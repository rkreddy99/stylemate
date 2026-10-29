"""SQLite-backed catalogue with a lexical index.

The catalogue can arrive three ways, in order of convenience:
  1. POST /api/catalogue/sync  - the browser pushes what it already loaded.
  2. `python -m app.catalogue.ingest --csv styles.csv` - Kaggle/HF CSV.
  3. `python -m app.catalogue.ingest --hf` - the HF datasets-server.

Whichever is used, the rows land in the same shape and the same index.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from ..config import settings
from .compat import slot_of

_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
  id TEXT PRIMARY KEY, sourceSku TEXT, cat TEXT, name TEXT, brand TEXT,
  articleType TEXT, masterCategory TEXT, subCategory TEXT, usage TEXT,
  season TEXT, year TEXT, colour TEXT, audience TEXT, spec TEXT,
  size TEXT, price REAL, image TEXT, slot TEXT, doc TEXT, raw TEXT
);
CREATE INDEX IF NOT EXISTS idx_cat ON products(cat);
CREATE INDEX IF NOT EXISTS idx_slot ON products(slot);
CREATE INDEX IF NOT EXISTS idx_colour ON products(colour);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""

_WORD = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


class Catalogue:
    def __init__(self, path=None):
        self.path = str(path or settings.db_path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self.rows: list[dict] = []
        self._df: Counter = Counter()
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._avg_len = 1.0
        self.pinned: dict | None = None
        self.reload()

    # -- writing ----------------------------------------------------------
    def upsert(self, items: Iterable[dict]) -> int:
        n = 0
        with _LOCK:
            cur = self._conn.cursor()
            for p in items:
                pid = str(p.get("id") or p.get("sourceSku") or "").strip()
                if not pid:
                    continue
                doc = " ".join(str(p.get(k) or "") for k in (
                    "name", "brand", "articleType", "masterCategory",
                    "subCategory", "usage", "season", "colour", "spec",
                    "audience", "cat", "desc"))
                cur.execute(
                    "INSERT INTO products (id,sourceSku,cat,name,brand,"
                    "articleType,masterCategory,subCategory,usage,season,year,"
                    "colour,audience,spec,size,price,image,slot,doc,raw) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "name=excluded.name,colour=excluded.colour,"
                    "price=excluded.price,image=excluded.image,doc=excluded.doc,"
                    "raw=excluded.raw,slot=excluded.slot",
                    (pid, str(p.get("sourceSku") or pid), str(p.get("cat") or ""),
                     str(p.get("name") or ""), str(p.get("brand") or ""),
                     str(p.get("articleType") or ""),
                     str(p.get("masterCategory") or ""),
                     str(p.get("subCategory") or ""), str(p.get("usage") or ""),
                     str(p.get("season") or ""), str(p.get("year") or ""),
                     str(p.get("colour") or ""), str(p.get("audience") or ""),
                     str(p.get("spec") or ""), str(p.get("size") or ""),
                     float(p.get("price") or 0),
                     str(p.get("image") or p.get("photo") or ""),
                     # An explicit slot wins: "Bracelet Style Strap Watch" is a
                     # watch, whatever the longest keyword in its name says.
                     str(p.get("slot") or "") or
                     slot_of(str(p.get("articleType") or ""),
                             str(p.get("name") or ""),
                             str(p.get("masterCategory") or "")) or "",
                     doc, json.dumps(p, ensure_ascii=False)))
                n += 1
            self._conn.commit()
        self.reload()
        return n

    def load_pinned(self, path: Path) -> None:
        """Make a catalogue file the single source of truth.

        The browser used to push whatever it had loaded (HuggingFace rows,
        synthetic rows) and the backend matched against that. With a pinned
        file the backend owns the shelf: the table is replaced when the file
        changes, and /api/catalogue/sync stops writing. Every feature - search,
        Shop a Look, the palette - then sees the same 1,000 SKUs as the page.
        """
        raw = Path(path).read_bytes()
        digest = hashlib.sha1(raw).hexdigest()
        data = json.loads(raw.decode("utf-8"))
        rows = data.get("rows") or []
        if self.get_meta("pinnedHash") != digest or len(self.rows) != len(rows):
            with _LOCK:
                self._conn.execute("DELETE FROM products")
                self._conn.commit()
            self.upsert(rows)
            self.set_meta("pinnedHash", digest)
            self.set_meta("lastSync", time.strftime("%Y-%m-%d %H:%M:%S"))
        self.pinned = {"source": data.get("source", Path(path).name),
                       "total": len(self.rows), "hash": digest[:12],
                       "path": str(path)}

    def set_meta(self, key: str, value: str) -> None:
        with _LOCK:
            self._conn.execute(
                "INSERT INTO meta (k,v) VALUES (?,?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (key, value))
            self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        r = self._conn.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
        return r["v"] if r else None

    # -- reading ----------------------------------------------------------
    def reload(self) -> None:
        cur = self._conn.execute("SELECT * FROM products")
        self.rows = [dict(r) for r in cur.fetchall()]
        self._build_index()

    def _build_index(self) -> None:
        self._df = Counter()
        self._postings = defaultdict(list)
        total = 0
        for i, r in enumerate(self.rows):
            toks = tokenise(r.get("doc", ""))
            total += len(toks)
            counts = Counter(toks)
            for t, c in counts.items():
                self._postings[t].append((i, c))
            self._df.update(counts.keys())
        self._avg_len = (total / len(self.rows)) if self.rows else 1.0

    def __len__(self) -> int:
        return len(self.rows)

    def get(self, pid: str) -> dict | None:
        for r in self.rows:
            if r["id"] == pid:
                return r
        return None

    def bm25(self, query: str, k1: float = 1.4, b: float = 0.72) -> dict[int, float]:
        """Classic BM25 over the product documents."""
        scores: dict[int, float] = defaultdict(float)
        N = max(1, len(self.rows))
        terms = tokenise(query)
        if not terms:
            return {}
        lens = {i: len(tokenise(r.get("doc", ""))) for i, r in enumerate(self.rows)}
        for t in set(terms):
            df = self._df.get(t, 0)
            if not df:
                continue
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            for i, tf in self._postings.get(t, []):
                dl = lens.get(i, 1) or 1
                denom = tf + k1 * (1 - b + b * dl / self._avg_len)
                scores[i] += idf * (tf * (k1 + 1)) / denom
        return scores

    def facets(self) -> dict[str, Any]:
        """Compact description of the catalogue, for the model's context."""
        def top(field, n=40):
            c = Counter(r[field] for r in self.rows if r.get(field))
            return [k for k, _ in c.most_common(n)]
        prices = sorted(r["price"] for r in self.rows if r.get("price"))
        return {
            "count": len(self.rows),
            "categories": top("cat"),
            "articleTypes": top("articleType", 70),
            "colours": top("colour", 45),
            "usages": top("usage", 15),
            "seasons": top("season", 6),
            "audiences": top("audience", 6),
            "slots": top("slot", 15),
            "priceRange": [prices[0], prices[-1]] if prices else [0, 0],
        }


catalogue = Catalogue()

# The shelf: built once by build_myntra_catalogue.py. SM_CATALOGUE_FILE points
# elsewhere; an empty value turns pinning off and the browser sync works again.
PINNED_FILE = os.getenv("SM_CATALOGUE_FILE",
                        str(Path(__file__).resolve().parent.parent / "data" /
                            "myntra_catalogue.json"))
if PINNED_FILE and Path(PINNED_FILE).is_file():
    catalogue.load_pinned(Path(PINNED_FILE))
