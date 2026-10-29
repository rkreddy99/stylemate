"""Semantic index over the catalogue: one embedding per SKU, built once.

Keyword search finds "black t-shirt" in a row that says "black t-shirt". It
does not find "crew neck tee" in a row that says "round neck t-shirt", or see
that "boxy" and "oversized" are the same request. Embeddings do: every SKU's
text and every query become vectors, and closeness of meaning is a dot product.

Cost, measured in API calls rather than guessed:
  * the catalogue: 1,000 texts in 10 batch requests, ONCE. The vectors are
    cached in var/cache/ keyed on the model and the exact texts, so a restart
    or a second look costs nothing. A rebuilt catalogue gets a new key.
  * a query: every item in a look goes in ONE batch request.
Nothing else is sent, and no catalogue row ever reaches a chat model.

When no embedding model is reachable (no key, offline) `ready` stays False
and callers fall back to BM25 + structured matching, saying so.
"""
from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import numpy as np

from ..catalogue.store import catalogue
from ..config import settings
from ..llm.router import llm

DIMS = 768


def doc_text(r: dict) -> str:
    """What a SKU means, in words. Brand is left out on purpose: a photo never
    tells us the brand, so it would only pull matches toward name-alikes."""
    import json
    extra = {}
    try:
        extra = json.loads(r.get("raw") or "{}")
    except Exception:                                   # noqa: BLE001
        pass
    desc = extra.get("desc") or ""
    aud = r.get("audience") or ""
    aud = {"men": "men's", "women": "women's"}.get(aud, aud)
    return (f"{aud} {r.get('colour') or ''} {r.get('articleType') or ''}. "
            f"{r.get('name') or ''}. {desc}. {r.get('usage') or ''} wear.").strip()


class SemanticIndex:
    def __init__(self):
        self.ids: list[str] = []
        self.pos: dict[str, int] = {}
        self.matrix: np.ndarray | None = None
        self.model: str | None = None
        self.error: str | None = None
        self.built_seconds: float | None = None
        self.from_cache = False
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.matrix is not None

    def status(self) -> dict:
        return {"ready": self.ready, "model": self.model, "skus": len(self.ids),
                "fromCache": self.from_cache, "buildSeconds": self.built_seconds,
                "error": self.error}

    def _provider(self):
        g = llm.gemini
        if g.api_key and llm.active_name == "gemini":
            return g
        return None

    def ensure(self) -> bool:
        """Build (or load) the index if the catalogue has changed. Safe to call
        on every request: it is a dict lookup once built."""
        rows = catalogue.rows
        if not rows:
            return False
        g = self._provider()
        if g is None:
            self.error = "No embedding model: semantic search needs a Gemini key."
            return False
        texts = [doc_text(r) for r in rows]
        ids = [r["id"] for r in rows]
        key = hashlib.sha1(("\n".join(texts) + g.embed_model).encode()).hexdigest()[:16]
        if self.ready and self.model == g.embed_model and getattr(self, "_key", "") == key:
            return True
        with self._lock:
            if self.ready and getattr(self, "_key", "") == key:
                return True
            path = settings.cache_dir / f"emb-{g.embed_model}-{key}.npz"
            # A copy ships with the code in app/data. A host's disk is wiped on
            # every deploy and restart, so without it each cold start would
            # re-embed the whole catalogue before the first look could use it.
            shipped = Path(__file__).resolve().parent.parent / "data" / path.name
            if not path.exists() and shipped.exists():
                path = shipped
            t0 = time.time()
            try:
                if path.exists():
                    z = np.load(path, allow_pickle=False)
                    mat, cached_ids = z["m"], [str(x) for x in z["ids"]]
                    if cached_ids != ids:
                        raise ValueError("stale cache")
                    self.from_cache = True
                else:
                    vecs = g.embed_batch(texts, task_type="RETRIEVAL_DOCUMENT", dims=DIMS)
                    mat = np.asarray(vecs, dtype=np.float32)
                    mat /= np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)
                    np.savez(path, m=mat, ids=np.asarray(ids))
                    self.from_cache = False
            except Exception as e:                      # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"[:300]
                return False
            self.matrix, self.ids, self._key = mat, ids, key
            self.pos = {pid: i for i, pid in enumerate(ids)}
            self.model, self.error = g.embed_model, None
            self.built_seconds = round(time.time() - t0, 2)
            return True

    def query(self, texts: list[str]) -> tuple[list[dict[str, float]] | None, dict]:
        """Similarity of every SKU to each text, in one embedding request.

        Returns ([{sku_id: cosine}, ...], meta), or (None, meta) when the
        index is unavailable."""
        meta = {"model": self.model, "queries": len(texts), "seconds": 0.0}
        if not texts or not self.ensure():
            meta["error"] = self.error
            return None, meta
        t0 = time.time()
        try:
            q = np.asarray(self._provider().embed_batch(
                texts, task_type="RETRIEVAL_QUERY", dims=DIMS), dtype=np.float32)
        except Exception as e:                          # noqa: BLE001
            meta["error"] = f"{type(e).__name__}: {e}"[:300]
            return None, meta
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
        # numpy 2.0 on macOS's Accelerate BLAS raises spurious divide/overflow
        # warnings on float32 matmul. The vectors are unit length and finite and
        # the result matches float64 to 1e-6, so the warnings are noise.
        with np.errstate(all="ignore"):
            sims = q @ self.matrix.T
        meta["seconds"] = round(time.time() - t0, 2)
        return [dict(zip(self.ids, row.tolist())) for row in sims], meta


index = SemanticIndex()


def warm_in_background() -> None:
    """Build the index at startup so the first look is not the slow one."""
    threading.Thread(target=index.ensure, name="semantic-warm", daemon=True).start()
