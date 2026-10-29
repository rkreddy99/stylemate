"""Build StyleMate catalogue v2 from the Kaggle "Fashion Product Images" set in ~/myntai.

    backend/.venv/bin/python build_catalogue_v2.py            # ~2,500 rows + images
    backend/.venv/bin/python build_catalogue_v2.py --no-images

Inputs (read-only):
  ~/myntai/data/raw/fashion-product-images-dataset.zip   styles.csv, styles/<id>.json, images/<id>.jpg
  ~/myntai/data/derived/k44.jsonl                        slim extract of the 44k
  ~/myntai/data/derived/ratings_44k.jsonl                simulated ratings (borrowed from donor products)
  ~/myntai/data/generated/reviews_v1/batch_*.jsonl       generated reviews

Outputs:
  backend/app/data/catalogue_v2.json    pinned catalogue (same row keys as v1, plus sizes/attrs/gallery/reviews)
  backend/app/data/img/<styleId>.jpg    main photo, 600x800 JPEG

Deterministic: same inputs and seed give the same file.
"""
from __future__ import annotations

import argparse
import csv
import glob
import html
import io
import json
import random
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE / "backend"
sys.path.insert(0, str(BACKEND))
from app.catalogue.compat import canonical_colour, colour_closeness, slot_of  # noqa: E402

MYNTAI = Path.home() / "myntai" / "data"
ZIP = MYNTAI / "raw" / "fashion-product-images-dataset.zip"
OUT = BACKEND / "app" / "data" / "catalogue_v2.json"
IMG = BACKEND / "app" / "data" / "img"

TOTAL = 2500
QUOTA = {"western": 0.34, "accessories": 0.26, "footwear": 0.17, "ethnic": 0.13, "sports": 0.10}
ETHNIC_TYPES = {"Kurtas", "Kurtis", "Sarees", "Kurta Sets", "Salwar", "Churidar", "Lehenga Choli",
                "Salwar and Dupatta", "Nehru Jackets", "Dhotis", "Patiala", "Dupatta"}
NO_SHOW = {"Briefs", "Bra", "Innerwear Vests", "Trunk", "Boxers", "Night suits", "Stockings",
           "Shapewear", "Bath Robe", "Camisoles", "Lounge Pants", "Lounge Shorts", "Nightdress",
           "Baby Dolls", "Robe", "Lingerie Set", "Socks", "Accessory Gift Set", "Free Gifts"}
ADULT_AGE = {"Adults-Men", "Adults-Women", "Adults-Unisex"}
KID_SIZE = re.compile(r"^\d+(-\d+)?\s*(YRS?|Y|M|MONTHS?)$", re.I)   # 4YRS, 2-3Y, 6M - not adult "M"
GENDER_WORD = re.compile(r"\b(Men's|Women's|Mens|Womens|Men|Women|Unisex|Boys|Girls)(?=[\s'A-Z]|$)\s*")
JUNK_TYPES = {"Shoe Accessories", "Key chain", "Ties and Cufflinks", "Swimwear", "Travel Accessory"}
SPORTS_TYPES = {"Sports Shoes", "Sports Sandals", "Track Pants", "Tracksuits"}
ETHNIC_BY_USAGE_NOT = {"Leggings", "Tops", "Shirts", "Tshirts", "Jeans", "Trousers"}
SIZE_MAP = {"ONESIZE": "Onesize", "FREESIZE": "Onesize", "FREESIZES": "Onesize", "FREE SIZE": "Onesize", "WP": "XS", "WS": "S",
            "WM": "M", "WL": "L", "WG": "XL", "WVG": "XXL"}
AUD = {"Men": "men", "Women": "women", "Unisex": "unisex"}


def cat_of(r: dict) -> str:
    if r["type"] in ETHNIC_TYPES or (r["master"] == "Apparel" and r["usage"] == "Ethnic"
                                     and r["type"] not in ETHNIC_BY_USAGE_NOT):
        return "ethnic"
    if r["type"] in SPORTS_TYPES or (r["usage"] == "Sports" and r["master"] in ("Apparel", "Footwear")):
        return "sports"
    if r["master"] == "Footwear":
        return "footwear"
    if r["master"] == "Accessories":
        return "accessories"
    return "western"


def clean_name(name: str, brand: str, colour: str = "", article: str = "") -> str:
    n = name.strip()
    squash = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    if brand:                                  # brand may be spelt differently in the name
        words = n.split()
        for k in range(min(4, len(words)), 0, -1):
            if squash(" ".join(words[:k])) == squash(brand):
                n = " ".join(words[k:]); break
    n = GENDER_WORD.sub("", n).strip()
    n = re.sub(r"^[&\s,/-]+", "", n)
    n = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", n, flags=re.I)
    n = re.sub(r"\s{2,}", " ", n).strip()
    if len(n.split()) <= 1 and colour and article:
        n = f"{colour} {article}"
    return n or name


def clean_sizes(sizes: list[str]) -> list[str]:
    out = []
    for s in sizes:
        s = str(s).strip().upper()
        s = SIZE_MAP.get(s, s)
        s = re.sub(r"^([A-Z]+)/\d+$", r"\1", s)         # "S/39" -> "S"
        if s and s not in out:
            out.append(s)
    uk = {s[2:] for s in out if s.startswith("UK")}
    return [s for s in out if not (s.isdigit() and s in uk)]


def clean_text(h: str, cap: int = 700) -> str:
    t = re.sub(r"<[^>]*>?", " ", h or "")             # closes cut-off tags too
    t = re.sub(r"\s+", " ", html.unescape(t)).replace("Â", " ").strip()
    t = re.sub(r"\b(Composition|Fitting|Fit|Wash ?care|Material|Care)\b\s*:?\s*", lambda m: m.group(1).capitalize() + ": ", t)
    t = t[:cap]
    end = max(t.rfind(". "), t.rfind("! "), t.rfind("? "), t.rfind(".") if t.endswith(".") else -1)
    return t[:end + 1].strip() if end > 40 else t.rsplit(" ", 1)[0] + "…" if len(t) >= cap else t


def load_inputs():
    z = zipfile.ZipFile(ZIP)
    styles = {}
    with z.open("fashion-dataset/styles.csv") as f:
        for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace")):
            try:
                styles[int(row["id"])] = row
            except (ValueError, KeyError):
                pass
    k44 = {}
    for line in open(MYNTAI / "derived" / "k44.jsonl", encoding="utf-8"):
        r = json.loads(line)
        k44[r["id"]] = r
    ratings = {}
    for line in open(MYNTAI / "derived" / "ratings_44k.jsonl", encoding="utf-8"):
        r = json.loads(line)
        ratings[r["id"]] = r
    reviews = {}
    for fn in sorted(glob.glob(str(MYNTAI / "generated" / "reviews_v1" / "batch_*.jsonl"))):
        for line in open(fn, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("reviews"):
                reviews[r["id"]] = [{"stars": int(v["stars"]), "text": v["text"]}
                                    for v in r["reviews"] if v.get("text")]
    return z, styles, k44, ratings, reviews


def candidate(r, st, rt, rv) -> bool:
    if r["gender"] not in AUD or r["master"] not in ("Apparel", "Footwear", "Accessories"):
        return False
    if r["type"] in JUNK_TYPES or (r.get("price") or 0) < 50 or "_" in r["name"]:
        return False
    if (r["type"] == "Kurtis" and r["gender"] == "Men") or (r["type"] == "Tshirts" and r["master"] == "Accessories"):
        return False
    cc = canonical_colour(r["name"])                  # source colour contradicts the name
    if cc and r.get("colour") and colour_closeness(cc, r["colour"]) == 0 and " & " not in r["name"] and " and " not in r["name"].lower():
        return False
    if r["type"] in NO_SHOW or slot_of(r["type"], r["name"], r["master"]) in (None, "", "beauty"):
        return False
    if not r.get("colour") or r["colour"] == "NA" or not r.get("sizes") or not st:
        return False
    if not st.get("season") or not r.get("usage") or r["usage"] == "NA":
        return False
    if any(KID_SIZE.search(s) for s in r["sizes"]):
        return False
    if not rt or not rt.get("rating_count") or not rv:
        return False
    return bool(r.get("price"))


def style_json(z, sid):
    try:
        return json.loads(z.read(f"fashion-dataset/styles/{sid}.json"))["data"]
    except (KeyError, ValueError):
        return None


def build(images: bool, seed: int) -> None:
    t0 = time.time()
    z, styles, k44, ratings, reviews = load_inputs()
    pool = defaultdict(lambda: defaultdict(list))   # cat -> (articleType, audience) -> [ids]
    for sid in sorted(k44):
        r = k44[sid]
        if candidate(r, styles.get(sid), ratings.get(sid), reviews.get(sid)):
            pool[cat_of(r)][f'{r["type"]}|{AUD[r["gender"]]}'].append(sid)
    seen = set()
    rnd = random.Random(seed)
    names = set(z.namelist())
    rows, counts = [], {}
    for cat, share in QUOTA.items():
        want = round(TOTAL * share)
        by_type = {t: rnd.sample(ids, len(ids)) for t, ids in sorted(pool[cat].items())}
        taken = 0
        while taken < want and any(by_type.values()):
            for t in sorted(by_type):                     # round-robin across article types
                if taken >= want or not by_type[t]:
                    continue
                sid = by_type[t].pop()
                row = make_row(z, names, sid, k44[sid], styles[sid], ratings[sid], reviews[sid], cat)
                key = row and (row["brand"].lower(), row["name"].lower(), row["colour"], row["price"], row["audience"])
                if row and key not in seen:
                    seen.add(key)
                    rows.append(row)
                    taken += 1
        counts[cat] = taken
    rows.sort(key=lambda p: p["id"])
    if images:
        IMG.mkdir(parents=True, exist_ok=True)
        write_images(z, rows)
    payload = {
        "source": "Kaggle Fashion Product Images (Myntra, 2011-2017) via ~/myntai",
        "built": time.strftime("%Y-%m-%d"),
        "version": 2,
        "total": len(rows),
        "categories": {c: {"name": n, "count": counts.get(c, 0)} for c, n in (
            ("western", "Western wear"), ("ethnic", "Ethnic wear"), ("footwear", "Footwear"),
            ("accessories", "Accessories"), ("sports", "Sports & activewear"))},
        "note": "Names, brands, prices, attributes, sizes and photos are Myntra's. Ratings are "
                "borrowed from similar real products; reviews are generated.",
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} rows={len(rows)} {counts} in {time.time()-t0:.0f}s")


def make_row(z, names, sid, r, st, rt, rv, cat):
    if f"fashion-dataset/fashion-dataset/images/{sid}.jpg" not in names:
        return None
    sj = style_json(z, sid)
    if not sj or sj.get("ageGroup") not in ADULT_AGE:
        return None
    si = sj.get("styleImages") or {}
    gallery = [si[a]["imageURL"].replace("http://", "https://", 1) for a in ("front", "back", "left", "right")
               if isinstance(si.get(a), dict) and si[a].get("imageURL")]
    if not gallery:
        return None
    mrp = float(r["price"])
    price = float(r.get("dprice") or mrp)
    attrs = {k: v for k, v in (r.get("attrs") or {}).items()
             if k in ("Fit", "Fabric", "Pattern", "Sleeve Length", "Neck", "Occasion", "Type") and v and v != "NA"}
    slot = slot_of(r["type"], r["name"], r["master"])
    name = clean_name(r["name"], r["brand"], r["colour"], r["type"])
    details = clean_text(r.get("desc", "")).replace("Â", " ").strip()
    sizes = clean_sizes(r["sizes"])
    spec = " · ".join([r["type"]] + [attrs[k] for k in ("Fit", "Fabric", "Pattern") if k in attrs])
    return {
        "id": f"k44-{sid}", "sourceSku": str(sid), "cat": cat, "slot": slot,
        "name": name, "brand": r["brand"], "articleType": r["type"],
        "masterCategory": r["master"], "subCategory": r["sub"], "usage": r["usage"],
        "season": st.get("season", ""), "year": st.get("year", ""),
        "colour": r["colour"], "audience": AUD[r["gender"]],
        "price": price, "mrp": mrp,
        "discount": round(100 - price / mrp * 100) if mrp > price else 0,
        "rating": rt.get("rating"), "ratingTotal": rt.get("rating_count"),
        "photo": f"/img/{sid}.jpg", "image": f"/img/{sid}.jpg", "gallery": gallery,
        "url": "https://www.myntra.com/" + (sj.get("landingPageUrl") or ""),
        "desc": f"{name} {r['brand']} {r['colour']} {r['type']} {' '.join(attrs.values())}".lower(),
        "spec": spec, "size": ", ".join(sizes), "sizes": sizes, "attrs": attrs,
        "details": details if len(details) >= 5 and details not in ("-", "NA") else spec, "reviews": rv[:8],
        "realPrice": True,
    }


def write_images(z, rows):
    import cv2
    import numpy as np
    done = 0
    for p in rows:
        out = IMG / f"{p['sourceSku']}.jpg"
        if out.exists():
            continue
        buf = np.frombuffer(z.read(f"fashion-dataset/fashion-dataset/images/{p['sourceSku']}.jpg"), np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if im is None:
            continue
        h, w = im.shape[:2]
        tw = 600
        th = round(h * tw / w)
        im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out), im, [cv2.IMWRITE_JPEG_QUALITY, 78])
        done += 1
        if done % 250 == 0:
            print(f"  images {done}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--seed", type=int, default=20261030)
    a = ap.parse_args()
    build(not a.no_images, a.seed)
