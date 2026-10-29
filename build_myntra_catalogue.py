#!/usr/bin/env python3
"""Build StyleMate's 1,000-SKU catalogue from the Myntra product CSV.

    python3 build_myntra_catalogue.py            # builds once, then refuses
    python3 build_myntra_catalogue.py --force    # rebuild on purpose

Run it once. The output is static data:

    backend/app/data/myntra_catalogue.json   the backend loads this at startup
    frontend/catalogue.js                    the same rows for a file:// page

The CSV has no category or colour column, but every product URL does:

    myntra.com/tshirts/<brand>/<brand>-men-mint-green-round-neck-...-t-shirt/16407468/buy
               ^type                ^gender, colour, neck, sleeve, fabric, fit

so the article type comes from the path and the colour, gender and the
search text come from the slug. That slug is the richest description in the
dataset, and it is what the semantic search embeds.

Selection is deterministic: per article type, the most-rated products first,
at most two per brand, alternating men and women where the type has both.
Standard library only, so it runs on the Python macOS ships.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "myntra dataset" / "myntra202305041052.csv"
OUT_JSON = HERE / "backend" / "app" / "data" / "myntra_catalogue.json"
OUT_JS = HERE / "frontend" / "catalogue.js"

# --------------------------------------------------------------------------
# What goes on the shelf. 5 categories, 1,000 SKUs, every category 50-400.
# (url type slug, articleType, display noun, slot, how many)
# --------------------------------------------------------------------------
PLAN = {
    "western": [
        ("tshirts", "Tshirts", "T-shirt", "top", 80),
        ("shirts", "Shirts", "Shirt", "top", 45),
        ("jeans", "Jeans", "Jeans", "bottom", 55),
        ("trousers", "Trousers", "Trousers", "bottom", 30),
        ("tops", "Tops", "Top", "top", 30),
        ("dresses", "Dresses", "Dress", "dress", 28),
        ("shorts", "Shorts", "Shorts", "bottom", 18),
        ("skirts", "Skirts", "Skirt", "bottom", 12),
        ("jackets", "Jackets", "Jacket", "outerwear", 15),
        ("sweatshirts", "Sweatshirts", "Sweatshirt", "top", 15),
        ("blazers", "Blazers", "Blazer", "outerwear", 12),
    ],
    "accessories": [
        ("watches", "Watches", "Watch", "watch", 50),
        ("smart-watches", "Smart Watches", "Smartwatch", "watch", 10),
        ("bracelet", "Bracelet", "Bracelet", "jewellery", 30),
        ("bangle", "Bangle", "Bangle", "jewellery", 10),
        ("necklace-and-chains", "Necklace and Chains", "Necklace", "jewellery", 30),
        ("pendant", "Pendant", "Pendant", "jewellery", 20),
        ("sunglasses", "Sunglasses", "Sunglasses", "eyewear", 30),
        ("belts", "Belts", "Belt", "belt", 22),
        ("earrings", "Earrings", "Earrings", "jewellery", 18),
        ("ring", "Ring", "Ring", "jewellery", 10),
        ("caps", "Caps", "Cap", "headwear", 12),
        ("handbags", "Handbags", "Handbag", "bag", 10),
        ("backpacks", "Backpacks", "Backpack", "bag", 8),
    ],
    "footwear": [
        ("casual-shoes", "Casual Shoes", "Casual Shoes", "footwear", 71),
        ("heels", "Heels", "Heels", "footwear", 25),
        ("flats", "Flats", "Flats", "footwear", 20),
        ("flip-flops", "Flip Flops", "Flip-Flops", "footwear", 20),
        ("boots", "Boots", "Boots", "footwear", 15),
        ("sandals", "Sandals", "Sandals", "footwear", 16),
        ("formal-shoes", "Formal Shoes", "Formal Shoes", "footwear", 3),
    ],
    "ethnic": [
        ("kurtas", "Kurtas", "Kurta", "top", 35),
        ("kurta-sets", "Kurta Sets", "Kurta Set", "top", 30),
        ("sarees", "Sarees", "Saree", "dress", 25),
        ("kurtis", "Kurtis", "Kurti", "top", 10),
        ("lehenga-choli", "Lehenga Choli", "Lehenga Choli", "dress", 10),
        ("nehru-jackets", "Nehru Jackets", "Nehru Jacket", "outerwear", 10),
        ("ethnic-dresses", "Ethnic Dresses", "Ethnic Dress", "dress", 10),
    ],
    "sports": [
        ("sports-shoes", "Sports Shoes", "Sports Shoes", "footwear", 35),
        ("track-pants", "Track Pants", "Track Pants", "bottom", 30),
        ("tracksuits", "Tracksuits", "Tracksuit", "top", 10),
        ("sports-sandals", "Sports Sandals", "Sports Sandals", "footwear", 10),
        # Not a Myntra type: t-shirts whose slug says they are for sport.
        ("@sports-tshirts", "Sports Tshirts", "Sports T-shirt", "top", 15),
    ],
}
SPORT_WORDS = {"sports", "sport", "running", "training", "gym", "workout",
               "dri", "athletic", "active", "activewear", "yoga"}

CAT_NAMES = {"western": "Western wear", "ethnic": "Ethnic wear",
             "footwear": "Footwear", "accessories": "Accessories",
             "sports": "Sports & activewear"}
MASTER = {"western": "Apparel", "ethnic": "Apparel", "sports": "Apparel",
          "footwear": "Footwear", "accessories": "Accessories"}
SUBCAT = {"top": "Topwear", "outerwear": "Topwear", "bottom": "Bottomwear",
          "dress": "Dress", "footwear": "Shoes", "watch": "Watches",
          "jewellery": "Jewellery", "eyewear": "Eyewear", "belt": "Belts",
          "headwear": "Headwear", "bag": "Bags"}
WOMEN_ONLY = {"sarees", "lehenga-choli", "heels", "dresses", "skirts", "kurtis",
              "earrings", "handbags", "flats", "ethnic-dresses", "tops"}
MEN_ONLY = {"nehru-jackets"}
NOT_ADULT = {"boys", "girls", "kids", "kid", "infant", "infants", "baby",
             "toddler", "toddlers", "teen", "teens", "junior"}

# Colour phrases as they appear in slugs -> the names the page's palette knows.
COLOUR_PHRASES = [
    ("navy blue", "Navy Blue"), ("off white", "Off White"),
    ("sea green", "Sea Green"), ("olive green", "Olive"),
    ("lime green", "Lime Green"), ("fluorescent green", "Fluorescent Green"),
    ("emerald green", "Emerald Green"), ("bottle green", "Emerald Green"),
    ("dark green", "Emerald Green"), ("mint green", "Sea Green"),
    ("turquoise blue", "Turquoise Blue"), ("teal blue", "Teal"),
    ("sky blue", "Sky Blue"), ("light blue", "Sky Blue"),
    ("powder blue", "Sky Blue"), ("royal blue", "Royal Blue"),
    ("charcoal grey", "Charcoal"), ("grey melange", "Grey Melange"),
    ("coffee brown", "Coffee Brown"), ("mushroom brown", "Mushroom Brown"),
    ("rose gold", "Rose Gold"), ("gold toned", "Gold"), ("gold plated", "Gold"),
    ("silver toned", "Silver"), ("silver plated", "Silver"),
    ("rose pink", "Pink"), ("dusty pink", "Pink"),
    ("navy", "Navy Blue"), ("black", "Black"), ("white", "White"),
    ("grey", "Grey"), ("gray", "Grey"), ("charcoal", "Charcoal"),
    ("blue", "Blue"), ("indigo", "Blue"), ("cobalt", "Royal Blue"),
    ("teal", "Teal"), ("turquoise", "Turquoise Blue"), ("green", "Green"),
    ("olive", "Olive"), ("mint", "Sea Green"), ("lime", "Lime Green"),
    ("red", "Red"), ("maroon", "Maroon"), ("burgundy", "Burgundy"),
    ("wine", "Wine"), ("rust", "Rust"), ("orange", "Orange"),
    ("coral", "Coral"), ("peach", "Peach"), ("pink", "Pink"),
    ("magenta", "Magenta"), ("fuchsia", "Magenta"), ("purple", "Purple"),
    ("violet", "Purple"), ("lavender", "Lavender"), ("lilac", "Lilac"),
    ("mauve", "Mauve"), ("yellow", "Yellow"), ("mustard", "Mustard"),
    ("cream", "Cream"), ("ivory", "Ivory"), ("beige", "Beige"),
    ("khaki", "Khaki"), ("tan", "Tan"), ("camel", "Tan"), ("brown", "Brown"),
    ("taupe", "Taupe"), ("nude", "Nude"), ("gold", "Gold"),
    ("golden", "Gold"), ("silver", "Silver"), ("gunmetal", "Steel"),
    ("steel", "Steel"), ("copper", "Copper"), ("bronze", "Bronze"),
    ("metallic", "Metallic"), ("multicoloured", "Multi"), ("multi", "Multi"),
    ("rose", "Rose"),
]
COLOUR_INDEX = sorted(((p.split(), c) for p, c in COLOUR_PHRASES),
                      key=lambda x: -len(x[0]))

IMG_TEMPLATE = "https://assets.myntassets.com/h_480,q_80,w_360/v1/assets/images/{}"
_IMG_PATH = re.compile(r"assets/images/(\S+?\.(?:jpe?g|png|webp))", re.I)


def find_colour(tokens: list[str]) -> str:
    """First colour mentioned in the slug, multi-word phrases preferred."""
    for i in range(len(tokens)):
        for words, canon in COLOUR_INDEX:
            if tokens[i:i + len(words)] == words:
                return canon
    return "Not specified"


def slug_tokens(slug: str, brand_slug: str, seller: str = "") -> list[str]:
    s = slug.strip("-")
    if brand_slug and s.startswith(brand_slug + "-"):
        s = s[len(brand_slug) + 1:]
    s = s.replace("--", " and ")
    toks = [t for t in re.split(r"[-\s]+", s.lower()) if t]
    # The URL's brand slug and the slug's own prefix do not always agree
    # ("atun" vs "a-t-u-n-..."), so also strip the seller's name word by word.
    bt = [t for t in re.split(r"[^a-z0-9]+", seller.lower()) if t]
    if bt and toks[:len(bt)] == bt:
        toks = toks[len(bt):]
    # Drop catalogue codes ("8015", "sf0051", "c5"): they carry no meaning for
    # a shopper and only add noise to the search text.
    return [t for t in toks if not any(ch.isdigit() for ch in t)]


def gender_of(tokens: list[str], name: str, type_slug: str) -> str | None:
    words = tokens + name.lower().split()
    # Anywhere in the slug, not just its start: a brand prefix can push
    # "boys" past the first few words.
    if any(t in NOT_ADULT for t in words):
        return None
    for t in words:
        if t in ("men", "mens", "male"):
            return "men"
        if t in ("women", "womens", "woman", "female", "ladies"):
            return "women"
        if t == "unisex":
            return "unisex"
    if type_slug in WOMEN_ONLY:
        return "women"
    if type_slug in MEN_ONLY:
        return "men"
    return "unisex"


def gender_from_photo(img_path: str) -> str | None:
    """The photo's file name often says what the slug did not:
    "...Roadster-Men-Tshirts-1.jpg", "...USPoloAssnWomenNavyBlueJeans1.jpg".
    Without this a man's hoodie was "unisex" and shown to girls. camelCase is
    split first, so "MANGOMANMenShirts" reads "MANGOMAN Men Shirts". None for
    kids' wear or when the name says nothing."""
    s = img_path.rsplit("/", 1)[-1]
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    words = set(re.split(r"[^a-z]+", s.lower()))
    if words & NOT_ADULT:
        return None
    if words & {"women", "womens", "woman", "ladies"}:
        return "women"
    if words & {"men", "mens"}:
        return "men"
    return None


def refine_audience() -> int:
    """Re-label the built catalogue's "unisex" rows from their names and photo
    file names, without the CSV. The same 1,000 products stay on the shelf."""
    if not OUT_JSON.is_file():
        print(f"{OUT_JSON.relative_to(HERE)} not found - build the catalogue first.")
        return 1
    payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    changed = Counter()
    for r in payload["rows"]:
        if r.get("audience") != "unisex":
            continue
        g = gender_of((r.get("desc") or "").split(), r.get("name") or "", "")
        if g == "unisex":
            g = gender_from_photo(r.get("photo") or "") or "unisex"
        if g != "unisex":
            r["audience"] = g
            changed[(r["cat"], g)] += 1
    payload["note"] = ("Prices, MRP, ratings and photos are Myntra's. Colour, gender and "
                       "the search text are parsed from each product's URL slug; gender "
                       "falls back to the photo's file name.")
    write_outputs(payload)
    print(f"re-labelled {sum(changed.values())} unisex SKUs:")
    for (cat, g), n in sorted(changed.items()):
        print(f"  {cat:12} -> {g:6} {n}")
    return 0


def write_outputs(payload: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8")
    OUT_JS.write_text("/* Generated by build_myntra_catalogue.py - do not edit. */\n"
                      "window.STYLEMATE_CATALOGUE=" +
                      json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n",
                      encoding="utf-8")


def num(x, kind=float):
    try:
        return kind(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return kind(0)


def display_name(name: str, noun: str) -> str:
    """The CSV name rarely says what the thing is ("Men Solid Oversized
    Cotton"), so add the noun when it is missing."""
    low = re.sub(r"[^a-z]", "", name.lower())
    key = re.sub(r"[^a-z]", "", noun.lower())
    stems = {key, key.rstrip("s"), key.replace("sports", "")}
    if noun == "T-shirt":
        stems |= {"tshirt", "tee"}
    if any(s and s in low for s in stems):
        return name.strip()
    return f"{name.strip()} {noun}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--force", action="store_true",
                    help="rebuild even though the catalogue already exists")
    ap.add_argument("--refine-audience", action="store_true",
                    help="re-label unisex rows of the built catalogue from photo "
                         "file names; needs no CSV")
    args = ap.parse_args()
    if args.refine_audience:
        return refine_audience()

    if OUT_JSON.exists() and not args.force:
        print(f"{OUT_JSON.relative_to(HERE)} already exists. It is static data, so it is "
              f"built once. Use --force to rebuild it.")
        return 0
    src = Path(args.csv)
    if not src.is_file():
        print(f"CSV not found: {src}")
        return 1

    csv.field_size_limit(10 ** 9)
    wanted: dict[str, tuple] = {}
    for cat, rows in PLAN.items():
        for slug, atype, noun, slot, n in rows:
            wanted[slug] = (cat, atype, noun, slot, n)

    # One streaming pass. Keep the best few hundred per (type, gender) by
    # rating count, so the 1.4 GB file never has to sit in memory.
    KEEP = 600
    heaps: dict[tuple[str, str], list] = defaultdict(list)
    seen_ids: set[str] = set()
    # Myntra re-lists the same product under new ids. Same seller, name, price
    # and slug is the same thing on the shelf twice; keep the first.
    seen_sigs: set[tuple] = set()

    def sig(it) -> tuple:
        row, toks = it[3], it[4]
        return ((row.get("seller") or "").lower(), (row.get("name") or "").lower(),
                row.get("price"), " ".join(toks))
    t0 = time.time()
    scanned = 0
    with open(src, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            scanned += 1
            parts = (row.get("purl") or "").split("/")
            if len(parts) < 7:
                continue
            type_slug, brand_slug, prod_slug, pid = parts[3], parts[4], parts[5], parts[6]
            if not pid.isdigit():
                continue
            toks = slug_tokens(prod_slug, brand_slug, row.get("seller") or "")
            key = type_slug
            if type_slug == "tshirts" and SPORT_WORDS & set(toks):
                key = "@sports-tshirts"
            if key not in wanted:
                continue
            price, mrp = num(row.get("price"), int), num(row.get("mrp"), int)
            rating, votes = num(row.get("rating")), num(row.get("ratingTotal"), int)
            if price <= 0 or rating < 3.5 or votes < 10:
                continue
            m = _IMG_PATH.search(row.get("img") or "")
            if not m:
                continue
            g = gender_of(toks, row.get("name") or "", type_slug)
            if g == "unisex":
                g = gender_from_photo(m.group(1)) or g
            if not g:
                continue
            h = heaps[(key, g)]
            # `scanned` is unique, so ties never fall through to comparing dicts.
            item = (votes, int(pid), scanned, row, toks, brand_slug, m.group(1), g)
            if len(h) < KEEP:
                heapq.heappush(h, item)
            elif votes > h[0][0]:
                heapq.heapreplace(h, item)
    print(f"scanned {scanned:,} rows in {time.time() - t0:.0f}s")

    out: list[dict] = []
    shortfall = []
    for key, (cat, atype, noun, slot, quota) in wanted.items():
        by_gender = {g: sorted(heaps.get((key, g), []), reverse=True)
                     for g in ("men", "women", "unisex")}
        chosen: list = []
        for brand_cap in (2, 4, 1000):           # relax only if the quota is short
            per_brand: Counter = Counter()
            chosen = []
            queues = {g: list(v) for g, v in by_gender.items() if v}
            while len(chosen) < quota and queues:
                for g in list(queues):
                    q = queues[g]
                    while q:
                        it = q.pop(0)
                        if str(it[1]) in seen_ids or per_brand[it[5]] >= brand_cap:
                            continue
                        if sig(it) in seen_sigs or any(sig(c) == sig(it) for c in chosen):
                            continue
                        per_brand[it[5]] += 1
                        chosen.append(it)
                        break
                    if not q:
                        del queues[g]
                    if len(chosen) >= quota:
                        break
            if len(chosen) >= quota:
                break
        if len(chosen) < quota:
            shortfall.append(f"{key}: {len(chosen)}/{quota}")
        for it in chosen:
            seen_sigs.add(sig(it))
        for votes, pid, _seq, row, toks, brand_slug, img_path, g in chosen:
            seen_ids.add(str(pid))
            name = display_name(row.get("name") or noun, noun)
            colour = find_colour(toks)
            if colour == "Not specified":
                colour = find_colour(re.split(r"[^a-z]+", (row.get("name") or "").lower()))
            usage = ("Ethnic" if cat == "ethnic" else "Sports" if cat == "sports"
                     else "Formal" if ("formal" in toks or key in ("formal-shoes", "blazers"))
                     else "Party" if "party" in toks else "Casual")
            price, mrp = num(row.get("price"), int), num(row.get("mrp"), int)
            desc = " ".join(toks)
            out.append({
                "id": f"myn-{pid}", "sourceSku": str(pid), "cat": cat, "slot": slot,
                "name": name, "brand": (row.get("seller") or "").strip() or brand_slug.replace("-", " ").title(),
                "articleType": atype, "masterCategory": MASTER[cat],
                "subCategory": SUBCAT[slot], "usage": usage, "season": "",
                "colour": colour, "audience": g,
                "price": price, "mrp": max(mrp, price),
                "discount": num(row.get("discount"), int),
                "rating": round(num(row.get("rating")), 1), "ratingTotal": votes,
                "photo": IMG_TEMPLATE.format(img_path),
                "url": row.get("purl"),
                "desc": desc,
                "spec": " · ".join(x for x in (atype, usage, MASTER[cat]) if x),
            })

    out.sort(key=lambda r: (list(PLAN).index(r["cat"]), r["articleType"], -r["ratingTotal"]))
    counts = Counter(r["cat"] for r in out)
    meta = {
        "source": f"Myntra product dataset ({src.name})",
        "built": time.strftime("%Y-%m-%d %H:%M"),
        "total": len(out),
        "categories": {c: {"name": CAT_NAMES[c], "count": counts[c]} for c in PLAN},
        "note": "Prices, MRP, ratings and photos are Myntra's. Colour, gender and "
                "the search text are parsed from each product's URL slug; gender "
                "falls back to the photo's file name.",
    }
    write_outputs({**meta, "rows": out})

    print(f"wrote {len(out)} SKUs -> {OUT_JSON.relative_to(HERE)} and {OUT_JS.relative_to(HERE)}")
    for c in PLAN:
        print(f"  {CAT_NAMES[c]:22} {counts[c]:4}")
    if shortfall:
        print("  short of quota: " + ", ".join(shortfall))
    return 0


if __name__ == "__main__":
    sys.exit(main())
