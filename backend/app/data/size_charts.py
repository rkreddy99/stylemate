"""Garment measurement charts.

IMPORTANT - these are GARMENT measurements (the flat dimensions of the item),
not body measurements. The two are not interchangeable: a size M t-shirt with a
100 cm chest is cut to go over a ~94 cm chest with ~6 cm of ease.

Provenance is tracked explicitly on every chart:
    "supplier"  - loaded from supplier_charts.json, keyed by SKU or brand+type.
                  These are the real thing.
    "reference" - the generic family charts below. Realistic for Indian RTW but
                  NOT this SKU's actual measurements. Anything derived from a
                  reference chart is reported at reduced confidence and the UI
                  says so.

Drop a supplier_charts.json next to this file to override per SKU; see
README "Supplying real size charts".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# Which measurements matter for which garment family. The fit engine only
# compares dimensions listed here, so a missing one is reported, not guessed.
FAMILY_DIMS: dict[str, list[str]] = {
    "top":     ["chest", "shoulder", "length", "sleeve"],
    "shirt":   ["chest", "shoulder", "length", "sleeve"],
    "dress":   ["chest", "waist", "hip", "length", "shoulder"],
    "kurta":   ["chest", "shoulder", "length", "sleeve"],
    "outer":   ["chest", "shoulder", "length", "sleeve"],
    "bottom":  ["waist", "hip", "inseam", "thigh"],
    "skirt":   ["waist", "hip", "length"],
    "shoe":    ["footLength"],
}

# articleType / name keyword -> family. Longest match wins.
FAMILY_KEYWORDS: list[tuple[str, str]] = [
    ("tshirt", "top"), ("t-shirt", "top"), ("tee", "top"), ("top", "top"),
    ("blouse", "top"), ("camisole", "top"), ("tank", "top"),
    ("shirt", "shirt"), ("kurta", "kurta"), ("kurti", "kurta"),
    ("sherwani", "kurta"), ("tunic", "kurta"),
    ("dress", "dress"), ("gown", "dress"), ("jumpsuit", "dress"),
    ("saree", None), ("dupatta", None), ("stole", None), ("scarf", None),
    ("jacket", "outer"), ("blazer", "outer"), ("coat", "outer"),
    ("sweater", "outer"), ("sweatshirt", "outer"), ("hoodie", "outer"),
    ("cardigan", "outer"), ("waistcoat", "outer"), ("nehru", "outer"),
    ("jean", "bottom"), ("trouser", "bottom"), ("pant", "bottom"),
    ("chino", "bottom"), ("cargo", "bottom"), ("jogger", "bottom"),
    ("legging", "bottom"), ("churidar", "bottom"), ("salwar", "bottom"),
    ("pyjama", "bottom"), ("track pant", "bottom"), ("short", "bottom"),
    ("skirt", "skirt"), ("palazzo", "bottom"), ("dhoti", "bottom"),
    ("shoe", "shoe"), ("sneaker", "shoe"), ("sandal", "shoe"),
    ("heel", "shoe"), ("loafer", "shoe"), ("boot", "shoe"),
    ("flip flop", "shoe"), ("slipper", "shoe"), ("flat", "shoe"),
]

_SIZES = ["XS", "S", "M", "L", "XL", "XXL"]

# Base size-M garment measurements in cm, then a per-size step.
# Separate blocks for men's and women's cuts because they are genuinely
# different patterns, not the same chart relabelled.
_BASE: dict[tuple[str, str], dict[str, float]] = {
    ("top", "male"):     {"chest": 104, "shoulder": 44.5, "length": 71,  "sleeve": 21},
    ("top", "female"):   {"chest": 94,  "shoulder": 37.5, "length": 62,  "sleeve": 18},
    ("shirt", "male"):   {"chest": 110, "shoulder": 45.5, "length": 75,  "sleeve": 62},
    ("shirt", "female"): {"chest": 98,  "shoulder": 38.5, "length": 64,  "sleeve": 57},
    ("kurta", "male"):   {"chest": 112, "shoulder": 45.5, "length": 102, "sleeve": 61},
    ("kurta", "female"): {"chest": 100, "shoulder": 38.0, "length": 94,  "sleeve": 46},
    ("dress", "female"): {"chest": 94,  "waist": 78, "hip": 100, "length": 96, "shoulder": 36.5},
    ("dress", "male"):   {"chest": 104, "waist": 92, "hip": 106, "length": 100, "shoulder": 44.0},
    ("outer", "male"):   {"chest": 116, "shoulder": 47.0, "length": 72,  "sleeve": 64},
    ("outer", "female"): {"chest": 104, "shoulder": 40.0, "length": 64,  "sleeve": 59},
    ("bottom", "male"):  {"waist": 82, "hip": 104, "inseam": 78, "thigh": 60},
    ("bottom", "female"):{"waist": 72, "hip": 100, "inseam": 74, "thigh": 57},
    ("skirt", "female"): {"waist": 72, "hip": 100, "length": 62},
    ("skirt", "male"):   {"waist": 82, "hip": 104, "length": 62},
}
# cm added per size step up from M.
_STEP: dict[str, dict[str, float]] = {
    "top":    {"chest": 5, "shoulder": 1.3, "length": 2.0, "sleeve": 1.0},
    "shirt":  {"chest": 5, "shoulder": 1.3, "length": 2.0, "sleeve": 1.5},
    "kurta":  {"chest": 5, "shoulder": 1.3, "length": 2.5, "sleeve": 1.5},
    "dress":  {"chest": 5, "waist": 5, "hip": 5, "length": 2.0, "shoulder": 1.0},
    "outer":  {"chest": 6, "shoulder": 1.5, "length": 2.0, "sleeve": 1.5},
    "bottom": {"waist": 5, "hip": 5, "inseam": 1.0, "thigh": 2.5},
    "skirt":  {"waist": 5, "hip": 5, "length": 1.5},
}

# UK shoe size -> internal foot length in cm (the dimension that decides fit).
SHOE_CHART: dict[str, float] = {
    "UK 5": 24.1, "UK 6": 24.9, "UK 7": 25.7, "UK 8": 26.5,
    "UK 9": 27.3, "UK 10": 28.1, "UK 11": 28.9, "UK 12": 29.7,
}

# Fabric stretch, as a fraction of the flat measurement that the garment can
# give before it reads as tight. Keyed by material word found in the product
# text. Absent material -> None, and the fit report says stretch is unknown.
FABRIC_STRETCH: dict[str, float] = {
    "lycra": 0.14, "spandex": 0.14, "elastane": 0.12, "jersey": 0.08,
    "knit": 0.08, "rib": 0.10, "cotton blend": 0.04, "polyester": 0.03,
    "denim": 0.02, "linen": 0.01, "silk": 0.01, "cotton": 0.02,
    "wool": 0.03, "georgette": 0.02, "chiffon": 0.02, "rayon": 0.04,
    "viscose": 0.04, "fleece": 0.07, "terry": 0.06,
}


def family_for(article_type: str, name: str = "", category: str = "") -> str | None:
    """Return the garment family, or None when the item is not body-sized."""
    hay = f"{article_type} {name} {category}".lower()
    # Prefer the keyword that appears EARLIEST, then the longest. Product names
    # lead with the primary piece, so "Kurta & Trouser Set" is a kurta; a
    # longest-match-wins rule called it a trouser.
    best: tuple[int, int, str | None] | None = None
    for kw, fam in FAMILY_KEYWORDS:
        at = hay.find(kw)
        if at < 0:
            continue
        cand = (at, -len(kw), fam)
        if best is None or cand[:2] < best[:2]:
            best = cand
    return best[2] if best else None


def reference_chart(family: str, gender: str) -> dict[str, dict[str, float]]:
    """Generate the reference garment chart for a family across all sizes."""
    gender = "female" if str(gender).lower().startswith("f") else "male"
    base = _BASE.get((family, gender)) or _BASE.get((family, "male"))
    if not base:
        return {}
    step = _STEP.get(family, {})
    mid = _SIZES.index("M")
    out: dict[str, dict[str, float]] = {}
    for i, size in enumerate(_SIZES):
        d = i - mid
        out[size] = {
            dim: round(v + step.get(dim, 0) * d, 1) for dim, v in base.items()
        }
    return out


class ChartStore:
    """Supplier charts first, reference charts as the labelled fallback."""

    def __init__(self, path: Path | None = None):
        self.path = path or (HERE / "supplier_charts.json")
        self.by_sku: dict[str, Any] = {}
        self.by_key: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        self.by_sku, self.by_key = {}, {}
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for entry in raw.get("charts", []):
            sizes = entry.get("sizes") or {}
            if not sizes:
                continue
            rec = {"sizes": sizes, "source": "supplier",
                   "supplier": entry.get("supplier", "supplier"),
                   "unit": entry.get("unit", "cm")}
            for sku in entry.get("skus", []):
                self.by_sku[str(sku)] = rec
            key = entry.get("brandArticleKey")
            if key:
                self.by_key[key.strip().lower()] = rec

    def chart_for(self, *, sku: str, brand: str, article_type: str,
                  name: str, category: str, gender: str) -> dict[str, Any]:
        """Return {sizes, source, dims, family, supplier?} for a product."""
        family = family_for(article_type, name, category)
        if family == "shoe":
            return {"family": "shoe", "source": "reference",
                    "dims": FAMILY_DIMS["shoe"],
                    "sizes": {k: {"footLength": v} for k, v in SHOE_CHART.items()}}
        if not family:
            return {"family": None, "source": None, "dims": [], "sizes": {}}

        hit = self.by_sku.get(str(sku))
        if not hit:
            hit = self.by_key.get(f"{brand}|{article_type}".strip().lower())
        if hit:
            sizes = hit["sizes"]
            if hit.get("unit") == "in":
                sizes = {s: {d: round(v * 2.54, 1) for d, v in m.items()}
                         for s, m in sizes.items()}
            dims = sorted({d for m in sizes.values() for d in m})
            return {"family": family, "source": "supplier", "dims": dims,
                    "sizes": sizes, "supplier": hit.get("supplier")}

        return {"family": family, "source": "reference",
                "dims": FAMILY_DIMS.get(family, []),
                "sizes": reference_chart(family, gender)}


def stretch_for(text: str) -> tuple[float | None, str | None]:
    """Fabric stretch allowance from the product text, or (None, None)."""
    hay = (text or "").lower()
    # A blend behaves like its stretchiest fibre: "cotton lycra" is a stretch
    # knit, not cotton. Matching by word length picked "cotton" and lost that.
    best: tuple[float, str] | None = None
    for word, val in FABRIC_STRETCH.items():
        if word in hay and (best is None or val > best[0]):
            best = (val, word)
    return best if best else (None, None)


charts = ChartStore()
