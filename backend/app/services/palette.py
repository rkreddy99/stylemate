"""Seasonal colour analysis, and the products that follow from it.

The Colour Studio reads an undertone (warm / cool / neutral) and a depth
(light / medium / deep) off a photo. Those two axes place a person in the
twelve-season system used by stylists, and each season carries a palette.

What this module adds: the palette is expressed in the CATALOGUE's own colour
vocabulary, so it can be turned straight into product recommendations - shirts,
t-shirts, trousers, dresses in the colours that suit the person, rather than a
row of swatches they then have to shop for themselves.

Honest about its limits: a real analysis is done by draping fabric in daylight
and comparing. Two numbers off a webcam frame put you in the right
neighbourhood, not the exact house. Every response says so, and the shopper can
override the season by hand.

Season colours follow standard seasonal analysis, cross-checked against the
reference the shopper supplied (ajeworld.com "What are my colours").
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from ..catalogue.store import catalogue
from ..catalogue.compat import SLOT_CATEGORIES


@dataclass
class Season:
    key: str
    name: str
    family: str                      # Spring | Summer | Autumn | Winter
    undertone: str
    depth: str
    summary: str
    colours: list[str]               # catalogue colour names, best first
    neutrals: list[str]
    metals: list[str]
    avoid: list[str]

    def dict(self) -> dict:
        return asdict(self)


# Catalogue colour names are title-case in the dataset ("Navy Blue"), so the
# palettes are written that way and matched case-insensitively.
SEASONS: dict[str, Season] = {
    "light_spring": Season(
        "light_spring", "Light Spring", "Spring", "warm", "light",
        "Warm but delicate. Clear, light colours with a touch of warmth lift "
        "you; anything heavy or muddy wears you instead.",
        ["Peach", "Coral", "Turquoise Blue", "Sea Green", "Pink", "Yellow",
         "Lavender", "Aqua", "Blue", "Mint Green", "Rose", "Nude"],
        ["Cream", "Off White", "Beige", "Tan", "Grey Melange"],
        ["gold", "rose gold"],
        ["Black", "Charcoal", "Maroon", "Coffee Brown"]),
    "warm_spring": Season(
        "warm_spring", "Warm Spring", "Spring", "warm", "medium",
        "Golden and clear. Colours with sunlight in them - coral, tangerine, "
        "warm greens - look alive on you.",
        ["Coral", "Orange", "Yellow", "Mustard", "Sea Green", "Turquoise Blue",
         "Green", "Peach", "Gold", "Rust", "Teal", "Red"],
        ["Ivory", "Cream", "Beige", "Camel", "Tan", "Khaki"],
        ["gold"],
        ["Black", "Silver", "Grey", "Mauve"]),
    "bright_spring": Season(
        "bright_spring", "Bright Spring", "Spring", "warm", "medium",
        "High contrast and warm. Clear, saturated colour suits you; anything "
        "dusty or greyed goes flat.",
        ["Red", "Turquoise Blue", "Magenta", "Blue", "Green", "Coral",
         "Yellow", "Purple", "Teal", "Orange", "Pink"],
        ["White", "Off White", "Navy Blue", "Grey"],
        ["gold", "silver"],
        ["Beige", "Khaki", "Mauve", "Grey Melange"]),
    "light_summer": Season(
        "light_summer", "Light Summer", "Summer", "cool", "light",
        "Cool and soft. Light, slightly powdery colours - cornflower, rose, "
        "lavender - sit well against your skin.",
        ["Lavender", "Pink", "Blue", "Sea Green", "Turquoise Blue", "Mauve",
         "Rose", "Purple", "Mint Green", "Peach", "Grey"],
        ["Off White", "Grey Melange", "Silver", "Steel", "Navy Blue"],
        ["silver", "white gold", "platinum"],
        ["Orange", "Mustard", "Rust", "Coffee Brown"]),
    "cool_summer": Season(
        "cool_summer", "Cool Summer", "Summer", "cool", "medium",
        "Properly cool. Navy, rose, raspberry and emerald read clean on you; "
        "warm earth tones pull you down.",
        ["Navy Blue", "Blue", "Pink", "Rose", "Magenta", "Purple", "Teal",
         "Sea Green", "Green", "Burgundy", "Lavender", "Grey"],
        ["Off White", "Grey", "Charcoal", "Steel", "Silver"],
        ["silver", "platinum", "white gold"],
        ["Orange", "Mustard", "Gold", "Rust", "Khaki"]),
    "soft_summer": Season(
        "soft_summer", "Soft Summer", "Summer", "neutral", "medium",
        "Muted and cool-leaning. Softened colours flatter you; anything "
        "neon or stark competes with your face.",
        ["Mauve", "Rose", "Sea Green", "Teal", "Blue", "Burgundy", "Olive",
         "Grey", "Lavender", "Pink", "Purple"],
        ["Grey Melange", "Off White", "Charcoal", "Taupe", "Navy Blue"],
        ["silver", "rose gold"],
        ["Orange", "Yellow", "Black", "Magenta"]),
    "soft_autumn": Season(
        "soft_autumn", "Soft Autumn", "Autumn", "neutral", "medium",
        "Warm but gentle. Muted earth colours - sage, old gold, light olive - "
        "suit you better than anything bright.",
        ["Olive", "Khaki", "Beige", "Peach", "Sea Green", "Teal", "Coral",
         "Mustard", "Rose", "Tan", "Green", "Brown"],
        ["Cream", "Off White", "Beige", "Taupe", "Grey Melange"],
        ["gold", "rose gold", "bronze"],
        ["Black", "Magenta", "Turquoise Blue", "Silver"]),
    "warm_autumn": Season(
        "warm_autumn", "Warm Autumn", "Autumn", "warm", "medium",
        "Golden and earthy. Rust, saffron, olive and warm reds are your "
        "strongest colours.",
        ["Rust", "Mustard", "Olive", "Orange", "Brown", "Coral", "Green",
         "Teal", "Maroon", "Gold", "Khaki", "Red"],
        ["Cream", "Beige", "Camel", "Tan", "Coffee Brown"],
        ["gold", "bronze", "copper"],
        ["Silver", "Pink", "Lavender", "Grey"]),
    "deep_autumn": Season(
        "deep_autumn", "Deep Autumn", "Autumn", "warm", "deep",
        "Warm and rich. Deep, saturated earth colours - forest, true red, "
        "dark olive, chocolate - have the weight to match your colouring.",
        ["Maroon", "Rust", "Olive", "Coffee Brown", "Brown", "Red", "Teal",
         "Mustard", "Green", "Burgundy", "Gold", "Tan"],
        ["Cream", "Off White", "Beige", "Khaki", "Charcoal"],
        ["gold", "bronze", "copper"],
        ["Pastel Pink", "Lavender", "Silver", "Grey Melange"]),
    "deep_winter": Season(
        "deep_winter", "Deep Winter", "Winter", "cool", "deep",
        "Cool and deep. High contrast suits you - true white against black, "
        "emerald, sapphire, cherry.",
        ["Black", "Navy Blue", "Burgundy", "Magenta", "Purple", "Green",
         "Blue", "Red", "Teal", "Maroon", "Charcoal"],
        ["White", "Off White", "Charcoal", "Grey", "Silver"],
        ["silver", "platinum", "white gold"],
        ["Orange", "Mustard", "Beige", "Peach"]),
    "cool_winter": Season(
        "cool_winter", "Cool Winter", "Winter", "cool", "medium",
        "Icy and clear. Emerald, cobalt, fuchsia and cool ruby are yours; "
        "warm or muted colours look dusty on you.",
        ["Navy Blue", "Blue", "Magenta", "Purple", "Green", "Red", "Pink",
         "Teal", "Burgundy", "Black", "Turquoise Blue"],
        ["White", "Off White", "Grey", "Charcoal", "Silver"],
        ["silver", "platinum", "white gold"],
        ["Orange", "Mustard", "Rust", "Khaki", "Gold"]),
    "bright_winter": Season(
        "bright_winter", "Bright Winter", "Winter", "cool", "light",
        "Cool and vivid. Hot pink, cobalt, true red and emerald hold their "
        "own against your contrast.",
        ["Magenta", "Blue", "Red", "Purple", "Green", "Turquoise Blue",
         "Pink", "Black", "Navy Blue", "Teal"],
        ["White", "Off White", "Grey", "Silver", "Charcoal"],
        ["silver", "platinum"],
        ["Beige", "Khaki", "Olive", "Mustard"]),
}

# (undertone, depth) -> season. Clarity is not measurable from a webcam frame,
# so the middle option of each family is chosen and the shopper can switch.
_MAP = {
    ("warm", "light"):    "light_spring",
    ("warm", "medium"):   "warm_autumn",
    ("warm", "deep"):     "deep_autumn",
    ("cool", "light"):    "light_summer",
    ("cool", "medium"):   "cool_summer",
    ("cool", "deep"):     "deep_winter",
    ("neutral", "light"): "light_summer",
    ("neutral", "medium"): "soft_summer",
    ("neutral", "deep"):  "deep_winter",
}

# Alternatives offered next to the chosen season, because warm/medium could
# reasonably be Warm Spring or Soft Autumn depending on clarity.
_NEIGHBOURS = {
    "light_spring": ["warm_spring", "light_summer"],
    "warm_spring": ["bright_spring", "warm_autumn"],
    "bright_spring": ["warm_spring", "bright_winter"],
    "light_summer": ["soft_summer", "cool_summer"],
    "cool_summer": ["soft_summer", "deep_winter"],
    "soft_summer": ["soft_autumn", "cool_summer"],
    "soft_autumn": ["warm_autumn", "soft_summer"],
    "warm_autumn": ["deep_autumn", "soft_autumn"],
    "deep_autumn": ["warm_autumn", "deep_winter"],
    "deep_winter": ["cool_winter", "deep_autumn"],
    "cool_winter": ["bright_winter", "deep_winter"],
    "bright_winter": ["cool_winter", "bright_spring"],
}


def season_for(undertone: str, depth: str) -> Season:
    u = (undertone or "neutral").strip().lower()
    d = (depth or "medium").strip().lower()
    if u not in ("warm", "cool", "neutral"):
        u = "neutral"
    if d not in ("light", "medium", "deep"):
        d = "medium"
    return SEASONS[_MAP[(u, d)]]


# Which slots to show products for, and what to call each row.
SHOP_ROWS = [
    ("top", "Tops and shirts"),
    ("bottom", "Trousers and jeans"),
    ("dress", "Dresses and ethnic wear"),
    ("outerwear", "Layers"),
    ("footwear", "Footwear"),
    ("jewellery", "Jewellery"),
    ("bag", "Bags"),
]


def recommend(undertone: str, depth: str, *, season_key: str | None = None,
              audience: str | None = None, per_row: int = 8,
              budget_max: float | None = None) -> dict:
    """Season + the actual products in that season's colours."""
    season = SEASONS.get(season_key or "") or season_for(undertone, depth)
    palette = [c.lower() for c in season.colours]
    neutrals = [c.lower() for c in season.neutrals]
    rank = {c: i for i, c in enumerate(palette)}
    avoid = {c.lower() for c in season.avoid}

    rows = []
    for slot, label in SHOP_ROWS:
        cats = SLOT_CATEGORIES.get(slot, [])
        picks = []
        for i, r in enumerate(catalogue.rows):
            if r["slot"] != slot:
                continue
            if cats and r["cat"] not in cats:
                continue
            if audience and r["audience"] != audience:      # women's only, no unisex
                continue
            if budget_max is not None and (r["price"] or 0) > budget_max:
                continue
            colour = (r["colour"] or "").lower()
            if colour in rank:
                score = 10.0 - rank[colour] * 0.25
                why = f"{r['colour']} is a core {season.name} colour"
            elif colour in neutrals:
                score = 5.0
                why = f"{r['colour']} is a neutral that works with your palette"
            else:
                continue
            picks.append((score, i, r, why))
        picks.sort(key=lambda t: (-t[0], t[1]))
        # Round-robin across colours. Sorting by score alone returned eight
        # maroon shirts, which shows the palette's rank rather than the
        # palette itself.
        by_colour: dict[str, list] = {}
        for pick in picks:
            by_colour.setdefault((pick[2]["colour"] or "").lower(), []).append(pick)
        order = sorted(by_colour,
                       key=lambda c: rank.get(c, 100 + neutrals.index(c)
                                              if c in neutrals else 200))
        spread = []
        depth_i = 0
        while len(spread) < per_row and any(by_colour.values()):
            placed = False
            for c in order:
                bucket = by_colour.get(c) or []
                if depth_i < len(bucket):
                    spread.append(bucket[depth_i])
                    placed = True
                    if len(spread) >= per_row:
                        break
            if not placed:
                break
            depth_i += 1

        items = []
        for _s, _i, r, why in spread:
            row = dict(r)
            row.pop("raw", None)
            row.pop("doc", None)
            row["why"] = why
            items.append(row)
        if items:
            rows.append({"slot": slot, "label": label, "items": items,
                         "total": len(picks)})

    return {
        "season": season.dict(),
        "alternatives": [SEASONS[k].dict() for k in _NEIGHBOURS.get(season.key, [])],
        "rows": rows,
        "avoid": season.avoid,
        "note": (
            f"Read as {season.name} from a {undertone or 'neutral'} undertone "
            f"and {depth or 'medium'} depth. Seasonal analysis is normally done "
            f"by draping fabric in daylight; two numbers from a photo put you "
            f"in the right family, not necessarily the exact sub-season. If the "
            f"colours look wrong, switch to one of the alternatives."),
        "counts": {r["slot"]: r["total"] for r in rows},
    }
