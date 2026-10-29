"""Styling knowledge used to answer relational queries.

"Something that goes with silver jewellery" and "something to wear with a
skirt" are not keyword searches - the words "silver" and "skirt" must NOT
appear in the results. They are constraints on what else works. This module
turns those phrases into (a) a slot filter, so a skirt query never returns
skirts, and (b) a colour preference, so silver pulls cool tones.

Everything here is a deterministic rule table. The LLM decides WHICH rules a
sentence invokes; it does not invent the rules.
"""
from __future__ import annotations

# --- colour temperature ---------------------------------------------------
COOL = {"navy blue", "blue", "teal", "turquoise blue", "grey", "charcoal",
        "silver", "white", "off white", "black", "purple", "lavender",
        "sea green", "green", "steel", "grey melange", "burgundy"}
WARM = {"red", "maroon", "orange", "rust", "mustard", "yellow", "gold",
        "beige", "tan", "khaki", "brown", "coffee brown", "olive", "peach",
        "cream", "copper", "bronze", "camel"}
NEUTRAL = {"black", "white", "off white", "grey", "charcoal", "beige",
           "navy blue", "cream", "ivory"}

# Metal -> colours that read well against it. Jewellery is the most common
# "goes with" in Indian fashion search, so it gets its own table.
METAL_AFFINITY = {
    "silver":   ["navy blue", "charcoal", "white", "grey", "black", "blue",
                 "sea green", "teal", "lavender", "burgundy"],
    "platinum": ["navy blue", "charcoal", "white", "grey", "black", "blue"],
    "white gold": ["navy blue", "charcoal", "white", "grey", "blue"],
    "gold":     ["maroon", "red", "mustard", "olive", "rust", "cream",
                 "beige", "green", "magenta", "brown"],
    "rose gold": ["peach", "pink", "cream", "beige", "maroon", "brown"],
    "oxidised": ["mustard", "olive", "maroon", "rust", "black", "indigo"],
}

# --- garment slots --------------------------------------------------------
# Which StyleMate categories can fill each slot.
SLOT_CATEGORIES = {
    "top":        ["western", "ethnic", "sports"],
    "bottom":     ["western", "ethnic", "sports"],
    "dress":      ["western", "ethnic"],
    "outerwear":  ["western", "ethnic", "sports"],
    "footwear":   ["footwear"],
    "jewellery":  ["accessories"],
    "bag":        ["accessories"],
    "watch":      ["accessories"],
    "eyewear":    ["accessories"],
    "belt":       ["accessories"],
    "headwear":   ["accessories"],
    "beauty":     ["beauty"],
}

# articleType keywords that identify a slot.
SLOT_KEYWORDS = {
    "top": ["tshirt", "t-shirt", "shirt", "top", "blouse", "kurta", "kurti",
            "tunic", "sweater", "sweatshirt", "camisole", "tank"],
    "bottom": ["jean", "trouser", "pant", "short", "skirt", "legging",
               "chino", "cargo", "jogger", "churidar", "salwar", "palazzo",
               "track pant", "capri", "dhoti"],
    "dress": ["dress", "gown", "jumpsuit", "saree", "lehenga"],
    "outerwear": ["jacket", "blazer", "coat", "cardigan", "waistcoat",
                  "shrug", "nehru"],
    "footwear": ["shoe", "sneaker", "sandal", "heel", "loafer", "boot",
                 "flip flop", "slipper", "flat", "wedge", "mule"],
    "jewellery": ["earring", "necklace", "bracelet", "bangle", "ring",
                  "pendant", "jewellery", "jewelry", "anklet", "chain",
                  "nose pin", "maang"],
    "bag": ["bag", "handbag", "clutch", "backpack", "wallet", "purse",
            "tote", "sling"],
    "watch": ["watch", "smartwatch"],
    "eyewear": ["sunglass", "eyewear", "spectacle", "frame"],
    "belt": ["belt"],
    "headwear": ["cap", "hat", "turban", "headband", "scarf", "stole"],
    "beauty": ["lipstick", "foundation", "kajal", "perfume", "fragrance",
               "nail", "serum", "cream", "lotion", "shampoo", "kohl",
               "mascara", "eyeliner", "compact", "deodorant"],
}

# Given an anchor the shopper already owns, which slots make sense to buy.
# The anchor's OWN slot is excluded - that is the whole point.
PAIRS_WITH = {
    "top":       ["bottom", "outerwear", "footwear", "jewellery", "bag", "watch", "belt"],
    "bottom":    ["top", "outerwear", "footwear", "jewellery", "bag", "belt"],
    "dress":     ["outerwear", "footwear", "jewellery", "bag", "belt"],
    "outerwear": ["top", "bottom", "footwear", "jewellery", "bag"],
    "footwear":  ["top", "bottom", "dress", "outerwear", "bag"],
    "jewellery": ["top", "dress", "ethnic", "bottom", "outerwear", "bag"],
    "bag":       ["top", "bottom", "dress", "outerwear", "footwear"],
    "watch":     ["top", "outerwear", "bottom"],
    "belt":      ["top", "bottom", "footwear"],
    "eyewear":   ["top", "dress", "outerwear"],
    "headwear":  ["top", "dress", "outerwear"],
    "beauty":    ["beauty"],
}

# --- occasion and season --------------------------------------------------
OCCASION_USAGE = {
    "wedding": ["Ethnic", "Party", "Formal"],
    "sangeet": ["Ethnic", "Party"], "mehendi": ["Ethnic", "Party"],
    "reception": ["Party", "Ethnic", "Formal"],
    "festival": ["Ethnic", "Party"], "diwali": ["Ethnic", "Party"],
    "puja": ["Ethnic"], "navratri": ["Ethnic", "Party"],
    "office": ["Formal", "Casual"], "work": ["Formal", "Casual"],
    "interview": ["Formal"], "meeting": ["Formal"],
    "party": ["Party"], "club": ["Party"], "night out": ["Party"],
    "date": ["Party", "Casual"], "brunch": ["Casual", "Smart Casual"],
    "casual": ["Casual"], "everyday": ["Casual"], "college": ["Casual"],
    "gym": ["Sports"], "workout": ["Sports"], "running": ["Sports"],
    "yoga": ["Sports"], "travel": ["Casual", "Sports"],
    "beach": ["Casual"], "holiday": ["Casual"],
}

MONTH_SEASON = {
    1: "Winter", 2: "Winter", 3: "Spring", 4: "Summer", 5: "Summer",
    6: "Summer", 7: "Monsoon", 8: "Monsoon", 9: "Fall", 10: "Fall",
    11: "Winter", 12: "Winter",
}
MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


# --- colour names ----------------------------------------------------------
# A vision model says "dark navy" or "light wash denim blue"; the catalogue says
# "Navy Blue". These map free text onto the catalogue's names (the same set
# build_myntra_catalogue.py parses from product URLs), longest phrase first.
_COLOUR_WORDS = [
    ("navy blue", "Navy Blue"), ("off white", "Off White"), ("sea green", "Sea Green"),
    ("olive green", "Olive"), ("lime green", "Lime Green"), ("emerald", "Emerald Green"),
    ("bottle green", "Emerald Green"), ("dark green", "Emerald Green"),
    ("mint", "Sea Green"), ("turquoise", "Turquoise Blue"), ("teal", "Teal"),
    ("sky blue", "Sky Blue"), ("light blue", "Sky Blue"), ("powder blue", "Sky Blue"),
    ("royal blue", "Royal Blue"), ("cobalt", "Royal Blue"), ("charcoal", "Charcoal"),
    ("grey melange", "Grey Melange"), ("heather grey", "Grey Melange"),
    ("rose gold", "Rose Gold"), ("coffee", "Coffee Brown"), ("navy", "Navy Blue"),
    ("denim", "Blue"), ("indigo", "Blue"), ("black", "Black"), ("white", "White"),
    ("grey", "Grey"), ("gray", "Grey"), ("blue", "Blue"), ("green", "Green"),
    ("olive", "Olive"), ("khaki", "Khaki"), ("beige", "Beige"), ("cream", "Cream"),
    ("ivory", "Ivory"), ("tan", "Tan"), ("camel", "Tan"), ("brown", "Brown"),
    ("taupe", "Taupe"), ("maroon", "Maroon"), ("burgundy", "Burgundy"),
    ("wine", "Wine"), ("red", "Red"), ("rust", "Rust"), ("orange", "Orange"),
    ("coral", "Coral"), ("peach", "Peach"), ("pink", "Pink"), ("magenta", "Magenta"),
    ("fuchsia", "Magenta"), ("purple", "Purple"), ("violet", "Purple"),
    ("lavender", "Lavender"), ("lilac", "Lilac"), ("mauve", "Mauve"),
    ("mustard", "Mustard"), ("yellow", "Yellow"), ("gold", "Gold"),
    ("silver", "Silver"), ("steel", "Steel"), ("gunmetal", "Steel"),
    ("copper", "Copper"), ("bronze", "Bronze"), ("metallic", "Metallic"),
    ("nude", "Nude"), ("multi", "Multi"), ("rose", "Rose"),
]
COLOUR_FAMILY = {
    "blue": {"Navy Blue", "Blue", "Sky Blue", "Royal Blue", "Teal", "Turquoise Blue"},
    "green": {"Green", "Sea Green", "Olive", "Lime Green", "Emerald Green", "Teal"},
    "grey": {"Grey", "Charcoal", "Grey Melange", "Steel", "Silver", "Metallic"},
    "light": {"White", "Off White", "Cream", "Ivory", "Beige"},
    "brown": {"Brown", "Tan", "Coffee Brown", "Taupe", "Khaki", "Beige", "Rust",
              "Copper", "Bronze", "Mushroom Brown"},
    "red": {"Red", "Maroon", "Burgundy", "Wine", "Rust", "Coral"},
    "pink": {"Pink", "Rose", "Magenta", "Peach", "Coral", "Mauve", "Rose Gold"},
    "purple": {"Purple", "Lavender", "Lilac", "Mauve", "Magenta"},
    "yellow": {"Yellow", "Mustard", "Gold", "Orange"},
    "metal": {"Gold", "Silver", "Rose Gold", "Steel", "Metallic", "Copper", "Bronze"},
}


def canonical_colour(text: str | None) -> str | None:
    import re
    words = re.findall(r"[a-z]+", (text or "").lower())
    joined = " " + " ".join(words) + " "
    for phrase, canon in _COLOUR_WORDS:
        if f" {phrase} " in joined:
            return canon
    return None


def colour_closeness(want: str | None, have: str | None) -> float:
    """1 same colour, 0.6 same family (navy vs blue), 0 otherwise."""
    if not want or not have or have == "Not specified":
        return 0.0
    if want == have:
        return 1.0
    for fam in COLOUR_FAMILY.values():
        if want in fam and have in fam:
            return 0.6
    return 0.0


def slot_of(article_type: str, name: str = "", category: str = "") -> str | None:
    hay = f"{article_type} {name} {category}".lower()
    best = None
    for slot, kws in SLOT_KEYWORDS.items():
        for kw in kws:
            if kw in hay and (best is None or len(kw) > best[0]):
                best = (len(kw), slot)
    return best[1] if best else None


def colours_for_metal(metal: str) -> list[str]:
    return METAL_AFFINITY.get((metal or "").strip().lower(), [])


def complement_slots(anchor_slot: str | None) -> list[str]:
    if not anchor_slot:
        return []
    return PAIRS_WITH.get(anchor_slot, [])


def temperature(colour: str) -> str:
    c = (colour or "").strip().lower()
    if c in NEUTRAL:
        return "neutral"
    if c in COOL:
        return "cool"
    if c in WARM:
        return "warm"
    return "unknown"


def goes_with_colour(anchor: str, candidate: str) -> float:
    """0..1 affinity between two colour names."""
    a, b = (anchor or "").lower(), (candidate or "").lower()
    if not a or not b:
        return 0.5
    if b in NEUTRAL or a in NEUTRAL:
        return 0.85
    ta, tb = temperature(a), temperature(b)
    if ta == "unknown" or tb == "unknown":
        return 0.5
    if a == b:
        return 0.55            # head-to-toe one colour is rarely the answer
    return 0.8 if ta == tb else 0.65


def season_for_month(month: int | None) -> str | None:
    return MONTH_SEASON.get(month) if month else None


def usages_for_occasion(occasion: str | None) -> list[str]:
    if not occasion:
        return []
    o = occasion.strip().lower()
    for key, val in OCCASION_USAGE.items():
        if key in o:
            return val
    return []
