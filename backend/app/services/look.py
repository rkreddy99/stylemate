"""Shop a look: recognise what a person is wearing, then find it here.

The image is sent ONCE. Everything after that is text:

  1. SEE        one vision call (Gemini through the router). The photo is
                downscaled to 768px first, which bounds the image tokens, and
                a response schema makes the reply pure JSON - no prose, no
                repair pass. The model returns each visible item as a slot, a
                catalogue article type, a colour, listing-style details and a
                one-line description written like a product title. The result
                is cached by the image's hash, so the same photo is never paid
                for twice in a session.

  2. MATCH      plain code over the catalogue, with no chat model involved:
                hard filters (slot, and women's or men's only: the person in
                the photo, else the shopper's saved choice, else all), then
                a hybrid score - semantic similarity between the item's
                description and every SKU (one embedding request for all the
                items in the look), BM25 over the same words, and exact type
                and colour agreement.

  3. RE-MATCH   POST /api/look/match takes the step-1 text and runs step 2
                again (another audience, more results) - zero image tokens.

If no vision model answers, ZONE ANALYSIS (OpenCV) reads the dominant colour of
each band of the figure instead. It knows where an item sits, not what it is,
and the payload says so.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict

import numpy as np

from ..catalogue.compat import (SLOT_CATEGORIES, canonical_colour,
                                colour_closeness, slot_of)
from ..catalogue.store import catalogue, tokenise
from ..config import settings
from ..llm.router import LLMUnavailable, llm
from . import semantic

SLOTS = ["top", "bottom", "dress", "outerwear", "footwear", "watch", "jewellery",
         "eyewear", "belt", "headwear", "bag"]

VISION_SYSTEM = """You tag what a person in a photo is wearing, so a fashion \
store can search its catalogue for similar products. You are a catalogue \
tagger, not a stylist.

For every visible garment, shoe and accessory give:
- slot: one of {slots}
- article_type: the closest of: {types}
- colour: the main colour in plain words; secondary_colour if there is a clear second one
- pattern: solid, striped, checked, printed, floral, graphic, textured, embroidered or null
- details: 2-5 short phrases a product listing would use - fit, neck, sleeve,
  length, rise, fabric, closure; for accessories the dial, strap, metal, stone
  or shape (e.g. "slim fit", "round neck", "short sleeves", "mid-rise",
  "analogue dial", "metal strap", "gold-plated chain", "aviator")
- worn_by: menswear, womenswear or unisex - judged from the item's cut and
  styling alone. A plain tee, plain jeans or sneakers is unisex.
- description: ONE line written like a product title, using the words above,
  e.g. "men black slim fit round neck cotton t-shirt"
- confidence: 0-1

Also give person_detected, a one-sentence summary of the look, its style, and
wearer: women if the person wearing it is a woman or girl, men if a man or
boy, unclear if there is no person or you cannot tell. The store opens the
matches in that section; the shopper can switch.

Work through the person in this order and list what is there: top, bottom or
dress or set, outerwear, footwear; then look closely at the wrists (watch,
bracelet, band, bangles), hands (rings), neck (chain, necklace, pendant),
ears (earrings), face (sunglasses), waist (belt), head (cap) and any bag.
Small accessories matter as much as clothes.

Ethnic wear: a kurta worn with a dupatta or matching bottoms is a Kurta Set
(put the dupatta in its details); a saree is a Saree; a long kurta is not a
dress.

Rules: one entry per item, at most 8. Skip what you cannot actually see. Never name a brand. Do not describe the
person's body, face, skin or hair, or the background. If there is no person or
no clothing, set person_detected false and return no items."""

VISION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "person_detected": {"type": "BOOLEAN"},
        "summary": {"type": "STRING"},
        "style": {"type": "STRING"},
        "wearer": {"type": "STRING", "enum": ["women", "men", "unclear"]},
        "items": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {
                "slot": {"type": "STRING", "enum": SLOTS},
                "article_type": {"type": "STRING"},
                "colour": {"type": "STRING"},
                "secondary_colour": {"type": "STRING", "nullable": True},
                "pattern": {"type": "STRING", "nullable": True},
                "details": {"type": "ARRAY", "items": {"type": "STRING"}},
                "worn_by": {"type": "STRING", "enum": ["menswear", "womenswear", "unisex"]},
                "description": {"type": "STRING"},
                "confidence": {"type": "NUMBER"},
            },
            "required": ["slot", "article_type", "colour", "worn_by", "description"],
        }},
    },
    "required": ["person_detected", "items"],
}

MAX_EDGE = 768          # px. Bounds the image tokens; enough to see a pendant.

# Which band of the figure each slot normally occupies, as fractions of the
# subject's height measured from the top of the head.
BANDS = {
    "headwear": (0.00, 0.12),
    "top":      (0.20, 0.48),
    "outerwear": (0.20, 0.52),
    "bottom":   (0.50, 0.88),
    "footwear": (0.90, 1.00),
}


@dataclass
class DetectedItem:
    slot: str
    label: str
    article_type: str | None = None
    colour: str | None = None
    secondary_colour: str | None = None
    pattern: str | None = None
    material: str | None = None
    worn_by: str | None = None       # menswear | womenswear | unisex
    wearer: str | None = None        # women | men - the person in the photo, not the cut
    confidence: float = 0.0
    source: str = "vision"
    rgb: list | None = None
    details: list = field(default_factory=list)
    description: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "DetectedItem":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})

    def dict(self) -> dict:
        return asdict(self)


# --- catalogue colour vocabulary -----------------------------------------
BASE_RGB = {
    "black": (26, 26, 26), "white": (245, 245, 245), "off white": (238, 234, 222),
    "grey": (138, 138, 138), "charcoal": (74, 78, 82), "navy blue": (28, 42, 86),
    "blue": (48, 96, 180), "teal": (36, 122, 122), "green": (56, 122, 64),
    "olive": (110, 112, 60), "red": (176, 42, 42), "maroon": (108, 30, 42),
    "pink": (222, 136, 160), "purple": (108, 70, 150), "lavender": (190, 176, 220),
    "yellow": (226, 196, 72), "mustard": (196, 154, 48), "orange": (214, 118, 48),
    "rust": (162, 80, 48), "brown": (112, 78, 56), "coffee brown": (82, 58, 44),
    "beige": (214, 196, 166), "cream": (238, 226, 198), "tan": (196, 158, 110),
    "khaki": (178, 162, 118), "gold": (198, 162, 72), "silver": (190, 194, 198),
    "peach": (240, 186, 158), "magenta": (186, 52, 130), "turquoise blue": (62, 186, 194),
}


def nearest_colour(rgb) -> str:
    r, g, b = [float(x) for x in rgb]
    best, bd = "black", 1e9
    for name, (nr, ng, nb) in BASE_RGB.items():
        d = (r - nr) ** 2 * 0.30 + (g - ng) ** 2 * 0.59 + (b - nb) ** 2 * 0.11
        if d < bd:
            best, bd = name, d
    return best


def _is_skin(r, g, b) -> bool:
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return 55 < y and 77 <= cb <= 132 and 132 <= cr <= 178 and r > g >= b - 12


def _skin_mask(img):
    import cv2
    b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
    # Skin is chromatic and not blown out. Without these two bounds a white or
    # cream shirt passes every YCbCr test and gets mistaken for a face, which
    # then anchors the whole body in the wrong place.
    m = ((y > 55) & (cb >= 77) & (cb <= 132) & (cr >= 132) & (cr <= 178)
         & (r > g) & (g >= b - 14)
         & (sat > 0.15) & (mx < 245)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _find_face(img):
    """Largest skin region in the upper half, as (cx, cy, w, h).

    This build of OpenCV has no cascade or HOG detector, so the face is found
    by skin colour and connected components. It is the anchor for everything
    else: without a face we do not know where the person is, and guessing
    produces the nonsense this replaced - "turquoise headwear" read off a
    night sky.
    """
    import cv2
    h, w = img.shape[:2]
    mask = _skin_mask(img)
    mask[int(h * 0.62):, :] = 0                 # faces are not at the ankles
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < (h * w) * 0.0012:
            continue
        if area > (h * w) * 0.10:               # too big to be a face
            continue
        ratio = bh / max(1, bw)
        if not (0.6 <= ratio <= 2.4):           # a face is roughly upright-oval
            continue
        # Prefer blobs that are high in the frame and near the centre: the face
        # sits above the hands and any other skin the photo shows.
        height_bonus = 1.0 - (cents[i][1] / h)
        centre_bonus = 1.0 - abs(cents[i][0] / w - 0.5)
        score = area * (0.45 + height_bonus) * centre_bonus
        if best is None or score > best[0]:
            best = (score, int(cents[i][0]), int(cents[i][1]), int(bw), int(bh))
    if not best:
        return None
    return best[1], best[2], best[3], best[4]


def detect_zones(image_bytes: bytes) -> list[DetectedItem]:
    """Fallback detector: anchor on the face, then read each band's colour.

    Always available - no model, no download. It knows WHERE an item sits, not
    what kind of garment it is, and it refuses to report anything when it
    cannot find a person.
    """
    import cv2
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    h, w = img.shape[:2]
    scale = 640.0 / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]

    face = _find_face(img)
    if not face:
        return []
    fx, fy, fw, fh = face
    head = max(fh, fw * 1.25)

    # Adult proportions: about 7.5 head-heights tall, shoulders ~1.6 heads wide.
    crown = max(0, fy - int(head * 0.55))
    body_px = head * 7.5
    # Shoulders are about two head-widths across, so sample the torso core
    # rather than a box wide enough to be mostly background.
    half_w = int(head * 0.95)
    x0, x1 = max(0, fx - half_w), min(w, fx + half_w)
    if x1 - x0 < 12:
        return []

    bg = np.median(np.concatenate([
        img[0:10, 0:10].reshape(-1, 3), img[0:10, -10:].reshape(-1, 3),
        img[-10:, 0:10].reshape(-1, 3), img[-10:, -10:].reshape(-1, 3)]), axis=0)

    skin = _skin_mask(img)
    bands = [
        ("headwear", -0.60, -0.15),
        ("top",       0.95,  2.90),
        ("bottom",    3.10,  5.90),
        ("footwear",  6.60,  7.40),
    ]
    out: list[DetectedItem] = []
    for slot, a, b in bands:
        y0 = crown + int(head * a)
        y1 = crown + int(head * b)
        y0, y1 = max(0, y0), min(h, y1)
        if y1 - y0 < 8:
            continue
        patch = img[y0:y1, x0:x1].reshape(-1, 3)
        spatch = skin[y0:y1, x0:x1].reshape(-1)
        # Drop skin AND anything that matches the background, before measuring
        # dominance - otherwise a busy backdrop decides the answer.
        notbg = np.linalg.norm(patch.astype(float) - bg, axis=1) > 46
        keep = (spatch == 0) & notbg
        if keep.sum() < 60 or keep.sum() < 0.12 * len(patch):
            continue
        px = patch[keep]
        data = np.float32(px)
        k = min(3, max(1, len(data) // 80))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 14, 1.0)
        _, labels, centres = cv2.kmeans(data, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.flatten(), minlength=k)
        idx = int(np.argmax(counts))
        c = centres[idx]
        share = float(counts[idx] / counts.sum())
        # A band that is mostly background, or has no dominant colour at all,
        # is not a garment.
        if float(np.linalg.norm(c - bg)) < 46:
            continue
        if share < 0.40:
            continue
        rgb = [int(c[2]), int(c[1]), int(c[0])]
        name = nearest_colour(rgb)
        out.append(DetectedItem(
            slot=slot, label=f"{name} item on the {slot}", colour=name,
            confidence=round(min(0.6, 0.2 + share * 0.45), 2),
            source="zones", rgb=rgb))
    return out


# ------------------------------------------------------------------ step 1
def prepare_image(data: bytes, max_edge: int = MAX_EDGE) -> tuple[bytes, list[int]]:
    """Downscale and re-encode before the model sees it.

    Image tokens grow with resolution; a 4,000px phone photo costs several
    times what a 768px one does and recognises a t-shirt no better. The page
    already shrinks the photo, this makes the bound hold for any client."""
    try:
        from PIL import Image, ImageOps
        im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_edge, max_edge))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85, optimize=True)
        return buf.getvalue(), [im.width, im.height]
    except Exception:                                   # noqa: BLE001
        return data, [0, 0]


def _catalogue_types() -> list[str]:
    seen = []
    for r in catalogue.rows:
        t = r.get("articleType")
        if t and t not in seen:
            seen.append(t)
    return seen


# Recognition results by image hash. Text only - the photo itself is never kept.
_SEEN: "OrderedDict[str, dict]" = OrderedDict()
_SEEN_MAX = 48


def detect_vision(image: bytes) -> tuple[list[DetectedItem], dict]:
    """One vision call. Returns the items and what the call cost."""
    types = _catalogue_types()
    system = VISION_SYSTEM.format(slots=", ".join(SLOTS),
                                  types=", ".join(types) or "any garment noun")
    kw = dict(images=[image], temperature=0.1, json_mode=True)
    if llm.active_name == "gemini":
        kw.update(response_schema=VISION_SCHEMA, max_output_tokens=2048)
    else:
        kw.update(model=settings.vision_model)
    reply = llm.chat(system, "Tag every item this person is wearing.", **kw)
    from ..llm.ollama import extract_json
    data = extract_json(reply.text)
    raw = data.get("items") if isinstance(data, dict) else data
    wearer = str(data.get("wearer") or "").lower() if isinstance(data, dict) else ""
    wearer = wearer if wearer in ("women", "men") else None
    items: list[DetectedItem] = []
    for it in (raw or [])[:8]:
        if not isinstance(it, dict):
            continue
        slot = str(it.get("slot") or "").lower().strip()
        if slot not in SLOT_CATEGORIES:
            slot = slot_of(str(it.get("article_type") or ""),
                           str(it.get("description") or it.get("label") or "")) or ""
        if not slot:
            continue
        details = [str(x) for x in (it.get("details") or []) if x][:6]
        desc = str(it.get("description") or it.get("label") or "").strip()
        items.append(DetectedItem(
            slot=slot,
            label=desc or f"{it.get('colour') or ''} {it.get('article_type') or slot}".strip(),
            article_type=(str(it["article_type"]) if it.get("article_type") else None),
            colour=(str(it["colour"]).lower() if it.get("colour") else None),
            secondary_colour=(str(it["secondary_colour"]).lower()
                              if it.get("secondary_colour") else None),
            pattern=(str(it["pattern"]) if it.get("pattern") else None),
            material=(str(it["material"]) if it.get("material") else None),
            worn_by=(str(it["worn_by"]).lower() if it.get("worn_by") else None),
            wearer=wearer,
            confidence=float(it.get("confidence") or 0.5),
            source="vision", details=details, description=desc or None))
    meta = {
        "model": reply.model,
        "promptTokens": reply.prompt_tokens,
        "outputTokens": reply.eval_tokens,
        "thoughtTokens": getattr(reply, "thought_tokens", 0),
        "seconds": round(reply.seconds, 2),
        "personDetected": bool(data.get("person_detected", True)) if isinstance(data, dict) else True,
        "summary": str(data.get("summary") or "") if isinstance(data, dict) else "",
        "style": str(data.get("style") or "") if isinstance(data, dict) else "",
        "wearer": wearer,
    }
    return items, meta


def detect(image_bytes: bytes, *, use_vision: bool = True) -> dict:
    """Step 1 with its cache. Never calls the model twice for the same photo."""
    small, size = prepare_image(image_bytes)
    # The prompt is part of the key: change the instructions and the same photo
    # is read afresh instead of returning an answer to the old question.
    key = hashlib.sha256(small + VISION_SYSTEM.encode()).hexdigest()
    if use_vision and key in _SEEN:
        _SEEN.move_to_end(key)
        hit = dict(_SEEN[key])
        hit["vision"] = dict(hit["vision"], cached=True)
        return hit

    detector, note, vision = "zones", "", {"called": False}
    items: list[DetectedItem] = []
    if use_vision:
        try:
            items, vision = detect_vision(small)
            vision.update(called=True, cached=False, imagePx=size,
                          imageKB=round(len(small) / 1024, 1))
            detector = "vision"
        except (LLMUnavailable, ValueError) as e:
            note = (f"Vision model unavailable ({e}). Fell back to colour-and-"
                    f"placement analysis, which reads colours off the photo but "
                    f"cannot name the garment type.")
        except Exception as e:                  # noqa: BLE001
            note = (f"Vision model failed ({type(e).__name__}: {e}). Fell back "
                    f"to colour-and-placement analysis.")
    if detector == "vision" and not items and vision.get("personDetected") is False:
        note = "No person or clothing was found in that photo."
    elif not items:
        items = detect_zones(image_bytes)
        if detector == "vision" and not note:
            note = "The vision model returned nothing; used zone analysis."
        detector = "zones"

    out = {"lookId": key[:16], "detector": detector, "note": note,
           "items": [it.dict() for it in items], "vision": vision}
    if detector == "vision":
        _SEEN[key] = out
        while len(_SEEN) > _SEEN_MAX:
            _SEEN.popitem(last=False)
    return out


# ------------------------------------------------------------------ step 2
# Catalogue audience values for each garment-gender the model may report.
_WORN_BY_AUDIENCE = {
    "menswear": "men", "mens": "men", "men": "men", "male": "men",
    "womenswear": "women", "womens": "women", "women": "women",
    "female": "women",
    "unisex": None, "boys": "men", "girls": "women",
}
_NOISE = {"and", "with", "the", "a", "of", "for", "men", "women", "mens", "womens",
          "unisex", "s", "in", "on", "style", "look", "wear"}


def look_audience(wearer: str | None, saved: str | None) -> tuple[str | None, str]:
    """Which section the matches open in, and why.

    The person in the photo first: a girl's photo opens on women's, a man's on
    men's. A photo with no clear person (a flat lay) uses the shopper's saved
    choice, else all. The shopper switches with Women / Men / All afterwards."""
    if wearer in ("women", "men"):
        return wearer, "photo"
    if saved in ("women", "men"):
        return saved, "saved"
    return None, "none"


_GENDER_WORDS = re.compile(r"\b(men|women|man|woman|boys?|girls?|male|female|"
                           r"unisex|mens|womens)\b(?:'s|')?", re.I)


def query_text(it: DetectedItem, audience: str | None = None) -> str:
    """The words the search runs on. The model's one-line description first,
    then anything it listed that the description left out. Its gender words
    are swapped for the audience being shopped, so "men black jeans" on a girl
    searches as "women's black jeans"."""
    parts = [_GENDER_WORDS.sub("", it.description or it.label or "")]
    parts[0] = re.sub(r"\s{2,}", " ", parts[0]).strip()
    for x in [it.article_type, it.colour, it.secondary_colour, it.pattern,
              it.material] + list(it.details or []):
        if x and str(x).lower() not in parts[0].lower():
            parts.append(str(x))
    if audience in ("men", "women"):
        parts.insert(0, audience + "'s")
    return " ".join(p for p in parts if p).strip()


def _row_desc(r: dict) -> str:
    try:
        return (json.loads(r.get("raw") or "{}") or {}).get("desc") or ""
    except Exception:                                   # noqa: BLE001
        return ""


def _public(r: dict) -> dict:
    """A catalogue row as the page wants it - the raw JSON merged back in."""
    out = dict(r)
    raw = out.pop("raw", None)
    out.pop("doc", None)
    try:
        out.update({k: v for k, v in (json.loads(raw) or {}).items()
                    if k not in out or not out[k]})
    except Exception:                                   # noqa: BLE001
        pass
    return out


def match_item(item: DetectedItem, *, limit: int = 6, audience: str | None = None,
               sem: dict[str, float] | None = None) -> list[dict]:
    """Rank catalogue products against one detected item. Deterministic.

    Hard: the slot, and with an audience ("women"/"men") that section only -
    no unisex rows and no fallback to the other section, even if that leaves
    few or none. None is "All". Soft, summed to a 0-100 score: meaning
    (embeddings), shared words (BM25), the same article type, the same or a
    related colour, and under All the garment's cut.
    """
    rows = catalogue.rows
    cats = SLOT_CATEGORIES.get(item.slot, [])
    pool = [i for i, r in enumerate(rows)
            if (r["slot"] == item.slot) or (not r["slot"] and r["cat"] in cats)]
    if not pool and cats:
        pool = [i for i, r in enumerate(rows) if r["cat"] in cats]
    if audience:
        pool = [i for i in pool if rows[i]["audience"] == audience]
    cut = _WORN_BY_AUDIENCE.get(item.worn_by or "")

    q = query_text(item, audience)
    bm = catalogue.bm25(q) if q else {}
    bm_max = max((bm.get(i, 0.0) for i in pool), default=0.0) or 1.0
    sims = [sem.get(rows[i]["id"], 0.0) for i in pool] if sem else []
    s_lo, s_hi = (min(sims), max(sims)) if sims else (0.0, 0.0)
    want_colour = canonical_colour(item.colour) or canonical_colour(item.description)
    want_type = (item.article_type or "").strip().lower()
    q_words = {w for w in tokenise(q) if len(w) > 2 and w not in _NOISE}

    w_sem, w_lex, w_type, w_col, w_aud = ((0.45, 0.20, 0.15, 0.15, 0.05) if sem
                                         else (0.0, 0.45, 0.25, 0.25, 0.05))
    scored = []
    for k, i in enumerate(pool):
        r = rows[i]
        s_sem = ((sims[k] - s_lo) / (s_hi - s_lo)) if sem and s_hi > s_lo else 0.0
        s_lex = bm.get(i, 0.0) / bm_max
        rt = (r.get("articleType") or "").lower()
        s_type = 1.0 if want_type and (want_type == rt or want_type.rstrip("s") == rt.rstrip("s")) else 0.0
        s_col = colour_closeness(want_colour, r.get("colour"))
        # Under All, lean towards the garment's cut; otherwise every row fits.
        s_aud = 1.0 if audience or not cut or r["audience"] in (cut, "unisex") else 0.0
        score = w_sem * s_sem + w_lex * s_lex + w_type * s_type + w_col * s_col + w_aud * s_aud
        scored.append((score, i, s_sem, s_type, s_col))

    scored.sort(key=lambda t: (-t[0], t[1]))
    out, shown = [], set()
    for score, i, s_sem, s_type, s_col in scored:
        if len(out) >= limit:
            break
        r = rows[i]
        sig = (r.get("brand"), r.get("name"), r.get("colour"), r.get("price"))
        if sig in shown:
            continue
        shown.add(sig)
        why = []
        if s_type:
            why.append(f"same type ({r['articleType']})")
        if s_col == 1.0:
            why.append(f"same colour ({r['colour']})")
        elif s_col:
            why.append(f"{r['colour']} is close to {want_colour}")
        shared = [w for w in tokenise(_row_desc(r)) if w in q_words and
                  w not in (want_type or "") and not canonical_colour(w)]
        shared = list(dict.fromkeys(shared))[:3]
        if shared:
            why.append("also " + ", ".join(shared))
        if not why and s_sem > 0.7:
            why.append("closest in description")
        p = _public(r)
        out.append({k: p.get(k) for k in (
            "id", "name", "brand", "articleType", "colour", "audience", "price",
            "mrp", "discount", "rating", "ratingTotal", "image", "photo", "url", "slot")}
            | {"score": round(max(0.0, score) * 100), "why": why or ["same part of the outfit"]})
    return out


def match_all(items: list[DetectedItem], *, per_item: int = 6,
              audience: str | None = None) -> tuple[list[dict], dict]:
    """Step 2 for every item, with ONE embedding request for the whole look."""
    audience = audience if audience in ("women", "men") else None
    queries = [query_text(it, audience) for it in items]
    sims, sem_meta = semantic.index.query(queries) if items else (None, {})
    results = []
    for k, it in enumerate(items):
        results.append({
            "detected": it.dict(),
            "query": queries[k],
            "matches": match_item(it, limit=per_item, audience=audience,
                                  sem=(sims[k] if sims else None)),
        })
    sem_meta = dict(sem_meta, used=bool(sims), index=semantic.index.status())
    return results, sem_meta


LIMITS = {
    "vision": ("Matches are StyleMate products that resemble what was detected. "
               "They are not the same product, and no brand is identified from a "
               "photograph."),
    "zones": ("Without a vision model this reads the dominant colour of each part "
              "of the figure. It knows WHERE an item sits, not what kind of "
              "garment it is."),
}


def analyse(image_bytes: bytes, *, audience: str | None = None,
            per_item: int = 6, use_vision: bool = True) -> dict:
    """Full pipeline: see once, then match in code.

    `audience` is the shopper's saved "shopping for" choice, used only when
    the photo shows no clear person. See look_audience()."""
    t0 = time.time()
    det = detect(image_bytes, use_vision=use_vision)
    items = [DetectedItem.from_dict(d) for d in det["items"]]
    v = det["vision"]
    aud, aud_from = look_audience(v.get("wearer"), audience)
    results, sem_meta = match_all(items, per_item=per_item, audience=aud)
    return {
        "lookId": det["lookId"],
        "detector": det["detector"],
        "note": det["note"] or (
            "Recognised by " + str(v.get("model")) + ", matched in code against "
            f"{len(catalogue)} SKUs" + (" by meaning and keywords." if sem_meta.get("used")
                                         else " by keywords, type and colour.")),
        "summary": v.get("summary", ""),
        "style": v.get("style", ""),
        "wearer": v.get("wearer"),
        "audience": aud or "all",          # women | men | all
        "audienceFrom": aud_from,          # photo | saved | none
        "items": results,
        "usage": {"vision": v, "semantic": sem_meta,
                  "seconds": round(time.time() - t0, 2),
                  "catalogueSkus": len(catalogue)},
        "limits": LIMITS[det["detector"]],
    }


def rematch(items: list[dict], *, audience: str | None = None,
            per_item: int = 6) -> dict:
    """Step 2 again from step 1's text. No image, no vision tokens."""
    t0 = time.time()
    objs = [DetectedItem.from_dict(d) for d in items if isinstance(d, dict) and d.get("slot")]
    # The shopper picked Women, Men or All; "all" (or nothing) is no filter.
    results, sem_meta = match_all(objs, per_item=per_item, audience=audience)
    return {"audience": audience if audience in ("women", "men") else "all",
            "items": results,
            "usage": {"vision": {"called": False}, "semantic": sem_meta,
                      "seconds": round(time.time() - t0, 2),
                      "catalogueSkus": len(catalogue)}}
