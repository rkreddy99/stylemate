"""AI search: free text -> structured intent -> ranked catalogue results.

Two stages, deliberately separated:

  UNDERSTANDING (model)  - turn a sentence into a QuerySpec. This is where a
      language model earns its place: "a wedding dress for December under 2000
      for my sister" has budget, occasion, season and audience folded into
      ordinary English, and "something that goes with silver jewellery" has no
      searchable keyword in it at all.

  RETRIEVAL (deterministic) - score the catalogue against that spec. BM25 over
      product text, plus colour affinity, occasion->usage, month->season and
      slot rules from compat.py. Reproducible, explainable, fast.

A model is never asked whether a product matches a budget. It is asked what
the sentence means. The rules do the rest.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Any

from ..catalogue.compat import (MONTH_NAMES, METAL_AFFINITY, colours_for_metal,
                                complement_slots, goes_with_colour, slot_of,
                                season_for_month, usages_for_occasion,
                                SLOT_CATEGORIES)
from ..catalogue.store import catalogue, tokenise
from ..llm.router import LLMUnavailable, llm


@dataclass
class QuerySpec:
    raw: str = ""
    intent: str = "find"                     # find | pair | browse
    audience: str | None = None              # men | women | unisex
    occasion: str | None = None
    month: int | None = None
    season: str | None = None
    budget_min: float | None = None
    budget_max: float | None = None
    categories: list[str] = field(default_factory=list)
    slots: list[str] = field(default_factory=list)
    article_types: list[str] = field(default_factory=list)
    colours: list[str] = field(default_factory=list)
    exclude_colours: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    pair_anchor: str | None = None           # "silver jewellery", "a skirt"
    pair_slot: str | None = None             # slot of the anchor
    pair_colour: str | None = None
    pair_metal: str | None = None
    for_whom: str | None = None
    source: str = "rules"                    # rules | model
    notes: list[str] = field(default_factory=list)

    def dict(self) -> dict:
        return asdict(self)


SYSTEM = """You turn a shopper's sentence into a JSON search specification for \
an Indian fashion catalogue. You do not choose products. You only describe what \
the sentence asks for.

Return ONLY this JSON object:
{
 "intent": "find" | "pair" | "browse",
 "audience": "men" | "women" | "unisex" | null,
 "occasion": string|null,          // wedding, office, gym, party, festival...
 "month": 1-12 or null,            // if a month or "this December" is named
 "budget_min": number|null,
 "budget_max": number|null,        // rupees; "under 2000" -> 2000; "2k" -> 2000
 "categories": [string],           // ONLY from the categories list given
 "slots": [string],                // ONLY from the slots list given
 "article_types": [string],        // ONLY from the article types list given
 "colours": [string],              // ONLY from the colours list given
 "exclude_colours": [string],
 "keywords": [string],             // descriptive words: slim, printed, floral
 "pair_anchor": string|null,       // the thing they ALREADY have, if any
 "pair_colour": string|null,
 "pair_metal": "silver"|"gold"|"rose gold"|"platinum"|"oxidised"|null,
 "for_whom": string|null           // "myself", "my sister", "my wife"
}

Rules that matter:
- intent="pair" whenever the sentence is about something they ALREADY HAVE and \
what to put WITH it. All of these are pair queries, and X is the pair_anchor:
    "goes with X" / "to wear with X" / "matches my X"
    "what suits me better with X" / "what will suit me of a X"
    "what should I wear with X" / "what looks good on X"
    "styling my X" / "pair my X" / "X - what top?"
  X is what they OWN. NEVER put X in article_types, categories, slots or \
keywords: returning another X is the single worst answer to a pair query. If \
they say "blue jeans", pair_anchor is "blue jeans" and pair_colour is "blue".
- If a pair query names no target garment, leave slots empty - the system will \
work out what goes with the anchor.
- Jewellery metals: set pair_metal, and leave colours empty. The system knows \
which colours suit each metal.
- Only use vocabulary from the lists supplied. If the sentence names something \
not in the lists, put it in keywords instead of inventing a value.
- "for her"/"for my sister"/"for women" -> audience "women". "for me" with no \
other signal -> null.
"""


def _facet_block() -> str:
    f = catalogue.facets()
    return (
        f"categories: {f['categories']}\n"
        f"slots: {f['slots']}\n"
        f"article types: {f['articleTypes'][:60]}\n"
        f"colours: {f['colours']}\n"
        f"usages: {f['usages']}\n"
        f"price range in rupees: {f['priceRange']}\n"
        f"catalogue size: {f['count']} products"
    )


# --- deterministic parser (always runs; the model refines it) -------------
_BUDGET = [
    (re.compile(r"(?:under|below|less than|within|upto|up to|max(?:imum)?)\s*"
                r"(?:rs\.?|inr|₹)?\s*([\d,]+)\s*(k|thousand)?", re.I), "max"),
    (re.compile(r"(?:above|over|more than|at ?least|min(?:imum)?)\s*"
                r"(?:rs\.?|inr|₹)?\s*([\d,]+)\s*(k|thousand)?", re.I), "min"),
    (re.compile(r"(?:budget(?:\s+of)?|around|about)\s*"
                r"(?:rs\.?|inr|₹)?\s*([\d,]+)\s*(k|thousand)?", re.I), "max"),
    (re.compile(r"₹\s*([\d,]+)\s*(k|thousand)?", re.I), "max"),
]
_PAIR = re.compile(
    r"(?:goes?|go|going|look)[a-z ]*?\s(?:good|well|nice|great)?\s*"
    r"(?:with|on|over|under)\s+(?:my |a |an |the |some )?([a-z \-]{2,40})"
    r"|(?:to (?:wear|go) with|pair(?:s|ed)? with|match(?:es|ing)? (?:my |a |the )?)"
    r"\s*(?:my |a |an |the |some )?([a-z \-]{2,40})", re.I)
_WOMEN = re.compile(r"\b(women|woman|female|ladies|girl|her|sister|wife|mom|"
                    r"mother|daughter|she)\b", re.I)
_MEN = re.compile(r"\b(men|man|male|gents|boy|him|brother|husband|dad|father|"
                  r"son|he)\b", re.I)


def _money(m: re.Match) -> float:
    v = float(m.group(1).replace(",", ""))
    if m.group(2):
        v *= 1000
    return v


# "what will suit me better of a blue jeans", "what should I wear with a skirt"
_PAIR_SUIT = re.compile(
    r"(?:what|which)[a-z ,']*?(?:suit|go|look|wear|pair|style)[a-z ,']*?"
    r"\b(?:with|of|on|for)\s+(?:a |an |the |my |some )?([a-z \-]{2,40})", re.I)
# "styling my denim jacket", "pair my white shirt"
_PAIR_VERB = re.compile(
    r"\b(?:styl(?:e|ing)|pair(?:ing)?|accessoris(?:e|ing))\s+"
    r"(?:a |an |the |my |some )?([a-z \-]{2,40})", re.I)


def parse_rules(text: str) -> QuerySpec:
    spec = QuerySpec(raw=text, source="rules")
    low = (text or "").lower()

    for rx, which in _BUDGET:
        m = rx.search(text or "")
        if m:
            v = _money(m)
            if which == "max" and spec.budget_max is None:
                spec.budget_max = v
            elif which == "min" and spec.budget_min is None:
                spec.budget_min = v

    for name, num in MONTH_NAMES.items():
        if re.search(rf"\b{name}\b", low):
            spec.month = num
            break

    if _WOMEN.search(low) and not _MEN.search(low):
        spec.audience = "women"
    elif _MEN.search(low) and not _WOMEN.search(low):
        spec.audience = "men"

    for occ in ("wedding", "sangeet", "mehendi", "reception", "festival",
                "diwali", "navratri", "puja", "office", "work", "interview",
                "meeting", "party", "club", "date", "brunch", "gym",
                "workout", "running", "yoga", "travel", "beach", "casual"):
        if re.search(rf"\b{occ}\b", low):
            spec.occasion = occ
            break

    m = _PAIR.search(text or "")
    anchor = ""
    if m:
        anchor = (m.group(1) or m.group(2) or "").strip(" .,")
    if not anchor:
        for rx in (_PAIR_SUIT, _PAIR_VERB):
            m2 = rx.search(text or "")
            if m2:
                anchor = (m2.group(1) or "").strip(" .,")
                if anchor:
                    break
    if anchor:
        # Trim trailing qualifiers: "blue jeans for casual date" -> "blue jeans"
        anchor = re.split(r"\b(?:for|under|below|in|on|at|with|to|and)\b",
                          anchor, flags=re.I)[0].strip(" .,")
        anchor = anchor.lower()
        if anchor:
            spec.intent = "pair"
            spec.pair_anchor = anchor
            for metal in METAL_AFFINITY:
                if metal in anchor:
                    spec.pair_metal = metal
            spec.pair_slot = slot_of(anchor, anchor)
            # The anchor's colour matters for what goes with it.
            for colour in catalogue.facets()["colours"]:
                if colour.lower() in anchor.lower():
                    spec.pair_colour = colour
                    break

    facets = catalogue.facets()
    for colour in facets["colours"]:
        if re.search(rf"\b{re.escape(colour.lower())}\b", low):
            if not (spec.pair_anchor and colour.lower() in spec.pair_anchor):
                spec.colours.append(colour)
    for at in facets["articleTypes"]:
        if re.search(rf"\b{re.escape(at.lower())}s?\b", low):
            if not (spec.pair_anchor and at.lower() in spec.pair_anchor):
                spec.article_types.append(at)

    stop = {"i", "a", "an", "the", "for", "with", "my", "need", "want", "some",
            "something", "under", "below", "in", "on", "to", "me", "and",
            "that", "goes", "good", "well", "looking", "would", "is", "it",
            "this", "of", "please", "give", "show", "find", "buy", "get",
            "wear", "wearing", "go", "going", "match", "matches", "pair",
            "paired", "nice", "great", "which", "anything", "budget"}
    # Words belonging to the thing they ALREADY own are not search terms.
    # Without this, "goes with silver jewellery" retrieves silver clothing.
    anchor_tokens = set(tokenise(spec.pair_anchor or ""))
    spec.keywords = [t for t in tokenise(low)
                     if t not in stop and t not in anchor_tokens
                     and not t.isdigit() and len(t) > 2][:12]
    # A garment noun in the request pins the slot: "wedding dress" must not
    # come back as a skirt, however well the skirt scores on occasion.
    SLOT_NOUNS = {
        "dress": "dress", "gown": "dress", "saree": "dress", "lehenga": "dress",
        "shirt": "top", "tshirt": "top", "t-shirt": "top", "top": "top",
        "kurta": "top", "kurti": "top", "blouse": "top", "sweater": "top",
        "jeans": "bottom", "trouser": "bottom", "pant": "bottom",
        "skirt": "bottom", "shorts": "bottom", "legging": "bottom",
        "jacket": "outerwear", "blazer": "outerwear", "coat": "outerwear",
        "shoe": "footwear", "shoes": "footwear", "sneaker": "footwear",
        "heel": "footwear", "heels": "footwear", "sandal": "footwear",
        "earring": "jewellery", "necklace": "jewellery", "bangle": "jewellery",
        "watch": "watch", "bag": "bag", "handbag": "bag", "belt": "belt",
    }
    if not spec.slots:
        for noun, slot in SLOT_NOUNS.items():
            if re.search(rf"\b{re.escape(noun)}s?\b", low):
                if spec.pair_anchor and noun in spec.pair_anchor:
                    continue          # that is the anchor, not the target
                spec.slots.append(slot)
        spec.slots = sorted(set(spec.slots))
    spec.season = season_for_month(spec.month)
    return spec


def parse(text: str, use_model: bool = True) -> QuerySpec:
    """Rules first, then let the model refine. The model never removes a
    budget or a month the rules already proved from the text."""
    spec = parse_rules(text)
    if not use_model:
        return spec
    try:
        data, reply = llm.chat_json(
            SYSTEM,
            f"Catalogue vocabulary:\n{_facet_block()}\n\n"
            f"Today is {date.today().isoformat()}.\n"
            f'Shopper said: "{text}"',
            temperature=0.1)
    except (LLMUnavailable, ValueError) as e:
        spec.notes.append(f"Model unavailable, used rule parsing only: {e}")
        return spec
    except Exception as e:                      # noqa: BLE001
        # Understanding is an enhancement; retrieval must still answer.
        spec.notes.append(
            f"Model call failed ({type(e).__name__}: {e}); "
            f"used rule parsing only.")
        return spec
    if not isinstance(data, dict):
        spec.notes.append("Model returned an unexpected shape; kept rules.")
        return spec

    spec.source = "model"
    spec.intent = str(data.get("intent") or spec.intent)
    spec.audience = data.get("audience") or spec.audience
    spec.occasion = data.get("occasion") or spec.occasion
    spec.for_whom = data.get("for_whom") or spec.for_whom
    if spec.month is None and isinstance(data.get("month"), (int, float)):
        spec.month = int(data["month"])
    # Budgets proved by regex win: the model sometimes rounds them.
    if spec.budget_max is None and isinstance(data.get("budget_max"), (int, float)):
        spec.budget_max = float(data["budget_max"])
    if spec.budget_min is None and isinstance(data.get("budget_min"), (int, float)):
        spec.budget_min = float(data["budget_min"])

    def take(key, cur):
        v = data.get(key)
        return sorted({*(cur or []), *[str(x) for x in v if x]}) if isinstance(v, list) else cur

    spec.categories = take("categories", spec.categories)
    spec.slots = take("slots", spec.slots)
    spec.article_types = take("article_types", spec.article_types)
    spec.colours = take("colours", spec.colours)
    spec.exclude_colours = take("exclude_colours", spec.exclude_colours)
    spec.keywords = take("keywords", spec.keywords)
    if data.get("pair_anchor"):
        spec.intent = "pair"
        spec.pair_anchor = str(data["pair_anchor"])
        spec.pair_slot = spec.pair_slot or slot_of(spec.pair_anchor,
                                                   spec.pair_anchor)
    spec.pair_metal = data.get("pair_metal") or spec.pair_metal
    spec.pair_colour = data.get("pair_colour") or spec.pair_colour
    spec.season = season_for_month(spec.month) or spec.season
    spec.notes.append(
        f"Understood by {reply.model} in {reply.seconds:.1f}s "
        f"({reply.prompt_tokens} prompt tokens).")
    if reply.truncated:
        spec.notes.append(
            "The prompt filled the context window - raise SM_NUM_CTX.")
    return spec


ANSWER_SYSTEM = """You are StyleMate's shopping agent, talking to one shopper \
in a fashion app. You are given their request and the products the catalogue \
actually returned, with colours, prices and categories.

Write a short, direct answer: 2-4 sentences, no headings, no lists, no \
markdown. Say what you would wear and why it works - colour, occasion, \
formality. Name at most two of the products, exactly as they are written in \
the data. Speak to them as "you".

Hard rules:
- Only mention products from the list you were given. Never invent one.
- Never quote a price that is not in the data.
- If they already own something (the anchor), talk about what goes WITH it and \
do not suggest another one of the same thing.
- If the list is empty, say so plainly in one sentence and suggest loosening \
the budget or the occasion.
- No preamble like "Certainly" or "Here are". Start with the advice."""


def compose_answer(spec: QuerySpec, results: list[dict]) -> tuple[str, bool]:
    """A short styling answer from the model. Returns (text, from_model)."""
    if not results:
        return ("Nothing in the catalogue matched that. Try widening the "
                "budget, or naming the occasion differently."), False
    lines = []
    for r in results[:10]:
        lines.append(
            f"- {r.get('name')} | {r.get('colour')} | {r.get('cat')} | "
            f"{r.get('slot') or 'item'} | Rs {int(r.get('price') or 0)} | "
            f"{r.get('usage') or 'any occasion'}")
    brief = [f'Shopper said: "{spec.raw}"']
    if spec.pair_anchor:
        brief.append(f"They already own: {spec.pair_anchor}. These are things "
                     f"to wear WITH it.")
    if spec.occasion:
        brief.append(f"Occasion: {spec.occasion}.")
    if spec.budget_max:
        brief.append(f"Budget cap: Rs {int(spec.budget_max)}.")
    if spec.season:
        brief.append(f"Season: {spec.season}.")
    try:
        reply = llm.chat(ANSWER_SYSTEM,
                         "\n".join(brief) + "\n\nProducts returned:\n"
                         + "\n".join(lines),
                         temperature=0.4)
        text = (reply.text or "").strip()
        if text:
            return text, True
    except Exception:                           # noqa: BLE001
        pass
    return _fallback_answer(spec, results), False


def _fallback_answer(spec: QuerySpec, results: list[dict]) -> str:
    """Written from the data when no model is connected."""
    top = results[0]
    bits = []
    if spec.pair_anchor:
        bits.append(
            f"With {spec.pair_anchor}, the safest move is a "
            f"{(top.get('slot') or 'piece')} that does not fight it \u2014 "
            f"the {top.get('name')} in {top.get('colour')} works")
    else:
        bits.append(f"The {top.get('name')} in {top.get('colour')} is the "
                    f"closest match")
    if spec.occasion:
        bits.append(f"and reads right for a {spec.occasion}")
    if spec.budget_max:
        bits.append(f"at Rs {int(top.get('price') or 0)}, inside your "
                    f"Rs {int(spec.budget_max)} budget")
    out = ", ".join(bits) + "."
    if len(results) > 1:
        second = results[1]
        out += (f" If you want a second option, the {second.get('name')} in "
                f"{second.get('colour')} is the same idea in another colour.")
    out += (" Connect a model under Agent model for a fuller styling answer.")
    return out


# --- retrieval ------------------------------------------------------------
def _allowed_slots(spec: QuerySpec) -> set[str] | None:
    if spec.slots:
        return set(spec.slots)
    if spec.intent == "pair" and spec.pair_slot:
        return set(complement_slots(spec.pair_slot))
    if spec.pair_metal:
        # Jewellery anchors ask for clothing, not more jewellery.
        return set(complement_slots("jewellery"))
    return None


def search(text: str, *, limit: int = 24, use_model: bool = True,
           profile: dict | None = None) -> dict:
    spec = parse(text, use_model=use_model)
    rows = catalogue.rows
    if not rows:
        return {"spec": spec.dict(), "results": [], "total": 0,
                "explanation": "The catalogue is empty. Sync it first.",
                "filtersApplied": []}

    # When the shopper does not say who it is for, shop for THEM. Without this
    # a man asking what goes with his jeans was offered women's blazers.
    if not spec.audience and profile:
        g = str(profile.get("gender") or "").lower()
        if g.startswith("f"):
            spec.audience = "women"
            spec.notes.append("Shopping for you (women's), from your saved profile.")
        elif g.startswith("m"):
            spec.audience = "men"
            spec.notes.append("Shopping for you (men's), from your saved profile.")

    applied: list[str] = []
    idx = list(range(len(rows)))

    if spec.budget_max is not None:
        idx = [i for i in idx if (rows[i]["price"] or 0) <= spec.budget_max]
        applied.append(f"price at or under Rs {int(spec.budget_max):,}")
    if spec.budget_min is not None:
        idx = [i for i in idx if (rows[i]["price"] or 0) >= spec.budget_min]
        applied.append(f"price at or above Rs {int(spec.budget_min):,}")
    if spec.audience:
        # Strict: "women" means women's products only, not unisex ones too,
        # even when that leaves fewer to choose from.
        idx = [i for i in idx if rows[i]["audience"] == spec.audience]
        applied.append(f"{spec.audience}'s only")

    slots = _allowed_slots(spec)
    if slots:
        keep = [i for i in idx if rows[i]["slot"] in slots]
        if keep:
            idx = keep
            if spec.intent == "pair" and spec.pair_anchor:
                applied.append(
                    f"only things worn WITH {spec.pair_anchor} "
                    f"({', '.join(sorted(slots))}) - no more "
                    f"{spec.pair_slot or 'of the same kind'}")
            else:
                applied.append("slots: " + ", ".join(sorted(slots)))

    if spec.categories:
        keep = [i for i in idx if rows[i]["cat"] in spec.categories]
        if keep:
            idx, _ = keep, applied.append("categories: " + ", ".join(spec.categories))

    want_colours = [c.lower() for c in spec.colours]
    if spec.pair_metal and not want_colours:
        want_colours = [c.lower() for c in colours_for_metal(spec.pair_metal)]
        if want_colours:
            applied.append(
                f"colours that sit well against {spec.pair_metal}: "
                + ", ".join(want_colours[:6]))
    if spec.exclude_colours:
        ex = {c.lower() for c in spec.exclude_colours}
        idx = [i for i in idx if (rows[i]["colour"] or "").lower() not in ex]
        applied.append("excluding " + ", ".join(spec.exclude_colours))

    usages = usages_for_occasion(spec.occasion)
    query_text = " ".join(spec.keywords + spec.article_types)
    bm = catalogue.bm25(query_text) if query_text else {}
    bm_max = max(bm.values()) if bm else 1.0

    scored = []
    for i in idx:
        r = rows[i]
        s, why = 0.0, []
        lex = (bm.get(i, 0.0) / bm_max) if bm_max else 0.0
        s += lex * 3.2
        if lex > 0.35:
            why.append("matches the words you used")

        colour = (r["colour"] or "").lower()
        if want_colours:
            if colour in want_colours:
                rank = want_colours.index(colour)
                s += 2.6 - rank * 0.08
                why.append(
                    f"{r['colour']} works against {spec.pair_metal}"
                    if spec.pair_metal else f"{r['colour']} is a colour you asked for")
            elif spec.pair_colour:
                s += goes_with_colour(spec.pair_colour, colour) * 1.2
        if spec.pair_colour and colour:
            aff = goes_with_colour(spec.pair_colour, colour)
            if aff >= 0.8:
                why.append(f"{r['colour']} sits well with {spec.pair_colour}")

        if usages and r["usage"]:
            if r["usage"] in usages:
                s += 2.0
                why.append(f"{r['usage']} pieces suit a {spec.occasion}")
            else:
                s -= 0.5
        if spec.season and r["season"]:
            if r["season"] == spec.season:
                s += 0.9
                why.append(f"{r['season']} piece, which fits {_month_name(spec.month)}")
        if spec.article_types and r["articleType"] in spec.article_types:
            s += 1.6
            why.append(f"it is a {r['articleType'].lower()}")
        if profile:
            pal = [c.lower() for c in (profile.get("paletteColours") or [])]
            if pal and colour in pal:
                s += 0.8
                why.append("in the palette you saved")

        # With a pair anchor there are often no search words at all, so rank
        # on how well the piece actually completes the outfit instead of
        # leaving every score at zero.
        if spec.intent == "pair" and spec.pair_anchor:
            slot = r["slot"] or ""
            order = complement_slots(spec.pair_slot or "jewellery")
            if slot in order:
                s += 1.8 - order.index(slot) * 0.12
                if not why:
                    why.append(
                        f"a {slot} is the usual next piece with "
                        f"{spec.pair_anchor}")
            if colour:
                neutral_bonus = 0.9 if colour in (
                    "black", "white", "navy blue", "grey", "charcoal",
                    "beige", "cream", "off white") else 0.0
                s += neutral_bonus
                if neutral_bonus and len(why) < 2:
                    why.append(f"{r['colour']} is easy to pair")
            if r["usage"] in ("Party", "Formal", "Smart Casual"):
                s += 0.3
        scored.append((s, i, why))

    # Tie-break on catalogue order, not price: sorting ties by price pushed the
    # cheapest row of every group to the top regardless of relevance.
    scored.sort(key=lambda t: (-t[0], t[1]))
    top = scored[:limit]
    results = []
    for s, i, why in top:
        r = dict(rows[i])
        raw = r.pop("raw", None)
        try:
            r.update({k: v for k, v in (__import__("json").loads(raw) or {}).items()
                      if k not in r or not r[k]})
        except Exception:                      # noqa: BLE001
            pass
        r["score"] = round(s, 3)
        r["why"] = why[:3]
        results.append(r)

    answer, from_model = (compose_answer(spec, results) if use_model
                          else (_fallback_answer(spec, results) if results
                                else "Nothing matched that.", False))
    return {
        "spec": spec.dict(),
        "results": results,
        "total": len(idx),
        "filtersApplied": applied,
        "answer": answer,
        "answerFromModel": from_model,
        "explanation": _explain(spec, applied, len(idx), len(results)),
    }


def _month_name(m: int | None) -> str:
    return {v: k for k, v in MONTH_NAMES.items() if len(k) > 3}.get(m, "that month") if m else "the season"


def _explain(spec: QuerySpec, applied: list[str], total: int, shown: int) -> str:
    bits = []
    if spec.intent == "pair" and spec.pair_anchor:
        bits.append(
            f"You already have {spec.pair_anchor}, so this looks for things to "
            f"wear with it rather than more of the same.")
    if spec.occasion:
        bits.append(f"Read as a {spec.occasion}.")
    if spec.month:
        bits.append(f"{_month_name(spec.month).title()} reads as {spec.season}.")
    if spec.budget_max:
        bits.append(f"Capped at Rs {int(spec.budget_max):,}.")
    bits.append(f"{total} products matched; showing {shown}.")
    return " ".join(bits)
