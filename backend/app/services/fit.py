"""Deterministic garment-fit engine.

Every number here is arithmetic on measurements. No LLM is involved in deciding
whether something fits - a language model may later phrase the result, but it
never computes it.

Core idea: EASE = garment measurement - body measurement, at the same landmark.
Ease is judged per dimension against bands that differ by garment family and by
landmark (a jacket needs more chest ease than a t-shirt; a waistband needs
almost none). Where a dimension is missing from either side we report it as
missing and lower the confidence rather than filling it in.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from ..data.size_charts import charts, stretch_for, FAMILY_DIMS

CM_PER_IN = 2.54

# Which body measurement answers which garment dimension.
DIM_TO_BODY = {
    "chest": "chest", "waist": "waist", "hip": "hips",
    "shoulder": "shoulder", "length": "frontLength",
    "footLength": "foot", "inseam": "inseam", "thigh": "thigh",
}
# Dimensions we report but never compare, because no shopper saves them.
INFO_ONLY = {"sleeve"}

# Ease bands in cm: (too_tight_below, snug_below, regular_below, relaxed_below).
# Above the last value reads as oversized.
EASE_BANDS: dict[tuple[str, str], tuple[float, float, float, float]] = {
    ("top", "chest"):     (2, 6, 12, 20),
    ("shirt", "chest"):   (4, 8, 15, 24),
    ("kurta", "chest"):   (4, 9, 16, 26),
    ("outer", "chest"):   (6, 12, 20, 30),
    ("dress", "chest"):   (0, 4, 10, 18),
    ("dress", "waist"):   (0, 4, 10, 18),
    ("dress", "hip"):     (1, 5, 11, 19),
    ("bottom", "waist"):  (-1, 1, 4, 9),
    ("bottom", "hip"):    (1, 4, 9, 16),
    ("bottom", "thigh"):  (1, 4, 9, 16),
    ("skirt", "waist"):   (-1, 1, 4, 9),
    ("skirt", "hip"):     (1, 4, 9, 16),
}
# Shoulder is a seam placement, not a circumference: the tolerance is tiny and
# symmetric, so it gets its own rule rather than an ease band.
SHOULDER_TOL = 1.5      # cm either side of the body shoulder width
# Foot length wants a toe allowance rather than "ease".
TOE_ALLOWANCE = (0.5, 1.5)   # cm; below -> tight, above -> sloppy

VERDICTS = ["too tight", "snug", "regular", "relaxed", "oversized"]


@dataclass
class DimResult:
    dim: str
    garment_cm: float | None
    body_cm: float | None
    ease_cm: float | None
    verdict: str                 # one of VERDICTS, or "unknown"
    note: str
    ok: bool                     # wearable at this dimension

    def dict(self) -> dict:
        return asdict(self)


@dataclass
class FitReport:
    sku: str
    size: str
    family: str | None
    chart_source: str | None     # "supplier" | "reference" | None
    supplier: str | None
    overall: str
    fits: bool
    recommended_size: str | None
    size_change_reason: str | None
    dims: list[dict]
    missing_body: list[str]
    missing_garment: list[str]
    confidence: str              # high | medium | low | none
    confidence_reason: str
    fabric_stretch: float | None
    fabric_word: str | None
    length_note: str | None
    summary: str

    def dict(self) -> dict:
        return asdict(self)


def to_cm(value: float | None, unit: str) -> float | None:
    if value is None:
        return None
    return round(float(value) * (CM_PER_IN if unit == "in" else 1.0), 1)


def _band_verdict(family: str, dim: str, ease: float,
                  stretch_cm: float) -> tuple[str, bool]:
    bands = EASE_BANDS.get((family, dim))
    if not bands:
        return "unknown", True
    tight, snug, regular, relaxed = bands
    # Stretch fabric effectively lends the wearer extra centimetres.
    e = ease + stretch_cm
    if e < tight:
        return "too tight", False
    if e < snug:
        return "snug", True
    if e < regular:
        return "regular", True
    if e < relaxed:
        return "relaxed", True
    return "oversized", True


def _shoulder_verdict(ease: float) -> tuple[str, bool]:
    if ease < -SHOULDER_TOL:
        return "too tight", False
    if ease < -0.4:
        return "snug", True
    if ease <= SHOULDER_TOL:
        return "regular", True
    if ease <= SHOULDER_TOL * 2.5:
        return "relaxed", True
    return "oversized", True


def _shoe_verdict(allow: float) -> tuple[str, bool]:
    lo, hi = TOE_ALLOWANCE
    if allow < 0:
        return "too tight", False
    if allow < lo:
        return "snug", True
    if allow <= hi:
        return "regular", True
    if allow <= hi + 1.0:
        return "relaxed", True
    return "oversized", False


def evaluate(product: dict, size: str, body: dict,
             unit: str = "cm") -> FitReport:
    """Compare one product at one size against one body.

    `body` carries whatever the shopper has saved; missing keys stay missing.
    `unit` is the unit of the incoming body values ("cm" or "in").
    """
    sku = str(product.get("id") or product.get("sku") or "")
    chart = charts.chart_for(
        sku=str(product.get("sourceSku") or sku),
        brand=str(product.get("brand") or ""),
        article_type=str(product.get("articleType") or ""),
        name=str(product.get("name") or ""),
        category=str(product.get("masterCategory") or product.get("cat") or ""),
        gender=str(body.get("gender") or "male"),
    )
    family = chart.get("family")
    if not family:
        return FitReport(
            sku=sku, size=size, family=None, chart_source=None, supplier=None,
            overall="not sized", fits=True, recommended_size=None,
            size_change_reason=None, dims=[], missing_body=[],
            missing_garment=[], confidence="none",
            confidence_reason="This item is not sized by body measurement.",
            fabric_stretch=None, fabric_word=None, length_note=None,
            summary="This item is not sized against the body, so there is "
                    "nothing to compute.")

    sizes = chart.get("sizes") or {}
    if size not in sizes:
        size = _closest_size(sizes, size)
    garment = sizes.get(size, {})
    dims_wanted = chart.get("dims") or FAMILY_DIMS.get(family, [])

    stretch, fabric_word = stretch_for(
        f"{product.get('spec','')} {product.get('name','')} "
        f"{product.get('articleType','')}")

    body_cm = {k: to_cm(v, unit) for k, v in body.items()
               if isinstance(v, (int, float))}

    results: list[DimResult] = []
    missing_body: list[str] = []
    missing_garment: list[str] = []

    for dim in dims_wanted:
        g = garment.get(dim)
        b_key = DIM_TO_BODY.get(dim)
        b = body_cm.get(b_key) if b_key else None
        if g is None:
            missing_garment.append(dim)
        if b is None and dim not in INFO_ONLY and dim != "length":
            missing_body.append(b_key or dim)
        if dim in INFO_ONLY:
            results.append(DimResult(
                dim=dim, garment_cm=g, body_cm=None, ease_cm=None,
                verdict="unknown", ok=True,
                note=(f"Sleeve length {g} cm. Reported for reference; it is not "
                      "compared, because arm length is not part of the saved "
                      "profile." if g is not None else
                      "The chart does not list a sleeve length.")))
            continue
        if g is None or b is None:
            results.append(DimResult(
                dim=dim, garment_cm=g, body_cm=b, ease_cm=None,
                verdict="unknown", ok=True,
                note=_missing_note(dim, g, b, b_key, chart.get("source"))))
            continue

        if dim == "footLength":
            allow = round(g - b, 1)
            v, ok = _shoe_verdict(allow)
            note = (f"Internal length {g} cm against a {b} cm foot leaves "
                    f"{allow} cm of toe room.")
            results.append(DimResult(dim, g, b, allow, v, note, ok))
            continue

        ease = round(g - b, 1)
        stretch_cm = round((stretch or 0.0) * g, 1)
        if dim == "shoulder":
            v, ok = _shoulder_verdict(ease)
            note = (f"Shoulder seam at {g} cm against a {b} cm shoulder: "
                    f"{_signed(ease)} cm.")
        elif dim == "length":
            v, ok = "unknown", True
            note = (f"Garment length {g} cm."
                    + (f" Your front length is {b} cm, so it falls "
                       f"{_signed(round(g - b, 1))} cm past it."
                       if b else ""))
            if b:
                v, ok = _length_verdict(family, g - b)
        else:
            v, ok = _band_verdict(family, dim, ease, stretch_cm)
            note = (f"{dim.capitalize()} {g} cm against your {b} cm: "
                    f"{_signed(ease)} cm of ease"
                    + (f", plus about {stretch_cm} cm the {fabric_word} can "
                       f"give, so it behaves like "
                       f"{_signed(round(ease + stretch_cm, 1))} cm."
                       if stretch_cm else "."))
        results.append(DimResult(dim, g, b, ease, v, note, ok))

    overall, fits = _overall(results)
    rec, reason = _recommend(sizes, size, body_cm, family, stretch or 0.0,
                             dims_wanted, results)
    conf, conf_reason = _confidence(chart.get("source"), missing_body,
                                    missing_garment, stretch)
    length_note = next((r.note for r in results if r.dim == "length"), None)

    return FitReport(
        sku=sku, size=size, family=family,
        chart_source=chart.get("source"), supplier=chart.get("supplier"),
        overall=overall, fits=fits, recommended_size=rec,
        size_change_reason=reason,
        dims=[r.dict() for r in results],
        missing_body=sorted(set(missing_body)),
        missing_garment=sorted(set(missing_garment)),
        confidence=conf, confidence_reason=conf_reason,
        fabric_stretch=stretch, fabric_word=fabric_word,
        length_note=length_note,
        summary=_summary(size, overall, results, rec, reason, conf,
                         chart.get("source")),
    )


# -- helpers --------------------------------------------------------------
def _signed(v: float) -> str:
    return f"+{v}" if v >= 0 else str(v)


def _closest_size(sizes: dict, want: str) -> str:
    keys = list(sizes)
    if not keys:
        return want
    return "M" if "M" in keys else keys[len(keys) // 2]


def _missing_note(dim, g, b, b_key, source) -> str:
    if g is None and b is None:
        return (f"No {dim} measurement on either side, so this dimension "
                f"cannot be checked.")
    if g is None:
        return (f"The {source or 'available'} chart does not list a {dim} "
                f"measurement for this item, so it is not checked.")
    return (f"You have not saved a {b_key or dim} measurement, so the "
            f"{dim} of {g} cm cannot be compared.")


def _length_verdict(family: str, delta: float) -> tuple[str, bool]:
    """Garment length relative to the wearer's front length."""
    if family in ("top", "shirt", "outer"):
        if delta < -4:
            return "too tight", True      # reads as cropped, still wearable
        if delta > 16:
            return "oversized", True
        return "regular", True
    if family in ("kurta", "dress"):
        return "regular", True
    return "regular", True


def _overall(results: list[DimResult]) -> tuple[str, bool]:
    known = [r for r in results if r.verdict != "unknown"]
    if not known:
        return "cannot be judged", True
    if any(not r.ok for r in known):
        return "too tight", False
    rank = {v: i for i, v in enumerate(VERDICTS)}
    worst = max(known, key=lambda r: abs(rank[r.verdict] - 2))
    return worst.verdict, True


def _recommend(sizes: dict, current: str, body_cm: dict, family: str,
               stretch: float, dims: list[str],
               results: list[DimResult]) -> tuple[str | None, str | None]:
    """Score every size on the chart and name a better one if there is one."""
    if not sizes or family == "shoe":
        return _recommend_shoe(sizes, current, body_cm)
    scored: list[tuple[float, str]] = []
    for sz, g in sizes.items():
        pen = 0.0
        for dim in dims:
            if dim in ("length", "sleeve") or dim not in g:
                continue
            b_key = DIM_TO_BODY.get(dim)
            b = body_cm.get(b_key) if b_key else None
            if b is None:
                continue
            ease = g[dim] - b + stretch * g[dim]
            if dim == "shoulder":
                pen += abs(ease) * 2.5
                continue
            bands = EASE_BANDS.get((family, dim))
            if not bands:
                continue
            tight, snug, regular, relaxed = bands
            ideal = (snug + regular) / 2
            pen += abs(ease - ideal)
            if ease < tight:
                pen += 25                      # unwearable, heavily penalised
        scored.append((pen, sz))
    if not scored:
        return None, None
    scored.sort()
    best = scored[0][1]
    current_wearable = all(r.ok for r in results if r.verdict != "unknown")
    if best == current:
        return current, None
    cur_pen = next((p for p, s in scored if s == current), None)
    # Only call a small improvement "not worth it" when the current size is
    # actually wearable. If any dimension came out unwearable, we always move.
    if (current_wearable and cur_pen is not None
            and cur_pen - scored[0][0] < 2.0):
        return current, None
    # Name the dimension that actually drives the change: an unwearable one
    # first, otherwise the one furthest from a regular fit.
    judged = [r for r in results if r.verdict != "unknown"]
    failing = [r for r in judged if not r.ok]
    pool = failing or judged
    worst = min(pool, key=lambda r: r.ease_cm if r.ease_cm is not None else 0,
                default=None)
    why = (f"{current} comes out {worst.verdict} at the "
           f"{worst.dim}" if worst and worst.verdict != "regular"
           else f"{best} sits closer to the middle of the range")
    return best, f"{why}; {best} is the better match on this chart."


def _recommend_shoe(sizes: dict, current: str,
                    body_cm: dict) -> tuple[str | None, str | None]:
    foot = body_cm.get("foot")
    if not foot or not sizes:
        return None, None
    lo, hi = TOE_ALLOWANCE
    target = foot + (lo + hi) / 2
    best = min(sizes, key=lambda s: abs(sizes[s]["footLength"] - target))
    if best == current:
        return current, None
    return best, (f"{best} leaves about "
                  f"{round(sizes[best]['footLength'] - foot, 1)} cm of toe room "
                  f"for a {foot} cm foot.")


def _confidence(source, missing_body, missing_garment,
                stretch) -> tuple[str, str]:
    if source is None:
        return "none", "No chart applies to this item."
    bits = []
    score = 3 if source == "supplier" else 2
    if source != "supplier":
        bits.append("the chart is a generic reference, not this supplier's")
    if missing_body:
        score -= 1
        bits.append("you have not saved " + ", ".join(sorted(set(missing_body))))
    if missing_garment:
        score -= 1
        bits.append("the chart omits " + ", ".join(sorted(set(missing_garment))))
    if stretch is None:
        bits.append("the fabric is not stated, so stretch is not allowed for")
    level = {3: "high", 2: "medium", 1: "low"}.get(max(score, 1), "low")
    if not bits:
        return level, "Supplier measurements and a complete body profile."
    return level, "Reduced because " + "; ".join(bits) + "."


def _summary(size, overall, results, rec, reason, conf, source) -> str:
    head = f"Size {size} should come out {overall}."
    known = [r for r in results if r.verdict != "unknown"]
    if known:
        drivers = ", ".join(
            f"{r.dim} {r.verdict}" for r in known
            if r.verdict not in ("regular", "unknown"))
        if drivers:
            head += f" Driven by {drivers}."
    if rec and rec != size and reason:
        head += f" {reason}"
    if source == "reference":
        head += (" Based on a generic reference chart, not this item's own "
                 "supplier measurements.")
    return head


def best_size(product: dict, body: dict, unit: str = "cm") -> str | None:
    """Convenience: the size the engine would pick, with no size supplied."""
    chart = charts.chart_for(
        sku=str(product.get("sourceSku") or product.get("id") or ""),
        brand=str(product.get("brand") or ""),
        article_type=str(product.get("articleType") or ""),
        name=str(product.get("name") or ""),
        category=str(product.get("masterCategory") or product.get("cat") or ""),
        gender=str(body.get("gender") or "male"))
    sizes = chart.get("sizes") or {}
    if not sizes:
        return None
    start = "M" if "M" in sizes else list(sizes)[len(sizes) // 2]
    return evaluate(product, start, body, unit).recommended_size
