"""Translate a product's own size labels to and from the fit engine's chart.

Catalogue v2 lists sizes the way Myntra did: S-XXL, waist inches (28-40),
collar or chest inches (36-48), UK/US/EURO shoe sizes, UK dress sizes,
Onesize. The reference chart only knows XS-XXL and "UK 5".."UK 12", and used
to swap anything else for "M" or a mid size without saying so. Every fit call
now goes product label -> chart key -> report, and the answer comes back in
the product's own labels.
"""
from __future__ import annotations

import re

ALPHA = ["XS", "S", "M", "L", "XL", "XXL"]
_ALPHA_ALIAS = {"XXS": "XS", "XXXL": "XXL", "3XL": "XXL", "4XL": "XXL", "2XL": "XXL"}
_WAIST = {24: "XS", 26: "XS", 28: "XS", 30: "S", 32: "M", 34: "L", 36: "XL", 38: "XXL", 40: "XXL", 42: "XXL"}
_CHEST = {34: "XS", 36: "XS", 38: "S", 39: "M", 40: "M", 42: "L", 44: "XL", 46: "XXL", 48: "XXL"}
_DRESS_UK = {4: "XS", 6: "XS", 8: "S", 10: "M", 12: "L", 14: "XL", 16: "XXL", 18: "XXL"}


def _norm(label: str) -> str:
    return re.sub(r"\s+", "", str(label or "")).upper()


def to_chart(label: str | None, product: dict) -> str | None:
    """The chart key for one of the product's size labels, or None when the
    label is not a body size (Onesize, 85CM belts, bag dimensions)."""
    n = _norm(label)
    if not n:
        return None
    if n in ALPHA:
        return n
    if n in _ALPHA_ALIAS:
        return _ALPHA_ALIAS[n]
    women = product.get("audience") == "women"
    slot = product.get("slot") or ""
    m = re.fullmatch(r"(UK|US|EURO?|EU)(\d+(?:\.5)?)", n)
    if m:
        sys_, v = m.group(1), float(m.group(2))
        if slot != "footwear" and sys_ == "UK" and women:          # UK dress sizes
            return _DRESS_UK.get(int(v))
        uk = v if sys_ == "UK" else v - (2 if women else 1) if sys_ == "US" else v - 34
        uk = int(round(min(12, max(5, uk))))
        return f"UK {uk}"
    if re.fullmatch(r"\d+", n):
        v = int(n)
        if slot == "bottom":
            return _WAIST.get(v)
        if slot in ("top", "outerwear", "dress"):
            return _CHEST.get(v) or (_DRESS_UK.get(v) if women else None)
    return None


def from_chart(chart: str | None, product: dict) -> str | None:
    """The product's own label closest to a chart key."""
    labels = product.get("sizes") or []
    if not chart or not labels:
        return chart
    mapped = [(lab, to_chart(lab, product)) for lab in labels]
    exact = [lab for lab, c in mapped if c == chart]
    if exact:
        return exact[len(exact) // 2]
    def rank(c):
        if c in ALPHA:
            return ALPHA.index(c)
        m = re.fullmatch(r"UK (\d+)", c or "")
        return int(m.group(1)) if m else None
    want = rank(chart)
    cands = [(abs(rank(c) - want), lab) for lab, c in mapped if c and rank(c) is not None and want is not None]
    return min(cands)[1] if cands else None


def report_in_labels(report: dict, label: str | None, product: dict) -> dict:
    """Restate a fit report in the product's own labels."""
    chart = report.get("size")
    report["chart_size"] = chart
    report["size"] = label or from_chart(chart, product) or chart
    rec = report.get("recommended_size")
    if rec:
        report["recommended_chart"] = rec
        report["recommended_size"] = from_chart(rec, product) or rec
    return report
