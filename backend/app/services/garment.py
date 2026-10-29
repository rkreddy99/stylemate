"""Garment geometry for the 3D viewer.

HONEST STATEMENT OF WHAT THIS PRODUCES
--------------------------------------
There are no 3D meshes for the catalogue products. The source dataset carries a
photograph and text attributes, nothing else. A flat photo cannot be turned into
an accurate garment pattern, so this module does NOT pretend to.

What it builds instead is a MEASUREMENT SHELL: a surface lofted at the garment's
own chart circumferences around the avatar, spanning the garment's chart length.
Every dimension of that shell traces back to a number in the size chart, so the
shape genuinely shows how much room the garment has at the chest, waist and hip,
and where it ends on the body. It is real geometry driven by real measurements.

It is NOT:
  * the product's actual cut, seams, darts or drape;
  * simulated cloth (no solver, no collision, no gravity);
  * a guarantee of how the item will look.

Every payload carries assetKind="measurement_shell" and the viewer labels it.

PLUGGING IN REAL ASSETS
-----------------------
`GarmentAssetResolver` looks for a real mesh first. Drop GLB/glTF files into
var/garment_assets/ named <sku>.glb (or map them in garment_assets.json) and
the API returns assetKind="supplier_mesh" with a URL instead of a shell, and
the viewer loads the real geometry. The same hook is where a cloth-simulation
service would attach.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..config import settings
from ..catalogue.compat import slot_of
from ..data.size_charts import charts
from .avatar import ASPECT, SQUARE, _limb, _section, leg_axis

# Vertical span of each family as a fraction of stature, measured from the
# shoulder (tops) or the waist/hip (bottoms). Length from the chart overrides
# the lower bound where available.
SPANS = {
    "top":    {"top": 0.818, "ref": "shoulder"},
    "shirt":  {"top": 0.818, "ref": "shoulder"},
    "kurta":  {"top": 0.818, "ref": "shoulder"},
    "outer":  {"top": 0.830, "ref": "shoulder"},
    "dress":  {"top": 0.810, "ref": "shoulder"},
    "bottom": {"top": 0.620, "ref": "waist"},
    "skirt":  {"top": 0.620, "ref": "waist"},
}

# Which chart dimension governs which body level inside the garment's span.
LEVEL_DIMS = {
    "shoulder": "chest", "chest": "chest", "waist": "waist", "hip": "hip",
}


class GarmentAssetResolver:
    """Finds a real 3D asset for a SKU, if one has been supplied."""

    def __init__(self):
        self.dir = settings.cache_dir.parent / "garment_assets"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.map_path = self.dir / "garment_assets.json"
        self.map: dict[str, str] = {}
        if self.map_path.exists():
            try:
                self.map = json.loads(self.map_path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                self.map = {}

    def find(self, sku: str) -> str | None:
        name = self.map.get(str(sku))
        if name and (self.dir / name).exists():
            return f"/assets/garments/{name}"
        for ext in (".glb", ".gltf"):
            if (self.dir / f"{sku}{ext}").exists():
                return f"/assets/garments/{sku}{ext}"
        return None


assets = GarmentAssetResolver()


def _interp_levels(family: str, garment: dict, body_m: dict,
                   height_cm: float) -> list[tuple[float, float]]:
    """Return [(y_cm, circumference_cm)] down the garment's span."""
    span = SPANS.get(family)
    if not span:
        return []
    top_y = height_cm * span["top"]

    length = garment.get("length")
    if length:
        bottom_y = max(0.04 * height_cm, top_y - float(length))
    else:
        bottom_y = height_cm * (0.47 if family in ("top", "shirt", "outer")
                                else 0.10)

    # Anchor circumferences at the levels the chart actually gives.
    anchors: list[tuple[float, float]] = []
    if family in ("top", "shirt", "kurta", "outer", "dress"):
        chest_c = garment.get("chest")
        if chest_c:
            anchors.append((height_cm * 0.720, float(chest_c)))
            shoulder_w = garment.get("shoulder")
            if shoulder_w:
                # Same width-to-girth conversion the body uses, so the garment
                # shoulder sits on the avatar's shoulder instead of overhanging.
                shoulder_c = float(shoulder_w) * 2.0 * (1 + 0.62) / 2.2
                anchors.append((top_y, shoulder_c))
                anchors.append((top_y - height_cm * 0.030,
                                (shoulder_c + float(chest_c)) / 2.0))
            else:
                anchors.append((top_y, float(chest_c) * 0.90))
        waist_c = garment.get("waist")
        if waist_c:
            anchors.append((height_cm * 0.620, float(waist_c)))
        hip_c = garment.get("hip")
        if hip_c:
            anchors.append((height_cm * 0.520, float(hip_c)))
        if not waist_c and chest_c:
            # A straight-cut top keeps roughly its chest girth down the body.
            anchors.append((height_cm * 0.560, float(chest_c) * 0.99))
    else:
        waist_c = garment.get("waist")
        hip_c = garment.get("hip")
        thigh_c = garment.get("thigh")
        if waist_c:
            anchors.append((height_cm * 0.620, float(waist_c)))
        if hip_c:
            anchors.append((height_cm * 0.520, float(hip_c)))
        if thigh_c:
            anchors.append((height_cm * 0.400, float(thigh_c)))
            anchors.append((bottom_y, float(thigh_c) * 0.62))
    if not anchors:
        return []

    anchors.sort(key=lambda t: t[0])
    ys = np.array([a[0] for a in anchors])
    cs = np.array([a[1] for a in anchors])
    out = []
    for i in range(22):
        y = bottom_y + (top_y - bottom_y) * i / 21.0
        out.append((float(y), float(np.interp(y, ys, cs))))
    return out


def _merge(parts):
    """Combine [(verts, faces)] into one indexed mesh."""
    V, F, off = [], [], 0
    for verts, faces in parts:
        if len(verts) == 0:
            continue
        V.extend(verts)
        F.extend([(a + off, b + off, c + off) for a, b, c in faces])
        off += len(verts)
    return V, F


def _tube(p0, p1, r0, r1, rings=7, segments=16):
    """A tapered tube between two points, as (verts, faces)."""
    v, f = _limb(np.array(p0, dtype=float), np.array(p1, dtype=float),
                 float(r0), float(r1), rings, segments)
    return (list(v), f)


def _loft(levels, segments, aspect_for, square_for):
    """Loft horizontal sections into a surface."""
    verts, faces, rings = [], [], []
    for y, circ, key in levels:
        rings.append(_section(circ, aspect_for(key), square_for(key), segments))
    for (y, _c, _k), sec in zip(levels, rings):
        for x, z in sec:
            verts.append([x, y, z])
    for i in range(len(rings) - 1):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            faces.append((a, b, b + segments))
            faces.append((a, b + segments, a + segments))
    return verts, faces


def _arm_points(height_cm, shoulder_cm, sign):
    """Shoulder, elbow and wrist for one arm, matching the avatar's limbs."""
    h, sw = height_cm, shoulder_cm / 2.0
    return (np.array([sign * sw * 0.66, h * 0.812, 0.0]),
            np.array([sign * sw * 0.97, h * 0.600, 0.0]),
            np.array([sign * sw * 1.02, h * 0.425, 0.0]))


def _sleeves(garment, body_m, height_cm, chest_c):
    """Sleeve tubes down each arm, as long as the chart says."""
    sleeve = garment.get("sleeve")
    if not sleeve:
        return []
    # Use the GARMENT's shoulder where the chart gives one: the sleeve head
    # belongs at the garment's shoulder seam, not the body's.
    shoulder_cm = float(garment.get("shoulder") or body_m["shoulder"]) * 1.02
    r_arm = body_m["chest"] * 0.105 / (2 * math.pi) * 2.0
    r0 = r_arm * 1.45          # armhole is roomier than the arm
    parts = []
    for sign in (-1, 1):
        sh, el, wr = _arm_points(height_cm, shoulder_cm, sign)
        # Pull the sleeve head inboard and up so it overlaps the body shell.
        # Sink the sleeve head into the body shell so the join reads as a
        # seam rather than a flap floating beside the shoulder.
        sh = sh + np.array([-sign * r0 * 1.15, height_cm * 0.012, 0.0])
        seg1 = float(np.linalg.norm(el - sh))
        seg2 = float(np.linalg.norm(wr - el))
        want = float(sleeve)
        if want <= seg1:
            end = sh + (el - sh) * (want / max(seg1, 1e-6))
            parts.append(_tube(sh, end, r0, r0 * 0.88, rings=5))
        else:
            parts.append(_tube(sh, el, r0, r0 * 0.86, rings=5))
            rem = min(want - seg1, seg2)
            end = el + (wr - el) * (rem / max(seg2, 1e-6))
            parts.append(_tube(el, end, r0 * 0.86, r0 * 0.70, rings=5))
    return parts


def _trouser_legs(garment, body_m, height_cm):
    """Two real legs rather than one pinched tube."""
    h = height_cm
    inseam = garment.get("inseam") or h * 0.44
    thigh_c = garment.get("thigh") or (body_m["thigh"] * 1.12)
    leg_x = leg_axis(body_m)
    crotch_y = h * 0.500
    end_y = max(h * 0.030, crotch_y - float(inseam))
    r_top = float(thigh_c) / (2 * math.pi)
    # Taper toward the ankle the way a trouser leg does, and never let the
    # hem close tighter than the calf it has to pass over.
    r_end = max(body_m["calf"] / (2 * math.pi) * 1.15, r_top * 0.52)
    parts = []
    for sign in (-1, 1):
        lx = sign * leg_x
        parts.append(_tube([lx, crotch_y, 0.0], [lx * 1.02, end_y, 0.0],
                           r_top, r_end, rings=9, segments=18))
    return parts


def _shoes(body_m, height_cm):
    """A shoe on each foot."""
    h = height_cm
    leg_x = leg_axis(body_m)
    r_ankle = body_m["ankle"] / (2 * math.pi) * 1.35
    foot = body_m["foot"]
    parts = []
    for sign in (-1, 1):
        lx = sign * leg_x * 1.02
        parts.append(_tube([lx, h * 0.075, -foot * 0.08],
                           [lx, h * 0.030, 0.0], r_ankle * 0.92, r_ankle, rings=4))
        parts.append(_tube([lx, h * 0.028, 0.0],
                           [lx, h * 0.020, foot * 0.62], r_ankle, r_ankle * 0.72,
                           rings=5))
    return parts


def _cap(body_m, height_cm):
    """A dome over the crown, for headwear."""
    h = height_cm
    head_r = h * 0.045 * 1.06
    base_y = h * 0.935
    verts, faces = [], []
    rings, segments = 6, 20
    for i in range(rings):
        t = i / (rings - 1)
        r = head_r * math.cos(t * math.pi / 2.4)
        y = base_y + h * 0.058 * t
        for j in range(segments):
            a = 2 * math.pi * j / segments
            verts.append([r * math.cos(a), y, r * 0.90 * math.sin(a)])
    for i in range(rings - 1):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            faces.append((a, b, b + segments))
            faces.append((a, b + segments, a + segments))
    top = len(verts)
    verts.append([0.0, base_y + h * 0.058, 0.0])
    last = (rings - 1) * segments
    for j in range(segments):
        faces.append((last + j, last + (j + 1) % segments, top))
    # peak
    peak = _tube([0.0, base_y + h * 0.004, head_r * 0.80],
                 [0.0, base_y - h * 0.004, head_r * 1.95],
                 head_r * 0.52, head_r * 0.42, rings=3, segments=12)
    return [(verts, faces), peak]


def build_shell(product: dict, size: str, body_m: dict,
                height_cm: float) -> dict:
    """Build the wearable geometry for one product at one size."""
    sku = str(product.get("id") or product.get("sourceSku") or "")
    source_sku = str(product.get("sourceSku") or sku)
    real = assets.find(source_sku) or assets.find(sku)
    chart = charts.chart_for(
        sku=source_sku, brand=str(product.get("brand") or ""),
        article_type=str(product.get("articleType") or ""),
        name=str(product.get("name") or ""),
        category=str(product.get("masterCategory") or product.get("cat") or ""),
        gender=str(body_m.get("gender") or "male"))
    family = chart.get("family")
    slot = slot_of(str(product.get("articleType") or ""),
                   str(product.get("name") or ""),
                   str(product.get("masterCategory") or ""))

    base = {
        "sku": sku, "size": size, "family": family, "slot": slot,
        "chartSource": chart.get("source"),
        "assetKind": None, "assetUrl": None,
        "positions": [], "indices": [], "note": "",
    }

    if real:
        base.update(assetKind="supplier_mesh", assetUrl=real,
                    note="Loaded from a supplied 3D asset for this SKU.")
        return base

    parts: list = []
    note = ""
    garment = (chart.get("sizes") or {}).get(size) or {}

    if family == "shoe":
        parts = _shoes(body_m, height_cm)
        note = ("Footwear shown at the size you selected. The shape is a "
                "stand-in, not this shoe's own last.")
    elif slot == "headwear":
        parts = _cap(body_m, height_cm)
        note = ("Headwear shown as a generic cap shape - the catalogue has no "
                "3D model of this item.")
    elif family in ("top", "shirt", "kurta", "outer", "dress", "bottom", "skirt"):
        if not garment:
            base.update(assetKind="none", note=f"No chart row for size {size}.")
            return base
        levels = _interp_levels(family, garment, body_m, height_cm)
        if not levels:
            base.update(assetKind="none",
                        note="The chart does not carry enough dimensions to "
                             "place this garment on the body.")
            return base
        segments = 28
        if family in ("bottom",):
            # Waistband plus two legs.
            waist_c = garment.get("waist") or body_m["waist"] * 1.04
            hip_c = garment.get("hip") or body_m["hips"] * 1.06
            band = [(height_cm * 0.620, float(waist_c), "waist"),
                    (height_cm * 0.560, float(hip_c) * 0.99, "hip"),
                    (height_cm * 0.520, float(hip_c), "hip"),
                    (height_cm * 0.500, float(hip_c) * 0.97, "hip")]
            parts.append(_loft(band, segments,
                               lambda k: ASPECT[k], lambda k: SQUARE.get(k, 2.2)))
            parts.extend(_trouser_legs(garment, body_m, height_cm))
            note = ("Waist, hip and inseam come from this item's "
                    + ("supplier" if chart.get("source") == "supplier" else "reference")
                    + " chart at size " + size + ". The legs are placed on the "
                    "avatar; the cut is not the product's own pattern.")
        else:
            lv = [(y, c, ("chest" if y > height_cm * 0.66
                          else "waist" if y > height_cm * 0.57 else "hip"))
                  for y, c in levels]
            parts.append(_loft(lv, segments,
                               lambda k: ASPECT[k], lambda k: SQUARE.get(k, 2.2)))
            if family != "skirt":
                parts.extend(_sleeves(garment, body_m, height_cm,
                                      garment.get("chest")))
            note = ("Chest, shoulder, length"
                    + (" and sleeve" if garment.get("sleeve") else "")
                    + " come from this item's "
                    + ("supplier" if chart.get("source") == "supplier" else "reference")
                    + " chart at size " + size + ". It is worn on the avatar at "
                    "those measurements; the cut, seams and drape are not the "
                    "product's own, and no cloth simulation has been run.")
    else:
        base.update(assetKind="none", note=(
            "Jewellery, bags and watches are not drawn on the avatar - there is "
            "no 3D model of them in the catalogue, and inventing one would "
            "misrepresent the product."))
        return base

    V, F = _merge(parts)
    if not V:
        base.update(assetKind="none", note="Nothing to draw for this item.")
        return base

    A = np.array(V, dtype=float)
    A[:, 0] -= float(A[:, 0].mean())
    A = A / 100.0
    base.update(
        assetKind="measurement_shell",
        positions=[round(float(x), 5) for x in A.reshape(-1)],
        indices=[int(i) for t in F for i in t],
        vertexCount=len(A), triangleCount=len(F),
        topY=round(float(A[:, 1].max()), 4),
        bottomY=round(float(A[:, 1].min()), 4),
        dims={k: v for k, v in garment.items()},
        note=note)
    return base


def _split_legs(verts, rings, segments, height_cm):
    """Pinch the lower rings toward two leg centres."""
    out = [list(v) for v in verts]
    crotch_y = height_cm * 0.46
    for i, (y, _sec) in enumerate(rings):
        if y >= crotch_y:
            continue
        t = min(1.0, (crotch_y - y) / max(1.0, crotch_y * 0.55))
        for j in range(segments):
            idx = i * segments + j
            x = out[idx][0]
            centre = math.copysign(abs(x) * 0.55 + 6.0, x if x else 1.0)
            out[idx][0] = x * (1 - 0.45 * t) + centre * (0.45 * t)
    return out
