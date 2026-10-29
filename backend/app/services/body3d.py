"""Fitting-room body and garments on a realistic human mesh.

The bodies are Anny meshes (NAVER LABS, Apache 2.0, built on MakeHuman CC0
assets), baked once by tools/bake_bodies.py into data/bodies/anny_*.npz: a
small grid of male and female builds (weight x muscle), arms lowered. Nothing
here needs torch.

build_body(): blend the baked builds nearest to the shopper, scale to their
height, then widen or narrow the trunk and thighs so the chest, waist, hip and
thigh circumferences equal what they saved. Circumference is measured the way
a tape does it: the convex hull of the cross-section.

garment_shell(): cut the garment out of that body surface (torso and sleeves
for a top, waist to hem for trousers, feet for shoes) and push it out along
the surface normal by the room the size chart leaves at each height. A loose
size reads loose and a snug one hugs. Below the hips, dresses, kurtas and
skirts hang as a lofted skirt instead of wrapping each leg.

What it is not: cloth simulation, or the garment's real cut and print.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DIR = Path(__file__).resolve().parent.parent / "data" / "bodies"
TWO_PI = 2 * math.pi
_SETS: dict[str, dict] = {}


def available() -> bool:
    return all((DIR / f"anny_{g}.npz").exists() for g in ("male", "female"))


# --------------------------------------------------------------- geometry
def _edges(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def _section(V: np.ndarray, E: np.ndarray, y: float) -> np.ndarray:
    """XZ points where the plane at height y cuts the given edges."""
    a, b = V[E[:, 0]], V[E[:, 1]]
    cross = (a[:, 1] - y) * (b[:, 1] - y) < 0
    if not cross.any():
        return np.zeros((0, 2), np.float32)
    a, b = a[cross], b[cross]
    t = (y - a[:, 1]) / (b[:, 1] - a[:, 1])
    p = a + (b - a) * t[:, None]
    return p[:, [0, 2]].astype(np.float32)


def _hull(pts: np.ndarray):
    if len(pts) < 3:
        return None
    return cv2.convexHull(pts.reshape(-1, 1, 2)).reshape(-1, 2)


def _girth(pts: np.ndarray) -> float:
    h = _hull(pts)
    return 0.0 if h is None else float(cv2.arcLength(h.reshape(-1, 1, 2), True))


def _centre(pts: np.ndarray) -> np.ndarray:
    h = _hull(pts)
    return np.zeros(2) if h is None else h.mean(axis=0).astype(np.float64)


def _normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def _interp(y, knots):
    ks = sorted(knots)
    return np.interp(y, [k[0] for k in ks], [k[1] for k in ks])


# ------------------------------------------------------------- body sets
def _load(gender: str) -> dict:
    if gender in _SETS:
        return _SETS[gender]
    d = np.load(DIR / f"anny_{gender}.npz")
    names = [str(x) for x in d["region_names"]]
    s = {
        "F": d["faces"].astype(np.int64), "region": d["regions"],
        "arm_w": d["arm_w"].astype(np.float64), "leg_w": d["leg_w"].astype(np.float64),
        "builds": d["vertices"].astype(np.float64), "joints": d["joints"].astype(np.float64),
        "jn": [str(x) for x in d["joint_names"]], "R": {n: i for i, n in enumerate(names)},
        "source": str(d["source"]),
    }
    s["E"] = _edges(s["F"])
    arm = s["arm_w"] >= 0.5
    leg = s["leg_w"] >= 0.5
    s["E_trunk"] = s["E"][~arm[s["E"]].any(axis=1)]
    s["E_leg"] = s["E"][(leg[s["E"]].all(axis=1))]
    s["stats"] = []
    for V, J in zip(s["builds"], s["joints"]):
        lm = _landmarks(s, V, dict(zip(s["jn"], J)))
        H = V[:, 1].max()
        s["stats"].append(np.array([lm["C_chest"] / H, lm["C_waist"] / H, lm["C_hip"] / H]))
    _SETS[gender] = s
    return s


def _landmarks(s: dict, V: np.ndarray, J: dict) -> dict:
    """Heights and girths (metres) of chest, waist, hip, crotch and thigh."""
    H = float(V[:, 1].max())
    E = s["E_trunk"]
    hip_j = (J["hip_l"][1] + J["hip_r"][1]) / 2
    trunk_v = s["arm_w"] < 0.5
    mid = trunk_v & (np.abs(V[:, 0]) < 0.012) & (V[:, 1] > J["knee_l"][1]) & (V[:, 1] < hip_j)
    crotch = float(V[mid, 1].min()) if mid.any() else hip_j - 0.06 * H

    def scan(lo, hi, pick):
        ys = np.arange(lo, hi, 0.005)
        g = np.array([_girth(_section(V, E, y)) for y in ys])
        i = int(pick(g))
        return float(ys[i]), float(g[i])

    chest_y, C_chest = scan(0.68 * H, 0.77 * H, np.argmax)
    hip_y, C_hip = scan(crotch + 0.02, hip_j + 0.04 * H, np.argmax)
    waist_y, C_waist = scan(hip_y + 0.06 * H, chest_y - 0.04 * H, np.argmin)
    thigh_y = crotch - 0.03
    legs = [_section(V, s["E_leg"], thigh_y)]
    lp = legs[0]
    C_thigh = (_girth(lp[lp[:, 0] > 0]) + _girth(lp[lp[:, 0] < 0])) / 2
    return {"H": H, "crotch": crotch, "chest_y": chest_y, "waist_y": waist_y,
            "hip_y": hip_y, "thigh_y": thigh_y, "C_chest": C_chest,
            "C_waist": C_waist, "C_hip": C_hip, "C_thigh": C_thigh,
            "knee_y": float((J["knee_l"][1] + J["knee_r"][1]) / 2),
            "ankle_y": float((J["ankle_l"][1] + J["ankle_r"][1]) / 2),
            "neck_y": float(J["neck"][1]), "head_y": float(J["head"][1])}


@dataclass
class Body:
    gender: str
    V: np.ndarray
    N: np.ndarray
    J: dict
    lm: dict
    s: dict

    def girth_trunk(self, y: float) -> float:
        return _girth(_section(self.V, self.s["E_trunk"], y))

    def girth_leg(self, y: float) -> float:
        p = _section(self.V, self.s["E_leg"], y)
        return (_girth(p[p[:, 0] > 0]) + _girth(p[p[:, 0] < 0])) / 2


_BODIES: dict[tuple, "Body"] = {}


def build_body(m: dict) -> Body:
    """The baked body nearest to measurements `m` (cm), fitted to them. Cached."""
    key = tuple(sorted((k, v) for k, v in m.items() if isinstance(v, (int, float, str))))
    if key not in _BODIES:
        if len(_BODIES) > 64:
            _BODIES.clear()
        _BODIES[key] = _build_body(m)
    return _BODIES[key]


def _build_body(m: dict) -> Body:
    gender = "female" if m.get("gender") == "female" else "male"
    s = _load(gender)
    H_t = m["height"] / 100.0
    target = np.array([m["chest"], m["waist"], m["hips"]]) / m["height"]
    err = np.array([np.sum(((st - target) / target) ** 2) for st in s["stats"]])
    w = np.exp(-(err - err.min()) / 0.002)
    w /= w.sum()
    V = np.tensordot(w, s["builds"], axes=1)
    Jarr = np.tensordot(w, s["joints"], axes=1)
    k = H_t / V[:, 1].max()
    V, Jarr = V * k, Jarr * k
    J = dict(zip(s["jn"], Jarr))
    lm = _landmarks(s, V, J)

    clamp = lambda f: float(min(1.35, max(0.75, f)))
    # Each pass moves where the widest/narrowest section sits, so repeat until it settles.
    for _ in range(3):
        fc = clamp(m["chest"] / 100 / lm["C_chest"])
        fw = clamp(m["waist"] / 100 / lm["C_waist"])
        fh = clamp(m["hips"] / 100 / lm["C_hip"])
        ft = clamp(m["thigh"] / 100 / lm["C_thigh"]) if m.get("_thigh_supplied") else fh

        y = V[:, 1]
        # Trunk: scale each cross-section about its own centre.
        knots = [(lm["crotch"] - 0.05, fh), (lm["hip_y"], fh), (lm["waist_y"], fw),
                 (lm["chest_y"], fc), (lm["neck_y"] - 0.03, 1 + (fc - 1) * 0.5),
                 (lm["neck_y"], 1.0), (lm["H"] + 1, 1.0)]
        f_trunk = _interp(y, knots)
        ys = np.arange(lm["crotch"] - 0.05, lm["neck_y"] + 0.01, 0.01)
        cen = np.array([_centre(_section(V, s["E_trunk"], yy)) for yy in ys])
        cx, cz = np.interp(y, ys, cen[:, 0]), np.interp(y, ys, cen[:, 1])
        d_trunk = np.zeros_like(V)
        d_trunk[:, 0] = (f_trunk - 1) * (V[:, 0] - cx)
        d_trunk[:, 2] = (f_trunk - 1) * (V[:, 2] - cz)

        # Legs: thigh girth fades to nothing at the knee; legs move apart with the hips.
        g_leg = _interp(y, [(lm["knee_y"], 1.0), (lm["thigh_y"], ft), (lm["crotch"] + 0.02, ft),
                            (-1, 1.0), (lm["H"] + 1, ft)])
        d_leg = np.zeros_like(V)
        side = np.sign(V[:, 0])
        leg_cx = np.where(side > 0, J["hip_l"][0], J["hip_r"][0])
        leg_cz = (J["hip_l"][2] + J["hip_r"][2]) / 2
        d_leg[:, 0] = (g_leg - 1) * (V[:, 0] - leg_cx) + (fh - 1) * leg_cx
        d_leg[:, 2] = (g_leg - 1) * (V[:, 2] - leg_cz)

        # Arms ride outward with the chest (and the hips, so hands clear them).
        sx = abs(J["shoulder_l"][0])
        shift = max((fc - 1) * sx, (fh - 1) * sx * 0.8, -0.03)
        d_arm = np.zeros_like(V)
        d_arm[:, 0] = side * shift

        aw, lw = s["arm_w"][:, None], s["leg_w"][:, None]
        tw = np.clip(1 - aw - lw, 0, 1)
        V = V + tw * d_trunk + lw * d_leg + aw * d_arm
        for key in ("shoulder", "elbow", "wrist"):
            J[f"{key}_l"] = J[f"{key}_l"] + np.array([shift, 0, 0])
            J[f"{key}_r"] = J[f"{key}_r"] - np.array([shift, 0, 0])
        lm = _landmarks(s, V, J)
    return Body(gender, V, _normals(V, s["F"]), J, lm, s)


def mesh_payload(b: Body) -> dict:
    V, F = b.V, b.s["F"]
    return {
        "positions": np.round(V, 4).reshape(-1).tolist(),
        "indices": F.reshape(-1).tolist(),
        "vertexCount": int(len(V)), "triangleCount": int(len(F)),
        "heightM": round(float(V[:, 1].max()), 4), "gender": b.gender,
        "model": "anny", "hair": _hair(b), "base": _base_layer(b),
        "girthsCm": {k: round(v * 100, 1) for k, v in (
            ("chest", b.girth_trunk(b.lm["chest_y"])), ("waist", b.girth_trunk(b.lm["waist_y"])),
            ("hips", b.girth_trunk(b.lm["hip_y"])), ("thigh", b.girth_leg(b.lm["thigh_y"])))},
    }


# --------------------------------------------------------------- garments
def _subset(b: Body, keep: np.ndarray, offset: np.ndarray, clip=None,
            drop: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Faces touching `keep`, pushed out by `offset`. The ring of vertices just
    past the edge is then snapped onto the cut by `clip`, so hems, cuffs and
    necklines come out as clean lines rather than the triangles' sawtooth."""
    F = b.s["F"]
    faces = F[keep[F].any(axis=1)] if clip else F[keep[F].all(axis=1)]
    if drop is not None:
        faces = faces[~drop[faces].any(axis=1)]
    used = np.unique(faces)
    remap = -np.ones(len(b.V), np.int64)
    remap[used] = np.arange(len(used))
    P = b.V[used] + b.N[used] * offset[used, None]
    if clip:
        P = clip(P, used)
    return P, remap[faces]


def _clip_y(lo: float = -1.0, hi: float = 9.0):
    def f(P, used):
        P = P.copy()
        P[:, 1] = np.clip(P[:, 1], lo, hi)
        return P
    return f


def _hair(b: Body) -> dict:
    head = b.s["region"] == b.s["R"]["head"]
    y, z = b.V[:, 1], b.V[:, 2]
    hz = b.J["head"][2]
    top = b.lm["H"] - 0.085 if b.gender == "male" else b.lm["H"] - 0.11
    if b.gender == "female":
        back = (y > b.lm["neck_y"] + 0.01) & (z < hz - 0.005)
        keep = (head | (b.s["region"] == b.s["R"]["neck"])) & ((y > top) | back)
    else:
        back = (y > b.J["head"][1] + 0.01) & (z < hz - 0.01)
        keep = head & ((y > top) | back)
    off = np.full(len(b.V), 0.006 if b.gender == "male" else 0.012)
    P, I = _subset(b, keep, off)
    return {"positions": np.round(P, 4).reshape(-1).tolist(), "indices": I.reshape(-1).tolist()}


def _base_layer(b: Body) -> dict:
    """Plain underwear, so an empty fitting room never shows a bare body.
    It sits just off the skin, under every garment."""
    V, lm = b.V, b.lm
    y = V[:, 1]
    aw, lw = b.s["arm_w"], b.s["leg_w"]
    off = np.full(len(V), 0.0015)
    parts = [_subset(b, (aw < 0.5) & (y <= lm["hip_y"] + 0.05) & (y >= lm["crotch"] - 0.045), off,
                     _clip_y(lm["crotch"] - 0.045, lm["hip_y"] + 0.05))]
    if b.gender == "female":
        lo, hi = lm["chest_y"] - 0.06, lm["chest_y"] + 0.045
        parts.append(_subset(b, (aw < 0.5) & (y >= lo) & (y <= hi), off, _clip_y(lo, hi),
                             drop=aw >= 0.5))
    P, I = _join(parts)
    return {"positions": P, "indices": I}


def _sleeve_cm(product: dict, family: str, chart_sleeve) -> float:
    attr = str((product.get("attrs") or {}).get("Sleeve Length") or "")
    name = str(product.get("name") or "")
    if re.search(r"sleeveless|no sleeves|tank|vest", attr + " " + name, re.I):
        return 0.0
    if re.search(r"three-quarter|3/4", attr, re.I):
        return 42.0
    if re.search(r"long", attr, re.I):
        return float(chart_sleeve) if chart_sleeve and chart_sleeve > 40 else 60.0
    if re.search(r"short", attr, re.I):
        return float(chart_sleeve) if chart_sleeve and chart_sleeve < 35 else 22.0
    if chart_sleeve:
        return float(chart_sleeve)
    return {"shirt": 60.0, "outer": 62.0, "kurta": 55.0, "dress": 0.0}.get(family, 22.0)


def _arm_t(b: Body) -> np.ndarray:
    """Distance (m) of each arm vertex from the shoulder, along the arm."""
    V, J = b.V, b.J
    t = np.full(len(V), np.inf)
    R = b.s["R"]
    reg = b.s["region"]
    for sfx, sel in (("l", V[:, 0] > 0), ("r", V[:, 0] <= 0)):
        s, e, w = J[f"shoulder_{sfx}"], J[f"elbow_{sfx}"], J[f"wrist_{sfx}"]
        u1 = (e - s) / np.linalg.norm(e - s)
        u2 = (w - e) / np.linalg.norm(w - e)
        up = sel & (reg == R["upperarm"])
        lo = sel & (reg == R["forearm"])
        sh = sel & (b.s["arm_w"] >= 0.5) & ~up & ~lo & (reg != R["hand"])
        hd = sel & (reg == R["hand"])
        # Row-wise dot products (not `@`: Accelerate's matmul raises spurious FP warnings).
        t[up] = ((V[up] - s) * u1).sum(axis=1)
        t[lo | hd] = np.linalg.norm(e - s) + ((V[lo | hd] - e) * u2).sum(axis=1)
        t[sh] = 0.0
    return t


def _clip_sleeve(b: Body, sleeve: float, lo: float, hi: float):
    """Pull arm vertices past the cuff back along the arm; clip the body to [lo, hi]."""
    t_all = _arm_t(b)
    J = b.J
    def f(P, used):
        P = P.copy()
        body = b.s["arm_w"][used] < 0.5             # sleeves hang below the hem; only clip the body
        P[body, 1] = np.clip(P[body, 1], lo, hi)
        t = t_all[used]
        arm = np.isfinite(t) & (t > sleeve) & (b.s["arm_w"][used] >= 0.5)
        for sfx, side in (("l", P[:, 0] > 0), ("r", P[:, 0] <= 0)):
            s_, e_, w_ = J[f"shoulder_{sfx}"], J[f"elbow_{sfx}"], J[f"wrist_{sfx}"]
            L1 = np.linalg.norm(e_ - s_)
            u = (e_ - s_) / L1 if sleeve <= L1 else (w_ - e_) / np.linalg.norm(w_ - e_)
            k = arm & side
            P[k] -= u * (t[k] - sleeve)[:, None]
        return P
    return f


def _offset_from(G: np.ndarray, C: np.ndarray, floor: float) -> np.ndarray:
    return np.clip((G - C) / TWO_PI, floor, 0.09)


def _skirt(b: Body, y_top: float, y_hem: float, girth_top: float, flare: float,
           floor: float, start: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """A hanging tube from y_top to y_hem around everything but the arms."""
    E = b.s["E_trunk"]
    seg = 64
    ang = np.linspace(-math.pi, math.pi, seg, endpoint=False)
    levels = np.arange(y_top, y_hem - 1e-6, -0.02).tolist() + [y_hem]
    rings, prev, prev_h = [], None, None
    # One axis for the whole skirt (the hip section's centre). Per-level centres
    # wander front to back by several cm and would make the hem ripple.
    c = _centre(_section(b.V, E, b.lm["hip_y"]))
    for y in levels:
        h = _hull(_section(b.V, E, y))
        if h is None:
            h = prev_h
        if h is None:
            continue
        prev_h = h
        # Densify the hull before going polar: interpolating radius by angle across a
        # long straight edge (the front of two thighs) bulges outward by several cm.
        nxt = np.roll(h, -1, axis=0)
        steps = np.maximum(1, (np.linalg.norm(nxt - h, axis=1) / 0.005).astype(int))
        h = np.concatenate([h[i] + (nxt[i] - h[i]) * np.linspace(0, 1, n, endpoint=False)[:, None]
                            for i, n in enumerate(steps)])
        a = np.arctan2(h[:, 1] - c[1], h[:, 0] - c[0])
        r = np.hypot(h[:, 0] - c[0], h[:, 1] - c[1])
        o = np.argsort(a)
        a, r = a[o], r[o]
        rr = np.interp(ang, np.concatenate([a - TWO_PI, a, a + TWO_PI]), np.tile(r, 3))
        frac = (y_top - y) / max(1e-6, y_top - y_hem)
        G = girth_top * (1 + flare * frac)
        if prev is None:
            # Top ring: the body plus the room the chart leaves (tape girth, as elsewhere).
            rr = rr + max(floor, (G - _girth(prev_h)) / TWO_PI)
            if start is not None and len(start) >= 3:
                # Begin from the bodice's own bottom edge so the two parts meet without a step.
                sa = np.arctan2(start[:, 1] - c[1], start[:, 0] - c[0])
                sr = np.hypot(start[:, 0] - c[0], start[:, 1] - c[1])
                o = np.argsort(sa)
                rr = np.maximum(rr, np.interp(ang, np.concatenate([sa[o] - TWO_PI, sa[o], sa[o] + TWO_PI]),
                                              np.tile(sr[o], 3)))
        else:
            # Below it the cloth hangs from the ring above (scaled for any flare) and
            # only moves out where the body itself would poke through.
            drape = 0.985 if flare <= 0 else 1.0       # unflared tails fall in a little
            rr = np.maximum(prev[1] * (G / prev[2]) * drape, rr + floor)
        prev = (c, rr, G)
        rings.append((c, y, rr))
    # Hull samples are coarse; smooth the radius around each ring and down the skirt.
    k = np.array([1, 2, 3, 2, 1], float) / 9
    R = np.array([r for _, _, r in rings])
    for _ in range(2):
        R = sum(w * np.roll(R, sh, axis=1) for w, sh in zip(k, range(-2, 3)))
    if len(R) > 2:
        R[1:-1] = (R[:-2] + 2 * R[1:-1] + R[2:]) / 4
    rings = [np.stack([c[0] + r * np.cos(ang), np.full(seg, y), c[1] + r * np.sin(ang)], axis=1)
             for (c, y, _), r in zip(rings, R)]
    P = np.concatenate(rings)
    I = []
    for i in range(len(rings) - 1):
        for j in range(seg):
            a0, a1 = i * seg + j, i * seg + (j + 1) % seg
            b0, b1 = a0 + seg, a1 + seg
            I += [[a0, b0, a1], [a1, b0, b1]]
    return P, np.array(I, np.int64)


def _join(parts):
    P, I, n = [], [], 0
    for p, i in parts:
        if len(p) and len(i):
            P.append(p)
            I.append(i + n)
            n += len(p)
    if not P:
        return [], []
    return np.round(np.concatenate(P), 4).reshape(-1).tolist(), np.concatenate(I).reshape(-1).tolist()


def garment_shell(b: Body, product: dict, family: str | None, slot: str,
                  chart_row: dict) -> dict:
    """Positions/indices for one garment worn on body `b`."""
    V, lm = b.V, b.lm
    y = V[:, 1]
    reg, R = b.s["region"], b.s["R"]
    aw, lwt = b.s["arm_w"], b.s["leg_w"]
    cm = lambda k: (chart_row.get(k) or 0) / 100.0
    parts = []

    if family == "shoe":
        keep = (reg == R["foot"]) | ((lwt >= 0.5) & (y < lm["ankle_y"] + 0.035))
        off = np.where(y < 0.012, 0.004, 0.009)         # a thicker upper over a thin sole
        parts.append(_subset(b, keep, off, _clip_y(hi=lm["ankle_y"] + 0.035)))
    elif slot == "headwear":
        keep = (reg == R["head"]) & (y > lm["H"] - 0.075)
        parts.append(_subset(b, keep, np.full(len(V), 0.008), _clip_y(lo=lm["H"] - 0.075)))
    elif family in ("top", "shirt", "kurta", "outer", "dress"):
        # Layers stack outward: bottoms, then tops and dresses, then outerwear.
        layer = 0.010 if family == "outer" else 0.004
        floor = 0.008 if family == "outer" else 0.003
        G_chest = cm("chest") or lm["C_chest"] * 1.08
        hem = lm["neck_y"] + 0.025 - (cm("length") or 0.7)     # chart length runs from the shoulder top
        hem = max(hem, lm["ankle_y"] + 0.15)
        # Above the hips the garment follows the body; below them it hangs straight
        # (shirt tails) or flares (kurtas, dresses) instead of wrapping each leg.
        hang_from = lm["waist_y"] if family in ("dress", "kurta") else lm["hip_y"]
        hangs = hem < hang_from - 0.01
        torso_bottom = max(hem, hang_from)
        neck_cut = lm["neck_y"] + 0.012
        trunk = (aw < 0.5) & (reg != R["head"]) & (y >= torso_bottom) & (y <= neck_cut)
        reach = min(np.linalg.norm(b.J["elbow_l"] - b.J["shoulder_l"])
                    + np.linalg.norm(b.J["wrist_l"] - b.J["elbow_l"]) - 0.01, 0.75)
        sleeve = min(_sleeve_cm(product, family, chart_row.get("sleeve")) / 100.0, reach)
        arm = (aw >= 0.5) & (_arm_t(b) <= sleeve) if sleeve > 0 else np.zeros(len(V), bool)
        ys = np.arange(torso_bottom - 0.02, lm["neck_y"] + 0.02, 0.01)
        C = np.array([b.girth_trunk(yy) for yy in ys])
        if family == "dress":
            G = _interp(ys, [(lm["chest_y"], G_chest),
                             (lm["waist_y"], cm("waist") or G_chest),
                             (lm["hip_y"], cm("hip") or G_chest), (-1, cm("hip") or G_chest)])
        else:
            G = np.full_like(ys, G_chest)          # straight cut from the chest down
        off_y = _offset_from(G, C, floor)
        i_ch = int(np.argmin(np.abs(ys - lm["chest_y"])))
        off_y[i_ch:] = np.linspace(off_y[i_ch], floor + 0.002, len(ys) - i_ch)
        off = np.interp(y, ys, off_y)
        sleeve_off = floor + (0.008 if family != "dress" else 0.004)
        off = off * (1 - aw) + sleeve_off * aw
        clip = _clip_sleeve(b, sleeve, torso_bottom, neck_cut)
        if sleeve <= 0:
            # Sleeveless: a clean armhole, cut on a vertical plane inside the shoulder joint.
            ax = abs(b.J["shoulder_l"][0]) - 0.012
            trunk = trunk & (np.abs(V[:, 0]) <= ax)
            def clip(P, used, _c=clip, _ax=ax):
                P = _c(P, used)
                P[:, 0] = np.clip(P[:, 0], -_ax, _ax)
                return P
        parts.append(_subset(b, trunk | arm, off + layer, clip,
                             drop=(aw >= 0.5) if sleeve <= 0 else None))
        if hangs:
            flare = {"dress": 0.22, "kurta": 0.10}.get(family, 0.0)
            g_top = max(cm("hip") or 0, G_chest if family != "dress" else 0,
                        b.girth_trunk(lm["hip_y"]) * 1.02)
            if family in ("dress", "kurta"):
                g_top = max(cm("waist") or 0, b.girth_trunk(lm["waist_y"]) * 1.02)
            bod = parts[-1][0]
            edge = bod[(bod[:, 1] < torso_bottom + 0.012) & (np.abs(bod[:, 0]) < abs(b.J["shoulder_l"][0]))]
            parts.append(_skirt(b, hang_from + 0.005, hem, g_top, flare, floor + 0.004 + layer,
                                start=edge[:, [0, 2]] if len(edge) else None))
    elif family == "skirt":
        top = lm["waist_y"] - 0.01
        keep = (aw < 0.5) & (y <= top) & (y >= lm["hip_y"])
        G = cm("waist") or lm["C_waist"] * 1.03
        parts.append(_subset(b, keep, np.full(len(V), max(0.003, (G - lm["C_waist"]) / TWO_PI)),
                             _clip_y(lm["hip_y"], top)))
        hem = max(top - (cm("length") or 0.6), lm["ankle_y"] + 0.05)
        g_top = max(cm("hip") or 0, b.girth_trunk(lm["hip_y"]) * 1.04)
        parts.append(_skirt(b, lm["hip_y"] + 0.01, hem, g_top, 0.18, 0.003))
    elif family == "bottom":
        top = lm["waist_y"] - 0.025
        inseam = cm("inseam") or (lm["crotch"] - lm["ankle_y"])
        hem = max(lm["crotch"] - inseam, lm["ankle_y"] + 0.01)
        keep = (aw < 0.5) & (y <= top) & (y >= hem) & (reg != R["foot"])
        Gw = cm("waist") or lm["C_waist"] * 1.03
        Gh = cm("hip") or b.girth_trunk(lm["hip_y"]) * 1.06
        ys = np.arange(lm["crotch"] - 0.01, top + 0.02, 0.01)
        C = np.array([b.girth_trunk(yy) for yy in ys])
        G = _interp(ys, [(top, Gw), (lm["hip_y"], Gh), (-1, Gh)])
        off_t = np.interp(y, ys, _offset_from(G, C, 0.003))
        Gt = cm("thigh") or b.girth_leg(lm["thigh_y"]) * 1.12
        knee_g = max(b.girth_leg(lm["knee_y"]) + 0.07, Gt * 0.72)
        ly = np.arange(hem - 0.01, lm["crotch"] + 0.01, 0.01)
        Cl = np.array([b.girth_leg(yy) for yy in ly])
        Gl = _interp(ly, [(lm["thigh_y"], Gt), (lm["knee_y"], knee_g), (-1, knee_g * 0.97)])
        off_l = np.interp(y, ly, _offset_from(Gl, Cl, 0.003))
        off = off_t * (1 - lwt) + off_l * lwt
        parts.append(_subset(b, keep, off, _clip_y(hem, top)))
    else:
        return {"positions": [], "indices": []}

    P, I = _join(parts)
    return {"positions": P, "indices": I}


def shell_for(b: Body, product: dict, size: str | None) -> dict:
    """Chart lookup as garment.build_shell does it, then garment_shell()."""
    from ..catalogue.compat import slot_of
    from ..data.size_charts import charts
    chart = charts.chart_for(
        sku=str(product.get("sourceSku") or product.get("id") or ""),
        brand=str(product.get("brand") or ""),
        article_type=str(product.get("articleType") or ""),
        name=str(product.get("name") or ""),
        category=str(product.get("masterCategory") or product.get("cat") or ""),
        gender=b.gender)
    slot = slot_of(str(product.get("articleType") or ""), str(product.get("name") or ""),
                   str(product.get("masterCategory") or ""))
    row = (chart.get("sizes") or {}).get(size) or {}
    family = chart.get("family")
    if not family and slot == "dress":
        # Saris and the like have no size chart: draw a short-sleeved blouse
        # over a floor-length drape so the colour still reads on the body.
        family = "dress"
        row = {"length": (b.lm["neck_y"] + 0.025 - b.lm["ankle_y"] - 0.02) * 100, "sleeve": 15}
    out = garment_shell(b, product, family, slot, row)
    out.update(family=family, slot=slot, chartSource=chart.get("source"),
               assetKind="body_surface")
    return out
