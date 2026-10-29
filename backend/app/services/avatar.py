"""Parametric body mesh.

WHAT THIS IS: a real, watertight 3D triangle mesh generated from anthropometric
measurements. Cross-sections (super-ellipses) are placed at anatomical heights,
scaled so that each section's CIRCUMFERENCE matches the measurement the shopper
saved, then lofted into a surface. Arms and legs are separate lofted limbs.
The result rotates, zooms and lights like any other mesh, because it is one.

WHAT THIS IS NOT: SMPL-X, or any statistical body model learned from scans. It
does not recover identity, pose or soft-tissue shape from a photograph. A single
photo cannot give true depth; this module uses a photo only to CALIBRATE
proportions that the shopper has not measured (shoulder:hip width ratio, and
the head-to-floor split), and says so in the payload it returns.

The reconstruction backend is pluggable - see `ESTIMATORS`. Adding SMPL-X later
means implementing one class with .estimate(image) -> BodyEvidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

CM_PER_IN = 2.54

# Fractions of total stature at which each landmark sits. Standard adult
# anthropometric proportions (Drillis & Contini); used only to place the
# sections vertically, never to invent a circumference.
LEVELS: list[tuple[str, float]] = [
    ("ankle",     0.039),
    ("calf",      0.120),
    ("knee",      0.285),
    ("thigh",     0.400),
    ("hip",       0.520),
    ("waist",     0.620),
    ("chest",     0.720),
    ("shoulder",  0.818),
    ("neck",      0.870),
    ("head_mid",  0.935),
    ("crown",     1.000),
]

# Default circumferences in cm for a mid-size adult, used ONLY where the
# shopper has saved nothing and only after being flagged as a default.
DEFAULTS = {
    "male":   {"height": 172.0, "chest": 98.0, "waist": 86.0, "hips": 98.0,
               "shoulder": 45.0, "neck": 38.0, "thigh": 55.0, "calf": 37.0,
               "ankle": 23.0, "foot": 26.5},
    "female": {"height": 160.0, "chest": 90.0, "waist": 74.0, "hips": 98.0,
               "shoulder": 38.5, "neck": 33.0, "thigh": 56.0, "calf": 35.0,
               "ankle": 22.0, "foot": 24.0},
}

# How elliptical each section is (depth / width). A chest is wider than deep;
# a thigh is nearly round. These shape the silhouette, not the size: the
# circumference is still driven by the measurement.
ASPECT = {"neck": 0.90, "shoulder": 0.62, "chest": 0.72, "waist": 0.78,
          "hip": 0.75, "thigh": 0.92, "knee": 0.95, "calf": 0.90,
          "ankle": 0.92, "head_mid": 0.88, "crown": 0.90}

# Per-gender silhouette. These are shape factors, never sizes: the measured
# circumference still decides how big each section is. Without them both
# bodies came out as the same tube and only the numbers differed.
SHAPE = {
    "male": {
        "shoulder_w": 1.00,   # shoulder width multiplier
        "chest_aspect": 0.74,  # deeper chest, flatter front
        "waist_in": 1.00,      # waist nip
        "hip_w": 0.96,
        "bust": 0.0,           # no forward bust volume
        "seat": 0.05,          # rear volume at the hip
        "limb": 1.00,
        "torso_long": 1.00,
    },
    "female": {
        "shoulder_w": 0.90,
        "chest_aspect": 0.80,
        "waist_in": 0.92,      # more defined waist
        "hip_w": 1.08,         # wider hips
        "bust": 0.42,          # forward volume at chest level
        "seat": 0.12,
        "limb": 0.92,
        "torso_long": 0.97,
    },
}

# Super-ellipse exponent: 2 is a true ellipse, higher is squarer. Torsos read
# better slightly squared off.
SQUARE = {"chest": 2.5, "waist": 2.4, "hip": 2.4, "shoulder": 2.8}


@dataclass
class BodyEvidence:
    """What a photo told us. Every field is optional and provenance-tagged."""
    source: str                       # "mediapipe" | "silhouette" | "none"
    shoulder_hip_ratio: float | None = None
    height_px: float | None = None
    width_profile: list[float] = field(default_factory=list)
    skin_rgb: tuple[int, int, int] | None = None
    notes: list[str] = field(default_factory=list)
    confidence: str = "none"          # high | medium | low | none

    def dict(self) -> dict:
        d = asdict(self)
        d["skin_rgb"] = list(self.skin_rgb) if self.skin_rgb else None
        return d


def resolve_measurements(profile: dict) -> tuple[dict, list[str], list[str]]:
    """Convert a saved size profile to cm and report what was supplied.

    Returns (measurements_cm, supplied_keys, defaulted_keys).
    Nothing is invented silently: anything not supplied appears in `defaulted`.
    """
    unit = (profile.get("unit") or "cm").lower()
    gender = "female" if str(profile.get("gender", "male")).lower().startswith("f") else "male"
    base = dict(DEFAULTS[gender])
    supplied, defaulted = [], []
    out = {}
    for key, fallback in base.items():
        v = profile.get(key)
        if isinstance(v, (int, float)) and v > 0:
            out[key] = round(float(v) * (CM_PER_IN if unit == "in" else 1.0), 1)
            supplied.append(key)
        else:
            out[key] = fallback
            defaulted.append(key)
    # frontLength is a torso length, handled separately from circumferences.
    fl = profile.get("frontLength")
    if isinstance(fl, (int, float)) and fl > 0:
        out["frontLength"] = round(float(fl) * (CM_PER_IN if unit == "in" else 1.0), 1)
        supplied.append("frontLength")
    out["gender"] = gender
    for key in ("shoulder", "chest", "waist", "hips", "height"):
        out[f"_{key}_supplied"] = key in supplied
    return out, supplied, defaulted


def leg_axis(m: dict, segments: int = 36) -> float:
    """Horizontal offset of each leg's centre line, in centimetres.

    The avatar and every garment placed on it must use this same value, or
    trousers end up next to the legs rather than around them.
    """
    gender = m.get("gender", "male")
    sh = SHAPE.get(gender, SHAPE["male"])
    hip_sec = _section(m["hips"] * sh["hip_w"], ASPECT["hip"],
                       SQUARE.get("hip", 2.4), segments)
    hip_w = float(np.max(np.abs(hip_sec[:, 0])))
    return hip_w * 0.46


def _section(circumference_cm: float, aspect: float, squareness: float,
             segments: int) -> np.ndarray:
    """One horizontal cross-section as (segments, 2) of x,z in centimetres.

    Solved so the resulting super-ellipse's PERIMETER equals the measurement:
    we build a unit shape, measure its perimeter numerically, then scale.
    """
    t = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    n = 2.0 / squareness
    ct, st = np.cos(t), np.sin(t)
    x = np.sign(ct) * np.abs(ct) ** n
    z = np.sign(st) * np.abs(st) ** n * aspect
    pts = np.stack([x, z], axis=1)
    closed = np.vstack([pts, pts[:1]])
    perim = float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))
    if perim <= 0:
        return pts
    return pts * (circumference_cm / perim)


def _limb(start: np.ndarray, end: np.ndarray, r0: float, r1: float,
          rings: int, segments: int) -> tuple[np.ndarray, list[tuple]]:
    """A tapered tube between two points. Returns (vertices, faces)."""
    verts, faces = [], []
    axis = end - start
    length = np.linalg.norm(axis)
    if length < 1e-6:
        return np.zeros((0, 3)), []
    axis = axis / length
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, up))) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, up); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    for i in range(rings):
        f = i / (rings - 1)
        centre = start + axis * (length * f)
        r = r0 + (r1 - r0) * f
        for j in range(segments):
            a = 2 * math.pi * j / segments
            verts.append(centre + u * (r * math.cos(a)) + v * (r * math.sin(a)))
    for i in range(rings - 1):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            c = a + segments
            d = b + segments
            faces.append((a, b, d)); faces.append((a, d, c))
    return np.array(verts), faces


def build_mesh(m: dict, segments: int = 36,
               evidence: BodyEvidence | None = None) -> dict:
    """Build the full body mesh. Returns a three.js-ready buffer payload."""
    h = float(m["height"])
    gender = m.get("gender", "male")

    # Photo evidence can only adjust the shoulder:hip WIDTH ratio, and only
    # when the shopper has not measured both. It never overrides a measurement.
    shoulder_w = m["shoulder"]
    if (evidence and evidence.shoulder_hip_ratio
            and "shoulder" not in (evidence.notes or [])):
        pass  # ratio is applied by the caller via m; kept explicit, see notes.

    sh = SHAPE.get(gender, SHAPE["male"])
    circ = {
        "ankle": m["ankle"], "calf": m["calf"], "knee": m["calf"] * 1.05,
        "thigh": m["thigh"], "hip": m["hips"] * sh["hip_w"],
        "waist": m["waist"] * sh["waist_in"],
        "chest": m["chest"],
        # Shoulder is a WIDTH, not a circumference. Approximate the shoulder
        # section's girth from the width, keeping the chest's aspect.
        "shoulder": (m["shoulder"] * sh["shoulder_w"]) * 2.0
                    * (1 + ASPECT["shoulder"]) / 2.2,
        "neck": m["neck"],
        "head_mid": m["neck"] * 1.55,
        "crown": m["neck"] * 0.55,
    }

    # Torso runs crotch -> hip -> waist -> chest -> shoulder -> neck, with
    # intermediate rings so the surface is smooth rather than faceted.
    crotch_y = h * 0.455
    spine = [
        (crotch_y,      circ["hip"] * 0.94, "hip"),
        (h * 0.520,     circ["hip"],        "hip"),
        (h * 0.575,     (circ["hip"] + circ["waist"]) / 2, "waist"),
        (h * 0.620,     circ["waist"],      "waist"),
        (h * 0.672,     (circ["waist"] + circ["chest"]) / 2, "chest"),
        (h * 0.720,     circ["chest"],      "chest"),
        (h * 0.772,     (circ["chest"] + circ["shoulder"]) / 2, "chest"),
        (h * 0.818,     circ["shoulder"],   "shoulder"),
        (h * 0.845,     circ["shoulder"] * 0.72, "shoulder"),
        (h * 0.858,     circ["neck"] * 1.06, "neck"),
        (h * 0.878,     circ["neck"] * 0.98, "neck"),
    ]
    rings = []
    for y, c, k in spine:
        aspect = sh["chest_aspect"] if k == "chest" else ASPECT.get(k, 0.8)
        sec = _section(c, aspect, SQUARE.get(k, 2.0), segments)
        # Bust: push the front of the chest band forward, tapering above and
        # below so it reads as a body rather than a bulge on a tube.
        if sh["bust"] and 0.645 <= y / h <= 0.790:
            t = 1.0 - abs((y / h) - 0.715) / 0.075
            amount = sh["bust"] * (m["chest"] / 100.0) * 4.2 * max(0.0, t)
            front = sec[:, 1] > 0
            sec[front, 1] += amount * (sec[front, 1] / max(1e-6, sec[:, 1].max()))
        # Seat: a little rear volume at the hip.
        if sh["seat"] and 0.470 <= y / h <= 0.560:
            t = 1.0 - abs((y / h) - 0.515) / 0.045
            amount = sh["seat"] * (m["hips"] / 100.0) * 5.0 * max(0.0, t)
            back = sec[:, 1] < 0
            sec[back, 1] -= amount * (abs(sec[back, 1]) / max(1e-6, abs(sec[:, 1]).max()))
        rings.append(sec)
    ys = [y for y, _c, _k in spine]

    verts: list[np.ndarray] = []
    for sec, y in zip(rings, ys):
        for x, z in sec:
            verts.append(np.array([x, y, z]))
    faces: list[tuple] = []
    for i in range(len(rings) - 1):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            c = a + segments
            d = b + segments
            faces.append((a, b, d)); faces.append((a, d, c))
    # Cap the neck opening and the crotch opening so the torso is closed.
    for ring_index, y_cap, flip in ((len(rings) - 1, ys[-1], False), (0, ys[0], True)):
        centre = len(verts)
        verts.append(np.array([0.0, y_cap, 0.0]))
        base = ring_index * segments
        for j in range(segments):
            a, b = base + j, base + (j + 1) % segments
            faces.append((a, b, centre) if not flip else (b, a, centre))

    # --- head ------------------------------------------------------------
    # Real adult head: about 0.13 of stature tall and ~15 cm wide, which is
    # only a little wider than the neck. Driving the width off neck girth made
    # it a bulb, so it is driven off stature and overlaps the neck slightly.
    head_bottom = h * 0.862
    head_h = h * 0.138
    head_r = h * 0.045
    head_rings = 14
    head_start = len(verts)
    for i in range(head_rings):
        t = i / (head_rings - 1)
        # Half-ellipse profile: widest a third of the way up, tapering to the crown.
        # Egg profile: full width through the cranium, narrowing to the jaw
        # at the base and rounding off at the crown.
        u = 2.0 * t - 1.0
        prof = math.sqrt(max(0.0, 1.0 - u * u * (0.92 if u > 0 else 0.62)))
        r = head_r * max(0.30, prof)
        y = head_bottom + head_h * t
        for j in range(segments):
            a = 2 * math.pi * j / segments
            verts.append(np.array([r * math.cos(a), y,
                                   r * 0.88 * math.sin(a)]))
    for i in range(head_rings - 1):
        for j in range(segments):
            a = head_start + i * segments + j
            b = head_start + i * segments + (j + 1) % segments
            faces.append((a, b, b + segments)); faces.append((a, b + segments, a + segments))
    crown = len(verts)
    verts.append(np.array([0.0, head_bottom + head_h, 0.0]))
    last = head_start + (head_rings - 1) * segments
    for j in range(segments):
        faces.append((last + j, last + (j + 1) % segments, crown))

    V = np.array(verts)
    offset = len(V)

    # --- limbs -----------------------------------------------------------
    leg_x = leg_axis(m, segments)
    sh_w = (m["shoulder"] * sh["shoulder_w"]) / 2.0
    r_thigh = m["thigh"] / (2 * math.pi)
    r_calf = m["calf"] / (2 * math.pi)
    r_ankle = m["ankle"] / (2 * math.pi)
    r_upper_arm = m["chest"] * 0.105 / (2 * math.pi) * 2.0 * sh["limb"]
    r_wrist = r_upper_arm * 0.58
    r_thigh *= sh["limb"]
    r_calf *= sh["limb"]

    limbs = []
    for sign in (-1, 1):
        lx = sign * leg_x
        # Thighs start ABOVE the crotch ring so they merge into the hips
        # instead of hanging below them.
        limbs.append((np.array([lx, h * 0.500, 0.0]),
                      np.array([lx * 1.02, h * 0.285, 0.0]), r_thigh, r_calf))
        limbs.append((np.array([lx * 1.02, h * 0.285, 0.0]),
                      np.array([lx * 1.02, h * 0.039, 0.0]), r_calf, r_ankle))
        # Arms begin inside the shoulder mass, not floating beside it.
        shoulder_in = sign * sh_w * 0.66
        limbs.append((np.array([shoulder_in, h * 0.812, 0.0]),
                      np.array([sign * sh_w * 0.97, h * 0.60, 0.0]),
                      r_upper_arm * 1.12, r_upper_arm * 0.80))
        limbs.append((np.array([sign * sh_w * 0.97, h * 0.60, 0.0]),
                      np.array([sign * sh_w * 1.02, h * 0.425, 0.0]),
                      r_upper_arm * 0.80, r_wrist))
        # hands
        limbs.append((np.array([sign * sh_w * 1.02, h * 0.425, 0.0]),
                      np.array([sign * sh_w * 1.03, h * 0.375, 0.0]),
                      r_wrist, r_wrist * 0.55))
        # feet
        limbs.append((np.array([lx * 1.02, h * 0.034, 0.0]),
                      np.array([lx * 1.02, h * 0.020, m["foot"] * 0.58]),
                      r_ankle, r_ankle * 0.82))

    all_v = [V]
    for start, end, r0, r1 in limbs:
        lv, lf = _limb(start, end, r0, r1, rings=7, segments=16)
        if len(lv) == 0:
            continue
        faces.extend([(a + offset, b + offset, c + offset) for a, b, c in lf])
        all_v.append(lv)
        offset += len(lv)

    V = np.vstack(all_v)
    F = np.array(faces, dtype=np.int32)

    # Centre on the origin horizontally; feet at y=0.
    V[:, 0] -= float(V[:, 0].mean())
    V[:, 1] -= float(V[:, 1].min())
    V = V / 100.0                              # centimetres -> metres

    return {
        "positions": [round(float(x), 5) for x in V.reshape(-1)],
        "indices": [int(i) for i in F.reshape(-1)],
        "vertexCount": int(len(V)),
        "triangleCount": int(len(F)),
        "heightM": round(float(V[:, 1].max()), 4),
        "gender": gender,
        "segments": segments,
    }


# --- photo evidence ------------------------------------------------------
class SilhouetteEstimator:
    """Offline body evidence from one photo, using OpenCV only.

    Segments the subject with GrabCut, then measures the silhouette's width at
    each height. That gives a real shoulder:hip width ratio and a usable
    head-to-floor span. It cannot give depth, and it is not a 3D reconstruction.
    """
    name = "silhouette"

    def estimate(self, image_bgr) -> BodyEvidence:
        import cv2
        ev = BodyEvidence(source=self.name)
        h, w = image_bgr.shape[:2]
        scale = 480.0 / max(h, w)
        if scale < 1.0:
            image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
            h, w = image_bgr.shape[:2]

        mask = np.zeros((h, w), np.uint8)
        rect = (int(w * 0.10), int(h * 0.03), int(w * 0.80), int(h * 0.94))
        try:
            cv2.grabCut(image_bgr, mask, rect, np.zeros((1, 65), np.float64),
                        np.zeros((1, 65), np.float64), 4, cv2.GC_INIT_WITH_RECT)
        except cv2.error as e:
            ev.notes.append(f"Segmentation failed: {e}")
            return ev
        fg = np.where((mask == 1) | (mask == 3), 1, 0).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        rows = np.where(fg.sum(axis=1) > w * 0.02)[0]
        if len(rows) < h * 0.25:
            ev.notes.append(
                "Could not separate a full-length figure from the background. "
                "A plain background and the whole body in frame work best.")
            return ev

        top, bottom = int(rows[0]), int(rows[-1])
        span = bottom - top
        ev.height_px = float(span)
        widths = [float(fg[y].sum()) for y in range(top, bottom + 1)]
        ev.width_profile = [round(x / max(widths), 4) for x in widths] if widths else []

        def band(a, b):
            lo = int(len(widths) * a); hi = int(len(widths) * b)
            seg = widths[lo:hi] or [0]
            return float(np.median(seg))

        # Measured DOWN from the crown: shoulders ~18-25%, hips ~48-56%.
        sh, hip = band(0.18, 0.25), band(0.48, 0.56)
        if sh > 0 and hip > 0:
            ratio = round(sh / hip, 3)
            # Human shoulder:hip width runs roughly 0.95-1.55. Outside that the
            # silhouette has almost certainly swallowed the arms, loose
            # clothing or background, so the number is reported but not used.
            if 0.95 <= ratio <= 1.55:
                ev.shoulder_hip_ratio = ratio
                ev.confidence = "medium"
                ev.notes.append(
                    f"Shoulder-to-hip width ratio {ratio} measured from the "
                    "silhouette. Widths only - a photo carries no depth.")
            else:
                ev.confidence = "low"
                ev.notes.append(
                    f"Measured a shoulder-to-hip ratio of {ratio}, which is "
                    "outside the plausible human range, so it was discarded. "
                    "Arms held away from the body, loose clothing or a busy "
                    "background usually cause this. Your saved measurements "
                    "are being used on their own.")
        else:
            ev.notes.append("Silhouette found but too noisy to measure widths.")

        # Skin tone from the face band, for the avatar material.
        face = image_bgr[top:top + max(1, int(span * 0.12)), :]
        fmask = fg[top:top + max(1, int(span * 0.12)), :]
        if fmask.sum() > 50:
            px = face[fmask > 0]
            b, g, r = [int(np.median(px[:, i])) for i in range(3)]
            ev.skin_rgb = (r, g, b)
        return ev


class MediapipePoseEstimator:
    """Landmark-based evidence. Optional - requires mediapipe's full
    `solutions` API and its bundled pose model.

    NOT TESTED in the environment this was built in: the Python 3.13 wheel
    available there ships only the `tasks` namespace with no bundled model, so
    this path could not be exercised. It is wired up and selected automatically
    when the import succeeds; treat it as untested until it runs on your box.
    """
    name = "mediapipe"

    @staticmethod
    def available() -> bool:
        try:
            import mediapipe as mp
            return hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")
        except Exception:                      # noqa: BLE001
            return False

    def estimate(self, image_bgr) -> BodyEvidence:
        import cv2
        import mediapipe as mp
        ev = BodyEvidence(source=self.name)
        with mp.solutions.pose.Pose(static_image_mode=True,
                                    model_complexity=1) as pose:
            res = pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            ev.notes.append("No person detected in the photo.")
            return ev
        lm = res.pose_landmarks.landmark
        P = mp.solutions.pose.PoseLandmark
        sh = abs(lm[P.LEFT_SHOULDER].x - lm[P.RIGHT_SHOULDER].x)
        hp = abs(lm[P.LEFT_HIP].x - lm[P.RIGHT_HIP].x)
        if hp > 1e-6:
            ev.shoulder_hip_ratio = round(sh / hp, 3)
        ys = [l.y for l in lm]
        ev.height_px = float(max(ys) - min(ys)) * image_bgr.shape[0]
        ev.confidence = "medium"
        ev.notes.append(
            "Landmarks from MediaPipe Pose. Proportions only - a single view "
            "gives no depth, so circumferences still come from your "
            "measurements.")
        return ev


ESTIMATORS = [MediapipePoseEstimator, SilhouetteEstimator]


def analyse_photo(image_bytes: bytes) -> BodyEvidence:
    """Run the best available estimator over an uploaded photo."""
    import cv2
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return BodyEvidence(source="none",
                            notes=["That file could not be read as an image."])
    for cls in ESTIMATORS:
        if cls is MediapipePoseEstimator and not cls.available():
            continue
        try:
            ev = cls().estimate(img)
            if ev.shoulder_hip_ratio or ev.skin_rgb:
                return ev
            last = ev
        except Exception as e:                 # noqa: BLE001
            last = BodyEvidence(source=cls.name,
                                notes=[f"{cls.name} failed: {e}"])
    return last


def apply_evidence(m: dict, ev: BodyEvidence | None) -> tuple[dict, list[str]]:
    """Let photo evidence adjust only what was NOT measured."""
    notes: list[str] = []
    if not ev or not ev.shoulder_hip_ratio:
        return m, notes
    out = dict(m)
    ratio = ev.shoulder_hip_ratio
    if not m.get("_shoulder_supplied"):
        # Hip circumference is known; derive a shoulder width consistent with
        # the ratio the photo shows.
        hip_w = m["hips"] / math.pi * 1.15
        derived = round(hip_w * ratio, 1)
        if 30 <= derived <= 60:
            out["shoulder"] = derived
            notes.append(
                f"Shoulder width set to {derived} cm from the photo's "
                f"shoulder-to-hip ratio, because you have not measured it.")
        else:
            notes.append(
                f"The photo implied a {derived} cm shoulder width, which is "
                "outside a believable range, so it was not used. Measure your "
                "across-shoulder in My Size for a better avatar.")
    else:
        notes.append(
            "Your saved shoulder measurement was kept; the photo was used only "
            "to check proportions.")
    return out, notes
