"""Bake the fitting-room body meshes from Anny (NAVER LABS, Apache 2.0; MakeHuman
CC0 assets). Run once on a dev machine - the server never imports anny or torch.

    ANNY_CACHE_DIR=... <venv with `pip install anny==0.6.2`>/bin/python tools/bake_bodies.py

Writes backend/app/data/bodies/anny_{female,male}.npz. Each file holds one
shared triangle list and a small grid of posed bodies (weight x muscle), in
metres, Y up, feet on y=0, facing +Z, arms lowered from Anny's A-pose.
Per vertex it also stores a body-region code from the dominant skinning bone,
which the server uses to cut garments out of the body surface.
"""
from __future__ import annotations

import math
import pathlib
import re

import anny
import numpy as np
import torch

OUT = pathlib.Path(__file__).resolve().parent.parent / "backend/app/data/bodies"
WEIGHTS = [0.2, 0.5, 0.8]
MUSCLES = [0.3, 0.7]
ARM_DROP_DEG = 32          # rotate each upper arm down towards the body

REGIONS = ["head", "neck", "torso", "hips", "upperarm", "forearm", "hand",
           "thigh", "shin", "foot"]
_BONE_REGION = [
    (r"^(head|eye)", "head"), (r"^neck", "neck"),
    (r"^(spine|clavicle|shoulder)", "torso"), (r"^(root|pelvis)", "hips"),
    (r"^upperarm", "upperarm"), (r"^lowerarm", "forearm"),
    (r"^(wrist|finger|metacarpal)", "hand"), (r"^upperleg", "thigh"),
    (r"^lowerleg", "shin"), (r"^(foot|toe)", "foot"),
]
JOINTS = {"shoulder_l": "upperarm01.L", "shoulder_r": "upperarm01.R",
          "elbow_l": "lowerarm01.L", "elbow_r": "lowerarm01.R",
          "wrist_l": "wrist.L", "wrist_r": "wrist.R",
          "hip_l": "upperleg01.L", "hip_r": "upperleg01.R",
          "knee_l": "lowerleg01.L", "knee_r": "lowerleg01.R",
          "ankle_l": "foot.L", "ankle_r": "foot.R",
          "neck": "neck01", "head": "head"}


def _rot_y(deg: float) -> torch.Tensor:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)


def _to_y_up(p: np.ndarray, floor: float) -> np.ndarray:
    # Anny is Z up and faces -Y. Rotate -90 deg about X: (x, y, z) -> (x, z, -y).
    q = np.stack([p[..., 0], p[..., 2] - floor, -p[..., 1]], axis=-1)
    return q.astype(np.float32)


def bone_regions(model) -> np.ndarray:
    labels = list(model.bone_labels)
    code = []
    for b in labels:
        for pat, reg in _BONE_REGION:
            if re.match(pat, b):
                code.append(REGIONS.index(reg))
                break
        else:
            raise ValueError(f"unmapped bone {b}")
    idx = model.vertex_bone_indices.cpu().numpy()      # (V, K)
    w = model.vertex_bone_weights.cpu().numpy()        # (V, K)
    dominant = idx[np.arange(len(idx)), w.argmax(axis=1)]
    code = np.asarray(code, dtype=np.uint8)
    limb = code[idx]
    def share(*regs):
        return (w * np.isin(limb, [REGIONS.index(r) for r in regs])).sum(axis=1).astype(np.float32)
    return code[dominant], share("upperarm", "forearm", "hand"), share("thigh", "shin", "foot")


def bake(model, gender: float) -> dict:
    labels = list(model.bone_labels)
    pose = torch.eye(4)[None, None].repeat(1, model.bone_count, 1, 1)
    pose[0, labels.index("upperarm01.L"), :3, :3] = _rot_y(ARM_DROP_DEG)
    pose[0, labels.index("upperarm01.R"), :3, :3] = _rot_y(-ARM_DROP_DEG)
    bodies, joints = [], []
    for w in WEIGHTS:
        for mu in MUSCLES:
            ph = {"gender": gender, "age": 0.5, "muscle": mu, "weight": w,
                  "height": 0.5, "proportions": 0.5}
            out = model(pose_parameters=pose, phenotype_kwargs=ph)
            v = out["vertices"][0].detach().cpu().numpy()
            heads = out["bone_poses"][0][:, :3, 3].detach().cpu().numpy()
            floor = float(v[:, 2].min())
            bodies.append(_to_y_up(v, floor))
            joints.append(np.stack([_to_y_up(heads[labels.index(b)], floor)
                                    for b in JOINTS.values()]))
    return {"vertices": np.stack(bodies), "joints": np.stack(joints)}


def main() -> None:
    model = anny.Anny().to(dtype=torch.float32)
    faces = np.asarray(model.faces, dtype=np.int32)
    regions, arm_w, leg_w = bone_regions(model)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, g in (("male", 0.0), ("female", 1.0)):   # anny: gender anchors ["male", "female"]
        d = bake(model, g)
        np.savez_compressed(
            OUT / f"anny_{name}.npz", faces=faces, regions=regions, arm_w=arm_w, leg_w=leg_w,
            region_names=np.array(REGIONS), vertices=d["vertices"],
            joints=d["joints"], joint_names=np.array(list(JOINTS)),
            weights=np.array(WEIGHTS, dtype=np.float32),
            muscles=np.array(MUSCLES, dtype=np.float32),
            source=np.array(f"anny {anny.__version__}, rig anny, topology anny, "
                            f"age 0.5, height 0.5, proportions 0.5, arm drop {ARM_DROP_DEG} deg"))
        v = d["vertices"]
        print(name, v.shape, "height m", float(v[0, :, 1].max()),
              "width m", float(np.ptp(v[0, :, 0])))
    print("regions", {r: int((regions == i).sum()) for i, r in enumerate(REGIONS)})


if __name__ == "__main__":
    main()
