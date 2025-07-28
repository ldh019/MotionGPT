#!/usr/bin/env python3
"""Convert SMPL pose files to BVH motion files.

This utility reads an ``.npz`` or ``.pkl`` file containing SMPL pose
parameters and saves the motion in BVH format.  It mirrors the structure
of the other converter scripts in this repository.
"""

from __future__ import annotations

import os
from pathlib import Path
import pickle
from typing import Sequence

import numpy as np
import smplx
import torch

from mGPT.utils.rotation_conversions import matrix_to_axis_angle

from bvh_utils import bvh, quat


NAMES: Sequence[str] = [
    "Pelvis",
    "Left_hip",
    "Right_hip",
    "Spine1",
    "Left_knee",
    "Right_knee",
    "Spine2",
    "Left_ankle",
    "Right_ankle",
    "Spine3",
    "Left_foot",
    "Right_foot",
    "Neck",
    "Left_collar",
    "Right_collar",
    "Head",
    "Left_shoulder",
    "Right_shoulder",
    "Left_elbow",
    "Right_elbow",
    "Left_wrist",
    "Right_wrist",
    "Left_palm",
    "Right_palm",
]


def _mirror_rot_trans(lrot: np.ndarray, trans: np.ndarray, names: Sequence[str], parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joints_mirror = np.array([
        (
            names.index("Left" + n[5:]) if n.startswith("Right") else (
                names.index("Right" + n[4:]) if n.startswith("Left") else names.index(n)
            )
        )
        for n in names
    ])

    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([1, 1, -1, -1])
    grot = quat.fk_rot(lrot, parents)
    trans_mirror = mirror_pos * trans
    grot_mirror = mirror_rot * grot[:, joints_mirror]
    return quat.ik_rot(grot_mirror, parents), trans_mirror


def convert(
    input_path: Path,
    output_path: Path,
    model_path: Path = Path("data/smpl"),
    model_type: str = "smpl",
    gender: str = "MALE",
    num_betas: int = 10,
    fps: int = 60,
    mirror: bool = False,
    overwrite: bool = True,
) -> Path:
    """Convert ``input_path`` SMPL pose to BVH and save to ``output_path``."""

    model = smplx.create(str(model_path), model_type=model_type, gender=gender, batch_size=1)
    parents = model.parents.detach().cpu().numpy()

    # rest = model()
    with open(input_path, "rb") as f:
        if input_path.suffix == ".npz":
            data = np.load(f)
        elif input_path.suffix == ".pkl":
            data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file: {input_path}")

        # ``pose`` may have various names and shapes.  We expect rotation
        # matrices with shape (F, 24, 3, 3).
        if "pose" in data:
            rots = data["pose"]
        elif "poses" in data:
            rots = data["poses"]
        elif "smpl_poses" in data:
            rots = data["smpl_poses"]
        else:
            raise KeyError("pose data not found in input file")

        trans = data["trans"] if "trans" in data else data.get("smpl_trans")
        scaling = data.get("smpl_scaling")
        betas = data.get("betas") or data.get("smpl_betas")

        # Remove any leading singleton dimension.
        rots = np.asarray(rots)
        if rots.ndim == 5 and rots.shape[0] == 1:
            rots = rots[0]
        if trans is not None:
            trans = np.asarray(trans)

    if betas is not None:
        betas_t = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
        rest = model(betas=betas_t)
    else:
        rest = model()
    #

    rest_pose = rest.joints.detach().cpu().numpy().squeeze()[:24, :]

    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = root_offset
    offsets *= 100

    # ``rots`` could be provided as rotation matrices. Convert to axis-angle if needed.
    if rots.ndim == 4 and rots.shape[-1] == 3 and rots.shape[-2] == 3:
        rots = matrix_to_axis_angle(torch.from_numpy(rots)).numpy()

    if scaling is not None:
        trans = trans / scaling

    rots = quat.from_axis_angle(rots)
    order = "zyx"
    pos = offsets[None].repeat(len(rots), axis=0)
    positions = pos.copy()
    positions[:, 0] += trans * 100
    rotations = np.degrees(quat.to_euler(rots, order=order))

    bvh_data = {
        "rotations": rotations,
        "positions": positions,
        "offsets": offsets,
        "parents": parents,
        "names": NAMES,
        "order": order,
        "frametime": 1 / fps,
    }

    if output_path.suffix != ".bvh":
        output_path = output_path.with_suffix(".bvh")

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists. Use --overwrite to replace")

    bvh.save(str(output_path), bvh_data)

    if mirror:
        rots_mirror, trans_mirror = _mirror_rot_trans(rots, trans, NAMES, parents)
        positions_mirror = pos.copy()
        positions_mirror[:, 0] += trans_mirror
        rotations_mirror = np.degrees(quat.to_euler(rots_mirror, order=order))

        bvh_data = {
            "rotations": rotations_mirror,
            "positions": positions_mirror,
            "offsets": offsets,
            "parents": parents,
            "names": NAMES,
            "order": order,
            "frametime": 1 / fps,
        }
        mirror_path = output_path.with_name(output_path.stem + "_mirror.bvh")
        bvh.save(str(mirror_path), bvh_data)

    return output_path
def main() -> None:
    input_folder = Path("results/smpl")
    output_folder = Path("results/bvh")
    model_folder = Path("deps/smpl_models")

    with open("input_scripts/names.txt", "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    output_folder.mkdir(parents=True, exist_ok=True)

    for name in names:
        input_path = input_folder / f"{name}.npz"
        if not input_path.exists():
            raise FileNotFoundError(f"{input_path} not found.")
        output_path = output_folder / f"{name}.bvh"
        out = convert(input_path, output_path, model_folder)
        print(f"BVH saved to {out}")


if __name__ == "__main__":
    os.environ.setdefault("DISPLAY", ":0.0")
    main()