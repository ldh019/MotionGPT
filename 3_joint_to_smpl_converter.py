#!/usr/bin/env python3
"""Convert joint sequences to SMPL pose matrices."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import numpy as np

from mGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat


def convert(input_path: Path, output_path: Path) -> Path:
    """Convert joints in ``input_path`` to SMPL pose and save as ``output_path``."""
    data = np.load(str(input_path))

    if isinstance(data, np.lib.npyio.NpzFile):
        joints = data["joints"]
        root_trans = data.get("trans")
        root_rot = data.get("root_rot")
    else:
        joints = data
        root_trans = None
        root_rot = None

    if joints.ndim == 4:
        joints = joints[0]

    if root_trans is not None:
        joints_local = joints - root_trans.reshape(-1, 1, 3)
    else:
        joints_local = joints - joints[0, 0]

    pose_generator = HybrIKJointsToRotmat()
    pose = pose_generator(joints_local)

    identity = np.stack([np.eye(3)] * pose.shape[0], 0)
    pose = np.concatenate([pose, np.stack([identity] * 2, 1)], 1)

    if root_trans is None:
        root_trans = np.zeros((pose.shape[0], 3))
    if root_rot is None:
        root_rot = pose[:, 0]

    np.savez(str(output_path), pose=pose, trans=root_trans, root_rot=root_rot)
    return output_path


def main() -> None:
    with open("input_scripts/names.txt", "r", encoding="utf-8") as f:
        input_files = [line.strip() for line in f if line.strip()]

    input_folder = "results/joint/"
    output_folder = "results/smpl"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for input_file in input_files:
        if not input_file.endswith(".npz"):
            input_file += ".npz"
        if not os.path.exists(input_folder + input_file):
            raise FileNotFoundError(f"{input_folder + input_file} not found.")
        input_path = Path(input_folder + input_file)
        output_path = Path(output_folder + f"/{input_path.stem}.npz")

        out = convert(input_path, output_path)
        print(f"SMPL pose saved to {out}")


if __name__ == "__main__":
    main()