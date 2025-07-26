#!/usr/bin/env python3
"""Convert motion feature ``.npy`` files to SMPL pose files.

The script reads an ``.npy`` file containing motion features (as used in the
HumanML3D dataset) and outputs the corresponding SMPL pose matrices.

It can be used as a module or run directly as a script.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import sys
import torch

from mGPT.data.humanml.scripts.motion_process import recover_root_rot_pos
from mGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat
from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.utils.rotation_conversions import quaternion_to_matrix


def _load_datamodule():
    """Load :class:`HumanML3DDataModule` with default configs."""
    saved = sys.argv
    sys.argv = [sys.argv[0]]  # use defaults
    cfg = parse_args(phase="webui")
    sys.argv = saved
    return build_data(cfg, phase="test")


def convert(input_path: Path, output_path: Path, datamodule, overwrite: bool = True) -> Path:
    """Convert feature ``input_path`` to SMPL pose ``output_path``."""
    feats_np = np.load(str(input_path))
    if feats_np.ndim == 3 and feats_np.shape[0] == 1:
        feats_np = feats_np[0]
    feats = torch.tensor(feats_np, dtype=torch.float32)

    joints = datamodule.feats2joints(feats).cpu()
    if joints.ndim == 4:
        joints = joints[0]

    feats_denorm = datamodule.denormalize(feats)
    r_quat, r_pos = recover_root_rot_pos(feats_denorm)
    root_trans = r_pos.cpu().numpy()
    root_rot = quaternion_to_matrix(r_quat).cpu().numpy()

    joints = joints - r_pos.unsqueeze(1)
    pose_generator = HybrIKJointsToRotmat()
    pose = pose_generator(joints.numpy())

    identity = np.stack([np.eye(3)] * pose.shape[0], 0)
    pose = np.concatenate([pose, np.stack([identity] * 2, 1)], 1)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists. Use --overwrite to replace")

    np.savez(str(output_path), pose = pose, trans = root_trans, root_rot = root_rot)
    return output_path


def main() -> None:
    os.environ.setdefault("DISPLAY", ":0.0")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    input_folder = "results/npy/"
    with open("input_scripts/names.txt", "r", encoding="utf-8") as f:
        input_files = [line.strip() for line in f if line.strip()]

    output_folder = "results/smpl"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for input_file in input_files:
        if not input_file.endswith(".npy"):
            input_file += ".npy"
        if not os.path.exists(input_folder + input_file):
            raise FileNotFoundError(f"{input_folder + input_file} not found.")
        input_path = Path(input_folder + input_file)
        output_path = Path(output_folder + f"/{input_path.stem}_pose")

        print(input_path)

        datamodule = _load_datamodule()

        out = convert(input_path, output_path, datamodule)
        print(f"SMPL pose saved to {out}")


if __name__ == "__main__":
    main()