#!/usr/bin/env python3
"""Convert motion feature ``.npy`` files to joint sequences.

This script loads motion features as produced by MotionGPT datasets and
converts them to global joint positions. In addition to the joint array,
root translation and rotation are stored so that a subsequent conversion
to SMPL pose can recover the global transform.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.data.humanml.scripts.motion_process import recover_root_rot_pos
from mGPT.utils.rotation_conversions import quaternion_to_matrix


def _load_datamodule():
    """Instantiate ``HumanML3DDataModule`` with default configs."""
    saved = sys.argv
    sys.argv = [sys.argv[0]]
    cfg = parse_args(phase="webui")
    sys.argv = saved
    return build_data(cfg, phase="test")


def convert(input_path: Path, output_path: Path, datamodule) -> Path:
    """Convert feature file ``input_path`` to joints and save to ``output_path``."""
    feats_np = np.load(str(input_path))
    if feats_np.ndim == 3 and feats_np.shape[0] == 1:
        feats_np = feats_np[0]
    feats = torch.tensor(feats_np, dtype=torch.float32)

    joints = datamodule.feats2joints(feats).cpu()
    if joints.ndim == 4:
        joints = joints[0]
    joints_np = joints.numpy()

    feats_denorm = datamodule.denormalize(feats)
    r_quat, r_pos = recover_root_rot_pos(feats_denorm)
    root_trans = r_pos.cpu().numpy()
    root_rot = quaternion_to_matrix(r_quat).cpu().numpy()

    np.savez(str(output_path), joints=joints_np, trans=root_trans, root_rot=root_rot)
    return output_path


def main() -> None:
    with open("input_scripts/names.txt", "r", encoding="utf-8") as f:
        input_files = [line.strip() for line in f if line.strip()]

    input_folder = "results/feature/"
    output_folder = "results/joint"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for input_file in input_files:
        if not input_file.endswith(".npy"):
            input_file += ".npy"
        if not os.path.exists(input_folder + input_file):
            raise FileNotFoundError(f"{input_folder + input_file} not found.")
        input_path = Path(input_folder + input_file)
        output_path = Path(output_folder + f"/{input_path.stem}.npz")

        datamodule = _load_datamodule()

        out = convert(input_path, output_path, datamodule)
        print(f"Joint data saved to {out}")


if __name__ == "__main__":
    main()