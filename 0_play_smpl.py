#!/usr/bin/env python3
"""Render SMPL pose sequences to GIF.

This utility loads a ``.npz`` file produced by
``feature_to_smpl_converter.py`` and renders the animation to a GIF file.
It performs a simple coordinate system conversion so that the avatar is
upright on the floor and the camera is placed fairly close to the body.
"""

from __future__ import annotations

from pathlib import Path

import imageio
import numpy as np
from scipy.spatial.transform import Rotation as RRR
import os

from mGPT.render.pyrender.smpl_render import SMPLRender


def npz_to_gif(
    npz_path: Path,
    gif_path: Path,
    smpl_model_path: str = "deps/smpl/smpl_models/smpl",
    fps: float = 20.0,
    camera_scale: float = 0.7
) -> Path:
    """Render ``npz_path`` (containing ``pose``, ``trans`` and ``root_rot``) to
    ``gif_path``."""

    data = np.load(str(npz_path))
    pose = data["pose"]
    trans = data["trans"]
    root_rot = data["root_rot"]

    # Combine root rotation if provided
    if root_rot.shape == pose[:, 0].shape:
        pose[:, 0] = root_rot

    # Subtract initial translation to keep the motion near the origin
    trans = trans - trans[0]

    # Rotate pose and translation to match pyrender coordinates
    r = RRR.from_rotvec(np.array([np.pi, 0.0, 0.0]))
    rot = r.as_matrix()
    pose[:, 0] = np.einsum("ij,tjk->tik", rot, pose[:, 0])
    trans = np.einsum("ij,tj->ti", rot, trans)

    params = dict(pred_shape=np.zeros([1, 10]),
                  pred_root=trans,
                  pred_pose=pose)
    renderer = SMPLRender(smpl_model_path)
    renderer.init_renderer([768, 768, 3], params)
    # Move the camera slightly closer to the avatar
    renderer.renderer.camera_pose[2][3] *= camera_scale

    frames = [renderer.render(i)[:, :, :3] for i in range(pose.shape[0])]
    imageio.mimsave(str(gif_path), frames, duration=1000.0 / fps)
    return gif_path


def main() -> None:
    with open("input_scripts/names.txt", "r", encoding="utf-8") as f:
        input_files = [line.strip() for line in f if line.strip()]

    input_folder = "results/smpl/"
    output_folder = "results/gif"
    model_folder = "deps/smpl_models/smpl"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for input_file in input_files:
        if not input_file.endswith(".npz"):
            input_file += ".npz"
        if not os.path.exists(input_folder + input_file):
            raise FileNotFoundError(f"{input_folder + input_file} not found.")
        input_path = Path(input_folder + input_file)
        output_path = Path(output_folder + f"/{input_path.stem}.gif")

        npz_to_gif(input_path, output_path, smpl_model_path=model_folder, fps=20)
        print(f"GIF saved to {output_path}")


if __name__ == "__main__":
    main()
