#!/usr/bin/env python3
"""Simple utility to convert joint npy files to SMPL pose npy files.

This script loads a numpy array containing 3D joint locations and
converts them to SMPL rotation matrices using the ``HybrIKJointsToRotmat``
class already provided in ``mGPT.render.pyrender.hybrik_loc2rot``. The
resulting pose array can then be used by other utilities in the
repository for visualization or further processing.

Example
-------
    python npy_to_smpl_converter.py input_joints.npy --output output_pose.npy

If ``--output`` is not given, ``_pose.npy`` will be appended to the input
file name.
"""
import argparse
import numpy as np

from mGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat


def convert(input_path: str, output_path: str | None = None) -> str:
    """Convert joint positions stored in ``input_path`` to SMPL pose matrices.

    Parameters
    ----------
    input_path: str
        Path to the ``.npy`` file containing joint positions with shape
        ``(F, N, 3)`` where ``F`` is the number of frames and ``N`` the
        number of joints.
    output_path: str | None, optional
        Where to save the resulting SMPL pose file. If ``None``,
        ``input_path`` with ``_pose.npy`` suffix will be used.

    Returns
    -------
    str
        The path to the written pose file.
    """
    data = np.load(input_path)
    if data.ndim == 4:
        data = data[0]

    # Center the motion on the first joint of the first frame
    data = data - data[0, 0]

    pose_generator = HybrIKJointsToRotmat()
    pose = pose_generator(data)

    # Append identity rotations for the last two SMPL joints
    identity = np.stack([np.eye(3)] * pose.shape[0], 0)
    pose = np.concatenate([pose, np.stack([identity] * 2, 1)], 1)

    if output_path is None:
        output_path = input_path.replace('.npy', '_pose.npy')

    np.save(output_path, pose)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a joint npy file to a SMPL pose npy file")
    parser.add_argument('input', help='Path to the input joint npy file')
    parser.add_argument('--output', help='Destination path for the SMPL pose')
    args = parser.parse_args()

    output_path = convert(args.input, args.output)
    print(f"SMPL pose saved to {output_path}")


if __name__ == '__main__':
    main()
