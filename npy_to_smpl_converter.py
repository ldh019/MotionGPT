#!/usr/bin/env python3
"""Convert joint ``.npy`` files to SMPL pose ``.npy`` files.

The script is a tiny wrapper around the :class:`HybrIKJointsToRotmat`
class found in :mod:`mGPT.render.pyrender.hybrik_loc2rot`.  It reads one
or more joint files, infers rotation matrices and writes them back next
to the inputs.  Each output file receives a ``_pose.npy`` suffix unless
an explicit output location is provided.

Examples
--------
Convert a single file::

    python npy_to_smpl_converter.py path/to/joints.npy

Or convert multiple files at once::

    python npy_to_smpl_converter.py poses/seq_*.npy --output-dir smpl_poses
"""
import argparse
from pathlib import Path
import numpy as np

from mGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat


def convert(input_path: Path, output_path: Path, overwrite: bool = False) -> Path:
    """Convert joint positions stored in ``input_path`` to SMPL pose matrices.

    Parameters
    ----------
    input_path: :class:`~pathlib.Path`
        Path to the ``.npy`` file containing joint positions with shape
        ``(F, N, 3)`` where ``F`` is the number of frames and ``N`` the
        number of joints.
    output_path: :class:`~pathlib.Path`
        Where to save the resulting SMPL pose file.
    overwrite: bool, optional
        Whether to overwrite ``output_path`` if it already exists.

    Returns
    -------
    :class:`~pathlib.Path`
        The path to the written pose file.
    """
    data = np.load(str(input_path))
    if data.ndim == 4:
        data = data[0]

    # Center the motion on the first joint of the first frame
    data = data - data[0, 0]

    pose_generator = HybrIKJointsToRotmat()
    pose = pose_generator(data)

    # Append identity rotations for the last two SMPL joints
    identity = np.stack([np.eye(3)] * pose.shape[0], 0)
    pose = np.concatenate([pose, np.stack([identity] * 2, 1)], 1)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists. Use --force to overwrite")
    np.save(output_path, pose)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert joint npy files to SMPL pose files")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input joint npy files. Shell globs are supported by most shells",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to store resulting pose files. Defaults to alongside inputs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing pose files",
    )
    parser.add_argument(
        "--output",
        help="Destination file when a single input is provided",
    )
    args = parser.parse_args()

    inputs = [Path(p) for p in args.inputs]
    output_dir = Path(args.output_dir) if args.output_dir else None

    output_paths = []
    for i, inp in enumerate(inputs):
        if output_dir:
            out_path = output_dir / f"{inp.stem}_pose.npy"
        elif i == 0 and args.output:
            out_path = Path(args.output)
        else:
            out_path = inp.with_name(f"{inp.stem}_pose.npy")
        out = convert(inp, out_path, overwrite=args.force)
        output_paths.append(out)
        print(f"SMPL pose saved to {out}")


if __name__ == '__main__':
    main()
