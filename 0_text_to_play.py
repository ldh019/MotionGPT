from __future__ import annotations

import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation as RRR

from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.data.humanml.scripts.motion_process import recover_root_rot_pos
from mGPT.models.build_model import build_model
from mGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat
from mGPT.render.pyrender.smpl_render import SMPLRender
from mGPT.utils.rotation_conversions import quaternion_to_matrix


def load_model():
    """Build datamodule and model and load checkpoints."""
    saved = sys.argv
    sys.argv = [sys.argv[0]]  # use defaults
    cfg = parse_args(phase="webui")
    sys.argv = saved

    datamodule = build_data(cfg, phase="test")
    model = build_model(cfg, datamodule)
    if cfg.TEST.CHECKPOINTS:
        state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")[
            "state_dict"
        ]
        model.load_state_dict(state_dict)
    else:
        print("Warning: no checkpoints provided, using untrained model")
    device = torch.device("cuda" if cfg.ACCELERATOR == "gpu" else "cpu")
    model.to(device)
    model.eval()
    return model, datamodule, device


def text_to_motion(model, datamodule, device, texts):
    """Generate motion npy arrays from input texts.txt."""
    lengths = [datamodule.hparams.max_motion_length] * len(texts)
    batch = {"text": texts, "length": lengths}
    outputs = model(batch, task="t2m")
    motions = []
    for j, l in zip(outputs["feats"], outputs["length"]):
        motions.append(j[:l].detach().cpu().numpy())
    return motions


def text_to_feature():
    with open("input_scripts/texts.txt", "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    with open("input_scripts/names.txt", "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    output_path = Path("results/feature")

    model, datamodule, device = load_model()

    motions = text_to_motion(model, datamodule, device, texts)

    if len(motions) == 1 and not output_path.exists():
        np.save(output_path, motions[0])
        print(f"Motion saved to {output_path}")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(motions):
            fname = output_path / f"{names[i]}.npy"
            np.save(fname, m)
            print(f"Motion saved to {fname}")


def _load_datamodule():
    """Instantiate ``HumanML3DDataModule`` with default configs."""
    saved = sys.argv
    sys.argv = [sys.argv[0]]
    cfg = parse_args(phase="webui")
    sys.argv = saved
    return build_data(cfg, phase="test")


def convert_to_joint(input_path: Path, output_path: Path, datamodule) -> Path:
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


def feature_to_joint() -> None:
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

        out = convert_to_joint(input_path, output_path, datamodule)
        print(f"Joint data saved to {out}")


def convert_to_smpl(input_path: Path, output_path: Path) -> Path:
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


def joint_to_smpl() -> None:
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

        out = convert_to_smpl(input_path, output_path)
        print(f"SMPL pose saved to {out}")


def npz_to_gif(
    npz_path: Path,
    gif_path: Path,
    smpl_model_path: str = "deps/smpl/smpl_models/smpl",
    fps: float = 20.0,
    camera_scale: float = 0.7,
    camera_elevation: float = -30.0,
    camera_azimuth: float = 0.0,
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
    # Adjust camera placement
    cam_pose = np.array(renderer.renderer.camera_pose)
    # distance scaling
    cam_pose[2, 3] *= camera_scale
    # rotate around y axis (azimuth)
    yaw_r = RRR.from_euler("y", camera_azimuth, degrees=True).as_matrix()
    pitch_r = RRR.from_euler("x", camera_elevation, degrees=True).as_matrix()
    cam_pose[:3, 3] = yaw_r.dot(cam_pose[:3, 3])
    cam_pose[:3, :3] = yaw_r.dot(pitch_r)
    renderer.renderer.camera_pose = cam_pose

    frames = [renderer.render(i)[:, :, :3] for i in range(pose.shape[0])]
    imageio.mimsave(str(gif_path), frames, duration=1000.0 / fps)
    return gif_path


def smpl_play() -> None:
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
    text_to_feature()
    feature_to_joint()
    joint_to_smpl()
    smpl_play()