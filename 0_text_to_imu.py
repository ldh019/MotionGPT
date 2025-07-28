from __future__ import annotations
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Sequence

import imageio
import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as RRR

from bvh_utils import bvh, quat
from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.data.humanml.scripts.motion_process import recover_root_rot_pos
from mGPT.models.build_model import build_model
from mGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat
from mGPT.render.pyrender.smpl_render import SMPLRender
from mGPT.utils.rotation_conversions import matrix_to_axis_angle
from mGPT.utils.rotation_conversions import quaternion_to_matrix

from pathlib import Path

from imusim.io.bvh import BVHLoader
from imusim.trajectories.rigid_body import SplinedBodyModel
import time

try:
    import cupy as cp
    import cupyx.scipy.interpolate as cupy_interpolate
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CuPy not available, falling back to CPU processing")

# double check
version = 'ideal'
# version = 'sim'

_samplingPeriod = 0.
calibSamples = 1000
calibRotVel = 20

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


def smpl_to_bvh() -> None:
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


def gpu_quaternion_operations(quaternions):
    """GPU-accelerated quaternion operations"""
    if not CUDA_AVAILABLE:
        return quaternions

    # Move to GPU
    quat_gpu = cp.asarray(quaternions, dtype=cp.float32)

    # Normalize quaternions on GPU
    norms = cp.linalg.norm(quat_gpu, axis=1, keepdims=True)
    quat_gpu = quat_gpu / norms

    # Additional quaternion operations can be added here
    return cp.asnumpy(quat_gpu)


class GPUAcceleratedIMU:
    """GPU-accelerated IMU simulation wrapper"""

    def __init__(self, splined_model, joint_names, sampling_period):
        self.splined_model = splined_model
        self.joint_names = joint_names
        self.sampling_period = sampling_period
        self.start_time = splined_model.startTime
        self.end_time = splined_model.endTime

        # Pre-compute time array
        self.time_array = np.arange(self.start_time, self.end_time, sampling_period)

    def extract_trajectories_gpu(self):
        """Extract all joint trajectories using GPU acceleration"""
        trajectories = {}

        for joint_name in self.joint_names:
            joint = self.splined_model.getJoint(joint_name)

            # Extract position and orientation data
            positions = []
            orientations = []

            for t in self.time_array:
                pos = joint.position(t)
                ori = joint.rotation(t)
                positions.append([pos[0], pos[1], pos[2]])
                orientations.append([ori.w, ori.x, ori.y, ori.z])

            positions = np.array(positions)
            orientations = np.array(orientations)

            # GPU-accelerated processing
            if CUDA_AVAILABLE:
                orientations = gpu_quaternion_operations(orientations)

            trajectories[joint_name] = {
                'positions': positions,
                'orientations': orientations
            }

        return trajectories

    def compute_accelerations_gpu(self, trajectories):
        """Compute accelerations using GPU"""
        accelerations = {}

        for joint_name, traj in trajectories.items():
            positions = traj['positions']

            if CUDA_AVAILABLE:
                pos_gpu = cp.asarray(positions, dtype=cp.float32)
                # Compute second derivative (acceleration)
                vel_gpu = cp.gradient(pos_gpu, self.sampling_period, axis=0)
                acc_gpu = cp.gradient(vel_gpu, self.sampling_period, axis=0)
                accelerations[joint_name] = cp.asnumpy(acc_gpu)
            else:
                # Fallback to CPU
                vel = np.gradient(positions, self.sampling_period, axis=0)
                acc = np.gradient(vel, self.sampling_period, axis=0)
                accelerations[joint_name] = acc

        return accelerations

    def compute_angular_velocities_gpu(self, trajectories):
        """Compute angular velocities using GPU"""
        angular_velocities = {}

        for joint_name, traj in trajectories.items():
            orientations = traj['orientations']

            if CUDA_AVAILABLE:
                # Convert quaternions to angular velocities on GPU
                quat_gpu = cp.asarray(orientations, dtype=cp.float32)

                # Compute quaternion derivatives
                quat_dot = cp.gradient(quat_gpu, self.sampling_period, axis=0)

                # Convert to angular velocity
                # ω = 2 * q_dot * q_conj
                angular_vel = cp.zeros((len(orientations), 3), dtype=cp.float32)

                for i in range(len(orientations)):
                    q = quat_gpu[i]
                    q_dot = quat_dot[i]

                    # Quaternion multiplication for angular velocity
                    angular_vel[i, 0] = 2.0 * (
                                q_dot[0] * (-q[1]) + q_dot[1] * q[0] + q_dot[2] * (-q[3]) + q_dot[3] * q[2])
                    angular_vel[i, 1] = 2.0 * (
                                q_dot[0] * (-q[2]) + q_dot[1] * q[3] + q_dot[2] * q[0] + q_dot[3] * (-q[1]))
                    angular_vel[i, 2] = 2.0 * (
                                q_dot[0] * (-q[3]) + q_dot[1] * (-q[2]) + q_dot[2] * q[1] + q_dot[3] * q[0])

                angular_velocities[joint_name] = cp.asnumpy(angular_vel)
            else:
                # Fallback to CPU implementation
                angular_vel = np.zeros((len(orientations), 3))
                for i in range(1, len(orientations)):
                    dt = self.sampling_period
                    q1 = orientations[i - 1]
                    q2 = orientations[i]

                    # Simple finite difference approximation
                    dq = (q2 - q1) / dt
                    angular_vel[i] = 2.0 * np.array([
                        dq[0] * (-q1[1]) + dq[1] * q1[0] + dq[2] * (-q1[3]) + dq[3] * q1[2],
                        dq[0] * (-q1[2]) + dq[1] * q1[3] + dq[2] * q1[0] + dq[3] * (-q1[1]),
                        dq[0] * (-q1[3]) + dq[1] * (-q1[2]) + dq[2] * q1[1] + dq[3] * q1[0]
                    ])

                angular_velocities[joint_name] = angular_vel

        return angular_velocities


def extract_vir_imu_cuda(file_bvh, file_imu):
    """CUDA-accelerated IMU extraction"""

    closest_joint_from_sensor = {
        'Left_knee': 'Left_knee',
        'Left_ankle': 'Left_ankle',
        'Head': 'Head',
        'Left_wrist': 'Left_wrist',
        'Spine1': 'Spine1',
        'Spine3': 'Spine3',
        'Left_elbow': 'Left_elbow'
    }

    print(f"Using {'CUDA' if CUDA_AVAILABLE else 'CPU'} acceleration")
    print(f"Sensor mapping: {closest_joint_from_sensor}")

    # Create joint to sensor mapping
    list_joint2sensor = {}
    for sensor_name in closest_joint_from_sensor:
        joint_name = closest_joint_from_sensor[sensor_name]
        list_joint2sensor[joint_name] = sensor_name

    # Load BVH file
    start_time = time.time()
    with open(file_bvh, 'r') as bvh_file:
        loader = BVHLoader(bvh_file, 1)
        loader._readHeader()
        loader._readMotionData()
        model = loader.model
    print(f"BVH loading took: {time.time() - start_time:.2f}s")

    # Create splined model
    start_time = time.time()
    splined_model = SplinedBodyModel(model)
    print(f"Spline model creation took: {time.time() - start_time:.2f}s")

    sampling_period = (splined_model.endTime - splined_model.startTime) / loader.frameCount
    print(f'Frame count: {loader.frameCount}')
    print(f'Sampling period: {sampling_period}')

    # GPU-accelerated processing
    start_time = time.time()
    gpu_imu = GPUAcceleratedIMU(splined_model, list_joint2sensor.keys(), sampling_period)

    # Extract trajectories
    trajectories = gpu_imu.extract_trajectories_gpu()
    print(f"Trajectory extraction took: {time.time() - start_time:.2f}s")

    # Compute sensor data
    start_time = time.time()
    accelerations = gpu_imu.compute_accelerations_gpu(trajectories)
    angular_velocities = gpu_imu.compute_angular_velocities_gpu(trajectories)
    print(f"Sensor data computation took: {time.time() - start_time:.2f}s")

    # Organize results
    acc_seq = {}
    gyro_seq = {}

    for joint_name in list_joint2sensor:
        sensor_name = list_joint2sensor[joint_name]
        acc_seq[sensor_name] = accelerations[joint_name]
        gyro_seq[sensor_name] = angular_velocities[joint_name]
        print(f"{sensor_name}: acc {accelerations[joint_name].shape}, gyro {angular_velocities[joint_name].shape}")

    # Save results
    np.savez(file_imu, accel=acc_seq, gyro=gyro_seq)
    print(f'Saved all IMU data to : {file_imu}')

    return {
        'acc': acc_seq,
        'gyro': gyro_seq
    }

def bvh_to_imu() -> None:
    input_folder = Path("results/bvh")
    output_folder = Path("results/imu")

    with open("input_scripts/names.txt", "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    output_folder.mkdir(parents=True, exist_ok=True)

    for name in names:
        input_path = input_folder / f"{name}.bvh"
        if not input_path.exists():
            raise FileNotFoundError(f"{input_path} not found.")
        output_path = output_folder / f"{name}.npz"
        extract_vir_imu_cuda(input_path, output_path)
        print(f"IMU saved to {name}")


if __name__ == "__main__":
    text_to_feature()
    feature_to_joint()
    joint_to_smpl()
    smpl_play()
    smpl_to_bvh()
    bvh_to_imu()