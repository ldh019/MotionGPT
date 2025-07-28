#!/usr/bin/env python3
"""Convert SMPL pose files to BVH motion files.

This utility reads an ``.npz`` or ``.pkl`` file containing SMPL pose
parameters and saves the motion in BVH format.  It mirrors the structure
of the other converter scripts in this repository.
"""

from pathlib import Path

import numpy as np

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

def main() -> None:
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
    main()