#!/usr/bin/env python3
"""Convert SMPL pose files to BVH motion files.

This utility reads an ``.npz`` or ``.pkl`` file containing SMPL pose
parameters and saves the motion in BVH format.  It mirrors the structure
of the other converter scripts in this repository.
"""

from pathlib import Path

import numpy as np

from imusim.behaviours.imu import BasicIMUBehaviour
from imusim.environment.base import Environment
from imusim.io.bvh import BVHLoader
from imusim.platforms.imus import IdealIMU, Orient3IMU
from imusim.simulation.base import Simulation
from imusim.simulation.calibrators import ScaleAndOffsetCalibrator
from imusim.trajectories.rigid_body import SplinedBodyModel

# double check
version = 'ideal'
# version = 'sim'

_samplingPeriod = 0.
calibSamples = 1000
calibRotVel = 20


def extract_vir_imu(file_bvh, file_acc, file_gyro):
    closest_joint_from_sensor = {'Left_knee': 'Left_knee',
                                 'Left_ankle': 'Left_ankle',
                                 'Head': 'Head',
                                 'Left_wrist': 'Left_wrist',
                                 'Spine1': 'Spine1',
                                 'Spine3': 'Spine3',
                                 'Left_elbow': 'Left_elbow'}

    print(closest_joint_from_sensor)

    list_joint2sensor = {}
    for sensor_name in closest_joint_from_sensor:
        joint_name = closest_joint_from_sensor[sensor_name]
        list_joint2sensor[joint_name] = sensor_name

    sensor = {}

    # Extact virtual sensor with imusim
    # updated to python 3 version
    path_bvh = file_bvh
    path_acc = file_acc
    path_gyro = file_gyro

    # load mocap
    with open(path_bvh, 'r') as bvhFile:
        conversionFactor = 1
        loader = BVHLoader(bvhFile, conversionFactor)
        loader._readHeader()
        loader._readMotionData()
        model = loader.model
    print('load mocap from ...', path_bvh)

    # spline intrepolation
    splinedModel = SplinedBodyModel(model)
    startTime = splinedModel.startTime
    endTime = splinedModel.endTime

    if _samplingPeriod == 0.:
        samplingPeriod = (endTime - startTime) / loader.frameCount
    else:
        samplingPeriod = _samplingPeriod
    print('frameCount:', loader.frameCount)
    print('samplingPeriod:', samplingPeriod)

    if version == 'ideal':
        print('Simulating ideal IMU.')

        # set simulation
        sim = Simulation()
        sim.time = startTime

        # run simulation
        dict_imu = {}
        for joint_name in list_joint2sensor:
            imu = IdealIMU()
            imu.simulation = sim
            imu.trajectory = splinedModel.getJoint(joint_name)

            BasicIMUBehaviour(imu, samplingPeriod)

            dict_imu[joint_name] = imu

        sim.run(endTime)

    elif version == 'sim':
        print('Simulating Orient3IMU.')

        # set simulation
        env = Environment()
        calibrator = ScaleAndOffsetCalibrator(env, calibSamples, samplingPeriod, calibRotVel)
        sim = Simulation(environment=env)
        sim.time = startTime

        # run simulation
        dict_imu = {}
        for joint_name in list_joint2sensor:
            imu = Orient3IMU()
            calibration = calibrator.calibrate(imu)
            print('imu calibration:', joint_name)

            imu.simulation = sim
            imu.trajectory = splinedModel.getJoint(joint_name)

            BasicIMUBehaviour(imu, samplingPeriod, calibration, initialTime=sim.time)

            dict_imu[joint_name] = imu

        sim.run(endTime)

    # collect sensor values
    acc_seq = {}
    gyro_seq = {}
    for joint_name in list_joint2sensor:
        sensor_name = list_joint2sensor[joint_name]
        imu = dict_imu[joint_name]

        if version == 'ideal':
            acc_seq[sensor_name] = imu.accelerometer.rawMeasurements.values.T
            gyro_seq[sensor_name] = imu.gyroscope.rawMeasurements.values.T
        elif version == 'sim':
            acc_seq[sensor_name] = imu.accelerometer.calibratedMeasurements.values.T
            gyro_seq[sensor_name] = imu.gyroscope.calibratedMeasurements.values.T

        print(sensor_name, acc_seq[sensor_name].shape, gyro_seq[sensor_name].shape)

    # save
    np.savez(path_acc, **acc_seq)
    print('save in ...', path_acc)
    np.savez(path_gyro, **gyro_seq)
    print('save in ...', path_gyro)

    acc = acc_seq
    gyro = gyro_seq

    sensor = {
        'acc': acc,
        'gyro': gyro}

    return sensor


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
        output_acc_path = output_folder / f"{name}_acc.npz"
        output_gyro_path = output_folder / f"{name}_gyro.npz"
        extract_vir_imu(input_path, output_acc_path, output_gyro_path)
        print(f"BVH saved to {name}")

if __name__ == "__main__":
    main()