"""Replaceable state-estimation boundary used by the policy controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np

from locomotion.imu import ICM42688_CONFIG, Icm42688OrientationEstimator
from locomotion.robot_protocol import RawRobotState


@dataclass(frozen=True, slots=True)
class EstimatedRobotState:
    sample_time_ns: int
    joint_sample_time_ns: int
    imu_sample_time_ns: int
    joint_names: tuple[str, ...]
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    base_angular_velocity: np.ndarray
    projected_gravity: np.ndarray


class StateEstimator(Protocol):
    def update(self, raw_state: RawRobotState) -> None: ...
    def latest(self) -> EstimatedRobotState | None: ...
    def reset(self) -> None: ...


class Icm42688StateEstimator:
    """Estimate policy state from raw ICM-42688-P gyro and acceleration."""

    def __init__(self, *, sensor_to_body: np.ndarray | None = None):
        self.imu = Icm42688OrientationEstimator(
            ICM42688_CONFIG, sensor_to_body=sensor_to_body
        )
        self._latest: EstimatedRobotState | None = None

    def reset(self) -> None:
        self.imu.reset()
        self._latest = None

    def update(self, raw_state: RawRobotState) -> None:
        raw_state.validate()
        joint_sample_time_ns = (
            raw_state.sample_time_ns
            if raw_state.joint_sample_time_ns is None
            else raw_state.joint_sample_time_ns
        )
        imu_sample_time_ns = (
            raw_state.sample_time_ns
            if raw_state.imu_sample_time_ns is None
            else raw_state.imu_sample_time_ns
        )
        self.imu.update(
            np.asarray(raw_state.imu_angular_velocity),
            np.asarray(raw_state.imu_linear_acceleration),
            imu_sample_time_ns,
        )
        self._latest = EstimatedRobotState(
            sample_time_ns=raw_state.sample_time_ns,
            joint_sample_time_ns=joint_sample_time_ns,
            imu_sample_time_ns=imu_sample_time_ns,
            joint_names=raw_state.joint_names,
            joint_position=np.asarray(raw_state.joint_position, dtype=np.float32),
            joint_velocity=np.asarray(raw_state.joint_velocity, dtype=np.float32),
            base_angular_velocity=self.imu.angular_velocity.copy(),
            projected_gravity=self.imu.projected_gravity.copy(),
        )

    def latest(self) -> EstimatedRobotState | None:
        return self._latest


class PassthroughStateEstimator:
    """Legacy diagnostic estimator that uses a supplied perfect orientation."""

    def __init__(self):
        self._latest: EstimatedRobotState | None = None

    def reset(self) -> None:
        self._latest = None

    def update(self, raw_state: RawRobotState) -> None:
        raw_state.validate()
        joint_sample_time_ns = (
            raw_state.sample_time_ns
            if raw_state.joint_sample_time_ns is None
            else raw_state.joint_sample_time_ns
        )
        imu_sample_time_ns = (
            raw_state.sample_time_ns
            if raw_state.imu_sample_time_ns is None
            else raw_state.imu_sample_time_ns
        )
        if raw_state.base_orientation_wxyz is None:
            raise ValueError(
                "Passthrough estimator requires base_orientation_wxyz; replace it "
                "with an IMU estimator for raw hardware gyro/accelerometer data"
            )
        quaternion = np.asarray(raw_state.base_orientation_wxyz, dtype=np.float64)
        norm = np.linalg.norm(quaternion)
        if norm < 1.0e-8:
            raise ValueError("base orientation quaternion has zero norm")
        quaternion /= norm
        inverse_quaternion = quaternion.copy()
        inverse_quaternion[1:] *= -1.0
        projected_gravity = np.empty(3, dtype=np.float64)
        mujoco.mju_rotVecQuat(
            projected_gravity,
            np.array([0.0, 0.0, -1.0]),
            inverse_quaternion,
        )
        self._latest = EstimatedRobotState(
            sample_time_ns=raw_state.sample_time_ns,
            joint_sample_time_ns=joint_sample_time_ns,
            imu_sample_time_ns=imu_sample_time_ns,
            joint_names=raw_state.joint_names,
            joint_position=np.asarray(raw_state.joint_position, dtype=np.float32),
            joint_velocity=np.asarray(raw_state.joint_velocity, dtype=np.float32),
            base_angular_velocity=np.asarray(
                raw_state.imu_angular_velocity, dtype=np.float32
            ),
            projected_gravity=projected_gravity.astype(np.float32),
        )

    def latest(self) -> EstimatedRobotState | None:
        return self._latest
