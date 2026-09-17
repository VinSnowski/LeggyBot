"""Coherent device-level observation age for native policy runtimes."""

from __future__ import annotations

import numpy as np


class CorrelatedDeviceDelay:
    """Select current or previous device sample with temporally correlated lag."""

    def __init__(self, *, hold_probability: float = 0.9, seed: int = 0) -> None:
        if not 0.0 <= hold_probability <= 1.0:
            raise ValueError("hold_probability must be in [0, 1]")
        self.hold_probability = hold_probability
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self._previous: tuple[np.ndarray, ...] | None = None
        self._lag = 0

    def apply(self, *current: np.ndarray) -> tuple[np.ndarray, ...]:
        values = tuple(np.asarray(value).copy() for value in current)
        if self._previous is None:
            self._previous = values
            self._lag = int(self._rng.integers(0, 2))
            return values
        if self._rng.random() > self.hold_probability:
            self._lag = int(self._rng.integers(0, 2))
        output = self._previous if self._lag else values
        self._previous = values
        return tuple(value.copy() for value in output)


class ObservationAgeModel:
    """Independent coherent age processes for servo and IMU observations."""

    def __init__(self, *, hold_probability: float = 0.9, seed: int = 0) -> None:
        self.servo = CorrelatedDeviceDelay(
            hold_probability=hold_probability, seed=seed
        )
        self.imu = CorrelatedDeviceDelay(
            hold_probability=hold_probability, seed=seed + 1
        )

    def reset(self) -> None:
        self.servo.reset()
        self.imu.reset()

    def apply(
        self,
        angular_velocity: np.ndarray,
        projected_gravity: np.ndarray,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        delayed_gyro, delayed_gravity = self.imu.apply(
            angular_velocity, projected_gravity
        )
        delayed_position, delayed_velocity = self.servo.apply(
            joint_position, joint_velocity
        )
        return (
            delayed_gyro,
            delayed_gravity,
            delayed_position,
            delayed_velocity,
        )
