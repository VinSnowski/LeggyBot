"""Servo register conversions shared by training and deployment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence

import numpy as np
import torch


# The STS3215 position register/feedback uses 4096 counts per revolution.
STS3215_POSITION_COUNTS_PER_REVOLUTION = 4096
STS3215_POSITION_RESOLUTION_RAD = (
    2.0 * math.pi / STS3215_POSITION_COUNTS_PER_REVOLUTION
)
# Weight of the newest finite-difference sample in the causal velocity filter.
# At 50 Hz this is a modest low-pass filter while preserving gait dynamics.
STS3215_VELOCITY_FILTER_ALPHA = 0.5

# Feetech's position-mode profile registers use encoder counts. LeRobot sets
# both acceleration registers to their largest finite value and removes the
# default response delay. We additionally pin the volatile goal velocity/time
# registers instead of relying on their power-on state; velocity 0 is unlimited
# in position mode and time 0 selects velocity-based motion.
STS3215_MAX_SPEED_COUNTS_PER_S = 3400
STS3215_ACCELERATION_COUNTS_PER_S2_PER_UNIT = 100
STS3215_MAX_ACCELERATION_REGISTER = 254
STS3215_MAX_PROFILE_SPEED_RAD_S = (
    STS3215_MAX_SPEED_COUNTS_PER_S * STS3215_POSITION_RESOLUTION_RAD
)
STS3215_MAX_PROFILE_ACCELERATION_RAD_S2 = (
    STS3215_MAX_ACCELERATION_REGISTER
    * STS3215_ACCELERATION_COUNTS_PER_S2_PER_UNIT
    * STS3215_POSITION_RESOLUTION_RAD
)


@dataclass(frozen=True, slots=True)
class Sts3215FirmwareConfig:
    """Explicit STS3215 firmware contract used by simulation and hardware."""

    operating_mode: int = 0
    position_p_coefficient: int = 16
    position_i_coefficient: int = 0
    position_d_coefficient: int = 32
    return_delay_time: int = 0
    maximum_acceleration: int = STS3215_MAX_ACCELERATION_REGISTER
    acceleration: int = STS3215_MAX_ACCELERATION_REGISTER
    goal_velocity: int = 0
    goal_time: int = 0
    clear_phase_angle_overflow_bit: bool = True

    def __post_init__(self) -> None:
        if self.operating_mode != 0:
            raise ValueError("LeggyBot STS3215 servos must use position mode 0")
        for name, value in (
            ("P coefficient", self.position_p_coefficient),
            ("I coefficient", self.position_i_coefficient),
            ("D coefficient", self.position_d_coefficient),
        ):
            if not 0 <= value <= 254:
                raise ValueError(f"{name} must be in [0, 254]")
        if not 0 <= self.return_delay_time <= 254:
            raise ValueError("return delay register must be in [0, 254]")
        if not 0 <= self.maximum_acceleration <= 254:
            raise ValueError("maximum acceleration register must be in [0, 254]")
        if not 0 <= self.acceleration <= self.maximum_acceleration:
            raise ValueError("acceleration must be within the configured maximum")
        if not 0 <= self.goal_velocity <= 32767:
            raise ValueError("goal velocity register must be in [0, 32767]")
        if not 0 <= self.goal_time <= 32767:
            raise ValueError("goal time register must be in [0, 32767]")


LEGGYBOT_STS3215_FIRMWARE = Sts3215FirmwareConfig()


class LeRobotFeetechBus(Protocol):
    """Small structural subset of LeRobot's ``FeetechMotorsBus`` API."""

    def write(self, data_name: str, motor: str, value: int) -> None: ...

    def read(self, data_name: str, motor: str, *, normalize: bool = False) -> int: ...


def configure_sts3215_like_lerobot(
    bus: LeRobotFeetechBus,
    motors: Sequence[str],
    config: Sts3215FirmwareConfig = LEGGYBOT_STS3215_FIRMWARE,
    *,
    verify: bool = True,
) -> None:
    """Configure each servo for low-latency, maximum-response position control.

    The function intentionally uses LeRobot's public register-name API so a
    future hardware daemon can depend on LeRobot without coupling policy code
    to its serial implementation. It extends LeRobot's setup by making volatile
    goal velocity/time explicit instead of relying on power-on SRAM values.
    """

    expected = {
        "Operating_Mode": config.operating_mode,
        "P_Coefficient": config.position_p_coefficient,
        "I_Coefficient": config.position_i_coefficient,
        "D_Coefficient": config.position_d_coefficient,
        "Return_Delay_Time": config.return_delay_time,
        "Maximum_Acceleration": config.maximum_acceleration,
        "Acceleration": config.acceleration,
        "Goal_Velocity": config.goal_velocity,
        "Goal_Time": config.goal_time,
    }
    for motor in motors:
        for register, value in expected.items():
            bus.write(register, motor, value)
        if config.clear_phase_angle_overflow_bit:
            phase = int(bus.read("Phase", motor, normalize=False))
            if phase & 0x10:
                bus.write("Phase", motor, phase & ~0x10)
                expected_phase = phase & ~0x10
            else:
                expected_phase = phase
        else:
            expected_phase = None

        if not verify:
            continue
        mismatches: list[str] = []
        for register, value in expected.items():
            actual = int(bus.read(register, motor, normalize=False))
            if actual != value:
                mismatches.append(f"{register}={actual}, expected {value}")
        if expected_phase is not None:
            actual_phase = int(bus.read("Phase", motor, normalize=False))
            if actual_phase != expected_phase:
                mismatches.append(
                    f"Phase={actual_phase}, expected {expected_phase}"
                )
        if mismatches:
            raise RuntimeError(
                f"STS3215 configuration verification failed for {motor}: "
                + "; ".join(mismatches)
            )


class PositionSetpointProfiler:
    """Acceleration- and velocity-limited STS3215 firmware setpoint model."""

    def __init__(
        self,
        initial_position: np.ndarray,
        max_velocity_rad_s: float | np.ndarray = STS3215_MAX_PROFILE_SPEED_RAD_S,
        max_acceleration_rad_s2: float | np.ndarray = STS3215_MAX_PROFILE_ACCELERATION_RAD_S2,
    ) -> None:
        initial = np.asarray(initial_position, dtype=np.float64)
        self.position = initial.copy()
        self.goal = initial.copy()
        self.velocity = np.zeros_like(initial)
        self.max_velocity = np.broadcast_to(
            np.asarray(max_velocity_rad_s, dtype=np.float64), initial.shape
        ).copy()
        self.max_acceleration = np.broadcast_to(
            np.asarray(max_acceleration_rad_s2, dtype=np.float64), initial.shape
        ).copy()
        if np.any(self.max_velocity <= 0.0):
            raise ValueError("profile velocity limits must be positive")
        if np.any(self.max_acceleration <= 0.0):
            raise ValueError("profile acceleration limits must be positive")

    def reset(self, position: np.ndarray) -> None:
        value = np.asarray(position, dtype=np.float64)
        if value.shape != self.position.shape:
            raise ValueError("profile reset position has an invalid shape")
        self.position[:] = value
        self.goal[:] = value
        self.velocity.fill(0.0)

    def set_goal(self, goal: np.ndarray) -> None:
        value = np.asarray(goal, dtype=np.float64)
        if value.shape != self.goal.shape or not np.isfinite(value).all():
            raise ValueError("profile goal has an invalid value or shape")
        self.goal[:] = value

    def update(self, dt: float) -> np.ndarray:
        if dt <= 0.0:
            raise ValueError("profile timestep must be positive")
        error = self.goal - self.position
        acceleration_step = self.max_acceleration * dt
        # Discrete safe speed: after advancing by v*dt, the remaining distance
        # still admits a full acceleration-limited stop. This avoids the
        # overshoot caused by using the continuous sqrt(2*a*d) bound directly.
        stopping_speed = np.sqrt(
            acceleration_step**2
            + 2.0 * self.max_acceleration * np.abs(error)
        ) - acceleration_step
        desired_velocity = np.sign(error) * np.minimum(
            self.max_velocity, stopping_speed
        )
        velocity_delta = np.clip(
            desired_velocity - self.velocity,
            -acceleration_step,
            acceleration_step,
        )
        next_velocity = self.velocity + velocity_delta
        step = next_velocity * dt
        reached = (
            (np.abs(step) >= np.abs(error) - 1.0e-15)
            & (np.sign(step) == np.sign(error))
            & (np.abs(next_velocity) <= acceleration_step + 1.0e-12)
        )
        self.position[:] = np.where(reached, self.goal, self.position + step)
        self.velocity[:] = np.where(reached, 0.0, next_velocity)
        return self.position.copy()


def quantize_position_numpy(
    position: np.ndarray, resolution_rad: float = STS3215_POSITION_RESOLUTION_RAD
) -> np.ndarray:
    """Round joint positions to the nearest representable servo count."""
    if resolution_rad <= 0.0:
        raise ValueError("position resolution must be positive")
    value = np.asarray(position)
    return np.rint(value / resolution_rad) * resolution_rad


def quantize_position_torch(
    position: torch.Tensor, resolution_rad: float = STS3215_POSITION_RESOLUTION_RAD
) -> torch.Tensor:
    """Round joint positions to the nearest representable servo count."""
    if resolution_rad <= 0.0:
        raise ValueError("position resolution must be positive")
    return torch.round(position / resolution_rad) * resolution_rad


class PositionBasedVelocityEstimator:
    """Causal joint velocity estimate from policy-rate position samples.

    Inputs must already have the same position quantization/calibration used by
    the policy observation.  The estimator deliberately uses the fixed policy
    period rather than packet timestamps: simulation and hardware consequently
    behave identically when the policy samples the latest available feedback.
    """

    def __init__(
        self,
        sample_period_s: float,
        filter_alpha: float = STS3215_VELOCITY_FILTER_ALPHA,
    ) -> None:
        if sample_period_s <= 0.0:
            raise ValueError("sample period must be positive")
        if not 0.0 < filter_alpha <= 1.0:
            raise ValueError("filter alpha must be in (0, 1]")
        self.sample_period_s = float(sample_period_s)
        self.filter_alpha = float(filter_alpha)
        self._previous_position: np.ndarray | None = None
        self._filtered_velocity: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_position = None
        self._filtered_velocity = None

    def update(self, observed_position: np.ndarray) -> np.ndarray:
        position = np.asarray(observed_position, dtype=np.float32)
        if self._previous_position is None:
            self._previous_position = position.copy()
            self._filtered_velocity = np.zeros_like(position)
            return self._filtered_velocity.copy()

        assert self._filtered_velocity is not None
        raw_velocity = (position - self._previous_position) / self.sample_period_s
        self._filtered_velocity += self.filter_alpha * (
            raw_velocity - self._filtered_velocity
        )
        self._previous_position[...] = position
        return self._filtered_velocity.copy()
