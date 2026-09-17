import math

import numpy as np

from locomotion.imu import (
    ICM42688_CONFIG,
    STANDARD_GRAVITY,
    Icm42688OrientationEstimator,
)


def test_icm42688_selected_ranges_and_resolution() -> None:
    cfg = ICM42688_CONFIG
    assert cfg.sample_hz == 50.0
    assert cfg.gyro_full_scale_rad_s == math.radians(500.0)
    assert cfg.accel_full_scale_m_s2 == 16.0 * STANDARD_GRAVITY
    assert cfg.gyro_resolution_rad_s == 2.0 * math.radians(500.0) / 65_536
    assert cfg.accel_resolution_m_s2 == 32.0 * STANDARD_GRAVITY / 65_536


def test_gravity_estimator_tracks_roll_without_a_quaternion() -> None:
    estimator = Icm42688OrientationEstimator()
    roll_rate = math.radians(30.0)
    target = np.array((0.0, 0.0, -1.0))
    for index in range(51):
        roll = roll_rate * index * 0.02
        target = np.array((0.0, -math.sin(roll), -math.cos(roll)))
        estimator.update(
            np.array((roll_rate, 0.0, 0.0)),
            -STANDARD_GRAVITY * target,
            index * 20_000_000,
        )
    angle_error = math.degrees(math.acos(np.clip(
        np.dot(target, estimator.projected_gravity), -1.0, 1.0
    )))
    assert angle_error < 1.5


def test_estimator_downsamples_to_policy_rate() -> None:
    estimator = Icm42688OrientationEstimator()
    upright_accel = np.array((0.0, 0.0, STANDARD_GRAVITY))
    assert estimator.update(np.zeros(3), upright_accel, 0)
    assert not estimator.update(np.ones(3), upright_accel, 5_000_000)
    assert estimator.update(np.array((1.0, 0.0, 0.0)), upright_accel, 20_000_000)
    assert estimator.angular_velocity[0] > 0.5
    assert estimator.update(np.array((1.0, 0.0, 0.0)), upright_accel, 40_000_000)
    assert estimator.angular_velocity[0] > 0.5
