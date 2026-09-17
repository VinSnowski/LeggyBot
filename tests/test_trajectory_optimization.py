import json

import numpy as np

from locomotion.robot_profiles import LEGGYBOT_FIXED_ANKLE_PROFILE
from locomotion.trajectory_optimization import (
    PeriodicSpline,
    bilateral_gait_seed,
    compute_torque_speed_metrics,
    load_trajectory,
    save_trajectory,
    torque_speed_limit,
    trajectory_fingerprint,
    weight_transfer_gait_seed,
)


def test_periodic_spline_seam_and_vectorization():
    spline = PeriodicSpline(np.arange(48, dtype=float).reshape(8, 6) / 100.0, 0.8)
    np.testing.assert_allclose(spline.evaluate(0.0), spline.evaluate(0.8))
    assert spline.evaluate(np.array([0.0, 0.1])).shape == (2, 6)
    assert spline.evaluate(0.1).shape == (6,)


def test_bilateral_seed_is_in_profile_bounds():
    seed = bilateral_gait_seed()
    lo, hi = LEGGYBOT_FIXED_ANKLE_PROFILE.joint_lower, LEGGYBOT_FIXED_ANKLE_PROFILE.joint_upper
    assert seed.knots.shape == (8, 6)
    assert np.all(seed.knots >= np.asarray(lo))
    assert np.all(seed.knots <= np.asarray(hi))


def test_weight_transfer_seed_is_bilateral_and_in_bounds():
    seed = weight_transfer_gait_seed()
    lo, hi = LEGGYBOT_FIXED_ANKLE_PROFILE.action_bounds()
    assert np.all(seed.knots >= lo)
    assert np.all(seed.knots <= hi)
    np.testing.assert_allclose(seed.knots[:, 3], seed.knots[:, 0])
    np.testing.assert_allclose(
        seed.knots[:, 4] - LEGGYBOT_FIXED_ANKLE_PROFILE.default_joint_pos[4],
        np.roll(seed.knots[:, 1] - LEGGYBOT_FIXED_ANKLE_PROFILE.default_joint_pos[1], 4),
    )


def test_torque_speed_envelope_and_safe_metrics():
    assert torque_speed_limit(0.0) == 2.942
    metrics = compute_torque_speed_metrics(np.zeros((4, 6)), np.zeros((4, 6)))
    assert metrics.speed_ok and metrics.torque_ok
    assert compute_torque_speed_metrics(np.full((1, 1), 3.0), np.zeros((1, 1))).torque_ok is False


def test_trajectory_artifact_round_trip(tmp_path):
    trajectory = bilateral_gait_seed()
    path = save_trajectory(tmp_path / "gait.npz", trajectory)
    restored = load_trajectory(path)
    np.testing.assert_allclose(restored.knots, trajectory.knots)
    assert trajectory_fingerprint(restored) == trajectory_fingerprint(trajectory)
