import math

import numpy as np
import pytest
import torch

from locomotion.robot_profiles import GO1_PROFILE, LEGGYBOT_FIXED_ANKLE_PROFILE
from locomotion.servo import (
    LEGGYBOT_STS3215_FIRMWARE,
    PositionBasedVelocityEstimator,
    PositionSetpointProfiler,
    STS3215_MAX_PROFILE_ACCELERATION_RAD_S2,
    STS3215_MAX_PROFILE_SPEED_RAD_S,
    STS3215_POSITION_COUNTS_PER_REVOLUTION,
    STS3215_POSITION_RESOLUTION_RAD,
    configure_sts3215_like_lerobot,
    quantize_position_numpy,
    quantize_position_torch,
)


def test_sts3215_resolution_is_4096_counts_per_revolution() -> None:
    assert STS3215_POSITION_COUNTS_PER_REVOLUTION == 4096
    assert STS3215_POSITION_RESOLUTION_RAD == 2.0 * math.pi / 4096
    assert STS3215_MAX_PROFILE_SPEED_RAD_S == pytest.approx(
        3400 * 2.0 * math.pi / 4096
    )
    assert STS3215_MAX_PROFILE_ACCELERATION_RAD_S2 == pytest.approx(
        25400 * 2.0 * math.pi / 4096
    )


def test_numpy_and_torch_position_quantizers_round_to_servo_counts() -> None:
    step = STS3215_POSITION_RESOLUTION_RAD
    values = np.array([-1.6 * step, -0.49 * step, 0.51 * step, 2.4 * step])
    expected = np.array([-2.0 * step, 0.0, step, 2.0 * step])
    np.testing.assert_allclose(quantize_position_numpy(values), expected)
    torch.testing.assert_close(
        quantize_position_torch(torch.tensor(values)), torch.tensor(expected)
    )


def test_leggybot_quantizes_feedback_and_position_targets() -> None:
    profile = LEGGYBOT_FIXED_ANKLE_PROFILE
    step = STS3215_POSITION_RESOLUTION_RAD
    feedback = profile.observed_joint_position(
        profile.array("default_joint_pos") + 0.37 * step
    )
    np.testing.assert_allclose(feedback / step, np.rint(feedback / step))

    target = profile.action_target_from_normalized(
        np.array([0.123, -0.456, 0.789, -0.234, 0.567, -0.891])
    )
    np.testing.assert_allclose(target / step, np.rint(target / step))
    assert np.all(target >= profile.array("joint_lower"))
    assert np.all(target <= profile.array("joint_upper"))


def test_profiles_without_discrete_servos_remain_continuous() -> None:
    values = np.linspace(-0.1234, 0.5678, GO1_PROFILE.action_dim)
    np.testing.assert_array_equal(GO1_PROFILE.observed_joint_position(values), values)


def test_position_based_velocity_uses_policy_period_and_causal_filter() -> None:
    step = STS3215_POSITION_RESOLUTION_RAD
    estimator = PositionBasedVelocityEstimator(0.02, filter_alpha=0.5)

    np.testing.assert_array_equal(estimator.update(np.array([3.0 * step])), [0.0])
    first = estimator.update(np.array([5.0 * step]))
    np.testing.assert_allclose(first, [0.5 * 2.0 * step / 0.02])
    second = estimator.update(np.array([5.0 * step]))
    np.testing.assert_allclose(second, 0.5 * first)

    estimator.reset()
    np.testing.assert_array_equal(estimator.update(np.array([100.0 * step])), [0.0])


def test_position_setpoint_profiler_limits_speed_acceleration_and_overshoot() -> None:
    profiler = PositionSetpointProfiler(
        np.array([0.0]), max_velocity_rad_s=1.0, max_acceleration_rad_s2=2.0
    )
    profiler.set_goal(np.array([0.2]))
    first = profiler.update(0.01)
    np.testing.assert_allclose(first, [0.0002])
    np.testing.assert_allclose(profiler.velocity, [0.02])
    previous_velocity = profiler.velocity.copy()
    for _ in range(200):
        profiler.update(0.01)
        assert np.abs(profiler.velocity - previous_velocity).max() <= 0.0200001
        assert np.abs(profiler.velocity).max() <= 1.000001
        assert profiler.position[0] <= 0.200001
        previous_velocity = profiler.velocity.copy()
    np.testing.assert_allclose(profiler.position, [0.2], atol=1e-8)
    np.testing.assert_allclose(profiler.velocity, [0.0], atol=1e-8)


def test_lerobot_sts3215_configuration_is_explicit_and_verified() -> None:
    class FakeBus:
        def __init__(self):
            self.values = {("Phase", name): 0x10 for name in ("right", "left")}
            self.writes = []

        def write(self, register, motor, value):
            self.values[(register, motor)] = value
            self.writes.append((register, motor, value))

        def read(self, register, motor, *, normalize=False):
            assert not normalize
            return self.values[(register, motor)]

    bus = FakeBus()
    configure_sts3215_like_lerobot(bus, ("right", "left"))
    for motor in ("right", "left"):
        assert bus.values[("Operating_Mode", motor)] == 0
        assert bus.values[("P_Coefficient", motor)] == 16
        assert bus.values[("I_Coefficient", motor)] == 0
        assert bus.values[("D_Coefficient", motor)] == 32
        assert bus.values[("Return_Delay_Time", motor)] == 0
        assert bus.values[("Maximum_Acceleration", motor)] == 254
        assert bus.values[("Acceleration", motor)] == 254
        assert bus.values[("Goal_Velocity", motor)] == 0
        assert bus.values[("Goal_Time", motor)] == 0
        assert bus.values[("Phase", motor)] == 0
    assert LEGGYBOT_STS3215_FIRMWARE.acceleration == 254
