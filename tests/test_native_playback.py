from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from locomotion.native_policy import load_native_policy
from locomotion.play_native import (
    _control_ids,
    _joint_addresses,
    apply_policy_action,
    benchmark,
    build_actor_observation,
    make_native_go1_model,
    reset_robot,
)
from locomotion.robot_profiles import (
    DEFAULT_ROBOT_NAME,
    GO1_PROFILE,
    LEGGYBOT_FIXED_ANKLE_PROFILE,
    LEGGYBOT_PROFILE,
    get_robot_profile,
    make_native_model,
)
from locomotion.servo import PositionBasedVelocityEstimator
from locomotion.watch_policy_sequence import (
    TOTAL_DURATION_S,
    WALKING_COMMAND_SEQUENCE,
    _local_pose_delta,
    command_segment_at,
    command_sequence,
    run_sequence,
    sequence_duration,
)


def test_native_loop_feeds_back_clipped_action() -> None:
    profile = LEGGYBOT_FIXED_ANKLE_PROFILE

    class SaturatingPolicy:
        task_id = "WalkingTask"

        def __init__(self):
            self.observations = []

        def __call__(self, observation):
            self.observations.append(observation.copy())
            return np.array([100., -200., .3, -500., 600., -.4], dtype=np.float32)

    policy = SaturatingPolicy()
    benchmark(policy, .06, profile)
    assert len(policy.observations) == 3
    np.testing.assert_array_equal(policy.observations[0][18:24], 0.)
    for observation in policy.observations[1:]:
        np.testing.assert_allclose(observation[18:24], [1., -1., .3, -1., 1., -.4])


def test_scripted_command_sequence_has_exact_half_open_boundaries() -> None:
    assert TOTAL_DURATION_S == 55.
    assert [segment.command for segment in WALKING_COMMAND_SEQUENCE] == [
        (0., 0., 0.),
        (.08, 0., 0.),
        (0., .06, 0.),
        (0., -.06, 0.),
        (.08, 0., .30),
        (.08, 0., -.30),
    ]
    assert command_segment_at(4.999)[0] == 0
    assert command_segment_at(5.)[0] == 1
    assert command_segment_at(15.)[0] == 2
    assert command_segment_at(25.)[0] == 3
    assert command_segment_at(35.)[0] == 4
    assert command_segment_at(45.)[0] == 5
    with pytest.raises(ValueError, match="complete"):
        command_segment_at(55.)


def test_local_pose_delta_uses_the_segment_start_frame() -> None:
    start = np.array([1., 2., np.pi / 2])
    end = np.array([.5, 3., np.pi])
    np.testing.assert_allclose(
        _local_pose_delta(start, end), [1., .5, np.pi / 2], atol=1e-12
    )


def test_no_lateral_sequence_retains_forward_and_both_curves() -> None:
    sequence = command_sequence(no_lateral=True)
    assert sequence == (
        WALKING_COMMAND_SEQUENCE[0],
        WALKING_COMMAND_SEQUENCE[1],
        WALKING_COMMAND_SEQUENCE[4],
        WALKING_COMMAND_SEQUENCE[5],
    )
    assert sequence_duration(sequence) == 35.
    assert command_segment_at(15., sequence)[1].command == (.08, 0., .30)
    assert command_segment_at(25., sequence)[1].command == (.08, 0., -.30)
    with pytest.raises(ValueError, match="35 seconds"):
        command_segment_at(35., sequence)


def test_scripted_native_playback_runs_policy_at_50_hz(capsys) -> None:
    profile = LEGGYBOT_FIXED_ANKLE_PROFILE

    class ZeroPolicy:
        task_id = "WalkingTask"
        observation_dim = 29
        action_dim = 6

        def __init__(self):
            self.calls = 0

        def __call__(self, observation):
            assert observation.shape == (self.observation_dim,)
            self.calls += 1
            return np.zeros(self.action_dim, dtype=np.float32)

    policy = ZeroPolicy()
    result = run_sequence(policy, profile, headless=True, stop_after_s=.06)
    assert result.completed
    assert result.control_steps == policy.calls == 3
    assert result.physics_steps == 12
    assert result.simulated_seconds == pytest.approx(.06)
    output = capsys.readouterr().out
    assert "commanded integral" in output
    assert "actual integrated body motion" in output


def test_rsl_checkpoint_loader_reconstructs_normalized_actor(tmp_path: Path) -> None:
    state = {
        "obs_normalizer._mean": torch.tensor([[1.0, 2.0]]),
        "obs_normalizer._var": torch.ones(1, 2),
        "obs_normalizer._std": torch.tensor([[2.0, 4.0]]),
        "obs_normalizer.count": torch.tensor(100),
        "distribution.std_param": torch.ones(1),
        "mlp.0.weight": torch.tensor([[2.0, -1.0]]),
        "mlp.0.bias": torch.tensor([0.5]),
    }
    checkpoint = tmp_path / "rsl.pt"
    torch.save(
        {
            "actor_state_dict": state,
            "critic_state_dict": {},
            "optimizer_state_dict": {},
            "iter": 1,
            "task_id": LEGGYBOT_FIXED_ANKLE_PROFILE.task_id,
        },
        checkpoint,
    )

    policy, backend = load_native_policy(checkpoint)
    observation = np.array([5.0, 10.0], dtype=np.float32)
    normalized = (observation - np.array([1.0, 2.0])) / np.array([2.01, 4.01])
    expected = 2.0 * normalized[0] - normalized[1] + 0.5

    assert backend == "rsl-rl"
    assert policy.task_id == LEGGYBOT_FIXED_ANKLE_PROFILE.task_id
    np.testing.assert_allclose(policy(observation)[0], expected, rtol=1e-6)


def test_native_go1_policy_contract() -> None:
    model = make_native_go1_model()
    data = mujoco.MjData(model)
    qpos_addresses, dof_addresses = _joint_addresses(model, GO1_PROFILE)
    control_ids = _control_ids(model, GO1_PROFILE)
    reset_robot(model, data, qpos_addresses, GO1_PROFILE)

    previous_action = np.zeros(12, dtype=np.float32)
    command = np.array([0.5, -0.1, 0.2], dtype=np.float32)
    observation = build_actor_observation(
        model,
        data,
        qpos_addresses,
        dof_addresses,
        previous_action, command, GO1_PROFILE
    )

    assert model.opt.timestep * GO1_PROFILE.control_decimation == GO1_PROFILE.control_dt
    assert observation.shape == (GO1_PROFILE.observation_dim,)
    np.testing.assert_allclose(observation[6:18], 0.0, atol=1e-6)
    np.testing.assert_allclose(observation[-3:], command)

    action = np.ones(12, dtype=np.float32)
    apply_policy_action(model, data, control_ids, action, GO1_PROFILE)
    np.testing.assert_allclose(
        data.ctrl[control_ids],
        GO1_PROFILE.array("default_joint_pos") + GO1_PROFILE.array("action_scale"),
        rtol=1e-7,
    )


def test_new_leggybot_is_the_default_policy_contract() -> None:
    profile = get_robot_profile()
    assert DEFAULT_ROBOT_NAME == LEGGYBOT_FIXED_ANKLE_PROFILE.name
    assert profile is LEGGYBOT_FIXED_ANKLE_PROFILE
    model = make_native_model(profile)
    data = mujoco.MjData(model)
    qpos_addresses, dof_addresses = _joint_addresses(model)
    control_ids = _control_ids(model)
    reset_robot(model, data, qpos_addresses)

    observation = build_actor_observation(
        model,
        data,
        qpos_addresses,
        dof_addresses,
        np.zeros(profile.action_dim, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        velocity_estimator=PositionBasedVelocityEstimator(profile.control_dt),
    )
    assert observation.shape == (27,)

    apply_policy_action(
        model,
        data,
        control_ids,
        np.ones(profile.action_dim, dtype=np.float32),
    )
    np.testing.assert_allclose(
        data.ctrl[control_ids],
        profile.action_target_from_normalized(np.ones(profile.action_dim)),
        rtol=1e-7,
    )


def test_native_ankle8_profile_uses_its_offset_pose() -> None:
    model = make_native_model(LEGGYBOT_FIXED_ANKLE_PROFILE)
    data = mujoco.MjData(model)
    qpos_addresses, _ = _joint_addresses(model, LEGGYBOT_FIXED_ANKLE_PROFILE)
    reset_robot(model, data, qpos_addresses, LEGGYBOT_FIXED_ANKLE_PROFILE)

    np.testing.assert_allclose(
        data.qpos[qpos_addresses], LEGGYBOT_FIXED_ANKLE_PROFILE.default_joint_pos
    )
    assert LEGGYBOT_FIXED_ANKLE_PROFILE.observation_dim == LEGGYBOT_PROFILE.observation_dim
    assert LEGGYBOT_FIXED_ANKLE_PROFILE.action_dim == LEGGYBOT_PROFILE.action_dim


def test_native_gait_observation_appends_the_training_clock() -> None:
    model = make_native_model(LEGGYBOT_FIXED_ANKLE_PROFILE)
    data = mujoco.MjData(model)
    qpos_addresses, dof_addresses = _joint_addresses(model, LEGGYBOT_FIXED_ANKLE_PROFILE)
    reset_robot(model, data, qpos_addresses, LEGGYBOT_FIXED_ANKLE_PROFILE)
    data.time = 0.2

    observation = build_actor_observation(
        model,
        data,
        qpos_addresses,
        dof_addresses,
        np.zeros(6, dtype=np.float32),
        np.array([0.2, 0.0, 0.0], dtype=np.float32),
        LEGGYBOT_FIXED_ANKLE_PROFILE,
        gait_period_s=0.8,
        velocity_estimator=PositionBasedVelocityEstimator(
            LEGGYBOT_FIXED_ANKLE_PROFILE.control_dt
        ),
    )

    assert observation.shape == (29,)
    np.testing.assert_allclose(observation[-2:], [1.0, 0.0], atol=1e-6)
