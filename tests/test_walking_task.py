from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import load_rl_cfg

from locomotion.robot_profiles import get_robot_profile
from locomotion.task import make_env_cfg, register_tasks
from locomotion.tasks import (
    LOCOMOTION_TASK_IDS,
    WALKING_TASK_ID,
    WalkingCommand,
    get_locomotion_task,
    locomotion_tasks,
)
from locomotion.tasks.omni_tracking import integrate_twist
from test_omni_tracking import advance, fixture


def test_walking_is_the_only_registered_project_task() -> None:
    register_tasks()
    assert LOCOMOTION_TASK_IDS == (WALKING_TASK_ID,)
    assert tuple(task.task_id for task in locomotion_tasks()) == (WALKING_TASK_ID,)
    assert get_locomotion_task(WALKING_TASK_ID).task_id == WALKING_TASK_ID
    with pytest.raises(ValueError, match="No implemented locomotion task"):
        get_locomotion_task("RemovedTask")


def test_walking_registers_its_command_distribution_and_symmetry() -> None:
    cfg = make_env_cfg(WALKING_TASK_ID, num_envs=2, seed=1)
    command = cfg.commands["twist"]
    assert command.reverse_probability == command.full_omni_probability == 0
    assert sum(
        (
            command.forward_straight_probability,
            command.forward_curve_probability,
            command.pure_lateral_probability,
            command.stop_probability,
        )
    ) == pytest.approx(1)
    assert command.ranges.lin_vel_x == (0.0, 0.10)
    assert command.protected_forward_range == (0.04, 0.10)
    assert not command.scale_yaw_with_forward_speed
    assert "lateral_foot_line" in cfg.rewards
    assert get_robot_profile().gait_period_for_task(WALKING_TASK_ID) == 0.33
    symmetry = load_rl_cfg(WALKING_TASK_ID).algorithm.symmetry_cfg
    assert symmetry["use_data_augmentation"]
    assert symmetry["use_mirror_loss"]
    assert symmetry["mirror_loss_coeff"] == 0.05


def test_curve_yaw_has_full_range_even_at_minimum_forward_speed() -> None:
    cfg = make_env_cfg(
        WALKING_TASK_ID, num_envs=1, seed=1
    ).commands["twist"]
    command = object.__new__(WalkingCommand)
    command.cfg = cfg
    command._env = SimpleNamespace(device="cpu")
    count = 16_384
    command.vel_command_b = torch.zeros(count, 3)
    command.vel_command_w = torch.zeros(count, 3)
    command.is_heading_env = torch.zeros(count, dtype=torch.bool)
    command.is_standing_env = torch.zeros(count, dtype=torch.bool)
    command.is_world_env = torch.zeros(count, dtype=torch.bool)
    command.is_forward_env = torch.zeros(count, dtype=torch.bool)
    command._resample_command(torch.arange(count))

    curve = command.vel_command_b[:, 2].abs() > 0.0
    slow_curve = curve & (command.vel_command_b[:, 0] <= 0.05)
    yaw = command.vel_command_b[curve, 2].abs()
    assert torch.all(yaw >= 0.10)
    assert torch.all(yaw <= 0.40)
    assert torch.any(command.vel_command_b[slow_curve, 2].abs() > 0.30)


def test_visual_sequence_covers_both_curves_without_reverse() -> None:
    from locomotion.watch_policy_sequence import (
        WALKING_COMMAND_SEQUENCE,
        command_segment_at,
    )

    assert len(WALKING_COMMAND_SEQUENCE) == 6
    assert all(segment.command[0] >= 0 for segment in WALKING_COMMAND_SEQUENCE)
    assert {segment.command[2] for segment in WALKING_COMMAND_SEQUENCE} == {
        -0.3,
        0.0,
        0.3,
    }
    assert command_segment_at(45.0)[1].command == (0.08, 0.0, -0.3)


def test_cross_track_penalizes_both_signs_without_punishing_curves_or_stop() -> None:
    env, data, commands, term = fixture(count=6, horizon=1.0)
    commands[:] = torch.tensor(
        [
            [0.08, 0, 0],
            [0.08, 0, 0],
            [0, 0.06, 0],
            [0, -0.06, 0],
            [0.08, 0, 0.3],
            [0, 0, 0],
        ]
    )
    actual = commands.clone()
    actual[:2, 1] = torch.tensor([0.02, -0.02])
    actual[2:4, 0] = 0.02
    actual[4:, 0] += 0.02
    pose = torch.zeros(6, 3)
    term(env)
    for _ in range(70):
        pose = integrate_twist(pose, actual, env.step_dt)
        advance(env, data, pose)
        term(env)
    cost = term(env, linear_scale=0.01, output="cross_track_cost")
    torch.testing.assert_close(
        cost[:4], torch.full((4,), 5**0.5 - 1), atol=1e-5, rtol=0
    )
    assert (cost[4:] == 0).all()
    term.reset(torch.tensor([0]))
    assert term(env, output="cross_track_cost")[0] == 0
