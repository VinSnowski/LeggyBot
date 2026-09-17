import torch
import pytest
from types import SimpleNamespace
from locomotion.task import make_env_cfg, register_tasks
from locomotion.tasks.skills import (
    NATURAL_OMNIDIRECTIONAL_V4_TASK_ID,
    QuasiOmnidirectionalMixtureCommand,
)
from locomotion.robot_profiles import get_robot_profile
from mjlab.tasks.registry import load_rl_cfg
from test_omni_tracking import fixture, advance
from locomotion.tasks.omni_tracking import integrate_twist


def test_v4_registers_deployable_exclusive_modes_and_symmetry():
    register_tasks()
    cfg = make_env_cfg(NATURAL_OMNIDIRECTIONAL_V4_TASK_ID, num_envs=2, seed=1)
    c = cfg.commands['twist']
    assert c.reverse_probability == c.full_omni_probability == 0
    assert sum((c.forward_straight_probability, c.forward_curve_probability,
                c.pure_lateral_probability, c.stop_probability)) == pytest.approx(1)
    assert c.ranges.lin_vel_x[0] >= 0
    assert c.ranges.lin_vel_x == (0.0, .20)
    assert c.protected_forward_range == (.04, .20)
    assert not c.scale_yaw_with_forward_speed
    assert 'lateral_foot_line' in cfg.rewards
    assert get_robot_profile().gait_period_for_task(NATURAL_OMNIDIRECTIONAL_V4_TASK_ID) == .33
    sym = load_rl_cfg(NATURAL_OMNIDIRECTIONAL_V4_TASK_ID).algorithm.symmetry_cfg
    assert sym['use_data_augmentation'] and sym['use_mirror_loss']
    assert sym['mirror_loss_coeff'] == .05


def test_v4_curve_yaw_has_full_range_even_at_minimum_forward_speed():
    cfg = make_env_cfg(
        NATURAL_OMNIDIRECTIONAL_V4_TASK_ID, num_envs=1, seed=1
    ).commands["twist"]
    command = object.__new__(QuasiOmnidirectionalMixtureCommand)
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


def test_v4_visual_sequence_covers_both_curves_without_reverse():
    from locomotion.watch_policy_sequence import V4_COMMAND_SEQUENCE, command_segment_at
    assert len(V4_COMMAND_SEQUENCE) == 6
    assert all(s.command[0] >= 0 for s in V4_COMMAND_SEQUENCE)
    assert {s.command[2] for s in V4_COMMAND_SEQUENCE} == {-.3, 0., .3}
    assert command_segment_at(45., V4_COMMAND_SEQUENCE)[1].command == (.08, 0., -.3)


def test_cross_track_penalizes_both_signs_without_punishing_curves_or_stop():
    env, data, commands, term = fixture(count=6, horizon=1.)
    commands[:] = torch.tensor([[.08,0,0],[.08,0,0],[0,.06,0],[0,-.06,0],[.08,0,.3],[0,0,0]])
    actual=commands.clone()
    actual[:2,1]=torch.tensor([.02,-.02])
    actual[2:4,0]=.02
    actual[4:,0]+=.02
    pose=torch.zeros(6,3)
    term(env)
    for _ in range(70):
        pose=integrate_twist(pose,actual,env.step_dt)
        advance(env,data,pose)
        term(env)
    cost=term(env,linear_scale=.01,output='cross_track_cost')
    torch.testing.assert_close(cost[:4],torch.full((4,),5**.5-1),atol=1e-5,rtol=0)
    assert (cost[4:]==0).all()
    term.reset(torch.tensor([0]))
    assert term(env,output='cross_track_cost')[0] == 0
