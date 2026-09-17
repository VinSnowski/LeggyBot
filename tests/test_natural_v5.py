from types import SimpleNamespace

import pytest
import torch

from locomotion.task import make_env_cfg, register_tasks
from locomotion.tasks.gait import ForwardSpeedGateCurriculum
from locomotion.tasks.skills import (
    NATURAL_OMNIDIRECTIONAL_V4_TASK_ID,
    NATURAL_OMNIDIRECTIONAL_V5_TASK_ID,
)
from mjlab.tasks.registry import load_rl_cfg


class _CommandManager:
    def __init__(self):
        cfg = SimpleNamespace(
            ranges=SimpleNamespace(lin_vel_x=(0.0, 0.15)),
            protected_forward_range=(0.04, 0.15),
        )
        self.term = SimpleNamespace(cfg=cfg)

    def get_term(self, name):
        assert name == "twist"
        return self.term


def _fake_env(success: float):
    return SimpleNamespace(
        num_envs=4,
        device="cpu",
        common_step_counter=0,
        command_manager=_CommandManager(),
        metrics_manager=SimpleNamespace(
            _episode_sums={"tracking_success": torch.full((4,), success * 1000)},
            _step_count=torch.full((4,), 1000, dtype=torch.long),
        ),
    )


def test_v5_changes_only_speed_curriculum_and_starts_at_point_fifteen():
    register_tasks()
    v4 = make_env_cfg(NATURAL_OMNIDIRECTIONAL_V4_TASK_ID, num_envs=2, seed=1)
    v5 = make_env_cfg(NATURAL_OMNIDIRECTIONAL_V5_TASK_ID, num_envs=2, seed=1)
    command = v5.commands["twist"]
    assert command.ranges.lin_vel_x == (0.0, 0.15)
    assert command.protected_forward_range == (0.04, 0.15)
    assert command.forward_straight_probability == v4.commands["twist"].forward_straight_probability
    assert command.forward_curve_probability == v4.commands["twist"].forward_curve_probability
    assert set(v5.rewards) == set(v4.rewards)
    assert {name: term.weight for name, term in v5.rewards.items()} == {
        name: term.weight for name, term in v4.rewards.items()
    }
    event = v5.events["domain_randomization"]
    assert event.params == {"symmetric_steps": 0, "independent_ramp_steps": 0}
    curriculum = v5.curriculum["forward_speed_gate"].func
    assert isinstance(curriculum, ForwardSpeedGateCurriculum)
    assert curriculum.speed_levels[-1] == 0.50
    assert curriculum.minimum_gate_iterations == 100
    assert curriculum.success_threshold == 0.70
    v4_rl = load_rl_cfg(NATURAL_OMNIDIRECTIONAL_V4_TASK_ID)
    v5_rl = load_rl_cfg(NATURAL_OMNIDIRECTIONAL_V5_TASK_ID)
    assert v5_rl.actor == v4_rl.actor
    assert v5_rl.critic == v4_rl.critic
    assert v5_rl.algorithm == v4_rl.algorithm
    assert v5_rl.num_steps_per_env == v4_rl.num_steps_per_env


def test_forward_speed_gate_requires_success_and_one_hundred_iterations():
    curriculum = ForwardSpeedGateCurriculum(success_window_iterations=50)
    env = _fake_env(0.90)
    curriculum(env)
    env.common_step_counter = 99 * 24
    state = curriculum(env)
    assert state["forward_speed_max_mps"] == pytest.approx(0.15)
    env.common_step_counter = 100 * 24
    state = curriculum(env)
    assert state["forward_speed_max_mps"] == pytest.approx(0.20)
    assert env.command_manager.term.cfg.protected_forward_range == (0.04, 0.20)

    env.metrics_manager._episode_sums["tracking_success"].fill_(0.69 * 1000)
    env.common_step_counter = 250 * 24
    state = curriculum(env)
    assert state["forward_speed_max_mps"] == pytest.approx(0.20)


def test_forward_speed_gate_state_round_trip():
    curriculum = ForwardSpeedGateCurriculum()
    env = _fake_env(0.90)
    env.common_step_counter = 100 * 24
    curriculum(env)
    restored = ForwardSpeedGateCurriculum()
    restored.load_state_dict(curriculum.state_dict())
    assert restored.level == 1
    assert restored.gate_start_iteration == 100
