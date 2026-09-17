from pathlib import Path

import pytest
import torch

from locomotion.custom_robot import CustomRobotContract
from locomotion.play_keyboard import (
    KeyboardVelocityCommand,
    detect_checkpoint_backend,
    optimize_env_cfg_for_playback,
)
from locomotion.ppo import ActorCritic, PPOConfig, RunningMeanStd, compute_gae


def test_gae_bootstraps_truncation_but_not_termination() -> None:
    rewards = torch.tensor([[1.0, 1.0]])
    values = torch.tensor([[2.0, 2.0]])
    next_values = torch.tensor([[3.0, 3.0]])
    terminated = torch.tensor([[False, True]])
    truncated = torch.tensor([[True, False]])

    advantages = compute_gae(
        rewards,
        values,
        next_values,
        terminated,
        truncated,
        gamma=0.9,
        gae_lambda=0.95,
    )

    assert advantages[0, 0].item() == pytest.approx(1.0 + 0.9 * 3.0 - 2.0)
    assert advantages[0, 1].item() == pytest.approx(1.0 - 2.0)


def test_actor_critic_supports_privileged_critic_observations() -> None:
    model = ActorCritic(10, 14, 4, PPOConfig(hidden_dims=(32, 16)))
    actor_obs = torch.randn(8, 10)
    critic_obs = torch.randn(8, 14)

    actions, log_probs, values = model.act(actor_obs, critic_obs)
    evaluated_log_probs, entropy, evaluated_values = model.evaluate_actions(
        actor_obs, critic_obs, actions
    )

    assert actions.shape == (8, 4)
    assert log_probs.shape == values.shape == (8,)
    assert torch.allclose(log_probs, evaluated_log_probs)
    assert torch.allclose(values, evaluated_values)
    assert torch.all(entropy > 0)


def test_running_statistics_are_checkpoint_state() -> None:
    normalizer = RunningMeanStd((2,))
    normalizer.update(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    restored = RunningMeanStd((2,))
    restored.load_state_dict(normalizer.state_dict())

    assert torch.allclose(restored.mean, normalizer.mean)
    assert torch.allclose(restored.var, normalizer.var)
    assert torch.allclose(restored.count, normalizer.count)


def test_custom_robot_contract_rejects_incomplete_hardware_description(
    tmp_path: Path,
) -> None:
    xml = tmp_path / "robot.xml"
    xml.write_text("<mujoco/>")
    contract = CustomRobotContract(
        mjcf_path=xml,
        root_body="base",
        foot_sites=("left", "right"),
        foot_geoms=("left_collision", "right_collision"),
        default_joint_positions={},
        action_scale={".*": 0.2},
    )
    with pytest.raises(ValueError, match="default joint pose"):
        contract.validate_static()


def test_keyboard_commands_are_latched_and_bounded() -> None:
    from mjlab.viewer.native.keys import KEY_E, KEY_W, KEY_X

    command = KeyboardVelocityCommand(linear_step=1.0, yaw_step=0.25, max_forward=1.5)
    command.key_callback(KEY_W)
    command.key_callback(KEY_W)
    command.key_callback(KEY_E)
    assert command.target == (1.5, 0.0, -0.25)
    command.key_callback(KEY_X)
    assert command.target == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ({"format": "mjlab-custom-ppo-v1"}, "custom"),
        (
            {
                "actor_state_dict": {},
                "critic_state_dict": {},
                "optimizer_state_dict": {},
            },
            "rsl-rl",
        ),
    ],
)
def test_checkpoint_backend_is_detected(
    tmp_path: Path, contents: dict, expected: str
) -> None:
    checkpoint = tmp_path / "policy.pt"
    torch.save(contents, checkpoint)
    assert detect_checkpoint_backend(checkpoint) == expected


def test_lightweight_playback_preserves_actor_abi() -> None:
    from locomotion.task import make_env_cfg

    from locomotion.tasks import STAND_TASK_ID

    cfg = make_env_cfg(STAND_TASK_ID, num_envs=1, seed=123, play=True)
    actor_terms = tuple(cfg.observations["actor"].terms)
    optimize_env_cfg_for_playback(cfg)

    assert tuple(cfg.observations["actor"].terms) == actor_terms
    assert tuple(cfg.observations["critic"].terms) == actor_terms
    assert cfg.scene.sensors == ()
    assert cfg.rewards == {}
    assert cfg.metrics == {}
    assert all(term.mode == "reset" for term in cfg.events.values())
