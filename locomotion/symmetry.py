"""Left/right symmetry transforms for LeggyBot PPO data augmentation.

The transform represents reflection in the robot's sagittal X-Z plane.  A
walking state is also shifted by half of its gait cycle: the reflected state
must exchange the scheduled right and left support legs, rather than merely
reflect a static pose.

This module is deliberately explicit about the actor and critic ABIs.  A
changed observation layout must update this transform and its accompanying
tests instead of silently training an incorrect augmented policy.
"""

from __future__ import annotations

import torch


_ACTOR_DIM = 29
_CRITIC_DIM = 44
_JOINT_MIRROR = (3, 4, 5, 0, 1, 2)


def _swap_and_sign(values: torch.Tensor, signs: tuple[float, ...]) -> torch.Tensor:
    """Swap right/left joint groups and apply the joint-coordinate reflection."""
    return values[..., _JOINT_MIRROR] * values.new_tensor(signs)


def _mirror_actor(flat: torch.Tensor) -> torch.Tensor:
    """Mirror the fixed 29-value deployment actor observation."""
    if flat.shape[-1] != _ACTOR_DIM:
        raise ValueError(
            f"LeggyBot symmetry expects {_ACTOR_DIM} actor values, got {flat.shape[-1]}"
        )
    mirrored = flat.clone()
    # Angular velocity is an axial vector; gravity is a polar vector.
    mirrored[..., 0:3] = flat[..., 0:3] * flat.new_tensor((-1.0, 1.0, -1.0))
    mirrored[..., 3:6] = flat[..., 3:6] * flat.new_tensor((1.0, -1.0, 1.0))
    # [R roll, R hip pitch, R knee, L roll, L hip pitch, L knee].
    joint_signs = (-1.0, 1.0, 1.0, -1.0, 1.0, 1.0)
    mirrored[..., 6:12] = _swap_and_sign(flat[..., 6:12], joint_signs)
    mirrored[..., 12:18] = _swap_and_sign(flat[..., 12:18], joint_signs)
    mirrored[..., 18:24] = _swap_and_sign(flat[..., 18:24], joint_signs)
    # Command is [forward, lateral, yaw].
    mirrored[..., 24:27] = flat[..., 24:27] * flat.new_tensor((1.0, -1.0, -1.0))
    # Sagittal reflection exchanges legs, hence phase advances by half a cycle.
    mirrored[..., 27:29] = -flat[..., 27:29]
    return mirrored


def _mirror_critic(flat: torch.Tensor) -> torch.Tensor:
    """Mirror the 44-value privileged critic observation used by this task."""
    if flat.shape[-1] != _CRITIC_DIM:
        raise ValueError(
            f"LeggyBot symmetry expects {_CRITIC_DIM} critic values, got {flat.shape[-1]}"
        )
    mirrored = flat.clone()
    # Critic prefix: base linear velocity, angular velocity, projected gravity.
    mirrored[..., 0:3] = flat[..., 0:3] * flat.new_tensor((1.0, -1.0, 1.0))
    mirrored[..., 3:6] = flat[..., 3:6] * flat.new_tensor((-1.0, 1.0, -1.0))
    mirrored[..., 6:9] = flat[..., 6:9] * flat.new_tensor((1.0, -1.0, 1.0))
    joint_signs = (-1.0, 1.0, 1.0, -1.0, 1.0, 1.0)
    mirrored[..., 9:15] = _swap_and_sign(flat[..., 9:15], joint_signs)
    mirrored[..., 15:21] = _swap_and_sign(flat[..., 15:21], joint_signs)
    mirrored[..., 21:27] = _swap_and_sign(flat[..., 21:27], joint_signs)
    mirrored[..., 27:30] = flat[..., 27:30] * flat.new_tensor((1.0, -1.0, -1.0))
    # Per-foot scalar diagnostics preserve their value and exchange feet.
    mirrored[..., 30:32] = flat[..., (31, 30)]  # height
    mirrored[..., 32:34] = flat[..., (33, 32)]  # air time
    mirrored[..., 34:36] = flat[..., (35, 34)]  # contact flag
    # Per-foot contact force is a polar vector: swap feet, negate Y.
    forces = flat[..., 36:42].reshape(*flat.shape[:-1], 2, 3)
    mirrored_forces = forces[..., (1, 0), :] * flat.new_tensor((1.0, -1.0, 1.0))
    mirrored[..., 36:42] = mirrored_forces.reshape(*flat.shape[:-1], 6)
    mirrored[..., 42:44] = -flat[..., 42:44]
    return mirrored


def mirror_leggybot_actions(actions: torch.Tensor) -> torch.Tensor:
    """Mirror normalized six-joint LeggyBot actions."""
    if actions.shape[-1] != 6:
        raise ValueError(f"LeggyBot symmetry expects 6 actions, got {actions.shape[-1]}")
    return _swap_and_sign(actions, (-1.0, 1.0, 1.0, -1.0, 1.0, 1.0))


def leggybot_left_right_augmentation(env, obs=None, actions=None):
    """Append reflected LeggyBot samples for RSL-RL's PPO augmentation hook.

    ``env`` is part of RSL-RL's extension contract.  The ABI is fixed by the
    task configuration rather than inferred from the live environment so that a
    change in observations fails loudly at training startup.
    """
    del env
    augmented_obs = None
    if obs is not None:
        mirrored = obs.clone()
        try:
            mirrored.set("actor", _mirror_actor(obs["actor"]))
            mirrored.set("critic", _mirror_critic(obs["critic"]))
        except KeyError as error:
            raise KeyError(
                "LeggyBot symmetry augmentation requires actor and critic observation groups"
            ) from error
        augmented_obs = torch.cat((obs, mirrored), dim=0)

    augmented_actions = None
    if actions is not None:
        augmented_actions = torch.cat((actions, mirror_leggybot_actions(actions)), dim=0)
    return augmented_obs, augmented_actions


__all__ = ["leggybot_left_right_augmentation", "mirror_leggybot_actions"]
