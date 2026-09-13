"""LeggyBot reward terms that are not provided by mjlab's velocity task."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def fixed_heading_error(
    env,
    target_heading: float = 0.0,
    max_penalty: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bounded squared yaw error from a fixed world-frame heading."""
    if max_penalty <= 0.0:
        raise ValueError("max_penalty must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    error = asset.data.heading_w - target_heading
    wrapped_error = torch.atan2(torch.sin(error), torch.cos(error))
    return torch.clamp(torch.square(wrapped_error), max=max_penalty)


def lateral_path_deviation(
    env,
    tolerance: float = 0.15,
    max_penalty: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Normalized squared distance from the environment's forward X axis."""
    if tolerance <= 0.0 or max_penalty <= 0.0:
        raise ValueError("tolerance and max_penalty must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    lateral_offset = asset.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return torch.clamp(
        torch.square(lateral_offset / tolerance), max=max_penalty
    )


def lateral_velocity_deviation(
    env,
    tolerance: float = 0.10,
    max_penalty: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize instantaneous sideways velocity, not accumulated drift."""
    if tolerance <= 0.0 or max_penalty <= 0.0:
        raise ValueError("tolerance and max_penalty must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    lateral_velocity = asset.data.root_link_lin_vel_b[:, 1]
    return torch.clamp(torch.square(lateral_velocity / tolerance), max=max_penalty)


def uncommanded_base_angular_velocity(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize roll/pitch rates and only the uncommanded part of yaw rate."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command {command_name!r} not found"

    angular_velocity = asset.data.root_link_ang_vel_b
    roll_pitch_cost = torch.sum(torch.square(angular_velocity[:, :2]), dim=1)
    yaw_error_cost = torch.square(angular_velocity[:, 2] - command[:, 2])
    return roll_pitch_cost + yaw_error_cost


def command_direction_progress(
    env,
    command_name: str,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward signed velocity progress along the requested translation direction.

    A motionless robot receives exactly zero. Matching the requested velocity
    receives one, moving in the wrong direction receives a negative reward, and
    clipping prevents excessive speed from becoming profitable.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command {command_name!r} not found"

    command_xy = command[:, :2]
    actual_xy = asset.data.root_link_lin_vel_b[:, :2]
    command_norm_sq = torch.sum(torch.square(command_xy), dim=1)
    progress = torch.sum(actual_xy * command_xy, dim=1) / torch.clamp(
        command_norm_sq, min=command_threshold**2
    )
    active = command_norm_sq > command_threshold**2
    return torch.clamp(progress, -1.0, 1.0) * active.float()
