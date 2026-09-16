"""LeggyBot reward terms that are not provided by mjlab's velocity task."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class UnwrappedFixedHeadingError(ManagerTermBase):
    """Bounded squared yaw error accumulated across angle wrapping."""

    def __init__(self, cfg, env) -> None:
        del cfg
        super().__init__(env)
        self._last_episode_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._last_heading = torch.zeros(self.num_envs, device=self.device)
        self._unwrapped_heading = torch.zeros(self.num_envs, device=self.device)

    @staticmethod
    def _wrap(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._last_episode_step[env_ids] = -1
        self._last_heading[env_ids] = 0.0
        self._unwrapped_heading[env_ids] = 0.0

    def __call__(
        self,
        env,
        target_heading: float = 0.0,
        max_penalty: float = 1.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        if max_penalty <= 0.0:
            raise ValueError("max_penalty must be positive")
        asset: Entity = env.scene[asset_cfg.name]
        current_heading = self._wrap(asset.data.heading_w)
        steps = env.episode_length_buf
        new_episode = (self._last_episode_step < 0) | (
            steps < self._last_episode_step
        )
        if torch.any(new_episode):
            self._last_heading[new_episode] = current_heading[new_episode]
            self._unwrapped_heading[new_episode] = current_heading[new_episode]
        continuing = ~new_episode
        if torch.any(continuing):
            self._unwrapped_heading[continuing] += self._wrap(
                current_heading[continuing] - self._last_heading[continuing]
            )
        self._last_heading.copy_(current_heading)
        self._last_episode_step.copy_(steps)
        error = self._unwrapped_heading - target_heading
        return torch.clamp(torch.square(error), max=max_penalty)


class UnwrappedHeadingMagnitude(UnwrappedFixedHeadingError):
    """Track accumulated heading magnitude for long-horizon diagnostics."""

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        current_heading = self._wrap(asset.data.heading_w)
        steps = env.episode_length_buf
        new_episode = (self._last_episode_step < 0) | (
            steps < self._last_episode_step
        )
        if torch.any(new_episode):
            self._last_heading[new_episode] = current_heading[new_episode]
            self._unwrapped_heading[new_episode] = current_heading[new_episode]
        continuing = ~new_episode
        if torch.any(continuing):
            self._unwrapped_heading[continuing] += self._wrap(
                current_heading[continuing] - self._last_heading[continuing]
            )
        self._last_heading.copy_(current_heading)
        self._last_episode_step.copy_(steps)
        return torch.abs(self._unwrapped_heading)


class GradualBoundedHeadingDriftCost(ManagerTermBase):
    """Penalize accumulated yaw drift from the episode's initial heading."""

    def __init__(self, cfg, env) -> None:
        del cfg
        super().__init__(env)
        self._last_episode_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._last_heading = torch.zeros(self.num_envs, device=self.device)
        self._heading_drift = torch.zeros(self.num_envs, device=self.device)
        self._ramp_start_step: float | None = None

    @staticmethod
    def _wrap(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._last_episode_step[env_ids] = -1
        self._last_heading[env_ids] = 0.0
        self._heading_drift[env_ids] = 0.0

    def __call__(
        self,
        env,
        deadband: float = 0.075,
        scale: float = 0.25,
        max_cost: float = 4.0,
        ramp_steps: int = 12_000,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        if deadband < 0.0 or scale <= 0.0 or max_cost <= 0.0:
            raise ValueError("invalid heading-drift cost parameters")
        if ramp_steps < 0:
            raise ValueError("ramp_steps must be non-negative")
        asset: Entity = env.scene[asset_cfg.name]
        current_heading = self._wrap(asset.data.heading_w)
        steps = env.episode_length_buf
        new_episode = (self._last_episode_step < 0) | (
            steps < self._last_episode_step
        )
        if torch.any(new_episode):
            self._last_heading[new_episode] = current_heading[new_episode]
            self._heading_drift[new_episode] = 0.0
        continuing = ~new_episode
        if torch.any(continuing):
            self._heading_drift[continuing] += self._wrap(
                current_heading[continuing] - self._last_heading[continuing]
            )
        self._last_heading.copy_(current_heading)
        self._last_episode_step.copy_(steps)

        normalized_excess = torch.relu(
            torch.abs(self._heading_drift) - deadband
        ) / scale
        squared_excess = torch.square(normalized_excess)
        cost = max_cost * (1.0 - torch.exp(-squared_excess / max_cost))
        if ramp_steps == 0:
            return cost
        current_step = float(getattr(env, "common_step_counter", 0))
        if self._ramp_start_step is None:
            self._ramp_start_step = current_step
        progress = min(
            1.0,
            max(0.0, current_step - self._ramp_start_step) / float(ramp_steps),
        )
        return progress * cost


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
