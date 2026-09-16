"""Deployment-compatible action mappings for project robots."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values


@dataclass(kw_only=True)
class AsymmetricJointPositionActionCfg(JointPositionActionCfg):
    """Position action whose two normalized sides have independent gains."""

    target_lower: dict[str, float]
    target_upper: dict[str, float]
    initial_range_fraction: float = 1.0
    range_ramp_steps: int = 0
    model_backlash: bool = True

    def build(self, env):
        return AsymmetricJointPositionAction(self, env)


class AsymmetricJointPositionAction(JointPositionAction):
    """Map -1/1 to lower/upper targets while keeping zero at nominal pose."""

    cfg: AsymmetricJointPositionActionCfg

    def __init__(self, cfg: AsymmetricJointPositionActionCfg, env):
        # The base class would apply a single symmetric scale around nominal.
        # Disable that path; this class performs the piecewise-affine mapping.
        cfg.use_default_offset = False
        cfg.scale = 1.0
        cfg.offset = 0.0
        super().__init__(cfg, env)
        _, _, lower = resolve_matching_names_values(
            cfg.target_lower, self._target_names
        )
        _, _, upper = resolve_matching_names_values(
            cfg.target_upper, self._target_names
        )
        self._target_lower = torch.tensor(lower, device=self.device)
        self._target_upper = torch.tensor(upper, device=self.device)
        if not 0.0 < cfg.initial_range_fraction <= 1.0:
            raise ValueError("initial_range_fraction must be in (0, 1]")
        if cfg.range_ramp_steps < 0:
            raise ValueError("range_ramp_steps must be non-negative")
        self._initial_range_fraction = cfg.initial_range_fraction
        self._range_ramp_steps = cfg.range_ramp_steps
        # The curriculum can take ownership of this value at runtime.  Keeping
        # it on the action term makes the policy/deployment mapping unchanged.
        self._range_fraction = self._initial_range_fraction
        nominal = self._entity.data.default_joint_pos[:, self._target_ids]
        # Half-width of a stateful play operator.  It remains zero in playback
        # and for tasks without servo randomization; reset events populate it
        # during sim-to-real training.
        self._backlash_half_width = torch.zeros_like(self._processed_actions)
        self._backlash_output = nominal.clone()
        if torch.any(self._target_lower >= nominal[0]) or torch.any(
            self._target_upper <= nominal[0]
        ):
            raise ValueError("normalized action bounds must contain nominal pose")

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        normalized = torch.clamp(self._raw_actions, -1.0, 1.0)
        if self._range_ramp_steps:
            progress = min(
                1.0,
                float(self._env.common_step_counter) / self._range_ramp_steps,
            )
            range_fraction = self._initial_range_fraction + (
                1.0 - self._initial_range_fraction
            ) * progress
        else:
            range_fraction = self._range_fraction
        normalized = normalized * range_fraction
        nominal = self._entity.data.default_joint_pos[:, self._target_ids]
        requested = nominal + torch.where(
            normalized >= 0.0,
            normalized * (self._target_upper - nominal),
            normalized * (nominal - self._target_lower),
        )
        if self.cfg.model_backlash:
            # A play/backlash operator: small reversals inside the gear-train
            # clearance do not move the output; larger commands take up the
            # clearance before motion resumes.
            self._backlash_output[:] = torch.maximum(
                torch.minimum(
                    self._backlash_output,
                    requested + self._backlash_half_width,
                ),
                requested - self._backlash_half_width,
            )
            self._processed_actions[:] = self._backlash_output
        else:
            self._processed_actions[:] = requested

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        nominal = self._entity.data.default_joint_pos[:, self._target_ids]
        self._processed_actions[env_ids] = nominal[env_ids]
        self._backlash_output[env_ids] = nominal[env_ids]

    def set_backlash_half_width(
        self, half_width: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        """Set per-environment joint backlash radii in radians."""
        if half_width.shape != (len(env_ids), self.action_dim):
            raise ValueError(
                "backlash half-width must have shape "
                f"({len(env_ids)}, {self.action_dim}), got {tuple(half_width.shape)}"
            )
        if torch.any(half_width < 0.0):
            raise ValueError("backlash half-width cannot be negative")
        self._backlash_half_width[env_ids] = half_width

    def set_range_fraction(self, fraction: float) -> None:
        """Set training-time action authority without changing the policy API."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError("action range fraction must be in (0, 1]")
        self._range_fraction = float(fraction)
