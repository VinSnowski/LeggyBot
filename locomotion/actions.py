"""Deployment-compatible action mappings for project robots."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

from locomotion.servo import (
    STS3215_MAX_PROFILE_ACCELERATION_RAD_S2,
    STS3215_MAX_PROFILE_SPEED_RAD_S,
    quantize_position_torch,
)


@dataclass(kw_only=True)
class AsymmetricJointPositionActionCfg(JointPositionActionCfg):
    """Position action whose two normalized sides have independent gains."""

    target_lower: dict[str, float]
    target_upper: dict[str, float]
    initial_range_fraction: float = 1.0
    range_ramp_steps: int = 0
    model_backlash: bool = True
    target_resolution_rad: float | None = None
    command_delay_max_steps: int = 7
    command_delay_hold_probability: float = 0.9
    missed_frame_probability: float = 0.01
    model_firmware_profile: bool = True
    firmware_max_velocity_rad_s: float = STS3215_MAX_PROFILE_SPEED_RAD_S
    firmware_max_acceleration_rad_s2: float = (
        STS3215_MAX_PROFILE_ACCELERATION_RAD_S2
    )

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
        if (
            cfg.target_resolution_rad is not None
            and cfg.target_resolution_rad <= 0.0
        ):
            raise ValueError("target_resolution_rad must be positive when provided")
        if cfg.command_delay_max_steps < 0:
            raise ValueError("command_delay_max_steps must be non-negative")
        if not 0.0 <= cfg.command_delay_hold_probability <= 1.0:
            raise ValueError("command_delay_hold_probability must be in [0, 1]")
        if not 0.0 <= cfg.missed_frame_probability <= 1.0:
            raise ValueError("missed_frame_probability must be in [0, 1]")
        if cfg.firmware_max_velocity_rad_s <= 0.0:
            raise ValueError("firmware profile velocity must be positive")
        if cfg.firmware_max_acceleration_rad_s2 <= 0.0:
            raise ValueError("firmware profile acceleration must be positive")
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
        # A servo count is a software-side coordinate.  The actual assembled
        # joint may have a zero offset and a small scale error.  mjlab already
        # owns the additive per-joint encoder_bias tensor, so extend that same
        # command/feedback mapping with a scale tensor rather than introducing
        # an unrelated observation perturbation.
        if not hasattr(self._entity.data, "encoder_scale"):
            self._entity.data.encoder_scale = torch.ones_like(
                self._entity.data.encoder_bias
            )

        # One FIFO models the synchronized serial packet for all six servos.
        # Deadlines are physics-step counters; keeping commands in order avoids
        # the nonphysical overtaking that can occur when a time-varying history
        # index is used as a delay model.
        self._delay_queue_capacity = max(
            3,
            (cfg.command_delay_max_steps + max(1, env.cfg.decimation) - 1)
            // max(1, env.cfg.decimation)
            + 2,
        )
        queue_shape = (env.num_envs, self._delay_queue_capacity)
        self._delay_targets = torch.zeros(
            (*queue_shape, self.action_dim), device=self.device
        )
        self._delay_deadlines = torch.zeros(
            queue_shape, dtype=torch.long, device=self.device
        )
        self._delay_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=self.device
        )
        self._all_env_ids = torch.arange(env.num_envs, device=self.device)
        self._delivered_target = nominal.clone()
        self._firmware_position = nominal.clone()
        self._firmware_velocity = torch.zeros_like(nominal)
        self._firmware_max_velocity = torch.full_like(
            nominal, cfg.firmware_max_velocity_rad_s
        )
        self._firmware_max_acceleration = torch.full_like(
            nominal, cfg.firmware_max_acceleration_rad_s2
        )
        self._base_delay_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=self.device
        )
        self._jitter_steps = torch.zeros_like(self._base_delay_steps)
        self._delay_strength = torch.zeros(env.num_envs, device=self.device)
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
        if self.cfg.target_resolution_rad is not None:
            # The STS3215 goal-position register accepts integer encoder
            # counts. Quantize the electrical command before the mechanical
            # play operator, while leaving MuJoCo's physical state continuous.
            requested = quantize_position_torch(
                requested, self.cfg.target_resolution_rad
            )
        # Convert the integer register command through the actual assembled
        # calibration before applying mechanical gearbox play. Backlash is a
        # load-side angular clearance, so its configured width is in physical
        # joint radians rather than software/count coordinates.
        physical_requested = self._software_to_physical(requested)
        self._processed_actions[:] = requested
        if self.cfg.model_backlash:
            # A play/backlash operator: small reversals inside the gear-train
            # clearance do not move the output; larger commands take up the
            # clearance before motion resumes.
            self._backlash_output[:] = torch.maximum(
                torch.minimum(
                    self._backlash_output,
                    physical_requested + self._backlash_half_width,
                ),
                physical_requested - self._backlash_half_width,
            )
            physical_target = self._backlash_output
        else:
            physical_target = physical_requested

        self._enqueue_physical_target(physical_target)

    def _software_to_physical(self, target: torch.Tensor) -> torch.Tensor:
        bias = self._entity.data.encoder_bias[:, self._target_ids]
        scale = self._entity.data.encoder_scale[:, self._target_ids]
        return (target - bias) / scale

    def _sample_command_delay(self) -> torch.Tensor:
        active = self._delay_strength > 0.0
        change = active & (
            torch.rand(self.num_envs, device=self.device)
            > self.cfg.command_delay_hold_probability
        )
        # Most packet-to-packet variation is within one 5 ms physics tick.
        candidates = torch.randint(
            -1, 2, (self.num_envs,), device=self.device
        )
        self._jitter_steps[:] = torch.where(
            change, candidates, self._jitter_steps
        )
        delay = self._base_delay_steps + self._jitter_steps
        missed = active & (
            torch.rand(self.num_envs, device=self.device)
            < self.cfg.missed_frame_probability * self._delay_strength
        )
        delay = delay + missed.long() * int(self._env.cfg.decimation)
        return torch.clamp(delay, 0, self.cfg.command_delay_max_steps)

    def _enqueue_physical_target(self, physical: torch.Tensor) -> None:
        delay = self._sample_command_delay()
        env_ids = self._all_env_ids
        slots = self._delay_count
        # Indexing fails loudly if the analytically sized FIFO ever overflows;
        # avoid a per-policy-step GPU-to-host synchronization for an assertion.
        deadline = int(self._env._sim_step_counter) + 1 + delay
        has_previous = slots > 0
        previous_slot = torch.clamp(slots - 1, min=0)
        previous_deadline = self._delay_deadlines[env_ids, previous_slot]
        # Serial packets preserve ordering. Multiple packets may become
        # available at the same physics boundary, in which case the newest one
        # correctly wins before integration.
        deadline = torch.where(
            has_previous, torch.maximum(deadline, previous_deadline), deadline
        )
        self._delay_targets[env_ids, slots] = physical
        self._delay_deadlines[env_ids, slots] = deadline
        self._delay_count += 1

    def apply_actions(self) -> None:
        current_step = int(self._env._sim_step_counter)
        # A maximum of a few commands can be pending, so bounded FIFO shifting
        # is simpler and safer than maintaining per-environment ring indices.
        for _ in range(self._delay_queue_capacity):
            due = (self._delay_count > 0) & (
                self._delay_deadlines[:, 0] <= current_step
            )
            ids = torch.where(due)[0]
            self._delivered_target[ids] = self._delay_targets[ids, 0]
            self._delay_targets[ids, :-1] = self._delay_targets[ids, 1:].clone()
            self._delay_deadlines[ids, :-1] = self._delay_deadlines[ids, 1:].clone()
            self._delay_count[ids] -= 1
        if self.cfg.model_firmware_profile:
            self._advance_firmware_profile()
        else:
            self._firmware_position[:] = self._delivered_target
            self._firmware_velocity.zero_()
        self._entity.set_joint_position_target(
            self._firmware_position, joint_ids=self._target_ids
        )

    def _advance_firmware_profile(self) -> None:
        """Advance the servo's internal position setpoint by one physics tick."""
        dt = float(self._env.physics_dt)
        error = self._delivered_target - self._firmware_position
        acceleration_step = self._firmware_max_acceleration * dt
        stopping_speed = torch.sqrt(
            acceleration_step.square()
            + 2.0 * self._firmware_max_acceleration * torch.abs(error)
        ) - acceleration_step
        desired_velocity = torch.sign(error) * torch.minimum(
            self._firmware_max_velocity, stopping_speed
        )
        next_velocity = self._firmware_velocity + torch.clamp(
            desired_velocity - self._firmware_velocity,
            -acceleration_step,
            acceleration_step,
        )
        step = next_velocity * dt
        reached = (
            (torch.abs(step) >= torch.abs(error) - 1.0e-12)
            & (torch.sign(step) == torch.sign(error))
            & (torch.abs(next_velocity) <= acceleration_step + 1.0e-12)
        )
        self._firmware_position[:] = torch.where(
            reached,
            self._delivered_target,
            self._firmware_position + step,
        )
        self._firmware_velocity[:] = torch.where(
            reached, torch.zeros_like(next_velocity), next_velocity
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        nominal = self._entity.data.default_joint_pos[:, self._target_ids]
        self._processed_actions[env_ids] = nominal[env_ids]
        physical_nominal = self._software_to_physical(nominal)
        self._backlash_output[env_ids] = physical_nominal[env_ids]
        self._delay_count[env_ids] = 0
        self._jitter_steps[env_ids] = 0
        self._delivered_target[env_ids] = physical_nominal[env_ids]
        self._firmware_position[env_ids] = physical_nominal[env_ids]
        self._firmware_velocity[env_ids] = 0.0

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

    def set_servo_calibration(
        self,
        zero_offset: torch.Tensor,
        scale: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Set the shared physical/count calibration for selected robots."""
        if zero_offset.shape != (len(env_ids), self.action_dim):
            raise ValueError("zero offset has an invalid shape")
        if scale.shape != zero_offset.shape or torch.any(scale <= 0.0):
            raise ValueError("servo calibration scale must be positive and aligned")
        self._entity.data.encoder_bias[
            env_ids[:, None], self._target_ids
        ] = zero_offset
        self._entity.data.encoder_scale[
            env_ids[:, None], self._target_ids
        ] = scale

    def set_command_delay(
        self,
        base_steps: torch.Tensor,
        strength: float,
        env_ids: torch.Tensor,
    ) -> None:
        """Set an episode base delay; per-packet jitter is sampled later."""
        if base_steps.shape != (len(env_ids),):
            raise ValueError("base command delay has an invalid shape")
        if torch.any(base_steps < 0):
            raise ValueError("base command delay cannot be negative")
        self._base_delay_steps[env_ids] = torch.clamp(
            base_steps, 0, self.cfg.command_delay_max_steps
        )
        self._delay_strength[env_ids] = float(strength)
        self._jitter_steps[env_ids] = 0

    def set_firmware_profile_limits(
        self,
        max_velocity: torch.Tensor,
        max_acceleration: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Set physical-radian limits for the internal servo setpoint."""
        expected = (len(env_ids), self.action_dim)
        if max_velocity.shape != expected or max_acceleration.shape != expected:
            raise ValueError("firmware profile limits have an invalid shape")
        if torch.any(max_velocity <= 0.0) or torch.any(max_acceleration <= 0.0):
            raise ValueError("firmware profile limits must be positive")
        self._firmware_max_velocity[env_ids] = max_velocity
        self._firmware_max_acceleration[env_ids] = max_acceleration

    def set_range_fraction(self, fraction: float) -> None:
        """Set training-time action authority without changing the policy API."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError("action range fraction must be in (0, 1]")
        self._range_fraction = float(fraction)
