"""Finite-horizon SE(2) tracking for a servo biped without measured odometry.

The reference follows commanded yaw, never measured yaw. Sliding windows keep
credit assignment local and do not let old mistakes cancel new mistakes.
"""
from __future__ import annotations

import math
import torch

from mjlab.managers.manager_base import ManagerTermBase


def integrate_twist(pose: torch.Tensor, command: torch.Tensor, dt: float) -> torch.Tensor:
    """Exact constant-body-twist integration, including the zero-yaw limit."""
    angle = command[:, 2] * dt
    half = angle * .5
    distance_scale = dt * torch.sinc(half / math.pi)
    heading = pose[:, 2] + half
    return pose + torch.stack((
        distance_scale * (heading.cos()*command[:, 0] - heading.sin()*command[:, 1]),
        distance_scale * (heading.sin()*command[:, 0] + heading.cos()*command[:, 1]),
        angle,
    ), dim=-1)


class WindowedSE2Tracking(ManagerTermBase):
    """Compare actual and desired motion over a sliding window in the start frame.

    Independent instances can serve rewards and metrics. Each instance updates
    at most once per episode step and supports asynchronous partial resets.
    No additional actor observations or simulator odometry are required.
    """
    def __init__(self, cfg, env):
        super().__init__(env)
        self.length = max(1, round(cfg.params.get("horizon_s", .66) / env.step_dt))
        self.actual = torch.zeros(self.length+1, env.num_envs, 3, device=env.device)
        self.reference = torch.zeros_like(self.actual)
        self.pose = torch.zeros(env.num_envs, 3, device=env.device)
        self.target = torch.zeros_like(self.pose)
        self.last_heading = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.age = torch.zeros_like(self.last_step)
        self.error = torch.zeros_like(self.pose)
        self.displacement_error = torch.zeros_like(self.pose)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.last_step[ids] = -1
        self.age[ids] = 0
        self.error[ids] = 0
        self.displacement_error[ids] = 0

    def errors(self, env, command_name="twist"):
        data = env.scene["robot"].data
        steps = env.episode_length_buf
        heading = data.heading_w
        fresh = (self.last_step < 0) | (steps < self.last_step)
        update = fresh | (steps > self.last_step)
        continuing = update & ~fresh
        ids = torch.arange(env.num_envs, device=env.device)
        if fresh.any():
            self.pose[fresh, :2] = data.root_link_pos_w[fresh, :2]
            self.pose[fresh, 2] = heading[fresh]
            self.target[fresh] = self.pose[fresh]
            self.age[fresh] = 0
            self.actual[:, fresh] = self.pose[fresh]
            self.reference[:, fresh] = self.target[fresh]
            self.error[fresh] = 0
            self.displacement_error[fresh] = 0
        if continuing.any():
            delta = heading - self.last_heading
            self.pose[continuing, :2] = data.root_link_pos_w[continuing, :2]
            self.pose[continuing, 2] += torch.atan2(delta.sin(), delta.cos())[continuing]
            command = env.command_manager.get_command(command_name)
            self.target[continuing] = integrate_twist(self.target[continuing], command[continuing], env.step_dt)
            self.age[continuing] += 1
        slot = self.age % (self.length+1)
        self.actual[slot[update], ids[update]] = self.pose[update]
        self.reference[slot[update], ids[update]] = self.target[update]
        start_slot = (self.age - self.length).clamp(min=0) % (self.length+1)
        start = self.actual[start_slot, ids]
        target_start = self.reference[start_slot, ids]
        actual_delta = self.pose - start
        target_delta = self.target - target_start

        def local(delta, yaw):
            return torch.stack((yaw.cos()*delta[:, 0]+yaw.sin()*delta[:, 1],
                                -yaw.sin()*delta[:, 0]+yaw.cos()*delta[:, 1],
                                delta[:, 2]), -1)

        displacement_error = (
            local(actual_delta, start[:, 2])
            - local(target_delta, target_start[:, 2])
        )
        elapsed = (self.age.clamp(min=1, max=self.length) * env.step_dt)[:, None]
        self.displacement_error[update] = displacement_error[update]
        self.error[update] = (displacement_error / elapsed)[update]
        self.last_heading[update] = heading[update]
        self.last_step[update] = steps[update]
        return self.error

    def __call__(self, env, command_name="twist", horizon_s=.66,
                 linear_scale=.04, yaw_scale=.15, output="reward",
                 lateral_direction: int | None = None,
                 min_lateral_speed: float = .04,
                 max_forward_speed: float = .01,
                 max_yaw_rate: float = .02):
        if min(horizon_s, linear_scale, yaw_scale) <= 0:
            raise ValueError("tracking horizons and scales must be positive")
        error = self.errors(env, command_name)
        linear_normalized = error[:, :2].norm(dim=-1) / linear_scale
        yaw_normalized = error[:, 2].abs() / yaw_scale
        if output == "linear_error":
            return error[:, :2].norm(dim=-1)
        if output == "yaw_error":
            return error[:, 2].abs()
        if output in ("cross_track_cost", "cross_track_error"):
            # Orthogonal error for axis-exclusive straight commands. Curves
            # retain the SE(2) objective; their two legs need unequal strides.
            command = env.command_manager.get_command(command_name)
            forward = (command[:, 0].abs() >= .02) & (command[:, 1].abs() <= .01)
            lateral = (command[:, 1].abs() >= .02) & (command[:, 0].abs() <= .01)
            active = (forward | lateral) & (command[:, 2].abs() <= .02)
            cross = torch.where(forward, error[:, 1], error[:, 0])
            value = cross.abs() if output == "cross_track_error" else (
                torch.sqrt(1.0 + (cross / linear_scale).square()) - 1.0
            )
            return value * active * (self.age >= self.length)
        if output == "linear_displacement_error":
            return self.displacement_error[:, :2].norm(dim=-1)
        if output == "yaw_displacement_error":
            return self.displacement_error[:, 2].abs()
        if output == "linear_reward":
            return torch.exp(-torch.square(linear_normalized))
        if output == "yaw_reward":
            return torch.exp(-torch.square(yaw_normalized))
        if output == "linear_cost":
            return torch.sqrt(1.0 + torch.square(linear_normalized)) - 1.0
        if output == "yaw_cost":
            return torch.sqrt(1.0 + torch.square(yaw_normalized)) - 1.0
        if output == "linear_displacement_cost":
            normalized = self.displacement_error[:, :2].norm(dim=-1) / linear_scale
            return torch.sqrt(1.0 + torch.square(normalized)) - 1.0
        if output == "yaw_displacement_cost":
            normalized = self.displacement_error[:, 2].abs() / yaw_scale
            return torch.sqrt(1.0 + torch.square(normalized)) - 1.0
        if output == "success":
            upright = -env.scene["robot"].data.projected_gravity_b[:, 2] > .9
            success = ((error[:, :2].norm(dim=-1) < linear_scale) &
                       (error[:, 2].abs() < yaw_scale) & upright &
                       (self.age >= self.length)).float()
            if lateral_direction is None:
                return success
            if lateral_direction not in (-1, 1):
                raise ValueError("lateral_direction must be -1, 1, or None")
            command = env.command_manager.get_command(command_name)
            active = (
                (command[:, 1] * lateral_direction >= min_lateral_speed)
                & (command[:, 0].abs() <= max_forward_speed)
                & (command[:, 2].abs() <= max_yaw_rate)
            )
            # MetricsManager averages all environments.  Inactive samples are
            # zero by design; the paired active-fraction metrics in V3 make
            # the conditional success rate recoverable without hiding a sign.
            return success * active.to(success.dtype)
        scaled = error / error.new_tensor((linear_scale, linear_scale, yaw_scale))
        cost = (torch.sqrt(1+scaled.square())-1).sum(-1)
        if output == "cost":
            return cost
        if output != "reward":
            raise ValueError(f"Unknown tracking output: {output}")
        # Smooth cost keeps an improvement signal far from the goal, while the
        # bonus makes accurate simultaneous tracking more valuable than standing.
        return torch.exp(-cost) - .25*cost


def tracking_success(env):
    """Use the same finite-window definition as the logged training metric."""
    manager = env.metrics_manager
    return manager._step_values[:, manager.active_terms.index("tracking_success")]
