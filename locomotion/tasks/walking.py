"""The complete LeggyBot walking task.

The task is intentionally defined in one place.  Its environment starts from
the robot/simulator foundation and declares its command distribution,
observations, randomization, objective, metrics, and PPO configuration
directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import list_tasks, register_mjlab_task
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

from locomotion.task_runner import task_runner
from locomotion.tasks.gait import (
    Sim2RealImuAngularVelocity,
    Sim2RealImuProjectedGravity,
    Sim2RealJointPosition,
    Sim2RealJointVelocity,
    actuator_effort_cost,
    commanded_lateral_foot_line_cost,
    commanded_motion_rate_safety_cost,
    commanded_neutral_stance,
    commanded_periodic_stance_foot_speed_cost,
    commanded_periodic_swing_force_cost,
    commanded_scheduled_foot_contact,
    gait_clock,
    lateral_command_active,
    lateral_foot_line_error,
    periodic_walk_evaluation_active,
    pitch_dominant_angular_velocity,
    pitch_upright,
    stopped_foot_alignment_cost,
    updated_default_domain_randomization,
)
from locomotion.tasks.leggybot import _leggybot_base_env_cfg, leggybot_ppo_runner_cfg
from locomotion.tasks.omni_tracking import WindowedSE2Tracking, tracking_success

WALKING_TASK_ID = "WalkingTask"
LOCOMOTION_TASK_IDS = (WALKING_TASK_ID,)

_FOOT_SITES = SceneEntityCfg(
    "robot", site_names=("right_foot", "left_foot"), preserve_order=True
)
_RANDOMIZATION_PAIRED_STEPS = 500 * 24
_RANDOMIZATION_RAMP_STEPS = 300 * 24


@dataclass(kw_only=True)
class WalkingCommandCfg(UniformVelocityCommandCfg):
    """Deployment-weighted mixture of standing, straight, and curved motion."""

    forward_straight_probability: float = 0.45
    forward_curve_probability: float = 0.45
    pure_lateral_probability: float = 0.0
    full_omni_probability: float = 0.0
    stop_probability: float = 0.10
    reverse_probability: float = 0.0
    protected_forward_range: tuple[float, float] = (0.04, 0.10)
    minimum_coupled_yaw_fraction: float = 0.25
    scale_yaw_with_forward_speed: bool = False
    minimum_lateral_speed: float = 0.04

    def build(self, env):
        return WalkingCommand(self, env)


class WalkingCommand(UniformVelocityCommand):
    """Sample mutually exclusive walking command modes."""

    cfg: WalkingCommandCfg

    _FORWARD_STRAIGHT = 0
    _FORWARD_CURVE = 1
    _PURE_LATERAL = 2
    _FULL_OMNI = 3
    _STOP = 4
    _REVERSE = 5

    def _resample_command(self, env_ids):
        probabilities = torch.as_tensor(
            (
                self.cfg.forward_straight_probability,
                self.cfg.forward_curve_probability,
                self.cfg.pure_lateral_probability,
                self.cfg.full_omni_probability,
                self.cfg.stop_probability,
                self.cfg.reverse_probability,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        if torch.any(probabilities < 0.0) or not torch.isclose(
            probabilities.sum(), probabilities.new_tensor(1.0), atol=1.0e-6
        ):
            raise ValueError("command mode probabilities must be non-negative and sum to 1")

        super()._resample_command(env_ids)
        count = len(env_ids)
        modes = torch.multinomial(probabilities, count, replacement=True)
        command = torch.zeros((count, 3), device=self.device)

        x_low, x_high = (float(value) for value in self.cfg.ranges.lin_vel_x)
        y_low, y_high = (float(value) for value in self.cfg.ranges.lin_vel_y)
        yaw_limit = max(
            abs(float(self.cfg.ranges.ang_vel_z[0])),
            abs(float(self.cfg.ranges.ang_vel_z[1])),
        )
        protected_low, protected_high = self.cfg.protected_forward_range
        if protected_low < 0.0 or protected_low > protected_high:
            raise ValueError("protected forward range must be ordered and non-negative")
        forward_limit = max(
            abs(x_low), abs(x_high), abs(protected_low), abs(protected_high)
        )

        def uniform(low: float, high: float) -> torch.Tensor:
            return torch.empty(count, device=self.device).uniform_(low, high)

        def conditional_yaw(vx: torch.Tensor) -> torch.Tensor:
            if forward_limit == 0.0 or yaw_limit == 0.0:
                return torch.zeros_like(vx)
            fraction = float(self.cfg.minimum_coupled_yaw_fraction)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("minimum coupled yaw fraction must be in [0, 1]")
            magnitude = fraction + (1.0 - fraction) * torch.rand(
                count, device=self.device
            )
            speed_scale = (
                torch.abs(vx) / forward_limit
                if self.cfg.scale_yaw_with_forward_speed
                else torch.ones_like(vx)
            )
            sign = 2.0 * torch.randint(
                0, 2, (count,), device=self.device
            ).float() - 1.0
            return sign * yaw_limit * speed_scale * magnitude

        straight = modes == self._FORWARD_STRAIGHT
        if torch.any(straight):
            command[straight, 0] = uniform(
                max(0.0, protected_low), max(0.0, protected_high)
            )[straight]

        curve = modes == self._FORWARD_CURVE
        if torch.any(curve):
            curve_vx = uniform(
                max(0.0, protected_low), max(0.0, protected_high)
            )
            command[curve, 0] = curve_vx[curve]
            command[curve, 2] = conditional_yaw(curve_vx)[curve]

        lateral = modes == self._PURE_LATERAL
        if torch.any(lateral):
            minimum = float(self.cfg.minimum_lateral_speed)
            lateral_limit = max(abs(y_low), abs(y_high))
            if not 0.0 <= minimum <= lateral_limit:
                raise ValueError("minimum lateral speed must fit the lateral range")
            if minimum > 0.0:
                magnitude = uniform(minimum, lateral_limit)
                sign = 2.0 * torch.randint(
                    0, 2, (count,), device=self.device
                ) - 1.0
                command[lateral, 1] = (sign * magnitude)[lateral]
            else:
                command[lateral, 1] = uniform(y_low, y_high)[lateral]

        omni = modes == self._FULL_OMNI
        if torch.any(omni):
            omni_vx = uniform(x_low, x_high)
            command[omni, 0] = omni_vx[omni]
            command[omni, 1] = uniform(y_low, y_high)[omni]
            command[omni, 2] = conditional_yaw(omni_vx)[omni]

        reverse = modes == self._REVERSE
        if torch.any(reverse):
            command[reverse, 0] = uniform(
                min(x_low, 0.0), min(x_high, 0.0)
            )[reverse]

        self.vel_command_b[env_ids] = command
        self.vel_command_w[env_ids] = command
        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = False
        self.is_world_env[env_ids] = False
        self.is_forward_env[env_ids] = False


def walking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the complete training or playback environment."""
    cfg = _leggybot_base_env_cfg(play=play)

    for group_name in ("actor", "critic"):
        cfg.observations[group_name].terms["gait_clock"] = ObservationTermCfg(
            func=gait_clock, params={"period": 0.33}
        )

    actor = cfg.observations["actor"]
    imu_params = {
        "gyro_sensor_name": "robot/imu_ang_vel",
        "accel_sensor_name": "robot/base_accelerometer",
    }
    actor.terms["base_ang_vel"].func = Sim2RealImuAngularVelocity
    actor.terms["base_ang_vel"].params = imu_params.copy()
    actor.terms["projected_gravity"].func = Sim2RealImuProjectedGravity
    actor.terms["projected_gravity"].params = imu_params.copy()
    actor.terms["joint_pos"].func = Sim2RealJointPosition
    actor.terms["joint_pos"].params = {}
    actor.terms["joint_vel"].func = Sim2RealJointVelocity
    actor.terms["joint_vel"].params = {"filter_alpha": 0.5}
    for group in cfg.observations.values():
        for term in group.terms.values():
            term.noise = None

    cfg.commands["twist"] = WalkingCommandCfg(
        entity_name="robot",
        resampling_time_range=(5.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_world_envs=0.0,
        rel_forward_envs=0.0,
        heading_command=False,
        init_velocity_prob=0.0,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.10),
            lin_vel_y=(-0.08, 0.08),
            ang_vel_z=(-0.40, 0.40),
            heading=None,
        ),
    )

    upright = cfg.rewards["upright"]
    upright.weight = 0.75
    dof_pos_limits = cfg.rewards["dof_pos_limits"]
    dof_pos_limits.weight = -2.0
    action_rate_l2 = cfg.rewards["action_rate_l2"]
    action_rate_l2.weight = -0.03
    cfg.rewards = {
        "upright": upright,
        "dof_pos_limits": dof_pos_limits,
        "action_rate_l2": action_rate_l2,
        "uncommanded_base_ang_vel": RewardTermCfg(
            func=pitch_dominant_angular_velocity,
            weight=-0.08,
            params={"command_name": "twist"},
        ),
        "termination": RewardTermCfg(func=mdp.is_terminated, weight=-250.0),
        "scheduled_contact": RewardTermCfg(
            func=commanded_scheduled_foot_contact,
            weight=0.40,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "period": 0.33,
                "duty_factor": 0.70,
                "full_motion_speed": 0.04,
            },
        ),
        "swing_foot_force": RewardTermCfg(
            func=commanded_periodic_swing_force_cost,
            weight=-0.35,
            params={
                "sensor_name": "feet_ground_contact",
                "force_scale": 4.0,
                "transition_fraction": 0.04,
                "command_name": "twist",
                "period": 0.33,
                "duty_factor": 0.70,
                "full_motion_speed": 0.04,
            },
        ),
        "stance_foot_speed": RewardTermCfg(
            func=commanded_periodic_stance_foot_speed_cost,
            weight=-0.35,
            params={
                "speed_scale": 0.06,
                "transition_fraction": 0.04,
                "asset_cfg": _FOOT_SITES,
                "command_name": "twist",
                "period": 0.33,
                "duty_factor": 0.70,
                "full_motion_speed": 0.04,
            },
        ),
        "motion_rate_safety": RewardTermCfg(
            func=commanded_motion_rate_safety_cost,
            weight=-0.05,
            params={"command_name": "twist"},
        ),
        "actuator_effort": RewardTermCfg(
            func=actuator_effort_cost,
            weight=-0.03,
            params={
                "effort_scale": 1.5,
                "joint_weights": (1.0, 0.33, 0.33, 1.0, 0.33, 0.33),
            },
        ),
        "pitch_upright": RewardTermCfg(
            func=pitch_upright, weight=1.50, params={"std": 0.10}
        ),
        "linear_tracking": RewardTermCfg(
            func=WindowedSE2Tracking,
            weight=3.0,
            params={
                "horizon_s": 0.66,
                "linear_scale": 0.035,
                "yaw_scale": 0.15,
                "output": "linear_reward",
            },
        ),
        "yaw_tracking": RewardTermCfg(
            func=WindowedSE2Tracking,
            weight=1.5,
            params={
                "horizon_s": 0.66,
                "linear_scale": 0.035,
                "yaw_scale": 0.12,
                "output": "yaw_reward",
            },
        ),
        "linear_path_error": RewardTermCfg(
            func=WindowedSE2Tracking,
            weight=-0.75,
            params={
                "horizon_s": 2.0,
                "linear_scale": 0.035,
                "yaw_scale": 0.12,
                "output": "linear_cost",
            },
        ),
        "yaw_path_error": RewardTermCfg(
            func=WindowedSE2Tracking,
            weight=-0.75,
            params={
                "horizon_s": 2.0,
                "linear_scale": 0.035,
                "yaw_scale": 0.04,
                "output": "yaw_cost",
            },
        ),
        "neutral_stance": RewardTermCfg(
            func=commanded_neutral_stance,
            weight=0.80,
            params={
                "command_name": "twist",
                "joint_std": 0.12,
                "full_motion_speed": 0.04,
            },
        ),
        "stopped_foot_alignment": RewardTermCfg(
            func=stopped_foot_alignment_cost,
            weight=-0.30,
            params={
                "command_name": "twist",
                "fore_aft_scale": 0.015,
                "full_motion_speed": 0.04,
                "asset_cfg": _FOOT_SITES,
            },
        ),
        "lateral_foot_line": RewardTermCfg(
            func=commanded_lateral_foot_line_cost,
            weight=-0.20,
            params={
                "command_name": "twist",
                "line_scale": 0.025,
                "period": 0.33,
                "duty_factor": 0.70,
                "transition_fraction": 0.04,
                "asset_cfg": _FOOT_SITES,
            },
        ),
        "cross_track_precision": RewardTermCfg(
            func=WindowedSE2Tracking,
            weight=-0.50,
            params={
                "horizon_s": 1.0,
                "linear_scale": 0.010,
                "output": "cross_track_cost",
            },
        ),
    }

    cfg.metrics = {
        "tracking_success": MetricsTermCfg(
            func=WindowedSE2Tracking,
            params={
                "horizon_s": 0.66,
                "linear_scale": 0.025,
                "yaw_scale": 0.10,
                "output": "success",
            },
        ),
        "window_linear_error_mps": MetricsTermCfg(
            func=WindowedSE2Tracking,
            params={"horizon_s": 0.66, "output": "linear_error"},
        ),
        "window_yaw_error_radps": MetricsTermCfg(
            func=WindowedSE2Tracking,
            params={"horizon_s": 0.66, "output": "yaw_error"},
        ),
        "lateral_foot_line_error_m": MetricsTermCfg(
            func=lateral_foot_line_error,
            params={
                "command_name": "twist",
                "min_lateral_speed": 0.04,
                "max_forward_speed": 0.01,
                "max_yaw_rate": 0.02,
                "asset_cfg": _FOOT_SITES,
            },
        ),
    }
    for direction, label in ((1, "left"), (-1, "right")):
        cfg.metrics[f"{label}_lateral_command_fraction"] = MetricsTermCfg(
            func=lateral_command_active,
            params={
                "command_name": "twist",
                "lateral_direction": direction,
                "min_lateral_speed": 0.04,
                "max_forward_speed": 0.01,
                "max_yaw_rate": 0.02,
            },
        )
        cfg.metrics[f"{label}_lateral_tracking_success"] = MetricsTermCfg(
            func=WindowedSE2Tracking,
            params={
                "horizon_s": 0.66,
                "linear_scale": 0.025,
                "yaw_scale": 0.10,
                "output": "success",
                "lateral_direction": direction,
                "min_lateral_speed": 0.04,
                "max_forward_speed": 0.01,
                "max_yaw_rate": 0.02,
            },
        )
        cfg.metrics[f"{label}_lateral_foot_line_error_m"] = MetricsTermCfg(
            func=lateral_foot_line_error,
            params={
                "command_name": "twist",
                "min_lateral_speed": 0.04,
                "max_forward_speed": 0.01,
                "max_yaw_rate": 0.02,
                "lateral_direction": direction,
                "asset_cfg": _FOOT_SITES,
            },
        )
    cfg.metrics["cross_track_error_mps"] = MetricsTermCfg(
        func=WindowedSE2Tracking,
        params={"horizon_s": 1.0, "output": "cross_track_error"},
    )

    cfg.events["push"] = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (-0.05, 0.05),
                "roll": (-0.20, 0.20),
                "pitch": (-0.20, 0.20),
                "yaw": (-0.20, 0.20),
            }
        },
    )
    cfg.manages_sim2real_randomization = True
    cfg.sim2real_randomization_enabled = not play
    cfg.curriculum = {}
    cfg.episode_length_s = 20.0
    if play:
        cfg.events.pop("domain_randomization", None)
        cfg.events.pop("push", None)
    else:
        cfg.events["domain_randomization"] = EventTermCfg(
            func=updated_default_domain_randomization,
            mode="reset",
            params={
                "symmetric_steps": _RANDOMIZATION_PAIRED_STEPS,
                "independent_ramp_steps": _RANDOMIZATION_RAMP_STEPS,
            },
        )
        # Keep the same event order as the established training environment.
        cfg.events["push"] = cfg.events.pop("push")
    return cfg


@dataclass(frozen=True)
class LocomotionTask:
    task_id: str
    environment_factory: object
    experiment_name: str
    success_metric: object
    active_metric: object
    success_threshold: float


_WALKING_TASK = LocomotionTask(
    task_id=WALKING_TASK_ID,
    environment_factory=walking_env_cfg,
    experiment_name="leggybot_walking",
    success_metric=tracking_success,
    active_metric=periodic_walk_evaluation_active,
    success_threshold=0.80,
)


def locomotion_tasks() -> tuple[LocomotionTask, ...]:
    return (_WALKING_TASK,)


def get_locomotion_task(task_id: str) -> LocomotionTask:
    if task_id != WALKING_TASK_ID:
        raise ValueError(f"No implemented locomotion task is named {task_id!r}")
    return _WALKING_TASK


def register_leggybot_tasks() -> None:
    if WALKING_TASK_ID in set(list_tasks()):
        return
    register_mjlab_task(
        task_id=WALKING_TASK_ID,
        env_cfg=walking_env_cfg(),
        play_env_cfg=walking_env_cfg(play=True),
        rl_cfg=leggybot_ppo_runner_cfg(
            _WALKING_TASK.experiment_name,
            use_symmetry_augmentation=True,
            symmetry_mirror_loss_coeff=0.05,
            bounded_exploration=True,
            exploration_init_std=0.15,
            exploration_std_range=(0.03, 0.40),
            learning_rate=1e-3,
        ),
        runner_cls=task_runner(WALKING_TASK_ID),
    )


__all__ = [
    "LOCOMOTION_TASK_IDS",
    "WALKING_TASK_ID",
    "LocomotionTask",
    "WalkingCommand",
    "WalkingCommandCfg",
    "get_locomotion_task",
    "locomotion_tasks",
    "register_leggybot_tasks",
    "walking_env_cfg",
]
