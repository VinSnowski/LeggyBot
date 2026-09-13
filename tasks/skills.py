"""Common LeggyBot locomotion skill tasks.

Every task deliberately exposes the same actor ABI (including the command and
gait-clock slots).  This allows an actor checkpoint to be transferred between
gates while the critic and optimizer are re-created for the new objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.registry import list_tasks, register_mjlab_task
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand

from locomotion.tasks.gait import (
    actuator_effort_cost,
    CycleGatedEpisodeWorldProgress,
    CycleGatedWorldForwardVelocity,
    CycleNetLateralVelocityCost,
    CycleNetMotionGate,
    CycleNetYawRateCost,
    animated_walk_phase_symmetry,
    both_feet_contact,
    conservative_walk_phase_symmetry,
    exact_scheduled_foot_contact,
    forward_speed_success,
    full_domain_randomization,
    gait_clock,
    lateral_support_balance,
    mid_swing_clearance_success,
    motion_rate_safety_cost,
    pitch_dominant_angular_velocity,
    pitch_upright,
    periodic_foot_exchange,
    periodic_stance_foot_speed_cost,
    periodic_swing_force_cost,
    periodic_walk_evaluation_active,
    periodic_walk_success,
    scheduled_foot_contact,
    scheduled_stance_contact,
    scheduled_swing_clearance,
    scheduled_swing_unloading,
    scheduled_swing_unloading_success,
    single_support_active,
    standing_still,
    sagittal_velocity_success,
    step_in_place_success,
    track_animated_walk_reference,
    track_conservative_shuffle_reference,
    track_gait_reference,
    track_unloading_walk_reference,
    unloading_walk_phase_symmetry,
    episode_average_world_forward_speed,
    episode_average_world_progress,
    symmetric_domain_randomization,
    world_forward_velocity_tracking,
    weight_transfer_target,
    zero_yaw_rate_cost,
)
from locomotion.tasks.rewards import fixed_heading_error, lateral_path_deviation
from locomotion.task_runner import task_runner
from locomotion.tasks.leggybot import _leggybot_base_env_cfg, leggybot_ppo_runner_cfg

STAND_TASK_ID = "Mjlab-Stand-LeggyBot-FixedAnkle6DOF"
WEIGHT_TRANSFER_TASK_ID = "Mjlab-WeightTransfer-LeggyBot-FixedAnkle6DOF"
STEP_IN_PLACE_TASK_ID = "Mjlab-StepInPlace-LeggyBot-FixedAnkle6DOF"
ANIMATED_REFERENCE_WALK_TASK_ID = "Mjlab-AnimatedReferenceWalk-LeggyBot-FixedAnkle6DOF"
CONSERVATIVE_WALK_TASK_ID = "Mjlab-ConservativeWalk-LeggyBot-FixedAnkle6DOF"
UNLOADING_WALK_TASK_ID = "Mjlab-UnloadingWalk-LeggyBot-FixedAnkle6DOF"
PERIODIC_WALK_TASK_ID = "Mjlab-PeriodicWalk-LeggyBot-FixedAnkle6DOF"
STRAIGHT_PERIODIC_WALK_TASK_ID = "Mjlab-StraightPeriodicWalk-LeggyBot-FixedAnkle6DOF"
SYMMETRIC_CYCLE_PERIODIC_WALK_TASK_ID = "Mjlab-SymmetricCyclePeriodicWalk-LeggyBot-FixedAnkle6DOF"
SYMMETRIC_PERIODIC_WALK_TASK_ID = "Mjlab-SymmetricPeriodicWalk-LeggyBot-FixedAnkle6DOF"
SAGITTAL_TASK_ID = "Mjlab-Sagittal-LeggyBot-FixedAnkle6DOF"

LOCOMOTION_TASK_IDS = (
    STAND_TASK_ID,
    WEIGHT_TRANSFER_TASK_ID,
    STEP_IN_PLACE_TASK_ID,
    ANIMATED_REFERENCE_WALK_TASK_ID,
    CONSERVATIVE_WALK_TASK_ID,
    UNLOADING_WALK_TASK_ID,
    PERIODIC_WALK_TASK_ID,
    STRAIGHT_PERIODIC_WALK_TASK_ID,
    SYMMETRIC_CYCLE_PERIODIC_WALK_TASK_ID,
    SYMMETRIC_PERIODIC_WALK_TASK_ID,
    SAGITTAL_TASK_ID,
)

_GAIT_PARAMS = {"period": 0.8, "duty_factor": 0.6}
_REFERENCE = SceneEntityCfg(
    "robot", joint_names=(".*",), preserve_order=True
)
_FOOT_SITES = SceneEntityCfg(
    "robot", site_names=("right_foot", "left_foot"), preserve_order=True
)


@dataclass(frozen=True)
class LocomotionTask:
    """One task definition shared by training, evaluation, and documentation."""

    task_id: str
    environment_factory: object
    experiment_name: str
    success_metric: object
    active_metric: object
    success_threshold: float
    accepts_sagittal_command: bool = False
    use_symmetry_augmentation: bool = False


def _always_active(env: object) -> torch.Tensor:
    return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)


@dataclass(kw_only=True)
class DiscreteSagittalCommandCfg(UniformVelocityCommandCfg):
    """Command sampler with explicit train-time forward/backward speed bins."""

    bins: tuple[float, ...] = (-0.50, -0.30, -0.15, 0.15, 0.30, 0.50)

    def build(self, env):
        return DiscreteSagittalCommand(self, env)


class DiscreteSagittalCommand(UniformVelocityCommand):
    cfg: DiscreteSagittalCommandCfg

    def _resample_command(self, env_ids):
        if not self.cfg.bins:
            raise ValueError("sagittal command bins must not be empty")
        indices = torch.randint(
            len(self.cfg.bins), (len(env_ids),), device=self.device
        )
        bins = torch.as_tensor(self.cfg.bins, device=self.device)
        self.vel_command_b[env_ids, 0] = bins[indices]
        self.vel_command_b[env_ids, 1:] = 0.0
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = False
        self.is_world_env[env_ids] = False
        self.is_forward_env[env_ids] = False


def _command_cfg(
    x: tuple[float, float], y: tuple[float, float] = (0.0, 0.0),
    yaw: tuple[float, float] = (0.0, 0.0),
) -> UniformVelocityCommandCfg:
    return UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_world_envs=0.0,
        rel_forward_envs=0.0,
        heading_command=False,
        init_velocity_prob=0.0,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=x, lin_vel_y=y, ang_vel_z=yaw, heading=None
        ),
    )


def _shared_skill_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _leggybot_base_env_cfg(play=play)
    # These are focused hardware skills, not the deleted exploratory gait
    # curriculum. Keep the same normalized action ABI but expose the full
    # asymmetric safe joint limits from the first rollout.
    action_cfg = cfg.actions["joint_pos"]
    action_cfg.initial_range_fraction = 1.0
    action_cfg.range_ramp_steps = 0
    cfg.commands["twist"] = _command_cfg((0.0, 0.0))
    for group_name in ("actor", "critic"):
        cfg.observations[group_name].terms["gait_clock"] = ObservationTermCfg(
            # The clock itself only needs the period; duty factor belongs to
            # the contact/clearance terms below.
            func=gait_clock, params={"period": _GAIT_PARAMS["period"]}
        )

    # Keep a stable actor ABI across every skill, while applying sensor
    # corruption explicitly so each one tests robustness.
    from locomotion.tasks.gait import (
        sim2real_base_ang_vel,
        sim2real_joint_pos,
        sim2real_joint_vel,
        sim2real_projected_gravity,
    )

    actor = cfg.observations["actor"]
    actor.terms["base_ang_vel"].func = sim2real_base_ang_vel
    actor.terms["base_ang_vel"].params = {"sensor_name": "robot/imu_ang_vel"}
    actor.terms["projected_gravity"].func = sim2real_projected_gravity
    actor.terms["projected_gravity"].params = {}
    actor.terms["joint_pos"].func = sim2real_joint_pos
    actor.terms["joint_pos"].params = {}
    actor.terms["joint_vel"].func = sim2real_joint_vel
    actor.terms["joint_vel"].params = {}
    for term in actor.terms.values():
        term.noise = None
    for term in cfg.observations["critic"].terms.values():
        term.noise = None

    cfg.events["domain_randomization"] = EventTermCfg(
        func=full_domain_randomization, mode="reset"
    )
    # Small, repeatable disturbances test recovery without making the first
    # stand gate equivalent to a fall-recovery benchmark.
    cfg.events["push"] = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {
                "x": (-0.15, 0.15), "y": (-0.15, 0.15), "z": (-0.05, 0.05),
                "roll": (-0.20, 0.20), "pitch": (-0.20, 0.20), "yaw": (-0.20, 0.20),
            }
        },
    )
    cfg.manages_sim2real_randomization = True
    cfg.sim2real_randomization_enabled = True
    cfg.curriculum = {}
    if play:
        cfg.events.pop("domain_randomization", None)
        cfg.events.pop("push", None)
        cfg.sim2real_randomization_enabled = False
    cfg.episode_length_s = 20.0

    # Remove velocity-task terms which have no useful interpretation in a
    # stationary/periodic skill. Individual builders add back the
    # terms they need.
    for name in (
        "pose", "body_ang_vel", "air_time", "foot_clearance",
        "foot_swing_height", "soft_landing", "angular_momentum",
    ):
        cfg.rewards.pop(name, None)
    cfg.rewards["upright"].weight = 3.0
    # Stand/transfer stages have zero velocity commands; velocity tracking is
    # redundant there and otherwise dilutes the task-specific objective.
    cfg.rewards["track_linear_velocity"].weight = 0.0
    cfg.rewards["track_angular_velocity"].weight = 0.0
    cfg.rewards["dof_pos_limits"].weight = -1.0
    cfg.rewards["action_rate_l2"].weight = -0.02
    cfg.rewards["foot_slip"].weight = -0.05
    cfg.rewards["termination"] = RewardTermCfg(
        func=mdp.is_terminated, weight=-250.0
    )
    return cfg


def _add_gait_terms(cfg: ManagerBasedRlEnvCfg, *, velocity: bool) -> None:
    cfg.rewards["gait_contact"] = RewardTermCfg(
        func=scheduled_foot_contact,
        weight=2.0,
        params={"sensor_name": "feet_ground_contact", **_GAIT_PARAMS},
    )
    cfg.rewards["swing_clearance"] = RewardTermCfg(
        func=scheduled_swing_clearance,
        weight=2.0,
        params={
            "height_sensor_name": "foot_height_scan",
            "target_height": 0.020,
            "std": 0.012,
            **_GAIT_PARAMS,
        },
    )
    cfg.rewards["gait_reference"] = RewardTermCfg(
        func=track_gait_reference,
        weight=1.5,
        params={"std": 0.20, "asset_cfg": _REFERENCE, **_GAIT_PARAMS},
    )
    cfg.rewards["clearance_success"] = RewardTermCfg(
        func=mid_swing_clearance_success,
        weight=1.0e-6,
        params={"height_sensor_name": "foot_height_scan", **_GAIT_PARAMS},
    )
    if velocity:
        cfg.rewards["track_linear_velocity"].weight = 4.0
        cfg.rewards["track_linear_velocity"].params["std"] = 0.15
    else:
        cfg.rewards["track_linear_velocity"].weight = 0.0
    cfg.rewards["track_angular_velocity"].weight = 0.0


def _configure_walk_stability(cfg: ManagerBasedRlEnvCfg) -> None:
    """Favor torso pitch stability without making yaw or roll rigid."""
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["pitch_upright"] = RewardTermCfg(
        func=pitch_upright, weight=5.0, params={"std": 0.10}
    )
    cfg.rewards["uncommanded_base_ang_vel"] = RewardTermCfg(
        func=pitch_dominant_angular_velocity,
        weight=-0.20,
        params={"command_name": "twist"},
    )
    # A body-frame velocity objective alone permits a robot to earn forward
    # reward while tracing circles.  These are deliberately weaker than pitch
    # stability: modest gait yaw is allowed, but cumulative turning/drift is
    # not a valid straight-walking solution.
    cfg.rewards["fixed_heading_error"] = RewardTermCfg(
        func=fixed_heading_error, weight=-0.40,
        params={"target_heading": 0.0, "max_penalty": 1.0},
    )
    cfg.rewards["lateral_path_deviation"] = RewardTermCfg(
        func=lateral_path_deviation, weight=-0.30,
        params={"tolerance": 0.10, "max_penalty": 4.0},
    )


def _add_forward_speed_metric(cfg: ManagerBasedRlEnvCfg) -> None:
    cfg.metrics["speed_success"] = MetricsTermCfg(
        func=forward_speed_success, params={"command_name": "twist"}, reduce="mean"
    )


def _set_gait_clock_period(cfg: ManagerBasedRlEnvCfg, period: float) -> None:
    """Keep the deployable actor clock and privileged critic clock aligned."""
    if period <= 0.0:
        raise ValueError("gait clock period must be positive")
    for group_name in ("actor", "critic"):
        cfg.observations[group_name].terms["gait_clock"].params["period"] = period


def stand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _shared_skill_cfg(play)
    cfg.rewards["standing_still"] = RewardTermCfg(
        func=standing_still, weight=5.0,
        params={"linear_std": 0.08, "angular_std": 0.15},
    )
    cfg.metrics["stand_success"] = MetricsTermCfg(
        func=standing_still,
        params={"linear_std": 0.08, "angular_std": 0.15},
        reduce="mean",
    )
    return cfg


def weight_transfer_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _shared_skill_cfg(play)
    cfg.rewards["weight_transfer"] = RewardTermCfg(
        func=weight_transfer_target, weight=1.5,
        params={
            "amplitude": 0.14,
            "period": 2.0,
            "initial_amplitude": 0.0,
            "ramp_steps": 12_000,
        },
    )
    # Optimize the same physical quantity used by the promotion metric:
    # balanced support at the center and an 80/20 load split at extrema.
    cfg.rewards["load_transfer"] = RewardTermCfg(
        func=lateral_support_balance,
        weight=10.0,
        params={
            "sensor_name": "feet_ground_contact",
            "amplitude": 0.14,
            "period": 2.0,
            "target_imbalance": 0.60,
            "std": 0.12,
            # Start with balanced support, then ramp the desired split to
            # 80/20 over roughly 500 PPO iterations (24 env steps/update).
            "initial_imbalance": 0.0,
            "ramp_steps": 12_000,
        },
    )
    cfg.rewards["both_feet_contact"] = RewardTermCfg(
        # Keep bilateral contact as a safety constraint, but make it less
        # profitable than actually redistributing the loads.
        func=both_feet_contact, weight=1.0,
        params={"sensor_name": "feet_ground_contact"},
    )
    cfg.metrics["transfer_success"] = MetricsTermCfg(
        func=lateral_support_balance,
        params={
            "sensor_name": "feet_ground_contact",
            "amplitude": 0.14,
            "period": 2.0,
            "std": 0.12,
        },
        reduce="mean",
    )
    return cfg


def step_in_place_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _shared_skill_cfg(play)
    _add_gait_terms(cfg, velocity=False)
    cfg.rewards["step_in_place_success"] = RewardTermCfg(
        func=step_in_place_success, weight=1.0e-6,
        params={"height_sensor_name": "foot_height_scan", **_GAIT_PARAMS},
    )
    cfg.metrics["step_success"] = MetricsTermCfg(
        func=step_in_place_success,
        params={"height_sensor_name": "foot_height_scan", **_GAIT_PARAMS},
        reduce="mean",
    )
    return cfg


def animated_reference_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """PPO follows a conservative 3 Hz joint animation while learning balance."""
    cfg = _shared_skill_cfg(play)
    gait = {"period": 0.33, "duty_factor": 0.70}
    _set_gait_clock_period(cfg, gait["period"])
    cfg.commands["twist"] = _command_cfg((0.08, 0.08))
    cfg.rewards["track_linear_velocity"].weight = 5.0
    cfg.rewards["track_linear_velocity"].params["std"] = 0.05
    cfg.rewards["command_direction_progress"].weight = 1.5
    cfg.rewards["animated_reference"] = RewardTermCfg(
        func=track_animated_walk_reference,
        weight=7.0,
        params={"std": 0.08, "asset_cfg": _REFERENCE, **gait},
    )
    cfg.rewards["scheduled_contact"] = RewardTermCfg(
        func=scheduled_foot_contact,
        weight=1.5,
        params={"sensor_name": "feet_ground_contact", **gait},
    )
    cfg.rewards["gait_symmetry"] = RewardTermCfg(
        func=animated_walk_phase_symmetry,
        weight=0.30,
        params={"std": 0.08, "asset_cfg": _REFERENCE, **gait},
    )
    cfg.rewards["shallow_clearance"] = RewardTermCfg(
        func=scheduled_swing_clearance,
        weight=1.5,
        params={
            "height_sensor_name": "foot_height_scan",
            # Native forward kinematics places the reference peak about
            # 5--6 mm above nominal.  This is a clearance guard, not a
            # high-stepping objective.
            "target_height": 0.006,
            "std": 0.006,
            **gait,
        },
    )
    _configure_walk_stability(cfg)
    _add_forward_speed_metric(cfg)
    cfg.metrics["alternating_contact_success"] = MetricsTermCfg(
        func=exact_scheduled_foot_contact,
        params={"sensor_name": "feet_ground_contact", **gait},
        reduce="mean",
    )
    return cfg


def conservative_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Low-clearance, bilateral-contact-biased 0.08 m/s shuffling walk."""
    cfg = _shared_skill_cfg(play)
    gait = {"period": 0.33, "duty_factor": 0.82}
    _set_gait_clock_period(cfg, gait["period"])
    cfg.commands["twist"] = _command_cfg((0.08, 0.08))
    cfg.rewards["track_linear_velocity"].weight = 5.0
    cfg.rewards["track_linear_velocity"].params["std"] = 0.05
    cfg.rewards["command_direction_progress"].weight = 1.5
    cfg.rewards["conservative_reference"] = RewardTermCfg(
        func=track_conservative_shuffle_reference,
        weight=6.0,
        params={"std": 0.06, "asset_cfg": _REFERENCE, **gait},
    )
    # This is intentionally not scheduled-contact reward: the desired behavior
    # is retained bilateral support, not a high-stepping aerial swing phase.
    cfg.rewards["bilateral_contact"] = RewardTermCfg(
        func=both_feet_contact,
        weight=3.0,
        params={"sensor_name": "feet_ground_contact"},
    )
    cfg.rewards["gait_symmetry"] = RewardTermCfg(
        func=conservative_walk_phase_symmetry,
        weight=0.30,
        params={"std": 0.06, "asset_cfg": _REFERENCE, **gait},
    )
    cfg.rewards["shallow_clearance"] = RewardTermCfg(
        func=scheduled_swing_clearance,
        weight=0.75,
        params={
            "height_sensor_name": "foot_height_scan",
            "target_height": 0.003,
            "std": 0.004,
            **gait,
        },
    )
    _configure_walk_stability(cfg)
    _add_forward_speed_metric(cfg)
    cfg.metrics["bilateral_contact_success"] = MetricsTermCfg(
        func=both_feet_contact,
        params={"sensor_name": "feet_ground_contact"},
        reduce="mean",
    )
    return cfg


def unloading_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Transition from a stable shuffle to scheduled pitch-joint unloading."""
    cfg = _shared_skill_cfg(play)
    gait = {"period": 0.33, "duty_factor": 0.67}
    _set_gait_clock_period(cfg, gait["period"])
    cfg.commands["twist"] = _command_cfg((0.08, 0.08))
    cfg.rewards["track_linear_velocity"].weight = 5.0
    cfg.rewards["track_linear_velocity"].params["std"] = 0.05
    cfg.rewards["command_direction_progress"].weight = 1.5
    cfg.rewards["unloading_reference"] = RewardTermCfg(
        func=track_unloading_walk_reference,
        weight=7.0,
        params={"std": 0.09, "asset_cfg": _REFERENCE, **gait},
    )
    cfg.rewards["gait_symmetry"] = RewardTermCfg(
        func=unloading_walk_phase_symmetry,
        weight=0.30,
        params={"std": 0.09, "asset_cfg": _REFERENCE, **gait},
    )
    # Keep both soles near the ground, but demand that the scheduled support
    # foot carries most of the load.  This creates an unload--move--reload
    # sequence without making high airborne clearance profitable.
    cfg.rewards["scheduled_stance_contact"] = RewardTermCfg(
        func=scheduled_stance_contact,
        weight=1.5,
        params={"sensor_name": "feet_ground_contact", **gait},
    )
    cfg.rewards["scheduled_unloading"] = RewardTermCfg(
        func=scheduled_swing_unloading,
        weight=5.0,
        params={
            "sensor_name": "feet_ground_contact",
            "stance_load_fraction": 0.75,
            "std": 0.14,
            **gait,
        },
    )
    cfg.rewards["shallow_clearance"] = RewardTermCfg(
        func=scheduled_swing_clearance,
        weight=0.75,
        params={
            "height_sensor_name": "foot_height_scan",
            "target_height": 0.004,
            "std": 0.005,
            **gait,
        },
    )
    _configure_walk_stability(cfg)
    _add_forward_speed_metric(cfg)
    cfg.metrics["unloading_success"] = MetricsTermCfg(
        func=scheduled_swing_unloading_success,
        params={
            "sensor_name": "feet_ground_contact",
            "stance_load_fraction": 0.75,
            "std": 0.14,
            "minimum_score": 0.70,
            **gait,
        },
        reduce="mean",
    )
    cfg.metrics["scheduled_stance_contact_success"] = MetricsTermCfg(
        func=scheduled_stance_contact,
        params={"sensor_name": "feet_ground_contact", **gait},
        reduce="mean",
    )
    return cfg


def periodic_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """True walking from periodic reward composition, adapted to LeggyBot.

    There is intentionally no joint-space animation and no clearance-height
    objective. A smooth phase schedule only says when a foot should be light
    (swing) and when it should be still on the ground (stance). World-frame
    velocity plus episode-average displacement make stepping necessary.
    """
    cfg = _shared_skill_cfg(play)
    gait = {"period": 0.33, "duty_factor": 0.70, "transition_fraction": 0.04}
    _set_gait_clock_period(cfg, gait["period"])
    cfg.commands["twist"] = _command_cfg((0.08, 0.08))

    # Remove the body-frame terms that let the old tasks earn progress while
    # rotating or rocking in place.  The new terms below are world-frame and
    # explicitly use net episode displacement.
    cfg.rewards["track_linear_velocity"].weight = 0.0
    cfg.rewards["command_direction_progress"].weight = 0.0
    cfg.rewards["foot_slip"].weight = 0.0
    cfg.rewards["track_angular_velocity"].weight = 0.0

    # Paper-derived periodic composition: discourage foot force only in swing,
    # and horizontal foot motion only in stance. Both are normalized over their
    # respective phase mass so long double-support intervals cannot hide error.
    cfg.rewards["swing_foot_force"] = RewardTermCfg(
        func=periodic_swing_force_cost,
        weight=-3.0,
        params={"sensor_name": "feet_ground_contact", "force_scale": 4.0, **gait},
    )
    cfg.rewards["stance_foot_speed"] = RewardTermCfg(
        func=periodic_stance_foot_speed_cost,
        weight=-2.0,
        params={"speed_scale": 0.06, "asset_cfg": _FOOT_SITES, **gait},
    )

    # The previous policies scored body-frame velocity. These two terms require
    # the base to travel along the laboratory X direction now and on average
    # across the whole episode, eliminating circular shuffles and transient
    # momentum tricks.
    cfg.rewards["world_velocity"] = RewardTermCfg(
        func=world_forward_velocity_tracking,
        weight=4.0,
        params={"command_name": "twist", "std": 0.05},
    )
    cfg.rewards["world_progress"] = RewardTermCfg(
        func=episode_average_world_progress,
        weight=5.0,
        params={"command_name": "twist", "std": 0.035, "warmup_s": 0.50},
    )
    cfg.rewards["foot_exchange"] = RewardTermCfg(
        func=periodic_foot_exchange,
        weight=0.75,
        params={
            "period": gait["period"], "separation": 0.012, "std": 0.018,
            "asset_cfg": _FOOT_SITES,
        },
    )
    cfg.rewards["actuator_effort"] = RewardTermCfg(
        func=actuator_effort_cost, weight=-0.08, params={"effort_scale": 1.5},
    )

    _configure_walk_stability(cfg)
    # Net displacement requires heading and path control to be appreciably
    # stronger than in the exploratory reference tasks, while pitch remains
    # the primary body-orientation constraint.
    cfg.rewards["fixed_heading_error"].weight = -1.0
    cfg.rewards["lateral_path_deviation"].weight = -0.75
    cfg.rewards["uncommanded_base_ang_vel"].weight = -0.12

    cfg.metrics["average_world_forward_speed"] = MetricsTermCfg(
        func=episode_average_world_forward_speed, reduce="mean"
    )
    cfg.metrics["swing_foot_force_cost"] = MetricsTermCfg(
        func=periodic_swing_force_cost,
        params={"sensor_name": "feet_ground_contact", "force_scale": 4.0, **gait},
        reduce="mean",
    )
    cfg.metrics["stance_foot_speed_cost"] = MetricsTermCfg(
        func=periodic_stance_foot_speed_cost,
        params={"speed_scale": 0.06, "asset_cfg": _FOOT_SITES, **gait}, reduce="mean",
    )
    cfg.metrics["foot_exchange"] = MetricsTermCfg(
        func=periodic_foot_exchange,
        params={
            "period": gait["period"], "separation": 0.012, "std": 0.018,
            "asset_cfg": _FOOT_SITES,
        },
        reduce="mean",
    )
    cfg.metrics["periodic_walk_success"] = MetricsTermCfg(
        func=periodic_walk_success,
        params={
            "sensor_name": "feet_ground_contact", "command_name": "twist",
            "foot_site_cfg": _FOOT_SITES, **gait,
        },
        reduce="mean",
    )
    return cfg


def symmetric_periodic_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Periodic walking with mirrored PPO updates and cycle-level straightness.

    This is intentionally a narrow evolution of :func:`periodic_walk_env_cfg`.
    It preserves the same robot, 0.33 s phase schedule, command, contact terms,
    action ABI, and evaluation criterion.  The changes only remove the known
    asymmetric curved-walk local optimum.
    """
    cfg = periodic_walk_env_cfg(play)
    gait = {"period": 0.33, "duty_factor": 0.70, "transition_fraction": 0.04}
    cycle_straightness = {
        "period": gait["period"],
        "heading_std": 0.10,
        "cycle_lateral_velocity_std": 0.020,
        "cycle_yaw_rate_std": 0.35,
    }

    # A phase-specific contact objective remains the source of stepping.  Net
    # lateral and yaw drift over a complete cycle gate forward credit, leaving
    # the instantaneous lateral motion needed for support transfer available.
    cfg.rewards["world_velocity"] = RewardTermCfg(
        func=CycleGatedWorldForwardVelocity,
        weight=4.0,
        params={"command_name": "twist", "std": 0.05, **cycle_straightness},
    )
    cfg.rewards["world_progress"] = RewardTermCfg(
        func=CycleGatedEpisodeWorldProgress,
        weight=5.0,
        params={
            "command_name": "twist", "std": 0.035, "warmup_s": 0.50,
            **cycle_straightness,
        },
    )
    # Foot exchange alone can be achieved by moving just one leg.  Preserve it
    # as a small shaping signal, while PPO augmentation supplies bilateral bias.
    cfg.rewards["foot_exchange"].weight = 0.10
    cfg.rewards.pop("world_lateral_velocity", None)
    cfg.rewards["cycle_lateral_drift"] = RewardTermCfg(
        func=CycleNetLateralVelocityCost,
        weight=-0.10,
        params={
            "period": gait["period"],
            "std": cycle_straightness["cycle_lateral_velocity_std"],
            "max_cost": 4.0,
        },
    )
    cfg.rewards["zero_yaw_rate"] = RewardTermCfg(
        func=zero_yaw_rate_cost, weight=-0.25,
        params={"std": cycle_straightness["cycle_yaw_rate_std"], "max_cost": 4.0},
    )
    # Keep pitch as the dominant stability requirement.  The dedicated yaw
    # term above strengthens only the uncommanded yaw component.
    cfg.rewards["fixed_heading_error"].weight = -1.0
    cfg.rewards["lateral_path_deviation"].weight = -0.75

    if not play:
        cfg.events["domain_randomization"] = EventTermCfg(
            func=symmetric_domain_randomization, mode="reset"
        )

    cfg.metrics["cycle_straightness_gate"] = MetricsTermCfg(
        func=CycleNetMotionGate, params=cycle_straightness, reduce="mean"
    )
    cfg.metrics["cycle_lateral_velocity_cost"] = MetricsTermCfg(
        func=CycleNetLateralVelocityCost,
        params={
            "period": gait["period"],
            "std": cycle_straightness["cycle_lateral_velocity_std"],
            "max_cost": 4.0,
        },
        reduce="mean",
    )
    cfg.metrics["cycle_yaw_rate_cost"] = MetricsTermCfg(
        func=CycleNetYawRateCost,
        params={
            "period": gait["period"],
            "std": cycle_straightness["cycle_yaw_rate_std"],
            "max_cost": 4.0,
        },
        reduce="mean",
    )
    cfg.metrics["zero_yaw_rate_cost"] = MetricsTermCfg(
        func=zero_yaw_rate_cost,
        params={"std": cycle_straightness["cycle_yaw_rate_std"], "max_cost": 4.0},
        reduce="mean",
    )
    return cfg


def straight_periodic_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Periodic walking that gives forward credit only to a straight path.

    This is the minimal correction to :func:`periodic_walk_env_cfg` for a
    policy that learned to walk while accumulating world-Y displacement and
    yaw. It deliberately does *not* use PPO symmetry augmentation or altered
    domain randomization: the phase terms, action space, contact objectives,
    and every original reward are retained.

    The existing path-displacement penalty remains useful but is tradeable.
    The two forward rewards below are not: *net* lateral displacement and yaw
    accumulated across a complete gait cycle remove their forward credit.
    Instantaneous lateral CoM transfer and counter-yaw within a cycle are
    normal biped motion, so they are left unpenalized unless unusually large.
    """
    cfg = periodic_walk_env_cfg(play)
    straightness = {
        # These are intentionally less abrupt than the later symmetry task so
        # an already-walking PPO policy can adapt rather than losing all
        # forward reward at the first update.
        "period": 0.33,
        "heading_std": 0.15,
        "cycle_lateral_velocity_std": 0.020,
        "cycle_yaw_rate_std": 0.25,
    }
    cfg.rewards["world_velocity"] = RewardTermCfg(
        func=CycleGatedWorldForwardVelocity,
        weight=4.0,
        params={"command_name": "twist", "std": 0.05, **straightness},
    )
    cfg.rewards["world_progress"] = RewardTermCfg(
        func=CycleGatedEpisodeWorldProgress,
        weight=5.0,
        params={"command_name": "twist", "std": 0.035, "warmup_s": 0.50, **straightness},
    )
    cfg.rewards.pop("world_lateral_velocity", None)
    cfg.rewards["cycle_lateral_drift"] = RewardTermCfg(
        func=CycleNetLateralVelocityCost,
        weight=-0.10,
        params={"period": 0.33, "std": 0.020, "max_cost": 4.0},
    )
    cfg.rewards["motion_rate_safety"] = RewardTermCfg(
        func=motion_rate_safety_cost,
        weight=-0.03,
        params={"command_name": "twist"},
    )
    cfg.metrics["cycle_straightness_gate"] = MetricsTermCfg(
        func=CycleNetMotionGate, params=straightness, reduce="mean"
    )
    cfg.metrics["cycle_lateral_velocity_cost"] = MetricsTermCfg(
        func=CycleNetLateralVelocityCost,
        params={"period": 0.33, "std": 0.020, "max_cost": 4.0},
        reduce="mean",
    )
    cfg.metrics["cycle_yaw_rate_cost"] = MetricsTermCfg(
        func=CycleNetYawRateCost,
        params={"period": 0.33, "std": 0.25, "max_cost": 4.0},
        reduce="mean",
    )
    cfg.metrics["motion_rate_safety_cost"] = MetricsTermCfg(
        func=motion_rate_safety_cost,
        params={"command_name": "twist"},
        reduce="mean",
    )
    return cfg


def symmetric_cycle_periodic_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Cycle-gated straight walking with symmetry-focused pretraining."""
    cfg = straight_periodic_walk_env_cfg(play)
    if not play:
        cfg.events["domain_randomization"] = EventTermCfg(
            func=symmetric_domain_randomization, mode="reset"
        )
    return cfg


def sagittal_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _shared_skill_cfg(play)
    cfg.commands["twist"] = DiscreteSagittalCommandCfg(
        entity_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_world_envs=0.0,
        rel_forward_envs=0.0,
        heading_command=False,
        init_velocity_prob=0.0,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.50, 0.50), lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0), heading=None,
        ),
    )
    _add_gait_terms(cfg, velocity=True)
    cfg.rewards["sagittal_success"] = RewardTermCfg(
        func=sagittal_velocity_success, weight=1.0e-6,
        params={"command_name": "twist"},
    )
    cfg.metrics["sagittal_success"] = MetricsTermCfg(
        func=sagittal_velocity_success, params={"command_name": "twist"}, reduce="mean"
    )
    return cfg


def locomotion_tasks() -> tuple[LocomotionTask, ...]:
    """Return all registered LeggyBot skills without a separate task category."""
    return (
        LocomotionTask(
            STAND_TASK_ID, stand_env_cfg, "leggybot_skill_stand",
            lambda env: standing_still(env), _always_active, 0.85,
        ),
        LocomotionTask(
            WEIGHT_TRANSFER_TASK_ID, weight_transfer_env_cfg, "leggybot_skill_transfer",
            lambda env: lateral_support_balance(env, "feet_ground_contact"),
            _always_active, 0.75,
        ),
        LocomotionTask(
            STEP_IN_PLACE_TASK_ID, step_in_place_env_cfg, "leggybot_skill_step",
            lambda env: step_in_place_success(
                env, "foot_height_scan", period=0.8, duty_factor=0.6
            ),
            lambda env: mid_swing_active(env, period=0.8, duty_factor=0.6),
            0.50,
        ),
        LocomotionTask(
            ANIMATED_REFERENCE_WALK_TASK_ID, animated_reference_walk_env_cfg,
            "leggybot_skill_animated_reference_walk",
            lambda env: forward_speed_success(env, "twist")
            * exact_scheduled_foot_contact(
                env, "feet_ground_contact", period=0.33, duty_factor=0.70
            ),
            lambda env: single_support_active(env, period=0.33, duty_factor=0.70),
            0.70,
        ),
        LocomotionTask(
            CONSERVATIVE_WALK_TASK_ID, conservative_walk_env_cfg,
            "leggybot_skill_conservative_walk",
            lambda env: forward_speed_success(env, "twist")
            * both_feet_contact(env, "feet_ground_contact"),
            _always_active,
            0.70,
        ),
        LocomotionTask(
            UNLOADING_WALK_TASK_ID, unloading_walk_env_cfg,
            "leggybot_skill_unloading_walk",
            lambda env: forward_speed_success(env, "twist")
            * scheduled_swing_unloading_success(
                env,
                "feet_ground_contact",
                period=0.33,
                duty_factor=0.67,
                stance_load_fraction=0.75,
                std=0.14,
                minimum_score=0.70,
            ),
            lambda env: single_support_active(env, period=0.33, duty_factor=0.67),
            0.65,
        ),
        LocomotionTask(
            PERIODIC_WALK_TASK_ID, periodic_walk_env_cfg,
            "leggybot_skill_periodic_walk",
            lambda env: periodic_walk_success(
                env,
                "feet_ground_contact",
                "twist",
                period=0.33,
                duty_factor=0.70,
                foot_site_cfg=_FOOT_SITES,
            ),
            periodic_walk_evaluation_active,
            0.65,
        ),
        LocomotionTask(
            STRAIGHT_PERIODIC_WALK_TASK_ID, straight_periodic_walk_env_cfg,
            "leggybot_skill_straight_periodic_walk",
            lambda env: periodic_walk_success(
                env,
                "feet_ground_contact",
                "twist",
                period=0.33,
                duty_factor=0.70,
                foot_site_cfg=_FOOT_SITES,
            ),
            periodic_walk_evaluation_active,
            0.65,
        ),
        LocomotionTask(
            SYMMETRIC_CYCLE_PERIODIC_WALK_TASK_ID,
            symmetric_cycle_periodic_walk_env_cfg,
            "leggybot_skill_symmetric_cycle_periodic_walk",
            lambda env: periodic_walk_success(
                env,
                "feet_ground_contact",
                "twist",
                period=0.33,
                duty_factor=0.70,
                foot_site_cfg=_FOOT_SITES,
            ),
            periodic_walk_evaluation_active,
            0.65,
            use_symmetry_augmentation=True,
        ),
        LocomotionTask(
            SYMMETRIC_PERIODIC_WALK_TASK_ID, symmetric_periodic_walk_env_cfg,
            "leggybot_skill_symmetric_periodic_walk",
            lambda env: periodic_walk_success(
                env,
                "feet_ground_contact",
                "twist",
                period=0.33,
                duty_factor=0.70,
                foot_site_cfg=_FOOT_SITES,
            ),
            periodic_walk_evaluation_active,
            0.65,
            use_symmetry_augmentation=True,
        ),
        LocomotionTask(
            SAGITTAL_TASK_ID, sagittal_env_cfg, "leggybot_skill_sagittal",
            lambda env: sagittal_velocity_success(env, "twist"), _always_active,
            0.70, accepts_sagittal_command=True,
        ),
    )


def get_locomotion_task(task_id: str) -> LocomotionTask:
    for task in locomotion_tasks():
        if task.task_id == task_id:
            return task
    raise ValueError(f"No implemented locomotion task is named {task_id!r}")


def register_leggybot_skill_tasks() -> None:
    registered = set(list_tasks())
    for task in locomotion_tasks():
        if task.task_id in registered:
            continue
        register_mjlab_task(
            task_id=task.task_id,
            env_cfg=task.environment_factory(),
            play_env_cfg=task.environment_factory(play=True),
            rl_cfg=leggybot_ppo_runner_cfg(
                task.experiment_name,
                use_symmetry_augmentation=task.use_symmetry_augmentation,
            ),
            runner_cls=task_runner(task.task_id),
        )


__all__ = [
    "ANIMATED_REFERENCE_WALK_TASK_ID", "CONSERVATIVE_WALK_TASK_ID",
    "SAGITTAL_TASK_ID", "STAND_TASK_ID", "LocomotionTask",
    "STEP_IN_PLACE_TASK_ID", "LOCOMOTION_TASK_IDS", "UNLOADING_WALK_TASK_ID",
    "PERIODIC_WALK_TASK_ID",
    "STRAIGHT_PERIODIC_WALK_TASK_ID",
    "SYMMETRIC_PERIODIC_WALK_TASK_ID",
    "WEIGHT_TRANSFER_TASK_ID", "animated_reference_walk_env_cfg", "conservative_walk_env_cfg",
    "get_locomotion_task", "locomotion_tasks",
    "periodic_walk_env_cfg", "register_leggybot_skill_tasks", "sagittal_env_cfg", "straight_periodic_walk_env_cfg", "symmetric_cycle_periodic_walk_env_cfg", "symmetric_periodic_walk_env_cfg", "unloading_walk_env_cfg",
    "stand_env_cfg", "step_in_place_env_cfg", "weight_transfer_env_cfg",
]
