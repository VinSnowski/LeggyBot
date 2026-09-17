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
from mjlab.managers.curriculum_manager import CurriculumTermCfg
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
    commanded_periodic_foot_exchange,
    commanded_neutral_stance,
    commanded_periodic_stance_foot_speed_cost,
    commanded_periodic_swing_force_cost,
    commanded_periodic_reward_composition_stance_speed_cost,
    commanded_periodic_reward_composition_swing_force_cost,
    commanded_scheduled_foot_contact,
    CommandedCycleDisplacementCost,
    CommandedCycleDisplacementError,
    CommandedCyclePlanarVelocityCost,
    CommandedEpisodeDisplacementCost,
    CommandedCycleTrackingGate,
    CommandedPlanarPathCost,
    CommandedPlanarPathError,
    CommandedPlanarVelocityError,
    CycleGatedEpisodeWorldProgress,
    CycleGatedWorldForwardVelocity,
    CycleNetLateralVelocityCost,
    CycleNetMotionGate,
    CycleNetYawRateCost,
    UpdatedDefaultCycleLateralVelocityCost,
    UpdatedDefaultCycleStraightnessGate,
    UpdatedDefaultCycleYawRateCost,
    animated_walk_phase_symmetry,
    both_feet_contact,
    conservative_walk_phase_symmetry,
    commanded_motion_rate_safety_cost,
    exact_scheduled_foot_contact,
    forward_speed_success,
    full_domain_randomization,
    gait_clock,
    lateral_support_balance,
    lateral_velocity_command_range_curriculum,
    lateral_command_active,
    lateral_foot_line_error,
    commanded_lateral_foot_line_cost,
    instantaneous_world_command_velocity_cost,
    mid_swing_clearance_success,
    motion_rate_safety_cost,
    base_pitch_magnitude,
    constant_alive_reward,
    PeriodicRewardCompositionPelvisMotionCost,
    periodic_reward_composition_action_difference_cost,
    periodic_reward_composition_orientation_cost,
    periodic_reward_composition_torque_cost,
    periodic_reward_composition_velocity_cost,
    pitch_dominant_angular_velocity,
    pitch_upright,
    periodic_foot_exchange,
    periodic_stance_foot_speed_cost,
    periodic_swing_force_cost,
    periodic_walk_evaluation_active,
    periodic_walk_success,
    quasi_omnidirectional_command_curriculum,
    ForwardSpeedGateCurriculum,
    StagedQuasiOmnidirectionalCurriculum,
    quasi_omnidirectional_command_success,
    quasi_omnidirectional_velocity_success,
    scheduled_foot_contact,
    scheduled_stance_contact,
    scheduled_swing_clearance,
    scheduled_swing_unloading,
    scheduled_swing_unloading_success,
    single_support_active,
    standing_still,
    stopped_foot_alignment_cost,
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
    updated_default_domain_randomization,
    updated_default_lateral_path_cost,
    updated_default_walking_success,
    updated_default_lateral_walking_success,
    velocity_command_range_curriculum,
    world_forward_velocity_tracking,
    world_forward_velocity_only,
    weight_transfer_target,
    zero_yaw_rate_cost,
)
from locomotion.tasks.rewards import (
    GradualBoundedHeadingDriftCost,
    UnwrappedFixedHeadingError,
    UnwrappedHeadingMagnitude,
    fixed_heading_error,
    lateral_path_deviation,
)
from locomotion.task_runner import task_runner
from locomotion.tasks.omni_tracking import WindowedSE2Tracking, tracking_success
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
UPDATED_DEFAULT_WALKING_TASK_ID = "UpdatedDefaultWalkingTask"
UPDATED_DEFAULT_LATERAL_WALKING_TASK_ID = "UpdatedDefaultLateralWalkingTask"
QUASI_OMNIDIRECTIONAL_WALKING_TASK_ID = "QuasiOmnidirectionalWalkingTask"
QUASI_OMNIDIRECTIONAL_WALKING_V2_TASK_ID = "QuasiOmnidirectionalWalkingTaskV2"
SE2_TRACKING_TASK_ID = "SE2TrackingWalkingTask"
NATURAL_OMNIDIRECTIONAL_TASK_ID = "NaturalOmnidirectionalWalkingTask"
NATURAL_OMNIDIRECTIONAL_V3_TASK_ID = "NaturalOmnidirectionalWalkingTaskV3"
NATURAL_OMNIDIRECTIONAL_V4_TASK_ID = "NaturalOmnidirectionalWalkingTaskV4"
NATURAL_OMNIDIRECTIONAL_V5_TASK_ID = "NaturalOmnidirectionalWalkingTaskV5"
PERIODIC_REWARD_COMPOSITION_WALKING_TASK_ID = "PeriodicRewardCompositionWalkingTask"

# RSL-RL's LeggyBot runner collects 24 environment steps per PPO iteration.
# V3 keeps the physical distribution paired for 500 iterations, then ramps
# the full independent randomization envelope over the next 300 iterations.
_NATURAL_V3_SYMMETRIC_STEPS = 500 * 24
_NATURAL_V3_MISMATCH_RAMP_STEPS = 300 * 24

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
    UPDATED_DEFAULT_WALKING_TASK_ID,
    UPDATED_DEFAULT_LATERAL_WALKING_TASK_ID,
    QUASI_OMNIDIRECTIONAL_WALKING_TASK_ID,
    QUASI_OMNIDIRECTIONAL_WALKING_V2_TASK_ID,
    SE2_TRACKING_TASK_ID,
    NATURAL_OMNIDIRECTIONAL_TASK_ID,
    NATURAL_OMNIDIRECTIONAL_V3_TASK_ID,
    NATURAL_OMNIDIRECTIONAL_V4_TASK_ID,
    NATURAL_OMNIDIRECTIONAL_V5_TASK_ID,
    PERIODIC_REWARD_COMPOSITION_WALKING_TASK_ID,
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
    symmetry_mirror_loss_coeff: float = 0.0
    bounded_exploration: bool = False
    exploration_init_std: float | None = None
    exploration_std_range: tuple[float, float] = (0.03, 0.40)
    learning_rate: float = 1e-3


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


@dataclass(kw_only=True)
class LateralVelocityCommandCfg(UniformVelocityCommandCfg):
    zero_lateral_probability: float = 0.35

    def build(self, env):
        return LateralVelocityCommand(self, env)


class LateralVelocityCommand(UniformVelocityCommand):
    cfg: LateralVelocityCommandCfg

    def _resample_command(self, env_ids):
        if not 0.0 <= self.cfg.zero_lateral_probability <= 1.0:
            raise ValueError("zero lateral probability must be in [0, 1]")
        super()._resample_command(env_ids)
        preserve = torch.rand(len(env_ids), device=self.device) < (
            self.cfg.zero_lateral_probability
        )
        preserve_ids = env_ids[preserve]
        self.vel_command_b[preserve_ids, 1] = 0.0
        self.vel_command_w[preserve_ids, 1] = 0.0


@dataclass(kw_only=True)
class QuasiOmnidirectionalCommandCfg(UniformVelocityCommandCfg):
    zero_forward_probability: float = 0.20
    zero_lateral_probability: float = 0.20
    zero_yaw_probability: float = 0.20

    def build(self, env):
        return QuasiOmnidirectionalCommand(self, env)


class QuasiOmnidirectionalCommand(UniformVelocityCommand):
    cfg: QuasiOmnidirectionalCommandCfg

    def _resample_command(self, env_ids):
        probabilities = (
            self.cfg.zero_forward_probability,
            self.cfg.zero_lateral_probability,
            self.cfg.zero_yaw_probability,
        )
        if any(not 0.0 <= probability <= 1.0 for probability in probabilities):
            raise ValueError("zero-command probabilities must be in [0, 1]")
        super()._resample_command(env_ids)
        random = torch.rand((len(env_ids), 3), device=self.device)
        zero_forward = random[:, 0] < self.cfg.zero_forward_probability
        zero_lateral = random[:, 1] < self.cfg.zero_lateral_probability
        self.vel_command_b[env_ids[zero_forward], 0] = 0.0
        self.vel_command_b[env_ids[zero_lateral], 1] = 0.0

        forward_limit = max(abs(value) for value in self.cfg.ranges.lin_vel_x)
        yaw_limit = max(abs(value) for value in self.cfg.ranges.ang_vel_z)
        if forward_limit == 0.0 or yaw_limit == 0.0:
            conditional_yaw_limit = torch.zeros(len(env_ids), device=self.device)
        else:
            conditional_yaw_limit = (
                yaw_limit
                * torch.abs(self.vel_command_b[env_ids, 0])
                / forward_limit
            )
        yaw_sample = 2.0 * random[:, 2] - 1.0
        yaw_sample *= conditional_yaw_limit
        zero_yaw = torch.rand(len(env_ids), device=self.device) < (
            self.cfg.zero_yaw_probability
        )
        yaw_sample[zero_yaw] = 0.0
        self.vel_command_b[env_ids, 2] = yaw_sample
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = False
        self.is_world_env[env_ids] = False
        self.is_forward_env[env_ids] = False


@dataclass(kw_only=True)
class QuasiOmnidirectionalMixtureCommandCfg(UniformVelocityCommandCfg):
    """Deployment-weighted command modes for omnidirectional walking."""

    forward_straight_probability: float = 0.30
    forward_curve_probability: float = 0.25
    pure_lateral_probability: float = 0.15
    full_omni_probability: float = 0.20
    stop_probability: float = 0.05
    reverse_probability: float = 0.05
    protected_forward_range: tuple[float, float] = (0.04, 0.10)
    minimum_coupled_yaw_fraction: float = 0.25
    scale_yaw_with_forward_speed: bool = True
    minimum_lateral_speed: float = 0.0

    def build(self, env):
        return QuasiOmnidirectionalMixtureCommand(self, env)


class QuasiOmnidirectionalMixtureCommand(UniformVelocityCommand):
    """Sample a deployment-weighted mixture of command modes.

    The mode probabilities are explicit so straight-forward walking is not
    diluted by the volume of the full command cube.  Full-omni and curve modes
    retain the original yaw/forward coupling, which prevents large in-place
    yaw commands while preserving simultaneous ``vx``, ``vy`` and yaw motion.
    """

    cfg: QuasiOmnidirectionalMixtureCommandCfg

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

        # Initialize the base command bookkeeping (heading/world/forward flags)
        # and then replace the sampled velocity with the selected mode.
        super()._resample_command(env_ids)
        count = len(env_ids)
        modes = torch.multinomial(probabilities, count, replacement=True)
        command = torch.zeros((count, 3), device=self.device)

        x_low, x_high = (float(v) for v in self.cfg.ranges.lin_vel_x)
        y_low, y_high = (float(v) for v in self.cfg.ranges.lin_vel_y)
        yaw_limit = max(
            abs(float(self.cfg.ranges.ang_vel_z[0])),
            abs(float(self.cfg.ranges.ang_vel_z[1])),
        )
        protected_low, protected_high = self.cfg.protected_forward_range
        if protected_low < 0.0 or protected_low > protected_high:
            raise ValueError("protected forward range must be ordered and non-negative")
        forward_limit = max(abs(x_low), abs(x_high), abs(protected_low), abs(protected_high))

        def uniform(low: float, high: float) -> torch.Tensor:
            values = torch.empty(count, device=self.device)
            return values.uniform_(low, high)

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
            return (
                (2.0 * torch.randint(0, 2, (count,), device=self.device).float() - 1.0)
                * yaw_limit
                * speed_scale
                * magnitude
            )

        straight = modes == self._FORWARD_STRAIGHT
        if torch.any(straight):
            forward_low, forward_high = self.cfg.protected_forward_range
            command[straight, 0] = uniform(
                max(0.0, forward_low), max(0.0, forward_high)
            )[straight]

        curve = modes == self._FORWARD_CURVE
        if torch.any(curve):
            forward_low, forward_high = self.cfg.protected_forward_range
            curve_vx = uniform(max(0.0, forward_low), max(0.0, forward_high))
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
                sign = 2.0 * torch.randint(0, 2, (count,), device=self.device) - 1.0
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
            command[reverse, 0] = uniform(min(x_low, 0.0), min(x_high, 0.0))[reverse]

        self.vel_command_b[env_ids] = command
        self.vel_command_w[env_ids] = command
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
    # curriculum. Keep the same normalized action ABI and full asymmetric safe
    # joint limits in both training and playback.
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
        Sim2RealImuAngularVelocity,
        Sim2RealJointPosition,
        Sim2RealJointVelocity,
        Sim2RealImuProjectedGravity,
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
    cfg.rewards["fixed_heading_error"].func = UnwrappedFixedHeadingError
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
    cfg.rewards["fixed_heading_error"].func = UnwrappedFixedHeadingError
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


def updated_default_walking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Standalone long-horizon walking task with a velocity curriculum."""
    cfg = _shared_skill_cfg(play)
    gait = {"period": 0.33, "duty_factor": 0.70, "transition_fraction": 0.04}
    _set_gait_clock_period(cfg, gait["period"])
    velocity_range = {
        "initial_range": (0.075, 0.085),
        "final_range": (0.0, 0.1),
        "ramp_steps": 12_000,
    }
    cfg.commands["twist"] = _command_cfg(
        velocity_range["final_range"] if play else velocity_range["initial_range"]
    )
    cfg.episode_length_s = 1.0e9 if play else 60.0

    cycle = {
        "period": gait["period"],
        "initial_std": 0.08,
        "final_std": 0.02,
        "initial_yaw_std": 0.45,
        "final_yaw_std": 0.12,
        "ramp_steps": 12_000,
        "max_cost": 4.0,
    }

    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "command_direction_progress",
        "foot_slip",
        "world_progress",
        "fixed_heading_error",
        "lateral_path_deviation",
    ):
        cfg.rewards.pop(name, None)

    cfg.rewards["world_velocity"] = RewardTermCfg(
        func=world_forward_velocity_only,
        weight=4.0,
        params={"command_name": "twist", "std": 0.05},
    )
    cfg.rewards["scheduled_contact"] = RewardTermCfg(
        func=scheduled_foot_contact,
        weight=1.25,
        params={
            "sensor_name": "feet_ground_contact",
            "period": gait["period"],
            "duty_factor": gait["duty_factor"],
        },
    )
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
    cfg.rewards["foot_exchange"] = RewardTermCfg(
        func=periodic_foot_exchange,
        weight=0.50,
        params={
            "period": gait["period"],
            "separation": 0.012,
            "std": 0.018,
            "asset_cfg": _FOOT_SITES,
        },
    )
    cfg.rewards["cycle_lateral_drift"] = RewardTermCfg(
        func=UpdatedDefaultCycleLateralVelocityCost,
        weight=-0.25,
        params={
            "period": gait["period"],
            "initial_std": cycle["initial_std"],
            "final_std": cycle["final_std"],
            "ramp_steps": cycle["ramp_steps"],
            "max_cost": cycle["max_cost"],
        },
    )
    cfg.rewards["cycle_yaw_drift"] = RewardTermCfg(
        func=UpdatedDefaultCycleYawRateCost,
        weight=-0.20,
        params={
            "period": gait["period"],
            "initial_std": cycle["initial_yaw_std"],
            "final_std": cycle["final_yaw_std"],
            "ramp_steps": cycle["ramp_steps"],
            "max_cost": cycle["max_cost"],
        },
    )
    cfg.rewards["lateral_path_drift"] = RewardTermCfg(
        func=updated_default_lateral_path_cost,
        weight=-0.12,
        params={
            "initial_tolerance": 0.20,
            "final_tolerance": 0.08,
            "ramp_steps": cycle["ramp_steps"],
            "max_cost": 4.0,
        },
    )
    cfg.rewards["motion_rate_safety"] = RewardTermCfg(
        func=motion_rate_safety_cost,
        weight=-0.03,
        params={"command_name": "twist"},
    )
    cfg.rewards["actuator_effort"] = RewardTermCfg(
        func=actuator_effort_cost,
        weight=-0.08,
        params={"effort_scale": 1.5},
    )
    _configure_walk_stability(cfg)
    cfg.rewards["uncommanded_base_ang_vel"].weight = -0.12
    cfg.rewards.pop("fixed_heading_error", None)
    cfg.rewards.pop("lateral_path_deviation", None)

    if not play:
        cfg.curriculum["velocity_command_range"] = CurriculumTermCfg(
            func=velocity_command_range_curriculum,
            params=velocity_range,
        )
        cfg.events["domain_randomization"] = EventTermCfg(
            func=updated_default_domain_randomization,
            mode="reset",
            params={"symmetric_steps": cycle["ramp_steps"]},
        )

    cfg.metrics["average_world_forward_speed"] = MetricsTermCfg(
        func=episode_average_world_forward_speed, reduce="mean"
    )
    cfg.metrics["scheduled_contact"] = MetricsTermCfg(
        func=scheduled_foot_contact,
        params={
            "sensor_name": "feet_ground_contact",
            "period": gait["period"],
            "duty_factor": gait["duty_factor"],
        },
        reduce="mean",
    )
    cfg.metrics["foot_exchange"] = MetricsTermCfg(
        func=periodic_foot_exchange,
        params={
            "period": gait["period"],
            "separation": 0.012,
            "std": 0.018,
            "asset_cfg": _FOOT_SITES,
        },
        reduce="mean",
    )
    cfg.metrics["swing_foot_force_cost"] = MetricsTermCfg(
        func=periodic_swing_force_cost,
        params={"sensor_name": "feet_ground_contact", "force_scale": 4.0, **gait},
        reduce="mean",
    )
    cfg.metrics["stance_foot_speed_cost"] = MetricsTermCfg(
        func=periodic_stance_foot_speed_cost,
        params={"speed_scale": 0.06, "asset_cfg": _FOOT_SITES, **gait},
        reduce="mean",
    )
    cfg.metrics["cycle_straightness_gate"] = MetricsTermCfg(
        func=UpdatedDefaultCycleStraightnessGate,
        params={
            "period": gait["period"],
            "initial_lateral_std": cycle["initial_std"],
            "final_lateral_std": cycle["final_std"],
            "initial_yaw_std": cycle["initial_yaw_std"],
            "final_yaw_std": cycle["final_yaw_std"],
            "ramp_steps": cycle["ramp_steps"],
        },
        reduce="mean",
    )
    cfg.metrics["cycle_lateral_velocity_cost"] = MetricsTermCfg(
        func=UpdatedDefaultCycleLateralVelocityCost,
        params={
            "period": gait["period"],
            "initial_std": cycle["initial_std"],
            "final_std": cycle["final_std"],
            "ramp_steps": cycle["ramp_steps"],
            "max_cost": cycle["max_cost"],
        },
        reduce="mean",
    )
    cfg.metrics["cycle_yaw_rate_cost"] = MetricsTermCfg(
        func=UpdatedDefaultCycleYawRateCost,
        params={
            "period": gait["period"],
            "initial_std": cycle["initial_yaw_std"],
            "final_std": cycle["final_yaw_std"],
            "ramp_steps": cycle["ramp_steps"],
            "max_cost": cycle["max_cost"],
        },
        reduce="mean",
    )
    cfg.metrics["long_horizon_heading_error"] = MetricsTermCfg(
        func=UnwrappedHeadingMagnitude, reduce="mean"
    )
    cfg.metrics["peak_lateral_deviation"] = MetricsTermCfg(
        func=lateral_path_deviation,
        params={"tolerance": 0.08, "max_penalty": 4.0},
        reduce="max",
    )
    cfg.metrics["mean_lateral_deviation"] = MetricsTermCfg(
        func=lateral_path_deviation,
        params={"tolerance": 0.08, "max_penalty": 4.0},
        reduce="mean",
    )
    return cfg


def updated_default_lateral_walking_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Continue the default walker into command-relative lateral locomotion."""
    cfg = updated_default_walking_env_cfg(play)
    gait_period = 0.33
    lateral_range = {
        "initial_range": (0.0, 0.0),
        "final_range": (-0.1, 0.1),
        "ramp_steps": 24_000,
    }
    cfg.commands["twist"] = LateralVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_world_envs=0.0,
        rel_forward_envs=0.0,
        heading_command=False,
        init_velocity_prob=0.0,
        debug_vis=False,
        zero_lateral_probability=0.35,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.1),
            lin_vel_y=(
                lateral_range["final_range"]
                if play
                else lateral_range["initial_range"]
            ),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )

    cfg.rewards.pop("cycle_lateral_drift", None)
    cfg.rewards["cycle_planar_velocity_error"] = RewardTermCfg(
        func=CommandedCyclePlanarVelocityCost,
        weight=-0.25,
        params={
            "command_name": "twist",
            "period": gait_period,
            "lateral_std": 0.02,
            "forward_std": 0.05,
            "lateral_gate_scale": 0.02,
            "forward_weight": 1.0,
            "max_cost": 4.0,
        },
    )
    cfg.rewards["cycle_yaw_drift"].params = {
        "period": gait_period,
        "initial_std": 0.12,
        "final_std": 0.12,
        "ramp_steps": 0,
        "max_cost": 4.0,
    }
    cfg.rewards.pop("lateral_path_drift", None)
    cfg.rewards["commanded_path_error"] = RewardTermCfg(
        func=CommandedPlanarPathCost,
        weight=-0.12,
        params={
            "command_name": "twist",
            "lateral_tolerance": 0.08,
            "forward_tolerance": 0.20,
            "lateral_gate_scale": 0.02,
            "forward_weight": 1.0,
            "max_cost": 4.0,
        },
    )
    cfg.rewards["heading_drift_prevention"] = RewardTermCfg(
        func=GradualBoundedHeadingDriftCost,
        weight=-0.05,
        params={
            "deadband": 0.075,
            "scale": 0.25,
            "max_cost": 4.0,
            "ramp_steps": 12_000,
        },
    )
    cfg.rewards["motion_rate_safety"] = RewardTermCfg(
        func=commanded_motion_rate_safety_cost,
        weight=-0.03,
        params={"command_name": "twist"},
    )

    for name in (
        "cycle_straightness_gate",
        "cycle_lateral_velocity_cost",
        "peak_lateral_deviation",
        "mean_lateral_deviation",
    ):
        cfg.metrics.pop(name, None)
    cfg.metrics["cycle_command_tracking_gate"] = MetricsTermCfg(
        func=CommandedCycleTrackingGate,
        params={
            "command_name": "twist",
            "period": gait_period,
            "lateral_std": 0.02,
            "forward_std": 0.05,
            "yaw_std": 0.12,
            "lateral_gate_scale": 0.02,
        },
        reduce="mean",
    )
    cfg.metrics["cycle_planar_velocity_cost"] = MetricsTermCfg(
        func=CommandedCyclePlanarVelocityCost,
        params={
            "command_name": "twist",
            "period": gait_period,
            "lateral_std": 0.02,
            "forward_std": 0.05,
            "lateral_gate_scale": 0.02,
            "forward_weight": 1.0,
            "max_cost": 4.0,
        },
        reduce="mean",
    )
    cfg.metrics["commanded_planar_path_error"] = MetricsTermCfg(
        func=CommandedPlanarPathError,
        params={"command_name": "twist"},
        reduce="mean",
    )
    cfg.metrics["peak_commanded_planar_path_error"] = MetricsTermCfg(
        func=CommandedPlanarPathError,
        params={"command_name": "twist"},
        reduce="max",
    )
    cfg.metrics["commanded_average_velocity_error"] = MetricsTermCfg(
        func=CommandedPlanarVelocityError,
        params={"command_name": "twist"},
        reduce="mean",
    )
    cfg.metrics["heading_drift_prevention_cost"] = MetricsTermCfg(
        func=GradualBoundedHeadingDriftCost,
        params={
            "deadband": 0.075,
            "scale": 0.25,
            "max_cost": 4.0,
            "ramp_steps": 0,
        },
        reduce="mean",
    )

    if not play:
        cfg.curriculum = {
            "lateral_velocity_command_range": CurriculumTermCfg(
                func=lateral_velocity_command_range_curriculum,
                params=lateral_range,
            )
        }
        cfg.events["domain_randomization"] = EventTermCfg(
            func=full_domain_randomization, mode="reset"
        )
    return cfg


# Single source of truth for the V2 reward weights.  Keep this as a plain dict
# so changing the objective does not require hunting through the environment
# factory.  ``termination`` remains intentionally much larger as a safety
# event; the two displacement costs are the largest locomotion penalties.
QUASI_OMNIDIRECTIONAL_V2_REWARD_WEIGHTS: dict[str, float] = {
    "upright": 2.0,
    "dof_pos_limits": -1.0,
    "action_rate_l2": -0.02,
    "uncommanded_base_ang_vel": -0.12,
    "termination": -250.0,
    "world_velocity": -0.25,
    "scheduled_contact": 5,
    "swing_foot_force": -3.0,
    "stance_foot_speed": -2.0,
    "foot_exchange": 0.50,
    "motion_rate_safety": -0.03,
    "actuator_effort": -0.01,
    "pitch_upright": 4.0,
    "cycle_command_displacement": -3.5,
    "episode_command_displacement": -8.0,
}

# Joint order is the LeggyBot actuator order: right hip/base, right thigh,
# right knee, left hip/base, left thigh, left knee.  Thigh/knee effort is
# retained as a safety regularizer but contributes only 25% of the old cost.
QUASI_OMNIDIRECTIONAL_V2_ACTUATOR_EFFORT_JOINT_WEIGHTS = (
    1.0,
    0.33,
    0.33,
    1.0,
    0.33,
    0.33,
)


def _quasi_omnidirectional_walking_env_cfg(
    play: bool,
    command_cfg_type: type[UniformVelocityCommandCfg],
    include_episode_displacement: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the shared quasi-omnidirectional reward and curriculum config."""
    cfg = updated_default_walking_env_cfg(play)
    v2_weights = (
        QUASI_OMNIDIRECTIONAL_V2_REWARD_WEIGHTS
        if include_episode_displacement
        else None
    )
    gait_period = 0.33
    final_linear_speed = 0.1
    final_yaw_rate = math.pi / 4.0
    ramp_steps = 36_000
    initial_linear_range = (
        (-final_linear_speed, final_linear_speed) if play else (0.0, 0.0)
    )
    initial_yaw_range = (
        (-final_yaw_rate, final_yaw_rate) if play else (0.0, 0.0)
    )
    cfg.commands["twist"] = command_cfg_type(
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
            lin_vel_x=initial_linear_range,
            lin_vel_y=initial_linear_range,
            ang_vel_z=initial_yaw_range,
            heading=None,
        ),
    )

    for name in (
        "cycle_lateral_drift",
        "cycle_yaw_drift",
        "lateral_path_drift",
    ):
        cfg.rewards.pop(name, None)
    cfg.rewards["cycle_command_displacement"] = RewardTermCfg(
        func=CommandedCycleDisplacementCost,
        # The two displacement terms are the dominant locomotion penalties in
        # V2; termination remains a separate safety penalty.
        weight=(v2_weights["cycle_command_displacement"] if v2_weights else -2.0),
        params={
            "command_name": "twist",
            "period": gait_period,
            "sagittal_scale": 0.012,
            "lateral_scale": 0.012,
            "yaw_scale": 0.08,
            "yaw_deadband": 0.02,
            "yaw_relative_scale": 0.5,
            "yaw_scale_floor": 0.05,
            "coupled_vx_threshold": 0.02,
            "max_cost": 12.0,
        },
    )
    if include_episode_displacement:
        cfg.rewards["episode_command_displacement"] = RewardTermCfg(
            func=CommandedEpisodeDisplacementCost,
            weight=v2_weights["episode_command_displacement"],
            params={
                "command_name": "twist",
                "linear_scale": 0.08,
                "yaw_scale": 0.40,
                "yaw_deadband": 0.02,
                "yaw_relative_scale": 0.5,
                "yaw_scale_floor": 0.05,
                "coupled_vx_threshold": 0.02,
                "max_cost": 12.0,
                "warmup_s": 0.50,
            },
        )
    cfg.rewards["world_velocity"] = RewardTermCfg(
        func=instantaneous_world_command_velocity_cost,
        weight=(v2_weights["world_velocity"] if v2_weights else -0.25),
        params={
            "command_name": "twist",
            "linear_scale": 0.08,
            "yaw_scale": 0.40,
            "yaw_deadband": 0.02,
            "yaw_relative_scale": 0.5,
            "yaw_scale_floor": 0.05,
            "coupled_vx_threshold": 0.02,
            "max_cost": 4.0,
        },
    )
    cfg.rewards["foot_exchange"] = RewardTermCfg(
        func=commanded_periodic_foot_exchange,
        weight=(v2_weights["foot_exchange"] if v2_weights else 0.50),
        params={
            "command_name": "twist",
            "period": gait_period,
            "reference_speed": 0.08,
            "reference_peak_to_peak": 0.024,
            "std": 0.018,
            "asset_cfg": _FOOT_SITES,
        },
    )
    cfg.rewards["motion_rate_safety"] = RewardTermCfg(
        func=commanded_motion_rate_safety_cost,
        weight=(v2_weights["motion_rate_safety"] if v2_weights else -0.03),
        params={"command_name": "twist"},
    )

    if v2_weights is not None:
        missing = set(v2_weights).difference(cfg.rewards)
        if missing:
            raise RuntimeError(
                f"V2 reward weights reference missing terms: {sorted(missing)}"
            )
        for name, weight in v2_weights.items():
            cfg.rewards[name].weight = weight
        cfg.rewards["actuator_effort"].params.update(
            {
                "joint_weights": QUASI_OMNIDIRECTIONAL_V2_ACTUATOR_EFFORT_JOINT_WEIGHTS,
            }
        )

    for name in (
        "average_world_forward_speed",
        "cycle_straightness_gate",
        "cycle_lateral_velocity_cost",
        "cycle_yaw_rate_cost",
        "long_horizon_heading_error",
        "peak_lateral_deviation",
        "mean_lateral_deviation",
    ):
        cfg.metrics.pop(name, None)
    cfg.metrics["cycle_command_displacement_cost"] = MetricsTermCfg(
        func=CommandedCycleDisplacementCost,
        params={
            "command_name": "twist",
            "period": gait_period,
            "sagittal_scale": 0.012,
            "lateral_scale": 0.012,
            "yaw_scale": 0.08,
            "yaw_deadband": 0.02,
            "yaw_relative_scale": 0.5,
            "yaw_scale_floor": 0.05,
            "coupled_vx_threshold": 0.02,
            "max_cost": 12.0,
        },
        reduce="mean",
    )
    if include_episode_displacement:
        cfg.metrics["episode_command_displacement_cost"] = MetricsTermCfg(
            func=CommandedEpisodeDisplacementCost,
            params={
                "command_name": "twist",
                "linear_scale": 0.08,
                "yaw_scale": 0.40,
                "yaw_deadband": 0.02,
                "yaw_relative_scale": 0.5,
                "yaw_scale_floor": 0.05,
                "coupled_vx_threshold": 0.02,
                "max_cost": 12.0,
                "warmup_s": 0.50,
            },
            reduce="mean",
        )
    for name, component in (
        ("cycle_sagittal_displacement_error", 0),
        ("cycle_lateral_displacement_error", 1),
        ("cycle_yaw_displacement_error", 2),
    ):
        cfg.metrics[name] = MetricsTermCfg(
            func=CommandedCycleDisplacementError,
            params={
                "command_name": "twist",
                "period": gait_period,
                "component": component,
            },
            reduce="mean",
        )
    cfg.metrics["instantaneous_world_velocity_cost"] = MetricsTermCfg(
        func=instantaneous_world_command_velocity_cost,
        params={
            "command_name": "twist",
            "linear_scale": 0.08,
            "yaw_scale": 0.40,
            "yaw_deadband": 0.02,
            "yaw_relative_scale": 0.5,
            "yaw_scale_floor": 0.05,
            "coupled_vx_threshold": 0.02,
            "max_cost": 4.0,
        },
        reduce="mean",
    )
    cfg.metrics["command_tracking_success"] = MetricsTermCfg(
        func=quasi_omnidirectional_command_success,
        params={
            "command_name": "twist",
            "linear_error": 0.05,
            "yaw_deadband": 0.02,
            "yaw_relative_scale": 0.5,
            "yaw_scale_floor": 0.05,
            "coupled_vx_threshold": 0.02,
        },
        reduce="mean",
    )
    cfg.metrics["foot_exchange"] = MetricsTermCfg(
        func=commanded_periodic_foot_exchange,
        params={
            "command_name": "twist",
            "period": gait_period,
            "reference_speed": 0.08,
            "reference_peak_to_peak": 0.024,
            "std": 0.018,
            "asset_cfg": _FOOT_SITES,
        },
        reduce="mean",
    )

    if not play:
        curriculum_func = (
            StagedQuasiOmnidirectionalCurriculum(
                final_linear_speed=final_linear_speed,
                final_yaw_rate=final_yaw_rate,
            )
            if include_episode_displacement
            else quasi_omnidirectional_command_curriculum
        )
        curriculum_params = (
            {}
            if include_episode_displacement
            else {
                "final_linear_speed": final_linear_speed,
                "final_yaw_rate": final_yaw_rate,
                "ramp_steps": ramp_steps,
            }
        )
        cfg.curriculum = {
            "quasi_omnidirectional_command_range": CurriculumTermCfg(
                func=curriculum_func,
                params=curriculum_params,
            )
        }
    return cfg


def quasi_omnidirectional_walking_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Train one blank policy with independent quasi-omnidirectional sampling."""
    return _quasi_omnidirectional_walking_env_cfg(
        play, QuasiOmnidirectionalCommandCfg
    )


def quasi_omnidirectional_walking_v2_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Train quasi-omnidirectional walking with deployment-weighted command modes."""
    return _quasi_omnidirectional_walking_env_cfg(
        play, QuasiOmnidirectionalMixtureCommandCfg, include_episode_displacement=True
    )


def se2_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Learn straight, lateral and curved travel together with local SE(2) tracking.

    Fixed, modest command bounds deliberately avoid a prerequisite that can
    starve whole motion modes. Larger yaw rates should follow measured success.
    """
    cfg = quasi_omnidirectional_walking_v2_env_cfg(play)
    cfg.episode_length_s = 20.0
    cfg.curriculum = {}
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (0.04, 0.10)
    command.ranges.lin_vel_y = (-0.08, 0.08)
    command.ranges.ang_vel_z = (-0.40, 0.40)
    command.protected_forward_range = (0.04, 0.10)
    command.forward_straight_probability = .35
    command.forward_curve_probability = .30
    command.pure_lateral_probability = .30
    command.stop_probability = .05
    command.full_omni_probability = 0.
    command.reverse_probability = 0.
    command.resampling_time_range = (5., 8.)
    # Preserve physical servo limits and the deployable 29-value actor ABI.
    # SE2 remains the historical independent-randomization ablation; V3
    # replaces this event with a paired-to-mismatched schedule below.
    if not play:
        cfg.events["domain_randomization"] = EventTermCfg(
            func=updated_default_domain_randomization,
            mode="reset",
            params={"symmetric_steps": 0, "independent_ramp_steps": 0},
        )
    for name in ("cycle_command_displacement", "episode_command_displacement",
                 "world_velocity", "foot_exchange"):
        cfg.rewards.pop(name, None)
    cfg.rewards["se2_tracking"] = RewardTermCfg(
        func=WindowedSE2Tracking, weight=4.,
        params={"horizon_s": .66, "linear_scale": .04, "yaw_scale": .15},
    )
    cfg.rewards["path_tracking"] = RewardTermCfg(
        func=WindowedSE2Tracking, weight=-1.5,
        params={"horizon_s": 2., "linear_scale": .025,
                "yaw_scale": .10, "output": "cost"},
    )
    # Contact timing shapes a gait; it must not outweigh moving correctly.
    for name, weight in {"upright": 1., "pitch_upright": 1.,
                         "scheduled_contact": .5, "swing_foot_force": -.5,
                         "stance_foot_speed": -.5}.items():
        cfg.rewards[name].weight = weight
    cfg.metrics = {
        "tracking_success": MetricsTermCfg(func=WindowedSE2Tracking,
            params={"horizon_s": .66, "linear_scale": .025,
                    "yaw_scale": .10, "output": "success"}),
        "window_linear_error_mps": MetricsTermCfg(func=WindowedSE2Tracking,
            params={"horizon_s": .66, "output": "linear_error"}),
        "window_yaw_error_radps": MetricsTermCfg(func=WindowedSE2Tracking,
            params={"horizon_s": .66, "output": "yaw_error"}),
    }
    return cfg


def natural_omnidirectional_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Preserve emergent stepping while removing V1's reward and PPO degeneracies."""
    cfg = se2_tracking_env_cfg(play)
    command = cfg.commands["twist"]
    command.forward_straight_probability = .30
    command.forward_curve_probability = .30
    command.pure_lateral_probability = .30
    command.stop_probability = .10
    command.minimum_lateral_speed = .04

    cfg.rewards.pop("se2_tracking", None)
    cfg.rewards.pop("path_tracking", None)
    cfg.rewards["linear_tracking"] = RewardTermCfg(
        func=WindowedSE2Tracking, weight=3.0,
        params={"horizon_s": .66, "linear_scale": .035,
                "yaw_scale": .15, "output": "linear_reward"},
    )
    cfg.rewards["yaw_tracking"] = RewardTermCfg(
        func=WindowedSE2Tracking, weight=1.5,
        params={"horizon_s": .66, "linear_scale": .035,
                "yaw_scale": .12, "output": "yaw_reward"},
    )
    cfg.rewards["linear_path_error"] = RewardTermCfg(
        func=WindowedSE2Tracking, weight=-.75,
        params={"horizon_s": 2.0, "linear_scale": .035,
                "yaw_scale": .12, "output": "linear_cost"},
    )
    cfg.rewards["yaw_path_error"] = RewardTermCfg(
        func=WindowedSE2Tracking, weight=-.50,
        params={"horizon_s": 2.0, "linear_scale": .035,
                "yaw_scale": .12, "output": "yaw_cost"},
    )

    gait = {"command_name": "twist", "period": .33,
            "duty_factor": .70, "full_motion_speed": .04}
    cfg.rewards["scheduled_contact"] = RewardTermCfg(
        func=commanded_scheduled_foot_contact, weight=.40,
        params={"sensor_name": "feet_ground_contact", **gait},
    )
    cfg.rewards["swing_foot_force"] = RewardTermCfg(
        func=commanded_periodic_swing_force_cost, weight=-.35,
        params={"sensor_name": "feet_ground_contact", "force_scale": 4.0,
                "transition_fraction": .04, **gait},
    )
    cfg.rewards["stance_foot_speed"] = RewardTermCfg(
        func=commanded_periodic_stance_foot_speed_cost, weight=-.35,
        params={"speed_scale": .06, "transition_fraction": .04,
                "asset_cfg": _FOOT_SITES, **gait},
    )
    cfg.rewards["neutral_stance"] = RewardTermCfg(
        func=commanded_neutral_stance, weight=.80,
        params={"command_name": "twist", "joint_std": .12,
                "full_motion_speed": .04},
    )
    cfg.rewards["stopped_foot_alignment"] = RewardTermCfg(
        func=stopped_foot_alignment_cost, weight=-.30,
        params={"command_name": "twist", "fore_aft_scale": .015,
                "full_motion_speed": .04, "asset_cfg": _FOOT_SITES},
    )
    cfg.rewards["upright"].weight = .75
    cfg.rewards["pitch_upright"].weight = .75
    cfg.rewards["uncommanded_base_ang_vel"].weight = -.08
    cfg.rewards["motion_rate_safety"].weight = -.05
    cfg.rewards["actuator_effort"].weight = -.03
    cfg.rewards["action_rate_l2"].weight = -.03
    cfg.rewards["dof_pos_limits"].weight = -2.0
    return cfg


def natural_omnidirectional_v3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Natural walking with exclusive planar modes and a lateral line cost.

    This is deliberately based on the proven Natural reward.  It adds only a
    stronger pitch-upright term and the small double-support lateral foot-line
    cost.  Command modes are limited to standing, straight forward/reverse
    motion, pure lateral motion, and forward-plus-yaw curves.
    """
    cfg = natural_omnidirectional_env_cfg(play)
    if not play:
        # The RSL-RL runner collects 24 environment steps per PPO iteration.
        # Hold a reflection-consistent physical distribution through iteration
        # 500, then ramp the full independent envelope through iteration 800.
        cfg.events["domain_randomization"] = EventTermCfg(
            func=updated_default_domain_randomization,
            mode="reset",
            params={
                "symmetric_steps": _NATURAL_V3_SYMMETRIC_STEPS,
                "independent_ramp_steps": _NATURAL_V3_MISMATCH_RAMP_STEPS,
            },
        )
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (-0.05, 0.10)
    command.ranges.lin_vel_y = (-0.08, 0.08)
    command.ranges.ang_vel_z = (-0.40, 0.40)
    command.forward_straight_probability = .35
    command.forward_curve_probability = .35
    command.pure_lateral_probability = .20
    command.full_omni_probability = .0
    command.stop_probability = .10
    command.reverse_probability = .0
    command.protected_forward_range = (.04, .10)
    command.minimum_lateral_speed = .04
    command.minimum_coupled_yaw_fraction = .25
    command.resampling_time_range = (5., 8.)
    cfg.rewards["pitch_upright"].weight = 1.50
    cfg.metrics["lateral_foot_line_error_m"] = MetricsTermCfg(
        func=lateral_foot_line_error,
        params={
            "command_name": "twist",
            "min_lateral_speed": 0.04,
            "max_forward_speed": 0.01,
            "max_yaw_rate": 0.02,
            "asset_cfg": _FOOT_SITES,
        },
    )
    for direction, label in ((1, "left"), (-1, "right")):
        cfg.metrics[f"{label}_lateral_command_fraction"] = MetricsTermCfg(
            func=lateral_command_active,
            params={
                "command_name": "twist",
                "lateral_direction": direction,
                "min_lateral_speed": .04,
                "max_forward_speed": .01,
                "max_yaw_rate": .02,
            },
        )
        cfg.metrics[f"{label}_lateral_tracking_success"] = MetricsTermCfg(
            func=WindowedSE2Tracking,
            params={
                "horizon_s": .66,
                "linear_scale": .025,
                "yaw_scale": .10,
                "output": "success",
                "lateral_direction": direction,
                "min_lateral_speed": .04,
                "max_forward_speed": .01,
                "max_yaw_rate": .02,
            },
        )
        cfg.metrics[f"{label}_lateral_foot_line_error_m"] = MetricsTermCfg(
            func=lateral_foot_line_error,
            params={
                "command_name": "twist",
                "min_lateral_speed": .04,
                "max_forward_speed": .01,
                "max_yaw_rate": .02,
                "lateral_direction": direction,
                "asset_cfg": _FOOT_SITES,
            },
        )
    return cfg


def natural_omnidirectional_v4_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """V3 stepping with task symmetry, lateral placement and drift precision.

    No instantaneous bilateral joint-equality reward: alternating stance and
    swing, lateral travel, and curves all require unequal simultaneous angles.
    The runner uses reflected state/action samples (including gait phase) and
    a soft mirror loss while retaining a freely adapting MLP for mismatch.
    """
    cfg = natural_omnidirectional_v3_env_cfg(play)
    command = cfg.commands['twist']
    command.ranges.lin_vel_x = (0.0, .10)
    command.protected_forward_range = (.04, .10)
    command.reverse_probability = 0.0
    command.forward_straight_probability = .45
    command.forward_curve_probability = .45
    command.pure_lateral_probability = .0
    command.scale_yaw_with_forward_speed = False
    # Deployment mixture: .45 straight, .45 curves, .10 stopped.
    cfg.rewards['lateral_foot_line'] = RewardTermCfg(
        func=commanded_lateral_foot_line_cost, weight=-.20,
        params={'command_name': 'twist', 'line_scale': .025,
                'period': .33, 'duty_factor': .70,
                'transition_fraction': .04, 'asset_cfg': _FOOT_SITES},
    )
    cfg.rewards['cross_track_precision'] = RewardTermCfg(
        func=WindowedSE2Tracking, weight=-.50,
        params={'horizon_s': 1.0, 'linear_scale': .010,
                'output': 'cross_track_cost'},
    )
    # Narrow the long-window yaw cost without shrinking the broad exploration
    # reward. At 0.03 rad/s bias this now costs ~0.19 instead of ~0.015.
    cfg.rewards['yaw_path_error'].weight = -.75
    cfg.rewards['yaw_path_error'].params['yaw_scale'] = .04
    cfg.metrics['cross_track_error_mps'] = MetricsTermCfg(
        func=WindowedSE2Tracking,
        params={'horizon_s': 1.0, 'output': 'cross_track_error'},
    )
    return cfg


def natural_omnidirectional_v5_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Continue a complete V4 policy while probing its maximum forward speed."""
    cfg = natural_omnidirectional_v4_env_cfg(play)
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (0.0, 0.15)
    command.protected_forward_range = (0.04, 0.15)
    cfg.curriculum = {
        "forward_speed_gate": CurriculumTermCfg(
            func=ForwardSpeedGateCurriculum(
                speed_levels=(0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
                success_threshold=0.70,
                success_window_iterations=50,
                minimum_gate_iterations=100,
                steps_per_iteration=24,
                minimum_forward_speed=0.04,
                metric_name="tracking_success",
            )
        )
    }
    if not play:
        # V5 starts from a V4 policy that has already completed the independent
        # mismatch ramp, so retain that terminal distribution from iteration 0.
        cfg.events["domain_randomization"] = EventTermCfg(
            func=updated_default_domain_randomization,
            mode="reset",
            params={"symmetric_steps": 0, "independent_ramp_steps": 0},
        )
    return cfg


def periodic_reward_composition_walking_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Walking-only periodic reward composition adapted to LeggyBot.

    This is the first controlled ablation of Siekmann et al. (2021): an MLP
    retains the deployable six-position action ABI while gait shaping is
    limited to phase-weighted swing force and stance-foot speed.  Straight
    and lateral commands are sampled as disjoint modes; curves and combined
    planar commands are structurally absent.
    """
    cfg = _shared_skill_cfg(play)
    cfg.episode_length_s = 20.0
    cfg.curriculum = {}
    cfg.events.pop("push", None)

    cfg.commands["twist"] = QuasiOmnidirectionalMixtureCommandCfg(
        entity_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_world_envs=0.0,
        rel_forward_envs=0.0,
        heading_command=False,
        init_velocity_prob=0.0,
        debug_vis=play,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.10),
            lin_vel_y=(-0.08, 0.08),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
        forward_straight_probability=0.45,
        forward_curve_probability=0.0,
        pure_lateral_probability=0.45,
        full_omni_probability=0.0,
        stop_probability=0.10,
        reverse_probability=0.0,
        protected_forward_range=(0.04, 0.10),
        minimum_coupled_yaw_fraction=0.0,
        minimum_lateral_speed=0.04,
    )

    gait = {
        "command_name": "twist",
        "period": 0.33,
        "duty_factor": 0.70,
        "transition_fraction": 0.04,
        "full_motion_speed": 0.04,
    }
    for group in cfg.observations.values():
        group.terms["gait_clock"].params["period"] = gait["period"]

    # The published generic-controller coefficients are used where the paper
    # gives them.  Velocity costs alone are weighted more strongly so standing
    # is not competitive at LeggyBot's low commanded speeds.  Constant beta=1
    # makes survival valuable.  Safety terms below are explicit robot additions.
    cfg.rewards = {
        "alive_bias": RewardTermCfg(func=constant_alive_reward, weight=1.0),
        "swing_foot_force": RewardTermCfg(
            func=commanded_periodic_reward_composition_swing_force_cost,
            weight=-0.40,
            params={"sensor_name": "feet_ground_contact", "force_scale": 10.0, **gait},
        ),
        "stance_foot_speed": RewardTermCfg(
            func=commanded_periodic_reward_composition_stance_speed_cost,
            weight=-0.40,
            params={"speed_coefficient": 2.0, "asset_cfg": _FOOT_SITES, **gait},
        ),
        "forward_velocity": RewardTermCfg(
            func=periodic_reward_composition_velocity_cost,
            weight=-1.20,
            params={"command_name": "twist", "axis": 0, "coefficient": 2.0},
        ),
        "lateral_velocity": RewardTermCfg(
            func=periodic_reward_composition_velocity_cost,
            weight=-1.20,
            params={"command_name": "twist", "axis": 1, "coefficient": 2.0},
        ),
        "base_orientation": RewardTermCfg(
            func=periodic_reward_composition_orientation_cost,
            weight=-0.30,
            params={"coefficient": 3.0},
        ),
        "action_difference": RewardTermCfg(
            func=periodic_reward_composition_action_difference_cost,
            weight=-0.10,
            params={"coefficient": 5.0},
        ),
        "actuator_torque": RewardTermCfg(
            func=periodic_reward_composition_torque_cost,
            weight=-0.10,
            params={"coefficient": 0.05},
        ),
        "pelvis_motion": RewardTermCfg(
            func=PeriodicRewardCompositionPelvisMotionCost,
            weight=-0.10,
            params={"coefficient": 0.10},
        ),
        "neutral_stance": RewardTermCfg(
            func=commanded_neutral_stance,
            weight=0.20,
            params={"command_name": "twist", "joint_std": 0.12, "full_motion_speed": 0.04},
        ),
        "stopped_foot_alignment": RewardTermCfg(
            func=stopped_foot_alignment_cost,
            weight=-0.10,
            params={
                "command_name": "twist",
                "fore_aft_scale": 0.015,
                "full_motion_speed": 0.04,
                "asset_cfg": _FOOT_SITES,
            },
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "termination": RewardTermCfg(func=mdp.is_terminated, weight=-250.0),
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
        "base_pitch_abs_rad": MetricsTermCfg(func=base_pitch_magnitude),
        "joint_soft_limit_excess_rad": MetricsTermCfg(func=mdp.joint_pos_limits),
    }
    robot = cfg.scene.entities["robot"]
    assert robot.articulation is not None
    robot.articulation.soft_joint_pos_limit_factor = 0.85
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
        LocomotionTask(
            UPDATED_DEFAULT_WALKING_TASK_ID,
            updated_default_walking_env_cfg,
            "leggybot_updated_default_walking",
            lambda env: updated_default_walking_success(
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
            UPDATED_DEFAULT_LATERAL_WALKING_TASK_ID,
            updated_default_lateral_walking_env_cfg,
            "leggybot_updated_default_lateral_walking",
            lambda env: updated_default_lateral_walking_success(
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
            QUASI_OMNIDIRECTIONAL_WALKING_TASK_ID,
            quasi_omnidirectional_walking_env_cfg,
            "leggybot_quasi_omnidirectional_walking",
            lambda env: quasi_omnidirectional_velocity_success(env, "twist"),
            periodic_walk_evaluation_active,
            0.65,
            use_symmetry_augmentation=True,
        ),
        LocomotionTask(
            QUASI_OMNIDIRECTIONAL_WALKING_V2_TASK_ID,
            quasi_omnidirectional_walking_v2_env_cfg,
            "leggybot_quasi_omnidirectional_walking_v2",
            lambda env: quasi_omnidirectional_velocity_success(env, "twist"),
            periodic_walk_evaluation_active,
            0.65,
            use_symmetry_augmentation=True,
        ),
        LocomotionTask(
            SE2_TRACKING_TASK_ID,
            se2_tracking_env_cfg,
            "leggybot_se2_tracking_walking",
            tracking_success,
            periodic_walk_evaluation_active,
            .80,
            use_symmetry_augmentation=False,
        ),
        LocomotionTask(
            NATURAL_OMNIDIRECTIONAL_TASK_ID,
            natural_omnidirectional_env_cfg,
            "leggybot_natural_omnidirectional_walking",
            tracking_success,
            periodic_walk_evaluation_active,
            .80,
            use_symmetry_augmentation=False,
            bounded_exploration=True,
        ),
        LocomotionTask(
            NATURAL_OMNIDIRECTIONAL_V3_TASK_ID,
            natural_omnidirectional_v3_env_cfg,
            "leggybot_natural_omnidirectional_walking_v3",
            tracking_success,
            periodic_walk_evaluation_active,
            .80,
            use_symmetry_augmentation=False,
            symmetry_mirror_loss_coeff=.10,
            bounded_exploration=True,
            exploration_init_std=.15,
            exploration_std_range=(.03, .40),
            learning_rate=1e-3,
        ),
        LocomotionTask(
            PERIODIC_REWARD_COMPOSITION_WALKING_TASK_ID,
            periodic_reward_composition_walking_env_cfg,
            "leggybot_periodic_reward_composition_walking",
            tracking_success,
            periodic_walk_evaluation_active,
            .80,
            use_symmetry_augmentation=False,
            symmetry_mirror_loss_coeff=.10,
            bounded_exploration=True,
            exploration_init_std=.12,
            exploration_std_range=(.04, .30),
            learning_rate=1e-4,
        ),
        LocomotionTask(
            NATURAL_OMNIDIRECTIONAL_V4_TASK_ID,
            natural_omnidirectional_v4_env_cfg,
            'leggybot_natural_omnidirectional_walking_v4',
            tracking_success, periodic_walk_evaluation_active, .80,
            use_symmetry_augmentation=True,
            symmetry_mirror_loss_coeff=.05,
            bounded_exploration=True,
            exploration_init_std=.15,
            exploration_std_range=(.03, .40),
            learning_rate=1e-3,
        ),
        LocomotionTask(
            NATURAL_OMNIDIRECTIONAL_V5_TASK_ID,
            natural_omnidirectional_v5_env_cfg,
            "leggybot_natural_omnidirectional_walking_v5",
            tracking_success, periodic_walk_evaluation_active, .80,
            use_symmetry_augmentation=True,
            symmetry_mirror_loss_coeff=.05,
            bounded_exploration=True,
            exploration_init_std=.15,
            exploration_std_range=(.03, .40),
            learning_rate=1e-3,
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
                symmetry_mirror_loss_coeff=task.symmetry_mirror_loss_coeff,
                bounded_exploration=task.bounded_exploration,
                exploration_init_std=task.exploration_init_std,
                exploration_std_range=task.exploration_std_range,
                learning_rate=task.learning_rate,
            ),
            runner_cls=task_runner(task.task_id),
        )


__all__ = [
    "SE2_TRACKING_TASK_ID", "se2_tracking_env_cfg",
    "NATURAL_OMNIDIRECTIONAL_TASK_ID", "natural_omnidirectional_env_cfg",
    "NATURAL_OMNIDIRECTIONAL_V3_TASK_ID", "natural_omnidirectional_v3_env_cfg",
    "PERIODIC_REWARD_COMPOSITION_WALKING_TASK_ID", "periodic_reward_composition_walking_env_cfg",
    "ANIMATED_REFERENCE_WALK_TASK_ID", "CONSERVATIVE_WALK_TASK_ID",
    "SAGITTAL_TASK_ID", "STAND_TASK_ID", "LocomotionTask",
    "STEP_IN_PLACE_TASK_ID", "LOCOMOTION_TASK_IDS", "UNLOADING_WALK_TASK_ID",
    "PERIODIC_WALK_TASK_ID",
    "QUASI_OMNIDIRECTIONAL_WALKING_TASK_ID",
    "QUASI_OMNIDIRECTIONAL_WALKING_V2_TASK_ID",
    "STRAIGHT_PERIODIC_WALK_TASK_ID",
    "SYMMETRIC_PERIODIC_WALK_TASK_ID",
    "UPDATED_DEFAULT_LATERAL_WALKING_TASK_ID",
    "UPDATED_DEFAULT_WALKING_TASK_ID",
    "WEIGHT_TRANSFER_TASK_ID", "animated_reference_walk_env_cfg", "conservative_walk_env_cfg",
    "get_locomotion_task", "locomotion_tasks",
    "periodic_walk_env_cfg", "register_leggybot_skill_tasks", "sagittal_env_cfg", "straight_periodic_walk_env_cfg", "symmetric_cycle_periodic_walk_env_cfg", "symmetric_periodic_walk_env_cfg", "unloading_walk_env_cfg",
    "quasi_omnidirectional_walking_env_cfg", "quasi_omnidirectional_walking_v2_env_cfg", "stand_env_cfg", "step_in_place_env_cfg", "updated_default_lateral_walking_env_cfg", "updated_default_walking_env_cfg", "weight_transfer_env_cfg",
    "QuasiOmnidirectionalCommandCfg", "QuasiOmnidirectionalCommand", "QuasiOmnidirectionalMixtureCommandCfg", "QuasiOmnidirectionalMixtureCommand",
]
