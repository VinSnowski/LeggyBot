"""Shared fixed-ankle LeggyBot model and PPO configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.spec_config import CollisionCfg

from locomotion.robot_profiles import LEGGYBOT_FIXED_ANKLE_PROFILE, LEGGYBOT_PROFILE
from locomotion.actions import AsymmetricJointPositionActionCfg
from locomotion.servo import (
    STS3215_MAX_PROFILE_ACCELERATION_RAD_S2,
    STS3215_MAX_PROFILE_SPEED_RAD_S,
    STS3215_POSITION_RESOLUTION_RAD,
)
from locomotion.tasks.rewards import (
    command_direction_progress,
    uncommanded_base_angular_velocity,
)


@dataclass
class SymmetryPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """RSL-RL PPO config with optional native symmetry augmentation."""

    symmetry_cfg: dict[str, Any] | None = None
FIXED_ANKLE_PITCH_DEG = 8.0
_LEGACY_MODEL_PATH = Path(__file__).parents[1] / "LeggyBot" / "mjcf" / "LeggyBot.xml"
_FIXED_ANKLE_MODEL_PATH = (
    Path(__file__).parents[1] / "NewLeggyBot" / "mjcf" / "NewLeggyBot.xml"
)

_FOOT_GEOMS = ("right_foot_collision", "left_foot_collision")
_FOOT_SITES = ("right_foot", "left_foot")
_NON_FOOT_GEOMS = (
    "base_collision",
    "right_hip_collision",
    "right_thigh_collision",
    "right_shin_collision",
    "left_hip_collision",
    "left_thigh_collision",
    "left_shin_collision",
)


def _get_robot_spec(ankle_pitch_deg: float = 0.0) -> mujoco.MjSpec:
    """Load the training MJCF and remove viewer-only world objects."""
    model_path = (
        _LEGACY_MODEL_PATH
        if ankle_pitch_deg == 0.0
        else _FIXED_ANKLE_MODEL_PATH
    )
    spec = mujoco.MjSpec.from_file(str(model_path))
    floor = spec.geom("floor")
    if floor is not None:
        spec.delete(floor)
    for light in list(spec.lights):
        spec.delete(light)
    for equality in list(spec.equalities):
        spec.delete(equality)
    # Entity options are not propagated by MjSpec.attach. Reset them here and
    # let this task's SimulationCfg be the single owner of solver settings.
    defaults = mujoco.MjOption()
    spec.option.timestep = defaults.timestep
    spec.option.integrator = defaults.integrator
    spec.option.iterations = defaults.iterations
    spec.option.ls_iterations = defaults.ls_iterations
    return spec


_FULL_COLLISION = CollisionCfg(
    geom_names_expr=(r".*_collision",),
    contype=1,
    conaffinity=1,
    solref=(0.01, 1),
    condim={r".*_foot_collision": 6, r".*_collision": 1},
    priority={r".*_foot_collision": 1, ".*": 0},
    friction={r".*_foot_collision": (0.9, 5e-3, 5e-4)},
)

def get_leggybot_robot_cfg(
    *,
    ankle_pitch_deg: float = 0.0,
) -> EntityCfg:
    if ankle_pitch_deg not in (0.0, FIXED_ANKLE_PITCH_DEG):
        raise ValueError("Only the validated 0- and 8-degree ankle variants exist")
    profile = (
        LEGGYBOT_FIXED_ANKLE_PROFILE
        if ankle_pitch_deg == FIXED_ANKLE_PITCH_DEG
        else LEGGYBOT_PROFILE
    )
    return EntityCfg(
        spec_fn=partial(_get_robot_spec, ankle_pitch_deg),
        init_state=EntityCfg.InitialStateCfg(
            # The CAD root is below the torso; this places the straight-leg foot
            # soles a few millimetres above the plane before reset settling.
            pos=profile.initial_root_pos,
            joint_pos=dict(zip(profile.joint_names, profile.default_joint_pos, strict=True)),
            joint_vel={".*": 0.0},
        ),
        collisions=(_FULL_COLLISION,),
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(
                XmlActuatorCfg(
                    target_names_expr=(joint_name,),
                    # Command timing is modeled once for the synchronized bus
                    # packet by AsymmetricJointPositionAction. Per-actuator
                    # buffers would duplicate latency and let paired joints
                    # receive one packet at different times.
                    delay_min_lag=0,
                    delay_max_lag=0,
                    delay_hold_prob=1.0,
                    delay_update_period=0,
                )
                for joint_name in profile.joint_names
            ),
            soft_joint_pos_limit_factor=0.90,
        ),
    )


def leggybot_ppo_runner_cfg(
    experiment_name: str = "leggybot_walking",
    *,
    use_symmetry_augmentation: bool = False,
    symmetry_mirror_loss_coeff: float = 0.0,
    bounded_exploration: bool = False,
    exploration_init_std: float | None = None,
    exploration_std_range: tuple[float, float] = (0.03, 0.40),
    learning_rate: float = 1e-3,
) -> RslRlOnPolicyRunnerCfg:
    if symmetry_mirror_loss_coeff < 0.0:
        raise ValueError("symmetry mirror loss coefficient must be non-negative")
    if exploration_init_std is None:
        exploration_init_std = 0.15 if bounded_exploration else 0.10
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                # Enough normalized exploration to discover a useful gait.
                "init_std": exploration_init_std,
                "std_type": "log" if bounded_exploration else "scalar",
                **(
                    {"std_range": exploration_std_range}
                    if bounded_exploration
                    else {}
                ),
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 128, 128), activation="elu", obs_normalization=True
        ),
        algorithm=SymmetryPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.001 if bounded_exploration else 0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=learning_rate,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            # PPOaug: each minibatch contains original and reflected
            # transitions.  This is deliberately data augmentation only, not
            # a hard-equivariant architecture or an extra mirror-loss term.
            symmetry_cfg=(
                {
                    "data_augmentation_func": "locomotion.symmetry:leggybot_left_right_augmentation",
                    "use_data_augmentation": use_symmetry_augmentation,
                    "use_mirror_loss": symmetry_mirror_loss_coeff > 0.0,
                    "mirror_loss_coeff": symmetry_mirror_loss_coeff,
                }
                if use_symmetry_augmentation or symmetry_mirror_loss_coeff > 0.0
                else None
            ),
        ),
        experiment_name=experiment_name,
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=10_000,
        clip_actions=1.0,
    )


def _leggybot_base_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the robot and simulator foundation for walking."""
    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {
        "robot": get_leggybot_robot_cfg(ankle_pitch_deg=FIXED_ANKLE_PITCH_DEG)
    }
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Flat ground needs only the per-foot height probe from the base task.
    cfg.scene.sensors = tuple(
        sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
    )
    for sensor in cfg.scene.sensors:
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=name, entity="robot")
                for name in _FOOT_SITES
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.02, num_samples=4)

    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom", pattern=_FOOT_GEOMS, entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    non_foot_ground = ContactSensorCfg(
        name="non_foot_ground_contact",
        primary=ContactMatch(
            mode="geom", pattern=_NON_FOOT_GEOMS, entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground, non_foot_ground)

    cfg.observations["actor"].terms.pop("height_scan", None)
    cfg.observations["critic"].terms.pop("height_scan", None)

    action_lower, action_upper = LEGGYBOT_FIXED_ANKLE_PROFILE.action_bounds()
    cfg.actions["joint_pos"] = AsymmetricJointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        preserve_order=False,
        target_lower=dict(
            zip(LEGGYBOT_FIXED_ANKLE_PROFILE.joint_names, action_lower.tolist(), strict=True)
        ),
        target_upper=dict(
            zip(LEGGYBOT_FIXED_ANKLE_PROFILE.joint_names, action_upper.tolist(), strict=True)
        ),
        # Walking uses the full safe joint range in both training and
        # playback.  Keeping this identical avoids an otherwise silent policy
        # to actuator-authority mismatch at evaluation or deployment time.
        initial_range_fraction=1.0,
        range_ramp_steps=0,
        # Stateful gear clearance is randomized by the task reset event.  The
        # play operator is also the seam where measured per-joint backlash can
        # replace the current STS3215/BIMO prior later.
        model_backlash=True,
        target_resolution_rad=STS3215_POSITION_RESOLUTION_RAD,
        # At 200 Hz, seven physics ticks cover the 5--15 ms base delay, one
        # tick of jitter, and a rare additional 20 ms late control frame.
        command_delay_max_steps=7,
        command_delay_hold_probability=0.9,
        missed_frame_probability=0.01,
        model_firmware_profile=True,
        firmware_max_velocity_rad_s=STS3215_MAX_PROFILE_SPEED_RAD_S,
        firmware_max_acceleration_rad_s2=STS3215_MAX_PROFILE_ACCELERATION_RAD_S2,
    )

    cfg.commands["twist"] = UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_world_envs=0.0,
        rel_forward_envs=0.0,
        heading_command=False,
        init_velocity_prob=0.0,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.20, 0.20),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )

    cfg.events["reset_base"].params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.005, 0.015),
        "roll": (-0.03, 0.03),
        "pitch": (-0.03, 0.03),
        "yaw": (0.0, 0.0),
    }
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.02, 0.02)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.05, 0.05)
    for event_name in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
        cfg.events.pop(event_name, None)

    cfg.rewards["command_direction_progress"] = RewardTermCfg(
        func=command_direction_progress,
        weight=1.0,
        params={"command_name": "twist", "command_threshold": 0.05},
    )
    cfg.rewards["upright"].params["std"] = math.sqrt(0.10)
    cfg.rewards["upright"].params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=("BaseLink",)
    )
    cfg.rewards.pop("pose", None)
    cfg.rewards.pop("body_ang_vel", None)
    cfg.rewards["uncommanded_base_ang_vel"] = RewardTermCfg(
        func=uncommanded_base_angular_velocity,
        weight=-0.075,
        params={"command_name": "twist"},
    )
    cfg.rewards.pop("angular_momentum", None)
    cfg.rewards["dof_pos_limits"].weight = -1.0
    cfg.rewards["action_rate_l2"].weight = -0.01
    cfg.rewards["foot_slip"].params["asset_cfg"] = SceneEntityCfg(
        "robot", site_names=_FOOT_SITES
    )
    for old_term in ("air_time", "foot_clearance", "foot_swing_height", "soft_landing"):
        cfg.rewards.pop(old_term, None)

    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}
    )
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": "non_foot_ground_contact", "force_threshold": 5.0},
    )
    cfg.curriculum = {}

    cfg.sim.njmax = 200
    cfg.sim.nconmax = None
    cfg.sim.contact_sensor_maxmatch = 32
    cfg.sim.mujoco.timestep = 0.005
    cfg.sim.mujoco.iterations = 10
    cfg.sim.mujoco.ls_iterations = 10
    cfg.decimation = 4
    cfg.viewer.body_name = "BaseLink"
    cfg.viewer.distance = 1.0
    cfg.viewer.elevation = -10.0

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}

    return cfg
