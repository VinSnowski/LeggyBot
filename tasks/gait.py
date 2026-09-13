"""Reusable rewards, metrics, noise, and gait-reference terms for LeggyBot."""

from __future__ import annotations

import math
import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as env_mdp
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_FOOT_SITES_CFG = SceneEntityCfg(
    "robot", site_names=("right_foot", "left_foot"), preserve_order=True
)


def standing_still(
    env,
    linear_std: float = 0.08,
    angular_std: float = 0.15,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward a stationary base; useful for the first hardware gate."""
    asset: Entity = env.scene[asset_cfg.name]
    linear = asset.data.root_link_lin_vel_b
    angular = asset.data.root_link_ang_vel_b
    return torch.exp(-torch.sum(torch.square(linear / linear_std), dim=1)) * torch.exp(
        -torch.sum(torch.square(angular / angular_std), dim=1)
    )


def weight_transfer_target(
    env,
    amplitude: float = 0.10,
    period: float = 2.0,
    initial_amplitude: float | None = None,
    ramp_steps: int = 0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward feasible side-to-side torso roll without requiring a step.

    The target amplitude may be ramped from zero so the policy first learns
    neutral bilateral support before it is asked to lean. The transfer task uses
    0.14 rad (about 8 degrees), below LeggyBot's 10-degree inward hip-roll
    limit, while its force term independently trains the final 80/20 load split.
    """
    if amplitude <= 0.0 or period <= 0.0:
        raise ValueError("invalid weight-transfer target parameters")
    if initial_amplitude is not None and not 0.0 <= initial_amplitude <= amplitude:
        raise ValueError("initial roll amplitude must be within the target range")
    if ramp_steps < 0:
        raise ValueError("ramp_steps must be non-negative")
    asset: Entity = env.scene[asset_cfg.name]
    gravity = asset.data.projected_gravity_b
    roll = torch.atan2(gravity[:, 1], -gravity[:, 2])
    phase = torch.remainder(env.episode_length_buf.float() * env.step_dt / period, 1.0)
    if initial_amplitude is not None and ramp_steps > 0:
        progress = min(
            1.0,
            float(getattr(env, "common_step_counter", 0)) / float(ramp_steps),
        )
        target_amplitude = initial_amplitude + (amplitude - initial_amplitude) * progress
    else:
        target_amplitude = amplitude
    target = target_amplitude * torch.sin(2.0 * math.pi * phase)
    return torch.exp(-torch.square((roll - target) / 0.04))


def both_feet_contact(env, sensor_name: str) -> torch.Tensor:
    """Binary metric/reward for maintaining bilateral support."""
    contact = _contact_mask(env, sensor_name)
    return torch.all(contact, dim=1).float()


def scheduled_stance_contact(
    env,
    sensor_name: str,
    period: float = 0.33,
    duty_factor: float = 0.67,
) -> torch.Tensor:
    """Reward contact beneath the scheduled support foot without forcing lift-off.

    The unloading skill deliberately allows the scheduled swing foot to remain
    lightly in contact.  It therefore checks only that the designated support
    foot is carrying contact, unlike ``scheduled_foot_contact`` which also
    rewards complete swing-foot separation.
    """
    actual = _contact_mask(env, sensor_name)
    stance = desired_stance_mask(env, period, duty_factor)
    stance_count = torch.sum(stance, dim=1)
    score = torch.sum((actual & stance).float(), dim=1) / torch.clamp(
        stance_count, min=1
    )
    return score * upright_gate(env)


def scheduled_swing_unloading(
    env,
    sensor_name: str,
    period: float = 0.33,
    duty_factor: float = 0.67,
    stance_load_fraction: float = 0.75,
    std: float = 0.14,
) -> torch.Tensor:
    """Track a feasible 75/25 support-to-swing load shift without high steps.

    During double support the desired split is 50/50.  During a single-support
    phase, the support foot should carry ``stance_load_fraction`` of the net
    vertical load and the other foot remains lightly loaded.  This teaches
    unloading before full airborne stepping is requested.
    """
    if not 0.5 < duty_factor < 1.0:
        raise ValueError("duty factor must be between 0.5 and 1.0")
    if not 0.5 < stance_load_fraction < 1.0 or std <= 0.0:
        raise ValueError("invalid unloading target parameters")
    forces = env.scene[sensor_name].data.force
    if forces is None:
        raise RuntimeError(f"contact sensor {sensor_name!r} does not provide force")
    while forces.ndim > 3:
        forces = torch.linalg.vector_norm(forces, dim=-2)
    vertical = torch.abs(forces[..., 2])
    if vertical.shape[1] < 2:
        raise RuntimeError("unloading reward requires two foot contacts")
    load_fraction = vertical[:, :2] / torch.clamp(
        torch.sum(vertical[:, :2], dim=1, keepdim=True), min=1.0e-6
    )
    stance = desired_stance_mask(env, period, duty_factor)
    single_support = torch.sum(stance, dim=1, keepdim=True) == 1
    shifted_target = torch.where(
        stance,
        torch.full_like(load_fraction, stance_load_fraction),
        torch.full_like(load_fraction, 1.0 - stance_load_fraction),
    )
    target = torch.where(
        single_support,
        shifted_target,
        torch.full_like(load_fraction, 0.5),
    )
    score = torch.mean(torch.exp(-torch.square((load_fraction - target) / std)), dim=1)
    has_load = torch.sum(vertical[:, :2], dim=1) > 1.0e-3
    return torch.where(has_load, score, torch.zeros_like(score)) * upright_gate(env)


def scheduled_swing_unloading_success(
    env,
    sensor_name: str,
    period: float = 0.33,
    duty_factor: float = 0.67,
    stance_load_fraction: float = 0.75,
    std: float = 0.14,
    minimum_score: float = 0.70,
) -> torch.Tensor:
    """Binary promotion signal for sufficiently strong scheduled unloading."""
    if not 0.0 < minimum_score <= 1.0:
        raise ValueError("minimum unloading score must be in (0, 1]")
    score = scheduled_swing_unloading(
        env, sensor_name, period, duty_factor, stance_load_fraction, std
    )
    return (score >= minimum_score).float()


def lateral_support_balance(
    env,
    sensor_name: str,
    amplitude: float = 0.10,
    period: float = 2.0,
    target_imbalance: float = 0.60,
    std: float = 0.20,
    initial_imbalance: float | None = None,
    ramp_steps: int = 0,
) -> torch.Tensor:
    """Measure physical left/right load transfer while both feet stay planted.

    The contact sensor provides one net force vector per foot.  We compare the
    normalized right-minus-left vertical load to a sinusoidal target: balanced
    near the center of the transfer and approximately 80/20 at each extremum.
    This prevents a policy from passing merely by keeping both feet touching.
    """
    if amplitude <= 0.0 or period <= 0.0 or not 0.0 < target_imbalance < 1.0:
        raise ValueError("invalid weight-transfer metric parameters")
    if std <= 0.0:
        raise ValueError("load-imbalance standard deviation must be positive")
    if initial_imbalance is not None and not 0.0 <= initial_imbalance < 1.0:
        raise ValueError("initial load imbalance must be in [0, 1)")
    if ramp_steps < 0:
        raise ValueError("ramp_steps must be non-negative")
    sensor = env.scene[sensor_name]
    forces = sensor.data.force
    if forces is None:
        raise RuntimeError(f"contact sensor {sensor_name!r} does not provide force")
    # ContactSensor uses [batch, primary, slots, xyz] only when slots > 1;
    # this task configures one slot, but reduce defensively for custom robots.
    while forces.ndim > 3:
        forces = torch.linalg.vector_norm(forces, dim=-2)
    vertical = torch.abs(forces[..., 2])
    if vertical.shape[1] < 2:
        raise RuntimeError("weight-transfer metric requires two foot contacts")
    right_load = vertical[:, 0]
    left_load = vertical[:, 1]
    total_load = right_load + left_load
    actual_imbalance = (right_load - left_load) / torch.clamp(total_load, min=1.0e-6)

    phase = torch.remainder(
        env.episode_length_buf.to(torch.float32) * env.step_dt / period, 1.0
    )
    target_roll = amplitude * torch.sin(2.0 * math.pi * phase)
    if initial_imbalance is not None and ramp_steps > 0:
        progress = min(
            1.0,
            float(getattr(env, "common_step_counter", 0)) / float(ramp_steps),
        )
        imbalance = initial_imbalance + (target_imbalance - initial_imbalance) * progress
    else:
        imbalance = target_imbalance
    desired_imbalance = imbalance * target_roll / amplitude
    tracking = torch.exp(
        -torch.square((actual_imbalance - desired_imbalance) / std)
    )
    both_planted = torch.all(_contact_mask(env, sensor_name), dim=1)
    has_load = total_load > 1.0e-3
    return torch.where(both_planted & has_load, tracking, torch.zeros_like(tracking))


def step_in_place_success(
    env,
    height_sensor_name: str,
    period: float = 0.8,
    duty_factor: float = 0.6,
    minimum_height: float = 0.0125,
) -> torch.Tensor:
    """Success metric for repeated alternating swing without translation."""
    clear = mid_swing_clearance_success(
        env,
        height_sensor_name=height_sensor_name,
        minimum_height=minimum_height,
        period=period,
        duty_factor=duty_factor,
    )
    contact = scheduled_foot_contact(
        env, "feet_ground_contact", period=period, duty_factor=duty_factor
    )
    return clear * contact


def mid_swing_active(
    env,
    period: float = 0.8,
    duty_factor: float = 0.6,
) -> torch.Tensor:
    """Whether this control step contains a measured mid-swing opportunity."""
    phase = gait_phase_scalar(env, period)
    local_phase = torch.stack(
        (phase, torch.remainder(phase + 0.5, 1.0)), dim=1
    )
    swing = local_phase >= duty_factor
    progress = torch.clamp(
        (local_phase - duty_factor) / (1.0 - duty_factor), 0.0, 1.0
    )
    return torch.any(swing & (progress >= 0.25) & (progress <= 0.75), dim=1)


def full_domain_randomization(env, env_ids: torch.Tensor) -> None:
    """Apply the task's configured reset randomization envelope."""
    sim2real_domain_randomization(env, env_ids)


_SYMMETRIC_ACTUATOR_PAIRS = (
    ("right_hip_roll", "left_hip_roll"),
    ("right_thigh", "left_thigh"),
    ("right_knee", "left_knee"),
)
_SYMMETRIC_JOINT_PAIRS = (
    ("RightBaseToHip", "LeftBaseToHip", -1.0),
    ("RightHipToThigh", "LeftHipToThigh", 1.0),
    ("RightKnee", "LeftKnee", 1.0),
)


def _single_actuator(env, name: str):
    cfg = SceneEntityCfg("robot", actuator_names=(name,), preserve_order=True)
    cfg.resolve(env.scene)
    actuator_ids = cfg.actuator_ids
    if isinstance(actuator_ids, list):
        if len(actuator_ids) != 1:
            raise ValueError(f"Expected one actuator for {name!r}, got {actuator_ids}")
        actuator_id = actuator_ids[0]
    elif isinstance(actuator_ids, slice):
        raise ValueError(f"Actuator selection for {name!r} did not resolve to one item")
    else:
        actuator_id = actuator_ids
    return cfg, env.scene["robot"].actuators[actuator_id]


def _copy_paired_actuator_fields(env, env_ids: torch.Tensor, right, left, fields: tuple[str, ...]) -> None:
    """Copy one sampled actuator realization to its morphological mirror."""
    right_ids = torch.as_tensor(
        right.global_ctrl_ids, device=env.device, dtype=torch.long
    ).reshape(-1)
    left_ids = torch.as_tensor(
        left.global_ctrl_ids, device=env.device, dtype=torch.long
    ).reshape(-1)
    if right_ids.numel() != left_ids.numel():
        raise ValueError("mirrored actuators must expose equally sized control vectors")
    for field in fields:
        values = getattr(env.sim.model, field)[env_ids[:, None], right_ids]
        getattr(env.sim.model, field)[env_ids[:, None], left_ids] = values


def symmetric_domain_randomization(env, env_ids: torch.Tensor) -> None:
    """Sim-to-real randomization preserving LeggyBot's left/right MDP symmetry.

    The symmetry-augmentation phase intentionally samples the same PD gain,
    effort limit and latency for each right/left joint pair.  It retains global
    friction and torso-inertia variation, but excludes pairwise-inconsistent
    leg-inertia perturbations.  A later robustness phase can deliberately add
    measured independent mismatch after the nominal symmetric gait is learned.
    """
    scale = sim2real_randomization_scale(env)
    robot = SceneEntityCfg("robot")
    feet = SceneEntityCfg(
        "robot", geom_names=("right_foot_collision", "left_foot_collision"), preserve_order=True
    )
    robot.resolve(env.scene)
    feet.resolve(env.scene)

    # One terrain/friction realization per environment already preserves the
    # reflection.  Restrict inertia variation to a reflection-symmetric torso
    # perturbation: no lateral CoM offset or xy/yz inertial cross terms.
    env_mdp.dr.geom_friction(
        env,
        env_ids,
        ranges=(0.9 - 0.4 * scale, 0.9 + 0.3 * scale),
        operation="abs",
        shared_random=True,
        asset_cfg=feet,
    )
    env_mdp.dr.pseudo_inertia(
        env,
        env_ids,
        alpha_range=(-0.04 * scale, 0.04 * scale),
        d_range=(-0.02 * scale, 0.02 * scale),
        s13_range=(-0.01 * scale, 0.01 * scale),
        t1_range=(-0.005 * scale, 0.005 * scale),
        t2_range=(0.0, 0.0),
        t3_range=(-0.005 * scale, 0.005 * scale),
        asset_cfg=SceneEntityCfg("robot", body_names=("BaseLink",), preserve_order=True),
    )

    for right_name, left_name in _SYMMETRIC_ACTUATOR_PAIRS:
        right_cfg, right = _single_actuator(env, right_name)
        _, left = _single_actuator(env, left_name)
        env_mdp.dr.pd_gains(
            env,
            env_ids,
            kp_range=(1.0 - 0.10 * scale, 1.0 + 0.10 * scale),
            kd_range=(1.0 - 0.10 * scale, 1.0 + 0.10 * scale),
            operation="scale",
            asset_cfg=right_cfg,
        )
        _copy_paired_actuator_fields(
            env, env_ids, right, left, ("actuator_gainprm", "actuator_biasprm")
        )
        env_mdp.dr.effort_limits(
            env,
            env_ids,
            effort_limit_range=(1.0 - 0.10 * scale, 1.0 + 0.05 * scale),
            operation="scale",
            asset_cfg=right_cfg,
        )
        _copy_paired_actuator_fields(env, env_ids, right, left, ("actuator_forcerange",))
        max_lag = 2 if scale >= 0.5 else (1 if scale > 0.0 else 0)
        lags = torch.randint(0, max_lag + 1, (len(env_ids),), device=env.device)
        right.set_lags(lags, env_ids)
        left.set_lags(lags, env_ids)

    # Encoder offsets are joint-coordinate quantities.  Hip-roll coordinates
    # reverse under reflection, while pitch and knee coordinates do not.
    for right_name, left_name, sign in _SYMMETRIC_JOINT_PAIRS:
        right_cfg = SceneEntityCfg("robot", joint_names=(right_name,), preserve_order=True)
        left_cfg = SceneEntityCfg("robot", joint_names=(left_name,), preserve_order=True)
        right_cfg.resolve(env.scene)
        left_cfg.resolve(env.scene)
        env_mdp.dr.encoder_bias(
            env, env_ids, bias_range=(-0.01 * scale, 0.01 * scale), asset_cfg=right_cfg
        )
        right_ids = torch.as_tensor(
            right_cfg.joint_ids, device=env.device, dtype=torch.long
        ).reshape(-1)
        left_ids = torch.as_tensor(
            left_cfg.joint_ids, device=env.device, dtype=torch.long
        ).reshape(-1)
        env.scene["robot"].data.encoder_bias[env_ids[:, None], left_ids] = (
            sign * env.scene["robot"].data.encoder_bias[env_ids[:, None], right_ids]
        )


def upright_gate(
    env,
    minimum_cosine: float = 0.5,
    full_reward_cosine: float = 0.9,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Fade gait guidance to zero as the torso approaches its fall limit."""
    if not -1.0 <= minimum_cosine < full_reward_cosine <= 1.0:
        raise ValueError("invalid upright gate cosine bounds")
    asset: Entity = env.scene[asset_cfg.name]
    upright_cosine = -asset.data.projected_gravity_b[:, 2]
    return torch.clamp(
        (upright_cosine - minimum_cosine)
        / (full_reward_cosine - minimum_cosine),
        0.0,
        1.0,
    )


def gait_phase_scalar(env, period: float = 0.8) -> torch.Tensor:
    """Per-environment gait phase in [0, 1), reset with each episode."""
    if period <= 0.0:
        raise ValueError("gait period must be positive")
    elapsed = env.episode_length_buf.to(torch.float32) * env.step_dt
    return torch.remainder(elapsed / period, 1.0)


def gait_clock(env, period: float = 0.8) -> torch.Tensor:
    """Deployable sine/cosine gait clock observation."""
    angle = 2.0 * math.pi * gait_phase_scalar(env, period)
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=1)


def smooth_swing_probability(
    env,
    period: float = 0.33,
    duty_factor: float = 0.70,
    transition_fraction: float = 0.04,
) -> torch.Tensor:
    """Return a soft right/left swing schedule instead of hard contact labels.

    The gait is phase periodic, with each leg offset by half a cycle.  The
    logistic edges are a compact deterministic analogue of the smooth phase
    probabilities used by periodic-reward-composition locomotion work: being
    a few milliseconds early or late is not treated as a different gait.
    """
    if not 0.5 < duty_factor < 1.0:
        raise ValueError("duty factor must be between 0.5 and 1.0")
    if not 0.0 < transition_fraction < min(duty_factor, 1.0 - duty_factor):
        raise ValueError("transition fraction must fit inside stance and swing")
    phase = gait_phase_scalar(env, period)
    local_phase = torch.stack((phase, torch.remainder(phase + 0.5, 1.0)), dim=1)
    swing_starts = torch.sigmoid((local_phase - duty_factor) / transition_fraction)
    swing_ends = torch.sigmoid((1.0 - local_phase) / transition_fraction)
    return swing_starts * swing_ends


def _foot_force_magnitudes(env, sensor_name: str) -> torch.Tensor:
    """Return the net force magnitude under right and left feet in newtons."""
    forces = env.scene[sensor_name].data.force
    if forces is None:
        raise RuntimeError(f"contact sensor {sensor_name!r} does not provide force")
    # A sensor with multiple contact slots stores [env, foot, slot, xyz].
    # Combine slots vectorially before taking the magnitude.
    while forces.ndim > 3:
        forces = torch.sum(forces, dim=-2)
    if forces.shape[1] < 2:
        raise RuntimeError("periodic walking requires right and left foot forces")
    return torch.linalg.vector_norm(forces[:, :2], dim=-1)


def periodic_swing_force_cost(
    env,
    sensor_name: str,
    period: float = 0.33,
    duty_factor: float = 0.70,
    transition_fraction: float = 0.04,
    force_scale: float = 4.0,
) -> torch.Tensor:
    """Cost swing-foot ground force, but deliberately do not score stance force.

    This is the key phase-composed term: a foot must unload to become a swing
    foot, yet its support counterpart is free to carry whatever force keeps the
    robot upright.  The weighted average is normalized by the phase weight, so
    double-support time cannot make a planted-foot solution look successful.
    """
    if force_scale <= 0.0:
        raise ValueError("force scale must be positive")
    swing = smooth_swing_probability(env, period, duty_factor, transition_fraction)
    normalized_force = 1.0 - torch.exp(-torch.square(_foot_force_magnitudes(env, sensor_name) / force_scale))
    return torch.sum(swing * normalized_force, dim=1) / torch.clamp(
        torch.sum(swing, dim=1), min=1.0e-6
    )


def periodic_stance_foot_speed_cost(
    env,
    period: float = 0.33,
    duty_factor: float = 0.70,
    transition_fraction: float = 0.04,
    speed_scale: float = 0.06,
    asset_cfg: SceneEntityCfg = _FOOT_SITES_CFG,
) -> torch.Tensor:
    """Cost horizontal foot motion during stance, while leaving swing free."""
    if speed_scale <= 0.0:
        raise ValueError("speed scale must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    foot_speed = torch.linalg.vector_norm(
        asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1
    )
    if foot_speed.shape[1] != 2:
        raise RuntimeError("periodic walking requires exactly right and left foot sites")
    stance = 1.0 - smooth_swing_probability(
        env, period, duty_factor, transition_fraction
    )
    normalized_speed = 1.0 - torch.exp(-torch.square(foot_speed / speed_scale))
    return torch.sum(stance * normalized_speed, dim=1) / torch.clamp(
        torch.sum(stance, dim=1), min=1.0e-6
    )


def periodic_foot_exchange(
    env,
    period: float = 0.33,
    separation: float = 0.012,
    std: float = 0.018,
    asset_cfg: SceneEntityCfg = _FOOT_SITES_CFG,
) -> torch.Tensor:
    """Small phase-locked fore/aft foot exchange that prevents a fixed scissor.

    This is a task-space regularizer, not a joint trajectory: it asks the two
    feet to exchange a realistic 24 mm peak-to-peak fore/aft separation.  The
    target is comparable to 0.08 m/s over one 0.33 s leg cycle and prevents the
    old failure mode where one foot remained permanently ahead of the other.
    """
    if separation <= 0.0 or std <= 0.0:
        raise ValueError("foot-exchange separation and standard deviation must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    foot_x = asset.data.site_pos_w[:, asset_cfg.site_ids, 0]
    if foot_x.shape[1] != 2:
        raise RuntimeError("periodic walking requires exactly right and left foot sites")
    phase = gait_phase_scalar(env, period)
    desired_difference = separation * torch.sin(2.0 * math.pi * phase)
    actual_difference = foot_x[:, 0] - foot_x[:, 1]
    return torch.exp(-torch.square((actual_difference - desired_difference) / std)) * upright_gate(env)


def world_forward_velocity_tracking(
    env,
    command_name: str,
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track the command in the fixed world frame, not the turning body frame."""
    if std <= 0.0:
        raise ValueError("world velocity standard deviation must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command {command_name!r} not found"
    error = asset.data.root_link_lin_vel_w[:, :2] - command[:, :2]
    return torch.exp(-torch.sum(torch.square(error / std), dim=1))


def episode_average_world_forward_speed(
    env,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Net world-X displacement divided by elapsed episode time in m/s."""
    asset: Entity = env.scene[asset_cfg.name]
    elapsed = env.episode_length_buf.to(torch.float32) * env.step_dt
    displacement_x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return displacement_x / torch.clamp(elapsed, min=0.20)


def episode_average_world_progress(
    env,
    command_name: str,
    std: float = 0.035,
    warmup_s: float = 0.50,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward sustained net displacement rather than a local rocking velocity."""
    if std <= 0.0 or warmup_s <= 0.0:
        raise ValueError("world-progress standard deviation and warmup must be positive")
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command {command_name!r} not found"
    elapsed = env.episode_length_buf.to(torch.float32) * env.step_dt
    average_speed = episode_average_world_forward_speed(env, asset_cfg)
    score = torch.exp(-torch.square((average_speed - command[:, 0]) / std))
    return score * torch.clamp(elapsed / warmup_s, max=1.0)


class _CycleNetMotionTerm(ManagerTermBase):
    """Track net world-Y and yaw change over completed gait cycles.

    A biped must move its CoM sideways and may counter-rotate its torso within
    a step.  These buffers therefore retain only *net* change from the last
    completed cycle, rather than penalizing the instantaneous motion that
    makes normal support transfer possible.
    """

    def __init__(self, cfg, env) -> None:
        del cfg
        super().__init__(env)
        self._last_episode_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._cycle_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._cycle_start_y = torch.zeros(self.num_envs, device=self.device)
        self._cycle_start_heading = torch.zeros(self.num_envs, device=self.device)
        self._completed_lateral_speed = torch.zeros(self.num_envs, device=self.device)
        self._completed_yaw_rate = torch.zeros(self.num_envs, device=self.device)

    @staticmethod
    def _wrap(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._last_episode_step[env_ids] = -1
        self._completed_lateral_speed[env_ids] = 0.0
        self._completed_yaw_rate[env_ids] = 0.0

    def _completed_cycle_rates(
        self, env, *, period: float, asset_cfg: SceneEntityCfg
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if period <= 0.0:
            raise ValueError("gait period must be positive")
        asset: Entity = env.scene[asset_cfg.name]
        steps = env.episode_length_buf
        cycle_index = torch.floor(steps.to(torch.float32) * env.step_dt / period).long()
        current_y = asset.data.root_link_pos_w[:, 1]
        current_heading = self._wrap(asset.data.heading_w)

        # A manager reset normally handles this case. The additional monotonic
        # check makes the term safe with explicit/asynchronous resets too.
        new_episode = (self._last_episode_step < 0) | (steps < self._last_episode_step)
        if torch.any(new_episode):
            self._cycle_index[new_episode] = cycle_index[new_episode]
            self._cycle_start_y[new_episode] = current_y[new_episode]
            self._cycle_start_heading[new_episode] = current_heading[new_episode]
            self._completed_lateral_speed[new_episode] = 0.0
            self._completed_yaw_rate[new_episode] = 0.0

        completed_cycle = (cycle_index != self._cycle_index) & ~new_episode
        if torch.any(completed_cycle):
            self._completed_lateral_speed[completed_cycle] = (
                current_y[completed_cycle] - self._cycle_start_y[completed_cycle]
            ) / period
            self._completed_yaw_rate[completed_cycle] = self._wrap(
                current_heading[completed_cycle]
                - self._cycle_start_heading[completed_cycle]
            ) / period
            self._cycle_index[completed_cycle] = cycle_index[completed_cycle]
            self._cycle_start_y[completed_cycle] = current_y[completed_cycle]
            self._cycle_start_heading[completed_cycle] = current_heading[completed_cycle]

        self._last_episode_step.copy_(steps)
        return self._completed_lateral_speed, self._completed_yaw_rate

    def _gate(
        self,
        env,
        *,
        period: float,
        heading_std: float,
        cycle_lateral_velocity_std: float,
        cycle_yaw_rate_std: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        if min(heading_std, cycle_lateral_velocity_std, cycle_yaw_rate_std) <= 0.0:
            raise ValueError("cycle straightness standard deviations must be positive")
        lateral_speed, yaw_rate = self._completed_cycle_rates(
            env, period=period, asset_cfg=asset_cfg
        )
        heading = self._wrap(env.scene[asset_cfg.name].data.heading_w)
        squared_error = (
            torch.square(heading / heading_std)
            + torch.square(lateral_speed / cycle_lateral_velocity_std)
            + torch.square(yaw_rate / cycle_yaw_rate_std)
        )
        return torch.exp(-squared_error)


class CycleGatedWorldForwardVelocity(_CycleNetMotionTerm):
    """World-forward tracking gated by completed-cycle lateral/yaw drift."""

    def __call__(
        self,
        env,
        command_name: str,
        std: float,
        period: float,
        heading_std: float,
        cycle_lateral_velocity_std: float,
        cycle_yaw_rate_std: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        return world_forward_velocity_tracking(
            env, command_name=command_name, std=std, asset_cfg=asset_cfg
        ) * self._gate(
            env,
            period=period,
            heading_std=heading_std,
            cycle_lateral_velocity_std=cycle_lateral_velocity_std,
            cycle_yaw_rate_std=cycle_yaw_rate_std,
            asset_cfg=asset_cfg,
        )


class CycleGatedEpisodeWorldProgress(_CycleNetMotionTerm):
    """Net world-X progress gated by completed-cycle lateral/yaw drift."""

    def __call__(
        self,
        env,
        command_name: str,
        std: float,
        warmup_s: float,
        period: float,
        heading_std: float,
        cycle_lateral_velocity_std: float,
        cycle_yaw_rate_std: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        return episode_average_world_progress(
            env,
            command_name=command_name,
            std=std,
            warmup_s=warmup_s,
            asset_cfg=asset_cfg,
        ) * self._gate(
            env,
            period=period,
            heading_std=heading_std,
            cycle_lateral_velocity_std=cycle_lateral_velocity_std,
            cycle_yaw_rate_std=cycle_yaw_rate_std,
            asset_cfg=asset_cfg,
        )


class CycleNetMotionGate(_CycleNetMotionTerm):
    """Diagnostic value of the completed-cycle straightness gate."""

    def __call__(
        self,
        env,
        period: float,
        heading_std: float,
        cycle_lateral_velocity_std: float,
        cycle_yaw_rate_std: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        return self._gate(
            env,
            period=period,
            heading_std=heading_std,
            cycle_lateral_velocity_std=cycle_lateral_velocity_std,
            cycle_yaw_rate_std=cycle_yaw_rate_std,
            asset_cfg=asset_cfg,
        )


class CycleNetLateralVelocityCost(_CycleNetMotionTerm):
    """Bounded penalty on net, not instantaneous, lateral velocity."""

    def __call__(
        self,
        env,
        period: float,
        std: float,
        max_cost: float = 4.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        if std <= 0.0 or max_cost <= 0.0:
            raise ValueError("cycle lateral cost parameters must be positive")
        lateral_speed, _ = self._completed_cycle_rates(
            env, period=period, asset_cfg=asset_cfg
        )
        return torch.clamp(torch.square(lateral_speed / std), max=max_cost)


class CycleNetYawRateCost(_CycleNetMotionTerm):
    """Bounded penalty on yaw accumulated across a complete gait cycle."""

    def __call__(
        self,
        env,
        period: float,
        std: float,
        max_cost: float = 4.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        if std <= 0.0 or max_cost <= 0.0:
            raise ValueError("cycle yaw cost parameters must be positive")
        _, yaw_rate = self._completed_cycle_rates(
            env, period=period, asset_cfg=asset_cfg
        )
        return torch.clamp(torch.square(yaw_rate / std), max=max_cost)


def motion_rate_safety_cost(
    env,
    lateral_soft_limit: float = 0.12,
    lateral_scale: float = 0.04,
    yaw_soft_limit: float = 0.90,
    yaw_scale: float = 0.35,
    max_cost: float = 4.0,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bound only unusually fast lateral or yaw motion, leaving normal sway free."""
    if min(lateral_soft_limit, lateral_scale, yaw_soft_limit, yaw_scale, max_cost) <= 0.0:
        raise ValueError("motion-rate safety parameters must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command {command_name!r} not found"
    lateral_excess = torch.relu(
        torch.abs(asset.data.root_link_lin_vel_w[:, 1]) - lateral_soft_limit
    ) / lateral_scale
    yaw_excess = torch.relu(
        torch.abs(asset.data.root_link_ang_vel_b[:, 2] - command[:, 2]) - yaw_soft_limit
    ) / yaw_scale
    return torch.clamp(torch.square(lateral_excess) + torch.square(yaw_excess), max=max_cost)


def straight_world_motion_gate(
    env,
    heading_std: float = 0.10,
    lateral_velocity_std: float = 0.025,
    yaw_rate_std: float = 0.35,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return one only for forward motion that remains a straight world path.

    A separate negative path reward can be traded against a large positive
    forward reward.  This multiplicative gate cannot: a policy earns little
    forward credit while it is turning or moving laterally.  Roll is purposely
    absent because lateral CoM motion is part of a bipedal gait.
    """
    if min(heading_std, lateral_velocity_std, yaw_rate_std) <= 0.0:
        raise ValueError("straight-world gate standard deviations must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    heading = torch.atan2(torch.sin(asset.data.heading_w), torch.cos(asset.data.heading_w))
    lateral_velocity = asset.data.root_link_lin_vel_w[:, 1]
    yaw_rate = asset.data.root_link_ang_vel_b[:, 2]
    squared_error = (
        torch.square(heading / heading_std)
        + torch.square(lateral_velocity / lateral_velocity_std)
        + torch.square(yaw_rate / yaw_rate_std)
    )
    return torch.exp(-squared_error)


def gated_world_forward_velocity_tracking(
    env,
    command_name: str,
    std: float = 0.05,
    heading_std: float = 0.10,
    lateral_velocity_std: float = 0.025,
    yaw_rate_std: float = 0.35,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track world-X speed, giving credit only to a straight trajectory."""
    return world_forward_velocity_tracking(
        env, command_name=command_name, std=std, asset_cfg=asset_cfg
    ) * straight_world_motion_gate(
        env,
        heading_std=heading_std,
        lateral_velocity_std=lateral_velocity_std,
        yaw_rate_std=yaw_rate_std,
        asset_cfg=asset_cfg,
    )


def gated_episode_average_world_progress(
    env,
    command_name: str,
    std: float = 0.035,
    warmup_s: float = 0.50,
    heading_std: float = 0.10,
    lateral_velocity_std: float = 0.025,
    yaw_rate_std: float = 0.35,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward sustained net progress only while the instantaneous path is straight."""
    return episode_average_world_progress(
        env, command_name=command_name, std=std, warmup_s=warmup_s, asset_cfg=asset_cfg
    ) * straight_world_motion_gate(
        env,
        heading_std=heading_std,
        lateral_velocity_std=lateral_velocity_std,
        yaw_rate_std=yaw_rate_std,
        asset_cfg=asset_cfg,
    )


def world_lateral_velocity_cost(
    env,
    std: float = 0.025,
    max_cost: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bounded cost on sideways velocity in the fixed world frame."""
    if std <= 0.0 or max_cost <= 0.0:
        raise ValueError("world lateral-velocity cost parameters must be positive")
    lateral_velocity = env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, 1]
    return torch.clamp(torch.square(lateral_velocity / std), max=max_cost)


def zero_yaw_rate_cost(
    env,
    std: float = 0.35,
    max_cost: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bounded zero-yaw-rate cost for a straight, non-turning walking command."""
    if std <= 0.0 or max_cost <= 0.0:
        raise ValueError("zero-yaw-rate cost parameters must be positive")
    yaw_rate = env.scene[asset_cfg.name].data.root_link_ang_vel_b[:, 2]
    return torch.clamp(torch.square(yaw_rate / std), max=max_cost)


def periodic_walk_success(
    env,
    sensor_name: str,
    command_name: str,
    period: float = 0.33,
    duty_factor: float = 0.70,
    transition_fraction: float = 0.04,
    minimum_speed_ratio: float = 0.70,
    maximum_speed_ratio: float = 1.35,
    maximum_swing_force_cost: float = 0.35,
    maximum_stance_speed_cost: float = 0.35,
    maximum_heading_error: float = 0.15,
    maximum_lateral_deviation: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    foot_site_cfg: SceneEntityCfg = _FOOT_SITES_CFG,
) -> torch.Tensor:
    """Strict held-out pass signal for true periodic walking."""
    if not 0.0 < minimum_speed_ratio < maximum_speed_ratio:
        raise ValueError("invalid periodic-walk speed-ratio bounds")
    if maximum_heading_error <= 0.0 or maximum_lateral_deviation <= 0.0:
        raise ValueError("periodic-walk path bounds must be positive")
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command {command_name!r} not found"
    average_speed = episode_average_world_forward_speed(env, asset_cfg)
    speed_ratio = average_speed / torch.clamp(command[:, 0], min=1.0e-4)
    speed_ok = (speed_ratio >= minimum_speed_ratio) & (speed_ratio <= maximum_speed_ratio)
    swing_ok = periodic_swing_force_cost(
        env,
        sensor_name,
        period=period,
        duty_factor=duty_factor,
        transition_fraction=transition_fraction,
    ) <= maximum_swing_force_cost
    stance_ok = (
        periodic_stance_foot_speed_cost(
            env,
            period=period,
            duty_factor=duty_factor,
            transition_fraction=transition_fraction,
            asset_cfg=foot_site_cfg,
        )
        <= maximum_stance_speed_cost
    )
    asset: Entity = env.scene[asset_cfg.name]
    heading = torch.atan2(torch.sin(asset.data.heading_w), torch.cos(asset.data.heading_w))
    lateral_displacement = torch.abs(
        asset.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    )
    path_ok = (torch.abs(heading) <= maximum_heading_error) & (
        lateral_displacement <= maximum_lateral_deviation
    )
    return (speed_ok & swing_ok & stance_ok & path_ok).float()


def periodic_walk_evaluation_active(env, warmup_s: float = 2.0) -> torch.Tensor:
    """Ignore start-up transient before judging a net-displacement gait."""
    if warmup_s <= 0.0:
        raise ValueError("evaluation warmup must be positive")
    elapsed = env.episode_length_buf.to(torch.float32) * env.step_dt
    return elapsed >= warmup_s


def actuator_effort_cost(
    env,
    effort_scale: float = 1.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bounded actuator-force cost, matching the paper's energy regularizer."""
    if effort_scale <= 0.0:
        raise ValueError("effort scale must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    effort = torch.abs(asset.data.qfrc_actuator)
    return torch.mean(1.0 - torch.exp(-effort / effort_scale), dim=1)


def desired_stance_mask(
    env, period: float = 0.8, duty_factor: float = 0.6
) -> torch.Tensor:
    """Desired right/left stance mask, with the legs half a cycle apart."""
    if not 0.5 < duty_factor < 1.0:
        raise ValueError("duty factor must be between 0.5 and 1.0")
    phase = gait_phase_scalar(env, period)
    local_phase = torch.stack((phase, torch.remainder(phase + 0.5, 1.0)), dim=1)
    return local_phase < duty_factor


def _contact_mask(env, sensor_name: str) -> torch.Tensor:
    found = env.scene[sensor_name].data.found
    assert found is not None
    contact = found > 0
    while contact.ndim > 2:
        contact = torch.any(contact, dim=-1)
    return contact


def scheduled_foot_contact(
    env,
    sensor_name: str,
    period: float = 0.8,
    duty_factor: float = 0.6,
) -> torch.Tensor:
    """Fraction of feet whose contact agrees with the gait schedule."""
    actual = _contact_mask(env, sensor_name)
    desired = desired_stance_mask(env, period, duty_factor)
    score = torch.mean((actual == desired).to(torch.float32), dim=1)
    return score * upright_gate(env)


def exact_scheduled_foot_contact(
    env,
    sensor_name: str,
    period: float = 0.8,
    duty_factor: float = 0.6,
) -> torch.Tensor:
    """Binary contact agreement for a scheduled single-support instant."""
    actual = _contact_mask(env, sensor_name)
    desired = desired_stance_mask(env, period, duty_factor)
    return torch.all(actual == desired, dim=1).float()


def single_support_active(
    env,
    period: float = 0.8,
    duty_factor: float = 0.6,
) -> torch.Tensor:
    """Select the phase intervals that genuinely require alternating feet."""
    desired = desired_stance_mask(env, period, duty_factor)
    return torch.sum(desired, dim=1) == 1


def scheduled_swing_clearance(
    env,
    height_sensor_name: str,
    target_height: float = 0.02,
    std: float = 0.0075,
    period: float = 0.8,
    duty_factor: float = 0.6,
) -> torch.Tensor:
    """Reward swing feet near target clearance; double support is neutral-good."""
    if target_height <= 0.0 or std <= 0.0:
        raise ValueError("clearance target and standard deviation must be positive")
    heights = env.scene[height_sensor_name].data.heights
    phase = gait_phase_scalar(env, period)
    local_phase = torch.stack(
        (phase, torch.remainder(phase + 0.5, 1.0)), dim=1
    )
    swing = local_phase >= duty_factor
    swing_progress = torch.clamp(
        (local_phase - duty_factor) / (1.0 - duty_factor), 0.0, 1.0
    )
    # Zero clearance at toe-off/touchdown and target_height at mid-swing.
    desired_height = target_height * torch.sin(math.pi * swing_progress)
    score = torch.exp(-torch.square((heights - desired_height) / std))
    swing_count = torch.sum(swing, dim=1)
    swing_score = torch.sum(score * swing, dim=1) / torch.clamp(swing_count, min=1)
    # Double-support frames provide no swing-clearance evidence and are neutral
    # rather than automatic successes.
    score = torch.where(swing_count > 0, swing_score, torch.zeros_like(swing_score))
    return score * upright_gate(env)


def mid_swing_clearance_success(
    env,
    height_sensor_name: str,
    minimum_height: float = 0.0125,
    period: float = 0.8,
    duty_factor: float = 0.6,
) -> torch.Tensor:
    """Binary mid-swing height metric used by the curriculum."""
    if minimum_height <= 0.0:
        raise ValueError("minimum swing height must be positive")
    heights = env.scene[height_sensor_name].data.heights
    phase = gait_phase_scalar(env, period)
    local_phase = torch.stack((phase, torch.remainder(phase + 0.5, 1.0)), dim=1)
    swing = local_phase >= duty_factor
    progress = torch.clamp((local_phase - duty_factor) / (1.0 - duty_factor), 0.0, 1.0)
    mid_swing = swing & (progress >= 0.25) & (progress <= 0.75)
    count = torch.sum(mid_swing, dim=1)
    success = (heights >= minimum_height) & mid_swing
    return torch.sum(success.float(), dim=1) / torch.clamp(count, min=1)


def mid_swing_peak_height(
    env,
    height_sensor_name: str,
    foot_index: int,
    period: float = 0.8,
    duty_factor: float = 0.6,
) -> torch.Tensor:
    """Instantaneous mid-swing foot height in millimetres.

    Configure this as a metrics term with ``reduce="max"`` to obtain the
    per-episode peak. Values outside that foot's mid-swing interval are zero.
    """
    if foot_index not in (0, 1):
        raise ValueError("foot_index must be 0 (right) or 1 (left)")
    heights = env.scene[height_sensor_name].data.heights
    phase = gait_phase_scalar(env, period)
    local_phase = torch.remainder(phase + 0.5 * foot_index, 1.0)
    swing = local_phase >= duty_factor
    progress = torch.clamp((local_phase - duty_factor) / (1.0 - duty_factor), 0.0, 1.0)
    mid_swing = swing & (progress >= 0.25) & (progress <= 0.75)
    return torch.where(mid_swing, heights[:, foot_index] * 1000.0, torch.zeros_like(phase))


def gait_reference_joint_positions(
    env,
    period: float = 0.8,
    duty_factor: float = 0.6,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """A smooth, kinematically validated right/left hip-and-knee reference.

    At mid-swing, the offsets (hip pitch +0.57 rad, knee -0.995 rad) place
    LeggyBot's foot approximately 15 mm forward and 15 mm above its nominal
    location. Stance sweeps the hip gently backward as the base advances.
    """
    asset: Entity = env.scene[asset_cfg.name]
    default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reference = default.clone()
    phase = gait_phase_scalar(env, period)
    phases = (phase, torch.remainder(phase + 0.5, 1.0))

    for leg_index, local_phase in enumerate(phases):
        stance_progress = torch.clamp(local_phase / duty_factor, 0.0, 1.0)
        stance_hip = 0.10 - 0.20 * stance_progress

        swing_progress = torch.clamp(
            (local_phase - duty_factor) / (1.0 - duty_factor), 0.0, 1.0
        )
        swing_bump = torch.sin(math.pi * swing_progress)
        swing_hip = -0.10 + 0.20 * swing_progress + 0.57 * swing_bump
        swing_knee = -0.995 * swing_bump
        in_stance = local_phase < duty_factor

        hip_index = 1 + 3 * leg_index
        knee_index = 2 + 3 * leg_index
        reference[:, hip_index] += torch.where(in_stance, stance_hip, swing_hip)
        reference[:, knee_index] += torch.where(
            in_stance, torch.zeros_like(swing_knee), swing_knee
        )
    return reference


def track_gait_reference(
    env,
    std: float = 0.30,
    period: float = 0.8,
    duty_factor: float = 0.6,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Exponential joint-space tracking of the lightweight gait reference."""
    if std <= 0.0:
        raise ValueError("reference tracking standard deviation must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    actual = asset.data.joint_pos[:, asset_cfg.joint_ids]
    reference = gait_reference_joint_positions(
        env, period=period, duty_factor=duty_factor, asset_cfg=asset_cfg
    )
    error = torch.mean(torch.square(actual - reference), dim=1)
    return torch.exp(-error / std**2) * upright_gate(env, asset_cfg=asset_cfg)


def _smoothstep(progress: torch.Tensor) -> torch.Tensor:
    """Cubic phase interpolation with zero velocity at its endpoints."""
    progress = torch.clamp(progress, 0.0, 1.0)
    return progress * progress * (3.0 - 2.0 * progress)


def animated_walk_reference_joint_positions(
    env,
    period: float = 0.33,
    duty_factor: float = 0.70,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Short-support, low-clearance animated reference for 0.08 m/s walking.

    This is deliberately a joint animation, not an open-loop controller. The
    feet spend 70% of each leg cycle in stance. During the short swing interval
    the hip advances smoothly and the knee extends only 0.12 rad, keeping the
    implied clearance small and the peak reference speed below the STS3215
    no-load speed at a 0.33 s period.
    """
    if not 0.5 < duty_factor < 1.0:
        raise ValueError("duty factor must be between 0.5 and 1.0")
    asset: Entity = env.scene[asset_cfg.name]
    reference = asset.data.default_joint_pos[:, asset_cfg.joint_ids].clone()
    phase = gait_phase_scalar(env, period)
    for leg_index, local_phase in enumerate(
        (phase, torch.remainder(phase + 0.5, 1.0))
    ):
        stance_progress = _smoothstep(local_phase / duty_factor)
        swing_progress = _smoothstep(
            (local_phase - duty_factor) / (1.0 - duty_factor)
        )
        in_stance = local_phase < duty_factor
        # +0.06 -> -0.06 rad stance sweep, then smooth return during swing.
        stance_hip = 0.06 - 0.12 * stance_progress
        swing_hip = -0.06 + 0.12 * swing_progress
        # On this CAD model positive knee motion raises the foot.  This sign
        # was checked directly with native MuJoCo forward kinematics: a
        # negative knee bump lowers the foot into the floor instead.
        swing_knee = 0.12 * torch.sin(math.pi * swing_progress)
        hip_index, knee_index = 1 + 3 * leg_index, 2 + 3 * leg_index
        reference[:, hip_index] += torch.where(in_stance, stance_hip, swing_hip)
        reference[:, knee_index] += torch.where(
            in_stance, torch.zeros_like(swing_knee), swing_knee
        )
    return reference


def conservative_shuffle_reference_joint_positions(
    env,
    period: float = 0.33,
    duty_factor: float = 0.82,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Contact-biased 0.08 m/s shuffling reference with only tiny knee motion.

    The intent is to translate the feet fore/aft using thigh and knee motion,
    while retaining bilateral contact whenever possible. A 0.050 rad knee bump
    provides only enough unloading authority for a small slip-free shuffle; it
    is not a high-stepping trajectory.
    """
    if not 0.5 < duty_factor < 1.0:
        raise ValueError("duty factor must be between 0.5 and 1.0")
    asset: Entity = env.scene[asset_cfg.name]
    reference = asset.data.default_joint_pos[:, asset_cfg.joint_ids].clone()
    phase = gait_phase_scalar(env, period)
    for leg_index, local_phase in enumerate(
        (phase, torch.remainder(phase + 0.5, 1.0))
    ):
        stance_progress = _smoothstep(local_phase / duty_factor)
        swing_progress = _smoothstep(
            (local_phase - duty_factor) / (1.0 - duty_factor)
        )
        in_stance = local_phase < duty_factor
        stance_hip = 0.040 - 0.080 * stance_progress
        swing_hip = -0.040 + 0.080 * swing_progress
        swing_knee = 0.050 * torch.sin(math.pi * swing_progress)
        hip_index, knee_index = 1 + 3 * leg_index, 2 + 3 * leg_index
        reference[:, hip_index] += torch.where(in_stance, stance_hip, swing_hip)
        reference[:, knee_index] += torch.where(
            in_stance, torch.zeros_like(swing_knee), swing_knee
        )
    return reference


def unloading_walk_reference_joint_positions(
    env,
    period: float = 0.33,
    duty_factor: float = 0.67,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Low-clearance reference that deliberately engages thigh and knee pitch.

    The 0.18-rad thigh sweep and 0.08-rad knee peak are substantially larger
    than the conservative shuffle, yet remain inside the current STS3215
    speed envelope at the 0.33-second cycle.  Native forward kinematics gives
    only a few millimetres of clearance at mid-swing: this is an unloading
    motion, not a high-stepping trajectory.
    """
    if not 0.5 < duty_factor < 1.0:
        raise ValueError("duty factor must be between 0.5 and 1.0")
    asset: Entity = env.scene[asset_cfg.name]
    reference = asset.data.default_joint_pos[:, asset_cfg.joint_ids].clone()
    phase = gait_phase_scalar(env, period)
    for leg_index, local_phase in enumerate(
        (phase, torch.remainder(phase + 0.5, 1.0))
    ):
        stance_progress = _smoothstep(local_phase / duty_factor)
        swing_progress = _smoothstep(
            (local_phase - duty_factor) / (1.0 - duty_factor)
        )
        in_stance = local_phase < duty_factor
        stance_hip = 0.090 - 0.180 * stance_progress
        swing_hip = -0.090 + 0.180 * swing_progress
        swing_knee = 0.080 * torch.sin(math.pi * swing_progress)
        hip_index, knee_index = 1 + 3 * leg_index, 2 + 3 * leg_index
        reference[:, hip_index] += torch.where(in_stance, stance_hip, swing_hip)
        reference[:, knee_index] += torch.where(
            in_stance, torch.zeros_like(swing_knee), swing_knee
        )
    return reference


def _track_reference(
    env,
    reference_fn,
    std: float,
    period: float,
    duty_factor: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    if std <= 0.0:
        raise ValueError("reference tracking standard deviation must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    actual = asset.data.joint_pos[:, asset_cfg.joint_ids]
    reference = reference_fn(
        env, period=period, duty_factor=duty_factor, asset_cfg=asset_cfg
    )
    error = torch.mean(torch.square(actual - reference), dim=1)
    return torch.exp(-error / std**2) * upright_gate(env, asset_cfg=asset_cfg)


def _track_bilateral_reference(
    env,
    reference_fn,
    std: float,
    period: float,
    duty_factor: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Track both legs; a well-tracked support leg cannot mask a poor swing.

    A joint-average lets a policy earn a respectable reference score by
    following only one leg.  Taking the lower per-leg score remains dense and
    differentiable almost everywhere, but rewards a trajectory only as well as
    its worse leg follows it.
    """
    if std <= 0.0:
        raise ValueError("reference tracking standard deviation must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    actual = asset.data.joint_pos[:, asset_cfg.joint_ids]
    reference = reference_fn(
        env, period=period, duty_factor=duty_factor, asset_cfg=asset_cfg
    )
    if actual.shape[1] != 6:
        raise RuntimeError("LeggyBot bilateral reference requires six joints")
    per_leg_error = torch.stack(
        (
            torch.mean(torch.square(actual[:, :3] - reference[:, :3]), dim=1),
            torch.mean(torch.square(actual[:, 3:] - reference[:, 3:]), dim=1),
        ),
        dim=1,
    )
    per_leg_score = torch.exp(-per_leg_error / std**2)
    return torch.amin(per_leg_score, dim=1) * upright_gate(env, asset_cfg=asset_cfg)


def bilateral_gait_tracking_symmetry(
    env,
    reference_fn,
    std: float,
    period: float,
    duty_factor: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Small regularizer for matching right/left tracking quality by phase."""
    if std <= 0.0:
        raise ValueError("symmetry standard deviation must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    actual = asset.data.joint_pos[:, asset_cfg.joint_ids]
    reference = reference_fn(
        env, period=period, duty_factor=duty_factor, asset_cfg=asset_cfg
    )
    if actual.shape[1] != 6:
        raise RuntimeError("LeggyBot phase symmetry requires six joints")
    errors = torch.stack(
        (
            torch.mean(torch.square(actual[:, :3] - reference[:, :3]), dim=1),
            torch.mean(torch.square(actual[:, 3:] - reference[:, 3:]), dim=1),
        ),
        dim=1,
    )
    return torch.exp(-torch.square((errors[:, 0] - errors[:, 1]) / std**2))


def track_animated_walk_reference(
    env,
    std: float = 0.10,
    period: float = 0.33,
    duty_factor: float = 0.70,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward tracking the explicit low-clearance animated gait."""
    return _track_bilateral_reference(
        env, animated_walk_reference_joint_positions, std, period, duty_factor,
        asset_cfg,
    )


def track_conservative_shuffle_reference(
    env,
    std: float = 0.07,
    period: float = 0.33,
    duty_factor: float = 0.82,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward the small-amplitude, contact-biased shuffling animation."""
    return _track_bilateral_reference(
        env, conservative_shuffle_reference_joint_positions, std, period,
        duty_factor, asset_cfg,
    )


def animated_walk_phase_symmetry(
    env,
    std: float = 0.08,
    period: float = 0.33,
    duty_factor: float = 0.70,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Keep the animated walk's right and left phase-tracking equally good."""
    return bilateral_gait_tracking_symmetry(
        env, animated_walk_reference_joint_positions, std, period, duty_factor,
        asset_cfg,
    )


def conservative_walk_phase_symmetry(
    env,
    std: float = 0.06,
    period: float = 0.33,
    duty_factor: float = 0.82,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Keep the conservative walk's two phase trajectories balanced."""
    return bilateral_gait_tracking_symmetry(
        env, conservative_shuffle_reference_joint_positions, std, period,
        duty_factor, asset_cfg,
    )


def track_unloading_walk_reference(
    env,
    std: float = 0.09,
    period: float = 0.33,
    duty_factor: float = 0.67,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track the low-clearance reference that activates both pitch joints."""
    return _track_bilateral_reference(
        env, unloading_walk_reference_joint_positions, std, period,
        duty_factor, asset_cfg,
    )


def unloading_walk_phase_symmetry(
    env,
    std: float = 0.09,
    period: float = 0.33,
    duty_factor: float = 0.67,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Keep unloading reference tracking balanced across the two legs."""
    return bilateral_gait_tracking_symmetry(
        env, unloading_walk_reference_joint_positions, std, period,
        duty_factor, asset_cfg,
    )


def pitch_upright(
    env,
    std: float = 0.10,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Keep torso pitch near upright while allowing normal yaw and some roll."""
    if std <= 0.0:
        raise ValueError("pitch standard deviation must be positive")
    gravity = env.scene[asset_cfg.name].data.projected_gravity_b
    pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
    return torch.exp(-torch.square(pitch / std))


def pitch_dominant_angular_velocity(
    env,
    command_name: str,
    pitch_weight: float = 3.0,
    roll_weight: float = 0.4,
    yaw_weight: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize pitch rate most; yaw is only softly held near its command."""
    if min(pitch_weight, roll_weight, yaw_weight) < 0.0:
        raise ValueError("angular-velocity weights must be non-negative")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    angular = asset.data.root_link_ang_vel_b
    return (
        roll_weight * torch.square(angular[:, 0])
        + pitch_weight * torch.square(angular[:, 1])
        + yaw_weight * torch.square(angular[:, 2] - command[:, 2])
    )


def forward_speed_success(
    env,
    command_name: str,
    minimum_ratio: float = 0.7,
    maximum_ratio: float = 1.3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Binary curriculum metric for useful forward speed."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    ratio = asset.data.root_link_lin_vel_b[:, 0] / torch.clamp(
        command[:, 0], min=1.0e-4
    )
    return ((ratio >= minimum_ratio) & (ratio <= maximum_ratio)).float()


def sagittal_velocity_success(
    env,
    command_name: str,
    minimum_ratio: float = 0.7,
    maximum_ratio: float = 1.3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Binary velocity-tracking metric valid for positive and negative commands."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    desired = command[:, 0]
    actual = asset.data.root_link_lin_vel_b[:, 0]
    active = torch.abs(desired) >= 0.05
    ratio = actual / torch.where(torch.abs(desired) >= 0.05, desired, torch.ones_like(desired))
    return (active & (ratio >= minimum_ratio) & (ratio <= maximum_ratio)).float()


def sim2real_randomization_scale(env) -> float:
    """Return the task's explicit simulation-randomization strength.

    Common tasks do not inherit the deleted all-in-one gait curriculum. They
    use full reset and observation randomization by default; a task may opt
    into a reduced strength by setting ``cfg.domain_randomization_scale``.
    """
    if not getattr(env.cfg, "sim2real_randomization_enabled", True):
        return 0.0
    return float(getattr(env.cfg, "domain_randomization_scale", 1.0))


@requires_model_fields(
    "geom_friction",
    "body_mass",
    "body_ipos",
    "body_inertia",
    "body_iquat",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
    "jnt_actfrcrange",
    "tendon_actfrcrange",
    recompute=RecomputeLevel.set_const,
)
def sim2real_domain_randomization(env, env_ids: torch.Tensor) -> None:
    """Apply a task-configured reset randomization envelope."""
    scale = sim2real_randomization_scale(env)
    robot = SceneEntityCfg("robot")
    feet = SceneEntityCfg(
        "robot", geom_names=("right_foot_collision", "left_foot_collision")
    )
    actuators = SceneEntityCfg("robot", actuator_names=(".*",))
    for asset_cfg in (robot, feet, actuators):
        asset_cfg.resolve(env.scene)

    env_mdp.dr.geom_friction(
        env,
        env_ids,
        ranges=(0.9 - 0.4 * scale, 0.9 + 0.3 * scale),
        operation="abs",
        shared_random=True,
        asset_cfg=feet,
    )
    env_mdp.dr.pseudo_inertia(
        env,
        env_ids,
        alpha_range=(-0.04 * scale, 0.04 * scale),
        d_range=(-0.02 * scale, 0.02 * scale),
        t_range=(-0.005 * scale, 0.005 * scale),
        asset_cfg=robot,
    )
    env_mdp.dr.pd_gains(
        env,
        env_ids,
        kp_range=(1.0 - 0.10 * scale, 1.0 + 0.10 * scale),
        kd_range=(1.0 - 0.10 * scale, 1.0 + 0.10 * scale),
        operation="scale",
        asset_cfg=actuators,
    )
    env_mdp.dr.effort_limits(
        env,
        env_ids,
        effort_limit_range=(1.0 - 0.10 * scale, 1.0 + 0.05 * scale),
        operation="scale",
        asset_cfg=actuators,
    )
    env_mdp.dr.encoder_bias(
        env,
        env_ids,
        bias_range=(-0.01 * scale, 0.01 * scale),
        asset_cfg=robot,
    )
    max_lag = 2 if scale >= 0.5 else (1 if scale > 0.0 else 0)
    for actuator in env.scene["robot"].actuators:
        lags = torch.randint(
            0, max_lag + 1, (len(env_ids),), device=env.device
        )
        actuator.set_lags(lags, env_ids)

def _scaled_noise(env, value: torch.Tensor, maximum: float) -> torch.Tensor:
    scale = sim2real_randomization_scale(env)
    if scale == 0.0:
        return value
    return value + (2.0 * torch.rand_like(value) - 1.0) * maximum * scale


def sim2real_base_ang_vel(env, sensor_name: str, maximum_noise: float = 0.2):
    return _scaled_noise(
        env, env_mdp.builtin_sensor(env, sensor_name), maximum_noise
    )


def sim2real_projected_gravity(env, maximum_noise: float = 0.05):
    return _scaled_noise(env, env_mdp.projected_gravity(env), maximum_noise)


def sim2real_joint_pos(env, maximum_noise: float = 0.01):
    return _scaled_noise(
        env, env_mdp.joint_pos_rel(env, biased=True), maximum_noise
    )


def sim2real_joint_vel(env, maximum_noise: float = 1.5):
    return _scaled_noise(env, env_mdp.joint_vel_rel(env), maximum_noise)
