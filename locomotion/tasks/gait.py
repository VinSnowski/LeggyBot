"""Reusable rewards, metrics, noise, and gait-reference terms for LeggyBot."""

from __future__ import annotations

import math
from dataclasses import dataclass
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


def full_domain_randomization(
    env,
    env_ids: torch.Tensor,
    randomize_latency: bool = True,
) -> None:
    """Apply the task's configured reset randomization envelope."""
    sim2real_domain_randomization(env, env_ids, randomize_latency=randomize_latency)


_SYMMETRIC_JOINT_PAIRS = (
    ("RightBaseToHip", "LeftBaseToHip", -1.0),
    ("RightHipToThigh", "LeftHipToThigh", 1.0),
    ("RightKnee", "LeftKnee", 1.0),
)

_STS3215_JOINT_NAMES = tuple(pair[i] for i in (0, 1) for pair in _SYMMETRIC_JOINT_PAIRS)


@dataclass(frozen=True)
class Sts3215RandomizationPrior:
    """Pre-measurement STS3215 priors, centralized for later system ID.

    The shared terms represent supply voltage and a common firmware/controller
    configuration.  Unit terms represent manufacturing and assembly mismatch.
    Ranges combine the STS3215 datasheet's roughly 10% torque/speed tolerance
    with BIMO's measurements of 2.7--2.94 Nm supply-dependent torque, 5 ms bus
    delay, and 1.0--2.4 degree assembled-joint backlash.
    """

    supply_factor: tuple[float, float] = (2.70 / 2.942, 1.0)
    unit_torque_factor: tuple[float, float] = (0.95, 1.05)
    unit_speed_factor: tuple[float, float] = (0.90, 1.10)
    common_stiffness_factor: tuple[float, float] = (0.90, 1.10)
    unit_stiffness_factor: tuple[float, float] = (0.95, 1.05)
    friction_factor: tuple[float, float] = (0.75, 1.25)
    armature_factor: tuple[float, float] = (0.80, 1.25)
    encoder_bias_rad: float = math.radians(1.0)
    backlash_total_rad: tuple[float, float] = (
        math.radians(0.5),
        math.radians(2.4),
    )
    latency_lags: tuple[int, int] = (1, 3)
    encoder_noise_rad: float = math.radians(0.20)


STS3215_RANDOMIZATION = Sts3215RandomizationPrior()


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


def _paired_or_independent_samples(
    env,
    batch: int,
    count: int,
    value_range: tuple[float, float],
    paired: bool,
) -> torch.Tensor:
    sample_count = count // 2 if paired else count
    values = torch.empty((batch, sample_count), device=env.device).uniform_(
        *value_range
    )
    return torch.cat((values, values), dim=1) if paired else values


def _randomize_sts3215_servos(
    env, env_ids: torch.Tensor, *, paired: bool, randomize_latency: bool = True
) -> None:
    """Sample a correlated STS3215 realization for selected environments."""
    prior = STS3215_RANDOMIZATION
    scale = sim2real_randomization_scale(env)
    n = len(_STS3215_JOINT_NAMES)
    selected = torch.arange(env.num_envs, device=env.device)[env_ids]
    batch = len(selected)
    model = env.sim.model
    defaults = env.sim.get_default_field

    ctrl_ids = []
    actuators = []
    for actuator_name in (
        "right_hip_roll", "right_thigh", "right_knee",
        "left_hip_roll", "left_thigh", "left_knee",
    ):
        _, actuator = _single_actuator(env, actuator_name)
        ctrl_ids.append(int(torch.as_tensor(actuator.global_ctrl_ids).reshape(-1)[0]))
        actuators.append(actuator)
    ctrl_ids = torch.tensor(ctrl_ids, device=env.device, dtype=torch.long)

    local_joint_ids, _ = env.scene["robot"].find_joints(
        _STS3215_JOINT_NAMES, preserve_order=True
    )
    local_joint_ids = torch.tensor(
        local_joint_ids, device=env.device, dtype=torch.long
    )
    dof_ids = env.scene["robot"].indexing.joint_v_adr[local_joint_ids]

    def samples(value_range: tuple[float, float]) -> torch.Tensor:
        raw = _paired_or_independent_samples(env, batch, n, value_range, paired)
        return 1.0 + scale * (raw - 1.0)

    supply_raw = torch.empty((batch, 1), device=env.device).uniform_(
        *prior.supply_factor
    )
    supply = 1.0 + scale * (supply_raw - 1.0)
    unit_torque = samples(prior.unit_torque_factor)
    strength = supply * unit_torque
    no_load_speed = supply * samples(prior.unit_speed_factor)
    stiffness = (
        1.0
        + scale
        * (
            torch.empty((batch, 1), device=env.device).uniform_(
                *prior.common_stiffness_factor
            )
            - 1.0
        )
    ) * samples(prior.unit_stiffness_factor)

    model.actuator_gainprm[selected[:, None], ctrl_ids] = defaults(
        "actuator_gainprm"
    )[ctrl_ids]
    model.actuator_biasprm[selected[:, None], ctrl_ids] = defaults(
        "actuator_biasprm"
    )[ctrl_ids]
    model.actuator_forcerange[selected[:, None], ctrl_ids] = defaults(
        "actuator_forcerange"
    )[ctrl_ids] * strength[..., None]
    model.actuator_gainprm[selected[:, None], ctrl_ids, 0] *= stiffness
    model.actuator_biasprm[selected[:, None], ctrl_ids, 1] *= stiffness

    # The passive damping is the slope of the servo's approximate linear
    # torque-speed curve: tau = tau_stall - damping * speed.  Correlating it
    # with randomized stall torque and no-load speed is more physical than an
    # independent damping draw.
    model.dof_damping[selected[:, None], dof_ids] = defaults("dof_damping")[
        dof_ids
    ] * strength / no_load_speed
    model.dof_frictionloss[selected[:, None], dof_ids] = defaults(
        "dof_frictionloss"
    )[dof_ids] * samples(prior.friction_factor)
    model.dof_armature[selected[:, None], dof_ids] = defaults("dof_armature")[
        dof_ids
    ] * samples(prior.armature_factor)

    bias_half_range = prior.encoder_bias_rad * scale
    right_bias = torch.empty((batch, 3), device=env.device).uniform_(
        -bias_half_range, bias_half_range
    )
    if paired:
        bias = torch.cat(
            (right_bias, right_bias * torch.tensor((-1.0, 1.0, 1.0), device=env.device)),
            dim=1,
        )
    else:
        bias = torch.empty((batch, n), device=env.device).uniform_(
            -bias_half_range, bias_half_range
        )
    env.scene["robot"].data.encoder_bias[selected[:, None], local_joint_ids] = bias

    backlash_total = _paired_or_independent_samples(
        env, batch, n, prior.backlash_total_rad, paired
    ) * scale
    action = env.action_manager.get_term("joint_pos")
    action_width = torch.zeros((batch, action.action_dim), device=env.device)
    action_name_to_id = {name: i for i, name in enumerate(action.target_names)}
    for source, name in enumerate(_STS3215_JOINT_NAMES):
        action_width[:, action_name_to_id[name]] = 0.5 * backlash_total[:, source]
    action.set_backlash_half_width(action_width, selected)

    # The bus delay is common to all servos in one robot. XmlActuator instances
    # own separate buffers, so assign the same per-environment lag to each and
    # keep it fixed for the episode (delay_hold_prob=1 in the robot config).
    if randomize_latency:
        if scale <= 0.0:
            lags = torch.zeros(batch, device=env.device, dtype=torch.long)
        else:
            lag_min = max(0, round(prior.latency_lags[0] * scale))
            lag_max = max(lag_min, round(prior.latency_lags[1] * scale))
            lags = torch.randint(
                lag_min, lag_max + 1, (batch,), device=env.device
            )
        for actuator in actuators:
            actuator.set_lags(lags, selected)


def symmetric_domain_randomization(env, env_ids: torch.Tensor) -> None:
    """Pair servo variation without introducing independent leg-inertia mismatch.

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
    # reflection. Torso perturbations retain the nominal lateral COM and
    # central-inertia axes; small existing CAD asymmetries are not removed.
    env_mdp.dr.geom_friction(
        env,
        env_ids,
        ranges=(0.9 - 0.4 * scale, 0.9 + 0.3 * scale),
        operation="abs",
        shared_random=True,
        asset_cfg=feet,
    )
    # Reset every robot inertia before constructing the paired endpoint. A
    # previous full/randomized episode must not leak leg mismatch into it.
    body_ids = env.scene["robot"].indexing.body_ids
    for field in ("body_mass", "body_ipos", "body_inertia", "body_iquat"):
        getattr(env.sim.model, field)[env_ids[:, None], body_ids] = (
            env.sim.get_default_field(field)[body_ids]
        )
    torso = SceneEntityCfg("robot", body_names=("BaseLink",), preserve_order=True)
    torso.resolve(env.scene)
    torso_ids = body_ids[torso.body_ids]
    # Perturb central inertia, not the CAD-origin pseudo-inertia. Scaling
    # around that offset origin inadvertently displaced COM laterally.
    shape = (len(env_ids), len(torso_ids))
    mass_scale = torch.exp(
        torch.empty(shape, device=env.device).uniform_(-.08 * scale, .08 * scale)
    )
    length_scale = torch.exp(
        torch.empty(shape, device=env.device).uniform_(-.02 * scale, .02 * scale)
    )
    model = env.sim.model
    model.body_mass[env_ids[:, None], torso_ids] *= mass_scale
    model.body_inertia[env_ids[:, None], torso_ids] *= (mass_scale * length_scale.square())[..., None]
    offset = torch.empty((*shape, 3), device=env.device).uniform_(-.005 * scale, .005 * scale)
    offset[..., 1] = 0.0
    model.body_ipos[env_ids[:, None], torso_ids] += offset

    _randomize_sts3215_servos(env, env_ids, paired=True)


def _interpolated_full_domain_randomization(
    env, env_ids: torch.Tensor, amplitude: float
) -> None:
    """Blend paired and full reset realizations at a continuous amplitude."""
    if not 0.0 < amplitude < 1.0:
        raise ValueError("interpolation amplitude must be in (0, 1)")
    model = env.sim.model
    # Only the current XML position-actuator fields are changed by full DR.
    # In particular, do not wrap an empty tendon vec2 array on CPU.
    field_names = (
        "geom_friction", "actuator_gainprm", "actuator_biasprm",
        "actuator_forcerange", "dof_damping", "dof_frictionloss", "dof_armature",
    )
    body_fields = ("body_mass", "body_ipos", "body_inertia", "body_iquat")
    body_ids = env.scene["robot"].indexing.body_ids
    symmetric_domain_randomization(env, env_ids)
    paired_fields = {name: getattr(model, name)[env_ids].clone() for name in field_names}
    paired_body = [getattr(model, name)[env_ids[:, None], body_ids].clone() for name in body_fields]
    encoder_bias = env.scene["robot"].data.encoder_bias
    paired_encoder_bias = encoder_bias[env_ids].clone()
    action = env.action_manager.get_term("joint_pos")
    paired_backlash = action._backlash_half_width[env_ids].clone()
    full_domain_randomization(env, env_ids, randomize_latency=False)
    for name in field_names:
        values = getattr(model, name)
        full_values = values[env_ids]
        blended = paired_fields[name] + amplitude * (
            full_values - paired_fields[name]
        )
        values[env_ids] = blended
    # Interpolate physical mass distributions, independent of eigenvector
    # ordering or quaternion sign. Float64 avoids cancellation at CAD origins.
    from mjlab.envs.mdp.dr.body import _reconstruct_pseudo_inertia_J, _decompose_pseudo_inertia_J
    full_body = [getattr(model, name)[env_ids[:, None], body_ids].clone() for name in body_fields]
    paired_j = _reconstruct_pseudo_inertia_J(*(value.double() for value in paired_body))
    full_j = _reconstruct_pseudo_inertia_J(*(value.double() for value in full_body))
    blended_body = _decompose_pseudo_inertia_J(paired_j + amplitude * (full_j - paired_j))
    for name, value, original in zip(body_fields, blended_body, paired_body, strict=True):
        getattr(model, name)[env_ids[:, None], body_ids] = value.to(original.dtype)
    full_encoder_bias = encoder_bias[env_ids]
    encoder_bias[env_ids] = paired_encoder_bias + amplitude * (
        full_encoder_bias - paired_encoder_bias
    )
    full_backlash = action._backlash_half_width[env_ids]
    action._backlash_half_width[env_ids] = paired_backlash + amplitude * (
        full_backlash - paired_backlash
    )


@requires_model_fields(
    "geom_friction",
    "body_mass",
    "body_ipos",
    "body_inertia",
    "body_iquat",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
    "dof_damping",
    "dof_frictionloss",
    "dof_armature",
    "jnt_actfrcrange",
    "tendon_actfrcrange",
    recompute=RecomputeLevel.set_const,
)
def updated_default_domain_randomization(
    env,
    env_ids: torch.Tensor,
    symmetric_steps: int = 12_000,
    independent_ramp_steps: int = 0,
) -> None:
    """Pretrain under paired variation, then ramp to full randomization.

    Existing callers retain the historical hard switch when
    ``independent_ramp_steps`` is zero.  Tasks that need a stable symmetric
    gait first can request a gradual full-domain transition instead. During the
    ramp, continuous simulator fields are interpolated between paired and full
    realizations; discrete latency uses an increasing independent fraction.
    """
    if symmetric_steps < 0:
        raise ValueError("symmetric randomization steps must be non-negative")
    if independent_ramp_steps < 0:
        raise ValueError("independent mismatch ramp must be non-negative")
    step = float(getattr(env, "common_step_counter", 0))
    if step < symmetric_steps:
        symmetric_domain_randomization(env, env_ids)
        return
    if independent_ramp_steps == 0:
        full_domain_randomization(env, env_ids)
        return

    mismatch_scale = min(
        1.0, max(0.0, (step - symmetric_steps) / float(independent_ramp_steps))
    )
    if mismatch_scale <= 0.0:
        symmetric_domain_randomization(env, env_ids)
    elif mismatch_scale >= 1.0:
        full_domain_randomization(env, env_ids)
    else:
        _interpolated_full_domain_randomization(env, env_ids, mismatch_scale)


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
    include_vertical: bool = False,
    asset_cfg: SceneEntityCfg = _FOOT_SITES_CFG,
) -> torch.Tensor:
    """Cost foot motion during stance, optionally including vertical chatter."""
    if speed_scale <= 0.0:
        raise ValueError("speed scale must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    dimensions = slice(None) if include_vertical else slice(0, 2)
    foot_speed = torch.linalg.vector_norm(
        asset.data.site_lin_vel_w[:, asset_cfg.site_ids, dimensions], dim=-1
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


def periodic_reward_composition_swing_force_cost(
    env,
    sensor_name: str,
    period: float = 0.33,
    duty_factor: float = 0.70,
    transition_fraction: float = 0.04,
    force_scale: float = 10.0,
) -> torch.Tensor:
    """Paper-style sum of bounded swing-foot force costs.

    Siekmann et al. compose one cost per foot rather than normalizing by the
    number of active phases.  ``smooth_swing_probability`` is the project's
    deterministic expected-interval approximation because the paper does not
    publish the Von Mises concentration used for its walking policy.
    """
    if force_scale <= 0.0:
        raise ValueError("force scale must be positive")
    swing = smooth_swing_probability(env, period, duty_factor, transition_fraction)
    force_cost = 1.0 - torch.exp(
        -torch.square(_foot_force_magnitudes(env, sensor_name) / force_scale)
    )
    return torch.sum(swing * force_cost, dim=1)


def periodic_reward_composition_stance_speed_cost(
    env,
    period: float = 0.33,
    duty_factor: float = 0.70,
    transition_fraction: float = 0.04,
    speed_coefficient: float = 2.0,
    asset_cfg: SceneEntityCfg = _FOOT_SITES_CFG,
) -> torch.Tensor:
    """Paper-style sum of bounded 3-D stance-foot speed costs."""
    if speed_coefficient <= 0.0:
        raise ValueError("speed coefficient must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    foot_speed_squared = torch.sum(
        torch.square(asset.data.site_lin_vel_w[:, asset_cfg.site_ids]), dim=-1
    )
    if foot_speed_squared.shape[1] != 2:
        raise RuntimeError("periodic walking requires exactly right and left foot sites")
    stance = 1.0 - smooth_swing_probability(
        env, period, duty_factor, transition_fraction
    )
    speed_cost = 1.0 - torch.exp(-speed_coefficient * foot_speed_squared)
    return torch.sum(stance * speed_cost, dim=1)


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


def commanded_periodic_foot_exchange(
    env,
    command_name: str,
    period: float = 0.33,
    reference_speed: float = 0.08,
    reference_peak_to_peak: float = 0.024,
    std: float = 0.018,
    asset_cfg: SceneEntityCfg = _FOOT_SITES_CFG,
) -> torch.Tensor:
    """Scale signed fore/aft foot exchange linearly with commanded forward speed."""
    if min(period, reference_speed, reference_peak_to_peak, std) <= 0.0:
        raise ValueError("commanded foot-exchange parameters must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command {command_name!r} not found"
    foot_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]
    if foot_xy.shape[1] != 2:
        raise RuntimeError("periodic walking requires exactly right and left foot sites")
    phase = gait_phase_scalar(env, period)
    amplitude = 0.5 * reference_peak_to_peak * command[:, 0] / reference_speed
    desired_difference = amplitude * torch.sin(2.0 * math.pi * phase)
    # The exchange is fore/aft relative to the robot, not fixed world X.  A
    # turning robot's sagittal foot separation therefore needs projection onto
    # its current body-forward axis before comparing it with the command.
    heading = asset.data.heading_w
    forward = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
    actual_difference = torch.sum(
        (foot_xy[:, 0] - foot_xy[:, 1]) * forward, dim=1
    )
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


def world_forward_velocity_only(
    env,
    command_name: str,
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track world-forward speed without suppressing normal lateral transfer."""
    if std <= 0.0:
        raise ValueError("world velocity standard deviation must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command {command_name!r} not found"
    error = asset.data.root_link_lin_vel_w[:, 0] - command[:, 0]
    return torch.exp(-torch.square(error / std))


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
        self._last_heading = torch.zeros(self.num_envs, device=self.device)
        self._unwrapped_heading = torch.zeros(self.num_envs, device=self.device)

    @staticmethod
    def _wrap(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._last_episode_step[env_ids] = -1
        self._completed_lateral_speed[env_ids] = 0.0
        self._completed_yaw_rate[env_ids] = 0.0
        self._last_heading[env_ids] = 0.0
        self._unwrapped_heading[env_ids] = 0.0

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
            self._last_heading[new_episode] = current_heading[new_episode]
            self._unwrapped_heading[new_episode] = current_heading[new_episode]
            self._cycle_index[new_episode] = cycle_index[new_episode]
            self._cycle_start_y[new_episode] = current_y[new_episode]
            self._cycle_start_heading[new_episode] = self._unwrapped_heading[new_episode]
            self._completed_lateral_speed[new_episode] = 0.0
            self._completed_yaw_rate[new_episode] = 0.0

        continuing = ~new_episode
        if torch.any(continuing):
            self._unwrapped_heading[continuing] += self._wrap(
                current_heading[continuing] - self._last_heading[continuing]
            )
        self._last_heading.copy_(current_heading)

        completed_cycle = (cycle_index != self._cycle_index) & ~new_episode
        if torch.any(completed_cycle):
            self._completed_lateral_speed[completed_cycle] = (
                current_y[completed_cycle] - self._cycle_start_y[completed_cycle]
            ) / period
            self._completed_yaw_rate[completed_cycle] = (
                self._unwrapped_heading[completed_cycle]
                - self._cycle_start_heading[completed_cycle]
            ) / period
            self._cycle_index[completed_cycle] = cycle_index[completed_cycle]
            self._cycle_start_y[completed_cycle] = current_y[completed_cycle]
            self._cycle_start_heading[completed_cycle] = self._unwrapped_heading[completed_cycle]

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
        heading = self._unwrapped_heading
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


def _curriculum_progress(env, ramp_steps: int) -> float:
    if ramp_steps < 0:
        raise ValueError("curriculum ramp steps must be non-negative")
    if ramp_steps == 0:
        return 1.0
    return min(
        1.0,
        float(getattr(env, "common_step_counter", 0)) / float(ramp_steps),
    )


def _curriculum_value(
    env,
    initial: float,
    final: float,
    ramp_steps: int,
) -> float:
    return initial + (final - initial) * _curriculum_progress(env, ramp_steps)
    return initial + (final - initial) * progress


def velocity_command_range_curriculum(
    env,
    env_ids: torch.Tensor | slice | None,
    initial_range: tuple[float, float] = (0.075, 0.085),
    final_range: tuple[float, float] = (0.0, 0.1),
    ramp_steps: int = 12_000,
) -> dict[str, float]:
    """Expand the forward command range over training."""
    del env_ids
    if (
        initial_range[0] < 0.0
        or initial_range[0] > initial_range[1]
        or final_range[0] < 0.0
        or final_range[0] > final_range[1]
    ):
        raise ValueError("velocity command ranges must be ordered and non-negative")
    progress = _curriculum_progress(env, ramp_steps)
    lower = initial_range[0] + (final_range[0] - initial_range[0]) * progress
    upper = initial_range[1] + (final_range[1] - initial_range[1]) * progress
    env.command_manager.get_term("twist").cfg.ranges.lin_vel_x = (lower, upper)
    return {"min": lower, "max": upper}


def lateral_velocity_command_range_curriculum(
    env,
    env_ids: torch.Tensor | slice | None,
    initial_range: tuple[float, float] = (0.0, 0.0),
    final_range: tuple[float, float] = (-0.1, 0.1),
    ramp_steps: int = 24_000,
) -> dict[str, float]:
    """Expand the lateral command range while retaining the final forward range."""
    del env_ids
    if initial_range[0] > initial_range[1] or final_range[0] > final_range[1]:
        raise ValueError("lateral velocity command ranges must be ordered")
    progress = _curriculum_progress(env, ramp_steps)
    lower = initial_range[0] + (final_range[0] - initial_range[0]) * progress
    upper = initial_range[1] + (final_range[1] - initial_range[1]) * progress
    env.command_manager.get_term("twist").cfg.ranges.lin_vel_y = (lower, upper)
    return {"min": lower, "max": upper}


def quasi_omnidirectional_command_curriculum(
    env,
    env_ids: torch.Tensor | slice | None,
    final_linear_speed: float = 0.1,
    final_yaw_rate: float = math.pi / 4.0,
    ramp_steps: int = 36_000,
) -> dict[str, float]:
    """Expand symmetric planar and conditional-yaw command ranges from zero."""
    del env_ids
    if final_linear_speed <= 0.0 or final_yaw_rate <= 0.0:
        raise ValueError("final command limits must be positive")
    progress = _curriculum_progress(env, ramp_steps)
    linear_limit = final_linear_speed * progress
    yaw_limit = final_yaw_rate * progress
    ranges = env.command_manager.get_term("twist").cfg.ranges
    ranges.lin_vel_x = (-linear_limit, linear_limit)
    ranges.lin_vel_y = (-linear_limit, linear_limit)
    ranges.ang_vel_z = (-yaw_limit, yaw_limit)
    return {
        "vx_min": -linear_limit,
        "vx_max": linear_limit,
        "vy_min": -linear_limit,
        "vy_max": linear_limit,
        "yaw_min": -yaw_limit,
        "yaw_max": yaw_limit,
    }


def quasi_omnidirectional_command_success(
    env,
    command_name: str,
    linear_error: float = 0.05,
    yaw_deadband: float = 0.02,
    yaw_relative_scale: float = 0.5,
    yaw_scale_floor: float = 0.05,
    coupled_vx_threshold: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Per-step success used by the staged omni curriculum.

    Linear tracking is always checked for an active planar command.  Yaw is
    checked only for a coupled turn (non-zero commanded forward speed and yaw),
    using a deadbanded relative tolerance.  Thus straight and pure-lateral
    samples do not get penalized for incidental yaw, while coupled turns cannot
    be solved by following only ``vx``.
    """
    if min(linear_error, yaw_relative_scale, yaw_scale_floor, coupled_vx_threshold) <= 0.0:
        raise ValueError("omnidirectional success tolerances must be positive")
    if yaw_deadband < 0.0:
        raise ValueError("yaw deadband must be non-negative")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command {command_name!r} not found"
    heading = asset.data.heading_w
    cosine = torch.cos(heading)
    sine = torch.sin(heading)
    target_world = torch.stack(
        (
            cosine * command[:, 0] - sine * command[:, 1],
            sine * command[:, 0] + cosine * command[:, 1],
        ),
        dim=1,
    )
    actual = asset.data.root_link_lin_vel_w[:, :2]
    linear_ok = torch.linalg.vector_norm(actual - target_world, dim=1) <= linear_error
    yaw_error = torch.abs(asset.data.root_link_ang_vel_b[:, 2] - command[:, 2])
    yaw_active = (torch.abs(command[:, 0]) >= coupled_vx_threshold) & (
        torch.abs(command[:, 2]) >= yaw_deadband
    )
    yaw_tolerance = torch.maximum(
        command[:, 2].abs() * yaw_relative_scale,
        command[:, 2].new_tensor(yaw_scale_floor),
    )
    yaw_ok = (~yaw_active) | (yaw_error <= yaw_tolerance)
    command_active = torch.linalg.vector_norm(command[:, :2], dim=1) >= 0.01
    upright = -asset.data.projected_gravity_b[:, 2] >= 0.9
    return ((~command_active) | (linear_ok & yaw_ok & upright)).float()


class StagedQuasiOmnidirectionalCurriculum:
    """Stateful, hysteretic command curriculum for V2.

    The curriculum is evaluated when episodes reset, where the metrics manager
    still contains the just-completed episode.  It aggregates fixed-size
    evaluation windows, advances after three passing windows, and backs off the
    newest difficulty after two failing windows.  The forward anchor range is
    never reduced.
    """

    _CURVE_LEVELS = (0.05, 0.15, 0.30)
    _LATERAL_LEVELS = (0.02, 0.05, 0.10)
    _OMNI_LEVELS = (0.25, 0.50, 0.75, 1.00)
    _CYCLE_THRESHOLDS = (0.75, 0.85, 1.00, 1.20, 1.35)
    _EPISODE_THRESHOLDS = (1.50, 1.75, 2.00, 2.50, 2.75)

    def __init__(
        self,
        window_episodes: int = 256,
        advance_windows: int = 3,
        regress_windows: int = 2,
        final_linear_speed: float = 0.10,
        final_yaw_rate: float = math.pi / 4.0,
    ) -> None:
        if min(window_episodes, advance_windows, regress_windows) <= 0:
            raise ValueError("curriculum window parameters must be positive")
        if final_linear_speed <= 0.0 or final_yaw_rate <= 0.0:
            raise ValueError("final command limits must be positive")
        self.window_episodes = int(window_episodes)
        self.advance_windows = int(advance_windows)
        self.regress_windows = int(regress_windows)
        self.final_linear_speed = float(final_linear_speed)
        self.final_yaw_rate = float(final_yaw_rate)
        self.stage = 0
        self.level = 0
        self.good_windows = 0
        self.bad_windows = 0
        self._window_count = 0
        self._window_sums: dict[str, float] = {}
        self._last_gate: bool | None = None
        self.restore_checkpoint_requested = False
        self._difficulty_scale = 1.0

    @staticmethod
    def _ids(env, env_ids):
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(env.num_envs, device=env.device)
        return env_ids

    @staticmethod
    def _episode_mean(env, name: str, ids: torch.Tensor) -> torch.Tensor:
        manager = getattr(env, "metrics_manager", None)
        sums = getattr(manager, "_episode_sums", {}).get(name)
        counts = getattr(manager, "_step_count", None)
        if sums is None or counts is None:
            return torch.zeros(len(ids), device=env.device)
        return sums[ids] / torch.clamp(counts[ids].float(), min=1.0)

    def _collect_window(self, env, ids: torch.Tensor) -> None:
        counts = getattr(getattr(env, "metrics_manager", None), "_step_count", None)
        if counts is None:
            return
        valid = counts[ids] > 0
        if not torch.any(valid):
            return
        ids = ids[valid]
        values = {
            "success": self._episode_mean(env, "command_tracking_success", ids),
            "cycle": self._episode_mean(env, "cycle_command_displacement_cost", ids),
            "episode": self._episode_mean(env, "episode_command_displacement_cost", ids),
        }
        termination = getattr(env, "termination_manager", None)
        timeout = termination.get_term("time_out")[ids].float() if termination and "time_out" in termination.active_terms else torch.zeros(len(ids), device=env.device)
        failures = torch.zeros(len(ids), device=env.device)
        if termination is not None:
            for name in ("fell_over", "illegal_contact"):
                if name in termination.active_terms:
                    failures = torch.maximum(failures, termination.get_term(name)[ids].float())
        self._window_count += len(ids)
        for name, value in values.items():
            self._window_sums[name] = self._window_sums.get(name, 0.0) + float(value.sum().item())
        self._window_sums["timeout"] = self._window_sums.get("timeout", 0.0) + float(timeout.sum().item())
        self._window_sums["failure"] = self._window_sums.get("failure", 0.0) + float(failures.sum().item())

    def _window_passes(self) -> bool:
        n = max(self._window_count, 1)
        cycle = self._window_sums.get("cycle", float("inf")) / n
        episode = self._window_sums.get("episode", float("inf")) / n
        success = self._window_sums.get("success", 0.0) / n
        timeout = self._window_sums.get("timeout", 0.0) / n
        failure = self._window_sums.get("failure", 1.0) / n
        return (
            success >= 0.85
            and timeout >= 0.95
            and failure <= 0.05
            and cycle <= self._CYCLE_THRESHOLDS[min(self.stage, 4)]
            and episode <= self._EPISODE_THRESHOLDS[min(self.stage, 4)]
        )

    def _clear_window(self) -> None:
        self._window_count = 0
        self._window_sums.clear()

    def _advance(self) -> None:
        if self.stage == 0:
            self.stage, self.level = 1, 0
        elif self.stage == 1 and self.level + 1 < len(self._CURVE_LEVELS):
            self.level += 1
        elif self.stage == 1:
            self.stage, self.level = 2, 0
        elif self.stage == 2 and self.level + 1 < len(self._LATERAL_LEVELS):
            self.level += 1
        elif self.stage == 2:
            self.stage, self.level = 3, 0
        elif self.stage == 3 and self.level + 1 < len(self._OMNI_LEVELS):
            self.level += 1
        elif self.stage == 3:
            self.stage, self.level = 4, 0
        self.good_windows = 0
        self.bad_windows = 0
        self.restore_checkpoint_requested = False
        self._difficulty_scale = 1.0

    def _regress(self) -> None:
        self.restore_checkpoint_requested = True
        # Hysteretic backoff: reduce only the currently unlocked range by 15%.
        # The stage/level fallback below additionally removes an axis only when
        # the lowest level remains unstable.
        self._difficulty_scale = max(0.50, self._difficulty_scale * 0.85)
        if self.stage == 1:
            if self.level > 0:
                self.level -= 1
            else:
                self.stage, self.level = 0, 0
        elif self.stage == 2:
            if self.level > 0:
                self.level -= 1
            else:
                self.stage, self.level = 1, len(self._CURVE_LEVELS) - 1
        elif self.stage == 3:
            if self.level > 0:
                self.level -= 1
            else:
                self.stage, self.level = 2, len(self._LATERAL_LEVELS) - 1
        elif self.stage == 4:
            self.stage, self.level = 3, len(self._OMNI_LEVELS) - 1
        self.good_windows = 0
        self.bad_windows = 0

    def _apply(self, env) -> dict[str, float]:
        command = env.command_manager.get_term("twist").cfg
        ranges = command.ranges
        # Protected forward anchor.  The sampler uses this range for straight
        # and curve modes even after the global omni range becomes symmetric.
        command.protected_forward_range = (0.04, self.final_linear_speed)
        if self.stage == 0:
            ranges.lin_vel_x = (0.04, self.final_linear_speed)
            ranges.lin_vel_y = (0.0, 0.0)
            ranges.ang_vel_z = (0.0, 0.0)
            probabilities = (0.65, 0.25, 0.0, 0.0, 0.10, 0.0)
        elif self.stage == 1:
            ranges.lin_vel_x = (0.04, self.final_linear_speed)
            ranges.lin_vel_y = (0.0, 0.0)
            yaw = self._CURVE_LEVELS[min(self.level, len(self._CURVE_LEVELS) - 1)] * self._difficulty_scale
            ranges.ang_vel_z = (-yaw, yaw)
            probabilities = (0.65, 0.25, 0.0, 0.0, 0.10, 0.0)
        elif self.stage == 2:
            ranges.lin_vel_x = (0.0, self.final_linear_speed)
            lateral = self._LATERAL_LEVELS[min(self.level, len(self._LATERAL_LEVELS) - 1)] * self._difficulty_scale
            ranges.lin_vel_y = (-lateral, lateral)
            ranges.ang_vel_z = (-self._CURVE_LEVELS[-1], self._CURVE_LEVELS[-1])
            probabilities = (0.60, 0.20, 0.15, 0.0, 0.05, 0.0)
        elif self.stage == 3:
            scale = self._OMNI_LEVELS[min(self.level, len(self._OMNI_LEVELS) - 1)]
            scale *= self._difficulty_scale
            ranges.lin_vel_x = (-self.final_linear_speed * scale, self.final_linear_speed * scale)
            ranges.lin_vel_y = (-self.final_linear_speed * scale, self.final_linear_speed * scale)
            yaw = self.final_yaw_rate * scale
            ranges.ang_vel_z = (-yaw, yaw)
            probabilities = (0.55, 0.15, 0.10, 0.15, 0.05, 0.0)
        else:
            ranges.lin_vel_x = (-self.final_linear_speed, self.final_linear_speed)
            ranges.lin_vel_y = (-self.final_linear_speed, self.final_linear_speed)
            ranges.ang_vel_z = (-self.final_yaw_rate, self.final_yaw_rate)
            probabilities = (0.50, 0.15, 0.10, 0.15, 0.05, 0.05)
        for name, value in zip(
            ("forward_straight_probability", "forward_curve_probability", "pure_lateral_probability", "full_omni_probability", "stop_probability", "reverse_probability"),
            probabilities,
            strict=True,
        ):
            if hasattr(command, name):
                setattr(command, name, value)
        return {
            "stage": float(self.stage),
            "level": float(self.level),
            "good_windows": float(self.good_windows),
            "bad_windows": float(self.bad_windows),
            "restore_checkpoint": float(self.restore_checkpoint_requested),
            "vx_min": float(ranges.lin_vel_x[0]),
            "vx_max": float(ranges.lin_vel_x[1]),
            "vy_min": float(ranges.lin_vel_y[0]),
            "vy_max": float(ranges.lin_vel_y[1]),
            "yaw_min": float(ranges.ang_vel_z[0]),
            "yaw_max": float(ranges.ang_vel_z[1]),
        }

    def __call__(self, env, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
        ids = self._ids(env, env_ids)
        self._collect_window(env, ids)
        while self._window_count >= self.window_episodes:
            # Use the first complete window; any surplus episodes remain for the
            # next evaluation window rather than causing multiple jumps.
            passed = self._window_passes()
            self._last_gate = passed
            if passed:
                self.good_windows += 1
                self.bad_windows = 0
                if self.good_windows >= self.advance_windows:
                    self._advance()
            else:
                self.bad_windows += 1
                self.good_windows = 0
                if self.bad_windows >= self.regress_windows:
                    self._regress()
            self._clear_window()
        return self._apply(env)

    def state_dict(self) -> dict[str, float | int | bool]:
        """Serialize hysteresis state so resumes do not restart at stage zero."""
        return {
            "stage": self.stage,
            "level": self.level,
            "good_windows": self.good_windows,
            "bad_windows": self.bad_windows,
            "difficulty_scale": self._difficulty_scale,
            "restore_checkpoint_requested": self.restore_checkpoint_requested,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.stage = max(0, min(4, int(state.get("stage", 0))))
        self.level = max(0, int(state.get("level", 0)))
        self.good_windows = max(0, int(state.get("good_windows", 0)))
        self.bad_windows = max(0, int(state.get("bad_windows", 0)))
        self._difficulty_scale = float(state.get("difficulty_scale", 1.0))
        self._difficulty_scale = max(0.50, min(1.0, self._difficulty_scale))
        self.restore_checkpoint_requested = bool(
            state.get("restore_checkpoint_requested", False)
        )


class _CommandedCycleMotionTerm(ManagerTermBase):
    """Measure cycle motion against commands integrated over the same cycle."""

    def __init__(self, cfg, env) -> None:
        del cfg
        super().__init__(env)
        self._last_episode_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._cycle_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._cycle_start_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._command_displacement = torch.zeros(self.num_envs, 2, device=self.device)
        self._completed_actual_velocity = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        self._completed_command_velocity = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        self._last_heading = torch.zeros(self.num_envs, device=self.device)
        self._unwrapped_heading = torch.zeros(self.num_envs, device=self.device)
        self._cycle_start_heading = torch.zeros(self.num_envs, device=self.device)
        self._completed_yaw_rate = torch.zeros(self.num_envs, device=self.device)

    @staticmethod
    def _wrap(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._last_episode_step[env_ids] = -1
        self._command_displacement[env_ids] = 0.0
        self._completed_actual_velocity[env_ids] = 0.0
        self._completed_command_velocity[env_ids] = 0.0
        self._last_heading[env_ids] = 0.0
        self._unwrapped_heading[env_ids] = 0.0
        self._completed_yaw_rate[env_ids] = 0.0

    def _completed_cycle_motion(
        self,
        env,
        *,
        command_name: str,
        period: float,
        asset_cfg: SceneEntityCfg,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if period <= 0.0:
            raise ValueError("gait period must be positive")
        asset: Entity = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        assert command is not None, f"Command {command_name!r} not found"
        steps = env.episode_length_buf
        cycle_index = torch.floor(steps.to(torch.float32) * env.step_dt / period).long()
        current_xy = asset.data.root_link_pos_w[:, :2]
        current_heading = self._wrap(asset.data.heading_w)
        new_episode = (self._last_episode_step < 0) | (steps < self._last_episode_step)

        if torch.any(new_episode):
            self._cycle_index[new_episode] = cycle_index[new_episode]
            self._cycle_start_xy[new_episode] = current_xy[new_episode]
            self._command_displacement[new_episode] = 0.0
            self._completed_actual_velocity[new_episode] = 0.0
            self._completed_command_velocity[new_episode] = 0.0
            self._last_heading[new_episode] = current_heading[new_episode]
            self._unwrapped_heading[new_episode] = current_heading[new_episode]
            self._cycle_start_heading[new_episode] = current_heading[new_episode]
            self._completed_yaw_rate[new_episode] = 0.0

        continuing = ~new_episode
        if torch.any(continuing):
            self._command_displacement[continuing] += (
                command[continuing, :2] * env.step_dt
            )
            self._unwrapped_heading[continuing] += self._wrap(
                current_heading[continuing] - self._last_heading[continuing]
            )
        self._last_heading.copy_(current_heading)

        completed = (cycle_index != self._cycle_index) & continuing
        if torch.any(completed):
            self._completed_actual_velocity[completed] = (
                current_xy[completed] - self._cycle_start_xy[completed]
            ) / period
            self._completed_command_velocity[completed] = (
                self._command_displacement[completed] / period
            )
            self._completed_yaw_rate[completed] = (
                self._unwrapped_heading[completed]
                - self._cycle_start_heading[completed]
            ) / period
            self._cycle_index[completed] = cycle_index[completed]
            self._cycle_start_xy[completed] = current_xy[completed]
            self._command_displacement[completed] = 0.0
            self._cycle_start_heading[completed] = self._unwrapped_heading[completed]

        self._last_episode_step.copy_(steps)
        return (
            self._completed_actual_velocity,
            self._completed_command_velocity,
            self._completed_yaw_rate,
        )


class _CommandedCycleDisplacementTerm(ManagerTermBase):
    """Integrate body-frame velocity commands and actual world motion per gait cycle."""

    def __init__(self, cfg, env) -> None:
        del cfg
        super().__init__(env)
        self._last_episode_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._cycle_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._cycle_start_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._cycle_start_heading = torch.zeros(self.num_envs, device=self.device)
        self._last_heading = torch.zeros(self.num_envs, device=self.device)
        self._unwrapped_heading = torch.zeros(self.num_envs, device=self.device)
        self._command_displacement = torch.zeros(self.num_envs, 3, device=self.device)
        self._completed_error = torch.zeros(self.num_envs, 3, device=self.device)
        self._completed_command = torch.zeros(self.num_envs, 3, device=self.device)

    @staticmethod
    def _wrap(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._last_episode_step[env_ids] = -1
        self._command_displacement[env_ids] = 0.0
        self._completed_error[env_ids] = 0.0
        self._completed_command[env_ids] = 0.0

    def _completed_cycle_error(
        self,
        env,
        *,
        command_name: str,
        period: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        if period <= 0.0:
            raise ValueError("gait period must be positive")
        asset: Entity = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        assert command is not None, f"Command {command_name!r} not found"
        steps = env.episode_length_buf
        cycle_index = torch.floor(steps.to(torch.float32) * env.step_dt / period).long()
        current_xy = asset.data.root_link_pos_w[:, :2]
        current_heading = self._wrap(asset.data.heading_w)
        new_episode = (self._last_episode_step < 0) | (steps < self._last_episode_step)

        if torch.any(new_episode):
            self._cycle_index[new_episode] = cycle_index[new_episode]
            self._cycle_start_xy[new_episode] = current_xy[new_episode]
            self._cycle_start_heading[new_episode] = current_heading[new_episode]
            self._last_heading[new_episode] = current_heading[new_episode]
            self._unwrapped_heading[new_episode] = current_heading[new_episode]
            self._command_displacement[new_episode] = 0.0
            self._completed_error[new_episode] = 0.0
            self._completed_command[new_episode] = 0.0

        continuing = ~new_episode
        if torch.any(continuing):
            heading_delta = self._wrap(
                current_heading[continuing] - self._last_heading[continuing]
            )
            midpoint_heading = self._unwrapped_heading[continuing] + 0.5 * heading_delta
            cosine = torch.cos(midpoint_heading)
            sine = torch.sin(midpoint_heading)
            vx = command[continuing, 0]
            vy = command[continuing, 1]
            self._command_displacement[continuing, 0] += (
                cosine * vx - sine * vy
            ) * env.step_dt
            self._command_displacement[continuing, 1] += (
                sine * vx + cosine * vy
            ) * env.step_dt
            self._command_displacement[continuing, 2] += (
                command[continuing, 2] * env.step_dt
            )
            self._unwrapped_heading[continuing] += heading_delta
        self._last_heading.copy_(current_heading)

        completed = (cycle_index != self._cycle_index) & continuing
        if torch.any(completed):
            actual_displacement = torch.cat(
                (
                    current_xy[completed] - self._cycle_start_xy[completed],
                    (
                        self._unwrapped_heading[completed]
                        - self._cycle_start_heading[completed]
                    )[:, None],
                ),
                dim=1,
            )
            self._completed_error[completed] = (
                actual_displacement - self._command_displacement[completed]
            )
            self._completed_command[completed] = (
                self._command_displacement[completed] / period
            )
            self._cycle_index[completed] = cycle_index[completed]
            self._cycle_start_xy[completed] = current_xy[completed]
            self._cycle_start_heading[completed] = self._unwrapped_heading[completed]
            self._command_displacement[completed] = 0.0

        self._last_episode_step.copy_(steps)
        return self._completed_error


class _CommandedEpisodeDisplacementTerm(ManagerTermBase):
    """Integrate command and actual SE(2) motion over a complete episode."""

    def __init__(self, cfg, env) -> None:
        del cfg
        super().__init__(env)
        self._last_episode_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._episode_start_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._episode_start_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._episode_start_heading = torch.zeros(self.num_envs, device=self.device)
        self._last_heading = torch.zeros(self.num_envs, device=self.device)
        self._unwrapped_heading = torch.zeros(self.num_envs, device=self.device)
        self._command_displacement = torch.zeros(
            self.num_envs, 3, device=self.device
        )

    @staticmethod
    def _wrap(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._last_episode_step[env_ids] = -1
        self._episode_start_step[env_ids] = 0
        self._command_displacement[env_ids] = 0.0
        self._last_heading[env_ids] = 0.0
        self._unwrapped_heading[env_ids] = 0.0

    def _episode_error(
        self,
        env,
        *,
        command_name: str,
        asset_cfg: SceneEntityCfg,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        asset: Entity = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        assert command is not None, f"Command {command_name!r} not found"
        steps = env.episode_length_buf
        current_xy = asset.data.root_link_pos_w[:, :2]
        current_heading = self._wrap(asset.data.heading_w)
        new_episode = (self._last_episode_step < 0) | (steps < self._last_episode_step)

        if torch.any(new_episode):
            self._episode_start_step[new_episode] = steps[new_episode]
            self._episode_start_xy[new_episode] = current_xy[new_episode]
            self._episode_start_heading[new_episode] = current_heading[new_episode]
            self._last_heading[new_episode] = current_heading[new_episode]
            self._unwrapped_heading[new_episode] = current_heading[new_episode]
            self._command_displacement[new_episode] = 0.0

        continuing = ~new_episode
        if torch.any(continuing):
            heading_delta = self._wrap(
                current_heading[continuing] - self._last_heading[continuing]
            )
            midpoint_heading = self._unwrapped_heading[continuing] + 0.5 * heading_delta
            cosine = torch.cos(midpoint_heading)
            sine = torch.sin(midpoint_heading)
            vx = command[continuing, 0]
            vy = command[continuing, 1]
            self._command_displacement[continuing, 0] += (
                cosine * vx - sine * vy
            ) * env.step_dt
            self._command_displacement[continuing, 1] += (
                sine * vx + cosine * vy
            ) * env.step_dt
            self._command_displacement[continuing, 2] += (
                command[continuing, 2] * env.step_dt
            )
            self._unwrapped_heading[continuing] += heading_delta
        self._last_heading.copy_(current_heading)
        self._last_episode_step.copy_(steps)

        actual_displacement = torch.cat(
            (
                current_xy - self._episode_start_xy,
                (self._unwrapped_heading - self._episode_start_heading)[:, None],
            ),
            dim=1,
        )
        error = actual_displacement - self._command_displacement
        elapsed = torch.clamp(
            (steps - self._episode_start_step).to(torch.float32) * env.step_dt,
            min=float(env.step_dt),
        )
        return error, elapsed


class CommandedEpisodeDisplacementCost(_CommandedEpisodeDisplacementTerm):
    """Long-horizon command tracking cost on episode-average SE(2) motion.

    The accumulated displacement is divided by elapsed episode time before
    applying the scales, keeping the reward magnitude independent of episode
    length while still exposing persistent under-travel across many gait cycles.
    """

    def __call__(
        self,
        env,
        command_name: str,
        linear_scale: float = 0.08,
        yaw_scale: float = 0.40,
        yaw_deadband: float | None = None,
        yaw_relative_scale: float | None = None,
        yaw_scale_floor: float = 0.05,
        coupled_vx_threshold: float = 0.02,
        max_cost: float = 12.0,
        warmup_s: float = 0.50,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        if min(linear_scale, yaw_scale, max_cost, warmup_s) <= 0.0:
            raise ValueError("episode displacement cost parameters must be positive")
        if yaw_deadband is not None and yaw_deadband < 0.0:
            raise ValueError("yaw deadband must be non-negative")
        if yaw_relative_scale is not None and yaw_relative_scale <= 0.0:
            raise ValueError("relative yaw scale must be positive")
        error, elapsed = self._episode_error(
            env, command_name=command_name, asset_cfg=asset_cfg
        )
        average_error = error / elapsed[:, None]
        cost = torch.sum(
            _pseudo_huber(average_error[:, :2] / linear_scale), dim=1
        )
        if yaw_relative_scale is None or yaw_deadband is None:
            cost += _pseudo_huber(average_error[:, 2] / yaw_scale)
        else:
            expected_yaw_rate = self._command_displacement[:, 2] / elapsed
            active = (expected_yaw_rate.abs() >= yaw_deadband) & (
                self._command_displacement[:, :2].norm(dim=1) >= coupled_vx_threshold * elapsed
            )
            denominator = torch.maximum(
                expected_yaw_rate.abs() * yaw_relative_scale,
                expected_yaw_rate.new_tensor(yaw_scale_floor),
            )
            normalized = torch.where(
                active,
                average_error[:, 2] / denominator,
                torch.zeros_like(average_error[:, 2]),
            )
            cost += _pseudo_huber(normalized)
        warmup = torch.clamp(elapsed / warmup_s, max=1.0)
        return torch.clamp(cost, max=max_cost) * warmup


def _pseudo_huber(normalized_error: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(1.0 + torch.square(normalized_error)) - 1.0


class CommandedCycleDisplacementCost(_CommandedCycleDisplacementTerm):
    """Primary command-tracking cost on integrated SE(2) cycle displacement."""

    def __call__(
        self,
        env,
        command_name: str,
        period: float,
        sagittal_scale: float = 0.012,
        lateral_scale: float = 0.012,
        yaw_scale: float = 0.08,
        yaw_deadband: float | None = None,
        yaw_relative_scale: float | None = None,
        yaw_scale_floor: float = 0.05,
        coupled_vx_threshold: float = 0.02,
        max_cost: float = 12.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        if min(sagittal_scale, lateral_scale, yaw_scale, max_cost) <= 0.0:
            raise ValueError("cycle displacement cost parameters must be positive")
        error = self._completed_cycle_error(
            env, command_name=command_name, period=period, asset_cfg=asset_cfg
        )
        scales = error.new_tensor((sagittal_scale, lateral_scale, yaw_scale))
        cost = torch.sum(_pseudo_huber(error[:, :2] / scales[:2]), dim=1)
        if yaw_relative_scale is None or yaw_deadband is None:
            cost += _pseudo_huber(error[:, 2] / yaw_scale)
        else:
            expected_yaw_rate = self._completed_command[:, 2]
            coupled_vx = self._completed_command[:, :2].norm(dim=1)
            active = (expected_yaw_rate.abs() >= yaw_deadband) & (
                coupled_vx >= coupled_vx_threshold
            )
            denominator = torch.maximum(
                expected_yaw_rate.abs() * yaw_relative_scale,
                expected_yaw_rate.new_tensor(yaw_scale_floor),
            )
            normalized = torch.where(
                active,
                error[:, 2] / denominator,
                torch.zeros_like(error[:, 2]),
            )
            cost += _pseudo_huber(normalized)
        return torch.clamp(cost, max=max_cost)


class CommandedCycleDisplacementError(_CommandedCycleDisplacementTerm):
    """Absolute completed-cycle displacement error for one SE(2) component."""

    def __call__(
        self,
        env,
        command_name: str,
        period: float,
        component: int,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        if component not in (0, 1, 2):
            raise ValueError("cycle displacement component must be 0, 1, or 2")
        error = self._completed_cycle_error(
            env, command_name=command_name, period=period, asset_cfg=asset_cfg
        )
        return torch.abs(error[:, component])


def instantaneous_world_command_velocity_cost(
    env,
    command_name: str,
    linear_scale: float = 0.08,
    yaw_scale: float = 0.40,
    max_cost: float = 4.0,
    yaw_deadband: float | None = None,
    yaw_relative_scale: float | None = None,
    yaw_scale_floor: float = 0.05,
    coupled_vx_threshold: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Secondary tracking cost for instantaneous world velocity and body yaw rate."""
    if min(linear_scale, yaw_scale, max_cost, yaw_scale_floor, coupled_vx_threshold) <= 0.0:
        raise ValueError("instantaneous velocity cost parameters must be positive")
    if yaw_deadband is not None and yaw_deadband < 0.0:
        raise ValueError("yaw deadband must be non-negative")
    if yaw_relative_scale is not None and yaw_relative_scale <= 0.0:
        raise ValueError("relative yaw scale must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command {command_name!r} not found"
    heading = asset.data.heading_w
    cosine = torch.cos(heading)
    sine = torch.sin(heading)
    target_x = cosine * command[:, 0] - sine * command[:, 1]
    target_y = sine * command[:, 0] + cosine * command[:, 1]
    linear_error = asset.data.root_link_lin_vel_w[:, :2] - torch.stack(
        (target_x, target_y), dim=1
    )
    yaw_error = asset.data.root_link_ang_vel_b[:, 2] - command[:, 2]
    cost = torch.sum(_pseudo_huber(linear_error / linear_scale), dim=1)
    if yaw_relative_scale is None or yaw_deadband is None:
        cost += _pseudo_huber(yaw_error / yaw_scale)
    else:
        yaw_active = (torch.abs(command[:, 0]) >= coupled_vx_threshold) & (
            torch.abs(command[:, 2]) >= yaw_deadband
        )
        denominator = torch.maximum(
            torch.abs(command[:, 2]) * yaw_relative_scale,
            command[:, 2].new_tensor(yaw_scale_floor),
        )
        normalized = torch.where(
            yaw_active,
            yaw_error / denominator,
            torch.zeros_like(yaw_error),
        )
        cost += _pseudo_huber(normalized)
    return torch.clamp(cost, max=max_cost)


def _lateral_command_activity(command_y: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0.0:
        raise ValueError("lateral command activity scale must be positive")
    squared = torch.square(command_y)
    return squared / (squared + scale**2)


class CommandedCyclePlanarVelocityCost(_CommandedCycleMotionTerm):
    """Cycle-average Y tracking plus X tracking activated by lateral commands."""

    def __call__(
        self,
        env,
        command_name: str,
        period: float,
        lateral_std: float = 0.02,
        forward_std: float = 0.05,
        lateral_gate_scale: float = 0.02,
        forward_weight: float = 1.0,
        max_cost: float = 4.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        if min(lateral_std, forward_std, forward_weight, max_cost) <= 0.0:
            raise ValueError("cycle planar velocity cost parameters must be positive")
        actual, commanded, _ = self._completed_cycle_motion(
            env, command_name=command_name, period=period, asset_cfg=asset_cfg
        )
        error = actual - commanded
        activity = _lateral_command_activity(commanded[:, 1], lateral_gate_scale)
        cost = torch.square(error[:, 1] / lateral_std)
        cost += forward_weight * activity * torch.square(error[:, 0] / forward_std)
        return torch.clamp(cost, max=max_cost)


class CommandedCycleTrackingGate(_CommandedCycleMotionTerm):
    """Diagnostic score for commanded planar cycle motion without net yaw."""

    def __call__(
        self,
        env,
        command_name: str,
        period: float,
        lateral_std: float = 0.02,
        forward_std: float = 0.05,
        yaw_std: float = 0.12,
        lateral_gate_scale: float = 0.02,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        if min(lateral_std, forward_std, yaw_std) <= 0.0:
            raise ValueError("cycle command tracking standard deviations must be positive")
        actual, commanded, yaw_rate = self._completed_cycle_motion(
            env, command_name=command_name, period=period, asset_cfg=asset_cfg
        )
        error = actual - commanded
        activity = _lateral_command_activity(commanded[:, 1], lateral_gate_scale)
        squared_error = torch.square(error[:, 1] / lateral_std)
        squared_error += activity * torch.square(error[:, 0] / forward_std)
        squared_error += torch.square(yaw_rate / yaw_std)
        return torch.exp(-squared_error)


class _CommandedPlanarPathTerm(ManagerTermBase):
    """Integrate the piecewise velocity command into a world-frame path."""

    def __init__(self, cfg, env) -> None:
        del cfg
        super().__init__(env)
        self._last_episode_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._reference_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._elapsed = torch.zeros(self.num_envs, device=self.device)
        self._lateral_command_integral = torch.zeros(
            self.num_envs, device=self.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._last_episode_step[env_ids] = -1
        self._reference_xy[env_ids] = 0.0
        self._elapsed[env_ids] = 0.0
        self._lateral_command_integral[env_ids] = 0.0

    def _path_state(
        self,
        env,
        *,
        command_name: str,
        asset_cfg: SceneEntityCfg,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        asset: Entity = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        assert command is not None, f"Command {command_name!r} not found"
        steps = env.episode_length_buf
        new_episode = (self._last_episode_step < 0) | (steps < self._last_episode_step)
        origins = env.scene.env_origins[:, :2]

        if torch.any(new_episode):
            elapsed = steps[new_episode].to(torch.float32) * env.step_dt
            self._reference_xy[new_episode] = origins[new_episode] + (
                command[new_episode, :2] * elapsed[:, None]
            )
            self._elapsed[new_episode] = elapsed
            self._lateral_command_integral[new_episode] = (
                torch.abs(command[new_episode, 1]) * elapsed
            )

        continuing = ~new_episode
        if torch.any(continuing):
            self._reference_xy[continuing] += command[continuing, :2] * env.step_dt
            self._elapsed[continuing] += env.step_dt
            self._lateral_command_integral[continuing] += (
                torch.abs(command[continuing, 1]) * env.step_dt
            )

        self._last_episode_step.copy_(steps)
        path_error = asset.data.root_link_pos_w[:, :2] - self._reference_xy
        average_velocity_error = path_error / torch.clamp(
            self._elapsed, min=0.20
        )[:, None]
        average_lateral_command = self._lateral_command_integral / torch.clamp(
            self._elapsed, min=env.step_dt
        )
        return path_error, average_velocity_error, average_lateral_command


class CommandedPlanarPathCost(_CommandedPlanarPathTerm):
    """Y path tracking plus X tracking activated by lateral command history."""

    def __call__(
        self,
        env,
        command_name: str,
        lateral_tolerance: float = 0.08,
        forward_tolerance: float = 0.20,
        lateral_gate_scale: float = 0.02,
        forward_weight: float = 1.0,
        max_cost: float = 4.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        if min(lateral_tolerance, forward_tolerance, forward_weight, max_cost) <= 0.0:
            raise ValueError("commanded path cost parameters must be positive")
        error, _, lateral_command = self._path_state(
            env, command_name=command_name, asset_cfg=asset_cfg
        )
        activity = _lateral_command_activity(lateral_command, lateral_gate_scale)
        cost = torch.square(error[:, 1] / lateral_tolerance)
        cost += forward_weight * activity * torch.square(
            error[:, 0] / forward_tolerance
        )
        return torch.clamp(cost, max=max_cost)


class CommandedPlanarPathError(_CommandedPlanarPathTerm):
    def __call__(
        self,
        env,
        command_name: str,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        error, _, _ = self._path_state(
            env, command_name=command_name, asset_cfg=asset_cfg
        )
        return torch.linalg.vector_norm(error, dim=1)


class CommandedPlanarVelocityError(_CommandedPlanarPathTerm):
    def __call__(
        self,
        env,
        command_name: str,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        _, error, _ = self._path_state(
            env, command_name=command_name, asset_cfg=asset_cfg
        )
        return torch.linalg.vector_norm(error, dim=1)


def commanded_motion_rate_safety_cost(
    env,
    command_name: str,
    lateral_soft_limit: float = 0.12,
    lateral_scale: float = 0.04,
    yaw_soft_limit: float = 0.90,
    yaw_scale: float = 0.35,
    max_cost: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bound fast motion relative to the requested lateral and yaw rates."""
    if min(lateral_soft_limit, lateral_scale, yaw_soft_limit, yaw_scale, max_cost) <= 0.0:
        raise ValueError("motion-rate safety parameters must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command {command_name!r} not found"
    # Commands are body-frame velocities, so compare them with the measured
    # body-frame lateral velocity.  Comparing world-Y to body-Y is incorrect
    # as soon as the robot has a nonzero heading.
    lateral_excess = torch.relu(
        torch.abs(asset.data.root_link_lin_vel_b[:, 1] - command[:, 1])
        - lateral_soft_limit
    ) / lateral_scale
    yaw_excess = torch.relu(
        torch.abs(asset.data.root_link_ang_vel_b[:, 2] - command[:, 2])
        - yaw_soft_limit
    ) / yaw_scale
    return torch.clamp(
        torch.square(lateral_excess) + torch.square(yaw_excess), max=max_cost
    )


class UpdatedDefaultCycleLateralVelocityCost(_CycleNetMotionTerm):
    """Gradually tighten the completed-cycle lateral-drift tolerance."""

    def __call__(
        self,
        env,
        period: float,
        initial_std: float = 0.08,
        final_std: float = 0.02,
        ramp_steps: int = 12_000,
        max_cost: float = 4.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        if initial_std <= 0.0 or final_std <= 0.0 or max_cost <= 0.0:
            raise ValueError("cycle lateral cost parameters must be positive")
        std = _curriculum_value(env, initial_std, final_std, ramp_steps)
        lateral_speed, _ = self._completed_cycle_rates(
            env, period=period, asset_cfg=asset_cfg
        )
        return torch.clamp(torch.square(lateral_speed / std), max=max_cost)


class UpdatedDefaultCycleYawRateCost(_CycleNetMotionTerm):
    """Gradually tighten the completed-cycle yaw-drift tolerance."""

    def __call__(
        self,
        env,
        period: float,
        initial_std: float = 0.45,
        final_std: float = 0.12,
        ramp_steps: int = 12_000,
        max_cost: float = 4.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        if initial_std <= 0.0 or final_std <= 0.0 or max_cost <= 0.0:
            raise ValueError("cycle yaw cost parameters must be positive")
        std = _curriculum_value(env, initial_std, final_std, ramp_steps)
        _, yaw_rate = self._completed_cycle_rates(
            env, period=period, asset_cfg=asset_cfg
        )
        return torch.clamp(torch.square(yaw_rate / std), max=max_cost)


class UpdatedDefaultCycleStraightnessGate(_CycleNetMotionTerm):
    """Diagnostic cycle straightness score with a gradually tighter tolerance."""

    def __call__(
        self,
        env,
        period: float,
        initial_lateral_std: float = 0.08,
        final_lateral_std: float = 0.02,
        initial_yaw_std: float = 0.45,
        final_yaw_std: float = 0.12,
        ramp_steps: int = 12_000,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        lateral_std = _curriculum_value(
            env, initial_lateral_std, final_lateral_std, ramp_steps
        )
        yaw_std = _curriculum_value(env, initial_yaw_std, final_yaw_std, ramp_steps)
        lateral_speed, yaw_rate = self._completed_cycle_rates(
            env, period=period, asset_cfg=asset_cfg
        )
        return torch.exp(
            -torch.square(lateral_speed / lateral_std)
            -torch.square(yaw_rate / yaw_std)
        )


def updated_default_lateral_path_cost(
    env,
    initial_tolerance: float = 0.20,
    final_tolerance: float = 0.08,
    ramp_steps: int = 12_000,
    max_cost: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gradually penalize accumulated world-Y path error after gait formation."""
    if initial_tolerance <= 0.0 or final_tolerance <= 0.0 or max_cost <= 0.0:
        raise ValueError("lateral path cost parameters must be positive")
    tolerance = _curriculum_value(
        env, initial_tolerance, final_tolerance, ramp_steps
    )
    asset: Entity = env.scene[asset_cfg.name]
    lateral_offset = asset.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return _curriculum_progress(env, ramp_steps) * torch.clamp(
        torch.square(lateral_offset / tolerance), max=max_cost
    )


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
    if foot_site_cfg.site_names is not None and isinstance(foot_site_cfg.site_ids, slice):
        foot_site_cfg.resolve(env.scene)
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


def quasi_omnidirectional_velocity_success(
    env,
    command_name: str,
    maximum_linear_error: float = 0.05,
    maximum_yaw_error: float = 0.25,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Command-relative held-out success signal for planar and yaw velocity."""
    if maximum_linear_error <= 0.0 or maximum_yaw_error <= 0.0:
        raise ValueError("velocity-success tolerances must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command {command_name!r} not found"
    heading = asset.data.heading_w
    cosine = torch.cos(heading)
    sine = torch.sin(heading)
    target_world = torch.stack(
        (
            cosine * command[:, 0] - sine * command[:, 1],
            sine * command[:, 0] + cosine * command[:, 1],
        ),
        dim=1,
    )
    linear_error = torch.linalg.vector_norm(
        asset.data.root_link_lin_vel_w[:, :2] - target_world, dim=1
    )
    yaw_error = torch.abs(asset.data.root_link_ang_vel_b[:, 2] - command[:, 2])
    upright = -asset.data.projected_gravity_b[:, 2] >= 0.9
    return (
        (linear_error <= maximum_linear_error)
        & (yaw_error <= maximum_yaw_error)
        & upright
    ).float()


def _updated_default_eval_heading(env, current_heading: torch.Tensor) -> torch.Tensor:
    state = getattr(env, "_updated_default_eval_heading_state", None)
    if state is None or state[0].shape != current_heading.shape:
        state = (
            current_heading.clone(),
            current_heading.clone(),
            torch.full_like(env.episode_length_buf, -1),
        )
        setattr(env, "_updated_default_eval_heading_state", state)
    last_heading, unwrapped_heading, last_step = state
    steps = env.episode_length_buf
    new_episode = (last_step < 0) | (steps < last_step)
    if torch.any(new_episode):
        last_heading[new_episode] = current_heading[new_episode]
        unwrapped_heading[new_episode] = current_heading[new_episode]
    continuing = ~new_episode
    if torch.any(continuing):
        unwrapped_heading[continuing] += torch.atan2(
            torch.sin(current_heading[continuing] - last_heading[continuing]),
            torch.cos(current_heading[continuing] - last_heading[continuing]),
        )
    last_heading.copy_(current_heading)
    last_step.copy_(steps)
    return unwrapped_heading


def updated_default_walking_success(
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
    maximum_heading_error: float = 0.20,
    maximum_lateral_deviation: float = 0.08,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    foot_site_cfg: SceneEntityCfg = _FOOT_SITES_CFG,
) -> torch.Tensor:
    """Long-horizon walking pass signal using unwrapped heading and path drift."""
    if foot_site_cfg.site_names is not None and isinstance(foot_site_cfg.site_ids, slice):
        foot_site_cfg.resolve(env.scene)
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
    stance_ok = periodic_stance_foot_speed_cost(
        env,
        period=period,
        duty_factor=duty_factor,
        transition_fraction=transition_fraction,
        asset_cfg=foot_site_cfg,
    ) <= maximum_stance_speed_cost
    asset: Entity = env.scene[asset_cfg.name]
    current_heading = torch.atan2(
        torch.sin(asset.data.heading_w), torch.cos(asset.data.heading_w)
    )
    heading = _updated_default_eval_heading(env, current_heading)
    lateral = torch.abs(
        asset.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    )
    path_ok = (torch.abs(heading) <= maximum_heading_error) & (
        lateral <= maximum_lateral_deviation
    )
    return (speed_ok & swing_ok & stance_ok & path_ok).float()


def updated_default_lateral_walking_success(
    env,
    sensor_name: str,
    command_name: str,
    period: float = 0.33,
    duty_factor: float = 0.70,
    transition_fraction: float = 0.04,
    maximum_average_velocity_error: float = 0.04,
    maximum_path_error: float = 0.20,
    maximum_swing_force_cost: float = 0.35,
    maximum_stance_speed_cost: float = 0.35,
    maximum_heading_error: float = 0.20,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    foot_site_cfg: SceneEntityCfg = _FOOT_SITES_CFG,
) -> torch.Tensor:
    """Evaluate piecewise planar command tracking with an unwrapped heading."""
    if min(maximum_average_velocity_error, maximum_path_error) <= 0.0:
        raise ValueError("planar walking evaluation tolerances must be positive")
    if foot_site_cfg.site_names is not None and isinstance(foot_site_cfg.site_ids, slice):
        foot_site_cfg.resolve(env.scene)
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command {command_name!r} not found"
    asset: Entity = env.scene[asset_cfg.name]
    steps = env.episode_length_buf
    state = getattr(env, "_updated_default_lateral_eval_path_state", None)
    if state is None or state[0].shape != command[:, :2].shape:
        state = (
            torch.zeros_like(command[:, :2]),
            torch.zeros(env.num_envs, device=env.device),
            torch.full_like(steps, -1),
        )
        setattr(env, "_updated_default_lateral_eval_path_state", state)
    reference_xy, elapsed, last_step = state
    new_episode = (last_step < 0) | (steps < last_step)
    origins = env.scene.env_origins[:, :2]
    if torch.any(new_episode):
        initial_elapsed = steps[new_episode].to(torch.float32) * env.step_dt
        reference_xy[new_episode] = origins[new_episode] + (
            command[new_episode, :2] * initial_elapsed[:, None]
        )
        elapsed[new_episode] = initial_elapsed
    continuing = ~new_episode
    if torch.any(continuing):
        reference_xy[continuing] += command[continuing, :2] * env.step_dt
        elapsed[continuing] += env.step_dt
    last_step.copy_(steps)

    path_error = torch.linalg.vector_norm(
        asset.data.root_link_pos_w[:, :2] - reference_xy, dim=1
    )
    average_velocity_error = path_error / torch.clamp(elapsed, min=0.20)
    velocity_ok = average_velocity_error <= maximum_average_velocity_error
    swing_ok = periodic_swing_force_cost(
        env,
        sensor_name,
        period=period,
        duty_factor=duty_factor,
        transition_fraction=transition_fraction,
    ) <= maximum_swing_force_cost
    stance_ok = periodic_stance_foot_speed_cost(
        env,
        period=period,
        duty_factor=duty_factor,
        transition_fraction=transition_fraction,
        asset_cfg=foot_site_cfg,
    ) <= maximum_stance_speed_cost
    current_heading = torch.atan2(
        torch.sin(asset.data.heading_w), torch.cos(asset.data.heading_w)
    )
    heading = _updated_default_eval_heading(env, current_heading)
    path_ok = (path_error <= maximum_path_error) & (
        torch.abs(heading) <= maximum_heading_error
    )
    return (velocity_ok & swing_ok & stance_ok & path_ok).float()


def periodic_walk_evaluation_active(env, warmup_s: float = 2.0) -> torch.Tensor:
    """Ignore start-up transient before judging a net-displacement gait."""
    if warmup_s <= 0.0:
        raise ValueError("evaluation warmup must be positive")
    elapsed = env.episode_length_buf.to(torch.float32) * env.step_dt
    return elapsed >= warmup_s


def actuator_effort_cost(
    env,
    effort_scale: float = 1.5,
    joint_weights: tuple[float, ...] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bounded actuator-force cost, optionally weighted per joint.

    ``joint_weights`` deliberately is not normalized: a value below one
    reduces that joint's contribution to the task's effort penalty, while a
    value above one increases it.  This lets a task preserve the global
    effort regularizer without making expensive thigh/knee motion impossible.
    """
    if effort_scale <= 0.0:
        raise ValueError("effort scale must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    effort = torch.abs(asset.data.qfrc_actuator)
    normalized = 1.0 - torch.exp(-effort / effort_scale)
    if joint_weights is not None:
        weights = effort.new_tensor(joint_weights)
        if weights.ndim != 1 or weights.numel() != effort.shape[1]:
            raise ValueError(
                "joint effort weights must contain one non-negative value per actuator"
            )
        if torch.any(weights < 0.0):
            raise ValueError("joint effort weights must be non-negative")
        normalized = normalized * weights[None, :]
    return torch.mean(normalized, dim=1)


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


def commanded_motion_gate(
    env,
    command_name: str,
    full_motion_speed: float = 0.04,
) -> torch.Tensor:
    """Smoothly enable gait shaping only when planar travel is requested."""
    if full_motion_speed <= 0.0:
        raise ValueError("full motion speed must be positive")
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command {command_name!r} not found"
    planar_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
    return torch.clamp(planar_speed / full_motion_speed, 0.0, 1.0)


def commanded_scheduled_foot_contact(
    env,
    command_name: str,
    sensor_name: str,
    period: float = 0.33,
    duty_factor: float = 0.70,
    full_motion_speed: float = 0.04,
) -> torch.Tensor:
    return commanded_motion_gate(env, command_name, full_motion_speed) * scheduled_foot_contact(
        env, sensor_name, period, duty_factor
    )


def commanded_periodic_swing_force_cost(
    env,
    command_name: str,
    sensor_name: str,
    period: float = 0.33,
    duty_factor: float = 0.70,
    transition_fraction: float = 0.04,
    force_scale: float = 4.0,
    full_motion_speed: float = 0.04,
) -> torch.Tensor:
    return commanded_motion_gate(env, command_name, full_motion_speed) * periodic_swing_force_cost(
        env, sensor_name, period, duty_factor, transition_fraction, force_scale
    )


def commanded_periodic_stance_foot_speed_cost(
    env,
    command_name: str,
    period: float = 0.33,
    duty_factor: float = 0.70,
    transition_fraction: float = 0.04,
    speed_scale: float = 0.06,
    include_vertical: bool = False,
    full_motion_speed: float = 0.04,
    asset_cfg: SceneEntityCfg = _FOOT_SITES_CFG,
) -> torch.Tensor:
    return commanded_motion_gate(env, command_name, full_motion_speed) * periodic_stance_foot_speed_cost(
        env, period, duty_factor, transition_fraction, speed_scale,
        include_vertical, asset_cfg
    )


def commanded_periodic_reward_composition_swing_force_cost(
    env,
    command_name: str,
    sensor_name: str,
    period: float = 0.33,
    duty_factor: float = 0.70,
    transition_fraction: float = 0.04,
    force_scale: float = 10.0,
    full_motion_speed: float = 0.04,
) -> torch.Tensor:
    """Enable the paper-style swing-force composition only while walking."""
    return commanded_motion_gate(
        env, command_name, full_motion_speed
    ) * periodic_reward_composition_swing_force_cost(
        env, sensor_name, period, duty_factor, transition_fraction, force_scale
    )


def commanded_periodic_reward_composition_stance_speed_cost(
    env,
    command_name: str,
    period: float = 0.33,
    duty_factor: float = 0.70,
    transition_fraction: float = 0.04,
    speed_coefficient: float = 2.0,
    full_motion_speed: float = 0.04,
    asset_cfg: SceneEntityCfg = _FOOT_SITES_CFG,
) -> torch.Tensor:
    """Enable the paper-style stance-speed composition only while walking."""
    return commanded_motion_gate(
        env, command_name, full_motion_speed
    ) * periodic_reward_composition_stance_speed_cost(
        env, period, duty_factor, transition_fraction, speed_coefficient, asset_cfg
    )


def commanded_neutral_stance(
    env,
    command_name: str,
    joint_std: float = 0.12,
    full_motion_speed: float = 0.04,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """At stop, settle near the deployable nominal pose instead of a scissor stance."""
    if joint_std <= 0.0:
        raise ValueError("joint standard deviation must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    error = asset.data.joint_pos - asset.data.default_joint_pos
    pose = torch.exp(-torch.mean(torch.square(error / joint_std), dim=1))
    return (1.0 - commanded_motion_gate(env, command_name, full_motion_speed)) * pose


def stopped_foot_alignment_cost(
    env,
    command_name: str,
    fore_aft_scale: float = 0.015,
    full_motion_speed: float = 0.04,
    asset_cfg: SceneEntityCfg = _FOOT_SITES_CFG,
) -> torch.Tensor:
    """Remove the permanent one-foot-forward resting solution without shaping steps."""
    if fore_aft_scale <= 0.0:
        raise ValueError("fore-aft scale must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    separation_w = (
        asset.data.site_pos_w[:, asset_cfg.site_ids[0]]
        - asset.data.site_pos_w[:, asset_cfg.site_ids[1]]
    )
    heading = asset.data.heading_w
    fore_aft = heading.cos() * separation_w[:, 0] + heading.sin() * separation_w[:, 1]
    stopped = 1.0 - commanded_motion_gate(env, command_name, full_motion_speed)
    return stopped * torch.clamp(torch.square(fore_aft / fore_aft_scale), max=4.0)


def _pure_lateral_command_mask(
    env,
    command_name: str,
    min_lateral_speed: float = 0.04,
    max_forward_speed: float = 0.01,
    max_yaw_rate: float = 0.02,
) -> torch.Tensor:
    """Select pure lateral commands without activating on curves or stops."""
    if min_lateral_speed <= 0.0 or max_forward_speed < 0.0 or max_yaw_rate < 0.0:
        raise ValueError("invalid pure-lateral command thresholds")
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command {command_name!r} not found"
    return (
        (command[:, 1].abs() >= min_lateral_speed)
        & (command[:, 0].abs() <= max_forward_speed)
        & (command[:, 2].abs() <= max_yaw_rate)
    ).to(torch.float32)


def lateral_command_active(
    env,
    command_name: str,
    lateral_direction: int,
    min_lateral_speed: float = 0.04,
    max_forward_speed: float = 0.01,
    max_yaw_rate: float = 0.02,
) -> torch.Tensor:
    """Metric mask for one signed pure-lateral command mode."""
    if lateral_direction not in (-1, 1):
        raise ValueError("lateral_direction must be -1 or 1")
    command = env.command_manager.get_command(command_name)
    pure = _pure_lateral_command_mask(
        env, command_name, min_lateral_speed, max_forward_speed, max_yaw_rate
    )
    return pure * (
        command[:, 1] * lateral_direction >= min_lateral_speed
    ).to(pure.dtype)


def lateral_foot_line_error(
    env,
    command_name: str,
    min_lateral_speed: float = 0.04,
    max_forward_speed: float = 0.01,
    max_yaw_rate: float = 0.02,
    asset_cfg: SceneEntityCfg = _FOOT_SITES_CFG,
    lateral_direction: int | None = None,
) -> torch.Tensor:
    """Body-frame fore/aft foot separation during pure lateral commands."""
    asset: Entity = env.scene[asset_cfg.name]
    separation_w = (
        asset.data.site_pos_w[:, asset_cfg.site_ids[0]]
        - asset.data.site_pos_w[:, asset_cfg.site_ids[1]]
    )
    heading = asset.data.heading_w
    fore_aft = heading.cos() * separation_w[:, 0] + heading.sin() * separation_w[:, 1]
    mask = _pure_lateral_command_mask(
        env, command_name, min_lateral_speed, max_forward_speed, max_yaw_rate
    )
    if lateral_direction is not None:
        mask = mask * lateral_command_active(
            env,
            command_name,
            lateral_direction,
            min_lateral_speed,
            max_forward_speed,
            max_yaw_rate,
        )
    return mask * fore_aft.abs()


def commanded_lateral_foot_line_cost(
    env,
    command_name: str,
    line_scale: float = 0.025,
    period: float = 0.33,
    duty_factor: float = 0.70,
    transition_fraction: float = 0.04,
    min_lateral_speed: float = 0.04,
    max_forward_speed: float = 0.01,
    max_yaw_rate: float = 0.02,
    asset_cfg: SceneEntityCfg = _FOOT_SITES_CFG,
    lateral_direction: int | None = None,
) -> torch.Tensor:
    """Keep lateral foot placement on one fore/aft line in double support.

    The term is zero for standing, sagittal walking, and curves.  During a
    pure lateral command it is also faded out while either scheduled foot is
    swinging, so it shapes foot placement without preventing a real step.
    """
    if line_scale <= 0.0:
        raise ValueError("lateral foot-line scale must be positive")
    error = lateral_foot_line_error(
        env,
        command_name,
        min_lateral_speed,
        max_forward_speed,
        max_yaw_rate,
        asset_cfg,
        lateral_direction,
    )
    stance = 1.0 - smooth_swing_probability(
        env, period, duty_factor, transition_fraction
    )
    double_support = torch.prod(stance, dim=1)
    # Smooth-L1/Huber shaping remains informative for badly misaligned feet;
    # the former exponential term saturated at roughly 3*line_scale.
    normalized = error / line_scale
    absolute = normalized.abs()
    huber = torch.where(
        absolute <= 1.0,
        0.5 * normalized.square(),
        absolute - 0.5,
    )
    return double_support * huber


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


def base_pitch_magnitude(
    env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Absolute base pitch in radians for training diagnostics."""
    gravity = env.scene[asset_cfg.name].data.projected_gravity_b
    return torch.abs(torch.atan2(gravity[:, 0], -gravity[:, 2]))


def periodic_reward_composition_velocity_cost(
    env,
    command_name: str,
    axis: int,
    coefficient: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bounded pelvis velocity-error cost from Table II of Siekmann et al."""
    if axis not in (0, 1):
        raise ValueError("planar velocity axis must be 0 or 1")
    if coefficient <= 0.0:
        raise ValueError("velocity coefficient must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command {command_name!r} not found"
    error = torch.abs(command[:, axis] - asset.data.root_link_lin_vel_b[:, axis])
    return 1.0 - torch.exp(-coefficient * error)


def periodic_reward_composition_orientation_cost(
    env,
    coefficient: float = 3.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bounded full-base orientation cost relative to the upright world frame.

    The reset yaw is zero for this walking-only task.  Squaring the quaternion
    dot product makes the cost invariant to the equivalent ``q`` and ``-q``
    representations, as in the paper's Table II.
    """
    if coefficient <= 0.0:
        raise ValueError("orientation coefficient must be positive")
    quaternion = env.scene[asset_cfg.name].data.root_link_quat_w
    identity_dot_squared = torch.square(quaternion[:, 0])
    return 1.0 - torch.exp(-coefficient * (1.0 - identity_dot_squared))


def periodic_reward_composition_action_difference_cost(
    env, coefficient: float = 5.0
) -> torch.Tensor:
    """Bounded norm of consecutive normalized policy actions."""
    if coefficient <= 0.0:
        raise ValueError("action-difference coefficient must be positive")
    difference = env.action_manager.action - env.action_manager.prev_action
    return 1.0 - torch.exp(
        -coefficient * torch.linalg.vector_norm(difference, dim=1)
    )


def periodic_reward_composition_torque_cost(
    env,
    coefficient: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bounded actuator-effort norm from the paper's smoothness objective."""
    if coefficient <= 0.0:
        raise ValueError("torque coefficient must be positive")
    effort = env.scene[asset_cfg.name].data.qfrc_actuator
    return 1.0 - torch.exp(
        -coefficient * torch.linalg.vector_norm(effort, dim=1)
    )


class PeriodicRewardCompositionPelvisMotionCost(ManagerTermBase):
    """Bound angular velocity and finite-difference pelvis acceleration."""

    def __init__(self, cfg, env) -> None:
        del cfg
        super().__init__(env)
        self._last_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._last_linear_velocity = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._cached = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._last_step[env_ids] = -1
        self._last_linear_velocity[env_ids] = 0.0
        self._cached[env_ids] = 0.0

    def __call__(
        self,
        env,
        coefficient: float = 0.10,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        if coefficient <= 0.0:
            raise ValueError("pelvis-motion coefficient must be positive")
        asset: Entity = env.scene[asset_cfg.name]
        step = env.episode_length_buf
        update = step != self._last_step
        new_episode = (self._last_step < 0) | (step < self._last_step)
        acceleration = torch.zeros_like(asset.data.root_link_lin_vel_w)
        continuing = update & ~new_episode
        acceleration[continuing] = (
            asset.data.root_link_lin_vel_w[continuing]
            - self._last_linear_velocity[continuing]
        ) / env.step_dt
        cost = 1.0 - torch.exp(
            -coefficient
            * (
                torch.linalg.vector_norm(asset.data.root_link_ang_vel_b, dim=1)
                + torch.linalg.vector_norm(acceleration, dim=1)
            )
        )
        self._cached[update] = cost[update]
        self._last_linear_velocity[update] = asset.data.root_link_lin_vel_w[update]
        self._last_step[update] = step[update]
        return self._cached


def constant_alive_reward(env) -> torch.Tensor:
    """Paper reward bias beta=1; termination remains a separate safety cost."""
    return torch.ones(env.num_envs, device=env.device)


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
    "dof_damping",
    "dof_frictionloss",
    "dof_armature",
    "jnt_actfrcrange",
    "tendon_actfrcrange",
    recompute=RecomputeLevel.set_const,
)
def sim2real_domain_randomization(
    env,
    env_ids: torch.Tensor,
    randomize_latency: bool = True,
) -> None:
    """Apply a task-configured reset randomization envelope."""
    scale = sim2real_randomization_scale(env)
    robot = SceneEntityCfg("robot")
    feet = SceneEntityCfg(
        "robot", geom_names=("right_foot_collision", "left_foot_collision")
    )
    for asset_cfg in (robot, feet):
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
    _randomize_sts3215_servos(
        env, env_ids, paired=False, randomize_latency=randomize_latency
    )

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


def sim2real_joint_pos(
    env, maximum_noise: float = STS3215_RANDOMIZATION.encoder_noise_rad
):
    return _scaled_noise(
        env, env_mdp.joint_pos_rel(env, biased=True), maximum_noise
    )


def sim2real_joint_vel(env, maximum_noise: float = 1.5):
    return _scaled_noise(env, env_mdp.joint_vel_rel(env), maximum_noise)
