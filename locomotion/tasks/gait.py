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

from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

from locomotion.servo import (
    STS3215_MAX_PROFILE_ACCELERATION_RAD_S2,
    STS3215_MAX_PROFILE_SPEED_RAD_S,
    STS3215_POSITION_RESOLUTION_RAD,
    STS3215_VELOCITY_FILTER_ALPHA,
    quantize_position_torch,
)

from locomotion.imu import SimulatedIcm42688

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

_FOOT_SITES_CFG = SceneEntityCfg(
    "robot", site_names=("right_foot", "left_foot"), preserve_order=True
)

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
    calibration_scale_factor: tuple[float, float] = (0.995, 1.005)
    firmware_velocity_factor: tuple[float, float] = (0.95, 1.05)
    firmware_acceleration_factor: tuple[float, float] = (0.80, 1.20)
    encoder_bias_rad: float = math.radians(1.0)
    backlash_total_rad: tuple[float, float] = (
        math.radians(0.5),
        math.radians(2.4),
    )
    latency_lags: tuple[int, int] = (1, 3)
    encoder_noise_rad: float = math.radians(0.20)
    position_resolution_rad: float = STS3215_POSITION_RESOLUTION_RAD

STS3215_RANDOMIZATION = Sts3215RandomizationPrior()

class FootGroundRandomizationPrior:
    """Pre-hardware sole, pad, contact, and floor uncertainty."""

    sole_roll_rad: float = math.radians(1.0)
    sole_pitch_rad: float = math.radians(1.0)
    sole_height_m: float = 0.0005
    # MuJoCo's default refsafety rule requires timeconst >= 2 * physics_dt.
    # At LeggyBot's 5 ms physics step, 10 ms is the smallest effective value.
    contact_time_constant_s: tuple[float, float] = (0.010, 0.015)
    contact_damping_ratio: tuple[float, float] = (0.7, 1.3)
    compliance_width_m: tuple[float, float] = (0.0005, 0.002)
    torsional_friction_factor: tuple[float, float] = (0.6, 1.4)
    rolling_friction_factor: tuple[float, float] = (0.6, 1.4)
    floor_roll_rad: float = math.radians(1.5)
    floor_pitch_rad: float = math.radians(1.5)

FOOT_GROUND_RANDOMIZATION = FootGroundRandomizationPrior()

def _global_geom_ids(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Resolve an entity-local geom selection to MuJoCo global geom IDs."""
    asset_cfg.resolve(env.scene)
    entity_ids = env.scene[asset_cfg.name].indexing.geom_ids
    local_ids = asset_cfg.geom_ids
    if isinstance(local_ids, slice):
        return entity_ids[local_ids].to(device=env.device, dtype=torch.long)
    return entity_ids[torch.as_tensor(local_ids, device=env.device)].long()

def _scaled_uniform(
    env,
    shape: tuple[int, ...],
    value_range: tuple[float, float],
    nominal: float,
) -> torch.Tensor:
    raw = torch.empty(shape, device=env.device).uniform_(*value_range)
    scale = sim2real_randomization_scale(env)
    return nominal + scale * (raw - nominal)

def _randomize_foot_ground(
    env,
    env_ids: torch.Tensor,
    *,
    paired_geometry: bool,
) -> None:
    """Randomize collision-only soles, compliant contact, and floor slope.

    Material/contact properties are global per robot because both pads are cut
    from the same sheet. Sole mounting errors are mirrored during the symmetry
    phase and independent during full domain randomization.
    """
    prior = FOOT_GROUND_RANDOMIZATION
    scale = sim2real_randomization_scale(env)
    selected = torch.arange(env.num_envs, device=env.device)[env_ids].long()
    batch = len(selected)
    feet_cfg = SceneEntityCfg(
        "robot",
        geom_names=("right_foot_collision", "left_foot_collision"),
        preserve_order=True,
    )
    terrain_cfg = SceneEntityCfg(
        "terrain", geom_names=("terrain",), preserve_order=True
    )
    foot_ids = _global_geom_ids(env, feet_cfg)
    terrain_ids = _global_geom_ids(env, terrain_cfg)
    if len(foot_ids) != 2 or len(terrain_ids) != 1:
        raise ValueError("expected two foot collision geoms and one terrain plane")

    def centered(maximum: float, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, device=env.device).uniform_(
            -maximum * scale, maximum * scale
        )

    if paired_geometry:
        # In each foot's local frame, X is lateral, Y is forward, and Z is the
        # sole normal. A sagittal reflection preserves pitch/height and flips
        # roll. quat_from_euler_xyz therefore receives equal local-X rotations
        # and opposite local-Y rotations for the two feet.
        sole_pitch = centered(prior.sole_pitch_rad, (batch, 1))
        sole_roll = centered(prior.sole_roll_rad, (batch, 1))
        local_x = torch.cat((-sole_pitch, -sole_pitch), dim=1)
        local_y = torch.cat((sole_roll, -sole_roll), dim=1)
        height = centered(prior.sole_height_m, (batch, 1)).expand(-1, 2)
    else:
        local_x = -centered(prior.sole_pitch_rad, (batch, 2))
        local_y = centered(prior.sole_roll_rad, (batch, 2))
        height = centered(prior.sole_height_m, (batch, 2))

    zeros = torch.zeros_like(local_x)
    perturbation = quat_from_euler_xyz(
        local_x.flatten(), local_y.flatten(), zeros.flatten()
    ).reshape(batch, 2, 4)
    default_quat = env.sim.get_default_field("geom_quat")[foot_ids]
    default_quat = default_quat.unsqueeze(0).expand(batch, -1, -1).contiguous()
    # Post-multiply so roll/pitch are expressed in the nominal sole frame.
    # This is equivalent for today's identity foot-geom quaternions, but also
    # remains correct if the CAD export later gives either geom a local pose.
    foot_quat = quat_mul(default_quat, perturbation)
    env_grid, foot_grid = torch.meshgrid(selected, foot_ids, indexing="ij")
    model = env.sim.model
    model.geom_quat[env_grid, foot_grid] = foot_quat
    default_pos = env.sim.get_default_field("geom_pos")[foot_ids]
    model.geom_pos[env_grid, foot_grid] = default_pos.unsqueeze(0)
    model.geom_pos[env_grid, foot_grid, 2] += height

    # Both pads share one material realization. The positive solref format is
    # (time constant, damping ratio); solimp width approximates the finite
    # compression distance of a thin silicone/polyurethane pad.
    time_constant = _scaled_uniform(
        env, (batch, 1), prior.contact_time_constant_s, 0.010
    )
    damping_ratio = _scaled_uniform(
        env, (batch, 1), prior.contact_damping_ratio, 1.0
    )
    compliance_width = _scaled_uniform(
        env, (batch, 1), prior.compliance_width_m, 0.001
    )
    model.geom_solref[env_grid, foot_grid, 0] = time_constant.expand(-1, 2)
    model.geom_solref[env_grid, foot_grid, 1] = damping_ratio.expand(-1, 2)
    default_solimp = env.sim.get_default_field("geom_solimp")[foot_ids]
    model.geom_solimp[env_grid, foot_grid] = default_solimp.unsqueeze(0)
    model.geom_solimp[env_grid, foot_grid, 2] = compliance_width.expand(-1, 2)

    tangential = _scaled_uniform(env, (batch, 1), (0.5, 1.2), 0.9)
    torsion_factor = _scaled_uniform(
        env, (batch, 1), prior.torsional_friction_factor, 1.0
    )
    rolling_factor = _scaled_uniform(
        env, (batch, 1), prior.rolling_friction_factor, 1.0
    )
    default_friction = env.sim.get_default_field("geom_friction")[foot_ids]
    model.geom_friction[env_grid, foot_grid] = default_friction.unsqueeze(0)
    model.geom_friction[env_grid, foot_grid, 0] = tangential.expand(-1, 2)
    model.geom_friction[env_grid, foot_grid, 1] *= torsion_factor.expand(-1, 2)
    model.geom_friction[env_grid, foot_grid, 2] *= rolling_factor.expand(-1, 2)

    floor_roll = centered(prior.floor_roll_rad, (batch, 1))
    floor_pitch = centered(prior.floor_pitch_rad, (batch, 1))
    floor_perturbation = quat_from_euler_xyz(
        floor_roll.flatten(), floor_pitch.flatten(), torch.zeros(batch, device=env.device)
    ).reshape(batch, 1, 4)
    default_floor_quat = env.sim.get_default_field("geom_quat")[terrain_ids]
    default_floor_quat = default_floor_quat.unsqueeze(0).expand(batch, -1, -1).contiguous()
    floor_quat = quat_mul(floor_perturbation, default_floor_quat)
    terrain_grid = terrain_ids.unsqueeze(0).expand(batch, -1)
    selected_grid = selected.unsqueeze(1).expand_as(terrain_grid)
    model.geom_quat[selected_grid, terrain_grid] = floor_quat

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
    for actuator_name in (
        "right_hip_roll", "right_thigh", "right_knee",
        "left_hip_roll", "left_thigh", "left_knee",
    ):
        _, actuator = _single_actuator(env, actuator_name)
        ctrl_ids.append(int(torch.as_tensor(actuator.global_ctrl_ids).reshape(-1)[0]))
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
    calibration_scale = samples(prior.calibration_scale_factor)
    firmware_velocity = samples(prior.firmware_velocity_factor)
    firmware_acceleration = samples(prior.firmware_acceleration_factor)

    backlash_total = _paired_or_independent_samples(
        env, batch, n, prior.backlash_total_rad, paired
    ) * scale
    action = env.action_manager.get_term("joint_pos")
    action_width = torch.zeros((batch, action.action_dim), device=env.device)
    action_bias = torch.zeros_like(action_width)
    action_scale = torch.ones_like(action_width)
    action_profile_velocity = torch.zeros_like(action_width)
    action_profile_acceleration = torch.zeros_like(action_width)
    action_name_to_id = {name: i for i, name in enumerate(action.target_names)}
    for source, name in enumerate(_STS3215_JOINT_NAMES):
        target = action_name_to_id[name]
        action_width[:, target] = 0.5 * backlash_total[:, source]
        action_bias[:, target] = bias[:, source]
        action_scale[:, target] = calibration_scale[:, source]
        # Register limits live in the software/count coordinate. Convert them
        # through the same calibration scale before applying the physical
        # setpoint profile.
        action_profile_velocity[:, target] = (
            STS3215_MAX_PROFILE_SPEED_RAD_S
            * firmware_velocity[:, source]
            / calibration_scale[:, source]
        )
        action_profile_acceleration[:, target] = (
            STS3215_MAX_PROFILE_ACCELERATION_RAD_S2
            * firmware_acceleration[:, source]
            / calibration_scale[:, source]
        )
    action.set_backlash_half_width(action_width, selected)
    action.set_servo_calibration(action_bias, action_scale, selected)
    action.set_firmware_profile_limits(
        action_profile_velocity, action_profile_acceleration, selected
    )

    # The bus delay is common to all servos in one synchronized packet. The
    # action term owns one FIFO per robot and adds correlated packet jitter.
    if randomize_latency:
        if scale <= 0.0:
            lags = torch.zeros(batch, device=env.device, dtype=torch.long)
        else:
            lag_min = max(0, round(prior.latency_lags[0] * scale))
            lag_max = max(lag_min, round(prior.latency_lags[1] * scale))
            lags = torch.randint(
                lag_min, lag_max + 1, (batch,), device=env.device
            )
        action.set_command_delay(lags, scale, selected)

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
    robot.resolve(env.scene)

    # Shared pad/contact samples preserve bilateral material properties.
    # Collision-only mounting errors remain mirrored until independent
    # mismatch is enabled; the small global floor slope is active throughout.
    _randomize_foot_ground(env, env_ids, paired_geometry=True)
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
        "actuator_gainprm", "actuator_biasprm", "actuator_forcerange",
        "dof_damping", "dof_frictionloss", "dof_armature",
    )
    body_fields = ("body_mass", "body_ipos", "body_inertia", "body_iquat")
    body_ids = env.scene["robot"].indexing.body_ids
    symmetric_domain_randomization(env, env_ids)
    paired_fields = {name: getattr(model, name)[env_ids].clone() for name in field_names}
    paired_body = [getattr(model, name)[env_ids[:, None], body_ids].clone() for name in body_fields]
    encoder_bias = env.scene["robot"].data.encoder_bias
    paired_encoder_bias = encoder_bias[env_ids].clone()
    action = env.action_manager.get_term("joint_pos")
    paired_encoder_scale = action._entity.data.encoder_scale[env_ids].clone()
    paired_backlash = action._backlash_half_width[env_ids].clone()
    paired_profile_velocity = action._firmware_max_velocity[env_ids].clone()
    paired_profile_acceleration = action._firmware_max_acceleration[env_ids].clone()
    foot_ids = _global_geom_ids(
        env,
        SceneEntityCfg(
            "robot",
            geom_names=("right_foot_collision", "left_foot_collision"),
            preserve_order=True,
        ),
    )
    selected = torch.arange(env.num_envs, device=env.device)[env_ids].long()
    env_grid, foot_grid = torch.meshgrid(selected, foot_ids, indexing="ij")
    paired_foot_pos = model.geom_pos[env_grid, foot_grid].clone()
    paired_foot_quat = model.geom_quat[env_grid, foot_grid].clone()
    full_domain_randomization(env, env_ids, randomize_latency=False)
    full_foot_pos = model.geom_pos[env_grid, foot_grid].clone()
    full_foot_quat = model.geom_quat[env_grid, foot_grid].clone()
    for name in field_names:
        values = getattr(model, name)
        full_values = values[env_ids]
        blended = paired_fields[name] + amplitude * (
            full_values - paired_fields[name]
        )
        values[env_ids] = blended
    model.geom_pos[env_grid, foot_grid] = paired_foot_pos + amplitude * (
        full_foot_pos - paired_foot_pos
    )
    # Normalized shortest-path interpolation is stable for this small (1 deg)
    # mismatch range and keeps every per-world geom quaternion valid.
    same_hemisphere = torch.sum(paired_foot_quat * full_foot_quat, dim=-1) >= 0.0
    full_foot_quat = torch.where(
        same_hemisphere[..., None], full_foot_quat, -full_foot_quat
    )
    blended_foot_quat = paired_foot_quat + amplitude * (
        full_foot_quat - paired_foot_quat
    )
    model.geom_quat[env_grid, foot_grid] = torch.nn.functional.normalize(
        blended_foot_quat, dim=-1
    )
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
    encoder_scale = action._entity.data.encoder_scale
    full_encoder_scale = encoder_scale[env_ids].clone()
    encoder_scale[env_ids] = paired_encoder_scale + amplitude * (
        full_encoder_scale - paired_encoder_scale
    )
    full_backlash = action._backlash_half_width[env_ids]
    action._backlash_half_width[env_ids] = paired_backlash + amplitude * (
        full_backlash - paired_backlash
    )
    full_profile_velocity = action._firmware_max_velocity[env_ids].clone()
    action._firmware_max_velocity[env_ids] = paired_profile_velocity + amplitude * (
        full_profile_velocity - paired_profile_velocity
    )
    full_profile_acceleration = action._firmware_max_acceleration[env_ids].clone()
    action._firmware_max_acceleration[env_ids] = (
        paired_profile_acceleration
        + amplitude * (full_profile_acceleration - paired_profile_acceleration)
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
    realizations. Packet latency stays fully randomized because one synchronized
    command packet supplies all six servos and has no left/right mismatch.
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

def sim2real_randomization_scale(env) -> float:
    """Return the task's explicit simulation-randomization strength.

    Common tasks do not inherit the deleted all-in-one gait curriculum. They
    use full reset and observation randomization by default; a task may opt
    into a reduced strength by setting ``cfg.domain_randomization_scale``.
    """
    if not getattr(env.cfg, "sim2real_randomization_enabled", True):
        return 0.0
    return float(getattr(env.cfg, "domain_randomization_scale", 1.0))

def sim2real_domain_randomization(
    env,
    env_ids: torch.Tensor,
    randomize_latency: bool = True,
) -> None:
    """Apply a task-configured reset randomization envelope."""
    scale = sim2real_randomization_scale(env)
    robot = SceneEntityCfg("robot")
    robot.resolve(env.scene)

    _randomize_foot_ground(env, env_ids, paired_geometry=False)
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

def _calibrated_joint_position(robot) -> torch.Tensor:
    """Map physical joint angle into the controller's software coordinate."""
    bias = getattr(robot.data, "encoder_bias", None)
    if bias is None:
        # Lightweight unit-test and third-party entities may not expose mjlab's
        # optional encoder fields.
        bias = torch.zeros_like(robot.data.joint_pos)
    scale = getattr(robot.data, "encoder_scale", None)
    if scale is None:
        scale = torch.ones_like(bias)
    return robot.data.joint_pos * scale + bias

def _simulated_imu(env) -> SimulatedIcm42688:
    estimator = getattr(env, "_simulated_icm42688", None)
    if estimator is None:
        estimator = SimulatedIcm42688(env.num_envs, env.device)
        env._simulated_icm42688 = estimator
    return estimator

def _simulated_imu_outputs(
    env, gyro_sensor_name: str, accel_sensor_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    estimator = _simulated_imu(env)
    return estimator.update(
        env_mdp.builtin_sensor(env, gyro_sensor_name),
        env_mdp.builtin_sensor(env, accel_sensor_name),
        env.episode_length_buf,
        sim2real_randomization_scale(env),
    )

class Sim2RealImuAngularVelocity(ManagerTermBase):
    def __init__(self, cfg, env) -> None:
        del cfg
        super().__init__(env)
        self._imu = _simulated_imu(env)

    def reset(self, env_ids=None) -> None:
        self._imu.reset(env_ids)

    def __call__(self, env, gyro_sensor_name: str, accel_sensor_name: str):
        return _simulated_imu_outputs(
            env, gyro_sensor_name, accel_sensor_name
        )[0]

class Sim2RealImuProjectedGravity(ManagerTermBase):
    def __init__(self, cfg, env) -> None:
        del cfg
        super().__init__(env)
        self._imu = _simulated_imu(env)

    def reset(self, env_ids=None) -> None:
        self._imu.reset(env_ids)

    def __call__(self, env, gyro_sensor_name: str, accel_sensor_name: str):
        return _simulated_imu_outputs(
            env, gyro_sensor_name, accel_sensor_name
        )[1]

class _ServoObservationAge:
    """Keep position and derived velocity on one randomized servo-bus age."""

    def __init__(self, env) -> None:
        shape = env.scene["robot"].data.joint_pos.shape
        self.device = env.device
        self.previous_position = torch.zeros(shape, device=self.device)
        self.previous_velocity = torch.zeros(shape, device=self.device)
        self.output_position = torch.zeros(shape, device=self.device)
        self.output_velocity = torch.zeros(shape, device=self.device)
        self.current_position = torch.zeros(shape, device=self.device)
        self.lag = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        self.last_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.pending_step = torch.full_like(self.last_step, -1)
        self.pending_fresh = torch.zeros_like(self.lag)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.last_step[ids] = -1
        self.pending_step[ids] = -1
        self.previous_velocity[ids] = 0.0
        self.output_velocity[ids] = 0.0

    def begin(
        self, position: torch.Tensor, step: torch.Tensor, strength: float
    ) -> torch.Tensor:
        update = step != self.last_step
        already_pending = step == self.pending_step
        update &= ~already_pending
        fresh = update & ((self.last_step < 0) | (step < self.last_step))
        continuing = update & ~fresh
        ids = torch.where(update)[0]
        if ids.numel():
            if torch.any(fresh):
                fresh_ids = torch.where(fresh)[0]
                self.lag[fresh_ids] = (
                    torch.rand(fresh_ids.numel(), device=self.device)
                    < 0.5 * float(strength)
                )
            if torch.any(continuing):
                continuing_ids = torch.where(continuing)[0]
                change = torch.rand(
                    continuing_ids.numel(), device=self.device
                ) > 0.9
                if torch.any(change):
                    changed_ids = continuing_ids[change]
                    self.lag[changed_ids] = (
                        torch.rand(changed_ids.numel(), device=self.device)
                        < 0.5 * float(strength)
                    )
            use_previous = self.lag[ids] & ~fresh[ids]
            self.current_position[ids] = position[ids]
            self.output_position[ids] = torch.where(
                use_previous[:, None], self.previous_position[ids], position[ids]
            )
            self.pending_step[ids] = step[ids]
            self.pending_fresh[ids] = fresh[ids]
        return self.output_position

    def finish(self, velocity: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        update = (self.pending_step == step) & (self.last_step != step)
        ids = torch.where(update)[0]
        if ids.numel():
            use_previous = self.lag[ids] & ~self.pending_fresh[ids]
            self.output_velocity[ids] = torch.where(
                use_previous[:, None], self.previous_velocity[ids], velocity[ids]
            )
            self.previous_position[ids] = self.current_position[ids]
            self.previous_velocity[ids] = velocity[ids]
            self.last_step[ids] = step[ids]
        return self.output_velocity

def _servo_observation_age(env) -> _ServoObservationAge:
    age = getattr(env, "_servo_observation_age", None)
    if age is None:
        age = _ServoObservationAge(env)
        env._servo_observation_age = age
    return age

class Sim2RealJointPosition(ManagerTermBase):
    def __init__(self, cfg, env) -> None:
        del cfg
        super().__init__(env)
        self._age = _servo_observation_age(env)

    def reset(self, env_ids=None) -> None:
        self._age.reset(env_ids)

    def __call__(
        self,
        env,
        maximum_noise: float = STS3215_RANDOMIZATION.encoder_noise_rad,
    ) -> torch.Tensor:
        robot = env.scene["robot"]
        measured = _scaled_noise(
            env, _calibrated_joint_position(robot), maximum_noise
        )
        observed = quantize_position_torch(
            measured, STS3215_RANDOMIZATION.position_resolution_rad
        )
        env._sim2real_joint_position_sample = observed
        env._sim2real_joint_position_sample_step = env.episode_length_buf.clone()
        delayed = self._age.begin(
            observed,
            env.episode_length_buf,
            sim2real_randomization_scale(env),
        )
        return delayed - robot.data.default_joint_pos

class Sim2RealJointVelocity(ManagerTermBase):
    """Estimate velocity from noisy, quantized 50 Hz position observations.

    This replaces access to MuJoCo's exact generalized velocity with the same
    causal backward difference and low-pass filter used by deployment.  The
    cached output also makes the term safe if an observation group is computed
    more than once during one environment step.
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(env)
        self._filter_alpha = float(
            cfg.params.get("filter_alpha", STS3215_VELOCITY_FILTER_ALPHA)
        )
        if not 0.0 < self._filter_alpha <= 1.0:
            raise ValueError("joint velocity filter alpha must be in (0, 1]")
        shape = (self.num_envs, env.scene["robot"].data.joint_pos.shape[1])
        self._previous_position = torch.zeros(shape, device=self.device)
        self._filtered_velocity = torch.zeros_like(self._previous_position)
        self._cached = torch.zeros_like(self._previous_position)
        self._last_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._age = _servo_observation_age(env)

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._last_step[env_ids] = -1
        self._previous_position[env_ids] = 0.0
        self._filtered_velocity[env_ids] = 0.0
        self._cached[env_ids] = 0.0
        self._age.reset(env_ids)

    def __call__(
        self,
        env,
        maximum_noise: float = STS3215_RANDOMIZATION.encoder_noise_rad,
        filter_alpha: float = STS3215_VELOCITY_FILTER_ALPHA,
    ) -> torch.Tensor:
        # Manager parameters are also supplied to __call__. Reject a mismatch
        # rather than silently using different filters during initialization
        # and evaluation.
        if filter_alpha != self._filter_alpha:
            raise ValueError("joint velocity filter alpha changed after initialization")
        robot = env.scene["robot"]
        step = env.episode_length_buf
        sampled = getattr(env, "_sim2real_joint_position_sample", None)
        sampled_step = getattr(env, "_sim2real_joint_position_sample_step", None)
        if (
            sampled is not None
            and sampled_step is not None
            and torch.equal(sampled_step, step)
        ):
            observed = sampled
        else:
            # This fallback keeps the term usable in isolation. During normal
            # actor observation construction, joint_pos precedes joint_vel and
            # supplies the shared sample above.
            measured = _scaled_noise(
                env, _calibrated_joint_position(robot), maximum_noise
            )
            observed = quantize_position_torch(
                measured, STS3215_RANDOMIZATION.position_resolution_rad
            )
        # Normally the joint-position term has already opened this coherent
        # device sample. The call is idempotent and also supports isolated
        # velocity-term tests or custom observation ordering.
        self._age.begin(
            observed, step, sim2real_randomization_scale(env)
        )
        update = step != self._last_step
        fresh = update & ((self._last_step < 0) | (step < self._last_step))
        continuing = update & ~fresh

        if torch.any(fresh):
            self._previous_position[fresh] = observed[fresh]
            self._filtered_velocity[fresh] = 0.0
            self._cached[fresh] = 0.0
        if torch.any(continuing):
            raw = (
                observed[continuing] - self._previous_position[continuing]
            ) / env.step_dt
            self._filtered_velocity[continuing] += self._filter_alpha * (
                raw - self._filtered_velocity[continuing]
            )
            self._previous_position[continuing] = observed[continuing]
            self._cached[continuing] = self._filtered_velocity[continuing]
        self._last_step[update] = step[update]
        return self._age.finish(self._cached, step)
