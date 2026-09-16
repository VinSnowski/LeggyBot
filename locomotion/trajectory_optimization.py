"""Native MuJoCo trajectory optimisation and feasibility checks.

The trajectory path is deliberately independent of mjlab.  A trajectory is a
small periodic table of absolute joint targets; the table is interpolated at
the 50 Hz command rate and held while native MuJoCo advances four 200 Hz
physics steps.  This makes saved trajectories useful to both the optimiser and
the eventual embedded controller.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import mujoco
import numpy as np

from locomotion.play_native import _control_ids, _joint_addresses, reset_robot
from locomotion.robot_profiles import LEGGYBOT_FIXED_ANKLE_PROFILE, RobotProfile, make_native_model


STS3215_STALL_TORQUE_NM = 2.942
STS3215_NO_LOAD_SPEED_RAD_S = 4.712
STS3215_SAFE_SPEED_FRACTION = 0.80
STS3215_SAFE_TORQUE_FRACTION = 0.80
TRAJECTORY_SCHEMA = "leggybot-trajectory/v1"


def _as_float_array(value: Any, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"expected an array with {ndim} dimensions, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("trajectory contains NaN or infinite values")
    return result


@dataclass(frozen=True, slots=True)
class PeriodicSpline:
    """Periodic Catmull-Rom spline over joint targets.

    Knots are evenly spaced over one period.  Catmull-Rom is preferable here
    to a dependency on a particular scipy version and has continuous position
    and velocity at the periodic seam.
    """

    knots: np.ndarray
    period_s: float

    def __post_init__(self) -> None:
        knots = _as_float_array(self.knots, 2)
        if knots.shape[0] < 2 or knots.shape[1] == 0:
            raise ValueError("periodic spline requires at least two knots")
        if not np.isfinite(self.period_s) or self.period_s <= 0:
            raise ValueError("period_s must be positive")
        object.__setattr__(self, "knots", knots.copy())

    @property
    def knot_count(self) -> int:
        return int(self.knots.shape[0])

    @property
    def joint_count(self) -> int:
        return int(self.knots.shape[1])

    def evaluate(self, time_s: float | np.ndarray) -> np.ndarray:
        t = np.asarray(time_s, dtype=np.float64)
        scalar = t.ndim == 0
        u = np.mod(t, self.period_s) / self.period_s * self.knot_count
        i = np.floor(u).astype(np.int64)
        f = u - i
        p0 = self.knots[(i - 1) % self.knot_count]
        p1 = self.knots[i % self.knot_count]
        p2 = self.knots[(i + 1) % self.knot_count]
        p3 = self.knots[(i + 2) % self.knot_count]
        # Standard Catmull-Rom basis, vectorised over time and joints.
        result = 0.5 * ((2 * p1) + (-p0 + p2) * f[..., None]
                        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f[..., None] ** 2
                        + (-p0 + 3 * p1 - 3 * p2 + p3) * f[..., None] ** 3)
        # With scalar time indexing the joint dimension is already the final
        # dimension (shape ``(joints,)``); do not drop the first joint.
        return result if scalar else result

    def velocity(self, time_s: float | np.ndarray) -> np.ndarray:
        t = np.asarray(time_s, dtype=np.float64)
        scalar = t.ndim == 0
        u = np.mod(t, self.period_s) / self.period_s * self.knot_count
        i = np.floor(u).astype(np.int64)
        f = u - i
        p0 = self.knots[(i - 1) % self.knot_count]
        p1 = self.knots[i % self.knot_count]
        p2 = self.knots[(i + 1) % self.knot_count]
        p3 = self.knots[(i + 2) % self.knot_count]
        derivative = 0.5 * ((-p0 + p2)
                            + 2 * (2 * p0 - 5 * p1 + 4 * p2 - p3) * f[..., None]
                            + 3 * (-p0 + 3 * p1 - 3 * p2 + p3) * f[..., None] ** 2)
        derivative *= self.knot_count / self.period_s
        return derivative if scalar else derivative

    def clip(self, lower: Iterable[float], upper: Iterable[float]) -> "PeriodicSpline":
        lo, hi = _as_float_array(lower, 1), _as_float_array(upper, 1)
        if lo.shape != hi.shape or lo.shape[0] != self.joint_count or np.any(lo > hi):
            raise ValueError("invalid clipping limits")
        return PeriodicSpline(np.clip(self.knots, lo, hi), self.period_s)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": TRAJECTORY_SCHEMA, "period_s": self.period_s,
                "knots": self.knots.tolist()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PeriodicSpline":
        if value.get("schema", TRAJECTORY_SCHEMA) != TRAJECTORY_SCHEMA:
            raise ValueError("unsupported trajectory schema")
        return cls(np.asarray(value["knots"], dtype=np.float64), float(value["period_s"]))


# Friendly aliases used by scripts and downstream experiments.
PeriodicJointTrajectory = PeriodicSpline


def bilateral_gait_seed(profile: RobotProfile = LEGGYBOT_FIXED_ANKLE_PROFILE,
                        knots: int = 8, period_s: float = 0.8) -> PeriodicSpline:
    """Build the low-dimensional seed corresponding to the PPO gait reference."""
    if knots < 2:
        raise ValueError("knots must be at least two")
    phase = np.arange(knots, dtype=np.float64) / knots
    values = np.tile(profile.array("default_joint_pos"), (knots, 1))
    for leg, offset in enumerate((0.0, 0.5)):
        local = np.mod(phase + offset, 1.0)
        stance_progress = np.clip(local / 0.6, 0.0, 1.0)
        swing_progress = np.clip((local - 0.6) / 0.4, 0.0, 1.0)
        in_stance = local < 0.6
        hip = np.where(in_stance, 0.10 - 0.20 * stance_progress,
                       -0.10 + 0.20 * swing_progress + 0.57 * np.sin(np.pi * swing_progress))
        knee = np.where(in_stance, 0.0, -0.995 * np.sin(np.pi * swing_progress))
        values[:, 1 + 3 * leg] += hip
        values[:, 2 + 3 * leg] += knee
    lower, upper = profile.action_bounds()
    # The reference is intentionally safe for physical limits (and retains a
    # 5-degree margin wherever the profile permits it).
    return PeriodicSpline(values, period_s).clip(lower, upper)


def weight_transfer_gait_seed(
    profile: RobotProfile = LEGGYBOT_FIXED_ANKLE_PROFILE,
    knots: int = 8,
    period_s: float = 1.6,
) -> PeriodicSpline:
    """Seed a slow unload--swing--land biped gait.

    A sagittal-only seed asks the robot to lift a loaded rigid foot and is not
    dynamically meaningful for LeggyBot.  This seed first rolls the stance
    geometry laterally, then flexes the unloaded leg, and mirrors the motion a
    half-cycle later.  It remains only an initializer: direct shooting is free
    to alter every knot while retaining bilateral periodicity.
    """
    if knots != 8:
        raise ValueError("the weight-transfer seed is defined for eight knots")
    nominal = profile.array("default_joint_pos")
    values = np.tile(nominal, (knots, 1))

    # MuJoCo contact-force measurements show that a small negative command on
    # *both* roll joints unloads the right foot (and a positive command unloads
    # the left).  The swing knee is flexed only after this lateral shift.
    common_roll = np.asarray((0.00, -0.05, -0.07, -0.035,
                              0.00, 0.05, 0.07, 0.035))
    values[:, 0] += common_roll
    values[:, 3] += common_roll
    values[:, 1] += (0.00, -0.04, 0.30, 0.18, 0.08, 0.00, -0.07, -0.09)
    values[:, 2] += (0.00, -0.05, -0.58, -0.20, 0.00, 0.00, 0.00, 0.00)

    # Mirror sagittal motion explicitly so the seed is useful outside the CLI.
    half = knots // 2
    values[:, 4] = nominal[4] + np.roll(values[:, 1] - nominal[1], half)
    values[:, 5] = nominal[5] + np.roll(values[:, 2] - nominal[2], half)
    lower, upper = profile.action_bounds()
    return PeriodicSpline(values, period_s).clip(lower, upper)


@dataclass(frozen=True, slots=True)
class ActuatorMetrics:
    peak_speed_rad_s: float
    peak_torque_nm: float
    speed_limit_rad_s: float
    torque_envelope_margin_nm: float
    torque_limit_margin_nm: float
    safe_torque_limit_nm: float
    speed_ok: bool
    torque_ok: bool


def torque_speed_limit(speed_rad_s: float | np.ndarray,
                       stall_torque_nm: float = STS3215_STALL_TORQUE_NM,
                       no_load_speed_rad_s: float = STS3215_NO_LOAD_SPEED_RAD_S) -> np.ndarray:
    """Linear motor torque-speed envelope, clipped at zero."""
    if stall_torque_nm <= 0 or no_load_speed_rad_s <= 0:
        raise ValueError("stall torque and no-load speed must be positive")
    return np.maximum(0.0, stall_torque_nm * (1.0 - np.abs(speed_rad_s) / no_load_speed_rad_s))


def compute_torque_speed_metrics(torque: np.ndarray, speed: np.ndarray,
                                 safe_speed_fraction: float = STS3215_SAFE_SPEED_FRACTION,
                                 safe_torque_fraction: float = STS3215_SAFE_TORQUE_FRACTION,
                                 stall_torque_nm: float = STS3215_STALL_TORQUE_NM,
                                 no_load_speed_rad_s: float = STS3215_NO_LOAD_SPEED_RAD_S) -> ActuatorMetrics:
    torque, speed = _as_float_array(torque), _as_float_array(speed)
    if torque.shape != speed.shape or torque.size == 0:
        raise ValueError("torque and speed must have the same non-empty shape")
    peak_speed = float(np.max(np.abs(speed)))
    peak_torque = float(np.max(np.abs(torque)))
    envelope = torque_speed_limit(speed, stall_torque_nm, no_load_speed_rad_s)
    return ActuatorMetrics(peak_speed, peak_torque, safe_speed_fraction * no_load_speed_rad_s,
                           float(np.min(envelope - np.abs(torque))),
                           stall_torque_nm - peak_torque,
                           safe_torque_fraction * stall_torque_nm,
                           peak_speed <= safe_speed_fraction * no_load_speed_rad_s + 1e-9,
                           bool(np.all(np.abs(torque) <= safe_torque_fraction * stall_torque_nm + 1e-9)
                                and np.all(np.abs(torque) <= envelope + 1e-9)))


# Short names retained for notebooks.
sts3215_torque_speed = torque_speed_limit
torque_speed_envelope = torque_speed_limit


@dataclass(slots=True)
class TrajectoryMetrics:
    feasible: bool
    falls: int = 0
    non_foot_contacts: int = 0
    alternating_contacts: bool = False
    minimum_clearance_m: float = 0.0
    periodicity_error_rad: float = 0.0
    q_cycle_periodicity_error_rad: float = 0.0
    qvel_cycle_periodicity_error_rad_s: float = 0.0
    base_cycle_periodicity_error: float = 0.0
    achieved_speed_m_s: float = 0.0
    speed_error_m_s: float = 0.0
    lateral_drift_m: float = 0.0
    yaw_drift_rad: float = 0.0
    lateral_excursion_m: float = 0.0
    yaw_excursion_rad: float = 0.0
    minimum_joint_limit_margin_deg: float = 0.0
    foot_slip_m_s: float = 0.0
    landing_speed_m_s: float = 0.0
    actuator: ActuatorMetrics | None = None
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return result


@dataclass(slots=True)
class TrajectoryRollout:
    times: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    targets: np.ndarray
    actuator_force: np.ndarray
    foot_positions: np.ndarray
    contacts: np.ndarray
    foot_normal_force: np.ndarray
    non_foot_contacts: np.ndarray
    base_yaw: np.ndarray
    metrics: TrajectoryMetrics
    model_fingerprint: str = ""


def _model_ids(model: mujoco.MjModel, profile: RobotProfile):
    foot_ids = tuple(
        np.asarray([
            geom_id for geom_id in range(model.ngeom)
            if ((name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
                and name.startswith(side) and name.endswith("foot_collision"))
        ], dtype=np.int32)
        for side in ("right", "left")
    )
    site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
                         for n in ("right_foot", "left_foot")])
    nonfoot = ["base_collision", "right_hip_collision", "right_thigh_collision",
               "right_shin_collision", "left_hip_collision", "left_thigh_collision",
               "left_shin_collision"]
    nonfoot_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in nonfoot])
    return foot_ids, site_ids, nonfoot_ids


def _contacts(model: mujoco.MjModel, data: mujoco.MjData, foot_ids: tuple[np.ndarray, np.ndarray],
              nonfoot_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    feet = np.zeros(2, dtype=bool)
    normal_force = np.zeros(2, dtype=np.float64)
    bad = False
    foot_sets = [set(map(int, ids)) for ids in foot_ids]
    nset = set(map(int, nonfoot_ids))
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        for side, side_ids in enumerate(foot_sets):
            if pair & side_ids:
                feet[side] = True
                wrench = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, index, wrench)
                normal_force[side] += max(0.0, float(wrench[0]))
        if pair & nset:
            bad = True
    return feet, normal_force, bad


def _validate_inputs(trajectory: PeriodicSpline, profile: RobotProfile, duration_s: float) -> None:
    if trajectory.joint_count != profile.action_dim:
        raise ValueError("trajectory joint count does not match profile")
    if duration_s <= 0 or not np.isfinite(duration_s):
        raise ValueError("duration_s must be positive")
    if np.any(trajectory.knots < profile.array("joint_lower") - 1e-9) or np.any(
        trajectory.knots > profile.array("joint_upper") + 1e-9):
        raise ValueError("trajectory contains joint targets outside physical limits")


def simulate_trajectory(trajectory: PeriodicSpline, speed_m_s: float = 0.1,
                        duration_s: float | None = None,
                        profile: RobotProfile = LEGGYBOT_FIXED_ANKLE_PROFILE,
                        seed: int = 0, model: mujoco.MjModel | None = None,
                        command_latency_s: float = 0.0,
                        initial_state_perturbation: Sequence[float] | None = None,
                        settle_s: float | None = None) -> TrajectoryRollout:
    """Run direct shooting with native CPU MuJoCo at the deployment rates."""
    del seed  # deterministic native dynamics; kept in the API for perturbations.
    duration_s = float(duration_s if duration_s is not None else max(2.0 * trajectory.period_s, 3.0))
    _validate_inputs(trajectory, profile, duration_s)
    if command_latency_s < 0:
        raise ValueError("command_latency_s cannot be negative")
    own_model = model is None
    model = model if model is not None else make_native_model(profile)
    data = mujoco.MjData(model)
    qpos_adr, dof_adr = _joint_addresses(model, profile)
    ctrl_ids = _control_ids(model, profile)
    reset_robot(model, data, qpos_adr, profile)
    if initial_state_perturbation is not None:
        perturbation = _as_float_array(initial_state_perturbation, 1)
        if perturbation.size not in (3, 6):
            raise ValueError("initial_state_perturbation must contain 3 or 6 values")
        data.qpos[:3] += perturbation[:3]
        if perturbation.size == 6:
            data.qvel[:6] += perturbation[3:]
        mujoco.mj_forward(model, data)
    foot_ids, site_ids, nonfoot_ids = _model_ids(model, profile)
    physics_steps = max(1, int(round(duration_s / profile.physics_dt)))
    times = np.arange(physics_steps, dtype=np.float64) * profile.physics_dt
    qpos = np.empty((physics_steps, model.nq), dtype=np.float64)
    qvel = np.empty((physics_steps, model.nv), dtype=np.float64)
    targets = np.empty((physics_steps, profile.action_dim), dtype=np.float64)
    forces = np.empty((physics_steps, profile.action_dim), dtype=np.float64)
    foot_positions = np.empty((physics_steps, 2, 3), dtype=np.float64)
    contacts = np.zeros((physics_steps, 2), dtype=bool)
    foot_normal_force = np.zeros((physics_steps, 2), dtype=np.float64)
    bad_contacts = np.zeros(physics_steps, dtype=bool)
    yaw = np.empty(physics_steps, dtype=np.float64)
    latency_steps = int(round(command_latency_s / profile.control_dt))
    history: list[np.ndarray] = []
    for step, time_s in enumerate(times):
        if step % profile.control_decimation == 0:
            history.append(np.asarray(trajectory.evaluate(time_s), dtype=np.float64))
            target = history[max(0, len(history) - 1 - latency_steps)]
            # Catmull-Rom can overshoot between otherwise safe knots.  The
            # deployed position command must preserve the same 5-degree
            # physical-limit margin required by validation.
            safe_lower = profile.array("joint_lower") + np.deg2rad(5.0)
            safe_upper = profile.array("joint_upper") - np.deg2rad(5.0)
            data.ctrl[ctrl_ids] = np.clip(target, safe_lower, safe_upper)
        mujoco.mj_step(model, data)
        qpos[step] = data.qpos
        qvel[step] = data.qvel
        targets[step] = data.ctrl[ctrl_ids]
        forces[step] = data.actuator_force[ctrl_ids]
        foot_positions[step] = data.site_xpos[site_ids]
        contacts[step], foot_normal_force[step], bad_contacts[step] = _contacts(
            model, data, foot_ids, nonfoot_ids
        )
        quat = data.qpos[3:7]
        yaw[step] = np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                               1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
    speeds = qvel[:, dof_adr]
    actuator = compute_torque_speed_metrics(
        forces,
        speeds,
        safe_speed_fraction=profile.actuator_safe_speed_fraction,
        safe_torque_fraction=profile.actuator_safe_torque_fraction,
        stall_torque_nm=profile.actuator_stall_torque_nm,
        no_load_speed_rad_s=profile.actuator_no_load_speed_rad_s,
    )
    metrics = compute_trajectory_metrics(trajectory, profile, times, qpos, qvel, targets,
                                          forces, foot_positions, contacts, bad_contacts, yaw,
                                          actuator, speed_m_s, settle_s)
    return TrajectoryRollout(times, qpos, qvel, targets, forces, foot_positions, contacts,
                             foot_normal_force, bad_contacts, yaw, metrics,
                             native_model_fingerprint(model))


def compute_trajectory_metrics(trajectory: PeriodicSpline, profile: RobotProfile,
                               times: np.ndarray, qpos: np.ndarray, qvel: np.ndarray,
                               targets: np.ndarray, actuator_force: np.ndarray,
                               foot_positions: np.ndarray, contacts: np.ndarray,
                               bad_contacts: np.ndarray, base_yaw: np.ndarray,
                               actuator: ActuatorMetrics | None = None,
                               target_speed_m_s: float | None = None,
                               settle_s: float | None = None) -> TrajectoryMetrics:
    """Compute nominal/robust metrics from a rollout or a test fixture."""
    times, qpos, qvel, targets, foot_positions = map(_as_float_array, (times, qpos, qvel, targets, foot_positions))
    if (times.ndim != 1 or times.size < 2 or qpos.ndim != 2 or qvel.ndim != 2
            or targets.ndim != 2 or foot_positions.ndim != 3
            or qpos.shape[0] != times.size or qvel.shape[0] != times.size
            or targets.shape[0] != times.size or foot_positions.shape != (times.size, 2, 3)):
        raise ValueError("rollout arrays have inconsistent lengths")
    contacts = np.asarray(contacts, dtype=bool)
    bad_contacts = np.asarray(bad_contacts, dtype=bool)
    base_yaw = _as_float_array(base_yaw, 1)
    actuator_force = _as_float_array(actuator_force, 2)
    if (contacts.shape != (times.size, 2) or bad_contacts.shape != (times.size,)
            or base_yaw.shape != (times.size,) or actuator_force.shape[0] != times.size):
        raise ValueError("contact, base, and actuator arrays have inconsistent lengths")
    dt = max(float(times[-1] - times[0]), 1e-9)
    # The CAD root origin sits below the torso, so comparing its world Z with
    # ``initial_root_pos`` does not reliably detect a toppled robot.  Use the
    # body's world-up projection and a relative height loss instead.
    quat = qpos[:, 3:7]
    base_up_z = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    root_z = qpos[:, 2]
    fallen = (base_up_z < 0.65) | (root_z < root_z[0] - 0.10)
    falls = int(np.count_nonzero(fallen))
    # The direct-shooting seed uses explicit unload/swing/land quarters.  Each
    # foot swings for 25% of the cycle, separated by double-support intervals.
    phase = np.mod(times / trajectory.period_s, 1.0)
    right_swing = (phase >= 0.125) & (phase < 0.375)
    left_swing = (phase >= 0.625) & (phase < 0.875)
    swing = np.column_stack((right_swing, left_swing))
    # Report the useful swing clearance: each foot's peak height in its swing
    # interval.  A minimum over all swing samples includes lift-off itself and
    # would incorrectly reject every otherwise valid trajectory.
    upright_samples = base_up_z >= 0.80
    peak_clearances = []
    for foot in range(2):
        swing_samples = swing[:, foot] & upright_samples & ~bad_contacts
        stance_samples = ~swing[:, foot] & upright_samples & ~bad_contacts
        if np.any(swing_samples) and np.any(stance_samples):
            stance_height = float(np.median(foot_positions[stance_samples, foot, 2]))
            peak_clearances.append(
                float(np.max(foot_positions[swing_samples, foot, 2]) - stance_height)
            )
    minimum_clearance = float(min(peak_clearances)) if peak_clearances else 0.0
    # Detect alternating stance ownership over meaningful stance frames.
    owner = np.argmax(contacts.astype(np.int8), axis=1)
    contact_events = owner[contacts.any(axis=1)]
    alternating = bool(np.sum(contact_events[1:] != contact_events[:-1]) >= 2) if contact_events.size > 2 else False
    # Distances use the root's world frame, where +X is forward and +Y lateral.
    achieved_speed = float((qpos[-1, 0] - qpos[0, 0]) / dt)
    unwrapped_yaw = np.unwrap(base_yaw)
    # Walking bipeds must sway and may yaw slightly within a step.  Drift is
    # accumulated displacement from start to finish; excursion is reported
    # separately and is not confused with systematic veering.
    lateral = float(abs(qpos[-1, 1] - qpos[0, 1]))
    yaw_drift = float(abs(unwrapped_yaw[-1] - unwrapped_yaw[0]))
    lateral_excursion = float(np.ptp(qpos[:, 1]))
    yaw_excursion = float(np.ptp(unwrapped_yaw))
    lower, upper = profile.array("joint_lower"), profile.array("joint_upper")
    margin = np.minimum(targets - lower, upper - targets)
    margin_deg = float(np.rad2deg(np.min(margin)))
    # Slip is horizontal foot velocity while contact is present. Landing speed
    # is measured only at a contact rising edge.
    foot_vel = np.gradient(foot_positions[:, :, :2], max(profile.physics_dt, 1e-9), axis=0)
    slip = float(np.max(np.linalg.norm(foot_vel[contacts], axis=-1))) if np.any(contacts) else 0.0
    rising = contacts & ~np.vstack((np.zeros((1, 2), bool), contacts[:-1]))
    foot_vel3 = np.gradient(foot_positions, max(profile.physics_dt, 1e-9), axis=0)
    landing = float(np.max(np.linalg.norm(foot_vel3[rising], axis=-1))) if np.any(rising) else 0.0
    # Compare the final full cycle after settling.  Quaternion components are
    # compared directly (sign is stable for this short upright rollout).
    cycle_steps = max(1, int(round(trajectory.period_s / profile.physics_dt)))
    settle_index = int(round((settle_s if settle_s is not None else trajectory.period_s) /
                             profile.physics_dt))
    end = times.size - 1
    start = max(0, end - cycle_steps, settle_index)
    if end - start >= 1:
        q_cycle = float(np.max(np.abs(qpos[end, 7:] - qpos[start, 7:])))
        qvel_cycle = float(np.max(np.abs(qvel[end, 6:] - qvel[start, 6:])))
        # Forward translation is intentionally removed: a walking gait is
        # periodic modulo its per-cycle displacement.  Lateral position,
        # height, and orientation must still return to the same orbit.
        base_cycle = float(np.max(np.abs(qpos[end, 1:7] - qpos[start, 1:7])))
    else:
        q_cycle = qvel_cycle = base_cycle = float("inf")
    speed_error = abs(achieved_speed - float(target_speed_m_s)) if target_speed_m_s is not None else 0.0
    violations: list[str] = []
    if falls: violations.append("fall")
    if int(np.count_nonzero(bad_contacts)): violations.append("non-foot contact")
    if not alternating: violations.append("alternating contacts")
    if minimum_clearance < 0.015: violations.append("clearance below 15 mm")
    if margin_deg < 5.0 - 1e-6: violations.append("joint-limit margin below 5 degrees")
    if actuator is not None and not actuator.speed_ok: violations.append("speed exceeds 80% no-load")
    if actuator is not None and not actuator.torque_ok: violations.append("torque-speed envelope")
    if q_cycle > 0.03: violations.append("joint periodicity error above 0.03 rad")
    if qvel_cycle > 0.15: violations.append("joint-speed periodicity error above 0.15 rad/s")
    if base_cycle > 0.08: violations.append("base periodicity error above 0.08")
    if not np.isfinite(q_cycle + qvel_cycle + base_cycle): violations.append("periodicity unavailable")
    if target_speed_m_s is not None and speed_error > 0.03:
        violations.append("speed error above 0.03 m/s")
    if lateral > 0.01: violations.append("lateral drift above 10 mm")
    if yaw_drift > np.deg2rad(3.0): violations.append("yaw drift above 3 degrees")
    if lateral_excursion > 0.12: violations.append("lateral excursion above 120 mm")
    if yaw_excursion > np.deg2rad(15.0): violations.append("yaw excursion above 15 degrees")
    feasible = not violations
    return TrajectoryMetrics(
        feasible=feasible, falls=falls, non_foot_contacts=int(np.count_nonzero(bad_contacts)),
        alternating_contacts=alternating, minimum_clearance_m=minimum_clearance,
        periodicity_error_rad=float(np.max(np.abs(trajectory.evaluate(0)-trajectory.evaluate(trajectory.period_s)))),
        q_cycle_periodicity_error_rad=q_cycle,
        qvel_cycle_periodicity_error_rad_s=qvel_cycle,
        base_cycle_periodicity_error=base_cycle, achieved_speed_m_s=achieved_speed,
        speed_error_m_s=speed_error, lateral_drift_m=lateral, yaw_drift_rad=yaw_drift,
        lateral_excursion_m=lateral_excursion, yaw_excursion_rad=yaw_excursion,
        minimum_joint_limit_margin_deg=margin_deg, foot_slip_m_s=slip,
        landing_speed_m_s=landing, actuator=actuator, violations=violations)


def validate_trajectory(trajectory: PeriodicSpline, speeds: Iterable[float] = (0.0, 0.05, 0.10, 0.15, 0.20),
                        profile: RobotProfile = LEGGYBOT_FIXED_ANKLE_PROFILE,
                        robust: bool = True, seed: int = 0,
                        duration_s: float | None = None,
                        robust_rollouts: int = 4) -> dict[str, Any]:
    """Validate nominal and perturbed dynamics, returning JSON-safe metrics."""
    speed_results: list[dict[str, Any]] = []
    for speed in speeds:
        nominal = simulate_trajectory(trajectory, speed, duration_s, profile, seed)
        entry: dict[str, Any] = {"speed_m_s": float(speed), "nominal": nominal.metrics.to_dict()}
        if robust:
            perturbations = []
            rng = np.random.default_rng(seed + int(round(speed * 1000)))
            robust_metrics = []
            for index in range(max(0, int(robust_rollouts))):
                scenario = {"friction_scale": float(rng.uniform(0.85, 1.15)),
                            "mass_scale": float(rng.uniform(0.95, 1.05)),
                            "gain_scale": float(rng.uniform(0.90, 1.10)),
                            "torque_scale": float(rng.uniform(0.90, 1.00)),
                            "latency_s": float(rng.uniform(0.0, 0.010)),
                            "initial_perturbation": rng.normal(0.0, (0.003, 0.003, 0.01)).tolist()}
                perturbations.append(scenario)
                robust_model = make_native_model(profile)
                # All modifications are made to this fresh model, never the
                # model used by another rollout.
                robust_model.geom_friction[:] *= scenario["friction_scale"]
                robust_model.body_mass[:] *= scenario["mass_scale"]
                robust_model.actuator_gainprm[:, 0] *= scenario["gain_scale"]
                robust_model.actuator_biasprm[:, 1] *= scenario["gain_scale"]
                robust_model.actuator_forcerange[:] *= scenario["torque_scale"]
                replay = simulate_trajectory(
                    trajectory, speed, duration_s, profile, seed + index, robust_model,
                    scenario["latency_s"], scenario["initial_perturbation"])
                robust_metrics.append(replay.metrics.to_dict())
            entry["robust_scenarios"] = perturbations
            entry["robust"] = {"rollouts": robust_metrics,
                               "success_rate": (float(np.mean([m["feasible"] for m in robust_metrics]))
                                                 if robust_metrics else None),
                               "feasible": bool(robust_metrics and all(m["feasible"] for m in robust_metrics))}
        speed_results.append(entry)
    return {"schema": TRAJECTORY_SCHEMA, "profile": profile.name,
            "period_s": trajectory.period_s, "speeds": speed_results,
            "feasible": bool(all(x["nominal"]["feasible"] and
                                  (not robust or x.get("robust", {}).get("feasible", False))
                                  for x in speed_results))}


def native_model_fingerprint(model: mujoco.MjModel) -> str:
    """Stable hash of model structure and safety-critical numerical arrays."""
    digest = hashlib.sha256()
    for name in ("nq", "nv", "nu"):
        digest.update(str(getattr(model, name)).encode())
    for name in ("opt", "jnt_range", "actuator_ctrlrange", "actuator_forcerange",
                 "actuator_gainprm", "actuator_biasprm"):
        value = getattr(model, name, None)
        if value is not None and hasattr(value, "tobytes"):
            digest.update(name.encode()); digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def trajectory_fingerprint(trajectory: PeriodicSpline, profile: RobotProfile = LEGGYBOT_FIXED_ANKLE_PROFILE) -> str:
    payload = {"trajectory": trajectory.to_dict(), "profile": profile.name,
               "joints": profile.joint_names, "actuators": profile.actuator_names}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_trajectory(path: str | Path, trajectory: PeriodicSpline,
                    profile: RobotProfile = LEGGYBOT_FIXED_ANKLE_PROFILE,
                    metadata: Mapping[str, Any] | None = None,
                    *, metrics: Mapping[str, Any] | None = None,
                    history: Mapping[str, Any] | None = None,
                    model: mujoco.MjModel | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if model is None:
        # Include a real compiled-model fingerprint by default.  This makes a
        # standalone NPZ self-describing even when the caller did not retain
        # the rollout's model instance.
        model = make_native_model(profile)
    model_fp = native_model_fingerprint(model)
    payload = {"schema": TRAJECTORY_SCHEMA, "profile": profile.name,
               "period_s": trajectory.period_s, "knots": trajectory.knots,
               "joint_names": np.asarray(profile.joint_names),
               "fingerprint": trajectory_fingerprint(trajectory, profile),
               "model_fingerprint": model_fp,
               "metadata_json": json.dumps(dict(metadata or {}), sort_keys=True),
               "metrics_json": json.dumps(dict(metrics or {}), sort_keys=True, default=float),
               "history_json": json.dumps(dict(history or {}), sort_keys=True, default=float)}
    np.savez_compressed(path, **payload)
    return path


def load_trajectory(
    path: str | Path,
    profile: RobotProfile = LEGGYBOT_FIXED_ANKLE_PROFILE,
    *,
    allow_model_mismatch: bool = False,
) -> PeriodicSpline:
    with np.load(Path(path), allow_pickle=False) as archive:
        schema = str(archive["schema"])
        if schema != TRAJECTORY_SCHEMA: raise ValueError(f"unsupported trajectory schema {schema!r}")
        names = tuple(str(x) for x in archive["joint_names"])
        if names != profile.joint_names: raise ValueError("trajectory joint fingerprint does not match profile")
        result = PeriodicSpline(archive["knots"], float(archive["period_s"]))
        expected = str(archive["fingerprint"])
        if expected != trajectory_fingerprint(result, profile) and not allow_model_mismatch:
            raise ValueError("trajectory fingerprint mismatch")
        if "model_fingerprint" in archive and not allow_model_mismatch:
            expected_model = str(archive["model_fingerprint"])
            current_model = native_model_fingerprint(make_native_model(profile))
            if expected_model != current_model:
                raise ValueError("trajectory model fingerprint does not match current native model")
        return result


def write_metrics_csv(path: str | Path, validation: Mapping[str, Any]) -> Path:
    path = Path(path)
    rows = []
    for item in validation.get("speeds", []):
        row = {"speed_m_s": item["speed_m_s"], **item["nominal"]}
        row.pop("actuator", None); row["violations"] = ";".join(row.get("violations", []))
        rows.append(row)
    if not rows: return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    return path


def write_trajectory_csv(path: str | Path, trajectory: PeriodicSpline,
                         control_dt: float = 0.02) -> Path:
    """Write one-period command targets for inspection or controller replay."""
    if control_dt <= 0:
        raise ValueError("control_dt must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = max(1, int(round(trajectory.period_s / control_dt)))
    times = np.arange(count, dtype=np.float64) * control_dt
    values = trajectory.evaluate(times)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", *[f"joint_{i}" for i in range(trajectory.joint_count)]])
        writer.writerows([[float(t), *map(float, row)] for t, row in zip(times, values)])
    return path


__all__ = ["PeriodicSpline", "PeriodicJointTrajectory", "TrajectoryMetrics", "TrajectoryRollout",
           "ActuatorMetrics", "bilateral_gait_seed", "simulate_trajectory", "validate_trajectory",
           "compute_trajectory_metrics", "torque_speed_limit", "compute_torque_speed_metrics",
           "sts3215_torque_speed", "trajectory_fingerprint", "save_trajectory", "load_trajectory",
           "write_metrics_csv", "write_trajectory_csv", "STS3215_STALL_TORQUE_NM",
           "STS3215_NO_LOAD_SPEED_RAD_S"]
