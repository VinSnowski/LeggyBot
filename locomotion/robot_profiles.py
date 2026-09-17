"""Robot-specific deployment contracts shared by simulation and controllers."""

from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np

from locomotion.servo import (
    STS3215_POSITION_RESOLUTION_RAD,
    quantize_position_numpy,
)


@dataclass(frozen=True, slots=True)
class RobotProfile:
    """Everything outside the neural network that defines its robot ABI."""

    name: str
    task_id: str | None
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    default_joint_pos: tuple[float, ...]
    action_scale: tuple[float, ...]
    joint_lower: tuple[float, ...]
    joint_upper: tuple[float, ...]
    initial_root_pos: tuple[float, float, float]
    root_body: str
    gyro_sensor: str
    accelerometer_sensor: str
    max_velocity_command: tuple[float, float, float]
    compatible_task_ids: tuple[str, ...] = ()
    gait_clock_periods: tuple[tuple[str, float], ...] = ()
    max_abs_curvature: float | None = None
    physics_dt: float = 0.005
    control_decimation: int = 4
    action_limit_margin_rad: float | None = None
    actuator_stall_torque_nm: float = 2.942
    actuator_no_load_speed_rad_s: float = 4.712
    actuator_safe_torque_fraction: float = 0.80
    actuator_safe_speed_fraction: float = 0.80
    position_resolution_rad: float | None = None

    def __post_init__(self) -> None:
        count = len(self.joint_names)
        for field_name in (
            "actuator_names",
            "default_joint_pos",
            "action_scale",
            "joint_lower",
            "joint_upper",
        ):
            if len(getattr(self, field_name)) != count:
                raise ValueError(f"{self.name}: {field_name} must contain {count} values")
        if len(set(self.joint_names)) != count:
            raise ValueError(f"{self.name}: duplicate joint names")
        if self.physics_dt <= 0.0 or self.control_decimation <= 0:
            raise ValueError(f"{self.name}: invalid simulation timing")
        if self.actuator_stall_torque_nm <= 0 or self.actuator_no_load_speed_rad_s <= 0:
            raise ValueError(f"{self.name}: invalid actuator limits")
        if (
            self.position_resolution_rad is not None
            and self.position_resolution_rad <= 0.0
        ):
            raise ValueError(f"{self.name}: invalid position resolution")
        for task_id, period in self.gait_clock_periods:
            if task_id not in self.compatible_task_ids and task_id != self.task_id:
                raise ValueError(f"{self.name}: gait task {task_id!r} is not compatible")
            if period <= 0.0:
                raise ValueError(f"{self.name}: gait period must be positive")

    @property
    def action_dim(self) -> int:
        return len(self.joint_names)

    @property
    def observation_dim(self) -> int:
        # gyro + gravity + joint pos + joint vel + previous action + command
        return 9 + 3 * self.action_dim

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.control_decimation

    def supports_task(self, task_id: str) -> bool:
        return self.task_id is not None and (
            task_id == self.task_id or task_id in self.compatible_task_ids
        )

    def gait_period_for_task(self, task_id: str | None) -> float | None:
        if task_id is None:
            return None
        return dict(self.gait_clock_periods).get(task_id)

    def observation_dim_for_task(self, task_id: str | None) -> int:
        extra = 2 if self.gait_period_for_task(task_id) is not None else 0
        return self.observation_dim + extra

    def array(self, field_name: str, *, dtype=np.float64) -> np.ndarray:
        return np.asarray(getattr(self, field_name), dtype=dtype)

    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the usable target range exposed to normalized actions."""
        if self.action_limit_margin_rad is None:
            default = self.array("default_joint_pos")
            scale = self.array("action_scale")
            return default - scale, default + scale
        margin = float(self.action_limit_margin_rad)
        lower = self.array("joint_lower") + margin
        upper = self.array("joint_upper") - margin
        if np.any(lower >= upper):
            raise ValueError(f"{self.name}: action margin removes a joint range")
        return lower, upper

    def action_target_from_normalized(
        self, action: np.ndarray, range_fraction: float = 1.0
    ) -> np.ndarray:
        """Map normalized actions to asymmetric, margin-limited targets."""
        if not 0.0 < range_fraction <= 1.0:
            raise ValueError("range_fraction must be in (0, 1]")
        normalized = np.clip(np.asarray(action), -1.0, 1.0)
        lower, upper = self.action_bounds()
        nominal = self.array("default_joint_pos")
        lower = nominal + range_fraction * (lower - nominal)
        upper = nominal + range_fraction * (upper - nominal)
        target = nominal + np.where(
            normalized >= 0.0,
            normalized * (upper - nominal),
            normalized * (nominal - lower),
        )
        lower_limit = self.array("joint_lower")
        upper_limit = self.array("joint_upper")
        target = np.clip(target, lower_limit, upper_limit)
        if self.position_resolution_rad is not None:
            resolution = self.position_resolution_rad
            target = quantize_position_numpy(target, resolution)
            # Keep the result both representable and inside the physical
            # limits even when a limit lies between two encoder counts.
            quantized_lower = np.ceil(lower_limit / resolution) * resolution
            quantized_upper = np.floor(upper_limit / resolution) * resolution
            target = np.clip(target, quantized_lower, quantized_upper)
        return target

    def observed_joint_position(self, position: np.ndarray) -> np.ndarray:
        """Apply the deployed servo's feedback resolution to joint positions."""
        return self.quantize_joint_position(position)

    def quantize_joint_position(self, position: np.ndarray) -> np.ndarray:
        """Round a feedback or goal position to the servo's register grid."""
        value = np.asarray(position)
        if self.position_resolution_rad is None:
            return value
        return quantize_position_numpy(value, self.position_resolution_rad)


LEGGYBOT_PROFILE = RobotProfile(
    name="leggybot",
    task_id=None,
    joint_names=(
        "RightBaseToHip",
        "RightHipToThigh",
        "RightKnee",
        "LeftBaseToHip",
        "LeftHipToThigh",
        "LeftKnee",
    ),
    actuator_names=(
        "right_hip_roll",
        "right_thigh",
        "right_knee",
        "left_hip_roll",
        "left_thigh",
        "left_knee",
    ),
    default_joint_pos=(0.0,) * 6,
    # Normalized policy actions are clipped to +/-1.  These scales make the
    # phase-conditioned reference reachable: hip pitch needs about 0.57 rad
    # and the knee about 1.0 rad during mid-swing.
    action_scale=(0.15, 0.65, 1.05, 0.15, 0.65, 1.05),
    joint_lower=(-0.174533, -1.047198, -2.094395, -0.785398, -1.047198, -2.094395),
    joint_upper=(0.785398, 1.047198, 0.261799, 0.174533, 1.047198, 0.261799),
    initial_root_pos=(0.0, 0.0, -0.070),
    root_body="BaseLink",
    gyro_sensor="imu_ang_vel",
    accelerometer_sensor="base_accelerometer",
    max_velocity_command=(0.5, 0.5, 0.5),
    max_abs_curvature=1.0,
    position_resolution_rad=STS3215_POSITION_RESOLUTION_RAD,
)


LEGGYBOT_FIXED_ANKLE_PROFILE = replace(
    LEGGYBOT_PROFILE,
    name="leggybot-fixed-ankle-6dof",
    task_id="Mjlab-Stand-LeggyBot-FixedAnkle6DOF",
    # NewLeggyBot has three actuated joints per leg and a fixed angled foot.
    # All implemented skills share one actor ABI.
    compatible_task_ids=(
        "Mjlab-WeightTransfer-LeggyBot-FixedAnkle6DOF",
        "Mjlab-StepInPlace-LeggyBot-FixedAnkle6DOF",
        "Mjlab-AnimatedReferenceWalk-LeggyBot-FixedAnkle6DOF",
        "Mjlab-ConservativeWalk-LeggyBot-FixedAnkle6DOF",
        "Mjlab-UnloadingWalk-LeggyBot-FixedAnkle6DOF",
        "Mjlab-PeriodicWalk-LeggyBot-FixedAnkle6DOF",
        "Mjlab-StraightPeriodicWalk-LeggyBot-FixedAnkle6DOF",
        "Mjlab-SymmetricCyclePeriodicWalk-LeggyBot-FixedAnkle6DOF",
        "Mjlab-SymmetricPeriodicWalk-LeggyBot-FixedAnkle6DOF",
        "Mjlab-Sagittal-LeggyBot-FixedAnkle6DOF",
        "UpdatedDefaultWalkingTask",
        "UpdatedDefaultLateralWalkingTask",
        "QuasiOmnidirectionalWalkingTask",
        "QuasiOmnidirectionalWalkingTaskV2",
        "SE2TrackingWalkingTask",
        "NaturalOmnidirectionalWalkingTask",
        "NaturalOmnidirectionalWalkingTaskV3",
        "NaturalOmnidirectionalWalkingTaskV4",
        "NaturalOmnidirectionalWalkingTaskV5",
        "PeriodicRewardCompositionWalkingTask",
    ),
    gait_clock_periods=(
        ("Mjlab-Stand-LeggyBot-FixedAnkle6DOF", 0.8),
        ("Mjlab-WeightTransfer-LeggyBot-FixedAnkle6DOF", 0.8),
        ("Mjlab-StepInPlace-LeggyBot-FixedAnkle6DOF", 0.8),
        ("Mjlab-AnimatedReferenceWalk-LeggyBot-FixedAnkle6DOF", 0.33),
        ("Mjlab-ConservativeWalk-LeggyBot-FixedAnkle6DOF", 0.33),
        ("Mjlab-UnloadingWalk-LeggyBot-FixedAnkle6DOF", 0.33),
        ("Mjlab-PeriodicWalk-LeggyBot-FixedAnkle6DOF", 0.33),
        ("Mjlab-StraightPeriodicWalk-LeggyBot-FixedAnkle6DOF", 0.33),
        ("Mjlab-SymmetricCyclePeriodicWalk-LeggyBot-FixedAnkle6DOF", 0.33),
        ("Mjlab-SymmetricPeriodicWalk-LeggyBot-FixedAnkle6DOF", 0.33),
        ("Mjlab-Sagittal-LeggyBot-FixedAnkle6DOF", 0.8),
        ("UpdatedDefaultWalkingTask", 0.33),
        ("UpdatedDefaultLateralWalkingTask", 0.33),
        ("QuasiOmnidirectionalWalkingTask", 0.33),
        ("QuasiOmnidirectionalWalkingTaskV2", 0.33),
        ("SE2TrackingWalkingTask", 0.33),
        ("NaturalOmnidirectionalWalkingTask", 0.33),
        ("NaturalOmnidirectionalWalkingTaskV3", 0.33),
        ("NaturalOmnidirectionalWalkingTaskV4", 0.33),
        ("NaturalOmnidirectionalWalkingTaskV5", 0.33),
        ("PeriodicRewardCompositionWalkingTask", 0.33),
    ),
    # An 8-degree hip offset keeps the torso upright and the angled sole flat.
    default_joint_pos=(0.0, -0.13962634, 0.0, 0.0, -0.13962634, 0.0),
    action_limit_margin_rad=0.035,
)

GO1_PROFILE = RobotProfile(
    name="go1",
    task_id="Mjlab-Velocity-Flat-Unitree-Go1",
    joint_names=(
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ),
    actuator_names=(
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ),
    default_joint_pos=(
        0.1,
        0.9,
        -1.8,
        -0.1,
        0.9,
        -1.8,
        0.1,
        0.9,
        -1.8,
        -0.1,
        0.9,
        -1.8,
    ),
    action_scale=(
        0.37275302,
        0.37275302,
        0.24850203,
        0.37275302,
        0.37275302,
        0.24850203,
        0.37275302,
        0.37275302,
        0.24850203,
        0.37275302,
        0.37275302,
        0.24850203,
    ),
    joint_lower=(-0.863, -0.686, -2.818) * 4,
    joint_upper=(0.863, 4.501, -0.888) * 4,
    initial_root_pos=(0.0, 0.0, 0.278),
    root_body="trunk",
    gyro_sensor="imu_ang_vel",
    accelerometer_sensor="imu_lin_acc",
    max_velocity_command=(1.5, 1.0, 0.7),
)

ROBOT_PROFILES = {
    profile.name: profile
    for profile in (
        LEGGYBOT_PROFILE,
        LEGGYBOT_FIXED_ANKLE_PROFILE,
        GO1_PROFILE,
    )
}
# This is intentionally a declaration, not a fabricated robot profile.  Until
# its URDF/MJCF, actuator names, limits and inertias are checked in, selecting
# it fails loudly rather than silently running the old six-DOF CAD model.
LEGGYBOT_8DOF_NAME = "leggybot-8dof"
DEFAULT_ROBOT_NAME = LEGGYBOT_FIXED_ANKLE_PROFILE.name


def robot_choices() -> tuple[str, ...]:
    """Names accepted by CLIs, including explicitly pending hardware."""
    return tuple(sorted((*ROBOT_PROFILES, LEGGYBOT_8DOF_NAME)))


def get_robot_profile(name: str = DEFAULT_ROBOT_NAME) -> RobotProfile:
    if name == LEGGYBOT_8DOF_NAME:
        raise ValueError(
            "LeggyBot8DOF has no checked-in URDF/MJCF, "
            "eight joint names, limits and actuator configuration are not available yet. "
            "Use --robot leggybot-fixed-ankle-6dof for NewLeggyBot."
        )
    try:
        return ROBOT_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(robot_choices())
        raise ValueError(f"Unknown robot {name!r}; choose one of: {choices}") from exc


def make_native_model(profile: RobotProfile) -> mujoco.MjModel:
    """Compile the profile's training robot and a native flat plane."""
    from mjlab.entity import Entity

    leggybot_names = {
        LEGGYBOT_PROFILE.name,
        LEGGYBOT_FIXED_ANKLE_PROFILE.name,
    }
    if profile.name in leggybot_names:
        from locomotion.tasks.leggybot import get_leggybot_robot_cfg

        ankle_pitch_deg = 0.0 if profile.name == LEGGYBOT_PROFILE.name else 8.0
        robot = Entity(get_leggybot_robot_cfg(ankle_pitch_deg=ankle_pitch_deg))
    elif profile.name == GO1_PROFILE.name:
        from mjlab.asset_zoo.robots.unitree_go1.go1_constants import get_go1_robot_cfg

        robot = Entity(get_go1_robot_cfg())
    else:  # A new profile must explicitly define its model construction here.
        raise ValueError(f"No native model factory registered for {profile.name!r}")

    spec = robot.spec
    floor = spec.worldbody.add_body(name="terrain")
    floor.add_geom(
        name="terrain",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0.0, 0.0, 0.01),
        friction=(1.0, 5.0e-3, 5.0e-4),
        contype=1,
        conaffinity=1,
    )
    model = spec.compile()
    model.opt.timestep = profile.physics_dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = 10
    model.opt.ls_iterations = 20 if profile.name == GO1_PROFILE.name else 10
    model.opt.gravity[:] = (0.0, 0.0, -9.81)
    if profile.name == GO1_PROFILE.name:
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        model.opt.impratio = 10.0
        model.opt.ccd_iterations = 50
    return model
