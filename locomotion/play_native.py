"""Real-time robot policy playback using native CPU MuJoCo.

This is the inference/deployment-style runtime. It intentionally bypasses
MuJoCo-Warp and every mjlab training manager for efficient single-robot use.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from threading import Lock

import mujoco
import mujoco.viewer
import numpy as np

from locomotion.imu import Icm42688OrientationEstimator
from locomotion.native_policy import NumpyPolicy, bounded_policy_action, load_native_policy
from locomotion.observation_latency import ObservationAgeModel
from locomotion.servo import PositionBasedVelocityEstimator, PositionSetpointProfiler
from locomotion.robot_profiles import (
    DEFAULT_ROBOT_NAME,
    GO1_PROFILE,
    LEGGYBOT_FIXED_ANKLE_PROFILE,
    RobotProfile,
    get_robot_profile,
    make_native_model,
    robot_choices,
)

# Compatibility aliases now describe the default robot. New code should use a profile.
PHYSICS_DT = LEGGYBOT_FIXED_ANKLE_PROFILE.physics_dt
CONTROL_DECIMATION = LEGGYBOT_FIXED_ANKLE_PROFILE.control_decimation
CONTROL_DT = LEGGYBOT_FIXED_ANKLE_PROFILE.control_dt
JOINT_NAMES = LEGGYBOT_FIXED_ANKLE_PROFILE.joint_names
DEFAULT_JOINT_POS = LEGGYBOT_FIXED_ANKLE_PROFILE.array("default_joint_pos")


class NativeKeyboardCommand:
    def __init__(
        self, linear_step: float, yaw_step: float, profile: RobotProfile
    ):
        self.linear_step = linear_step
        self.yaw_step = yaw_step
        self.profile = profile
        self._target = np.zeros(3, dtype=np.float32)
        self._reset_requested = False
        self._lock = Lock()

    def callback(self, key: int) -> None:
        # GLFW key values are stable ASCII codes for these uppercase letters.
        with self._lock:
            if key == ord("W"):
                self._target[0] += self.linear_step
            elif key == ord("S"):
                self._target[0] -= self.linear_step
            elif key == ord("A"):
                self._target[1] += self.linear_step
            elif key == ord("D"):
                self._target[1] -= self.linear_step
            elif key == ord("Q"):
                self._target[2] += self.yaw_step
            elif key == ord("E"):
                self._target[2] -= self.yaw_step
            elif key == ord("X"):
                self._target[:] = 0.0
            elif key in (257, 13):  # GLFW Enter and carriage return.
                self._reset_requested = True
                return
            else:
                return
            self._target[:] = np.clip(
                self._target,
                -np.asarray(self.profile.max_velocity_command),
                np.asarray(self.profile.max_velocity_command),
            )
            if self.profile.max_abs_curvature is not None:
                yaw_limit = self.profile.max_abs_curvature * abs(self._target[0])
                self._target[2] = np.clip(self._target[2], -yaw_limit, yaw_limit)
            print(
                f"command vx={self._target[0]:+.2f} "
                f"vy={self._target[1]:+.2f} yaw={self._target[2]:+.2f}"
            )

    def snapshot(self) -> tuple[np.ndarray, bool]:
        with self._lock:
            target = self._target.copy()
            reset = self._reset_requested
            self._reset_requested = False
        return target, reset


def make_native_go1_model() -> mujoco.MjModel:
    """Backward-compatible explicit Go1 model factory."""
    return make_native_model(GO1_PROFILE)


def _joint_addresses(
    model: mujoco.MjModel, profile: RobotProfile = LEGGYBOT_FIXED_ANKLE_PROFILE
) -> tuple[np.ndarray, np.ndarray]:
    joint_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in profile.joint_names
        ]
    )
    if np.any(joint_ids < 0):
        raise ValueError("Native model is missing a policy joint")
    return model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]


def _control_ids(
    model: mujoco.MjModel, profile: RobotProfile = LEGGYBOT_FIXED_ANKLE_PROFILE
) -> np.ndarray:
    actuator_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in profile.actuator_names
        ]
    )
    if np.any(actuator_ids < 0):
        raise ValueError("Native model is missing a policy actuator")
    return actuator_ids


def reset_robot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addresses: np.ndarray,
    profile: RobotProfile = LEGGYBOT_FIXED_ANKLE_PROFILE,
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = profile.initial_root_pos
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[qpos_addresses] = profile.array("default_joint_pos")
    mujoco.mj_forward(model, data)


def build_actor_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addresses: np.ndarray,
    dof_addresses: np.ndarray,
    previous_action: np.ndarray,
    command: np.ndarray,
    profile: RobotProfile = LEGGYBOT_FIXED_ANKLE_PROFILE,
    gait_period_s: float | None = None,
    velocity_estimator: PositionBasedVelocityEstimator | None = None,
    imu_estimator: Icm42688OrientationEstimator | None = None,
    observation_age: ObservationAgeModel | None = None,
) -> np.ndarray:
    gyro_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, profile.gyro_sensor
    )
    if gyro_id < 0:
        raise ValueError("Native model has no imu_ang_vel sensor")
    gyro_adr = model.sensor_adr[gyro_id]
    accel_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, profile.accelerometer_sensor
    )
    if accel_id < 0:
        raise ValueError("Native model has no accelerometer sensor")
    accel_adr = model.sensor_adr[accel_id]
    if imu_estimator is None:
        # Preserve the legacy Go1 contract. All STS3215 LeggyBot runtimes pass
        # the real 6-axis estimator explicitly below.
        base_ang_vel = data.sensordata[gyro_adr : gyro_adr + 3]
        inverse_quat = data.qpos[3:7].copy()
        inverse_quat[1:] *= -1.0
        projected_gravity = np.empty(3, dtype=np.float64)
        mujoco.mju_rotVecQuat(
            projected_gravity, np.array([0.0, 0.0, -1.0]), inverse_quat
        )
    else:
        imu_estimator.update(
            data.sensordata[gyro_adr : gyro_adr + 3],
            data.sensordata[accel_adr : accel_adr + 3],
            round(data.time * 1.0e9),
        )
        base_ang_vel = imu_estimator.angular_velocity
        projected_gravity = imu_estimator.projected_gravity

    observed_joint_position = profile.observed_joint_position(
        data.qpos[qpos_addresses]
    )
    if profile.position_resolution_rad is None:
        observed_joint_velocity = data.qvel[dof_addresses]
    else:
        if velocity_estimator is None:
            raise ValueError(
                "discrete-servo profiles require a position-based velocity estimator"
            )
        observed_joint_velocity = velocity_estimator.update(observed_joint_position)

    if observation_age is not None:
        (
            base_ang_vel,
            projected_gravity,
            observed_joint_position,
            observed_joint_velocity,
        ) = observation_age.apply(
            base_ang_vel,
            projected_gravity,
            observed_joint_position,
            observed_joint_velocity,
        )

    terms = [
            base_ang_vel,
            projected_gravity,
            observed_joint_position - profile.array("default_joint_pos"),
            observed_joint_velocity,
            previous_action,
            command,
    ]
    if gait_period_s is not None:
        angle = 2.0 * np.pi * ((data.time / gait_period_s) % 1.0)
        terms.append(np.asarray((np.sin(angle), np.cos(angle))))
    observation = np.concatenate(terms).astype(np.float32)
    expected_dim = profile.observation_dim + (2 if gait_period_s is not None else 0)
    if observation.shape != (expected_dim,):
        raise RuntimeError(f"Unexpected actor observation shape: {observation.shape}")
    return observation


def apply_policy_action(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    control_ids: np.ndarray,
    action: np.ndarray,
    profile: RobotProfile = LEGGYBOT_FIXED_ANKLE_PROFILE,
    position_profiler: PositionSetpointProfiler | None = None,
) -> None:
    target = profile.action_target_from_normalized(action)
    if position_profiler is None:
        data.ctrl[control_ids] = target
    else:
        position_profiler.set_goal(target)


def make_position_profiler(
    profile: RobotProfile,
) -> PositionSetpointProfiler | None:
    if profile.position_resolution_rad is None:
        return None
    return PositionSetpointProfiler(profile.array("default_joint_pos"))


def step_native_physics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    control_ids: np.ndarray,
    profile: RobotProfile,
    position_profiler: PositionSetpointProfiler | None,
) -> None:
    if position_profiler is not None:
        data.ctrl[control_ids] = position_profiler.update(profile.physics_dt)
    mujoco.mj_step(model, data)


def benchmark(policy: NumpyPolicy, seconds: float, profile: RobotProfile) -> float:
    model = make_native_model(profile)
    data = mujoco.MjData(model)
    qpos_adr, dof_adr = _joint_addresses(model, profile)
    ctrl_ids = _control_ids(model, profile)
    reset_robot(model, data, qpos_adr, profile)
    command = np.zeros(3, dtype=np.float32)
    action = np.zeros(profile.action_dim, dtype=np.float32)
    physics_steps = max(1, round(seconds / profile.physics_dt))
    gait_period = profile.gait_period_for_task(getattr(policy, "task_id", None))
    velocity_estimator = PositionBasedVelocityEstimator(profile.control_dt)
    imu_estimator = Icm42688OrientationEstimator()
    observation_age = (
        ObservationAgeModel() if profile.position_resolution_rad is not None else None
    )
    position_profiler = make_position_profiler(profile)

    started = time.perf_counter()
    for step in range(physics_steps):
        if step % profile.control_decimation == 0:
            observation = build_actor_observation(
                model, data, qpos_adr, dof_adr, action, command, profile, gait_period,
                velocity_estimator,
                imu_estimator,
                observation_age,
            )
            action = bounded_policy_action(policy, observation)
            apply_policy_action(
                model, data, ctrl_ids, action, profile, position_profiler
            )
        step_native_physics(
            model, data, ctrl_ids, profile, position_profiler
        )
    elapsed = time.perf_counter() - started
    return physics_steps * profile.physics_dt / elapsed


def run_viewer(
    policy: NumpyPolicy,
    keyboard: NativeKeyboardCommand,
    frame_rate: float,
    profile: RobotProfile,
) -> None:
    model = make_native_model(profile)
    data = mujoco.MjData(model)
    qpos_adr, dof_adr = _joint_addresses(model, profile)
    ctrl_ids = _control_ids(model, profile)
    reset_robot(model, data, qpos_adr, profile)
    action = np.zeros(profile.action_dim, dtype=np.float32)
    render_period = 1.0 / frame_rate
    gait_period = profile.gait_period_for_task(getattr(policy, "task_id", None))
    velocity_estimator = PositionBasedVelocityEstimator(profile.control_dt)
    imu_estimator = Icm42688OrientationEstimator()
    observation_age = (
        ObservationAgeModel() if profile.position_resolution_rad is not None else None
    )
    position_profiler = make_position_profiler(profile)

    with mujoco.viewer.launch_passive(
        model, data, key_callback=keyboard.callback, show_left_ui=False
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, profile.root_body
        )
        viewer.cam.distance = 1.5
        viewer.cam.elevation = -10.0
        viewer.cam.azimuth = 90.0
        next_control = time.perf_counter()
        next_render = next_control
        print("Controls: W/S forward, A/D lateral, Q/E yaw, X stop, Enter reset")

        while viewer.is_running():
            now = time.perf_counter()
            command, reset_requested = keyboard.snapshot()
            if reset_requested:
                reset_robot(model, data, qpos_adr, profile)
                action[:] = 0.0
                velocity_estimator.reset()
                imu_estimator.reset()
                if observation_age is not None:
                    observation_age.reset()
                if position_profiler is not None:
                    position_profiler.reset(profile.array("default_joint_pos"))
                next_control = now

            if now >= next_control:
                observation = build_actor_observation(
                    model,
                    data,
                    qpos_adr,
                    dof_adr,
                    action,
                    command,
                    profile,
                    gait_period,
                    velocity_estimator,
                    imu_estimator,
                    observation_age,
                )
                action = bounded_policy_action(policy, observation)
                apply_policy_action(
                    model,
                    data,
                    ctrl_ids,
                    action,
                    profile,
                    position_profiler,
                )
                for _ in range(profile.control_decimation):
                    step_native_physics(
                        model, data, ctrl_ids, profile, position_profiler
                    )
                next_control += profile.control_dt
                if now - next_control > profile.control_dt:
                    # Do not create an unbounded catch-up burst after window stalls.
                    next_control = now + profile.control_dt

            if now >= next_render:
                viewer.sync()
                next_render = now + render_period

            sleep_until = min(next_control, next_render)
            if sleep_until > time.perf_counter():
                time.sleep(min(sleep_until - time.perf_counter(), 0.002))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--robot", choices=robot_choices(), default=DEFAULT_ROBOT_NAME
    )
    parser.add_argument("--frame-rate", type=float, default=60.0)
    parser.add_argument("--linear-step", type=float, default=0.1)
    parser.add_argument("--yaw-step", type=float, default=0.1)
    parser.add_argument(
        "--benchmark",
        type=float,
        metavar="SECONDS",
        help="run headless for this many simulated seconds and report throughput",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.frame_rate <= 0.0:
        raise ValueError("--frame-rate must be positive")
    policy, backend = load_native_policy(args.checkpoint)
    profile = get_robot_profile(args.robot)
    checkpoint_task = getattr(policy, "task_id", None)
    if checkpoint_task is not None and not profile.supports_task(checkpoint_task):
        raise ValueError(
            f"checkpoint task {checkpoint_task!r} is not compatible with "
            f"{profile.name!r}"
        )
    expected_observation_dim = profile.observation_dim_for_task(checkpoint_task)
    if policy.observation_dim != expected_observation_dim or policy.action_dim != profile.action_dim:
        raise ValueError(
            f"{profile.name} requires policy dimensions "
            f"{expected_observation_dim} -> {profile.action_dim}, got "
            f"{policy.observation_dim} -> {policy.action_dim}. Check --robot."
        )
    print(f"Loaded {backend} {profile.name} actor from {args.checkpoint} on CPU")

    if args.benchmark is not None:
        if args.benchmark <= 0.0:
            raise ValueError("--benchmark must be positive")
        realtime = benchmark(policy, args.benchmark, profile)
        print(f"Native MuJoCo throughput: {realtime:.1f}x real time")
        return

    keyboard = NativeKeyboardCommand(args.linear_step, args.yaw_step, profile)
    run_viewer(policy, keyboard, args.frame_rate, profile)


if __name__ == "__main__":
    main()
