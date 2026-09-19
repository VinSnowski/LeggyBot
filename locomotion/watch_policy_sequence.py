"""Watch a walking checkpoint execute its command sequence in native MuJoCo.

The runtime uses the same deployment observation and action contract as
``locomotion.remote_controller``: physics advances at 200 Hz, the policy runs
at 50 Hz, its clipped previous action is fed back into the next observation,
and normalized actions are mapped to the robot profile's joint limits.
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from locomotion.imu import Icm42688OrientationEstimator
from locomotion.native_policy import NumpyPolicy, bounded_policy_action, load_native_policy
from locomotion.observation_latency import ObservationAgeModel
from locomotion.play_native import (
    _control_ids,
    _joint_addresses,
    build_actor_observation,
    make_position_profiler,
    reset_robot,
    step_native_physics,
)
from locomotion.robot_profiles import (
    DEFAULT_ROBOT_NAME,
    RobotProfile,
    get_robot_profile,
    make_native_model,
    robot_choices,
)
from locomotion.servo import PositionBasedVelocityEstimator


@dataclass(frozen=True, slots=True)
class CommandSegment:
    duration_s: float
    command: tuple[float, float, float]
    label: str


WALKING_COMMAND_SEQUENCE = (
    CommandSegment(5.0, (0.0, 0.0, 0.0), "stand"),
    CommandSegment(10.0, (0.08, 0.0, 0.0), "forward 0.08 m/s"),
    CommandSegment(10.0, (0.0, 0.06, 0.0), "left 0.06 m/s"),
    CommandSegment(10.0, (0.0, -0.06, 0.0), "right 0.06 m/s"),
    CommandSegment(10.0, (0.08, 0.0, 0.30), "forward + yaw left"),
    CommandSegment(10.0, (0.08, 0.0, -0.30), "forward + yaw right"),
)
TOTAL_DURATION_S = sum(segment.duration_s for segment in WALKING_COMMAND_SEQUENCE)
WALKING_NO_LATERAL_COMMAND_SEQUENCE = (
    WALKING_COMMAND_SEQUENCE[0],
    WALKING_COMMAND_SEQUENCE[1],
    WALKING_COMMAND_SEQUENCE[4],
    WALKING_COMMAND_SEQUENCE[5],
)


def sequence_duration(sequence: tuple[CommandSegment, ...]) -> float:
    return sum(segment.duration_s for segment in sequence)


def command_sequence(*, no_lateral: bool = False) -> tuple[CommandSegment, ...]:
    sequence = WALKING_COMMAND_SEQUENCE
    if no_lateral:
        sequence = tuple(
            segment for segment in sequence
            if not (
                abs(segment.command[1]) > 0.0
                and segment.command[0] == 0.0
                and segment.command[2] == 0.0
            )
        )
    return sequence


def command_segment_at(
    elapsed_s: float,
    sequence: tuple[CommandSegment, ...] = WALKING_COMMAND_SEQUENCE,
) -> tuple[int, CommandSegment]:
    """Return the active segment using half-open, exact schedule intervals."""
    if elapsed_s < 0.0:
        raise ValueError("elapsed time must be non-negative")
    end_s = 0.0
    for index, segment in enumerate(sequence):
        end_s += segment.duration_s
        if elapsed_s < end_s:
            return index, segment
    raise ValueError(
        f"command sequence is complete at {sequence_duration(sequence):g} seconds"
    )


@dataclass(frozen=True, slots=True)
class PlaybackResult:
    control_steps: int
    physics_steps: int
    simulated_seconds: float
    completed: bool


def _base_pose(data: mujoco.MjData) -> np.ndarray:
    """Return planar ``[x, y, yaw]`` from MuJoCo's free-joint pose."""
    w, x, y, z = data.qpos[3:7]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array((data.qpos[0], data.qpos[1], yaw), dtype=np.float64)


def _wrapped_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _local_pose_delta(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Express an endpoint pose relative to a starting body frame."""
    world_delta = end[:2] - start[:2]
    c, s = np.cos(start[2]), np.sin(start[2])
    return np.array((
        c * world_delta[0] + s * world_delta[1],
        -s * world_delta[0] + c * world_delta[1],
        _wrapped_angle(end[2] - start[2]),
    ))


def _format_motion(value: np.ndarray) -> str:
    return f"[{value[0]:+.4f} m {value[1]:+.4f} m {value[2]:+.4f} rad]"


def run_sequence(
    policy: NumpyPolicy,
    profile: RobotProfile,
    *,
    frame_rate: float = 50.0,
    action_range_fraction: float = 1.0,
    headless: bool = False,
    stop_after_s: float | None = None,
    no_lateral: bool = False,
) -> PlaybackResult:
    """Run the scripted sequence, in real time when the viewer is enabled."""
    if frame_rate <= 0.0:
        raise ValueError("frame rate must be positive")
    if not 0.0 < action_range_fraction <= 1.0:
        raise ValueError("action range fraction must be in (0, 1]")
    if not np.isclose(profile.control_dt, 1.0 / 50.0):
        raise ValueError(
            f"{profile.name} has a {1.0 / profile.control_dt:g} Hz control profile; "
            "this playback requires 50 Hz"
        )

    model = make_native_model(profile)
    data = mujoco.MjData(model)
    qpos_addresses, dof_addresses = _joint_addresses(model, profile)
    control_ids = _control_ids(model, profile)
    reset_robot(model, data, qpos_addresses, profile)

    action = np.zeros(profile.action_dim, dtype=np.float32)
    gait_period_s = profile.gait_period_for_task(getattr(policy, "task_id", None))
    velocity_estimator = PositionBasedVelocityEstimator(profile.control_dt)
    imu_estimator = Icm42688OrientationEstimator()
    observation_age = ObservationAgeModel()
    position_profiler = make_position_profiler(profile)
    sequence = command_sequence(no_lateral=no_lateral)
    duration_s = sequence_duration(sequence)
    if stop_after_s is None:
        stop_after_s = duration_s
    if not 0.0 < stop_after_s <= duration_s:
        raise ValueError(f"stop time must be in (0, {duration_s:g}]")
    total_control_steps = round(stop_after_s / profile.control_dt)
    viewer_context = (
        nullcontext(None)
        if headless
        else mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        )
    )

    control_step = 0
    physics_steps = 0
    active_segment = -1
    previous_pose = _base_pose(data)
    segment_start_pose = previous_pose.copy()
    segment_body_motion = np.zeros(3, dtype=np.float64)
    segment_command_motion = np.zeros(3, dtype=np.float64)
    segment_steps = 0
    window_body_motion = np.zeros(3, dtype=np.float64)
    window_steps = 0
    started = time.perf_counter()
    next_render = started
    render_period = 1.0 / frame_rate

    with viewer_context as viewer:
        if viewer is not None:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, profile.root_body
            )
            viewer.cam.distance = 1.5
            viewer.cam.elevation = -10.0
            viewer.cam.azimuth = 90.0

        while control_step < total_control_steps and (
            viewer is None or viewer.is_running()
        ):
            if viewer is not None:
                deadline = started + control_step * profile.control_dt
                remaining = deadline - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)

            elapsed_s = control_step * profile.control_dt
            segment_index, segment = command_segment_at(elapsed_s, sequence)
            if segment_index != active_segment:
                if active_segment >= 0:
                    duration = segment_steps * profile.control_dt
                    endpoint = _local_pose_delta(segment_start_pose, previous_pose)
                    print(
                        f"  segment summary ({duration:.2f}s): commanded integral "
                        f"{_format_motion(segment_command_motion)}; actual integrated "
                        f"body motion {_format_motion(segment_body_motion)}; endpoint "
                        f"in start frame {_format_motion(endpoint)}",
                        flush=True,
                    )
                active_segment = segment_index
                segment_start_pose = previous_pose.copy()
                segment_body_motion.fill(0.0)
                segment_command_motion.fill(0.0)
                segment_steps = 0
                window_body_motion.fill(0.0)
                window_steps = 0
                print(
                    f"t={elapsed_s:5.1f}s  {segment.label}: "
                    f"[vx vy yaw] = [{segment.command[0]:+.2f} "
                    f"{segment.command[1]:+.2f} {segment.command[2]:+.2f}]",
                    flush=True,
                )
            command = np.asarray(segment.command, dtype=np.float32)
            observation = build_actor_observation(
                model,
                data,
                qpos_addresses,
                dof_addresses,
                action,
                command,
                profile,
                gait_period_s,
                velocity_estimator,
                imu_estimator,
                observation_age,
            )
            action = bounded_policy_action(policy, observation)
            target = profile.action_target_from_normalized(
                action, action_range_fraction
            )
            if position_profiler is None:
                data.ctrl[control_ids] = target
            else:
                position_profiler.set_goal(target)
            for _ in range(profile.control_decimation):
                step_native_physics(
                    model, data, control_ids, profile, position_profiler
                )
                physics_steps += 1
            control_step += 1

            current_pose = _base_pose(data)
            world_delta = current_pose[:2] - previous_pose[:2]
            c, s = np.cos(previous_pose[2]), np.sin(previous_pose[2])
            body_step = np.array((
                c * world_delta[0] + s * world_delta[1],
                -s * world_delta[0] + c * world_delta[1],
                _wrapped_angle(current_pose[2] - previous_pose[2]),
            ))
            segment_body_motion += body_step
            window_body_motion += body_step
            segment_command_motion += command * profile.control_dt
            segment_steps += 1
            window_steps += 1
            previous_pose = current_pose

            if window_steps == round(1.0 / profile.control_dt):
                window_duration = window_steps * profile.control_dt
                actual_velocity = window_body_motion / window_duration
                segment_elapsed = segment_steps * profile.control_dt
                print(
                    f"  {segment_elapsed:4.1f}s window: actual average "
                    f"[vx vy yaw] = [{actual_velocity[0]:+.4f} "
                    f"{actual_velocity[1]:+.4f} {actual_velocity[2]:+.4f}] vs "
                    f"commanded [{command[0]:+.4f} {command[1]:+.4f} "
                    f"{command[2]:+.4f}]",
                    flush=True,
                )
                window_body_motion.fill(0.0)
                window_steps = 0

            now = time.perf_counter()
            if viewer is not None and now >= next_render:
                viewer.sync()
                next_render = now + render_period

    if active_segment >= 0 and segment_steps:
        duration = segment_steps * profile.control_dt
        endpoint = _local_pose_delta(segment_start_pose, previous_pose)
        print(
            f"  segment summary ({duration:.2f}s): commanded integral "
            f"{_format_motion(segment_command_motion)}; actual integrated body motion "
            f"{_format_motion(segment_body_motion)}; endpoint in start frame "
            f"{_format_motion(endpoint)}",
            flush=True,
        )

    completed = control_step == total_control_steps
    return PlaybackResult(
        control_steps=control_step,
        physics_steps=physics_steps,
        simulated_seconds=control_step * profile.control_dt,
        completed=completed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--robot", choices=robot_choices(), default=DEFAULT_ROBOT_NAME)
    parser.add_argument("--frame-rate", type=float, default=50.0)
    parser.add_argument(
        "--action-range-fraction",
        type=float,
        default=1.0,
        help="fraction of the training-time joint target range to expose",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run the same 55 simulated seconds without a viewer or real-time pacing",
    )
    parser.add_argument(
        "--no-lateral",
        action="store_true",
        help="skip the pure left and right lateral command segments",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    policy, backend = load_native_policy(args.checkpoint)
    profile = get_robot_profile(args.robot)
    checkpoint_task = getattr(policy, "task_id", None)
    if checkpoint_task is not None and not profile.supports_task(checkpoint_task):
        raise ValueError(
            f"checkpoint task {checkpoint_task!r} is not compatible with "
            f"{profile.name!r}"
        )
    expected_observation_dim = profile.observation_dim_for_task(checkpoint_task)
    if (
        policy.observation_dim != expected_observation_dim
        or policy.action_dim != profile.action_dim
    ):
        raise ValueError(
            f"{profile.name} requires policy dimensions "
            f"{expected_observation_dim} -> {profile.action_dim}, got "
            f"{policy.observation_dim} -> {policy.action_dim}; check --robot"
        )

    print(
        f"Loaded {backend} {profile.name} policy from {args.checkpoint} on CPU\n"
        f"MuJoCo physics: {1.0 / profile.physics_dt:g} Hz; "
        f"policy control: {1.0 / profile.control_dt:g} Hz; "
        f"duration: {sequence_duration(command_sequence(no_lateral=args.no_lateral)):g} s",
        flush=True,
    )
    result = run_sequence(
        policy,
        profile,
        frame_rate=args.frame_rate,
        action_range_fraction=args.action_range_fraction,
        headless=args.headless,
        no_lateral=args.no_lateral,
    )
    status = "completed" if result.completed else "viewer closed"
    print(
        f"Playback {status} after {result.simulated_seconds:.2f} simulated seconds "
        f"({result.control_steps} policy steps, {result.physics_steps} physics steps)."
    )


if __name__ == "__main__":
    main()
