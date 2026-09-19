"""Robot-agnostic policy controller using the versioned wire contract."""

from __future__ import annotations

import argparse
import math
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path
from types import TracebackType

import numpy as np

from locomotion.native_policy import NumpyPolicy, bounded_policy_action, load_native_policy
from locomotion.robot_profiles import (
    DEFAULT_ROBOT_NAME,
    LEGGYBOT_FIXED_ANKLE_PROFILE,
    RobotProfile,
    get_robot_profile,
    robot_choices,
)
from locomotion.servo import PositionBasedVelocityEstimator
from locomotion.robot_protocol import (
    ControllerTransport,
    JointCommand,
    UdpControllerTransport,
    monotonic_time_ns,
)
from locomotion.state_estimation import Icm42688StateEstimator, StateEstimator


class TerminalVelocityCommand:
    """Latched W/S, A/D, Q/E commands read from the controller terminal."""

    velocity_step = 0.01
    linear_velocity_limit = 0.10
    yaw_velocity_limit = math.pi / 4.0

    def __init__(
        self,
        initial: tuple[float, float, float],
        profile: RobotProfile,
        enabled: bool = True,
    ):
        self.target = np.asarray(initial, dtype=np.float32)
        self.profile = profile
        self._clamp_target()
        self.enabled = enabled and sys.stdin.isatty()
        self._original: list | None = None

    def __enter__(self) -> TerminalVelocityCommand:
        if self.enabled:
            self._original = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            os.set_blocking(sys.stdin.fileno(), False)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if self._original is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._original)

    def poll(self) -> None:
        if not self.enabled:
            return
        changed = False
        while select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1).lower()
            changed = self.apply_key(key) or changed
        if changed:
            print(
                f"command vx={self.target[0]:+.2f} "
                f"vy={self.target[1]:+.2f} yaw={self.target[2]:+.2f}"
            )

    def apply_key(self, key: str) -> bool:
        if key == "w":
            self.target[0] += self.velocity_step
        elif key == "s":
            self.target[0] -= self.velocity_step
        elif key == "a":
            self.target[1] += self.velocity_step
        elif key == "d":
            self.target[1] -= self.velocity_step
        elif key == "q":
            self.target[2] += self.velocity_step
        elif key == "e":
            self.target[2] -= self.velocity_step
        elif key == "x":
            self.target[:] = 0.0
        else:
            return False
        self._clamp_target()
        return True

    def _clamp_target(self) -> None:
        # Match the walking task envelope: linear commands are
        # bounded to +/-0.10 m/s and yaw to +/-pi/4 rad/s.  Do not apply the
        # legacy profile curvature cap here; it would prevent Q/E from
        # reaching the task's yaw range (and would couple yaw to vx).
        limits = np.asarray(
            (
                self.linear_velocity_limit,
                self.linear_velocity_limit,
                self.yaw_velocity_limit,
            ),
            dtype=np.float32,
        )
        self.target[:] = np.clip(
            self.target,
            -limits,
            limits,
        )


class PolicyController:
    """Frequency-decoupled state consumer and joint-command producer."""

    def __init__(
        self,
        policy: NumpyPolicy,
        transport: ControllerTransport,
        estimator: StateEstimator,
        profile: RobotProfile = LEGGYBOT_FIXED_ANKLE_PROFILE,
        *,
        control_hz: float = 50.0,
        state_timeout_s: float = 0.1,
        action_range_fraction: float = 1.0,
    ):
        if control_hz <= 0.0 or state_timeout_s <= 0.0:
            raise ValueError("control frequency and state timeout must be positive")
        self.policy = policy
        self.transport = transport
        self.estimator = estimator
        self.profile = profile
        if not 0.0 < action_range_fraction <= 1.0:
            raise ValueError("action range fraction must be in (0, 1]")
        self.action_range_fraction = action_range_fraction
        self.period_ns = round(1.0e9 / control_hz)
        self.state_timeout_ns = round(state_timeout_s * 1.0e9)
        self.last_action = np.zeros(profile.action_dim, dtype=np.float32)
        self.last_state_receive_ns: int | None = None
        self.last_state_sequence = -1
        self.received_state_count = 0
        self.command_sequence = 0
        self.dropped_or_reordered_states = 0
        self.gait_period_s = profile.gait_period_for_task(
            getattr(policy, "task_id", None)
        )
        self.gait_phase_origin_ns: int | None = None
        self.joint_velocity_estimator = PositionBasedVelocityEstimator(
            1.0 / control_hz
        )

    def receive_available_states(self, now_ns: int) -> None:
        for state in self.transport.receive_states():
            if state.robot_name != self.profile.name:
                raise ValueError(
                    f"received {state.robot_name!r} state for {self.profile.name!r} controller"
                )
            if state.joint_names != self.profile.joint_names:
                raise ValueError("robot joint order does not match the selected profile")
            if state.sequence <= self.last_state_sequence:
                self.dropped_or_reordered_states += 1
                continue
            if self.last_state_sequence >= 0:
                self.dropped_or_reordered_states += max(
                    0, state.sequence - self.last_state_sequence - 1
                )
            self.estimator.update(state)
            self.last_state_sequence = state.sequence
            self.last_state_receive_ns = now_ns
            self.received_state_count += 1

    def make_command(self, velocity_command: np.ndarray, now_ns: int) -> JointCommand:
        state = self.estimator.latest()
        stale = (
            state is None
            or self.last_state_receive_ns is None
            or now_ns - self.last_state_receive_ns > self.state_timeout_ns
        )
        if stale:
            self.last_action[:] = 0.0
            self.gait_phase_origin_ns = None
            self.joint_velocity_estimator.reset()
            self.estimator.reset()
            target = self.profile.quantize_joint_position(
                self.profile.array("default_joint_pos", dtype=np.float32)
            )
        else:
            default_pos = self.profile.array("default_joint_pos", dtype=np.float32)
            observed_position = self.profile.observed_joint_position(
                state.joint_position
            )
            observed_velocity = (
                state.joint_velocity
                if self.profile.position_resolution_rad is None
                else self.joint_velocity_estimator.update(observed_position)
            )
            terms = [
                    state.base_angular_velocity,
                    state.projected_gravity,
                    observed_position - default_pos,
                    observed_velocity,
                    self.last_action,
                    velocity_command,
            ]
            if self.gait_period_s is not None:
                if self.gait_phase_origin_ns is None:
                    self.gait_phase_origin_ns = now_ns
                elapsed_s = (now_ns - self.gait_phase_origin_ns) / 1.0e9
                angle = 2.0 * np.pi * ((elapsed_s / self.gait_period_s) % 1.0)
                terms.append(np.asarray((np.sin(angle), np.cos(angle))))
            observation = np.concatenate(terms).astype(np.float32)
            expected_dim = self.profile.observation_dim_for_task(
                getattr(self.policy, "task_id", None)
            )
            if observation.shape != (expected_dim,):
                raise RuntimeError(f"unexpected policy observation shape {observation.shape}")
            self.last_action = bounded_policy_action(self.policy, observation)
            target = self.profile.action_target_from_normalized(
                self.last_action, self.action_range_fraction
            )

        command = JointCommand(
            sequence=self.command_sequence,
            created_time_ns=now_ns,
            robot_name=self.profile.name,
            joint_names=self.profile.joint_names,
            position_target=tuple(float(value) for value in target),
            velocity_command=tuple(float(value) for value in velocity_command),
        )
        self.command_sequence += 1
        return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--robot", choices=robot_choices(), default=DEFAULT_ROBOT_NAME
    )
    parser.add_argument("--robot-host", default="127.0.0.1")
    parser.add_argument("--command-port", type=int, default=15000)
    parser.add_argument("--state-bind", default="0.0.0.0")
    parser.add_argument("--state-port", type=int, default=15001)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--state-timeout", type=float, default=0.1)
    parser.add_argument(
        "--action-range-fraction",
        type=float,
        default=1.0,
        help="training-time action authority to reproduce (e.g. 0.55); default: full range",
    )
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--no-keyboard", action="store_true")
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
    if policy.observation_dim != expected_observation_dim or policy.action_dim != profile.action_dim:
        raise ValueError(
            f"{profile.name} requires policy dimensions "
            f"{expected_observation_dim} -> {profile.action_dim}, got "
            f"{policy.observation_dim} -> {policy.action_dim}. Check --robot."
        )
    transport = UdpControllerTransport(
        state_bind=(args.state_bind, args.state_port),
        robot_command_address=(args.robot_host, args.command_port),
    )
    controller = PolicyController(
        policy,
        transport,
        Icm42688StateEstimator(),
        profile,
        control_hz=args.control_hz,
        state_timeout_s=args.state_timeout,
        action_range_fraction=args.action_range_fraction,
    )
    keyboard = TerminalVelocityCommand(
        (args.vx, args.vy, args.yaw), profile, enabled=not args.no_keyboard
    )
    print(
        f"Loaded {backend} {profile.name} policy; "
        f"action range={args.action_range_fraction:.2f}; "
        f"receiving state on {transport.local_address}, "
        f"sending commands to {(args.robot_host, args.command_port)}",
        flush=True,
    )
    print(
        "Controller keys: W/S forward, A/D lateral, Q/E yaw, X stop "
        "(0.01 steps; vx/vy +/-0.10, yaw +/-pi/4)",
        flush=True,
    )

    next_control_ns = monotonic_time_ns()
    next_status_ns = next_control_ns + 1_000_000_000
    previous_received_count = 0
    try:
        with keyboard:
            while True:
                now_ns = monotonic_time_ns()
                controller.receive_available_states(now_ns)
                keyboard.poll()
                if now_ns >= next_control_ns:
                    command = controller.make_command(keyboard.target, now_ns)
                    transport.send_command(command)
                    next_control_ns += controller.period_ns
                    if now_ns - next_control_ns > controller.period_ns:
                        next_control_ns = now_ns + controller.period_ns
                if now_ns >= next_status_ns:
                    received = controller.received_state_count - previous_received_count
                    age_ms = (
                        float("inf")
                        if controller.last_state_receive_ns is None
                        else (now_ns - controller.last_state_receive_ns) / 1.0e6
                    )
                    print(
                        f"state_rx={received}/s age={age_ms:.1f}ms "
                        f"sequence_gaps={controller.dropped_or_reordered_states} "
                        f"commands_sent={controller.command_sequence}",
                        flush=True,
                    )
                    previous_received_count = controller.received_state_count
                    next_status_ns += 1_000_000_000
                remaining = (next_control_ns - monotonic_time_ns()) / 1.0e9
                if remaining > 0.0:
                    time.sleep(min(remaining, 0.001))
    except KeyboardInterrupt:
        print("\nController stopped")
    finally:
        transport.close()


if __name__ == "__main__":
    main()
