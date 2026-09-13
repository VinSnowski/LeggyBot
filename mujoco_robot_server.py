"""Native CPU MuJoCo robot process controlled through the robot UDP contract."""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext

import mujoco
import mujoco.viewer
import numpy as np

from locomotion.play_native import _control_ids, _joint_addresses, reset_robot
from locomotion.robot_profiles import (
    DEFAULT_ROBOT_NAME,
    LEGGYBOT_FIXED_ANKLE_PROFILE,
    RobotProfile,
    get_robot_profile,
    make_native_model,
    robot_choices,
)
from locomotion.robot_protocol import (
    JointCommand,
    RawRobotState,
    RobotTransport,
    UdpRobotTransport,
    monotonic_time_ns,
)


class MujocoRobotServer:
    """Own physics and robot I/O; expose only state and joint-command messages."""

    def __init__(
        self,
        transport: RobotTransport,
        profile: RobotProfile = LEGGYBOT_FIXED_ANKLE_PROFILE,
        *,
        state_hz: float = 200.0,
        command_timeout_s: float = 0.1,
    ):
        if not 0.0 < state_hz <= 1.0 / profile.physics_dt:
            raise ValueError(f"state_hz must be in (0, {1.0 / profile.physics_dt}]")
        if command_timeout_s <= 0.0:
            raise ValueError("command timeout must be positive")
        self.transport = transport
        self.profile = profile
        self.model = make_native_model(profile)
        self.data = mujoco.MjData(self.model)
        self.qpos_addresses, self.dof_addresses = _joint_addresses(self.model, profile)
        self.control_ids = _control_ids(self.model, profile)
        self.gyro_address = self._sensor_address(profile.gyro_sensor)
        self.accel_address = self._sensor_address(profile.accelerometer_sensor)
        self.publish_period_s = 1.0 / state_hz
        self.next_publish_time_s = 0.0
        self.command_timeout_ns = round(command_timeout_s * 1.0e9)
        self.last_command_receive_ns: int | None = None
        self.last_command_sequence = -1
        self.state_sequence = 0
        self.timed_out = True
        reset_robot(self.model, self.data, self.qpos_addresses, profile)
        self.data.ctrl[self.control_ids] = profile.array("default_joint_pos")

    def _sensor_address(self, name: str) -> int:
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            raise ValueError(f"MuJoCo model has no sensor {name!r}")
        return int(self.model.sensor_adr[sensor_id])

    def _accept_latest_command(self, now_ns: int) -> JointCommand | None:
        latest = None
        for command in self.transport.receive_commands():
            if command.robot_name != self.profile.name:
                raise ValueError(
                    f"received {command.robot_name!r} command for {self.profile.name!r} robot"
                )
            if command.joint_names != self.profile.joint_names:
                raise ValueError("command joint order does not match the selected profile")
            if command.sequence > self.last_command_sequence:
                latest = command
                self.last_command_sequence = command.sequence
        if latest is not None:
            self.last_command_receive_ns = now_ns
        return latest

    def _make_state(self, now_ns: int) -> RawRobotState:
        state = RawRobotState(
            sequence=self.state_sequence,
            sample_time_ns=round(self.data.time * 1.0e9),
            sent_time_ns=now_ns,
            robot_name=self.profile.name,
            joint_names=self.profile.joint_names,
            joint_position=tuple(float(x) for x in self.data.qpos[self.qpos_addresses]),
            joint_velocity=tuple(float(x) for x in self.data.qvel[self.dof_addresses]),
            imu_angular_velocity=tuple(
                float(x)
                for x in self.data.sensordata[
                    self.gyro_address : self.gyro_address + 3
                ]
            ),
            imu_linear_acceleration=tuple(
                float(x)
                for x in self.data.sensordata[
                    self.accel_address : self.accel_address + 3
                ]
            ),
            base_orientation_wxyz=tuple(float(x) for x in self.data.qpos[3:7]),
        )
        self.state_sequence += 1
        return state

    def step(self, now_ns: int | None = None) -> None:
        now_ns = monotonic_time_ns() if now_ns is None else now_ns
        command = self._accept_latest_command(now_ns)
        if command is not None:
            self.data.ctrl[self.control_ids] = np.asarray(command.position_target)

        previously_timed_out = self.timed_out
        self.timed_out = (
            self.last_command_receive_ns is None
            or now_ns - self.last_command_receive_ns > self.command_timeout_ns
        )
        if previously_timed_out and not self.timed_out:
            print("Controller command stream active", flush=True)
        elif not previously_timed_out and self.timed_out:
            print("Command watchdog expired; holding nominal pose", flush=True)
        if self.timed_out:
            # A real robot driver should additionally cut torque or enter its own
            # certified safe state. The simulator holds this profile's nominal pose.
            self.data.ctrl[self.control_ids] = self.profile.array("default_joint_pos")

        mujoco.mj_step(self.model, self.data)
        if self.data.time + 1.0e-12 >= self.next_publish_time_s:
            self.transport.send_state(self._make_state(now_ns))
            self.next_publish_time_s += self.publish_period_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot", choices=robot_choices(), default=DEFAULT_ROBOT_NAME
    )
    parser.add_argument("--command-bind", default="0.0.0.0")
    parser.add_argument("--command-port", type=int, default=15000)
    parser.add_argument("--controller-host", default="127.0.0.1")
    parser.add_argument("--state-port", type=int, default=15001)
    parser.add_argument("--state-hz", type=float, default=200.0)
    parser.add_argument("--command-timeout", type=float, default=0.1)
    parser.add_argument("--render-hz", type=float, default=60.0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.render_hz <= 0.0:
        raise ValueError("render_hz must be positive")
    profile = get_robot_profile(args.robot)
    transport = UdpRobotTransport(
        command_bind=(args.command_bind, args.command_port),
        controller_state_address=(args.controller_host, args.state_port),
    )
    server = MujocoRobotServer(
        transport,
        profile,
        state_hz=args.state_hz,
        command_timeout_s=args.command_timeout,
    )
    print(
        f"MuJoCo {profile.name} listening for commands on {transport.local_address}; "
        f"publishing state to {(args.controller_host, args.state_port)}",
        flush=True,
    )

    viewer_context = (
        nullcontext(None)
        if args.headless
        else mujoco.viewer.launch_passive(
            server.model, server.data, show_left_ui=False, show_right_ui=False
        )
    )
    next_step = time.perf_counter()
    next_render = next_step
    render_period = 1.0 / args.render_hz
    try:
        with viewer_context as viewer:
            if viewer is not None:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = mujoco.mj_name2id(
                    server.model, mujoco.mjtObj.mjOBJ_BODY, profile.root_body
                )
                viewer.cam.distance = 1.5
                viewer.cam.elevation = -10.0
                viewer.cam.azimuth = 90.0
            while viewer is None or viewer.is_running():
                now = time.perf_counter()
                if now >= next_step:
                    server.step()
                    next_step += profile.physics_dt
                    if now - next_step > profile.physics_dt:
                        next_step = now + profile.physics_dt
                if viewer is not None and now >= next_render:
                    viewer.sync()
                    next_render = now + render_period
                deadline = min(next_step, next_render) if viewer is not None else next_step
                remaining = deadline - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(min(remaining, 0.002))
    except KeyboardInterrupt:
        print("\nMuJoCo robot server stopped")
    finally:
        transport.close()


if __name__ == "__main__":
    main()
