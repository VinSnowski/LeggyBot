"""Versioned asynchronous wire contract between robot I/O and controllers.

The UDP/JSON transport is intentionally small and inspectable.  The dataclasses
are the important boundary: a production DDS, Zenoh, ROS 2, shared-memory, or
hardware transport can implement the same interface without changing policy code.
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

PROTOCOL_VERSION = 2
MAX_DATAGRAM_BYTES = 65_507


def _finite_vector(name: str, values: tuple[float, ...], length: int) -> None:
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(values)}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity")


@dataclass(frozen=True, slots=True)
class RawRobotState:
    sequence: int
    sample_time_ns: int
    sent_time_ns: int
    robot_name: str
    joint_names: tuple[str, ...]
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    imu_angular_velocity: tuple[float, float, float]
    imu_linear_acceleration: tuple[float, float, float]
    # Perfect MuJoCo orientation for the placeholder estimator. A raw-hardware
    # transport may omit it once an IMU estimator is implemented here.
    base_orientation_wxyz: tuple[float, float, float, float] | None

    def validate(self) -> None:
        if self.sequence < 0 or self.sample_time_ns < 0 or self.sent_time_ns < 0:
            raise ValueError("state sequence and timestamps must be non-negative")
        if not self.robot_name:
            raise ValueError("robot_name must not be empty")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("state contains duplicate joint names")
        _finite_vector("joint_position", self.joint_position, len(self.joint_names))
        _finite_vector("joint_velocity", self.joint_velocity, len(self.joint_names))
        _finite_vector("imu_angular_velocity", self.imu_angular_velocity, 3)
        _finite_vector("imu_linear_acceleration", self.imu_linear_acceleration, 3)
        if self.base_orientation_wxyz is not None:
            _finite_vector("base_orientation_wxyz", self.base_orientation_wxyz, 4)


@dataclass(frozen=True, slots=True)
class JointCommand:
    sequence: int
    created_time_ns: int
    robot_name: str
    joint_names: tuple[str, ...]
    position_target: tuple[float, ...]
    velocity_command: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self) -> None:
        if self.sequence < 0 or self.created_time_ns < 0:
            raise ValueError("command sequence and timestamp must be non-negative")
        if not self.robot_name:
            raise ValueError("robot_name must not be empty")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("command contains duplicate joint names")
        _finite_vector("position_target", self.position_target, len(self.joint_names))
        _finite_vector("velocity_command", self.velocity_command, 3)


WireMessage = RawRobotState | JointCommand


def encode_message(message: WireMessage) -> bytes:
    message.validate()
    payload = asdict(message)
    payload["version"] = PROTOCOL_VERSION
    payload["type"] = "state" if isinstance(message, RawRobotState) else "command"
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")


def decode_message(payload: bytes) -> WireMessage:
    if len(payload) > MAX_DATAGRAM_BYTES:
        raise ValueError("datagram exceeds the UDP payload limit")
    raw = json.loads(payload)
    if raw.pop("version", None) != PROTOCOL_VERSION:
        raise ValueError("unsupported robot protocol version")
    message_type = raw.pop("type", None)
    for key in (
        "joint_names",
        "joint_position",
        "joint_velocity",
        "imu_angular_velocity",
        "imu_linear_acceleration",
        "base_orientation_wxyz",
        "position_target",
        "velocity_command",
    ):
        if key in raw and raw[key] is not None:
            raw[key] = tuple(raw[key])
    if message_type == "state":
        message: WireMessage = RawRobotState(**raw)
    elif message_type == "command":
        message = JointCommand(**raw)
    else:
        raise ValueError(f"unknown robot message type: {message_type!r}")
    message.validate()
    return message


class ControllerTransport(Protocol):
    def receive_states(self) -> list[RawRobotState]: ...
    def send_command(self, command: JointCommand) -> None: ...
    def close(self) -> None: ...


class RobotTransport(Protocol):
    def receive_commands(self) -> list[JointCommand]: ...
    def send_state(self, state: RawRobotState) -> None: ...
    def close(self) -> None: ...


class UdpControllerTransport:
    """Controller-side UDP endpoint; receives every queued state, latest wins."""

    def __init__(
        self,
        *,
        state_bind: tuple[str, int],
        robot_command_address: tuple[str, int],
    ):
        self.robot_command_address = robot_command_address
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(state_bind)
        self.socket.setblocking(False)

    @property
    def local_address(self) -> tuple[str, int]:
        host, port = self.socket.getsockname()[:2]
        return str(host), int(port)

    def receive_states(self) -> list[RawRobotState]:
        states = []
        while True:
            try:
                payload, _ = self.socket.recvfrom(MAX_DATAGRAM_BYTES)
            except BlockingIOError:
                break
            message = decode_message(payload)
            if isinstance(message, RawRobotState):
                states.append(message)
        return states

    def send_command(self, command: JointCommand) -> None:
        self.socket.sendto(encode_message(command), self.robot_command_address)

    def close(self) -> None:
        self.socket.close()


class UdpRobotTransport:
    """Robot/simulator-side endpoint with a fixed controller state destination."""

    def __init__(
        self,
        *,
        command_bind: tuple[str, int],
        controller_state_address: tuple[str, int],
    ):
        self.controller_state_address = controller_state_address
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(command_bind)
        self.socket.setblocking(False)

    @property
    def local_address(self) -> tuple[str, int]:
        host, port = self.socket.getsockname()[:2]
        return str(host), int(port)

    def receive_commands(self) -> list[JointCommand]:
        commands = []
        while True:
            try:
                payload, _ = self.socket.recvfrom(MAX_DATAGRAM_BYTES)
            except BlockingIOError:
                break
            message = decode_message(payload)
            if isinstance(message, JointCommand):
                commands.append(message)
        return commands

    def send_state(self, state: RawRobotState) -> None:
        self.socket.sendto(encode_message(state), self.controller_state_address)

    def close(self) -> None:
        self.socket.close()


def monotonic_time_ns() -> int:
    return time.monotonic_ns()
