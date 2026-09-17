import time

import numpy as np
import pytest

from locomotion.mujoco_robot_server import MujocoRobotServer
from locomotion.remote_controller import PolicyController, TerminalVelocityCommand
from locomotion.robot_profiles import LEGGYBOT_FIXED_ANKLE_PROFILE
from locomotion.tasks.skills import ANIMATED_REFERENCE_WALK_TASK_ID
from locomotion.robot_protocol import (
    JointCommand,
    RawRobotState,
    UdpControllerTransport,
    UdpRobotTransport,
    decode_message,
    encode_message,
)
from locomotion.state_estimation import Icm42688StateEstimator, PassthroughStateEstimator

DEFAULT_PROFILE = LEGGYBOT_FIXED_ANKLE_PROFILE


class ZeroPolicy:
    observation_dim = DEFAULT_PROFILE.observation_dim
    action_dim = DEFAULT_PROFILE.action_dim

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        assert observation.shape == (DEFAULT_PROFILE.observation_dim,)
        return np.zeros(DEFAULT_PROFILE.action_dim, dtype=np.float32)


def make_state(sequence: int = 0) -> RawRobotState:
    return RawRobotState(
        sequence=sequence,
        sample_time_ns=5_000_000,
        sent_time_ns=10_000_000,
        robot_name=DEFAULT_PROFILE.name,
        joint_names=DEFAULT_PROFILE.joint_names,
        joint_position=DEFAULT_PROFILE.default_joint_pos,
        joint_velocity=(0.0,) * DEFAULT_PROFILE.action_dim,
        imu_angular_velocity=(0.0, 0.0, 0.0),
        imu_linear_acceleration=(0.0, 0.0, 9.81),
        base_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )


def test_wire_messages_round_trip() -> None:
    state = make_state()
    command = JointCommand(
        sequence=3,
        created_time_ns=123,
        robot_name=DEFAULT_PROFILE.name,
        joint_names=DEFAULT_PROFILE.joint_names,
        position_target=DEFAULT_PROFILE.default_joint_pos,
        velocity_command=(0.1, 0.0, 0.02),
    )

    assert decode_message(encode_message(state)) == state
    assert decode_message(encode_message(command)) == command


def test_icm42688_estimator_uses_only_raw_six_axis_measurements() -> None:
    estimator = Icm42688StateEstimator()
    state = make_state()
    state = RawRobotState(
        sequence=state.sequence,
        sample_time_ns=0,
        sent_time_ns=state.sent_time_ns,
        robot_name=state.robot_name,
        joint_names=state.joint_names,
        joint_position=state.joint_position,
        joint_velocity=state.joint_velocity,
        imu_angular_velocity=(0.0, 0.0, 0.0),
        imu_linear_acceleration=(0.0, 0.0, 9.80665),
        # Deliberately absent: this is the real hardware contract.
        base_orientation_wxyz=None,
    )
    estimator.update(state)
    estimated = estimator.latest()
    assert estimated is not None
    np.testing.assert_allclose(estimated.base_angular_velocity, 0.0, atol=1e-6)
    np.testing.assert_allclose(
        estimated.projected_gravity, (0.0, 0.0, -1.0), atol=1e-5
    )


def test_icm42688_estimator_gates_non_gravitational_acceleration() -> None:
    estimator = Icm42688StateEstimator()
    template = make_state()
    for index in range(30):
        acceleration = (0.0, 0.0, 9.80665) if index == 0 else (19.6133, 0.0, 0.0)
        estimator.update(RawRobotState(
            sequence=index,
            sample_time_ns=index * 20_000_000,
            sent_time_ns=index * 20_000_000,
            robot_name=template.robot_name,
            joint_names=template.joint_names,
            joint_position=template.joint_position,
            joint_velocity=template.joint_velocity,
            imu_angular_velocity=(0.0, 0.0, 0.0),
            imu_linear_acceleration=acceleration,
            base_orientation_wxyz=None,
        ))
    estimated = estimator.latest()
    assert estimated is not None
    assert estimated.projected_gravity[2] < -0.98


def test_controller_feeds_back_clipped_action_before_range_scaling() -> None:
    class SaturatingPolicy:
        def __init__(self):
            self.observations = []

        def __call__(self, observation):
            self.observations.append(observation.copy())
            return np.array([100., -200., .3, -500., 600., -.4], dtype=np.float32)

    class MemoryTransport:
        def receive_states(self):
            return [make_state()]

    policy = SaturatingPolicy()
    controller = PolicyController(policy, MemoryTransport(), PassthroughStateEstimator(),
                                  state_timeout_s=1., action_range_fraction=.5)
    controller.receive_available_states(1_000_000_000)
    command = controller.make_command(np.zeros(3), 1_000_000_000)
    controller.make_command(np.zeros(3), 1_020_000_000)
    bounded = np.array([1., -1., .3, -1., 1., -.4])
    np.testing.assert_allclose(policy.observations[1][18:24], bounded)
    np.testing.assert_allclose(command.position_target,
        DEFAULT_PROFILE.action_target_from_normalized(bounded, .5))


def test_controller_derives_joint_velocity_from_quantized_positions() -> None:
    class RecordingPolicy:
        observation_dim = DEFAULT_PROFILE.observation_dim
        action_dim = DEFAULT_PROFILE.action_dim

        def __init__(self):
            self.observations = []

        def __call__(self, observation):
            self.observations.append(observation.copy())
            return np.zeros(self.action_dim, dtype=np.float32)

    class MemoryTransport:
        def __init__(self):
            self.states = []

        def receive_states(self):
            states, self.states = self.states, []
            return states

    transport = MemoryTransport()
    policy = RecordingPolicy()
    controller = PolicyController(
        policy, transport, PassthroughStateEstimator(), state_timeout_s=1.0
    )
    step = DEFAULT_PROFILE.position_resolution_rad
    assert step is not None
    first = make_state(sequence=0)
    transport.states.append(first)
    controller.receive_available_states(1_000_000_000)
    controller.make_command(np.zeros(3), 1_000_000_000)

    moved_position = np.asarray(first.joint_position) + 2.0 * step
    moved = RawRobotState(
        sequence=1,
        sample_time_ns=25_000_000,
        sent_time_ns=30_000_000,
        robot_name=first.robot_name,
        joint_names=first.joint_names,
        joint_position=tuple(moved_position),
        joint_velocity=(99.0,) * DEFAULT_PROFILE.action_dim,
        imu_angular_velocity=first.imu_angular_velocity,
        imu_linear_acceleration=first.imu_linear_acceleration,
        base_orientation_wxyz=first.base_orientation_wxyz,
    )
    transport.states.append(moved)
    controller.receive_available_states(1_020_000_000)
    controller.make_command(np.zeros(3), 1_020_000_000)

    np.testing.assert_array_equal(policy.observations[0][12:18], 0.0)
    np.testing.assert_allclose(
        policy.observations[1][12:18],
        0.5 * 2.0 * step / 0.02,
        rtol=1e-5,
    )


def test_udp_transport_is_bidirectional_and_asynchronous() -> None:
    controller = UdpControllerTransport(
        state_bind=("127.0.0.1", 0),
        robot_command_address=("127.0.0.1", 9),
    )
    robot = UdpRobotTransport(
        command_bind=("127.0.0.1", 0),
        controller_state_address=controller.local_address,
    )
    controller.robot_command_address = robot.local_address
    try:
        state = make_state()
        robot.send_state(state)
        assert controller.receive_states() == [state]

        command = JointCommand(
            sequence=0,
            created_time_ns=time.monotonic_ns(),
            robot_name=DEFAULT_PROFILE.name,
            joint_names=DEFAULT_PROFILE.joint_names,
            position_target=DEFAULT_PROFILE.default_joint_pos,
        )
        controller.send_command(command)
        assert robot.receive_commands() == [command]
    finally:
        controller.close()
        robot.close()


def test_controller_uses_latest_state_and_fails_safe_when_stale() -> None:
    class MemoryControllerTransport:
        def __init__(self):
            self.states = [make_state()]

        def receive_states(self):
            states, self.states = self.states, []
            return states

        def send_command(self, command):
            self.command = command

        def close(self):
            pass

    transport = MemoryControllerTransport()
    controller = PolicyController(
        ZeroPolicy(), transport, PassthroughStateEstimator(), state_timeout_s=0.1
    )
    now_ns = 1_000_000_000
    controller.receive_available_states(now_ns)
    command = controller.make_command(np.array([0.5, 0.0, 0.0]), now_ns)
    np.testing.assert_allclose(
        command.position_target,
        DEFAULT_PROFILE.action_target_from_normalized(
            np.zeros(DEFAULT_PROFILE.action_dim)
        ),
    )
    np.testing.assert_allclose(command.velocity_command, (0.5, 0.0, 0.0))

    stale = controller.make_command(np.zeros(3), now_ns + 200_000_000)
    np.testing.assert_allclose(
        stale.position_target,
        DEFAULT_PROFILE.action_target_from_normalized(
            np.zeros(DEFAULT_PROFILE.action_dim)
        ),
    )
    np.testing.assert_array_equal(controller.last_action, 0.0)


def test_terminal_velocity_keyboard_uses_point01_steps_and_point10_limit() -> None:
    command = TerminalVelocityCommand(
        (0.0, 0.0, 0.0), DEFAULT_PROFILE, enabled=False
    )
    for _ in range(10):
        assert command.apply_key("w")
    assert command.target[0] == pytest.approx(0.10)
    command.apply_key("w")
    assert command.target[0] == pytest.approx(0.10)
    command.apply_key("s")
    assert command.target[0] == pytest.approx(0.09)
    for _ in range(100):
        command.apply_key("q")
    assert command.target[2] == pytest.approx(np.pi / 4)
    command.apply_key("q")
    assert command.target[2] == pytest.approx(np.pi / 4)
    for _ in range(200):
        command.apply_key("e")
    assert command.target[2] == pytest.approx(-np.pi / 4)
    command.apply_key("x")
    assert tuple(command.target) == (0.0, 0.0, 0.0)


def test_controller_generates_the_same_gait_clock_as_training() -> None:
    class GaitPolicy:
        observation_dim = 29
        action_dim = 6
        task_id = ANIMATED_REFERENCE_WALK_TASK_ID

        def __init__(self):
            self.observations = []

        def __call__(self, observation):
            self.observations.append(observation.copy())
            return np.zeros(6, dtype=np.float32)

    class MemoryTransport:
        def __init__(self, state):
            self.states = [state]

        def receive_states(self):
            states, self.states = self.states, []
            return states

        def send_command(self, command):
            pass

        def close(self):
            pass

    profile = LEGGYBOT_FIXED_ANKLE_PROFILE
    state = RawRobotState(
        sequence=0,
        sample_time_ns=5_000_000,
        sent_time_ns=10_000_000,
        robot_name=profile.name,
        joint_names=profile.joint_names,
        joint_position=profile.default_joint_pos,
        joint_velocity=(0.0,) * profile.action_dim,
        imu_angular_velocity=(0.0, 0.0, 0.0),
        imu_linear_acceleration=(0.0, 0.0, 9.81),
        base_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    policy = GaitPolicy()
    controller = PolicyController(
        policy,
        MemoryTransport(state),
        PassthroughStateEstimator(),
        profile,
        state_timeout_s=1.0,
    )
    start = 1_000_000_000
    controller.receive_available_states(start)
    velocity = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    controller.make_command(velocity, start)
    controller.make_command(velocity, start + 82_500_000)

    np.testing.assert_allclose(policy.observations[0][-2:], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(policy.observations[1][-2:], [1.0, 0.0], atol=1e-6)


def test_controller_rejects_state_for_a_different_robot() -> None:
    class MemoryControllerTransport:
        def receive_states(self):
            state = make_state()
            return [
                RawRobotState(
                    sequence=state.sequence,
                    sample_time_ns=state.sample_time_ns,
                    sent_time_ns=state.sent_time_ns,
                    robot_name="go1",
                    joint_names=state.joint_names,
                    joint_position=state.joint_position,
                    joint_velocity=state.joint_velocity,
                    imu_angular_velocity=state.imu_angular_velocity,
                    imu_linear_acceleration=state.imu_linear_acceleration,
                    base_orientation_wxyz=state.base_orientation_wxyz,
                )
            ]

        def send_command(self, command):
            pass

        def close(self):
            pass

    controller = PolicyController(
        ZeroPolicy(), MemoryControllerTransport(), PassthroughStateEstimator()
    )
    with pytest.raises(ValueError, match="go1.*leggybot"):
        controller.receive_available_states(1_000_000_000)


def test_simulator_command_watchdog() -> None:
    class MemoryRobotTransport:
        def __init__(self):
            self.commands = []
            self.states = []

        def receive_commands(self):
            commands, self.commands = self.commands, []
            return commands

        def send_state(self, state):
            self.states.append(state)

        def close(self):
            pass

    transport = MemoryRobotTransport()
    server = MujocoRobotServer(transport, command_timeout_s=0.1)
    target = DEFAULT_PROFILE.array("default_joint_pos") + 0.05
    transport.commands.append(
        JointCommand(
            sequence=0,
            created_time_ns=1_000_000_000,
            robot_name=DEFAULT_PROFILE.name,
            joint_names=DEFAULT_PROFILE.joint_names,
            position_target=tuple(float(x) for x in target),
            velocity_command=(0.1, 0.0, 0.02),
        )
    )
    server.step(now_ns=1_000_000_000)
    quantized_target = DEFAULT_PROFILE.quantize_joint_position(target)
    np.testing.assert_allclose(server.position_profiler.goal, quantized_target)
    applied = server.data.ctrl[server.control_ids].copy()
    nominal = DEFAULT_PROFILE.quantize_joint_position(
        DEFAULT_PROFILE.default_joint_pos
    )
    assert np.all(applied > nominal)
    assert np.all(applied < quantized_target)
    np.testing.assert_allclose(server.last_velocity_command, (0.1, 0.0, 0.02))
    assert not server.timed_out
    assert transport.states
    assert transport.states[-1].base_orientation_wxyz is None
    observed = np.asarray(transport.states[-1].joint_position)
    resolution = DEFAULT_PROFILE.position_resolution_rad
    assert resolution is not None
    np.testing.assert_allclose(observed / resolution, np.rint(observed / resolution))

    server.step(now_ns=1_200_000_000)
    np.testing.assert_allclose(
        server.position_profiler.goal,
        nominal,
    )
    assert server.timed_out
