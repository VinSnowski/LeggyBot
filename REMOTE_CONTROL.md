# Decoupled simulation and robot controller

This architecture separates robot I/O from policy inference:

```text
             asynchronous RawRobotState (up to 200 Hz)
MuJoCo server --------------------------------------------> controller
or robot I/O                                               estimator
daemon        <-------------------------------------------- policy (50 Hz)
                  timestamped joint position commands
```

The controller never imports MuJoCo and does not know whether the peer is a
simulator or hardware. A future hardware I/O daemon must publish the same
`RawRobotState` and consume the same `JointCommand` contract.

## Run locally

NewLeggyBot (`leggybot-fixed-ankle-6dof`) is the default at both endpoints.
Start the native CPU MuJoCo process in one terminal and the controller in
another:

```bash
uv run python -m locomotion.mujoco_robot_server

uv run python -m locomotion.remote_controller \
  logs/rsl_rl/leggybot_skill_animated_reference_walk/<run>/model_4999.pt
```

Go1 remains available, but both endpoints must select it explicitly:

```bash
uv run python -m locomotion.mujoco_robot_server --robot go1

uv run python -m locomotion.remote_controller \
  logs/rsl_rl/go1_velocity/<run>/model_10000.pt --robot go1
```

`robot_profiles.py` is the single deployment registry for model construction,
joint and actuator ordering, dimensions, action scaling, limits, sensor names,
timing, command bounds, and safe nominal pose. `remote_controller.py`,
`mujoco_robot_server.py`, `play_native.py`, `robot_protocol.py`, and the state
estimator operate on that profile rather than embedding Go1 or LeggyBot constants.

Protocol v2 includes `robot_name` and `joint_names` in both directions. A robot
identity or joint-order mismatch is rejected before a command is applied. This is
intentional compatibility breakage from protocol v1; update simulator and
controller together.

The controller terminal accepts W/S for forward velocity, A/D for lateral
velocity, Q/E for yaw rate, and X to zero the command. For a fixed command or a
non-interactive process, use `--no-keyboard --vx 0.5`.

Defaults use UDP command port 15000, UDP state port 15001, 200 Hz physics/state
publication, and 50 Hz policy control. Hosts, ports, frequencies, rendering and
watchdog deadlines are configurable through `--help`. Bind to `127.0.0.1` rather
than `0.0.0.0` when external access is unnecessary.

## Frequency and clock model

The connection is asynchronous, not request/response:

- The robot owns its 200 Hz physics or motor loop and holds the latest joint
  command between controller updates.
- The robot publishes timestamped sensor samples independently, normally at
  200 Hz.
- The controller drains all available samples, updates its estimator in sample
  order, and runs the policy on the newest estimate at 50 Hz.
- Sequence numbers expose loss and reordering. Device sample time and local
  receive time are separate because clocks may not be synchronized.
- The controller emits a safe nominal pose when state is stale for 100 ms. The
  simulator independently applies its command watchdog after 100 ms.

UDP avoids head-of-line blocking: losing an old sample is preferable to waiting
for it while newer control data exists. On separate machines, synchronize clocks
with PTP when measured end-to-end latency or multi-sensor fusion depends on a
common clock. Never estimate latency by subtracting unsynchronized timestamps.

## State estimation

`PassthroughStateEstimator` is deliberately a placeholder. MuJoCo publishes
perfect base orientation plus IMU and joint sensor values; the placeholder uses
the orientation directly to calculate projected gravity. The policy controller
depends only on the `StateEstimator` interface.

A hardware estimator can replace it with an error-state EKF or invariant EKF
that consumes timestamped gyro, accelerometer, joint encoder, contact, and
optional external pose/velocity measurements. It should update at sensor rate
and provide a prediction at the policy evaluation timestamp. If the robot
firmware already publishes a trustworthy orientation/state estimate, an adapter
can populate the corrected state directly. Estimation mode and observation ABI
must match what was randomized and validated during training.

## Production assessment

The process boundary, independent rates, latest-state semantics, estimator
interface, timestamps, sequence counters and two-sided watchdogs are standard
professional robotics design choices. The current transport remains a prototype:

- UDP/JSON has no discovery, authentication, schema negotiation, QoS, redundancy,
  or delivery diagnostics beyond sequence gaps.
- Python is not a hard-real-time motor-control runtime.
- The nominal-pose watchdog is not an adequate hardware emergency stop.

For a real robot, retain the message semantics but normally move the hardware I/O
daemon and safety supervisor to C++ or Rust on a PREEMPT_RT system. Connect motors
over EtherCAT or CAN-FD, enforce position/velocity/torque/temperature/tilt limits
below the policy, and provide an independent physical emergency stop. DDS/ROS 2,
Zenoh, or a fixed binary shared-memory/UDP schema are suitable replacements when
their measured latency and jitter meet the control budget. The 1 kHz-class motor
servo belongs on the robot-side daemon; the learned policy can remain at 50 Hz.

Before hardware use, record raw state and command streams, replay them through
the controller offline, measure packet age and jitter, inject delay/loss, and
verify watchdog behavior. The controller-to-driver interface should remain
unchanged when switching from MuJoCo to the real robot.
