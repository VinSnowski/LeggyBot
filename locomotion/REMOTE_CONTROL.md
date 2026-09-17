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
velocity, Q/E for yaw rate, and X to zero the command. Each keypress changes its
command by 0.01. Forward/lateral velocity is bounded to +/-0.10 m/s and yaw
rate to `+/-pi/4` rad/s. For a fixed command or a
non-interactive process, use `--no-keyboard --vx 0.08`.

Defaults use UDP command port 15000, UDP state port 15001, 200 Hz physics/state
publication, and 50 Hz policy control. Hosts, ports, frequencies, rendering and
watchdog deadlines are configurable through `--help`. The simulator prints a
body-frame velocity comparison at 5 Hz by default:

```text
body_velocity actual=[+0.032 -0.001 +0.000 ; +0.000 +0.002 +0.041] command=[+0.050 +0.000 +0.000 ; +0.000 +0.000 +0.000] error=[-0.018 -0.001 +0.000 ; +0.000 +0.002 +0.041]
```

The first three values are torso linear velocity `(x, y, z)` and the last
three are torso angular velocity `(x, y, z)`. The command is the controller's
keyboard target, with yaw placed in angular `z`; `error` is actual minus
command. Use `--velocity-print-hz 0` to disable it or choose another rate.
Bind to `127.0.0.1` rather than `0.0.0.0` when external access is unnecessary.

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

## Servo calibration and command timing

Policy position targets and servo feedback are software joint coordinates. The
hardware bridge must own one calibration per joint and use it in both paths:

```text
reported_angle = scale * physical_angle + zero_offset
physical_target = (quantized_software_target - zero_offset) / scale
```

For an STS3215 bridge, perform target quantization in servo-count space before
the inverse mapping, and convert feedback counts through the forward mapping
before publishing `joint_position`. A horn/zero error can therefore change the
physical pose even while target and feedback counts agree. Do not add a second
observation-only offset in the policy controller.

Training models the six-servo synchronous write as one ordered command FIFO.
At the 200 Hz physics rate its base latency is 5--15 ms, packet jitter is
correlated and usually within 5 ms, and rare commands arrive one additional
20 ms policy frame late. The hardware bridge should timestamp command receipt
and actuator application when possible; these provisional ranges should be
replaced with measured bus/controller timing without changing the policy API.

## STS3215 firmware position profile

The hardware daemon must configure every servo explicitly at startup. Use
`locomotion.servo.configure_sts3215_like_lerobot` with the connected LeRobot
`FeetechMotorsBus`; it writes and verifies this contract:

```text
Operating_Mode       = 0    # bounded position mode
P_Coefficient        = 16
I_Coefficient        = 0
D_Coefficient        = 32
Return_Delay_Time    = 0
Maximum_Acceleration = 254
Acceleration         = 254
Goal_Velocity        = 0    # firmware maximum
Goal_Time            = 0
Phase bit 4           = 0    # bounded position mode
```

The PID values are LeRobot's current SO follower defaults. They make the
hardware behavior deterministic, but the mapping from firmware coefficients
to LeggyBot's effective joint stiffness and damping remains a pre-hardware
approximation. Treat any future PID change as an actuator-model change: update
the physical gains or identify a replacement model and retrain the policy.

Continue sending all six goal positions in one synchronous write. Speed and
acceleration are fixed servo configuration, not policy outputs and therefore
do not belong in `JointCommand`. Reapply and verify the SRAM-backed values
after reconnect or power-up; do not rely on a servo's previous state.

Simulation applies the same nominal firmware profile after the modeled packet
arrives: 3400 counts/s maximum velocity and 25,400 counts/s^2 acceleration,
equivalent to 5.216 rad/s and 38.96 rad/s^2 at 4096 counts/revolution. The
profile's setpoint then drives the existing MuJoCo motor model. Training varies
the profile limits across servos, so a policy cannot depend on every unit
having exactly the nominal response. These conversions are pre-hardware
approximations; record commanded and reported positions at 200 Hz once the
servos arrive, fit the actual ramp, and update the shared constants in
`locomotion/servo.py`.

## State estimation

`Icm42688StateEstimator` consumes only raw gyro and accelerometer samples and
produces the same filtered angular velocity and projected-gravity observations
used during training. `base_orientation_wxyz` is optional diagnostic ground
truth and is ignored by the production controller. `PassthroughStateEstimator`
remains available only for tests and diagnostics. MuJoCo can publish
perfect base orientation for diagnostics, but the simulated robot server now
omits it to exercise the hardware-compatible path. The policy controller
depends only on the `StateEstimator` interface.

The six-axis filter cannot observe absolute yaw. If a later controller needs
world position, heading, or velocity, add contact/kinematic and external
measurements through an error-state or invariant EKF and retrain against that
observation pipeline.

The hardware bridge must configure the ICM-42688-P for +/-500 deg/s and +/-16 g
and provide one coherent sample every 20 ms. Gyroscope values on the wire are
body-frame rad/s; accelerometer values are body-frame m/s^2 and read about
`[0, 0, +9.80665]` while the upright robot is stationary. Timestamp samples at
acquisition, map the board axes into the robot body frame before fusion, and
use a digital bandwidth near 20-25 Hz. If the sensor runs faster internally,
read its FIFO and downsample coherently instead of sending an arbitrary latest
register from each axis.

Wire protocol v3 carries `joint_sample_time_ns` and `imu_sample_time_ns`
separately. A hardware bridge should place the newest complete servo-bus sample
and newest complete IMU sample in each state packet, preserving their original
acquisition timestamps. The controller intentionally uses latest-wins data at
its 50 Hz policy boundary. Training independently randomizes servo and IMU age
between zero and one 20 ms policy frame while keeping all outputs from one
physical device mutually coherent.

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
