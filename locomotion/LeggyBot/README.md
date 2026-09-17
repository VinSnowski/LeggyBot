# LeggyBot simulation model

`urdf/LeggyBot.urdf` is the SolidWorks export with the requested joint limits.
`mjcf/LeggyBot.xml` is the native MuJoCo training model with a floating base,
visual STL meshes, simplified collision geometry, a floor, and IMU sensors.

The root coordinate frame follows the locomotion convention: +X forward, +Y left,
and +Z up. The SolidWorks mesh frame and child transforms are rotated at the root,
with the base inertia tensor transformed into that same basis.

Open it with:

```bash
uv run python -m locomotion.view_cad_robot
```

For joint-angle inspection, weld the floating torso to the world and display
the joint-axis markers:

```bash
uv run python -m locomotion.view_cad_robot --fixed-base
```

Use the six sliders in MuJoCo's **Control** panel to pose the joints. The weld
is disabled unless `--fixed-base` is supplied, so normal simulation and future
training retain the floating base.

Validate it headlessly with:

```bash
uv run python -m locomotion.view_cad_robot --headless --seconds 10
```

## Current fixed-ankle model

The current six-DOF task loads `../NewLeggyBot/mjcf/NewLeggyBot.xml`. That model
matches `LeggyBot.xml` outside its separate shin and fixed angled-foot bodies. A
-8-degree nominal hip-pitch offset keeps the torso upright and both soles flat.

Inspect the standalone simulated robot with:

```bash
uv run python -m locomotion.mujoco_robot_server --robot leggybot-fixed-ankle-6dof
```

Train any implemented skill explicitly, for example:

```bash
uv run python -m locomotion.train_rsl_rl \
  --task Mjlab-Stand-LeggyBot-FixedAnkle6DOF \
  --num-envs 4096 --iterations 1500 --run-name stand
```

Because the CAD export combines each shin and foot into one mesh, the displayed
STL cannot show the physical ankle bend. Physics uses the rotated primitive foot
collision and foot sensor site; the visual mesh is only an approximation. The
local CAD-derived shin/foot inertia is deliberately unchanged. That is appropriate
for this feasibility experiment, but separate shin and foot mass properties must
be recombined at the fixed transform before final sim-to-real training.

Stepping uses a 0.8-second alternating gait clock and a 20 mm swing target.
The short-cycle walking experiments use a 0.33-second per-leg cycle: one
follows a shallow knee/thigh animation, one biases toward retained bilateral
contact with a 3 mm clearance guard, and the unloading transition targets a
75/25 support-to-swing load split with larger thigh/knee movement. None rewards
a high-stepping gait. The subsequent periodic true-walking task removes all
joint references: it softly penalizes force beneath the scheduled swing foot
and horizontal motion of the scheduled stance foot, while requiring actual
world-frame forward displacement. Skill promotion is based on each task's
evaluator rather than an all-in-one gait curriculum.

Normalized actions use an asymmetric piecewise-affine mapping: zero is the
nominal joint pose, while -1 and +1 reach each joint limit with a 0.035 rad
margin. Current skills expose this full safe span.

## Joint ranges

| Joint | Minimum | Maximum |
|---|---:|---:|
| RightBaseToHip | -10 degrees (inward) | +45 degrees (outward) |
| LeftBaseToHip | -45 degrees (outward) | +10 degrees (inward) |
| RightHipToThigh | -60 degrees | +60 degrees |
| LeftHipToThigh | -60 degrees | +60 degrees |
| RightKnee | -120 degrees (flexion) | +15 degrees (hyperextension) |
| LeftKnee | -120 degrees (flexion) | +15 degrees (hyperextension) |

## STS3215 actuator baseline

The MJCF uses a pre-hardware baseline for the 12 V, 1:345 STS3215-C018:

| Parameter | Value |
|---|---:|
| Nominal stall torque | 2.942 N·m |
| No-load speed (datasheet) | 4.712 rad/s (45 RPM) |
| Joint damping | 0.60 N·m·s/rad |
| Coulomb friction | 0.052 N·m |
| Reflected armature | 0.028 kg·m² |
| MuJoCo position gain | 17.8 N·m/rad |
| Encoder resolution | 4096 counts/revolution |
| Datasheet backlash | at most 0.5 degrees |
| Firmware goal velocity | 0 (firmware maximum) |
| Firmware acceleration | 254 (maximum register value) |
| Modeled profile speed ceiling | 5.216 rad/s (3400 counts/s) |
| Modeled profile acceleration | 38.96 rad/s^2 (25,400 counts/s^2) |

The nominal torque comes from Feetech's 12 V specification. The damping,
friction, armature, and moderate position gain come from published STS3215
MuJoCo models used by SO-101 and Open Duck. This is intentionally conservative
and is more realistic than the former ideal inspection servo, but it is not a
substitute for identifying LeggyBot's production servos.

The sim-to-real path models backlash with a randomized stateful play operator.
Bus/controller delay, servo calibration, sensor noise, motor strength, speed,
friction, armature, and stiffness are randomized on the existing schedule.
Servo calibration is one shared invertible mapping: the software coordinate is
`scale * physical_angle + zero_offset`. Outgoing software targets are first
quantized to the STS3215's 4096-count register and then mapped to the physical
joint. Physical feedback is mapped forward into the software coordinate before
being quantized. At full randomization, zero offsets are within +/-1 degree and
angle scale is within +/-0.5%. MuJoCo dynamics remain continuous.

Command timing uses one ordered FIFO for the synchronized six-servo packet.
The episode base delay is 1--3 physics ticks (5--15 ms at 200 Hz), normally has
at most one tick of correlated packet jitter, and has a 1% full-strength chance
per policy command of one additional 20 ms late frame. Packets cannot overtake
one another. The old per-actuator delay buffers are disabled so latency is not
applied twice and one packet cannot reach paired joints at different times.

After a packet reaches a servo, training advances an acceleration- and
velocity-limited internal position setpoint instead of applying the new target
instantaneously. This starts from LeRobot's STS3215 setup: position mode uses
P/I/D coefficients 16/0/32, return delay is zero, both acceleration registers
are 254, and phase bit 4 is cleared. LeggyBot also
pins goal velocity and goal time to zero so the position command is unlimited
by either a stale speed or time setting. The profile runs at every 5 ms physics
step. Its nominal register-derived limits
are randomized by +/-5% for speed and +/-20% for acceleration; paired domain
randomization shares those samples between mirrored joints, while full domain
randomization samples each servo independently. The MuJoCo actuator still
models the motor and load response beneath this firmware setpoint, so its
4.712 rad/s no-load speed remains a separate physical limitation.

`locomotion.servo.configure_sts3215_like_lerobot` writes and reads back this
register contract through LeRobot's Feetech bus API. A hardware daemon must run
it at startup before enabling torque and must fail safe if verification does
not match. The register-unit conversion is a provisional pre-hardware model;
replace its speed and acceleration constants with measured motion from the
production servos without changing the policy interface.

Actor joint velocity is derived from the same noisy, quantized
policy-rate position samples with a 20 ms backward difference and a causal EMA
(`alpha=0.5`), both in training and deployment; the policy does not consume MuJoCo's exact joint
velocity or the servo's proprietary speed register. Voltage sag, current
limiting, and thermal protection still need
batch measurements and a hardware-control implementation; they cannot all be
represented faithfully by a single standard MJCF position actuator.

The policy IMU path targets an ICM-42688-P configured for +/-500 deg/s gyro,
+/-16 g acceleration, and 50 Hz fusion output. Training models the device's
ADC resolution, specified noise, bias and scale tolerances, cross-axis error,
mounting misalignment, slow bias drift, and additional servo vibration. A
causal complementary filter propagates gravity with the gyro, rejects
accelerometer correction during impacts, and estimates observable roll/pitch
gyro bias. A device-level observation-age model independently selects a current
or one-frame-old sample for the servo bus and IMU, retaining each lag state with
90% probability from one policy step to the next. Joint position/velocity stay
coherent, as do gyro/projected gravity. Deployment uses the same filter and
does not require a base quaternion. Absolute yaw is not observable from a
6-axis IMU and is intentionally not estimated.

## Foot and ground sim-to-real baseline

Reset randomization also models uncertainty in the printed feet, add-on pads,
contact solver, and floor. Only the two foot collision geoms are perturbed; the
render meshes, foot sites, joints, link masses, and inertias are unchanged by
this part of the model.

| Effect | Full-strength range |
|---|---:|
| Sole mounting roll and pitch | +/-1 degree |
| Sole collision height | +/-0.5 mm |
| Normal-contact time constant | 10--15 ms |
| Normal-contact damping ratio | 0.7--1.3 |
| Compliant contact width | 0.5--2.0 mm |
| Tangential friction | 0.5--1.2 |
| Torsional friction | 0.6--1.4 times nominal |
| Rolling friction | 0.6--1.4 times nominal |
| Floor roll and pitch | +/-1.5 degrees |

The 10 ms lower contact time constant is intentional. MuJoCo's default
reference-safety rule requires at least twice the 5 ms physics step, so a
nominal 5 ms sample would be clamped and would not provide the intended
variation. Contact softness and all three friction components are shared by
the two feet within one environment, matching two pads cut from the same
material. Floor slope and contact material are randomized from the beginning.

During paired domain randomization, sole height and pitch match and sole roll
is mirrored across the sagittal plane. During the independent-randomization
ramp, collision poses blend continuously toward independently sampled
right/left mounting errors. At full randomization, each sole pose is
independent. Setting `domain_randomization_scale=0` or disabling sim-to-real
randomization restores every affected contact and collision field exactly to
the MJCF default.

## CAD frame correction

The exported `RightBaseToHip` frame was located on a mounting hole rather than
at the center of the four-hole pattern. It is corrected by +4.95 mm in base X
and +4.95 mm in base Z (7.00 mm diagonally). The right-hip mesh, inertia,
collision geometry, and child-joint coordinates are counter-shifted so this
changes the rotation axis without changing the neutral assembly pose.

Both exported knee frames were located at the bottom-front mounting hole. Each
knee is corrected by +4.95 mm in thigh Y and +4.9497 mm in thigh Z (7.00 mm
diagonally), placing its origin at `(0, -0.1646, 0)`. Each shin/foot mesh, COM,
collision shape, and foot site is counter-shifted, preserving the neutral pose
and the CAD-derived inertia tensor about the COM.
