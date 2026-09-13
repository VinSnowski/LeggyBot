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

The nominal torque comes from Feetech's 12 V specification. The damping,
friction, armature, and moderate position gain come from published STS3215
MuJoCo models used by SO-101 and Open Duck. This is intentionally conservative
and is more realistic than the former ideal inspection servo, but it is not a
substitute for identifying LeggyBot's production servos.

Backlash is intentionally **not simulated** until an assembled LeggyBot joint
is bench-identified. A future model must reproduce measured target-to-output
hysteresis, including direction history, load, command rate, and the servo's
internal position response; a catalogue backlash angle alone is insufficient.
Bus/controller delay, encoder bias and sensor noise are randomized on the
existing sim-to-real schedule. Encoder quantization, voltage sag, motor-speed
variation, current limiting, and thermal protection still require batch
measurements and a hardware-control implementation; they cannot all be
represented faithfully by a single standard MJCF position actuator.

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
