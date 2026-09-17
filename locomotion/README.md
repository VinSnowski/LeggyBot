# LeggyBot locomotion manual

For the forward/lateral/curve tracking baseline, see
[omnidirectional diagnostics and SE2TrackingWalkingTask](../OMNI_WALKING_DIAGNOSTICS.md).
It includes measured v2 failures, the new task design, and training/evaluation commands.
The current task is `NaturalOmnidirectionalWalkingTaskV3`. It ports the proven
Natural reward to mutually exclusive standing, forward, lateral,
forward-plus-yaw, and backward commands. It adds stronger pitch-upright
shaping and a small double-support lateral foot-line cost, while preserving
Natural's PPO settings:

```bash
uv run python -m locomotion.train_rsl_rl \
  --task NaturalOmnidirectionalWalkingTaskV3 --num-envs 4096 \
  --iterations 1500 --run-name natural-v3-from-zero
```

The command proportions are 10% stop, 30% forward, 10% backward, 20% lateral,
and 30% curved forward. During pure lateral commands the foot-line penalty
discourages scissoring without constraining the swing phase. The task is
intended to train from a blank policy.

V3 pairs servo gain, effort-limit, and calibration samples through
iteration 500. It restores nominal leg inertias on each reset and varies torso
mass, central inertia and fore/aft/vertical COM without changing lateral COM.
From iterations 500–800 it interpolates physical pseudo-inertia matrices and
continuous servo parameters toward full independent samples. Command delay
remains common to all six servos because one synchronized packet owns it; its
episode baseline and runtime jitter remain randomized throughout. From
iteration 800 onward resets use full independent servo and leg randomization.
Episodes retain physical/calibration samples until reset.
`independent_ramp_steps=0` keeps the optional hard switch;
`full_domain_randomization` remains directly callable.

Signed TensorBoard metrics include zeros for inactive modes. For a conditional
left/right success estimate, divide the averaged signed `tracking_success` by
the averaged matching `command_fraction`; use the same ratio for signed
`foot_line_error_m`. Omit intervals whose command fraction is zero. These
ratios retain TensorBoard's episode weighting. Raw signed metrics are not
conditional success rates or conditional mean foot separation.

The Huber lateral cost remains unsaturated for large errors. Its larger scale
means it penalizes small offsets less strongly than the earlier exponential
cost; the larger reward weight does not make it uniformly stronger. This
shaping choice is retained while testing the corrected randomization.

The paper-style walking ablation is `PeriodicRewardCompositionWalkingTask`.
It trains one MLP policy from zero on mutually exclusive forward and lateral
commands: 45% forward, 45% lateral (left or right), and 10% stopped.  Its
sampler always sets yaw to zero and never makes both `vx` and `vy` nonzero.
Gait shaping consists only of smooth phase-weighted swing-foot force and
stance-foot velocity costs, following Siekmann et al.'s periodic reward
composition.  It intentionally has no scheduled-contact bonus, joint-motion
reference, curve command, or path-displacement reward.  The two bounded
velocity-error terms retain the paper kernel with a larger `-1.20` weight.

```bash
uv run python -m locomotion.train_rsl_rl \
  --task PeriodicRewardCompositionWalkingTask --num-envs 4096 \
  --iterations 1600 --run-name prc-walking-velocity-1p2-from-zero
```

Evaluate its four relevant modes with:

```bash
uv run python -m locomotion.evaluate_omni_run \
  logs/rsl_rl/leggybot_periodic_reward_composition_walking/<run-directory>
```

Watch any compatible checkpoint in raw CPU MuJoCo with a fixed 55-second,
50 Hz sequence covering stand, forward, backward, both lateral directions,
and forward-plus-yaw:

```bash
uv run python -m locomotion.watch_policy_sequence \
  logs/rsl_rl/<experiment>/<run>/model_<iteration>.pt
```

The task keeps the deployment-compatible 29-value observation and six-position
action ABI.  Its MLP and fixed servo gains are the first controlled reward
ablation; recurrent inference and policy-controlled PD gains are outside this
experiment.

Evaluate and rank every saved checkpoint in one omnidirectional run with:

```bash
uv run python -m locomotion.evaluate_omni_run \
  logs/rsl_rl/leggybot_natural_omnidirectional_walking_v3/<run-directory>
```

The command writes JSON and Markdown reports under `artifacts/omni_diagnostics`.
For V3 it evaluates stop, forward, backward, left, right, and both signed
forward-plus-yaw curves. It also checks the lateral foot-line objective in both
lateral modes. The paper-style axis-exclusive task requires stop, forward,
left, and right. All moving modes include stability, drift, orientation, and
bilateral-step checks.

This is the operational guide for the MuJoCo model, PPO training, policy
evaluation, native simulation, and controller interface. The project has one
rule: a training run receives a task ID; the task owns its environment, reward,
success metric, PPO configuration, and checkpoint metadata. The two PPO
trainers therefore use the same robot/task definition.

## Current robot status

NewLeggyBot is the global default as `leggybot-fixed-ankle-6dof`. It has three
actuated joints per leg and a fixed angled foot. Training commands default to
its standing task, while the other task IDs select its remaining skills.

`leggybot-8dof` remains an explicitly selectable future target that fails
clearly until its URDF/MJCF, joint names, limits, inertias, and actuator settings
are added.

## Install

Training with mjlab is intended for an NVIDIA GPU.

```bash
uv sync --dev
```

Run all commands below from the repository root. Use `uv run` rather than a
system Python interpreter.

## Inspect the current robot

Open the native MuJoCo model:

```bash
uv run python -m locomotion.view_cad_robot
```

For joint-axis and pose inspection with the floating base welded to the world:

```bash
uv run python -m locomotion.view_cad_robot --fixed-base
```

For a non-graphical dynamics smoke test:

```bash
uv run python -m locomotion.view_cad_robot --headless --seconds 10
```

The model details, joint limits, CAD-frame corrections, and current STS3215
baseline are documented in [LeggyBot/README.md](LeggyBot/README.md).

## Tasks and promotion sequence

All tasks share one actor ABI, so an actor can be initialized from the previous
successful policy. The critic, optimizer, and exploration state are always new
for the next task.

| Order | Task ID | What it validates | Evaluation pass condition |
| --- | --- | --- | --- |
| 1 | `Mjlab-Stand-LeggyBot-FixedAnkle6DOF` | Stable standing under reset randomization, noisy observations, and small pushes. | Stillness ≥ 0.85 and survival ≥ 0.90. |
| 2 | `Mjlab-WeightTransfer-LeggyBot-FixedAnkle6DOF` | Safe left/right load transfer without stepping. | Transfer score ≥ 0.75 and survival ≥ 0.90. |
| 3 | `Mjlab-StepInPlace-LeggyBot-FixedAnkle6DOF` | Repeated alternating contacts and a bounded periodic motion. | Step score ≥ 0.50 and survival ≥ 0.90. |
| 4A | `Mjlab-AnimatedReferenceWalk-LeggyBot-FixedAnkle6DOF` | Fixed 0.08 m/s walk guided by a shallow 0.33 s joint animation. | Forward-speed-and-alternating-contact score ≥ 0.70 and survival ≥ 0.90. |
| 4B | `Mjlab-ConservativeWalk-LeggyBot-FixedAnkle6DOF` | Fixed 0.08 m/s, 0.33 s contact-biased shuffle with retained bilateral support where possible. | Forward-speed-and-bilateral-contact score ≥ 0.70 and survival ≥ 0.90. |
| 5 | `Mjlab-UnloadingWalk-LeggyBot-FixedAnkle6DOF` | 0.08 m/s low-clearance transition that shifts 75/25 load support and engages thigh/knee pitch. | Forward-speed-and-unloading score ≥ 0.65 and survival ≥ 0.90. |
| 6 | `Mjlab-PeriodicWalk-LeggyBot-FixedAnkle6DOF` | True 0.08 m/s walking: smooth phase-weighted swing unloading, stance-foot anchoring, world-frame travel, and foot exchange. | Periodic-walk score ≥ 0.65 after a 2 s warm-up and survival ≥ 0.90. |
| 6S | `Mjlab-StraightPeriodicWalk-LeggyBot-FixedAnkle6DOF` | Minimal correction for a periodic walker that curves: world-forward rewards are gated by heading plus net world-Y and yaw accumulated across a full 0.33 s cycle. Normal within-cycle CoM transfer is free; only unusually large instantaneous rates incur a bounded safety cost. | Same periodic-walk score ≥ 0.65 after a 2 s warm-up and survival ≥ 0.90; inspect `cycle_straightness_gate`, `cycle_lateral_velocity_cost`, and `cycle_yaw_rate_cost`. |
| 6P | `Mjlab-SymmetricCyclePeriodicWalk-LeggyBot-FixedAnkle6DOF` | Symmetry-focused pretraining with the same cycle-level straightness rewards as 6S, paired left/right randomization, and mirrored PPO updates. | Same periodic-walk score ≥ 0.65 after a 2 s warm-up and survival ≥ 0.90; inspect `cycle_straightness_gate`, lateral drift, and bilateral foot exchange. |
| 6R | `Mjlab-SymmetricPeriodicWalk-LeggyBot-FixedAnkle6DOF` | Symmetric correction run using cycle-level net lateral/yaw drift for forward-credit gating, while retaining a weak instantaneous motion-safety bound. | Same periodic-walk score ≥ 0.65 after a 2 s warm-up and survival ≥ 0.90; also inspect `cycle_straightness_gate`, cycle lateral-drift cost, and cycle yaw-rate cost. |
| 7 | `Mjlab-Sagittal-LeggyBot-FixedAnkle6DOF` | Forward/backward speed tracking, acceleration, and braking. | Speed score ≥ 0.70 and survival ≥ 0.90. |

### Independent default walking task

`UpdatedDefaultWalkingTask` is a standalone NewLeggyBot task, not a stage in the
promotion sequence above. It keeps the existing actor inputs and action space,
starts its velocity curriculum at 0.075–0.085 m/s and expands it to 0–0.1 m/s
over 12,000 environment steps. It can start from the 0.08 m/s policy with
`--initialize-from`; when that option is omitted it starts from a fresh policy.
It pairs left/right randomization during the first 12,000 environment steps, and
then enables the full sim-to-real randomization envelope. Its 60-second episodes evaluate
unwrapped heading and accumulated lateral drift, while its walking rewards use
cycle-level drift, a ramped bounded path-deviation penalty, and scheduled contact
rather than instantaneous lateral-velocity suppression.

Start it from an empty policy with:

```bash
TASK=UpdatedDefaultWalkingTask
uv run python -m locomotion.train_rsl_rl \
  --task "$TASK" --iterations 4000 --run-name updated-default-walking
```

`UpdatedDefaultLateralWalkingTask` continues the completed default walker into
left/right and diagonal velocity commands. It starts with the final forward
range `[0, 0.1]` m/s, expands `vy` from zero to `[-0.1, 0.1]` m/s over 24,000
environment steps, and continues resampling commands every 4–8 seconds. Its
cycle and long-horizon path errors integrate every command segment. At `vy=0`,
the lateral terms reduce to the completed default task's straight-walking
terms; 35% of sampled commands retain exactly zero lateral velocity.

Initialize it from the completed forward curriculum:

```bash
TASK=UpdatedDefaultLateralWalkingTask
CHECKPOINT=logs/rsl_rl/leggybot_updated_default_walking/2026-09-13_06-56-49_updated-default-walking-velocity-curriculum/0_010_checkpoint.pt

uv run python -m locomotion.train_rsl_rl \
  --task "$TASK" --iterations 4000 \
  --run-name updated-default-lateral-walking \
  --initialize-from "$CHECKPOINT"
```

Tasks 4A and 4B are alternatives initialized from the same successful stepping
policy. Compare their evaluator result and native simulation; promote only the
one that truly alternates legs without a twitching or dragging exploit. The
values above are promotion gates, not a substitute for observing the policy in
simulation.

The promotion tasks use the same active sim-to-real envelope: collision-only
sole mounting and height error, compliant-contact dynamics, three-axis foot
friction, small floor slopes, mass/inertia, actuator PD gains and effort limits,
shared command/feedback servo calibration, an STS3215 firmware
velocity/acceleration profile, 5–15 ms base
command latency with correlated jitter, IMU/joint observation noise, and small
pushes. `UpdatedDefaultWalkingTask`
uses paired left/right actuator and encoder variation for its first 12,000 steps,
then the same full envelope. The current backlash placeholder is deliberately
disabled until an assembled joint is measured.

## Train any task

RSL-RL is the reference trainer. Replace `TASK`, `ITERATIONS`, and `RUN_NAME`
as needed:

```bash
TASK=Mjlab-Stand-LeggyBot-FixedAnkle6DOF
ITERATIONS=1500
RUN_NAME=stand

uv run python -m locomotion.train_rsl_rl \
  --task "$TASK" --iterations "$ITERATIONS" --run-name "$RUN_NAME"
```

For a smaller smoke test, add for example `--num-envs 256 --iterations 5`.
Production runs use the trainer default of 4096 environments unless an explicit
`--num-envs` is supplied.

The repository's compact PPO uses the identical environment and supports the
same task interface. Its default is 1024 environments:

```bash
uv run python -m locomotion.train_custom_ppo \
  --task "$TASK" --iterations "$ITERATIONS" --run-name "$RUN_NAME"
```

Both trainers write task-tagged `.pt` checkpoints. Custom PPO also attempts an
ONNX export; ONNX is an inference format, not a different policy or a training
method.

### Start a task from a successful policy

Use `--initialize-from` to copy only the compatible actor and its observation
normalization. This is the intended way to progress through the validation
sequence. It is not resume: use `--resume` only to continue the exact same
custom-PPO run with its critic and optimizer state.

```bash
SOURCE=logs/rsl_rl/leggybot_skill_stand/<timestamped-run>/model_1499.pt

uv run python -m locomotion.train_rsl_rl \
  --task Mjlab-WeightTransfer-LeggyBot-FixedAnkle6DOF \
  --iterations 2000 --run-name transfer \
  --initialize-from "$SOURCE"
```

For custom PPO, the source checkpoint must also be a custom-PPO checkpoint:

```bash
uv run python -m locomotion.train_custom_ppo \
  --task Mjlab-WeightTransfer-LeggyBot-FixedAnkle6DOF \
  --iterations 2000 --run-name transfer \
  --initialize-from logs/custom_ppo/leggybot_skill_stand/<run>/checkpoints/model_1500.pt
```

Actor transfer requires matching observation and action dimensions. It will fail
instead of copying a policy into an incompatible robot or task ABI.

## Recommended complete validation run

The following commands are the normal sequence. Replace every `<run>` with the
timestamped directory produced by the previous command. Each checkpoint name is
the final RSL-RL checkpoint for the requested iteration count.

```bash
export TASK_STAND=Mjlab-Stand-LeggyBot-FixedAnkle6DOF
export TASK_TRANSFER=Mjlab-WeightTransfer-LeggyBot-FixedAnkle6DOF
export TASK_STEP=Mjlab-StepInPlace-LeggyBot-FixedAnkle6DOF
export TASK_ANIMATED=Mjlab-AnimatedReferenceWalk-LeggyBot-FixedAnkle6DOF
export TASK_CONSERVATIVE=Mjlab-ConservativeWalk-LeggyBot-FixedAnkle6DOF
export TASK_UNLOADING=Mjlab-UnloadingWalk-LeggyBot-FixedAnkle6DOF
export TASK_PERIODIC=Mjlab-PeriodicWalk-LeggyBot-FixedAnkle6DOF
export TASK_STRAIGHT_PERIODIC=Mjlab-StraightPeriodicWalk-LeggyBot-FixedAnkle6DOF
export TASK_SYMMETRIC_CYCLE=Mjlab-SymmetricCyclePeriodicWalk-LeggyBot-FixedAnkle6DOF
export TASK_SYMMETRIC_PERIODIC=Mjlab-SymmetricPeriodicWalk-LeggyBot-FixedAnkle6DOF
export TASK_SAGITTAL=Mjlab-Sagittal-LeggyBot-FixedAnkle6DOF

# Focused three-task cycle/straightness pipeline:
# 1. Fresh symmetry-focused pretraining; keep the cycle-level straightness gate.
uv run python -m locomotion.train_rsl_rl --task "$TASK_SYMMETRIC_CYCLE" --iterations 4000 --run-name symmetric-cycle-pretrain
uv run python -m locomotion.evaluate_task logs/rsl_rl/leggybot_skill_symmetric_cycle_periodic_walk/<run>/model_3999.pt --task "$TASK_SYMMETRIC_CYCLE"

# 2. Full sim-to-real randomization and final straight-periodic adaptation.
uv run python -m locomotion.train_rsl_rl --task "$TASK_STRAIGHT_PERIODIC" --iterations 4000 --run-name straight-periodic-final --initialize-from logs/rsl_rl/leggybot_skill_symmetric_cycle_periodic_walk/<run>/model_3999.pt
uv run python -m locomotion.evaluate_task logs/rsl_rl/leggybot_skill_straight_periodic_walk/<run>/model_3999.pt --task "$TASK_STRAIGHT_PERIODIC"

# 3. Optional fallback only if the primary pipeline remains asymmetric.
# Start this task fresh; do not initialize it from an asymmetric gait.
uv run python -m locomotion.train_rsl_rl --task "$TASK_SYMMETRIC_PERIODIC" --iterations 6000 --run-name symmetric-periodic-fallback
uv run python -m locomotion.evaluate_task logs/rsl_rl/leggybot_skill_symmetric_periodic_walk/<run>/model_5999.pt --task "$TASK_SYMMETRIC_PERIODIC"

# 1. Stand — maximum 1,500 iterations.
uv run python -m locomotion.train_rsl_rl --task "$TASK_STAND" --iterations 1500 --run-name stand
uv run python -m locomotion.evaluate_task logs/rsl_rl/leggybot_skill_stand/<run>/model_1499.pt --task "$TASK_STAND"

# 2. Weight transfer — maximum 2,000 iterations.
uv run python -m locomotion.train_rsl_rl --task "$TASK_TRANSFER" --iterations 2000 --run-name transfer --initialize-from logs/rsl_rl/leggybot_skill_stand/<run>/model_1499.pt
uv run python -m locomotion.evaluate_task logs/rsl_rl/leggybot_skill_transfer/<run>/model_1999.pt --task "$TASK_TRANSFER"

# 3. Step in place — maximum 4,000 iterations.
uv run python -m locomotion.train_rsl_rl --task "$TASK_STEP" --iterations 4000 --run-name step-in-place --initialize-from logs/rsl_rl/leggybot_skill_transfer/<run>/model_1999.pt
uv run python -m locomotion.evaluate_task logs/rsl_rl/leggybot_skill_step/<run>/model_3999.pt --task "$TASK_STEP"

# 4A. Animated-reference walk — maximum 5,000 iterations.
uv run python -m locomotion.train_rsl_rl --task "$TASK_ANIMATED" --iterations 5000 --run-name animated-walk --initialize-from logs/rsl_rl/leggybot_skill_step/<run>/model_3999.pt
uv run python -m locomotion.evaluate_task logs/rsl_rl/leggybot_skill_animated_reference_walk/<run>/model_4999.pt --task "$TASK_ANIMATED"

# 4B. Conservative contact-biased walk — maximum 5,000 iterations.
uv run python -m locomotion.train_rsl_rl --task "$TASK_CONSERVATIVE" --iterations 5000 --run-name conservative-walk --initialize-from logs/rsl_rl/leggybot_skill_step/<run>/model_3999.pt
uv run python -m locomotion.evaluate_task logs/rsl_rl/leggybot_skill_conservative_walk/<run>/model_4999.pt --task "$TASK_CONSERVATIVE"

# 5. Low-clearance unloading walk — maximum 5,000 iterations.
uv run python -m locomotion.train_rsl_rl --task "$TASK_UNLOADING" --iterations 5000 --run-name unloading-walk --initialize-from logs/rsl_rl/leggybot_skill_conservative_walk/<run>/model_4999.pt
uv run python -m locomotion.evaluate_task logs/rsl_rl/leggybot_skill_unloading_walk/<run>/model_4999.pt --task "$TASK_UNLOADING"

# 6. True periodic walking — maximum 6,000 iterations.
# Start from stand or step-in-place, not from a failed reference/shuffle gait:
# this task deliberately has no joint trajectory for PPO to copy.
uv run python -m locomotion.train_rsl_rl --task "$TASK_PERIODIC" --iterations 6000 --run-name periodic-walk --initialize-from logs/rsl_rl/leggybot_skill_step/<run>/model_3999.pt
uv run python -m locomotion.evaluate_task logs/rsl_rl/leggybot_skill_periodic_walk/<run>/model_5999.pt --task "$TASK_PERIODIC"

# 6S. Minimal curved-walk correction — maximum 4,000 iterations. It copies
# the learned actor but starts a new critic/optimizer for the changed reward.
uv run python -m locomotion.train_rsl_rl --task "$TASK_STRAIGHT_PERIODIC" --iterations 4000 --run-name straight-periodic-walk --initialize-from logs/rsl_rl/leggybot_skill_periodic_walk/<restart-run>/model_4250.pt
uv run python -m locomotion.evaluate_task logs/rsl_rl/leggybot_skill_straight_periodic_walk/<run>/model_3999.pt --task "$TASK_STRAIGHT_PERIODIC"

# 6R. If periodic PPO settles into a curved or one-leg-dominant gait, use this
# close correction task. Start fresh: PPO augmentation assumes a near-symmetric
# initial policy, so do not initialize it from an asymmetric periodic policy.
uv run python -m locomotion.train_rsl_rl --task "$TASK_SYMMETRIC_PERIODIC" --iterations 6000 --run-name symmetric-periodic-walk
uv run python -m locomotion.evaluate_task logs/rsl_rl/leggybot_skill_symmetric_periodic_walk/<run>/model_5999.pt --task "$TASK_SYMMETRIC_PERIODIC"

# 7. Forward/backward sagittal tracking — maximum 6,000 iterations.
uv run python -m locomotion.train_rsl_rl --task "$TASK_SAGITTAL" --iterations 6000 --run-name sagittal --initialize-from logs/rsl_rl/leggybot_skill_periodic_walk/<run>/model_5999.pt
uv run python -m locomotion.evaluate_task logs/rsl_rl/leggybot_skill_sagittal/<run>/model_5999.pt --task "$TASK_SAGITTAL"

# Held-out sagittal checks after the final task.
uv run python -m locomotion.evaluate_task logs/rsl_rl/leggybot_skill_sagittal/<run>/model_5999.pt --task "$TASK_SAGITTAL" --sagittal-command -0.42
uv run python -m locomotion.evaluate_task logs/rsl_rl/leggybot_skill_sagittal/<run>/model_5999.pt --task "$TASK_SAGITTAL" --sagittal-command 0.42
```

You may stop a stage early after it passes `evaluate_task`; otherwise let it
reach its maximum, evaluate the last checkpoint, and do not promote a failure.

## Evaluate a checkpoint

The evaluator is deterministic apart from the task's normal sim-to-real
randomization. It checks that the checkpoint task matches `--task`, runs fixed
episodes, prints the task score and survival, and exits with code 0 only on a
pass.

```bash
uv run python -m locomotion.evaluate_task path/to/model.pt \
  --task Mjlab-AnimatedReferenceWalk-LeggyBot-FixedAnkle6DOF
```

Use `--no-sim2real-randomization` only for diagnosis, not for promotion. The
sagittal task additionally supports held commands such as
`--sagittal-command -0.42` or `--sagittal-command 0.42`.

## Run simulator and controller separately

This is the deployment-shaped setup. Native CPU MuJoCo owns physics and robot
I/O; the controller owns estimation and policy inference. The controller does
not import MuJoCo, so a real robot driver can replace the simulator while using
the same protocol.

Start the simulator in terminal 1:

```bash
uv run python -m locomotion.mujoco_robot_server \
  --robot leggybot-fixed-ankle-6dof \
  --command-bind 127.0.0.1 --command-port 15000 \
  --controller-host 127.0.0.1 --state-port 15001 \
  --state-hz 200 --render-hz 60 --velocity-print-hz 1
```

Start the controller in terminal 2:

```bash
CHECKPOINT=logs/rsl_rl/leggybot_skill_animated_reference_walk/<run>/model_4999.pt

uv run python -m locomotion.remote_controller "$CHECKPOINT" \
  --robot leggybot-fixed-ankle-6dof \
  --robot-host 127.0.0.1 --command-port 15000 \
  --state-bind 127.0.0.1 --state-port 15001 \
  --control-hz 50 --action-range-fraction 1.0
```

Use `W/S` for forward/backward command, `A/D` for lateral command, `Q/E` for
yaw, `X` to stop, and `Enter` to reset. Each keypress changes the command by
0.01. Forward/lateral velocity is bounded to +/-0.10 m/s and yaw rate to
`+/-pi/4` rad/s. A task may not have been trained for
every keyboard command; both short-walk tasks should be tested first with
`vx=0.08`, `vy=0`, `yaw=0`.

For a non-interactive test, add `--no-keyboard --vx 0.08 --vy 0 --yaw 0` to the
controller. Add `--headless` to the server when rendering is unnecessary.

The server publishes state at 200 Hz and the controller acts at 50 Hz. Both use
their own 100 ms watchdog. The latest state is used; old state and old commands
are never allowed to queue behind newer data. See [REMOTE_CONTROL.md](REMOTE_CONTROL.md)
for the protocol, velocity comparison output, estimation interface, safety
boundaries, and hardware notes.

## Other playback options

For an all-in-one viewer backed by mjlab, use:

```bash
uv run python -m locomotion.play_keyboard path/to/model.pt \
  --task Mjlab-AnimatedReferenceWalk-LeggyBot-FixedAnkle6DOF
```

For direct native MuJoCo playback in one process, use:

```bash
uv run python -m locomotion.play_native path/to/model.pt \
  --robot leggybot-fixed-ankle-6dof
```

Both recognize RSL-RL and custom-PPO `.pt` policies. A policy is independent
of which PPO implementation created it, but it is not independent of its robot
ABI: joint order, observations, normalization, action mapping, and limits must
all match.

## Trajectory feasibility diagnostic

This is not a replacement for PPO. It searches for and validates a periodic
open-loop trajectory to check whether the geometry, limits, and modeled torque
speed envelope permit a motion at all.

```bash
uv run python -m locomotion.optimize_gait \
  --robot leggybot-fixed-ankle-6dof --output artifacts/trajectory.npz
uv run python -m locomotion.validate_gait_trajectory \
  artifacts/trajectory.npz --robot leggybot-fixed-ankle-6dof
```

Read [../TRAJECTORY_FEASIBILITY.md](../TRAJECTORY_FEASIBILITY.md) for the exact
diagnostic interpretation.

## Add or replace a robot

Add the model, measured `RobotProfile`, native model factory, and a task factory
using the common task catalog. Reuse a policy only when the complete actor ABI
is identical. Add backlash/hysteresis only from bench data, and validate every
new robot with zero-action, random-action, short-training, and trajectory
feasibility checks before treating it as a sim-to-real candidate.
