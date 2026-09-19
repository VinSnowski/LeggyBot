# LeggyBot locomotion

This repository defines one production locomotion task: `WalkingTask`. It
trains the fixed-ankle six-joint LeggyBot to stand, walk forward, and follow
left/right forward curves with a 0.33-second gait clock.

The actor contract is fixed at 29 observations and six normalized joint-position
actions. The critic receives privileged simulation state during training.

## Install

Training with mjlab requires an NVIDIA GPU:

```bash
uv sync --dev
```

Run commands from the repository root.

## Train

`WalkingTask` is the default, so `--task` is optional:

```bash
uv run python -m locomotion.train_rsl_rl \
  --task WalkingTask \
  --num-envs 4096 \
  --iterations 2000 \
  --run-name walking
```

The training command distribution is 45% straight forward, 45% forward with
yaw, and 10% stopped. Forward speed is 0.04–0.10 m/s while moving; yaw is
bounded to ±0.40 rad/s. Pure lateral, reverse, and full-diagonal commands are
not sampled.

Training uses synchronized command delay, observation age, encoder/calibration,
servo, mass, inertia, center-of-mass, contact, and external-push randomization.
Physical parameters remain paired for 500 PPO iterations and transition to
independent samples over the next 300 iterations.

The policy uses reflected left/right samples and a soft mirror loss. The reward
combines finite-window linear/yaw tracking, long-window path tracking, upright
posture, scheduled contact, swing unloading, stance-foot stability, stopped
stance, effort, action-rate, and safety terms. The full definition is in
`locomotion/tasks/walking.py`.

## Evaluate

Rank every saved checkpoint in a run against stop, forward, and both curve
directions:

```bash
uv run python -m locomotion.evaluate_omni_run \
  logs/rsl_rl/leggybot_walking/<run-directory>
```

Reports are written under `artifacts/omni_diagnostics` by default. A passing
checkpoint must satisfy every command mode; modes are not averaged together.

For a quick fixed-command diagnostic:

```bash
uv run python -m locomotion.diagnose_omni \
  logs/rsl_rl/leggybot_walking/<run-directory>/model_1999.pt \
  --task WalkingTask \
  --output artifacts/omni_diagnostics/walking.json
```

The logged `tracking_success` is a strict binary window metric. It requires
planar error below 0.025 m/s, yaw error below 0.10 rad/s, an upright torso, and
a complete 0.66-second window. Read it together with
`window_linear_error_mps`, `window_yaw_error_radps`, episode length, and the
fixed-command report.

## Watch a policy

Run the deterministic native MuJoCo sequence:

```bash
uv run python -m locomotion.watch_policy_sequence \
  logs/rsl_rl/leggybot_walking/<run-directory>/model_1999.pt
```

Use `--no-lateral` to restrict playback to commands present in training: stop,
forward, and both forward curves.

For interactive mjlab playback:

```bash
uv run python -m locomotion.play_keyboard \
  logs/rsl_rl/leggybot_walking/<run-directory>/model_1999.pt \
  --task WalkingTask
```

## Decoupled simulator and controller

Start the robot-side simulator:

```bash
uv run python -m locomotion.mujoco_robot_server \
  --robot leggybot-fixed-ankle-6dof
```

Then start the joystick controller in another terminal:

```bash
uv run python -m locomotion.remote_controller_joystick \
  logs/rsl_rl/leggybot_walking/<run-directory>/model_1999.pt \
  --robot leggybot-fixed-ankle-6dof
```

The controller and robot process communicate asynchronously over UDP. The
robot owns the 200 Hz physics or motor loop; policy inference runs at 50 Hz.
Both sides reject robot identity or joint-order mismatches and fall back to a
safe nominal pose when state or commands become stale.

See [REMOTE_CONTROL.md](REMOTE_CONTROL.md) for protocol, estimator, actuator,
and hardware-safety details.

## Robot model

`NewLeggyBot/mjcf/NewLeggyBot.xml` is the authoritative fixed-ankle MuJoCo
model. Inspect it with:

```bash
uv run python -m locomotion.view_cad_robot \
  locomotion/NewLeggyBot/mjcf/NewLeggyBot.xml --fixed-base
```

Normalized actions use an asymmetric piecewise-affine map: zero is the nominal
pose and ±1 reach each joint's safe limit with a 0.035 rad margin. Training,
evaluation, native playback, and the remote controller share the same robot
profile and action mapping.
