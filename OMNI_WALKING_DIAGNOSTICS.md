# Omnidirectional walking diagnostics — 2026-09-14

## Precision follow-up from natural checkpoint 400 — 2026-09-15

The full ten-second evaluator ranked natural checkpoint 400 highest: stop,
forward, both lateral directions, and the left curve passed, while a persistent
right-turn rate bias accumulated too much heading error. Later checkpoints
lost lateral competence even though their aggregate TensorBoard score rose.

The former Natural V2 precision follow-up was evaluated and discarded: its
four-second displacement and velocity-precision penalties reduced tracking
success instead of improving the Natural baseline. The current follow-up is
V3, which is based directly on the Natural reward and adds only stronger pitch
upright shaping and the lateral foot-line term.

```bash
uv run python -m locomotion.train_rsl_rl \
  --task NaturalOmnidirectionalWalkingTaskV3 --num-envs 4096 \
  --iterations 1500 --run-name natural-v3-from-zero
```

## Follow-up: natural stepping run

The 4,000-iteration `SE2TrackingWalkingTask` run found a useful dynamic stepping
mode, but its optimization became pathological. The strongest evidence is the
Gaussian policy standard deviation: it rose from 0.10 to 79.68 while executed
actions were clipped to [-1, 1]. Over the same run, the action-rate reward cost
grew from -0.014 in iterations 150–350 to -0.257 near the end, and the joint-limit
cost grew from -0.001 to -0.147. The output-layer norm also grew from 0.61 at
checkpoint 150 to 15.63 at checkpoint 3999. PPO was optimizing probabilities in
an effectively unbounded Gaussian while the environment mostly received saturated
actions. The entropy term encouraged this divergence.

The reward also admitted visible behavioral degeneracies:

- Scheduled contact, swing unloading and stance-foot speed ran at zero command,
  so standing still was asked to follow a walking contact clock.
- There was no stopped-pose or stopped-foot-alignment objective after the old
  foot-exchange term was removed. A permanent scissor stance therefore had no
  direct cost.
- One combined SE(2) reward coupled linear and yaw errors: a large error on one
  axis suppressed the positive signal for the other axis.
- Pure lateral commands included magnitudes arbitrarily close to zero, weakening
  the association between a lateral command and an actual step.
- The actuator-effort contribution was only about -0.003 late in training. The
  visually effortless gait emerged from dynamics/contact shaping; it was not
  produced by a meaningful energy objective. This is encouraging, but it should
  not be interpreted as measured electrical or mechanical efficiency.

`NaturalOmnidirectionalWalkingTask` is the next experimental task. It retains
the useful 0.33-second phase clock and task-space contact shaping, while:

- using a log-standard-deviation Gaussian bounded to 0.03–0.40 and reducing the
  entropy coefficient from 0.01 to 0.001;
- enabling gait/contact shaping only for commanded planar motion;
- adding a nominal joint-pose reward and fore/aft foot-alignment cost at stop;
- separating linear and yaw tracking rewards and long-window costs;
- sampling lateral commands with at least 0.04 m/s magnitude;
- increasing action-rate, joint-limit and actuator-effort regularization modestly;
- keeping symmetry augmentation disabled so independent servo mismatch remains
  represented during training.

Actor-only initialization from `model_300.pt` is preferred. It keeps the stable
forward stepping network but creates a fresh critic, optimizer and bounded
exploration distribution. A 512-environment, 25-iteration transfer smoke run
completed with action standard deviation 0.144, tracking success 0.260, windowed
linear error 0.0437 m/s and yaw error 0.0634 rad/s. All 48 nominal ten-second
fixed-command trials survived. It retained forward and curve behavior; lateral
speed reached about ±0.025 m/s against ±0.060 m/s targets. This is validation of
the training direction, not a finished policy.

```bash
uv run python -m locomotion.train_rsl_rl \
  --task NaturalOmnidirectionalWalkingTask --num-envs 4096 \
  --iterations 1500 --run-name natural-from-300 \
  --initialize-from logs/rsl_rl/leggybot_se2_tracking_walking/\
2026-09-14_18-29-42_se2-tracking/model_300.pt
```

Do not select the final iteration automatically. Evaluate checkpoints every 50
iterations with `locomotion.diagnose_omni`; require all six modes to survive and
rank them by the worst command, especially both lateral directions and stop.
Inspect checkpoints around 300, 600, 900, 1200 and 1499 in the native
simulator/controller as well. Stop the run if tracking degrades for several
checkpoints while action standard deviation remains at its 0.40 ceiling.

The run-level evaluator automates this checkpoint selection and returns a
nonzero status unless at least one checkpoint passes every mode:

```bash
uv run python -m locomotion.evaluate_omni_run \
  logs/rsl_rl/leggybot_natural_omnidirectional_walking/<run-directory>
```

Its default objectives require 95% survival, at most 0.03 m/s net linear error,
0.10 rad/s yaw error, 0.20 m path RMSE, 0.30 rad heading RMSE, and genuine
bilateral liftoffs in at least 80% of moving trials. Stop additionally limits
creep, joint-pose error, fore/aft foot offset, and repeated stepping. Detailed
JSON and a concise Markdown table are written to `artifacts/omni_diagnostics`.

Implemented `SE2TrackingWalkingTask` as a new experimental baseline. V2 behavior
is preserved. The new task has passed numerical regression tests and a five-iteration
PPO smoke run; it has **not** been trained to convergence or validated on hardware.

## What is going wrong

1. **Zero-yaw commands disable the main yaw penalties.** V2 supplies
   `yaw_deadband` and `yaw_relative_scale` to the cycle, episode and instantaneous
   costs. Their inactive branch returns zero yaw cost. Straight walking and pure
   lateral walking therefore receive no yaw correction from these dominant
   terms. A small angular-rate regularizer remains, so yaw is not completely
   unpenalized. The staged curriculum's success metric also ignores yaw in these
   modes, and treats stop commands as successful without checking motion.
2. **The displacement reference follows measured heading.** The cycle and
   episode integrators rotate forward/lateral commands by actual heading. That
   is appropriate for instantaneous body-velocity tracking but does not specify
   an independent desired path: uncommanded turning bends the reference too.
3. **Episode averaging hides recent mistakes.** Dividing accumulated signed
   error by elapsed episode time allows errors in different directions to cancel
   and dilutes recent command-transition errors. Long histories are absent from
   the feedforward actor and critic. Increasing this reward's weight does not
   recover the missing information or improve temporal credit assignment.
4. **The staged curriculum starves the requested motions.** Saved states from
   the staged-recent `model_700.pt` and latest `model_450.pt` both remain at stage
   0, with difficulty scale 0.5. Stage 0 samples forward and stop commands only;
   its nominal curve mode has zero yaw. Their last logged command-tracking
   success values are 0.1315 and 0.0442. These policies have not received lateral
   or nonzero-yaw curve training in this stage.
5. **Instantaneous speed is a poor sole gate for this gait.** The older policy
   can average 0.073 m/s forward while showing 0.22 m/s instantaneous lateral
   velocity RMSE from its oscillatory gait. A strict per-step velocity gate can
   block curriculum progression even when net travel is useful. Evaluate net
   movement over several steps, plus independent long-horizon path error.

Do not interpret the logger's `Episode_Termination/time_out` directly as a
survival probability: saved logs contain values above one. The standalone
benchmark counts surviving environments directly.

## Measured baseline

`locomotion.diagnose_omni` evaluates six fixed commands for 10 seconds, eight
replicas each, seed 123, nominal playback physics, deterministic actions. It
excludes the first second from velocity/error statistics, tracks reference
heading from commanded yaw, and reports failures separately. Failed replicas
are excluded after their first failure; their subsequent engine resets do not
restore their survival status. Velocity statistics are pooled over surviving
samples and should always be read together with survival and valid-sample count.

Older run: `2026-09-14_15-13-17_quasi-omnidirectional-definitive/model_2350.pt`.
All 48 replicas survived. Mean heading-frame velocities and reference-path RMSE:

| Command | Measured vx | Measured vy | Measured yaw | Path RMSE |
|---|---:|---:|---:|---:|
| Stop | 0.0053 | -0.0036 | 0.0007 | 0.077 m |
| Forward: 0.08 m/s | 0.0727 | 0.0003 | 0.0186 | 0.029 m |
| Left: 0.06 m/s | 0.0119 | 0.0693 | 0.0088 | 0.106 m |
| Right: -0.06 m/s | 0.0137 | -0.0729 | 0.0075 | 0.113 m |
| Curve: vx=0.08, yaw=0.30 | 0.0917 | -0.0181 | 0.1446 | 0.387 m |
| Curve: vx=0.08, yaw=-0.30 | 0.0563 | 0.0214 | -0.1585 | 0.309 m |

Velocities are m/s and yaw rates rad/s. Both curves achieve only about half the
requested turning rate. Lateral motion leaks forward. These are measurable
tracking problems even when the robot remains upright.

Latest staged run: `2026-09-14_17-56-41_quasi-omnidirectional-latest/model_450.pt`.
Forward achieves 0.0785 m/s but adds -0.0166 m/s lateral drift and +0.0924 rad/s
yaw; heading RMSE is 0.531 rad. Requested left motion produces -0.0566 m/s
lateral velocity, and requested right motion produces +0.1150 m/s: both are
the wrong direction. All replicas survive these straight/lateral tests, but
**none survive either curve test for ten seconds**. Every left-curve replica
fails before the one-second warmup, so its tracking statistics are unavailable.
This directly confirms that survival during the forward-only stage does not
establish command generalization.

Raw results are in `artifacts/omni_diagnostics/v2_definitive.json` and
`artifacts/omni_diagnostics/v2_latest.json`. These evaluations use the current
task's nominal playback physics, not a reconstruction of every historical
training configuration. Eight randomized reset poses are a diagnostic sample,
not a broad robustness study.

## New task

The task retains the physical model, servo authority, 29-value actor,
44-value critic, 0.33-second gait clock, and deployment compatibility.

- Commands: 35% forward, 30% forward curves, 30% lateral, 5% stop. All modes
  occur from the beginning. Forward is 0.04–0.10 m/s; lateral is ±0.08 m/s;
  yaw is bounded by 0.40 rad/s and scaled by forward speed. No reverse or full
  diagonal command mode is included in this focused baseline.
- A 0.66-second sliding window measures net translation and yaw against exact
  constant-twist SE(2) integration. The reference turns with commanded yaw.
  Zero yaw always incurs an error when the robot turns. The reward combines a
  positive tracking bonus with a smooth error cost that remains informative
  away from the target.
- A two-second sliding window penalizes persistent trajectory error. It
  expires old mistakes rather than averaging the whole episode. Each window
  starts in the measured frame at its beginning: this trains local drift
  suppression, not return-to-path navigation after arbitrary external pushes.
- Contact timing and stance-foot speed remain modest gait regularizers.
  Removing the prescribed foot-exchange trajectory allows the policy to find
  lateral and turning motions without a competing foot-placement template.
- Twenty-second episodes allow more frequent resets and expose transitions.
  There is no command curriculum gate that can permanently exclude a mode.
  Independent servo randomization uses the existing gradual sim-to-real scale
  from the start, without the paired-motor pretraining interval.

The strict logged success metric checks both net planar error <0.025 m/s and
net yaw error <0.10 rad/s, including stop/straight/lateral commands, with an
upright condition and a full observation window. Use the six-command benchmark
as the acceptance test; a mixed average can still conceal a weak direction.

## Symmetry decision

Disable augmentation for this baseline. The current joint permutations,
polar/axial vector signs, command mirroring and half-cycle phase shift are
internally consistent and their tests pass. This does **not** prove that
augmentation caused the existing failures.

The installed RSL-RL extension repeats original log probabilities, returns and
advantages for mirrored samples. That is a stronger assumption than merely
reflecting joint targets. Real motor mismatch, imperfect CAD symmetry and
unobserved history make an unaugmented baseline useful. Independent identically
distributed randomization can still be symmetric in distribution; it is not by
itself proof that augmentation is invalid. Revisit augmentation only with a
matched multi-seed comparison using this same reward and command distribution.
Judge straight-path drift and both turn directions, not visual leg symmetry.

## Research and hardware implications

[Periodic Reward Composition](https://arxiv.org/abs/2011.01387) supports using
phase-based force and velocity rewards to shape biped gaits without requiring
one complete joint-space reference. That motivates keeping modest contact
guidance here; it does not establish these weights for LeggyBot.

[Learning Symmetric and Low-energy Locomotion](https://arxiv.org/abs/1801.08093)
demonstrates benefits from symmetry regularization, but does not establish that
our particular mirrored PPO data and hardware satisfy its assumptions.

[Extended Friction Models for Servo Actuators](https://arxiv.org/abs/2410.08650)
identifies friction from loaded pendulum trajectories and shows why simple
Coulomb-viscous models can miss important servo behavior. [BAM](https://github.com/Rhoban/bam)
provides identification tooling and includes STS3215 models. This is particularly
relevant to slow lateral stepping and direction reversals.

[Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini/blob/v2/README.md)
documents actuator identification as part of its workflow;
[Bimo](https://github.com/mekion/the-bimo-project) provides an open sim-to-real
platform and Isaac Lab implementation. Their successful transfer is motivation
for measuring our own actuators, not evidence that their reward weights or
joint trajectories will transfer to this fixed-ankle six-joint robot.

LeggyBot's existing servo documentation explicitly identifies unmeasured
backlash, supply effects and motor variation as remaining limitations. Keep
the present physical limits until loaded measurements justify changes. With
no actuated hip yaw, turning depends on coordinated foot placement and contact
rotation; an excessively strong no-slip objective may compete with it.

The actor has no deployed planar velocity/path estimate. This task can reduce
systematic drift through training, but cannot guarantee world-path correction
under arbitrary disturbances. Reliable navigation eventually needs a measured
heading/velocity estimate and an outer feedback controller, or a deliberately
expanded deployment observation/history interface.

## Reproduce and train

```bash
uv run python -m locomotion.train_rsl_rl \
  --task SE2TrackingWalkingTask --num-envs 4096 \
  --iterations 4000 --run-name se2-tracking

uv run python -m locomotion.diagnose_omni \
  path/to/model.pt --task SE2TrackingWalkingTask \
  --output artifacts/omni_diagnostics/se2_candidate.json
```

Start from a fresh actor for a clean comparison. An optional actor-only
initialization from the older `model_2350.pt` is plausible because it already
survives and responds to all modes; compare that separately, since it can retain
the old turning bias. The staged latest checkpoint is a poorer starting point
for the required lateral/curve coverage.

Do not expand toward ±π/4 rad/s until both curve directions pass the modest
envelope. A suggested next gate is ≥95% survival per command over multiple
seeds, <0.025 m/s net planar error, <0.10 rad/s net yaw error, then 30–60 second
path tests and randomized-model tests. The nominal ten-second benchmark alone
is not sufficient for deployment.

Validation: numerical tests cover analytic arcs, zero yaw, command changes,
heading wrap, drift, stationary under-travel, history expiry, idempotent calls,
partial resets and task registration. A 128-environment, five-iteration GPU PPO
run completed and exported `model_4.pt` and ONNX under
`logs/rsl_rl/leggybot_se2_tracking_walking/2026-09-14_18-24-55_se2-smoke`.
That artifact is a smoke-test policy, not a usable walking controller.

Final regression results: 90 tests passed in the sandbox; the remaining local
UDP test passed separately with socket access. The pre-existing v2 assertion
expecting -4.0 was updated to match its already configured -8.0 episode weight;
the v2 reward itself was not changed. Python compilation and whitespace checks
also passed.
