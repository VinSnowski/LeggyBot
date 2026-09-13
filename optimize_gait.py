"""Command-line direct-shooting optimiser for the native LeggyBot gait."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from locomotion.robot_profiles import (
    LEGGYBOT_FIXED_ANKLE_PROFILE,
    LEGGYBOT_PROFILE,
    get_robot_profile,
    make_native_model,
)
from locomotion.trajectory_optimization import (
    PeriodicSpline,
    bilateral_gait_seed,
    load_trajectory,
    weight_transfer_gait_seed,
    save_trajectory,
    simulate_trajectory,
    trajectory_fingerprint,
    validate_trajectory,
    write_trajectory_csv,
    write_metrics_csv,
)


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    knots: int = 8
    period_s: float = 0.8
    speeds: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20)
    iterations: int = 8
    generations: int | None = None
    population: int = 8
    sigma_rad: float = 0.04
    duration_s: float = 2.0
    seed: int = 0
    seeds: tuple[int, ...] | None = None
    seed_amplitude: float = 0.35
    enforce_bilateral_symmetry: bool = True
    seed_type: str = "weight-transfer"
    robust_objective_rollouts: int = 0


@dataclass(slots=True)
class OptimizationResult:
    trajectory: PeriodicSpline
    speed_results: list[dict]
    best_cost: float
    fingerprint: str
    history: list[dict] = field(default_factory=list)


def _cost(rollout, target_speed: float, period_s: float) -> float:
    m = rollout.metrics
    # Keep feasibility terms dominant so a slightly slower but upright gait is
    # preferred over one that wins speed by falling.
    cost = 1200.0 * abs(m.achieved_speed_m_s - target_speed)
    cost += 600.0 * m.lateral_drift_m + 1000.0 * m.yaw_drift_rad
    cost += 2000.0 * max(0.0, m.lateral_excursion_m - 0.12)
    cost += 1000.0 * max(0.0, m.yaw_excursion_rad - np.deg2rad(15.0))
    cost += 3000.0 * max(0.0, 0.020 - m.minimum_clearance_m)
    if m.minimum_clearance_m < 0.015:
        cost += 240.0
    cost += 10.0 * max(0.0, 5.0 - m.minimum_joint_limit_margin_deg) / 57.3
    cost += 20.0 * m.foot_slip_m_s + 5.0 * m.landing_speed_m_s
    samples = max(1, len(rollout.times))
    cost += 1200.0 * m.non_foot_contacts / samples
    cost += 1000.0 * int(m.falls > 0)
    quat = rollout.qpos[:, 3:7]
    up_z = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    cost += 800.0 * float(np.mean(np.maximum(0.0, 0.92 - up_z) ** 2))
    cost += 500.0 * float(max(0.0, 0.80 - np.min(up_z)))

    # Reward the actual contact sequence, not merely joint motion.  A missing
    # stance contact is more dangerous than a late swing-foot lift-off.
    phase = np.mod(rollout.times / period_s, 1.0)
    for foot, stance_foot, lo, hi in ((0, 1, 0.125, 0.375),
                                      (1, 0, 0.625, 0.875)):
        window = (phase >= lo) & (phase < hi)
        if np.any(window):
            cost += 80.0 * float(np.mean(rollout.contacts[window, foot]))
            cost += 240.0 * float(np.mean(~rollout.contacts[window, stance_foot]))
    cost += 30.0 * m.q_cycle_periodicity_error_rad
    cost += 5.0 * m.qvel_cycle_periodicity_error_rad_s
    if m.actuator is not None:
        cost += 20.0 * max(0.0, -m.actuator.torque_envelope_margin_nm)
        cost += 20.0 * max(0.0, m.actuator.peak_torque_nm - m.actuator.safe_torque_limit_nm)
        cost += 100.0 * max(0.0, m.actuator.peak_torque_nm
                           - 0.75 * m.actuator.safe_torque_limit_nm)
        cost += 10.0 * max(0.0, m.actuator.peak_speed_rad_s - m.actuator.speed_limit_rad_s)
    cost += 80.0 * len(m.violations)
    return float(cost)


def _bilateral_projection(knots: np.ndarray) -> np.ndarray:
    """Project an even-knot gait onto straight-line half-cycle symmetry."""
    result = np.asarray(knots, dtype=np.float64).copy()
    half = result.shape[0] // 2
    if result.shape[0] % 2:
        raise ValueError("bilateral gait symmetry requires an even knot count")
    right = result[:, :3].copy()
    # Straight walking shifts both hips in the same direction to move the COM
    # over the stance foot.  The roll waveform changes sign by itself over the
    # half cycle; copying it to the other hip preserves that common lean.
    result[:, 3] = right[:, 0]
    result[:, 4] = np.roll(right[:, 1], half)
    result[:, 5] = np.roll(right[:, 2], half)
    return result


def optimize_gait(config: OptimizerConfig = OptimizerConfig(), *, profile=LEGGYBOT_FIXED_ANKLE_PROFILE,
                  initial_trajectory: PeriodicSpline | None = None) -> OptimizationResult:
    """Run a small deterministic diagonal evolution strategy.

    This intentionally avoids the optional ``cma`` package.  It retains the
    important continuation and simulation-in-the-loop semantics while staying
    usable in a clean CPU checkout.
    """
    if config.knots != 8:
        raise ValueError("the validated workflow uses exactly 8 spline knots")
    if config.iterations < 0 or config.population < 1 or config.sigma_rad <= 0:
        raise ValueError("invalid optimiser settings")
    rng = np.random.default_rng(config.seed)
    generations = config.iterations if config.generations is None else config.generations
    if generations < 0:
        raise ValueError("generations must be non-negative")
    seeds = tuple(config.seeds or (config.seed,))
    if initial_trajectory is not None:
        if initial_trajectory.knot_count != config.knots:
            raise ValueError("initial trajectory knot count does not match optimizer")
        seed_trajectory = PeriodicSpline(initial_trajectory.knots, config.period_s)
    elif config.seed_type == "weight-transfer":
        seed_trajectory = weight_transfer_gait_seed(profile, config.knots, config.period_s)
    elif config.seed_type == "sagittal":
        seed_trajectory = bilateral_gait_seed(profile, config.knots, config.period_s)
    else:
        raise ValueError(f"unknown seed type: {config.seed_type}")
    nominal = profile.array("default_joint_pos")
    amplitude = 1.0 if initial_trajectory is not None else config.seed_amplitude
    scaled_seed = nominal + amplitude * (seed_trajectory.knots - nominal)
    if config.enforce_bilateral_symmetry:
        scaled_seed = _bilateral_projection(scaled_seed)
    current = PeriodicSpline(scaled_seed, config.period_s)
    action_lower, action_upper = profile.action_bounds()
    safety_margin = np.deg2rad(5.0)
    lower = np.maximum(action_lower, profile.array("joint_lower") + safety_margin)
    upper = np.minimum(action_upper, profile.array("joint_upper") - safety_margin)
    speed_results: list[dict] = []
    best_cost = float("inf")
    history: list[dict] = []
    for speed in config.speeds:
        robust_cases = []
        robust_rng = np.random.default_rng(config.seed + 7919 + int(round(speed * 1000)))
        for _ in range(max(0, config.robust_objective_rollouts)):
            robust_model = make_native_model(profile)
            robust_model.geom_friction[:] *= robust_rng.uniform(0.85, 1.15)
            robust_model.body_mass[:] *= robust_rng.uniform(0.95, 1.05)
            gain_scale = robust_rng.uniform(0.90, 1.10)
            robust_model.actuator_gainprm[:, 0] *= gain_scale
            robust_model.actuator_biasprm[:, 1] *= gain_scale
            robust_model.actuator_forcerange[:] *= robust_rng.uniform(0.90, 1.00)
            robust_cases.append((
                robust_model,
                float(robust_rng.uniform(0.0, 0.010)),
                robust_rng.normal(0.0, (0.003, 0.003, 0.01)),
            ))
        candidates_for_seed: list[tuple[float, np.ndarray]] = []
        for run_seed in seeds:
            mean = current.knots.copy()
            def objective(vector: np.ndarray) -> float:
                candidate = np.asarray(vector, dtype=np.float64).reshape(config.knots, profile.action_dim)
                if config.enforce_bilateral_symmetry:
                    candidate = _bilateral_projection(candidate)
                candidate = np.clip(candidate, lower, upper)
                trial = PeriodicSpline(candidate, current.period_s)
                costs = [_cost(simulate_trajectory(trial, speed, config.duration_s, profile,
                                                   seed=run_seed), speed, current.period_s)]
                for case_index, (robust_model, latency, perturbation) in enumerate(robust_cases):
                    replay = simulate_trajectory(
                        trial, speed, config.duration_s, profile,
                        seed=run_seed + case_index + 1, model=robust_model,
                        command_latency_s=latency,
                        initial_state_perturbation=perturbation,
                    )
                    costs.append(_cost(replay, speed, current.period_s))
                value = max(costs) + 0.10 * float(np.mean(costs))
                history.append({"speed_m_s": float(speed), "seed": int(run_seed), "cost": float(value)})
                return value
            if generations:
                # Keep ``population`` as the actual number of candidates per
                # generation. SciPy's differential_evolution multiplies this
                # by the 48-dimensional parameter count, making a full
                # continuation search needlessly huge on CPU.
                local_rng = np.random.default_rng(run_seed)
                best_local = mean.copy()
                best_value = objective(best_local.ravel())
                local_sigma = float(config.sigma_rad)
                for _generation in range(int(generations)):
                    candidates = [best_local]
                    for _ in range(max(1, int(config.population) - 1)):
                        candidates.append(np.clip(
                            best_local + local_rng.normal(0.0, local_sigma, best_local.shape),
                            lower, upper))
                    if config.enforce_bilateral_symmetry:
                        candidates = [_bilateral_projection(candidate) for candidate in candidates]
                    scored = sorted(
                        [(objective(candidate.ravel()), candidate) for candidate in candidates],
                        key=lambda item: item[0])
                    best_value, best_local = float(scored[0][0]), np.asarray(scored[0][1])
                    local_sigma = max(config.sigma_rad * 0.1, local_sigma * 0.92)
                candidates_for_seed.append((best_value, best_local))
            else:
                trial = PeriodicSpline(mean, current.period_s)
                candidates_for_seed.append((_cost(simulate_trajectory(trial, speed, config.duration_s,
                                                                       profile, seed=run_seed), speed,
                                                  current.period_s), mean))
        score, mean = min(candidates_for_seed, key=lambda pair: pair[0])
        best_cost = min(best_cost, score)
        current = PeriodicSpline(np.clip(mean, lower, upper), current.period_s)
        speed_results.append({"speed_m_s": float(speed),
                              "metrics": simulate_trajectory(current, speed, config.duration_s,
                                                              profile, seed=config.seed).metrics.to_dict()})
    return OptimizationResult(current, speed_results, best_cost, trajectory_fingerprint(current, profile), history)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot", default=LEGGYBOT_FIXED_ANKLE_PROFILE.name,
        choices=(LEGGYBOT_PROFILE.name, LEGGYBOT_FIXED_ANKLE_PROFILE.name),
        help="current six-DOF LeggyBot model used for feasibility search",
    )
    parser.add_argument("--output", type=Path, default=Path("trajectory.npz"))
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--generations", type=int, default=None,
                        help="differential-evolution generations (alias for iterations)")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--period", type=float, default=0.8)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--knots", type=int, default=8)
    parser.add_argument("--target-speed", type=float, default=None)
    parser.add_argument("--continuation-speeds", nargs="+", default=None,
                        help="speeds in m/s; accepts space- or comma-separated values")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="one or more random seeds")
    parser.add_argument("--seed-amplitude", type=float, default=0.35,
                        help="fraction of the original kinematic gait used to initialize search")
    parser.add_argument("--seed-type", choices=("weight-transfer", "sagittal"),
                        default="weight-transfer", help="kinematic initializer")
    parser.add_argument("--initial-trajectory", type=Path,
                        help="continue optimization from an existing trajectory artifact")
    parser.add_argument("--allow-initial-model-mismatch", action="store_true",
                        help="use a joint-compatible trajectory only as an optimization seed")
    parser.add_argument("--no-bilateral-symmetry", action="store_true",
                        help="allow small independent left/right corrections")
    parser.add_argument("--robust-objective-rollouts", type=int, default=0,
                        help="perturbed rollouts included in every optimizer evaluation")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    profile = get_robot_profile(args.robot)
    if args.continuation_speeds:
        speeds = tuple(float(token) for item in args.continuation_speeds
                       for token in str(item).split(",") if token.strip())
    else:
        speeds = (args.target_speed,) if args.target_speed is not None else OptimizerConfig.speeds
    if args.seeds and len(args.seeds) == 1 and args.seeds[0] > 1:
        seed_list = tuple(range(args.seed, args.seed + args.seeds[0]))
    else:
        seed_list = tuple(args.seeds) if args.seeds else None
    config = OptimizerConfig(knots=args.knots, speeds=speeds, period_s=args.period, iterations=args.iterations,
                              population=args.population, seed=args.seed,
                              duration_s=args.duration, generations=args.generations,
                              seeds=seed_list, seed_amplitude=args.seed_amplitude,
                              seed_type=args.seed_type,
                              enforce_bilateral_symmetry=not args.no_bilateral_symmetry,
                              robust_objective_rollouts=args.robust_objective_rollouts)
    initial = (load_trajectory(args.initial_trajectory, profile=profile,
                               allow_model_mismatch=args.allow_initial_model_mismatch)
               if args.initial_trajectory else None)
    result = optimize_gait(config, profile=profile, initial_trajectory=initial)
    save_trajectory(args.output, result.trajectory, profile=profile,
                    metadata={"best_cost": result.best_cost,
                                                               "fingerprint": result.fingerprint},
                    history={"evaluations": result.history})
    write_trajectory_csv(args.output.with_name("samples.csv"), result.trajectory,
                         profile.control_dt)
    history_path = args.output.with_name("optimization_history.csv")
    if result.history:
        with history_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("speed_m_s", "seed", "cost"))
            writer.writeheader(); writer.writerows(result.history)
    validation = validate_trajectory(result.trajectory, config.speeds, profile=profile,
                                     duration_s=args.duration)
    metrics_base = args.metrics or args.output.with_suffix("")
    metrics_base.parent.mkdir(parents=True, exist_ok=True)
    metrics_base.with_suffix(".json").write_text(json.dumps(validation, indent=2, default=float))
    write_metrics_csv(metrics_base.with_suffix(".csv"), validation)
    print(json.dumps({"output": str(args.output), "fingerprint": result.fingerprint,
                      "feasible": validation["feasible"], "best_cost": result.best_cost}, default=float))
    return 0 if validation["feasible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
