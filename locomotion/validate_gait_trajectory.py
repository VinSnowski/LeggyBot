"""Validate a saved trajectory under nominal and robust replay conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import mujoco.viewer

from locomotion.robot_profiles import (
    LEGGYBOT_FIXED_ANKLE_PROFILE,
    LEGGYBOT_PROFILE,
    get_robot_profile,
    make_native_model,
)
from locomotion.trajectory_optimization import load_trajectory, validate_trajectory, write_metrics_csv
from locomotion.trajectory_optimization import simulate_trajectory


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument(
        "--robot", default=LEGGYBOT_FIXED_ANKLE_PROFILE.name,
        choices=(LEGGYBOT_PROFILE.name, LEGGYBOT_FIXED_ANKLE_PROFILE.name),
        help="current six-DOF LeggyBot model used for validation",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-robust", action="store_true")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--speeds", type=float, nargs="+", default=None,
                        help="speeds to replay (default: continuation set)")
    parser.add_argument("--rollouts", "--robust-rollouts", dest="rollouts", type=int, default=4,
                        help="robust perturbation rollouts per speed")
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--viewer", action="store_true", help="replay the trajectory in MuJoCo viewer")
    args = parser.parse_args(argv)
    profile = get_robot_profile(args.robot)
    trajectory = load_trajectory(args.trajectory, profile=profile)
    duration = args.duration or max(trajectory.period_s * args.cycles, 2.0)
    report = validate_trajectory(trajectory, speeds=tuple(args.speeds) if args.speeds else
                                  (0.0, 0.05, 0.10, 0.15, 0.20),
                                  profile=profile, robust=not args.no_robust, duration_s=duration,
                                  robust_rollouts=args.rollouts)
    destination = args.output or args.trajectory.with_name("validation_report.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=float))
    write_metrics_csv(destination.with_suffix(".csv"), report)
    if args.viewer:
        model = make_native_model(profile)
        rollout = simulate_trajectory(trajectory, duration_s=duration,
                                      profile=profile, model=model)
        data = mujoco.MjData(model)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for i in range(len(rollout.times)):
                data.qpos[:] = rollout.qpos[i]
                data.qvel[:] = rollout.qvel[i]
                mujoco.mj_forward(model, data)
                viewer.sync()
    print(json.dumps(report, default=float))
    return 0 if report["feasible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
