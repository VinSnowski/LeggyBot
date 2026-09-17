"""Evaluate every selected checkpoint in one omnidirectional locomotion run.

The run passes when at least one checkpoint satisfies every fixed-command mode.
This deliberately selects by the worst configured mode instead of averaging
easy forward or stop samples over failed lateral or turning behavior. V3 is
checked in both signs of lateral and forward-plus-yaw motion, plus stop,
forward, and backward travel.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from locomotion.evaluate_task import _load_actor
from locomotion.task import make_env_cfg
from locomotion.tasks.skills import (
    NATURAL_OMNIDIRECTIONAL_TASK_ID,
    NATURAL_OMNIDIRECTIONAL_V3_TASK_ID,
    NATURAL_OMNIDIRECTIONAL_V4_TASK_ID,
    NATURAL_OMNIDIRECTIONAL_V5_TASK_ID,
    PERIODIC_REWARD_COMPOSITION_WALKING_TASK_ID,
    QUASI_OMNIDIRECTIONAL_WALKING_TASK_ID,
    QUASI_OMNIDIRECTIONAL_WALKING_V2_TASK_ID,
    SE2_TRACKING_TASK_ID,
)


COMMANDS: dict[str, tuple[float, float, float]] = {
    "stop": (0.0, 0.0, 0.0),
    "forward": (0.08, 0.0, 0.0),
    "left": (0.0, 0.06, 0.0),
    "right": (0.0, -0.06, 0.0),
    "curve_left": (0.08, 0.0, 0.30),
    "curve_right": (0.08, 0.0, -0.30),
    "backward": (-0.05, 0.0, 0.0),
}

V3_COMMANDS: dict[str, tuple[float, float, float]] = {
    "stop": COMMANDS["stop"],
    "forward": COMMANDS["forward"],
    "backward": COMMANDS["backward"],
    "left": COMMANDS["left"],
    "right": COMMANDS["right"],
    "curve_left": COMMANDS["curve_left"],
    "curve_right": COMMANDS["curve_right"],
}

V4_COMMANDS: dict[str, tuple[float, float, float]] = {
    name: COMMANDS[name]
    for name in ("stop", "forward", "curve_left", "curve_right")
}

AXIS_EXCLUSIVE_COMMANDS: dict[str, tuple[float, float, float]] = {
    name: COMMANDS[name] for name in ("stop", "forward", "left", "right")
}

SUPPORTED_TASKS = {
    NATURAL_OMNIDIRECTIONAL_TASK_ID,
    NATURAL_OMNIDIRECTIONAL_V3_TASK_ID,
    NATURAL_OMNIDIRECTIONAL_V4_TASK_ID,
    NATURAL_OMNIDIRECTIONAL_V5_TASK_ID,
    PERIODIC_REWARD_COMPOSITION_WALKING_TASK_ID,
    QUASI_OMNIDIRECTIONAL_WALKING_TASK_ID,
    QUASI_OMNIDIRECTIONAL_WALKING_V2_TASK_ID,
    SE2_TRACKING_TASK_ID,
}


def commands_for_task(task_id: str) -> dict[str, tuple[float, float, float]]:
    """Return only command modes that the selected task actually trains."""
    if task_id == PERIODIC_REWARD_COMPOSITION_WALKING_TASK_ID:
        return AXIS_EXCLUSIVE_COMMANDS
    if task_id == NATURAL_OMNIDIRECTIONAL_V3_TASK_ID:
        return V3_COMMANDS
    if task_id in (NATURAL_OMNIDIRECTIONAL_V4_TASK_ID, NATURAL_OMNIDIRECTIONAL_V5_TASK_ID):
        return V4_COMMANDS
    return COMMANDS


@dataclass(frozen=True)
class Objectives:
    minimum_survival: float = 0.95
    maximum_linear_error_mps: float = 0.03
    maximum_yaw_error_radps: float = 0.10
    maximum_path_rmse_m: float = 0.20
    maximum_heading_rmse_rad: float = 0.30
    minimum_step_fraction: float = 0.80
    minimum_liftoffs_per_foot: int = 2
    maximum_stop_speed_mps: float = 0.015
    maximum_stop_yaw_radps: float = 0.05
    maximum_stop_joint_rmse_rad: float = 0.12
    maximum_stop_foot_offset_m: float = 0.025
    maximum_stop_liftoffs_per_foot: int = 1
    maximum_mean_abs_base_pitch_deg: float = 8.0
    maximum_rms_base_pitch_deg: float = 12.0
    maximum_lateral_foot_line_error_m: float = 0.025


def checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("model_"))
    except ValueError as exc:
        raise ValueError(f"Checkpoint does not have a model_<iteration>.pt name: {path}") from exc


def discover_checkpoints(
    run_dir: Path, *, start: int, end: int | None, stride: int
) -> list[Path]:
    if stride <= 0:
        raise ValueError("checkpoint stride must be positive")
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=checkpoint_iteration)
    selected = []
    for checkpoint in checkpoints:
        iteration = checkpoint_iteration(checkpoint)
        if iteration < start or (end is not None and iteration > end):
            continue
        if (iteration - start) % stride == 0 or checkpoint == checkpoints[-1]:
            selected.append(checkpoint)
    if not selected:
        raise FileNotFoundError(f"No selected model_*.pt checkpoints in {run_dir}")
    return selected


def _contact_mask(env) -> torch.Tensor:
    found = env.scene["feet_ground_contact"].data.found
    if found is None:
        raise RuntimeError("feet_ground_contact does not expose contact matches")
    contact = found > 0
    while contact.ndim > 2:
        contact = torch.any(contact, dim=-1)
    if contact.shape[1] != 2:
        raise RuntimeError("run evaluation requires exactly two feet")
    return contact


def _mode_limits(
    name: str,
    objectives: Objectives,
    *,
    include_lateral_foot_line: bool = False,
) -> dict[str, float]:
    if name == "stop":
        return {
            "survival": objectives.minimum_survival,
            "stop_speed": objectives.maximum_stop_speed_mps,
            "stop_yaw": objectives.maximum_stop_yaw_radps,
            "joint_rmse": objectives.maximum_stop_joint_rmse_rad,
            "foot_offset": objectives.maximum_stop_foot_offset_m,
            "stop_liftoffs": float(objectives.maximum_stop_liftoffs_per_foot),
            "base_pitch_bias": objectives.maximum_mean_abs_base_pitch_deg,
            "base_pitch_rms": objectives.maximum_rms_base_pitch_deg,
        }
    limits = {
        "survival": objectives.minimum_survival,
        "linear_error": objectives.maximum_linear_error_mps,
        "yaw_error": objectives.maximum_yaw_error_radps,
        "path_rmse": objectives.maximum_path_rmse_m,
        "heading_rmse": objectives.maximum_heading_rmse_rad,
        "step_fraction": objectives.minimum_step_fraction,
        "base_pitch_bias": objectives.maximum_mean_abs_base_pitch_deg,
        "base_pitch_rms": objectives.maximum_rms_base_pitch_deg,
    }
    if include_lateral_foot_line and name in ("left", "right"):
        limits["lateral_foot_line"] = objectives.maximum_lateral_foot_line_error_m
    return limits


def judge_mode(
    name: str,
    metrics: dict[str, Any],
    objectives: Objectives,
    *,
    include_lateral_foot_line: bool = False,
) -> dict[str, Any]:
    limits = _mode_limits(
        name, objectives, include_lateral_foot_line=include_lateral_foot_line
    )
    if name == "stop":
        checks = {
            "survival": metrics["survival"] >= limits["survival"],
            "stop_speed": metrics["planar_speed_mps"] <= limits["stop_speed"],
            "stop_yaw": abs(metrics["mean_velocity"][2]) <= limits["stop_yaw"],
            "joint_rmse": metrics["joint_rmse_rad"] <= limits["joint_rmse"],
            "foot_offset": metrics["foot_fore_aft_offset_m"] <= limits["foot_offset"],
            "stop_liftoffs": metrics["maximum_mean_liftoffs_per_foot"] <= limits["stop_liftoffs"],
            "base_pitch_bias": abs(metrics["mean_base_pitch_deg"])
            <= limits["base_pitch_bias"],
            "base_pitch_rms": metrics["rms_base_pitch_deg"] <= limits["base_pitch_rms"],
        }
        ratios = {
            "survival": limits["survival"] / max(metrics["survival"], 1e-9),
            "stop_speed": metrics["planar_speed_mps"] / limits["stop_speed"],
            "stop_yaw": abs(metrics["mean_velocity"][2]) / limits["stop_yaw"],
            "joint_rmse": metrics["joint_rmse_rad"] / limits["joint_rmse"],
            "foot_offset": metrics["foot_fore_aft_offset_m"] / limits["foot_offset"],
            "stop_liftoffs": metrics["maximum_mean_liftoffs_per_foot"]
            / max(limits["stop_liftoffs"], 1.0),
            "base_pitch_bias": abs(metrics["mean_base_pitch_deg"])
            / limits["base_pitch_bias"],
            "base_pitch_rms": metrics["rms_base_pitch_deg"] / limits["base_pitch_rms"],
        }
    else:
        checks = {
            "survival": metrics["survival"] >= limits["survival"],
            "linear_error": metrics["linear_error_mps"] <= limits["linear_error"],
            "yaw_error": metrics["yaw_error_radps"] <= limits["yaw_error"],
            "path_rmse": metrics["path_rmse_m"] <= limits["path_rmse"],
            "heading_rmse": metrics["heading_rmse_rad"] <= limits["heading_rmse"],
            "step_fraction": metrics["bilateral_step_fraction"] >= limits["step_fraction"],
            "base_pitch_bias": abs(metrics["mean_base_pitch_deg"])
            <= limits["base_pitch_bias"],
            "base_pitch_rms": metrics["rms_base_pitch_deg"] <= limits["base_pitch_rms"],
        }
        ratios = {
            "survival": limits["survival"] / max(metrics["survival"], 1e-9),
            "linear_error": metrics["linear_error_mps"] / limits["linear_error"],
            "yaw_error": metrics["yaw_error_radps"] / limits["yaw_error"],
            "path_rmse": metrics["path_rmse_m"] / limits["path_rmse"],
            "heading_rmse": metrics["heading_rmse_rad"] / limits["heading_rmse"],
            "step_fraction": limits["step_fraction"]
            / max(metrics["bilateral_step_fraction"], 1e-9),
            "base_pitch_bias": abs(metrics["mean_base_pitch_deg"])
            / limits["base_pitch_bias"],
            "base_pitch_rms": metrics["rms_base_pitch_deg"] / limits["base_pitch_rms"],
        }
        if "lateral_foot_line" in limits:
            checks["lateral_foot_line"] = (
                metrics["lateral_foot_line_error_m"] <= limits["lateral_foot_line"]
            )
            ratios["lateral_foot_line"] = (
                metrics["lateral_foot_line_error_m"] / limits["lateral_foot_line"]
            )
    worst = max(ratios, key=ratios.get)
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "limits": limits,
        "worst_check": worst,
        "worst_ratio": ratios[worst],
    }


def evaluate_checkpoint(
    env,
    actor: torch.nn.Module,
    commands: torch.Tensor,
    mode_commands: dict[str, tuple[float, float, float]],
    *,
    seconds: float,
    warmup_s: float,
    replicas: int,
    objectives: Objectives,
    include_lateral_foot_line: bool = False,
) -> dict[str, Any]:
    observations, _ = env.reset()
    command_term = env.command_manager.get_term("twist")
    command_term.vel_command_b.copy_(commands)
    command_term.vel_command_w.copy_(commands)
    command_term.time_left.fill_(1.0e9)
    obs = observations["actor"]
    robot = env.scene["robot"].data
    count_envs = len(commands)
    start_xy = robot.root_link_pos_w[:, :2].clone()
    heading = robot.heading_w.clone()
    unwrapped_heading = heading.clone()
    reference_xy = start_xy.clone()
    reference_heading = heading.clone()
    alive = torch.ones(count_envs, dtype=torch.bool, device=env.device)
    survived_s = torch.zeros(count_envs, device=env.device)
    sample_count = torch.zeros(count_envs, device=env.device)
    velocity_sum = torch.zeros(count_envs, 3, device=env.device)
    path_squared = torch.zeros(count_envs, device=env.device)
    heading_squared = torch.zeros(count_envs, device=env.device)
    base_pitch_sum = torch.zeros(count_envs, device=env.device)
    base_pitch_squared = torch.zeros(count_envs, device=env.device)
    maximum_abs_base_pitch = torch.zeros(count_envs, device=env.device)
    base_roll_sum = torch.zeros(count_envs, device=env.device)
    maximum_abs_base_roll = torch.zeros(count_envs, device=env.device)
    joint_squared = torch.zeros(count_envs, device=env.device)
    joint_names = tuple(env.scene["robot"].joint_names)
    joint_count = len(joint_names)
    soft_limit_violation_sum = torch.zeros(count_envs, joint_count, device=env.device)
    soft_limit_violation_count = torch.zeros(count_envs, joint_count, device=env.device)
    minimum_physical_limit_margin = torch.full(
        (count_envs, joint_count), torch.inf, device=env.device
    )
    foot_offset_sum = torch.zeros(count_envs, device=env.device)
    lateral_foot_line_sum = torch.zeros(count_envs, device=env.device)
    liftoffs = torch.zeros(count_envs, 2, device=env.device)
    previous_contact = _contact_mask(env).clone()
    foot_site_ids = env.scene["robot"].find_sites(
        ("right_foot", "left_foot"), preserve_order=True
    )[0]
    steps = round(seconds / env.step_dt)

    for step in range(steps):
        with torch.no_grad():
            action = actor(obs).clamp(-1.0, 1.0)
            observations, _, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        alive &= ~done
        survived_s += alive * env.step_dt
        current_heading = robot.heading_w
        heading_delta = torch.atan2(
            torch.sin(current_heading - heading), torch.cos(current_heading - heading)
        )
        unwrapped_heading += heading_delta
        heading = current_heading.clone()

        midpoint = reference_heading + 0.5 * commands[:, 2] * env.step_dt
        reference_xy[:, 0] += (
            midpoint.cos() * commands[:, 0] - midpoint.sin() * commands[:, 1]
        ) * env.step_dt
        reference_xy[:, 1] += (
            midpoint.sin() * commands[:, 0] + midpoint.cos() * commands[:, 1]
        ) * env.step_dt
        reference_heading += commands[:, 2] * env.step_dt

        contact = _contact_mask(env)
        valid = alive & ((step + 1) * env.step_dt >= warmup_s)
        liftoffs += (previous_contact & ~contact & valid[:, None]).float()
        previous_contact = contact.clone()
        world_velocity = robot.root_link_lin_vel_w
        velocity_b = torch.stack(
            (
                heading.cos() * world_velocity[:, 0] + heading.sin() * world_velocity[:, 1],
                -heading.sin() * world_velocity[:, 0] + heading.cos() * world_velocity[:, 1],
                robot.root_link_ang_vel_w[:, 2],
            ),
            dim=1,
        )
        sample_count += valid
        velocity_sum += velocity_b * valid[:, None]
        path_squared += (
            torch.square(robot.root_link_pos_w[:, :2] - reference_xy).sum(dim=1) * valid
        )
        heading_squared += torch.square(unwrapped_heading - reference_heading) * valid
        gravity_b = robot.projected_gravity_b
        base_pitch = torch.atan2(gravity_b[:, 0], -gravity_b[:, 2])
        base_roll = torch.atan2(gravity_b[:, 1], -gravity_b[:, 2])
        base_pitch_sum += base_pitch * valid
        base_pitch_squared += torch.square(base_pitch) * valid
        maximum_abs_base_pitch = torch.maximum(
            maximum_abs_base_pitch, base_pitch.abs() * valid
        )
        base_roll_sum += base_roll * valid
        maximum_abs_base_roll = torch.maximum(
            maximum_abs_base_roll, base_roll.abs() * valid
        )
        joint_error = robot.joint_pos - robot.default_joint_pos
        joint_squared += torch.mean(torch.square(joint_error), dim=1) * valid
        soft_limits = robot.soft_joint_pos_limits
        physical_limits = robot.joint_pos_limits
        if soft_limits is None or physical_limits is None:
            raise RuntimeError("robot does not expose joint position limits")
        soft_limit_violation = torch.relu(soft_limits[..., 0] - robot.joint_pos)
        soft_limit_violation += torch.relu(robot.joint_pos - soft_limits[..., 1])
        soft_limit_violation_sum += soft_limit_violation * valid[:, None]
        soft_limit_violation_count += (soft_limit_violation > 0.0) * valid[:, None]
        physical_limit_margin = torch.minimum(
            robot.joint_pos - physical_limits[..., 0],
            physical_limits[..., 1] - robot.joint_pos,
        )
        minimum_physical_limit_margin = torch.minimum(
            minimum_physical_limit_margin,
            torch.where(valid[:, None], physical_limit_margin, torch.inf),
        )
        foot_sites = env.scene["robot"].data.site_pos_w[:, foot_site_ids]
        separation = foot_sites[:, 0] - foot_sites[:, 1]
        fore_aft = heading.cos() * separation[:, 0] + heading.sin() * separation[:, 1]
        foot_offset_sum += fore_aft.abs() * valid
        pure_lateral = (
            (commands[:, 1].abs() >= 0.04)
            & (commands[:, 0].abs() <= 0.01)
            & (commands[:, 2].abs() <= 0.02)
        )
        lateral_foot_line_sum += fore_aft.abs() * valid * pure_lateral
        obs = observations["actor"]

        done_ids = done.nonzero(as_tuple=False).flatten()
        if len(done_ids):
            reset_obs, _ = env.reset(env_ids=done_ids)
            command_term.vel_command_b[done_ids] = commands[done_ids]
            command_term.vel_command_w[done_ids] = commands[done_ids]
            command_term.time_left[done_ids] = 1.0e9
            obs = observations["actor"].clone()
            obs[done_ids] = reset_obs["actor"][done_ids]
            previous_contact[done_ids] = _contact_mask(env)[done_ids]

    results: dict[str, Any] = {}
    for mode_index, (name, command) in enumerate(mode_commands.items()):
        ids = slice(mode_index * replicas, (mode_index + 1) * replicas)
        samples = sample_count[ids].sum()
        if samples <= 0:
            metrics = {
                "command": list(command),
                "survival": alive[ids].float().mean().item(),
                "mean_survived_s": survived_s[ids].mean().item(),
                "valid_samples": 0,
            }
            results[name] = {
                "metrics": metrics,
                "judgement": {"passed": False, "worst_check": "survival", "worst_ratio": math.inf},
            }
            continue
        mean_velocity = velocity_sum[ids].sum(dim=0) / samples
        linear_error = torch.linalg.vector_norm(
            mean_velocity[:2] - mean_velocity.new_tensor(command[:2])
        )
        yaw_error = torch.abs(mean_velocity[2] - command[2])
        minimum_liftoffs = objectives.minimum_liftoffs_per_foot
        stepped = torch.all(liftoffs[ids] >= minimum_liftoffs, dim=1)
        metrics = {
            "command": list(command),
            "survival": alive[ids].float().mean().item(),
            "mean_survived_s": survived_s[ids].mean().item(),
            "valid_samples": int(samples.item()),
            "mean_velocity": mean_velocity.tolist(),
            "planar_speed_mps": torch.linalg.vector_norm(mean_velocity[:2]).item(),
            "linear_error_mps": linear_error.item(),
            "yaw_error_radps": yaw_error.item(),
            "path_rmse_m": torch.sqrt(path_squared[ids].sum() / samples).item(),
            "heading_rmse_rad": torch.sqrt(heading_squared[ids].sum() / samples).item(),
            "mean_base_pitch_deg": torch.rad2deg(
                base_pitch_sum[ids].sum() / samples
            ).item(),
            "rms_base_pitch_deg": torch.rad2deg(
                torch.sqrt(base_pitch_squared[ids].sum() / samples)
            ).item(),
            "maximum_abs_base_pitch_deg": torch.rad2deg(
                maximum_abs_base_pitch[ids].max()
            ).item(),
            "mean_base_roll_deg": torch.rad2deg(
                base_roll_sum[ids].sum() / samples
            ).item(),
            "maximum_abs_base_roll_deg": torch.rad2deg(
                maximum_abs_base_roll[ids].max()
            ).item(),
            "joint_rmse_rad": torch.sqrt(joint_squared[ids].sum() / samples).item(),
            "mean_soft_limit_violation_rad_by_joint": dict(
                zip(
                    joint_names,
                    (soft_limit_violation_sum[ids].sum(dim=0) / samples).tolist(),
                    strict=True,
                )
            ),
            "soft_limit_violation_fraction_by_joint": dict(
                zip(
                    joint_names,
                    (soft_limit_violation_count[ids].sum(dim=0) / samples).tolist(),
                    strict=True,
                )
            ),
            "minimum_physical_limit_margin_deg_by_joint": dict(
                zip(
                    joint_names,
                    torch.rad2deg(minimum_physical_limit_margin[ids].amin(dim=0)).tolist(),
                    strict=True,
                )
            ),
            "foot_fore_aft_offset_m": (foot_offset_sum[ids].sum() / samples).item(),
            "lateral_foot_line_error_m": (
                lateral_foot_line_sum[ids].sum() / samples
            ).item(),
            "mean_liftoffs_per_foot": liftoffs[ids].mean(dim=0).tolist(),
            "maximum_mean_liftoffs_per_foot": liftoffs[ids].mean(dim=0).max().item(),
            "bilateral_step_fraction": stepped.float().mean().item(),
        }
        results[name] = {
            "metrics": metrics,
            "judgement": judge_mode(
                name,
                metrics,
                objectives,
                include_lateral_foot_line=include_lateral_foot_line,
            ),
        }
    return results


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Checkpoint evaluation: {Path(report['run_dir']).name}",
        "",
        f"Result: **{'PASS' if report['run_passed'] else 'FAIL'}**",
        "",
        f"Task: `{report['task_id']}`  ",
        f"Best checkpoint: `{report['best_checkpoint']}`  ",
        f"Selected checkpoints: {len(report['checkpoints'])}",
        "",
        "| Iteration | Result | Modes passed | Worst mode | Worst check | Worst ratio |",
        "|---:|:---:|---:|---|---|---:|",
    ]
    for checkpoint in report["checkpoints"]:
        lines.append(
            f"| {checkpoint['iteration']} | {'PASS' if checkpoint['passed'] else 'FAIL'} "
            f"| {checkpoint['modes_passed']}/{report['mode_count']} | {checkpoint['worst_mode']} "
            f"| {checkpoint['worst_check']} | {checkpoint['worst_ratio']:.2f} |"
        )
    lines.extend(("", "A ratio at or below 1.0 passes that objective. The run passes only when one checkpoint passes every configured mode.", ""))
    best = next(item for item in report["checkpoints"] if item["checkpoint"] == report["best_checkpoint"])
    lines.extend(("## Best checkpoint modes", "", "| Mode | Result | Survival | Velocity [vx, vy, yaw] | Path RMSE | Lateral line | Mean pitch | RMS pitch | Steps |", "|---|:---:|---:|---|---:|---:|---:|---:|---:|"))
    for name, mode in best["modes"].items():
        metrics = mode["metrics"]
        velocity = metrics.get("mean_velocity")
        velocity_text = "n/a" if velocity is None else "[" + ", ".join(f"{v:.3f}" for v in velocity) + "]"
        lines.append(
            f"| {name} | {'PASS' if mode['judgement']['passed'] else 'FAIL'} "
            f"| {metrics['survival']:.1%} | {velocity_text} "
            f"| {metrics.get('path_rmse_m', float('nan')):.3f} "
            f"| {metrics.get('lateral_foot_line_error_m', float('nan')):.3f} "
            f"| {metrics.get('mean_base_pitch_deg', float('nan')):.1f} deg "
            f"| {metrics.get('rms_base_pitch_deg', float('nan')):.1f} deg "
            f"| {metrics.get('bilateral_step_fraction', 0.0):.1%} |"
        )
    lines.extend(
        (
            "",
            "## Best checkpoint joint-limit pressure",
            "",
            "Mean excess is the distance beyond the task's soft limit. The minimum margin is measured against the physical limit.",
            "",
            "| Mode | Joint | Samples beyond soft limit | Mean excess | Minimum physical margin |",
            "|---|---|---:|---:|---:|",
        )
    )
    for name, mode in best["modes"].items():
        metrics = mode["metrics"]
        fractions = metrics.get("soft_limit_violation_fraction_by_joint", {})
        excesses = metrics.get("mean_soft_limit_violation_rad_by_joint", {})
        margins = metrics.get("minimum_physical_limit_margin_deg_by_joint", {})
        for joint_name, fraction in fractions.items():
            lines.append(
                f"| {name} | {joint_name} | {fraction:.1%} "
                f"| {math.degrees(excesses[joint_name]):.2f} deg "
                f"| {margins[joint_name]:.2f} deg |"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--task", help="Defaults to the checkpoint's task_id.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument(
        "--forward-speed", type=float,
        help="Override vx for forward and curve modes (useful for speed curricula).",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if args.seconds <= 1.0 or not 0.0 <= args.warmup < args.seconds:
        raise ValueError("seconds must exceed one and warmup must be within the horizon")
    if args.replicas <= 0:
        raise ValueError("replicas must be positive")
    checkpoints = discover_checkpoints(run_dir, start=args.start, end=args.end, stride=args.stride)
    device_name = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    first_actor, checkpoint_task, _ = _load_actor(checkpoints[0], device)
    task_id = args.task or checkpoint_task
    if task_id is None:
        raise ValueError("Checkpoint has no task_id; provide --task")
    if task_id not in SUPPORTED_TASKS:
        raise ValueError(f"{task_id!r} is not an omnidirectional evaluation task")

    replicas = args.replicas
    mode_commands = commands_for_task(task_id)
    if args.forward_speed is not None:
        if args.forward_speed <= 0.0:
            raise ValueError("--forward-speed must be positive")
        mode_commands = {
            name: (
                (args.forward_speed, command[1], command[2])
                if name in ("forward", "curve_left", "curve_right")
                else command
            )
            for name, command in mode_commands.items()
        }
    commands = torch.tensor(list(mode_commands.values()), device=device).repeat_interleave(
        replicas, dim=0
    )
    cfg = make_env_cfg(
        task_id, num_envs=len(commands), seed=123, play=True,
        explicit_resets=True, sim2real_randomization=False,
    )
    cfg.episode_length_s = max(cfg.episode_length_s, args.seconds + 1.0)
    cfg.commands["twist"].resampling_time_range = (1.0e9, 1.0e9)
    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(cfg, device=str(device))
    objectives = (
        Objectives(maximum_linear_error_mps=.02, maximum_yaw_error_radps=.04,
                   maximum_path_rmse_m=.06, maximum_heading_rmse_rad=.12)
        if task_id in (NATURAL_OMNIDIRECTIONAL_V4_TASK_ID, NATURAL_OMNIDIRECTIONAL_V5_TASK_ID)
        else Objectives()
    )
    evaluated = []
    try:
        for index, checkpoint in enumerate(checkpoints):
            if index == 0:
                actor = first_actor
                current_task = checkpoint_task
            else:
                actor, current_task, _ = _load_actor(checkpoint, device)
            if current_task is not None and current_task != task_id:
                raise ValueError(f"{checkpoint} belongs to task {current_task!r}, expected {task_id!r}")
            modes = evaluate_checkpoint(
                env, actor, commands, mode_commands, seconds=args.seconds, warmup_s=args.warmup,
                replicas=replicas, objectives=objectives,
                include_lateral_foot_line=(task_id in (
                    NATURAL_OMNIDIRECTIONAL_V3_TASK_ID, NATURAL_OMNIDIRECTIONAL_V4_TASK_ID)),
            )
            mode_ratios = {name: mode["judgement"]["worst_ratio"] for name, mode in modes.items()}
            worst_mode = max(mode_ratios, key=mode_ratios.get)
            item = {
                "checkpoint": checkpoint.name,
                "iteration": checkpoint_iteration(checkpoint),
                "passed": all(mode["judgement"]["passed"] for mode in modes.values()),
                "modes_passed": sum(mode["judgement"]["passed"] for mode in modes.values()),
                "worst_mode": worst_mode,
                "worst_check": modes[worst_mode]["judgement"]["worst_check"],
                "worst_ratio": mode_ratios[worst_mode],
                "modes": modes,
            }
            evaluated.append(item)
            print(
                f"iteration={item['iteration']} result={'PASS' if item['passed'] else 'FAIL'} "
                f"modes={item['modes_passed']}/{len(mode_commands)} worst={worst_mode}:"
                f"{item['worst_check']} ratio={item['worst_ratio']:.2f}",
                flush=True,
            )
    finally:
        env.close()

    best = min(evaluated, key=lambda item: (-item["modes_passed"], item["worst_ratio"]))
    report = {
        "run_dir": str(run_dir),
        "task_id": task_id,
        "device": str(device),
        "seconds": args.seconds,
        "warmup_s": args.warmup,
        "replicas": replicas,
        "objectives": asdict(objectives),
        "mode_count": len(mode_commands),
        "run_passed": any(item["passed"] for item in evaluated),
        "best_checkpoint": best["checkpoint"],
        "checkpoints": evaluated,
    }
    output = args.output or Path("artifacts/omni_diagnostics") / f"{run_dir.name}_evaluation.json"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    markdown = output.with_suffix(".md")
    markdown.write_text(_markdown(report))
    print(f"run_result={'PASS' if report['run_passed'] else 'FAIL'}")
    print(f"best_checkpoint={best['checkpoint']}")
    print(f"json={output}")
    print(f"report={markdown}")
    return 0 if report["run_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
