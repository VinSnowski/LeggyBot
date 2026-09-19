"""Fixed-command, finite-horizon diagnostics independent of training success gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from locomotion.evaluate_task import _load_actor
from locomotion.task import make_env_cfg

COMMANDS = {
    "stop": (0., 0., 0.),
    "forward": (.08, 0., 0.),
    "left": (0., .06, 0.),
    "right": (0., -.06, 0.),
    "curve_left": (.08, 0., .3),
    "curve_right": (.08, 0., -.3),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--task", default="WalkingTask")
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 1 or args.replicas < 1:
        parser.error("seconds must exceed 1 and replicas must be positive")
    device = torch.device(args.device)
    actor, _, _ = _load_actor(args.checkpoint, device)
    commands = torch.tensor(list(COMMANDS.values()), device=device).repeat_interleave(args.replicas, 0)
    cfg = make_env_cfg(args.task, num_envs=len(commands), seed=123, play=True,
                       explicit_resets=True, sim2real_randomization=False)
    # A requested diagnostic horizon must not be shortened by the task timeout.
    cfg.episode_length_s = max(cfg.episode_length_s, args.seconds + 1.)
    cfg.commands["twist"].resampling_time_range = (1.e9, 1.e9)
    from mjlab.envs import ManagerBasedRlEnv
    env = ManagerBasedRlEnv(cfg, device=str(device))
    try:
        env.reset(seed=123)
        term = env.command_manager.get_term("twist")
        term.vel_command_b.copy_(commands)
        term.time_left.fill_(1.e9)
        obs = env.observation_manager.compute()["actor"]
        robot = env.scene["robot"].data
        start = robot.root_link_pos_w[:, :2].clone()
        heading = robot.heading_w.clone()
        reference_xy = start.clone()
        reference_heading = heading.clone()
        unwrapped = heading.clone()
        alive = torch.ones(len(commands), dtype=torch.bool, device=device)
        count = torch.zeros(len(commands), device=device)
        velocity = torch.zeros(len(commands), 3, device=device)
        squared_error = velocity.clone()
        path_sq = count.clone()
        yaw_sq = count.clone()
        survived_s = count.clone()
        for step in range(round(args.seconds / env.step_dt)):
            with torch.no_grad():
                action = actor.deterministic(obs) if hasattr(actor, "deterministic") else actor(obs)
                observations, _, terminated, truncated, _ = env.step(action.clamp(-1, 1))
            alive &= ~(terminated | truncated)
            survived_s += alive * env.step_dt
            new_heading = robot.heading_w
            unwrapped += torch.atan2(torch.sin(new_heading-heading), torch.cos(new_heading-heading))
            heading = new_heading.clone()
            mid = reference_heading + .5 * commands[:, 2] * env.step_dt
            reference_xy[:, 0] += (mid.cos()*commands[:, 0]-mid.sin()*commands[:, 1])*env.step_dt
            reference_xy[:, 1] += (mid.sin()*commands[:, 0]+mid.cos()*commands[:, 1])*env.step_dt
            reference_heading += commands[:, 2]*env.step_dt
            # Heading-frame velocity avoids mixing torso lean with planar travel.
            world = robot.root_link_lin_vel_w
            actual = torch.stack((heading.cos()*world[:, 0]+heading.sin()*world[:, 1],
                                  -heading.sin()*world[:, 0]+heading.cos()*world[:, 1],
                                  robot.root_link_ang_vel_w[:, 2]), -1)
            valid = alive & ((step+1)*env.step_dt >= 1.)
            count += valid
            velocity += actual * valid[:, None]
            squared_error += (actual-commands).square()*valid[:, None]
            path_sq += (robot.root_link_pos_w[:, :2]-reference_xy).square().sum(-1)*valid
            yaw_sq += (unwrapped-reference_heading).square()*valid
            obs = observations["actor"]
            done_ids = (terminated | truncated).nonzero().flatten()
            if len(done_ids):
                # Reset failed simulations for the engine, but keep them excluded
                # from every subsequent measurement and the survival denominator.
                env.reset(env_ids=done_ids)
                term.vel_command_b.copy_(commands)
                term.time_left.fill_(1.e9)
                obs = env.observation_manager.compute()["actor"]
        result = {"checkpoint": str(args.checkpoint), "task": args.task,
                  "seconds": args.seconds, "replicas": args.replicas,
                  "seed": 123, "randomization": False, "commands": {}}
        for i, name in enumerate(COMMANDS):
            ids = slice(i*args.replicas, (i+1)*args.replicas)
            n = count[ids].sum().clamp(min=1)
            result["commands"][name] = {
                "command": list(COMMANDS[name]),
                "survival": alive[ids].float().mean().item(),
                "mean_survived_s": survived_s[ids].mean().item(),
                "valid_samples": int(count[ids].sum().item()),
                "mean_velocity": (velocity[ids].sum(0)/n).tolist(),
                "velocity_rmse": (squared_error[ids].sum(0)/n).sqrt().tolist(),
                "path_rmse_m": (path_sq[ids].sum()/n).sqrt().item(),
                "heading_rmse_rad": (yaw_sq[ids].sum()/n).sqrt().item(),
            }
            if not count[ids].sum():
                # A policy that falls before warmup has no tracking estimate.
                for key in ("mean_velocity", "velocity_rmse", "path_rmse_m", "heading_rmse_rad"):
                    result["commands"][name][key] = None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
