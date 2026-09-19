"""Deterministically evaluate any implemented locomotion task."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from locomotion import DEFAULT_TASK_ID
from locomotion.ppo import load_custom_policy
from locomotion.task import make_env_cfg
from locomotion.tasks import (
    LOCOMOTION_TASK_IDS,
    get_locomotion_task,
)


def _rsl_actor(checkpoint: dict) -> torch.nn.Module:
    from locomotion.native_policy import _RslActor

    return _RslActor(checkpoint["actor_state_dict"])


def _load_actor(
    path: Path, device: torch.device
) -> tuple[torch.nn.Module, str | None, int]:
    checkpoint = torch.load(path.expanduser().resolve(), map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model, checkpoint = load_custom_policy(path.expanduser().resolve(), device)
        common_step_counter = checkpoint.get("infos", {}).get("env_state", {}).get(
            "common_step_counter", 0
        )
        return model, checkpoint.get("task_id"), int(common_step_counter)
    if "actor_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    common_step_counter = checkpoint.get("infos", {}).get("env_state", {}).get(
        "common_step_counter", 0
    )
    return (
        _rsl_actor(checkpoint).to(device).eval(),
        checkpoint.get("task_id"),
        int(common_step_counter),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--task", default=DEFAULT_TASK_ID)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-sim2real-randomization", action="store_true")
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return result


def main() -> int:
    args = parse_args()
    if args.episodes <= 0 or args.num_envs <= 0:
        raise ValueError("episodes and num-envs must be positive")
    if args.task not in LOCOMOTION_TASK_IDS:
        raise ValueError("--task must be an implemented locomotion task")
    task = get_locomotion_task(args.task)
    device = _device(args.device)
    actor, checkpoint_task, common_step_counter = _load_actor(args.checkpoint, device)
    if checkpoint_task is not None and checkpoint_task != args.task:
        raise ValueError(
            f"checkpoint task {checkpoint_task!r} does not match --task {args.task!r}"
        )
    env_cfg = make_env_cfg(
        args.task,
        num_envs=args.num_envs,
        seed=args.seed,
        explicit_resets=True,
        sim2real_randomization=not args.no_sim2real_randomization,
    )
    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(cfg=env_cfg, device=str(device))
    try:
        env.common_step_counter = common_step_counter
        observations, _ = env.reset(seed=args.seed)
        actor_obs = observations["actor"]
        accum = torch.zeros(args.num_envs, device=device)
        active_counts = torch.zeros(args.num_envs, device=device)
        survival: list[float] = []
        values: list[float] = []
        while len(values) < args.episodes:
            with torch.inference_mode():
                if hasattr(actor, "deterministic"):
                    actions = actor.deterministic(actor_obs)
                else:
                    actions = actor(actor_obs)
            observations, _, terminated, truncated, _ = env.step(actions.clamp(-1.0, 1.0))
            metric = task.success_metric(env)
            active = task.active_metric(env)
            accum += metric * active
            active_counts += active
            done = terminated | truncated
            ids = done.nonzero(as_tuple=False).flatten()
            if len(ids):
                values.extend((accum[ids] / active_counts[ids].clamp(min=1.0)).detach().cpu().tolist())
                survival.extend((~terminated[ids]).float().detach().cpu().tolist())
                accum[ids] = 0.0
                active_counts[ids] = 0.0
                reset_obs, _ = env.reset(env_ids=ids)
                actor_obs = observations["actor"].clone()
                actor_obs[ids] = reset_obs["actor"][ids]
            else:
                actor_obs = observations["actor"]
        values = values[: args.episodes]
        survival = survival[: args.episodes]
        score = sum(values) / len(values)
        survive = sum(survival) / len(survival)
        threshold = task.success_threshold
        passed = score >= threshold and survive >= 0.90
        print(f"task={args.task}")
        print(f"episodes={len(values)} metric={score:.3f} threshold={threshold:.3f}")
        print(f"survival={survive:.3f} threshold=0.900")
        print(f"RESULT={'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
