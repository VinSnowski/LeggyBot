"""Train the shared mjlab velocity task with this repository's PPO."""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict, deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from locomotion import DEFAULT_TASK_ID
from locomotion.export import export_custom_policy
from locomotion.ppo import ActorCritic, PPO, PPOConfig, RolloutStorage, load_custom_policy
from locomotion.task import make_env_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK_ID)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--log-root", type=Path, default=Path("logs/custom_ppo"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initialize-from", type=Path,
        help="Initialize the actor from a previous custom-PPO checkpoint; critic/optimizer are fresh.",
    )
    parser.add_argument("--no-sim2real-randomization", action="store_true")
    parser.add_argument("--no-onnx", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _obs_tensor(observations: dict, group: str) -> torch.Tensor:
    tensor = observations[group]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Observation group {group!r} must be concatenated into a tensor")
    return tensor


def train(args: argparse.Namespace) -> Path:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    if args.num_envs <= 0 or args.iterations <= 0 or args.rollout_steps <= 0:
        raise ValueError("num-envs, iterations and rollout-steps must be positive")

    configure_torch_backends()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("[WARN] mjlab training is intended for an NVIDIA GPU; CPU will be very slow.")

    env_cfg = make_env_cfg(
        args.task,
        num_envs=args.num_envs,
        seed=args.seed,
        explicit_resets=True,
        sim2real_randomization=not args.no_sim2real_randomization,
    )
    env_cfg.sim.nan_guard.enabled = True
    env = ManagerBasedRlEnv(cfg=env_cfg, device=str(device))

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{args.run_name}" if args.run_name else ""
    log_dir = (args.log_root / f"{stamp}{suffix}").resolve()
    checkpoint_dir = log_dir / "checkpoints"
    writer = SummaryWriter(log_dir=str(log_dir))

    try:
        observations, _ = env.reset(seed=args.seed)
        actor_obs = _obs_tensor(observations, "actor")
        critic_obs = _obs_tensor(observations, "critic")
        action_dim = env.action_manager.total_action_dim
        config = PPOConfig(rollout_steps=args.rollout_steps)

        if args.resume is not None and args.initialize_from is not None:
            raise ValueError("--resume and --initialize-from are mutually exclusive")

        if args.resume is not None:
            model, checkpoint = load_custom_policy(args.resume, device)
            if checkpoint["task_id"] != args.task:
                raise ValueError(
                    f"Checkpoint task {checkpoint['task_id']} does not match {args.task}"
                )
            config = replace(
                PPOConfig(**checkpoint["ppo_config"]),
                rollout_steps=args.rollout_steps,
            )
            config.hidden_dims = tuple(config.hidden_dims)
            start_iteration = int(checkpoint["iteration"])
            env_steps = int(checkpoint["env_steps"])
        else:
            model = ActorCritic(
                actor_obs.shape[-1], critic_obs.shape[-1], action_dim, config
            ).to(device)
            # Seed normalization with reset states; subsequent updates happen only
            # between PPO iterations so likelihood ratios stay valid.
            model.actor_normalizer.update(actor_obs)
            model.critic_normalizer.update(critic_obs)
            start_iteration = 0
            env_steps = 0

            if args.initialize_from is not None:
                source_model, source_checkpoint = load_custom_policy(
                    args.initialize_from.expanduser().resolve(), device
                )
                if source_model.actor_obs_dim != model.actor_obs_dim:
                    raise ValueError("Initialized actor observation ABI does not match environment")
                if source_model.action_dim != model.action_dim:
                    raise ValueError("Initialized actor action ABI does not match environment")
                model.actor.load_state_dict(source_model.actor.state_dict())
                model.actor_normalizer.load_state_dict(source_model.actor_normalizer.state_dict())
                # Exploration is task-specific; keep the destination PPO
                # initialization instead of importing the previous log-std.
                with torch.no_grad():
                    model.log_std.fill_(math.log(0.10))
                print(
                    f"[INFO] initialized actor from {args.initialize_from}; "
                    "critic and optimizer are fresh"
                )

        if model.actor_obs_dim != actor_obs.shape[-1] or model.action_dim != action_dim:
            raise ValueError("Checkpoint observation/action ABI does not match environment")

        algorithm = PPO(model, config)
        if args.resume is not None:
            algorithm.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        storage = RolloutStorage(
            config.rollout_steps,
            args.num_envs,
            actor_obs.shape[-1],
            critic_obs.shape[-1],
            action_dim,
            device,
        )
        episode_returns = torch.zeros(args.num_envs, device=device)
        episode_lengths = torch.zeros(args.num_envs, dtype=torch.long, device=device)
        completed_returns: deque[float] = deque(maxlen=100)
        completed_lengths: deque[int] = deque(maxlen=100)
        environment_metrics: dict[str, list[float]] = defaultdict(list)
        start_time = time.perf_counter()
        collected_steps = 0

        print(
            f"[INFO] task={args.task} device={device} envs={args.num_envs} "
            f"actor_obs={actor_obs.shape[-1]} critic_obs={critic_obs.shape[-1]} "
            f"actions={action_dim}"
        )

        for iteration in range(start_iteration + 1, args.iterations + 1):
            for step in range(config.rollout_steps):
                actions, log_probs, values = model.act(actor_obs, critic_obs)
                next_observations, rewards, terminated, truncated, _ = env.step(actions)
                next_actor_obs = _obs_tensor(next_observations, "actor")
                next_critic_obs = _obs_tensor(next_observations, "critic")

                with torch.no_grad():
                    terminal_next_values = model.value(next_critic_obs)

                storage.actor_obs[step].copy_(actor_obs)
                storage.critic_obs[step].copy_(critic_obs)
                storage.actions[step].copy_(actions)
                storage.log_probs[step].copy_(log_probs)
                storage.values[step].copy_(values)
                storage.next_values[step].copy_(terminal_next_values)
                storage.rewards[step].copy_(rewards)
                storage.terminated[step].copy_(terminated)
                storage.truncated[step].copy_(truncated)

                episode_returns += rewards
                episode_lengths += 1
                done = terminated | truncated
                done_ids = done.nonzero(as_tuple=False).flatten()
                if len(done_ids) > 0:
                    completed_returns.extend(episode_returns[done_ids].cpu().tolist())
                    completed_lengths.extend(episode_lengths[done_ids].cpu().tolist())
                    episode_returns[done_ids] = 0.0
                    episode_lengths[done_ids] = 0

                    reset_observations, reset_info = env.reset(env_ids=done_ids)
                    for name, value in reset_info.get("log", {}).items():
                        if isinstance(value, torch.Tensor):
                            value = value.float().mean().item()
                        if isinstance(value, (int, float)):
                            environment_metrics[name].append(float(value))
                    reset_actor = _obs_tensor(reset_observations, "actor")
                    reset_critic = _obs_tensor(reset_observations, "critic")
                    # Preserve the exact step observations of environments that did
                    # not reset; partial reset recomputes all observation terms.
                    actor_obs = next_actor_obs.clone()
                    critic_obs = next_critic_obs.clone()
                    actor_obs[done_ids] = reset_actor[done_ids]
                    critic_obs[done_ids] = reset_critic[done_ids]
                else:
                    actor_obs = next_actor_obs
                    critic_obs = next_critic_obs

            storage.compute_returns(config.gamma, config.gae_lambda)
            metrics = algorithm.update(storage)
            transitions = config.rollout_steps * args.num_envs
            env_steps += transitions
            collected_steps += transitions
            elapsed = max(time.perf_counter() - start_time, 1.0e-6)
            sps = collected_steps / elapsed

            for name, value in metrics.items():
                writer.add_scalar(f"train/{name}", value, env_steps)
            writer.add_scalar("train/steps_per_second", sps, env_steps)
            if completed_returns:
                writer.add_scalar(
                    "episode/mean_return",
                    sum(completed_returns) / len(completed_returns),
                    env_steps,
                )
                writer.add_scalar(
                    "episode/mean_length",
                    sum(completed_lengths) / len(completed_lengths),
                    env_steps,
                )
            for name, values in environment_metrics.items():
                if values:
                    writer.add_scalar(
                        f"environment/{name}", sum(values) / len(values), env_steps
                    )
            environment_metrics.clear()

            if iteration == 1 or iteration % 10 == 0:
                mean_return = (
                    sum(completed_returns) / len(completed_returns)
                    if completed_returns
                    else float("nan")
                )
                print(
                    f"iter={iteration:05d} steps={env_steps:,} return={mean_return:8.2f} "
                    f"policy={metrics['policy_loss']:+.4f} value={metrics['value_loss']:.4f} "
                    f"kl={metrics['kl']:.5f} lr={metrics['learning_rate']:.2e} sps={sps:,.0f}"
                )

            if iteration % args.save_interval == 0 or iteration == args.iterations:
                checkpoint_path = checkpoint_dir / f"model_{iteration}.pt"
                algorithm.save(
                    checkpoint_path,
                    iteration=iteration,
                    task_id=args.task,
                    env_steps=env_steps,
                )
                if not args.no_onnx:
                    try:
                        export_custom_policy(
                            model,
                            checkpoint_dir / f"model_{iteration}.onnx",
                            task_id=args.task,
                            checkpoint_path=checkpoint_path,
                        )
                    except Exception as exc:
                        print(f"[WARN] ONNX export failed; checkpoint is valid: {exc}")
                print(f"[INFO] saved {checkpoint_path}")
            writer.flush()

        return checkpoint_dir / f"model_{args.iterations}.pt"
    finally:
        writer.close()
        env.close()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
