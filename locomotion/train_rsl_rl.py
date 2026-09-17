"""Train the shared locomotion task with mjlab's official RSL-RL runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from locomotion import DEFAULT_TASK_ID
from locomotion.task import make_env_cfg, register_tasks
from locomotion.task_runner import TaskStampedOnPolicyRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK_ID)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index; use -1 for CPU")
    parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    parser.add_argument("--no-sim2real-randomization", action="store_true")
    parser.add_argument(
        "--initialize-from", type=Path,
        help="Initialize only the actor from a previous RSL-RL checkpoint.",
    )
    parser.add_argument(
        "--preserve-policy-std",
        action="store_true",
        help="With --initialize-from, also retain the learned action standard deviation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_envs <= 0 or args.iterations <= 0:
        raise ValueError("num-envs and iterations must be positive")
    if args.preserve_policy_std and args.initialize_from is None:
        raise ValueError("--preserve-policy-std requires --initialize-from")

    register_tasks()
    # Keep upstream logging/storage behavior, but group the terminal summary
    # into readable curriculum/reward/termination/metric sections.
    import rsl_rl.runners.on_policy_runner as rsl_runner
    from locomotion.grouped_logger import GroupedLogger

    rsl_runner.Logger = GroupedLogger
    from mjlab.rl import RslRlOnPolicyRunnerCfg
    from mjlab.scripts.train import TrainConfig, launch_training
    from mjlab.tasks.registry import load_rl_cfg

    agent_cfg = load_rl_cfg(args.task)
    if not isinstance(agent_cfg, RslRlOnPolicyRunnerCfg):
        raise TypeError(f"Task {args.task} is not configured for PPO")
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.iterations
    agent_cfg.run_name = args.run_name
    agent_cfg.logger = args.logger
    agent_cfg.upload_model = args.logger == "wandb"

    initialize_source: Path | None = None
    if args.initialize_from is not None:
        source = args.initialize_from.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        # mjlab's public TrainConfig only exposes regex-based resume paths.
        # Install a process-local resolver for this explicit path; this keeps
        # the new run under the destination task's experiment directory.
        initialize_source = source
        source_run = source.parent
        agent_cfg.resume = True
        agent_cfg.load_run = source_run.name
        agent_cfg.load_checkpoint = source.name
        TaskStampedOnPolicyRunner.initialize_actor_only = True
        TaskStampedOnPolicyRunner.preserve_initialized_policy_std = args.preserve_policy_std

    env_cfg = make_env_cfg(
        args.task,
        num_envs=args.num_envs,
        seed=args.seed,
        explicit_resets=False,
        sim2real_randomization=not args.no_sim2real_randomization,
    )
    env_cfg.sim.nan_guard.enabled = True

    config = TrainConfig(
        env=env_cfg,
        agent=agent_cfg,
        gpu_ids=None if args.gpu < 0 else [args.gpu],
        log_root=str(args.log_root),
        enable_nan_guard=True,
    )
    if initialize_source is not None:
        import mjlab.scripts.train as mjlab_train

        original_resolver = mjlab_train.get_checkpoint_path
        mjlab_train.get_checkpoint_path = lambda *_, **__: initialize_source
        try:
            launch_training(args.task, config)
        finally:
            mjlab_train.get_checkpoint_path = original_resolver
            TaskStampedOnPolicyRunner.initialize_actor_only = False
            TaskStampedOnPolicyRunner.preserve_initialized_policy_std = False
    else:
        launch_training(args.task, config)


if __name__ == "__main__":
    main()
