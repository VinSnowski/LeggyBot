"""Run a trained locomotion policy in real time with keyboard velocity commands.

W/S change forward velocity, A/D change lateral velocity, Q/E change yaw rate,
and X stops.  Enter resets the robot.  Commands remain latched until changed.
Custom PPO and RSL-RL checkpoint formats are detected automatically.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import torch

from locomotion import DEFAULT_TASK_ID
from locomotion.ppo import load_custom_policy
from locomotion.task import make_env_cfg, register_tasks


class KeyboardVelocityCommand:
    def __init__(
        self,
        *,
        linear_step: float = 0.1,
        yaw_step: float = 0.1,
        max_forward: float = 1.5,
        max_lateral: float = 1.0,
        max_yaw: float = 0.7,
        max_abs_curvature: float | None = None,
    ):
        self.linear_step = linear_step
        self.yaw_step = yaw_step
        self.limits = (max_forward, max_lateral, max_yaw)
        self.max_abs_curvature = max_abs_curvature
        self._target = [0.0, 0.0, 0.0]
        self._lock = Lock()

    @property
    def target(self) -> tuple[float, float, float]:
        with self._lock:
            return tuple(self._target)  # type: ignore[return-value]

    def key_callback(self, key: int) -> None:
        # This callback runs on MuJoCo's viewer thread.  It only mutates local
        # command state; the main simulation thread writes to mjlab tensors.
        from mjlab.viewer.native.keys import (
            KEY_A,
            KEY_D,
            KEY_E,
            KEY_Q,
            KEY_S,
            KEY_W,
            KEY_X,
        )

        with self._lock:
            if key == KEY_W:
                self._target[0] += self.linear_step
            elif key == KEY_S:
                self._target[0] -= self.linear_step
            elif key == KEY_A:
                self._target[1] += self.linear_step
            elif key == KEY_D:
                self._target[1] -= self.linear_step
            elif key == KEY_Q:
                self._target[2] += self.yaw_step
            elif key == KEY_E:
                self._target[2] -= self.yaw_step
            elif key == KEY_X:
                self._target[:] = [0.0, 0.0, 0.0]
            else:
                return
            self._target[:] = [
                max(-limit, min(limit, value))
                for value, limit in zip(self._target, self.limits, strict=True)
            ]
            if self.max_abs_curvature is not None:
                yaw_limit = self.max_abs_curvature * abs(self._target[0])
                self._target[2] = max(
                    -yaw_limit, min(yaw_limit, self._target[2])
                )
            print(
                f"command vx={self._target[0]:+.2f} "
                f"vy={self._target[1]:+.2f} yaw={self._target[2]:+.2f}"
            )


class ManualVelocityCommandMixin:
    """Mixin whose update executes on mjlab's main simulation thread."""

    controller: KeyboardVelocityCommand

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids
        target = torch.tensor(self.controller.target, device=self.device)
        self.vel_command_b[:] = target
        self.vel_command_w[:] = target
        self.is_heading_env[:] = False
        self.is_standing_env[:] = False
        self.is_world_env[:] = False
        self.is_forward_env[:] = False


class CustomPolicy:
    def __init__(self, model: Any):
        self.model = model

    def __call__(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model.deterministic(observations["actor"])


def detect_checkpoint_backend(checkpoint_path: Path) -> str:
    """Identify which trainer produced a PyTorch checkpoint.

    ``.pt`` describes PyTorch serialization, not a common policy schema.  Keep
    the format-specific loading at this boundary so the viewer and environment
    can work with the same observation-to-action policy interface afterwards.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a state dict")
    if checkpoint.get("format") == "mjlab-custom-ppo-v1":
        return "custom"
    rsl_keys = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict"}
    if rsl_keys.issubset(checkpoint):
        return "rsl-rl"
    keys = ", ".join(sorted(map(str, checkpoint.keys())))
    raise ValueError(
        f"Cannot recognize checkpoint format for {checkpoint_path}; keys: {keys}"
    )


def optimize_env_cfg_for_playback(env_cfg: Any) -> None:
    """Remove training-only work without changing the deployed policy ABI.

    A single-world interactive viewer is latency-bound, whereas MuJoCo-Warp is
    designed to amortize its kernels over many worlds.  Rewards and privileged
    critic features are useful during learning but are not inputs to the actor.
    Removing them also lets us drop the associated contact/height sensors.
    """
    actor_terms_before = tuple(env_cfg.observations["actor"].terms)

    # RSL-RL constructs a critic even for inference. Give that unused critic a
    # cheap observation group while loading only the actor checkpoint weights.
    env_cfg.observations["critic"] = deepcopy(env_cfg.observations["actor"])
    env_cfg.rewards = {}
    env_cfg.metrics = {}
    env_cfg.scene.sensors = ()

    # Reset events are required. Startup domain randomization and interval pushes
    # are undesirable for deterministic policy inspection and expand model state.
    env_cfg.events = {
        name: term for name, term in env_cfg.events.items() if term.mode == "reset"
    }
    for command_cfg in env_cfg.commands.values():
        command_cfg.debug_vis = False

    actor_terms_after = tuple(env_cfg.observations["actor"].terms)
    if actor_terms_after != actor_terms_before:
        raise RuntimeError("Playback optimization changed the actor observation ABI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--backend",
        choices=("auto", "custom", "rsl-rl"),
        default="auto",
        help="checkpoint format; auto detects it by default",
    )
    parser.add_argument("--task", default=DEFAULT_TASK_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--linear-step", type=float, default=0.1)
    parser.add_argument("--yaw-step", type=float, default=0.1)
    parser.add_argument("--frame-rate", type=float, default=30.0)
    parser.add_argument(
        "--full-env",
        action="store_true",
        help="retain training rewards, privileged observations and sensors for debugging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    backend = (
        detect_checkpoint_backend(args.checkpoint)
        if args.backend == "auto"
        else args.backend
    )
    print(f"Loading {backend} policy from {args.checkpoint}")
    device_name = (
        "cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    register_tasks()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
    from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
    from mjlab.viewer import NativeMujocoViewer, VerbosityLevel

    class ManualVelocityCommand(ManualVelocityCommandMixin, UniformVelocityCommand):
        def __init__(self, cfg: Any, env: Any):
            self.controller = cfg.controller
            super().__init__(cfg, env)

    @dataclass(kw_only=True)
    class ManualVelocityCommandCfg(UniformVelocityCommandCfg):
        controller: KeyboardVelocityCommand

        def build(self, env: Any) -> ManualVelocityCommand:
            return ManualVelocityCommand(self, env)

    env_cfg = make_env_cfg(
        args.task,
        num_envs=1,
        seed=123,
        play=True,
        explicit_resets=False,
        sim2real_randomization=False,
    )
    if not args.full_env:
        optimize_env_cfg_for_playback(env_cfg)
    twist_cfg = env_cfg.commands["twist"]
    command = KeyboardVelocityCommand(
        linear_step=args.linear_step,
        yaw_step=args.yaw_step,
        max_forward=max(abs(value) for value in twist_cfg.ranges.lin_vel_x),
        max_lateral=max(abs(value) for value in twist_cfg.ranges.lin_vel_y),
        max_yaw=max(abs(value) for value in twist_cfg.ranges.ang_vel_z),
        max_abs_curvature=getattr(twist_cfg, "max_abs_curvature", None),
    )
    env_cfg.commands["twist"] = ManualVelocityCommandCfg(
        controller=command,
        entity_name=twist_cfg.entity_name,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=twist_cfg.debug_vis,
        heading_command=False,
        heading_control_stiffness=twist_cfg.heading_control_stiffness,
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_world_envs=0.0,
        rel_forward_envs=0.0,
        init_velocity_prob=0.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=twist_cfg.ranges.lin_vel_x,
            lin_vel_y=twist_cfg.ranges.lin_vel_y,
            ang_vel_z=twist_cfg.ranges.ang_vel_z,
            heading=None,
        ),
        viz=deepcopy(twist_cfg.viz),
    )

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=str(device))
    if backend == "custom":
        model, checkpoint = load_custom_policy(args.checkpoint, device)
        if checkpoint["task_id"] != args.task:
            base_env.close()
            raise ValueError("Checkpoint task does not match --task")
        env: Any = base_env
        policy: Any = CustomPolicy(model)
    else:
        agent_cfg = load_rl_cfg(args.task)
        env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=str(device))
        runner.load(
            str(args.checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=str(device),
        )
        policy = runner.get_inference_policy(device=str(device))

    env.reset()
    print("Controls: W/S forward, A/D lateral, Q/E yaw, X stop, Enter reset")
    NativeMujocoViewer(
        env,
        policy,
        frame_rate=args.frame_rate,
        key_callback=command.key_callback,
        enable_perturbations=False,
        verbosity=VerbosityLevel.INFO,
    ).run()


if __name__ == "__main__":
    main()
