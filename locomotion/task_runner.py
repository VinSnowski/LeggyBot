"""Task-neutral RSL-RL checkpoint support.

Task definitions provide the task identifier when registering a runner.  The
trainer itself never imports a robot-specific task module.
"""

from __future__ import annotations

import torch

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class TaskStampedOnPolicyRunner(VelocityOnPolicyRunner):
    """Store task metadata and support actor-only skill transfer."""

    task_id: str
    initialize_actor_only = False

    def _restore_environment_state(self, infos: dict | None) -> None:
        if not infos:
            return
        env_state = infos.get("env_state", {})
        if "common_step_counter" in env_state:
            self.env.unwrapped.common_step_counter = env_state[
                "common_step_counter"
            ]
        curriculum_state = env_state.get("curriculum_state", {})
        manager = getattr(self.env.unwrapped, "curriculum_manager", None)
        if not manager or not curriculum_state:
            return
        for name, state in curriculum_state.items():
            try:
                func = manager.get_term_cfg(name).func
            except (ValueError, AttributeError):
                continue
            if hasattr(func, "load_state_dict"):
                func.load_state_dict(state)

    def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None):
        if self.initialize_actor_only:
            checkpoint = torch.load(path, weights_only=False, map_location=map_location)
            source = checkpoint["actor_state_dict"]
            target = self.alg._raw_actor.state_dict()
            for key, value in source.items():
                if key in target and target[key].shape != value.shape:
                    raise ValueError(
                        "Initialized actor ABI mismatch for "
                        f"{key}: source {tuple(value.shape)}, "
                        f"destination {tuple(target[key].shape)}"
                    )
            # A task receives the learned actor and actor normalization, but a
            # fresh critic, optimizer, iteration counter and exploration std.
            actor_state = {
                key: value
                for key, value in source.items()
                if key in target and not key.endswith(("std_param", "log_std_param"))
            }
            self.alg._raw_actor.load_state_dict(actor_state, strict=False)
            infos = checkpoint.get("infos", {})
            if checkpoint.get("task_id") == self.task_id:
                self._restore_environment_state(infos)
            return infos
        infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        self._restore_environment_state(infos)
        return infos

    def save(self, path: str, infos=None) -> None:
        super().save(path, infos)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["task_id"] = self.task_id
        env_state = checkpoint.setdefault("infos", {}).setdefault("env_state", {})
        manager = getattr(self.env.unwrapped, "curriculum_manager", None)
        if manager:
            curriculum_state = {}
            for name in getattr(manager, "active_terms", []):
                try:
                    func = manager.get_term_cfg(name).func
                except (ValueError, AttributeError):
                    continue
                if hasattr(func, "state_dict"):
                    curriculum_state[name] = func.state_dict()
            if curriculum_state:
                env_state["curriculum_state"] = curriculum_state
        torch.save(checkpoint, path)


def task_runner(task_id: str) -> type[TaskStampedOnPolicyRunner]:
    """Create the small registered runner type for one arbitrary task ID."""
    return type(
        f"TaskRunner_{task_id.replace('-', '_').replace('.', '_')}",
        (TaskStampedOnPolicyRunner,),
        {"task_id": task_id},
    )
