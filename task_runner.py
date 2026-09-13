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
            return checkpoint.get("infos", {})
        return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)

    def save(self, path: str, infos=None) -> None:
        super().save(path, infos)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["task_id"] = self.task_id
        torch.save(checkpoint, path)


def task_runner(task_id: str) -> type[TaskStampedOnPolicyRunner]:
    """Create the small registered runner type for one arbitrary task ID."""
    return type(
        f"TaskRunner_{task_id.replace('-', '_').replace('.', '_')}",
        (TaskStampedOnPolicyRunner,),
        {"task_id": task_id},
    )
