"""Deployment export for custom PPO checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from locomotion.ppo import ActorCritic


class DeterministicPolicy(nn.Module):
    def __init__(self, model: ActorCritic):
        super().__init__()
        self.normalizer = model.actor_normalizer
        self.actor = model.actor

    def forward(self, actor_observation: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(actor_observation))


def export_custom_policy(
    model: ActorCritic,
    output_path: Path,
    *,
    task_id: str,
    checkpoint_path: Path,
) -> None:
    """Export deterministic action means plus an explicit observation ABI."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = DeterministicPolicy(model).eval()
    sample = torch.zeros(1, model.actor_obs_dim, device=next(model.parameters()).device)
    torch.onnx.export(
        wrapper,
        (sample,),
        output_path,
        input_names=["actor_observation"],
        output_names=["joint_position_action"],
        dynamic_shapes={"actor_observation": {0: "batch"}},
        dynamo=True,
    )
    metadata = {
        "format": "mjlab-custom-ppo-onnx-v1",
        "task_id": task_id,
        "source_checkpoint": str(checkpoint_path),
        "actor_observation_dim": model.actor_obs_dim,
        "action_dim": model.action_dim,
        "action_semantics": "normalized residual joint-position targets",
        "normalization": "embedded in ONNX graph",
        "safety": "Policy output must pass through joint, velocity, torque and watchdog limits.",
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
