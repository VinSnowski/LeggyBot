"""Minimal inference-only loaders for locomotion PyTorch checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn

from locomotion.play_keyboard import detect_checkpoint_backend
from locomotion.ppo import load_custom_policy


class NumpyPolicy(Protocol):
    observation_dim: int
    action_dim: int
    task_id: str | None

    def __call__(self, observation: np.ndarray) -> np.ndarray: ...


def bounded_policy_action(policy: NumpyPolicy, observation: np.ndarray) -> np.ndarray:
    """Match the training wrapper before applying AND remembering an action.

    Clipping only the joint target leaves raw network outputs in the next
    observation, which is a different closed-loop policy from training.
    """
    action = policy(observation)
    if not np.isfinite(action).all():
        raise ValueError("Policy returned a non-finite action")
    return np.clip(action, -1.0, 1.0)


class _TorchPolicy:
    def __init__(
        self,
        model: nn.Module,
        observation_dim: int,
        action_dim: int,
        task_id: str | None = None,
    ):
        self.model = model.eval()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.task_id = task_id

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        if observation.shape != (self.observation_dim,):
            raise ValueError(
                f"Expected observation shape {(self.observation_dim,)}, "
                f"got {observation.shape}"
            )
        tensor = torch.from_numpy(observation).to(dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            action = self.model(tensor)
        return action.squeeze(0).numpy(force=True)


class _CustomActor(nn.Module):
    def __init__(self, actor_critic: nn.Module):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor_critic.deterministic(observation)


class _RslActor(nn.Module):
    """Deterministic RSL-RL MLP reconstructed from its checkpoint state."""

    def __init__(self, state: dict[str, torch.Tensor]):
        super().__init__()
        self.register_buffer("mean", state["obs_normalizer._mean"].clone())
        self.register_buffer("std", state["obs_normalizer._std"].clone())

        layer_indices = sorted(
            int(key.split(".")[1])
            for key in state
            if key.startswith("mlp.") and key.endswith(".weight")
        )
        if not layer_indices:
            raise ValueError("RSL-RL checkpoint has no actor MLP layers")

        modules: list[nn.Module] = []
        for position, index in enumerate(layer_indices):
            weight = state[f"mlp.{index}.weight"]
            bias = state[f"mlp.{index}.bias"]
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(bias)
            modules.append(linear)
            if position != len(layer_indices) - 1:
                modules.append(nn.ELU())
        self.mlp = nn.Sequential(*modules)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # RSL-RL EmpiricalNormalization uses a fixed 1e-2 denominator epsilon.
        normalized = (observation - self.mean) / (self.std + 1.0e-2)
        return self.mlp(normalized)


def load_native_policy(checkpoint_path: Path) -> tuple[NumpyPolicy, str]:
    """Load either trainer's actor without constructing an RL environment."""
    backend = detect_checkpoint_backend(checkpoint_path)
    if backend == "custom":
        actor_critic, checkpoint = load_custom_policy(
            checkpoint_path, torch.device("cpu")
        )
        policy = _TorchPolicy(
            _CustomActor(actor_critic),
            actor_critic.actor_obs_dim,
            actor_critic.action_dim,
            checkpoint.get("task_id"),
        )
        return policy, backend

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor_state = checkpoint["actor_state_dict"]
    model = _RslActor(actor_state)
    first_weight = next(
        value
        for key, value in actor_state.items()
        if key.startswith("mlp.") and key.endswith(".weight")
    )
    last_index = max(
        int(key.split(".")[1])
        for key in actor_state
        if key.startswith("mlp.") and key.endswith(".weight")
    )
    last_weight = actor_state[f"mlp.{last_index}.weight"]
    return (
        _TorchPolicy(
            model,
            first_weight.shape[1],
            last_weight.shape[0],
            checkpoint.get("task_id"),
        ),
        backend,
    )
