"""Tensor-native PPO used by the custom mjlab training script."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class PPOConfig:
    rollout_steps: int = 24
    learning_epochs: int = 5
    num_minibatches: int = 4
    learning_rate: float = 1.0e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    desired_kl: float = 0.01
    min_learning_rate: float = 1.0e-5
    max_learning_rate: float = 1.0e-2
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    obs_clip: float = 10.0


class RunningMeanStd(nn.Module):
    """Streaming observation statistics stored as part of the policy."""

    def __init__(self, shape: tuple[int, ...], clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(1.0e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        values = values.reshape(-1, *self.mean.shape).float()
        if values.shape[0] == 0:
            return
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(values.shape[0]), device=values.device)

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = (values - self.mean) * torch.rsqrt(self.var + 1.0e-8)
        return normalized.clamp(-self.clip, self.clip)


def _mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(last_dim, width), nn.ELU()))
        last_dim = width
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Asymmetric actor-critic with independently normalized observations."""

    def __init__(
        self,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        config: PPOConfig,
    ):
        super().__init__()
        self.actor_obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.action_dim = action_dim
        self.actor_normalizer = RunningMeanStd((actor_obs_dim,), config.obs_clip)
        self.critic_normalizer = RunningMeanStd((critic_obs_dim,), config.obs_clip)
        self.actor = _mlp(actor_obs_dim, action_dim, config.hidden_dims)
        self.critic = _mlp(critic_obs_dim, 1, config.hidden_dims)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def distribution(self, actor_obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(self.actor_normalizer(actor_obs))
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.critic_normalizer(critic_obs)).squeeze(-1)

    @torch.no_grad()
    def act(
        self, actor_obs: torch.Tensor, critic_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(actor_obs)
        actions = distribution.sample()
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        return actions, log_prob, self.value(critic_obs)

    def evaluate_actions(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(actor_obs)
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy, self.value(critic_obs)

    @torch.no_grad()
    def deterministic(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.actor_normalizer(actor_obs))


class RolloutStorage:
    def __init__(
        self,
        steps: int,
        num_envs: int,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        device: torch.device,
    ):
        self.steps = steps
        self.num_envs = num_envs
        self.actor_obs = torch.empty(steps, num_envs, actor_obs_dim, device=device)
        self.critic_obs = torch.empty(steps, num_envs, critic_obs_dim, device=device)
        self.actions = torch.empty(steps, num_envs, action_dim, device=device)
        self.log_probs = torch.empty(steps, num_envs, device=device)
        self.values = torch.empty(steps, num_envs, device=device)
        self.next_values = torch.empty(steps, num_envs, device=device)
        self.rewards = torch.empty(steps, num_envs, device=device)
        self.terminated = torch.empty(steps, num_envs, dtype=torch.bool, device=device)
        self.truncated = torch.empty(steps, num_envs, dtype=torch.bool, device=device)
        self.advantages = torch.empty(steps, num_envs, device=device)
        self.returns = torch.empty(steps, num_envs, device=device)

    def compute_returns(self, gamma: float, gae_lambda: float) -> None:
        self.advantages.copy_(
            compute_gae(
                self.rewards,
                self.values,
                self.next_values,
                self.terminated,
                self.truncated,
                gamma,
                gae_lambda,
            )
        )
        self.returns.copy_(self.advantages + self.values)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """GAE that bootstraps time limits but never crosses an episode reset."""
    advantages = torch.empty_like(rewards)
    gae = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        bootstrap = (~terminated[step]).float()
        continue_episode = (~(terminated[step] | truncated[step])).float()
        delta = rewards[step] + gamma * next_values[step] * bootstrap - values[step]
        gae = delta + gamma * gae_lambda * continue_episode * gae
        advantages[step] = gae
    return advantages


class PPO:
    def __init__(self, model: ActorCritic, config: PPOConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def update(self, storage: RolloutStorage) -> dict[str, float]:
        cfg = self.config
        total = storage.steps * storage.num_envs
        if total % cfg.num_minibatches != 0:
            raise ValueError("rollout size must be divisible by num_minibatches")
        minibatch_size = total // cfg.num_minibatches

        actor_obs = storage.actor_obs.flatten(0, 1)
        critic_obs = storage.critic_obs.flatten(0, 1)
        actions = storage.actions.flatten(0, 1)
        old_log_probs = storage.log_probs.flatten()
        old_values = storage.values.flatten()
        returns = storage.returns.flatten()
        advantages = storage.advantages.flatten()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        updates = 0
        kl_sum = 0.0
        for _ in range(cfg.learning_epochs):
            indices = torch.randperm(total, device=actor_obs.device)
            for start in range(0, total, minibatch_size):
                batch = indices[start : start + minibatch_size]
                new_log_probs, entropy, new_values = self.model.evaluate_actions(
                    actor_obs[batch], critic_obs[batch], actions[batch]
                )
                log_ratio = new_log_probs - old_log_probs[batch]
                ratio = log_ratio.exp()
                surrogate = ratio * advantages[batch]
                clipped_surrogate = ratio.clamp(
                    1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio
                ) * advantages[batch]
                policy_loss = -torch.minimum(surrogate, clipped_surrogate).mean()

                clipped_values = old_values[batch] + (new_values - old_values[batch]).clamp(
                    -cfg.clip_ratio, cfg.clip_ratio
                )
                value_loss = torch.maximum(
                    (new_values - returns[batch]).square(),
                    (clipped_values - returns[batch]).square(),
                ).mean()
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + cfg.value_loss_coef * value_loss
                    - cfg.entropy_coef * entropy_mean
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                approx_kl = ((ratio - 1.0) - log_ratio).mean().detach()
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy_mean.item()
                metrics["kl"] += approx_kl.item()
                kl_sum += approx_kl.item()
                updates += 1

        for key in metrics:
            metrics[key] /= max(updates, 1)
        mean_kl = kl_sum / max(updates, 1)
        current_lr = self.optimizer.param_groups[0]["lr"]
        if mean_kl > 2.0 * cfg.desired_kl:
            current_lr = max(cfg.min_learning_rate, current_lr / 1.5)
        elif 0.0 < mean_kl < 0.5 * cfg.desired_kl:
            current_lr = min(cfg.max_learning_rate, current_lr * 1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = current_lr
        metrics["learning_rate"] = current_lr

        # Update only after PPO ratios are finished so old and new log-probs use
        # identical normalization statistics throughout this update.
        self.model.actor_normalizer.update(actor_obs)
        self.model.critic_normalizer.update(critic_obs)
        return metrics

    def save(
        self,
        path: Path,
        *,
        iteration: int,
        task_id: str,
        env_steps: int,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "mjlab-custom-ppo-v1",
                "task_id": task_id,
                "iteration": iteration,
                "env_steps": env_steps,
                "ppo_config": asdict(self.config),
                "actor_obs_dim": self.model.actor_obs_dim,
                "critic_obs_dim": self.model.critic_obs_dim,
                "action_dim": self.model.action_dim,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )


def load_custom_policy(
    checkpoint_path: Path, device: torch.device
) -> tuple[ActorCritic, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("format") != "mjlab-custom-ppo-v1":
        raise ValueError(f"Not a custom PPO checkpoint: {checkpoint_path}")
    config = PPOConfig(**checkpoint["ppo_config"])
    # JSON/YAML round trips may turn tuples into lists.
    config.hidden_dims = tuple(config.hidden_dims)
    model = ActorCritic(
        checkpoint["actor_obs_dim"],
        checkpoint["critic_obs_dim"],
        checkpoint["action_dim"],
        config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
