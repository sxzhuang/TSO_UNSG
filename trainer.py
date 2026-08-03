"""Paper-aligned Tree-Based Stochastic Optimization (TSO) training loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.optim as optim

from env.game import DefenderAction, Path as EvaderPath, UrbanSecurityGame
from network import DefenderTreeSampler, EvaderTreeSampler


logger = logging.getLogger(__name__)


def conditional_pruned_probability(
    raw_alternative_probability: torch.Tensor,
    original_behavior_probability: torch.Tensor,
) -> torch.Tensor:
    """Return the true probability after Sample-and-Prune.

    Let ``q`` be the pre-pruning epsilon-greedy behavior distribution, ``a``
    its first draw, and ``a'`` a later accepted draw.  Rejection sampling gives
    ``P(a' | a' != a) = q(a') / (1 - q(a))``.  Algorithm 1 uses this
    normalized probability as ``p`` in its importance-sampling estimator.
    """
    remaining_mass = (1.0 - original_behavior_probability).clamp_min(1e-12)
    return (raw_alternative_probability / remaining_mass).detach()


@dataclass(frozen=True)
class BatchResult:
    evader_loss: torch.Tensor
    defender_loss: torch.Tensor
    evader_sample_probability: torch.Tensor
    defender_sample_probability: torch.Tensor


class TSOTrainer:
    """Optimize tree policies using the estimator in paper Algorithm 1."""

    def __init__(
        self,
        evader: EvaderTreeSampler,
        defender: DefenderTreeSampler,
        game: UrbanSecurityGame,
        *,
        epsilon: float,
        tau: float,
        evader_lr: float,
        defender_lr: float,
        tau_decay: float,
        evader_lr_decay: float,
        defender_lr_decay: float,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1].")
        self.evader = evader
        self.defender = defender
        self.game = game
        self.epsilon = float(epsilon)
        self.tau = float(tau)
        self.tau_decay = float(tau_decay)
        self.evader_lr_decay = float(evader_lr_decay)
        self.defender_lr_decay = float(defender_lr_decay)
        self.optimizer_evader = optim.Adam(self.evader.parameters(), lr=evader_lr)
        self.optimizer_defender = optim.Adam(self.defender.parameters(), lr=defender_lr)

    def _sample_and_prune_evader(
        self, batch_size: int
    ) -> tuple[list[EvaderPath], list[EvaderPath], torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            original_actions, _, original_q = self.evader.sample_batch_paths(
                batch_size,
                sampling_epsilon=None,
                behavior_epsilon=self.epsilon,
            )
        alternative_actions, policy_probability, raw_q = self.evader.sample_batch_paths(
            batch_size,
            sampling_epsilon=self.epsilon,
            behavior_epsilon=self.epsilon,
            excluded_paths=original_actions,
        )
        assert original_q is not None and raw_q is not None
        return (
            original_actions,
            alternative_actions,
            policy_probability,
            conditional_pruned_probability(raw_q, original_q),
        )

    def _sample_and_prune_defender(
        self, batch_size: int
    ) -> tuple[list[DefenderAction], list[DefenderAction], torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            original_actions, _, original_q = self.defender.sample_batch_paths(
                batch_size,
                sampling_epsilon=None,
                behavior_epsilon=self.epsilon,
            )
        alternative_actions, policy_probability, raw_q = self.defender.sample_batch_paths(
            batch_size,
            sampling_epsilon=self.epsilon,
            behavior_epsilon=self.epsilon,
            excluded_paths=original_actions,
        )
        assert original_q is not None and raw_q is not None
        return (
            original_actions,
            alternative_actions,
            policy_probability,
            conditional_pruned_probability(raw_q, original_q),
        )

    def train_batch(self, batch_size: int) -> BatchResult:
        """Create the unbiased, stop-gradient estimator for one TSO update."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        evader_original, evader_alternative, evader_pi, evader_p = self._sample_and_prune_evader(batch_size)
        defender_original, defender_alternative, defender_pi, defender_p = self._sample_and_prune_defender(batch_size)

        # Paper notation: r_i = -G_i + tau * log x_i(a'_i).
        evader_utility = torch.tensor(
            [self.game.reward(path, defender_original[index])[0] for index, path in enumerate(evader_alternative)],
            dtype=torch.float32,
            device=self.evader.device,
        )
        defender_utility = torch.tensor(
            [self.game.reward(evader_original[index], action)[1] for index, action in enumerate(defender_alternative)],
            dtype=torch.float32,
            device=self.defender.device,
        )

        evader_cost = (-evader_utility + self.tau * torch.log(evader_pi.clamp_min(1e-12))).detach()
        defender_cost = (-defender_utility + self.tau * torch.log(defender_pi.clamp_min(1e-12))).detach()
        evader_baseline = evader_cost.mean()
        defender_baseline = defender_cost.mean()

        # This is <sg[(r-v)/p e_a], x>.  Both the cost/baseline and p are
        # detached, so gradients flow only through x(a') = policy_probability.
        evader_coefficient = ((evader_cost - evader_baseline) / evader_p.clamp_min(1e-12)).detach()
        defender_coefficient = ((defender_cost - defender_baseline) / defender_p.clamp_min(1e-12)).detach()
        evader_loss = (evader_coefficient * evader_pi).mean()
        defender_loss = (defender_coefficient * defender_pi).mean()

        return BatchResult(evader_loss, defender_loss, evader_p, defender_p)

    def optimization_step(self, batch_size: int) -> BatchResult:
        result = self.train_batch(batch_size)
        self.optimizer_evader.zero_grad(set_to_none=True)
        result.evader_loss.backward()
        self.optimizer_evader.step()

        self.optimizer_defender.zero_grad(set_to_none=True)
        result.defender_loss.backward()
        self.optimizer_defender.step()
        return result

    def decay_hyperparameters(self) -> None:
        """Apply Algorithm-1 learning-rate/tau decay without resetting Adam."""
        self.tau = max(self.tau * self.tau_decay, 1e-6)
        for group in self.optimizer_evader.param_groups:
            group["lr"] = max(group["lr"] * self.evader_lr_decay, 1e-7)
        for group in self.optimizer_defender.param_groups:
            group["lr"] = max(group["lr"] * self.defender_lr_decay, 1e-7)

    def strategy_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """Enumerate the small game's current strategy profile."""
        if self.game.evader_actions is None or self.game.defender_actions is None:
            raise ValueError("Strategy enumeration is unavailable for this non-enumerated environment.")
        evader_strategy = np.asarray(
            [self.evader.action_probability(action) for action in self.game.evader_actions], dtype=np.float64
        )
        defender_strategy = np.asarray(
            [self.defender.action_probability(action) for action in self.game.defender_actions], dtype=np.float64
        )
        return evader_strategy, defender_strategy

    def duality_gap(self) -> tuple[float, float, float]:
        """Return zero-sum duality gap, expected attacker, and defender utility."""
        if self.game.payoff_matrix is None:
            raise ValueError("Duality-gap evaluation requires an enumerated payoff matrix.")
        evader_strategy, defender_strategy = self.strategy_profile()
        if not np.isclose(evader_strategy.sum(), 1.0, atol=1e-5):
            raise RuntimeError("Enumerated evader action probabilities do not form a distribution.")
        if not np.isclose(defender_strategy.sum(), 1.0, atol=1e-5):
            raise RuntimeError("Enumerated defender action probabilities do not form a distribution.")

        payoff = self.game.payoff_matrix
        value = float(evader_strategy @ payoff @ defender_strategy)
        attacker_best_response = float(np.max(payoff @ defender_strategy))
        defender_best_response = float(np.min(evader_strategy @ payoff))
        gap = attacker_best_response - defender_best_response
        return gap, value, -value

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "tau": self.tau,
                "evader_state_dict": self.evader.state_dict(),
                "defender_state_dict": self.defender.state_dict(),
                "optimizer_evader_state_dict": self.optimizer_evader.state_dict(),
                "optimizer_defender_state_dict": self.optimizer_defender.state_dict(),
            },
            path,
        )

    def train(
        self,
        *,
        epochs: int,
        batch_size: int,
        update_interval: int,
        checkpoint_interval: int,
        evaluation_interval: int,
        output_dir: Path,
    ) -> dict[str, list[float]]:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if update_interval < 1:
            raise ValueError("update_interval must be positive")
        output_dir.mkdir(parents=True, exist_ok=True)
        history: dict[str, list[float]] = {"evader_loss": [], "defender_loss": [], "duality_gap": []}

        for epoch in range(1, epochs + 1):
            result = self.optimization_step(batch_size)
            history["evader_loss"].append(float(result.evader_loss.item()))
            history["defender_loss"].append(float(result.defender_loss.item()))

            if epoch % update_interval == 0:
                self.decay_hyperparameters()

            if evaluation_interval > 0 and (epoch % evaluation_interval == 0 or epoch == 1 or epoch == epochs):
                gap, attacker_value, defender_value = self.duality_gap()
                history["duality_gap"].append(gap)
                logger.info(
                    "epoch=%d/%d evader_loss=%.6g defender_loss=%.6g gap=%.6g value=(%.6g, %.6g) tau=%.4g",
                    epoch,
                    epochs,
                    result.evader_loss.item(),
                    result.defender_loss.item(),
                    gap,
                    attacker_value,
                    defender_value,
                    self.tau,
                )

            if epoch % checkpoint_interval == 0 or epoch == epochs:
                self.save_checkpoint(output_dir / "checkpoints" / f"epoch_{epoch}.pth", epoch)

        np.savez(
            output_dir / "metrics.npz",
            evader_loss=np.asarray(history["evader_loss"]),
            defender_loss=np.asarray(history["defender_loss"]),
            duality_gap=np.asarray(history["duality_gap"]),
        )
        return history
