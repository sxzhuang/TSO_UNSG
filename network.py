"""Neural tree policies for the attacker and the ordered defender team."""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from env.game import DefenderAction, Edge, Path, UrbanSecurityGame, _edge


def _init(layer: nn.Linear, std: float = np.sqrt(2.0), bias: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias)
    return layer


class MaskedCategoricalPolicy(nn.Module):
    """Small MLP whose output is restricted by a dynamic action mask."""

    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            _init(nn.Linear(observation_dim, 64)),
            nn.Tanh(),
            _init(nn.Linear(64, 64)),
            nn.Tanh(),
            _init(nn.Linear(64, action_dim), std=0.01),
        )

    def action_and_probability(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor,
        forced_action: int | None = None,
    ) -> tuple[int, torch.Tensor]:
        if action_mask.ndim != 1:
            raise ValueError("action_mask must be one-dimensional")
        if not bool(torch.any(action_mask > 0)):
            raise ValueError("Cannot sample from an empty action mask.")

        logits = self.actor(observation)
        masked_logits = logits.masked_fill(action_mask.unsqueeze(0) == 0, float("-inf"))
        distribution = Categorical(logits=masked_logits)
        if forced_action is None:
            action = distribution.sample()
        else:
            action = torch.tensor([forced_action], dtype=torch.long, device=observation.device)
        probability = torch.exp(distribution.log_prob(action)).squeeze(0)
        return int(action.item()), probability


class _TreeSampler(MaskedCategoricalPolicy):
    def _draw_action(
        self,
        observation: torch.Tensor,
        mask: np.ndarray,
        sampling_epsilon: float | None,
    ) -> tuple[int, torch.Tensor, int]:
        mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=observation.device)
        valid_count = int(mask_tensor.sum().item())
        if valid_count == 0:
            raise RuntimeError("Tree construction reached a state with no valid child.")

        forced_action: int | None = None
        if sampling_epsilon is not None and float(torch.rand((), device=observation.device)) < sampling_epsilon:
            forced_action = int(torch.multinomial(mask_tensor, 1).item())
        action, policy_probability = self.action_and_probability(observation, mask_tensor, forced_action)
        return action, policy_probability, valid_count

    @staticmethod
    def _behavior_step_probability(
        policy_probability: torch.Tensor,
        valid_count: int,
        behavior_epsilon: float | None,
    ) -> torch.Tensor | None:
        if behavior_epsilon is None:
            return None
        return (1.0 - behavior_epsilon) * policy_probability.detach() + behavior_epsilon / valid_count


class EvaderTreeSampler(_TreeSampler):
    """Dynamically construct an attacker path one graph node at a time."""

    def __init__(self, game: UrbanSecurityGame, device: torch.device) -> None:
        super().__init__(game.max_path_nodes, game.degree)
        self.game = game
        self.device = device

    def _observation(self, history: Sequence[int]) -> torch.Tensor:
        observation = torch.zeros((1, self.game.max_path_nodes), dtype=torch.float32, device=self.device)
        observation[0, : len(history)] = torch.tensor(history, dtype=torch.float32, device=self.device)
        return observation

    def _sample_path(
        self,
        source: int,
        sampling_epsilon: float | None,
        behavior_epsilon: float | None,
    ) -> tuple[Path, torch.Tensor, torch.Tensor | None]:
        history: list[int] = [source]
        source_probability = 1.0 / len(self.game.starts)
        policy_probability = torch.tensor(source_probability, dtype=torch.float32, device=self.device)
        behavior_probability = (
            torch.tensor(source_probability, dtype=torch.float32, device=self.device)
            if behavior_epsilon is not None
            else None
        )

        while history[-1] not in self.game.targets:
            if len(history) >= self.game.max_path_nodes:
                raise RuntimeError("Attacker path exceeded its maximum length before reaching a target.")
            mask = self.game.evader_action_mask(history)
            action, step_policy_probability, valid_count = self._draw_action(
                self._observation(history), mask, sampling_epsilon
            )
            policy_probability = policy_probability * step_policy_probability
            behavior_step = self._behavior_step_probability(
                step_policy_probability, valid_count, behavior_epsilon
            )
            if behavior_probability is not None and behavior_step is not None:
                behavior_probability = behavior_probability * behavior_step

            history.append(self.game.adjacency[history[-1]][action])

        return tuple(history), policy_probability, behavior_probability

    def sample_batch_paths(
        self,
        batch_size: int,
        *,
        sampling_epsilon: float | None,
        behavior_epsilon: float | None,
        excluded_paths: Sequence[Sequence[int]] | None = None,
        max_attempts: int = 10_000,
    ) -> tuple[list[Path], torch.Tensor, torch.Tensor | None]:
        """Sample paths and report their policy and raw behavior probabilities.

        If ``excluded_paths`` is provided, rejection sampling returns an action
        different from the corresponding excluded full path.  The returned
        behavior probability remains the *raw* pre-pruning probability ``q``;
        the trainer performs the Algorithm-1 normalization ``q / (1 - q(a))``.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if excluded_paths is not None and len(excluded_paths) != batch_size:
            raise ValueError("excluded_paths must have batch_size entries")

        paths: list[Path] = []
        policy_probabilities: list[torch.Tensor] = []
        behavior_probabilities: list[torch.Tensor] = []
        for index in range(batch_size):
            excluded = tuple(excluded_paths[index]) if excluded_paths is not None else None
            for _ in range(max_attempts):
                source = random.choice(self.game.starts)
                path, policy_probability, behavior_probability = self._sample_path(
                    source, sampling_epsilon, behavior_epsilon
                )
                if path != excluded:
                    paths.append(path)
                    policy_probabilities.append(policy_probability)
                    if behavior_probability is not None:
                        behavior_probabilities.append(behavior_probability)
                    break
            else:
                raise RuntimeError("Sample-and-Prune could not find an alternative attacker action.")

        behavior = torch.stack(behavior_probabilities) if behavior_epsilon is not None else None
        return paths, torch.stack(policy_probabilities), behavior

    def action_probability(self, action: Sequence[int]) -> float:
        """Probability of one complete attacker action under the policy."""
        path = tuple(int(node) for node in action)
        if not path or path[0] not in self.game.starts or path[-1] not in self.game.targets:
            return 0.0
        if len(path) > self.game.max_path_nodes:
            return 0.0

        with torch.no_grad():
            probability = torch.tensor(1.0 / len(self.game.starts), device=self.device)
            history = [path[0]]
            for next_node in path[1:]:
                mask = self.game.evader_action_mask(history)
                neighbors = self.game.adjacency[history[-1]]
                if next_node not in neighbors:
                    return 0.0
                action_index = neighbors.index(next_node)
                if mask[action_index] == 0:
                    return 0.0
                _, step_probability = self.action_and_probability(
                    self._observation(history),
                    torch.as_tensor(mask, dtype=torch.float32, device=self.device),
                    action_index,
                )
                probability = probability * step_probability
                history.append(next_node)
            return float(probability.item())


class DefenderTreeSampler(_TreeSampler):
    """Sample an ordered, no-shared-location defender joint action."""

    def __init__(self, game: UrbanSecurityGame, device: torch.device) -> None:
        observation_dim = 2 * max(game.num_defenders - 1, 0) + 1
        super().__init__(observation_dim, len(game.defender_locations))
        self.game = game
        self.device = device
        self.observation_dim = observation_dim

    def _observation(self, history: Sequence[Edge], defender_index: int) -> torch.Tensor:
        observation = torch.full((1, self.observation_dim), -1.0, dtype=torch.float32, device=self.device)
        for index, road in enumerate(history):
            observation[0, 2 * index : 2 * index + 2] = torch.tensor(
                road, dtype=torch.float32, device=self.device
            )
        observation[0, -1] = float(defender_index)
        return observation

    def _sample_action(
        self,
        sampling_epsilon: float | None,
        behavior_epsilon: float | None,
    ) -> tuple[DefenderAction, torch.Tensor, torch.Tensor | None]:
        history: list[Edge] = []
        policy_probability = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        behavior_probability = (
            torch.tensor(1.0, dtype=torch.float32, device=self.device)
            if behavior_epsilon is not None
            else None
        )

        for defender_index in range(self.game.num_defenders):
            mask = self.game.defender_action_mask(history, defender_index)
            action, step_policy_probability, valid_count = self._draw_action(
                self._observation(history, defender_index), mask, sampling_epsilon
            )
            policy_probability = policy_probability * step_policy_probability
            behavior_step = self._behavior_step_probability(
                step_policy_probability, valid_count, behavior_epsilon
            )
            if behavior_probability is not None and behavior_step is not None:
                behavior_probability = behavior_probability * behavior_step
            history.append(self.game.defender_locations[action])

        return tuple(history), policy_probability, behavior_probability

    def sample_batch_paths(
        self,
        batch_size: int,
        *,
        sampling_epsilon: float | None,
        behavior_epsilon: float | None,
        excluded_paths: Sequence[Sequence[Sequence[int]]] | None = None,
        max_attempts: int = 10_000,
    ) -> tuple[list[DefenderAction], torch.Tensor, torch.Tensor | None]:
        """Sample defender actions, pruning only an exactly equal ordered action."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if excluded_paths is not None and len(excluded_paths) != batch_size:
            raise ValueError("excluded_paths must have batch_size entries")

        paths: list[DefenderAction] = []
        policy_probabilities: list[torch.Tensor] = []
        behavior_probabilities: list[torch.Tensor] = []
        for index in range(batch_size):
            excluded = (
                tuple(_edge(road) for road in excluded_paths[index])
                if excluded_paths is not None
                else None
            )
            for _ in range(max_attempts):
                action, policy_probability, behavior_probability = self._sample_action(
                    sampling_epsilon, behavior_epsilon
                )
                if action != excluded:
                    paths.append(action)
                    policy_probabilities.append(policy_probability)
                    if behavior_probability is not None:
                        behavior_probabilities.append(behavior_probability)
                    break
            else:
                raise RuntimeError("Sample-and-Prune could not find an alternative defender action.")

        behavior = torch.stack(behavior_probabilities) if behavior_epsilon is not None else None
        return paths, torch.stack(policy_probabilities), behavior

    def action_probability(self, action: Sequence[Sequence[int]]) -> float:
        """Probability of one legal ordered defender action under the policy."""
        if len(action) != self.game.num_defenders:
            return 0.0
        normalized = tuple(_edge(road) for road in action)
        if len(set(normalized)) != self.game.num_defenders:
            return 0.0

        with torch.no_grad():
            probability = torch.tensor(1.0, device=self.device)
            history: list[Edge] = []
            for defender_index, road in enumerate(normalized):
                if road not in self.game._location_to_index:
                    return 0.0
                mask = self.game.defender_action_mask(history, defender_index)
                action_index = self.game._location_to_index[road]
                if mask[action_index] == 0:
                    return 0.0
                _, step_probability = self.action_and_probability(
                    self._observation(history, defender_index),
                    torch.as_tensor(mask, dtype=torch.float32, device=self.device),
                    action_index,
                )
                probability = probability * step_probability
                history.append(road)
            return float(probability.item())
