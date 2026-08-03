"""Minimal urban network security game used by the TSO implementation.

The environment deliberately exposes only the operations required by TSO:
dynamic action masks, a simulator, and optional action enumeration for the
small S-1 evaluation game.
"""

from __future__ import annotations

from collections import deque
from itertools import product
from pathlib import Path as FilePath
import pickle
from typing import Sequence

import networkx as nx
import numpy as np


Edge = tuple[int, int]
Path = tuple[int, ...]
DefenderAction = tuple[Edge, ...]


def _edge(edge: Sequence[int]) -> Edge:
    """Return the canonical representation of an undirected road."""
    if len(edge) != 2:
        raise ValueError(f"A road must have two endpoints, got {edge!r}")
    u, v = (int(edge[0]), int(edge[1]))
    return (u, v) if u < v else (v, u)


class UrbanSecurityGame:
    """Undirected urban network security game with sequential defender choices.

    Defenders are ordered: a joint action is ``(e_1, ..., e_N)``.  A road may
    be assigned to at most one defender.  Consequently, a common action set
    with ``m`` roads and two defenders has ``m * (m - 1)`` legal actions.
    """

    def __init__(
        self,
        adjacency: dict[int, Sequence[int]],
        starts: Sequence[int],
        targets: Sequence[int],
        defender_action_sets: Sequence[Sequence[Sequence[int]]],
        max_path_edges: int,
        target_rewards: dict[int, float] | None = None,
        enumerate_actions: bool = False,
    ) -> None:
        self.adjacency = {
            int(node): tuple(sorted(int(neighbor) for neighbor in neighbors))
            for node, neighbors in adjacency.items()
        }
        self.starts = tuple(int(node) for node in starts)
        self.targets = frozenset(int(node) for node in targets)
        self.max_path_edges = int(max_path_edges)
        self.max_path_nodes = self.max_path_edges + 1
        self.target_rewards = {int(node): float(value) for node, value in (target_rewards or {}).items()}

        self.graph = nx.Graph()
        for node, neighbors in self.adjacency.items():
            for neighbor in neighbors:
                self.graph.add_edge(node, neighbor)

        if not self.starts or not self.targets:
            raise ValueError("At least one attacker start and target are required.")
        if self.max_path_edges < 1:
            raise ValueError("max_path_edges must be positive.")

        self.degree = max(len(neighbors) for neighbors in self.adjacency.values())
        self.defender_action_sets = tuple(
            tuple(_edge(road) for road in action_set) for action_set in defender_action_sets
        )
        if not self.defender_action_sets:
            raise ValueError("At least one defender is required.")

        locations: list[Edge] = []
        for action_set in self.defender_action_sets:
            for road in action_set:
                if road not in locations:
                    locations.append(road)
        self.defender_locations = tuple(locations)
        self._location_to_index = {road: index for index, road in enumerate(self.defender_locations)}
        self.num_defenders = len(self.defender_action_sets)

        self.evader_actions: tuple[Path, ...] | None = None
        self.defender_actions: tuple[DefenderAction, ...] | None = None
        self.payoff_matrix: np.ndarray | None = None
        if enumerate_actions:
            self.enumerate_actions()

    def _can_reach_a_target(self, source: int, blocked: set[int], max_remaining_edges: int) -> bool:
        """Whether a target can be reached within the remaining edge budget."""
        if source in self.targets:
            return True
        if max_remaining_edges <= 0:
            return False

        queue: deque[tuple[int, int]] = deque([(source, 0)])
        visited = {source}
        while queue:
            node, distance = queue.popleft()
            if distance >= max_remaining_edges:
                continue
            for neighbor in self.adjacency[node]:
                if neighbor in visited or neighbor in blocked:
                    continue
                if neighbor in self.targets:
                    return True
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
        return False

    def evader_action_mask(self, history: Sequence[int]) -> np.ndarray:
        """Mask valid next road choices for a simple attacker path."""
        if not history:
            raise ValueError("Attacker history must contain the current node.")
        current = int(history[-1])
        if current in self.targets:
            return np.zeros(self.degree, dtype=np.float32)

        mask = np.zeros(self.degree, dtype=np.float32)
        visited = {int(node) for node in history}
        remaining_after_next = self.max_path_edges - len(history)
        for index, next_node in enumerate(self.adjacency[current]):
            if next_node in visited:
                continue
            if self._can_reach_a_target(next_node, visited, remaining_after_next):
                mask[index] = 1.0
        return mask

    def defender_action_mask(self, history: Sequence[Sequence[int]], defender_index: int) -> np.ndarray:
        """Mask roads legal for one ordered defender decision.

        The per-defender action set implements the paper's :math:`E_i`; the
        history exclusion implements the S-1 no-shared-location constraint.
        """
        if not 0 <= defender_index < self.num_defenders:
            raise IndexError(f"Invalid defender index: {defender_index}")
        chosen = {_edge(road) for road in history}
        allowed = set(self.defender_action_sets[defender_index])
        mask = np.zeros(len(self.defender_locations), dtype=np.float32)
        for index, road in enumerate(self.defender_locations):
            if road in allowed and road not in chosen:
                mask[index] = 1.0
        return mask

    def reward(self, evader_path: Sequence[int], defender_action: Sequence[Sequence[int]]) -> tuple[float, float]:
        """Return attacker and defender utilities for one joint action."""
        if not evader_path or int(evader_path[-1]) not in self.targets:
            return (0.0, 0.0)

        target = int(evader_path[-1])
        attacker_reward = self.target_rewards.get(target, 1.0)
        defended_roads = {_edge(road) for road in defender_action}
        traversed_roads = {
            _edge((evader_path[index], evader_path[index + 1]))
            for index in range(len(evader_path) - 1)
        }
        if defended_roads & traversed_roads:
            return (-attacker_reward, attacker_reward)
        return (attacker_reward, -attacker_reward)

    def enumerate_actions(self) -> None:
        """Enumerate legal actions and the zero-sum payoff matrix for S-1."""
        evader_actions: list[Path] = []
        for source in self.starts:
            for target in sorted(self.targets):
                for path in nx.all_simple_paths(
                    self.graph,
                    source=source,
                    target=target,
                    cutoff=self.max_path_edges,
                ):
                    if any(node in self.targets for node in path[:-1]):
                        continue
                    evader_actions.append(tuple(int(node) for node in path))

        defender_actions: list[DefenderAction] = []
        for action in product(*self.defender_action_sets):
            normalized = tuple(_edge(road) for road in action)
            if len(set(normalized)) == self.num_defenders:
                defender_actions.append(normalized)

        self.evader_actions = tuple(evader_actions)
        self.defender_actions = tuple(defender_actions)
        self.payoff_matrix = np.array(
            [
                [self.reward(attacker, defender)[0] for defender in self.defender_actions]
                for attacker in self.evader_actions
            ],
            dtype=np.float64,
        )


def make_s1() -> UrbanSecurityGame:
    """Create paper scenario S-1 with 92 attacker and 110 defender actions.

    The defender team has two ordered members and 11 candidate roads.  The
    no-shared-location constraint gives ``11 * 10 = 110`` legal joint actions.
    """
    adjacency = {
        1: [2, 5, 6],
        2: [1, 3, 6, 7],
        3: [2, 4, 7],
        4: [3, 8],
        5: [1, 6, 9],
        6: [1, 2, 5, 7, 10],
        7: [2, 3, 6, 8, 11],
        8: [4, 7, 12],
        9: [5, 10, 13],
        10: [6, 9, 11, 14],
        11: [7, 10, 12, 15],
        12: [8, 11, 16],
        13: [9, 14],
        14: [10, 13, 15],
        15: [11, 14, 16],
        16: [12, 15],
    }
    defender_locations = [
        (1, 2),
        (1, 5),
        (1, 6),
        (10, 14),
        (9, 13),
        (12, 16),
        (7, 8),
        (7, 11),
        (11, 15),
        (14, 15),
        (15, 16),
    ]
    game = UrbanSecurityGame(
        adjacency=adjacency,
        starts=[1],
        targets=[15],
        defender_action_sets=[defender_locations, defender_locations],
        max_path_edges=8,
        enumerate_actions=True,
    )
    assert game.evader_actions is not None and len(game.evader_actions) == 92
    assert game.defender_actions is not None and len(game.defender_actions) == 110
    return game


def save_game_pickle(game: UrbanSecurityGame, path: str | FilePath) -> None:
    """Save only the graph and scenario configuration needed by TSO.

    The compact format intentionally does not store all pure actions or a payoff
    matrix.  They can be generated on demand for small environments.
    """
    payload = {
        "format": "tso-core-game-v1",
        "adjacency": game.adjacency,
        "starts": game.starts,
        "targets": tuple(sorted(game.targets)),
        "defender_action_sets": game.defender_action_sets,
        "max_path_edges": game.max_path_edges,
        "target_rewards": game.target_rewards,
    }
    output_path = FilePath(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(payload, file)


def load_game_pickle(path: str | FilePath, *, enumerate_actions: bool = False) -> UrbanSecurityGame:
    """Load a compact TSO game file or an existing legacy workspace pickle.

    Legacy one-defender medium files retain their precomputed action lists and
    payoff matrices when enumeration is requested.  This avoids rebuilding
    their matrices while keeping the runtime policy implementation independent
    of the legacy environment classes.
    """
    with FilePath(path).open("rb") as file:
        payload = pickle.load(file)

    is_compact_format = payload.get("format") == "tso-core-game-v1"
    if is_compact_format:
        game = UrbanSecurityGame(
            adjacency=payload["adjacency"],
            starts=payload["starts"],
            targets=payload["targets"],
            defender_action_sets=payload["defender_action_sets"],
            max_path_edges=payload["max_path_edges"],
            target_rewards=payload.get("target_rewards"),
            enumerate_actions=enumerate_actions,
        )
        return game

    required = {
        "adjlist",
        "evader_initial_pos",
        "evader_target",
        "evader_max_len",
        "pursuer_num",
        "pursuer_available_pos",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Unsupported game pickle; missing fields: {sorted(missing)}")

    positions = payload["pursuer_available_pos"]
    game = UrbanSecurityGame(
        adjacency=payload["adjlist"],
        starts=payload["evader_initial_pos"],
        targets=payload["evader_target"],
        defender_action_sets=[positions] * int(payload["pursuer_num"]),
        max_path_edges=int(payload["evader_max_len"]),
        target_rewards=payload.get("evader_target_rewards"),
        enumerate_actions=False,
    )

    # M1--M3 have one defender, so all stored actions are legal under the
    # no-shared-location rule.  Reuse their historical enumeration verbatim.
    if (
        enumerate_actions
        and game.num_defenders == 1
        and payload.get("evader_actions") is not None
        and payload.get("pursuer_actions") is not None
        and payload.get("payoff_matrix") is not None
    ):
        game.evader_actions = tuple(tuple(int(node) for node in action) for action in payload["evader_actions"])
        game.defender_actions = tuple(
            tuple(_edge(road) for road in action) for action in payload["pursuer_actions"]
        )
        game.payoff_matrix = np.asarray(payload["payoff_matrix"][0], dtype=np.float64)
    elif enumerate_actions:
        game.enumerate_actions()
    return game
