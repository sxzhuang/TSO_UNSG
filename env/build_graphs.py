"""Recreate the S-1 and medium graph configurations used by this repository.

Example:
    python env/build_graphs.py --scenario m1 --output env/m1_compact.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx

try:  # Supports both `python -m env.build_graphs` and direct script execution.
    from env.game import UrbanSecurityGame, make_s1, save_game_pickle
except ModuleNotFoundError:
    from game import UrbanSecurityGame, make_s1, save_game_pickle


# Exact undirected edge set of the shared 64-node graph used by M1--M4.
MEDIUM_EDGES = (
    (1, 2), (1, 9), (2, 3), (2, 10), (3, 4), (3, 11), (4, 5), (4, 12), (4, 13), (5, 6),
    (5, 13), (5, 14), (6, 7), (6, 13), (6, 14), (7, 8), (7, 15), (8, 16), (9, 10), (9, 17),
    (9, 18), (10, 11), (10, 18), (11, 12), (11, 19), (11, 20), (12, 13), (12, 20), (13, 14), (13, 21),
    (13, 22), (14, 15), (14, 21), (14, 22), (15, 16), (15, 23), (15, 24), (16, 24), (17, 18), (17, 25),
    (17, 26), (18, 19), (18, 26), (19, 20), (19, 27), (19, 28), (20, 21), (20, 27), (20, 28), (21, 22),
    (21, 29), (21, 30), (22, 23), (22, 30), (23, 24), (23, 30), (23, 31), (24, 31), (24, 32), (25, 26),
    (25, 33), (26, 27), (26, 34), (26, 35), (27, 28), (27, 35), (27, 36), (28, 29), (28, 36), (29, 30),
    (29, 36), (29, 37), (30, 31), (30, 38), (30, 39), (31, 32), (31, 38), (31, 39), (31, 40), (32, 40),
    (33, 34), (33, 41), (33, 42), (34, 35), (34, 42), (34, 43), (35, 36), (35, 43), (36, 37), (36, 43),
    (36, 44), (36, 45), (37, 38), (37, 44), (37, 45), (37, 46), (38, 39), (38, 46), (38, 47), (39, 40),
    (39, 47), (39, 48), (40, 48), (41, 42), (41, 49), (42, 43), (42, 49), (42, 50), (43, 44), (43, 50),
    (43, 51), (44, 45), (44, 51), (44, 52), (45, 46), (45, 52), (45, 53), (45, 54), (46, 47), (46, 53),
    (46, 54), (46, 55), (47, 48), (47, 55), (48, 56), (49, 50), (49, 57), (50, 51), (50, 58), (51, 52),
    (51, 59), (52, 53), (52, 59), (52, 60), (53, 54), (53, 60), (53, 61), (54, 55), (54, 61), (54, 62),
    (55, 56), (55, 63), (55, 64), (56, 64), (57, 58), (58, 59), (59, 60), (60, 61), (61, 62), (62, 63),
    (63, 64),
)


MEDIUM_SCENARIOS = {
    "m1": {"max_path_edges": 7, "num_defenders": 1},
    "m2": {"max_path_edges": 8, "num_defenders": 1},
    "m3": {"max_path_edges": 9, "num_defenders": 1},
    "m4": {"max_path_edges": 6, "num_defenders": 2},
}


def make_medium(scenario: str) -> UrbanSecurityGame:
    """Build one medium graph scenario without enumerating its action space."""
    if scenario not in MEDIUM_SCENARIOS:
        raise ValueError(f"Unknown medium scenario: {scenario}")
    graph = nx.Graph()
    graph.add_nodes_from(range(1, 65))
    graph.add_edges_from(MEDIUM_EDGES)
    adjacency = {node: sorted(graph.neighbors(node)) for node in graph.nodes}
    positions = list(graph.edges())
    config = MEDIUM_SCENARIOS[scenario]
    return UrbanSecurityGame(
        adjacency=adjacency,
        starts=[29],
        targets=[1, 8, 25, 40, 57, 64],
        defender_action_sets=[positions] * config["num_defenders"],
        max_path_edges=config["max_path_edges"],
        enumerate_actions=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a compact TSO game pickle.")
    parser.add_argument("--scenario", choices=["s1", *MEDIUM_SCENARIOS])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    game = make_s1() if args.scenario == "s1" else make_medium(args.scenario)
    save_game_pickle(game, args.output)
    print(f"Saved {args.scenario} graph configuration to {args.output}")


if __name__ == "__main__":
    main()
