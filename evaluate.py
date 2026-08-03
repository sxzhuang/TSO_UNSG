"""Evaluate a saved TSO checkpoint on the enumerated S-1 game."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from env import load_game_pickle
from network import DefenderTreeSampler, EvaderTreeSampler
from trainer import TSOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TSO S-1 checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--env-path", type=Path, default=Path("env/s1.pkl"))
    parser.add_argument("--device-id", "--device_id", dest="device_id", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    game = load_game_pickle(args.env_path, enumerate_actions=True)
    evader = EvaderTreeSampler(game, device).to(device)
    defender = DefenderTreeSampler(game, device).to(device)
    evader.load_state_dict(checkpoint["evader_state_dict"])
    defender.load_state_dict(checkpoint["defender_state_dict"])
    trainer = TSOTrainer(
        evader,
        defender,
        game,
        epsilon=0.8,
        tau=float(checkpoint["tau"]),
        evader_lr=1e-4,
        defender_lr=1e-4,
        tau_decay=0.7,
        evader_lr_decay=0.8,
        defender_lr_decay=0.8,
    )
    duality_gap, evader_value, defender_value = trainer.duality_gap()
    print(f"checkpoint_epoch: {checkpoint['epoch']}")
    print(f"duality_gap: {duality_gap:.10f}")
    print(f"expected_evader_payoff: {evader_value:.10f}")
    print(f"expected_defender_payoff: {defender_value:.10f}")


if __name__ == "__main__":
    main()
