"""Train the paper-aligned core TSO implementation on scenario S-1."""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from env import load_game_pickle
from network import DefenderTreeSampler, EvaderTreeSampler
from trainer import TSOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tree-Based Stochastic Optimization (TSO).")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", "--batch_num", dest="batch_size", type=int, default=100)
    parser.add_argument("--total-epochs", "--total_epoch", dest="total_epochs", type=int, default=50_000)
    parser.add_argument("--lr-evader", "--lr_evader", dest="lr_evader", type=float, default=1e-4)
    parser.add_argument("--lr-defender", "--lr_pursuer", dest="lr_defender", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.8)
    parser.add_argument("--tau-decay", "--weight_tau", dest="tau_decay", type=float, default=0.7)
    parser.add_argument(
        "--evader-lr-decay", "--weight_evader_lr", dest="evader_lr_decay", type=float, default=0.8
    )
    parser.add_argument(
        "--defender-lr-decay", "--weight_pursuer_lr", dest="defender_lr_decay", type=float, default=0.8
    )
    parser.add_argument(
        "--update-percentage", "--update_percentage", dest="update_percentage", type=float, default=0.01
    )
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--evaluation-interval", type=int, default=500)
    parser.add_argument("--env-path", type=Path, default=Path("env/s1.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device-id", "--device_id", dest="device_id", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true", help="Run on CPU even when CUDA is available.")
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    args = parse_args()
    if not 0.0 < args.update_percentage <= 1.0:
        raise ValueError("update_percentage must lie in (0, 1].")
    set_seeds(args.seed)
    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config["device"] = str(device)
        json.dump(config, file, indent=2, sort_keys=True)

    game = load_game_pickle(args.env_path, enumerate_actions=args.evaluation_interval > 0)
    evader = EvaderTreeSampler(game, device).to(device)
    defender = DefenderTreeSampler(game, device).to(device)
    trainer = TSOTrainer(
        evader,
        defender,
        game,
        epsilon=args.epsilon,
        tau=args.tau,
        evader_lr=args.lr_evader,
        defender_lr=args.lr_defender,
        tau_decay=args.tau_decay,
        evader_lr_decay=args.evader_lr_decay,
        defender_lr_decay=args.defender_lr_decay,
    )
    update_interval = max(int(args.total_epochs * args.update_percentage), 1)
    trainer.train(
        epochs=args.total_epochs,
        batch_size=args.batch_size,
        update_interval=update_interval,
        checkpoint_interval=args.checkpoint_interval,
        evaluation_interval=args.evaluation_interval,
        output_dir=output_dir,
    )
    print(f"Saved TSO run to: {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
