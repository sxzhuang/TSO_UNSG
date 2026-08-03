# Tree-Based Stochastic Optimization for Urban Network Security Games

This repository contains the core implementation of Tree-Based Stochastic
Optimization (TSO) for urban network security games, including tree-based
attacker and defender policies and Sample-and-Prune training.

The `env/` directory contains S-1, M1, M2, and M3 game files in pkl format.

## Setup

Python 3.10+ is recommended.

```bash
python -m pip install -r requirements.txt
```

## Training

Run training with the paper-facing default configuration:

```bash
python train.py --env-path env/s1.pkl --output-dir runs
```

For example, a short CPU run on S-1 is:

```bash
python train.py \
  --no-cuda \
  --total-epochs 2 \
  --batch-size 4 \
  --env-path env/s1.pkl \
  --checkpoint-interval 2 \
  --evaluation-interval 1
```

Outputs, including checkpoints and metrics, are saved to `runs/<timestamp>/`.

To train on a medium environment, replace the path, for example:

```bash
python train.py --env-path env/m1.pkl --output-dir runs
```

## Building an environment

The graph definitions for S-1 and M1--M4 are provided in
`env/build_graphs.py`.  Generate a compact pkl configuration with:

```bash
python env/build_graphs.py --scenario s1 --output env/s1_compact.pkl
python env/build_graphs.py --scenario m1 --output env/m1_compact.pkl
```

Available scenarios are `s1`, `m1`, `m2`, `m3`, and `m4`.

## Evaluation

```bash
python evaluate.py \
  --no-cuda \
  --env-path env/s1.pkl \
  --checkpoint runs/<timestamp>/checkpoints/epoch_50000.pth
```
