# Experiment J04 — Real Tetris Candidate Selection

## Why J04 exists

J03 validated candidate ranking on synthetic feature tables. J04 removes the synthetic candidate set and puts selection inside a real sequential environment: the board state and current tetromino determine which placements are legal.

The decision loop is:

board state + current piece -> legal placements -> tiny scorer -> placement -> board update -> reward

The environment is deterministic under a seed and implements a 10×20 board, seven tetromino types, collision checks, gravity/drop placement and line clearing.

## Planner / selector separation

The environment is responsible for generating legal placements. The tiny model never invents coordinates. It receives only the compact board state and candidate features for legal placements.

A candidate is identified by stable (rotation, x, y) coordinates. The environment validates that the selected placement is currently legal before applying it.

## Candidate features

Each legal placement is described by normalized x, final drop y, lines cleared, holes after placement, maximum column height, aggregate height, bumpiness and rotation index.

The compact context contains the ten column heights, aggregate height, holes and current piece id.

## Training

The Tiny MLP is trained from states collected by random play. For each legal placement, the training target is a deterministic local Tetris heuristic based on line clears, holes, aggregate height, bumpiness and maximum height.

This is teacher-style supervision, not a hidden synthetic utility function. Final evaluation uses actual sequential game score, lines and survival length in the environment.

## Baselines

j04_heuristic applies the same local heuristic directly at decision time.

j04_random chooses uniformly from the legal placement set.

The trainable models are tiny_mlp and linear, reusing the J03 candidate scorer with J04's 21-dimensional context+candidate input.

## Metrics

The J04 runner reports mean/std game return, mean lines cleared, mean pieces survived, teacher agreement and local teacher regret, candidate-order invariance, legal-action rate, mean/p95 decision latency, parameter count and FP32 model bytes.

## Commands

Heuristic baseline:

```bash
jev-bench j04-run --policy heuristic --episodes 20 --max-pieces 300
```

Random baseline:

```bash
jev-bench j04-run --policy random --episodes 20 --max-pieces 300
```

Train and evaluate a Tiny MLP:

```bash
jev-bench j04-train --model tiny_mlp --hidden-units 8 --train-episodes 100 --test-episodes 20
```

Capacity sweep:

```bash
jev-bench j04-sweep --hidden-units 1,2,4,8,16,32,64 --output results/j04-sweep.json
```

The sweep emits a Pareto front with mean game return maximized and single-decision latency plus model size minimized.

## Interpretation

J04 is the first benchmark in the repository where a tiny selector acts repeatedly inside a changing environment and the final metric is downstream game performance rather than a synthetic ranking score.

It still uses a compact research environment rather than a third-party game implementation, so the next escalation should preserve the candidate-selection contract while moving to an established benchmark or richer simulator.