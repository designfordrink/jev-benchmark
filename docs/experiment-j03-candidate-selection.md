# Experiment J03 — Candidate Selection

## Purpose

J03 is the first benchmark task where the tiny model is not asked to classify the world directly. A planner generates a finite set of legal candidates, and the tiny model ranks those candidates using the compact state plus candidate features.

The execution path is:

`state -> planner -> legal candidates -> tiny scorer -> selected candidate -> reward`

The benchmark question is whether a very small scorer can make useful decisions while remaining cheap and robust to candidate ordering.

## Synthetic problem

Each problem contains:

- a 6-dimensional compact context;
- 8 legal candidates;
- 6-dimensional features for each candidate;
- a deterministic latent utility function known to the evaluator but never supplied to the policy.

The policy sees only the context and candidate features. The evaluator computes the actual utility after selection.

## Metrics

The runner reports selection accuracy, mean reward, mean oracle reward, mean regret, p95 regret, confidence, abstention rate, permutation invariance, latency, parameter count and FP32 parameter bytes.

`selection_accuracy` is not the only objective. A non-oracle candidate can still have small regret, so reward and regret are reported separately.

## Candidate permutation test

For every test problem, candidates are randomly permuted several times. The selected candidate is compared by stable `candidate_id`, not by list position.

A permutation-invariant scorer should preserve the selected candidate under every permutation. This prevents a solution that accidentally depends on candidate index.

## Models

### Linear baseline

`j03_linear` fits a closed-form linear regression from context plus candidate features to latent utility.

### Tiny MLP

`j03_tiny_mlp` fits a one-hidden-layer NumPy MLP and ranks the legal candidates by predicted utility.

With 12 input features and one scalar output, the parameter count is `14h + 1` for hidden width `h`.

## Commands

Single experiment:

```bash
jev-bench j03-train --model tiny_mlp --hidden-units 8 --train-problems 500 --test-problems 300
```

Classical baseline:

```bash
jev-bench j03-train --model linear --train-problems 500 --test-problems 300
```

Capacity sweep:

```bash
jev-bench j03-sweep --hidden-units 1,2,4,8,16,32,64 --train-problems 500 --test-problems 300 --output results/j03-sweep.json
```

The sweep computes a Pareto front using selection accuracy as the maximized objective and latency/model bytes as minimized objectives.

## Research interpretation

J03 separates:

1. planning — candidate generation;
2. selection — tiny model ranking;
3. execution/reward — deterministic evaluator.

This separation is intended to carry forward to future non-synthetic tasks such as Tetris, job-shop scheduling, VRP, browser actions and game actions.

J03 is synthetic and is a mechanism-validation benchmark, not evidence of real-world performance.