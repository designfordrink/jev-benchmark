# JEV Benchmark

A reproducible benchmark for evaluating tiny decision models across perception, decision, control, and hierarchical decision tasks.

## Status

**v0.1 — first vertical slices**

Implemented:
- **J01 CartPole** — control task with deterministic rule policy, tiny-model size probe, latency/confidence/fallback metrics.
- **J02 HARTH** — subject-aware streaming loader, trainable Tiny MLP, subject-disjoint LOSO, hidden-size sweep, risk-coverage and Pareto analysis.

J02 model training/evaluation code is implemented, but benchmark numbers still require the real HARTH CSV files. The repository never fabricates HARTH performance.

## Development

```bash
pip install -e '.[dev]'
pytest
jev-bench list-tasks
```

## J01

```bash
jev-bench run --task j01_cartpole --policy rule --episodes 20 --seed 0
jev-bench sweep --task j01_cartpole --hidden-units 1,2,4,8,16,32,64 --episodes 20 --seed 0
```

## J02 HARTH

The public HARTH dataset is external to this repository. After downloading a pinned release:

```bash
jev-bench harth-manifest --dataset-root /path/to/harth
jev-bench harth-inspect --dataset-root /path/to/harth --subject S015
```

See `docs/experiment-j02-harth.md`.

## Research rule

A benchmark result must come from an actual runner execution. Synthetic fixtures are used only for tests of parsing, contracts and invariants; they are never presented as HARTH performance.

## J02 multi-subject benchmark

Run LOSO on all subjects:

```bash
jev-bench harth-loso --dataset-root /path/to/harth --model tiny_mlp --hidden-units 8 --output results/j02-loso-h8.json
```

Run the size sweep across subjects:

```bash
jev-bench harth-loso-sweep --dataset-root /path/to/harth --hidden-units 1,2,4,8,16,32,64 --output results/j02-loso-sweep.json
```

The LOSO summary reports per-subject results, mean/std metrics and an aggregate risk-coverage curve. The size sweep also computes the Pareto front over macro-F1, inference latency and FP32 parameter bytes.

## J03 Candidate Selection

The first decision-track task separates candidate generation from candidate ranking. The planner supplies legal candidates; the tiny scorer ranks them, and the evaluator measures reward, regret, confidence, latency and candidate-order invariance.

```bash
jev-bench j03-train --model tiny_mlp --hidden-units 8
jev-bench j03-train --model linear
jev-bench j03-sweep --hidden-units 1,2,4,8,16,32,64 --output results/j03-sweep.json
```

See `docs/experiment-j03-candidate-selection.md`.
