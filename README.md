# JEV Benchmark

A reproducible benchmark for evaluating tiny decision models across perception, decision, control, and hierarchical decision tasks.

## Status

**v0.1 — first vertical slices**

Implemented:
- **J01 CartPole** — control task with deterministic rule policy, tiny-model size probe, latency/confidence/fallback metrics.
- **J02 HARTH** — subject-aware streaming loader, trainable Tiny MLP, subject-disjoint LOSO, hidden-size sweep, risk-coverage and Pareto analysis.
- **J03 Candidate Selection** — planner/selector separation on a synthetic decision task, with reward/regret, confidence and permutation invariance.
- **J04 Tetris Candidate Selection** — sequential deterministic environment with legal-placement planning, trainable tiny selectors, confidence-aware abstention/fallback, calibration diagnostics and risk-coverage evaluation.

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

## J04 Real Tetris Candidate Selection

J04 moves candidate selection from synthetic tables into a deterministic sequential Tetris environment. The environment generates legal placements; the tiny model ranks them; evaluation measures actual game return, lines, survival, regret, permutation invariance, latency and model size.

```bash
jev-bench j04-run --policy heuristic --episodes 20
jev-bench j04-run --policy random --episodes 20
jev-bench j04-train --model tiny_mlp --hidden-units 8
jev-bench j04-sweep --hidden-units 1,2,4,8,16,32,64 --output results/j04-sweep.json
jev-bench j04-risk-coverage --model tiny_mlp --hidden-units 8 --output results/j04-risk-coverage.json
```

J04 also supports `--abstain-threshold` on `j04-train`: low-confidence selector decisions fall back to the heuristic teacher. See `docs/experiment-j04-tetris.md` for the selective-decision protocol.

See `docs/experiment-j04-tetris.md`.
