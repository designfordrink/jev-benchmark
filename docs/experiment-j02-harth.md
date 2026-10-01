# Experiment J02 — HARTH data contract

## Purpose

J02 moves the benchmark from control (J01 CartPole) to perception: fixed-length dual-accelerometer windows are converted into a reproducible classification input.

The public HARTH release provides 50 Hz recordings from two 3-axis accelerometers, with six acceleration channels plus timestamp and activity label. The original dataset is organized as one CSV per subject. The benchmark therefore treats the subject as the split unit and forbids subject leakage between train and test.

## Current implementation

- streaming CSV reader; raw dataset is not committed to this repository;
- schema validation for timestamp, six acceleration columns and label;
- 128-sample windows × 6 channels by default;
- 128-sample stride by default;
- only pure-label windows are emitted, so a window crossing an activity boundary is discarded rather than assigned a synthetic majority label;
- subject identity is preserved in every emitted window;
- manifest and window-inspection CLI commands;
- tests use a tiny synthetic CSV fixture and do not fabricate HARTH benchmark results.

## Dataset source

Use the original HARTH release or the UCI v1.2 release. The UCI record documents 22 subjects, 50 Hz sampling, six acceleration channels and the activity-code mapping. The NTNU repository currently contains a newer v2.0 release with 31 subjects; for strict reproducibility, pin a dataset version when publishing results.

## Commands

Inspect the dataset without loading it into memory:

```bash
jev-bench harth-manifest --dataset-root /path/to/harth
```

Inspect up to 1,000 windows from one held-out subject:

```bash
jev-bench harth-inspect --dataset-root /path/to/harth --subject S015 --max-windows 1000
```

## Multi-subject LOSO layer

The benchmark now supports leave-one-subject-out evaluation over all discovered HARTH subject files, with optional subject selection for development runs. Each fold is trained only on the other subjects.

```bash
jev-bench harth-loso --dataset-root /path/to/harth --model tiny_mlp --hidden-units 8 --output results/j02-loso-h8.json
```

The LOSO result contains per-subject rows plus mean/std aggregates for accuracy, macro-F1, confidence, abstention, single-window latency and training time. It also aggregates the confidence risk-coverage curve across subjects.

A full hidden-width sweep can be run with:

```bash
jev-bench harth-loso-sweep --dataset-root /path/to/harth --hidden-units 1,2,4,8,16,32,64 --output results/j02-loso-sweep.json
```

The sweep computes a Pareto front using macro-F1 as the maximization objective and mean single-window inference latency plus mean FP32 parameter bytes as minimization objectives.

## Scientific safeguards

The benchmark reports both batch-amortized latency and single-window latency. The latter is the metric used for the JEV size/latency Pareto analysis. Confidence is score-derived maximum softmax probability and is not yet calibrated.

No HARTH performance number should be recorded until the actual dataset files are supplied and the benchmark runner executes them.
