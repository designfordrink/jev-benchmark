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

## Next J02 layer

1. Build subject-disjoint train/test folds.
2. Add a deterministic classical baseline.
3. Add a trainable tiny MLP over the 128×6 window.
4. Sweep hidden width and report parameter bytes, serialized size, latency, macro-F1 and abstention/confidence metrics.
5. Add a leakage test that fails if a subject appears in both train and test.

No model-performance number should be recorded until the actual HARTH files are supplied and the benchmark runner executes them.
