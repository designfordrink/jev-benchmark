# JEV Benchmark Runner v0.1

The runner is implemented as incremental end-to-end vertical slices.

## Implemented

- `Decision` contract with action, confidence, abstention and metadata.
- Task/policy separation.
- J01 CartPole with deterministic rule baseline.
- J01 tiny fixed-weight MLP inference-size probe.
- J01 latency, reward, confidence, abstention and fallback metrics.
- Deterministic seed handling and hidden-size sweep.
- J02 HARTH data contract: streaming CSV reader, schema validation, pure 128×6 windows.
- J02 subject-disjoint split with deterministic bounded window sampling.
- J02 nearest-centroid classical baseline.
- J02 trainable NumPy Tiny MLP.
- J02 accuracy, macro-F1, confidence, abstention, inference latency, parameter bytes and serialized model size.
- CLI commands for J02 manifest, inspection, training and hidden-size sweep.
- Unit tests for model invariants and subject leakage.
- GitHub Actions CI configuration.
- J02 multi-subject LOSO and hidden-size sweep orchestration.
- J02 aggregate risk-coverage and Pareto-front analysis.
- J03 synthetic candidate-selection benchmark with planner/scorer separation, reward/regret, confidence and permutation invariance.
- J03 linear baseline, trainable tiny MLP and capacity sweep.

## Scientific safeguards

The benchmark distinguishes model weights from serialized artifact size. Confidence is the model's score-derived maximum probability, not yet a calibrated confidence estimate. Raw HARTH data is external to the repository; synthetic fixtures are test-only and must never be reported as benchmark results.

For J02, the held-out subject is never included in the training set. Training normalization statistics are estimated from training windows only inside the Tiny MLP. Test labels not represented in the training subjects cause the run to fail rather than silently producing incomplete metrics.

## CLI

```bash
jev-bench list-tasks
jev-bench list-policies
jev-bench harth-manifest --dataset-root /path/to/harth
jev-bench harth-inspect --dataset-root /path/to/harth --subject S015
jev-bench harth-train --dataset-root /path/to/harth --test-subject S015 --model tiny_mlp --hidden-units 8
jev-bench harth-train --dataset-root /path/to/harth --test-subject S015 --model nearest_centroid
jev-bench harth-sweep --dataset-root /path/to/harth --test-subject S015 --hidden-units 1,2,4,8,16,32,64
```

## Next research layer

1. Add a common result schema and versioned JSON artifact.
2. Add confidence calibration and risk-coverage / abstention curves.
3. Add candidate permutation robustness where applicable.
4. Run J02 on pinned HARTH data across multiple held-out subjects rather than relying on one subject.
5. Add calibration-aware risk-coverage and selective prediction evaluation.
6. Add real-world J03 candidate environments and deterministic executors after the synthetic mechanism-validation task.