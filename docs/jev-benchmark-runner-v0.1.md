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
- J04 deterministic Tetris environment with legal-placement generation, heuristic/random baselines and trainable candidate selector.
- J04 confidence-aware abstention with heuristic fallback, model coverage/fallback metrics and ECE calibration diagnostic.
- J04 risk-coverage runner for evaluating one trained selector across multiple confidence thresholds.
- Shared versioned result schema for J01–J04 single-run and sweep CLI artifacts.
- J02 multi-seed LOSO with separate seed-variance and subject-variance aggregation.
- Legacy result migration command via `normalize-result`.

## Scientific safeguards

The benchmark distinguishes model weights from serialized artifact size. J04 confidence is the model's score-derived maximum probability; ECE is a calibration diagnostic against local-teacher agreement, not proof of calibrated probability or optimality. Raw HARTH data is external to the repository; synthetic fixtures are test-only and must never be reported as HARTH performance.

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

1. Run J02 on a pinned HARTH release across multiple held-out subjects and at least 3 training seeds.
2. Compare Tiny MLP against nearest-centroid under the same subject/seed matrix.
3. Add calibrated confidence methods beyond raw softmax-max and evaluate selective prediction.
4. Escalate J04 from the self-contained Tetris simulator to an established benchmark while preserving the legal-candidate/executor contract.
5. Build the cross-task Pareto report from the shared v1 result artifacts.



## North Star alignment

The runner treats each benchmark task as a **decision component inside a larger system**, not merely as a model-evaluation problem.

Every new task should make explicit:

1. **External work** — what perception, planning, candidate generation, supervision or constraint checking is performed outside the tiny runtime component.
2. **Runtime boundary** — exactly what input the tiny component receives and what decision it must return.
3. **Downstream outcome** — what happens after the decision and which system-level metric determines usefulness.
4. **Resource constraint** — model size, memory, latency or energy proxy relevant to deployment.
5. **Failure handling** — confidence, abstention, fallback or escalation when the tiny component is uncertain.
6. **Robustness controls** — held-out data, representation changes, candidate-order changes, seeds or adversarial cases appropriate to the task.

A task is not complete merely because a tiny model achieves a good offline score. It must show how that score translates into downstream system utility and what operating point remains viable as the model is made smaller.

### Required task report

For new tasks, prefer a report of the form:

```
external work
    -> compact decision interface
    -> tiny runtime component
    -> downstream outcome

capacity / size / latency
    -> decision quality
    -> selective reliability
    -> system utility
```

The primary output is a **trade-off profile / operating-point frontier**, not a universal scalar score.
