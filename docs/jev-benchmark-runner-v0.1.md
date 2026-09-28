# JEV Benchmark Runner v0.1

This repository implements the benchmark runner as an incremental set of end-to-end vertical slices.

## Implemented in bootstrap

- `Decision` contract: action, confidence, metadata
- task/policy protocol separation
- J01 CartPole environment adapter
- deterministic rule baseline
- CLI: `list-tasks`, `list-policies`, `run`
- reward and action-latency metrics
- deterministic seed handling
- regression test

## Next

1. tiny neural policy adapter and model-size accounting
2. confidence, abstention, and fallback contract
3. J02 HARTH
4. J03 candidate selection
5. `train`, `evaluate`, and `sweep`
6. result artifact schema and Pareto analysis
