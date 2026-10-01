# Versioned Result Schema v1

JEV Benchmark command outputs use two shared JSON artifact schemas.

## Single result

Schema identifier:

```text
jev-benchmark.result/v1
```

Shape:

```json
{
  "schema_version": "jev-benchmark.result/v1",
  "task": "j04_tetris",
  "protocol": "sequential-train-evaluate",
  "model": {
    "model": "tiny_mlp",
    "hidden_units": 8,
    "parameter_count": 185,
    "model_size_bytes_fp32": 740
  },
  "evaluation": {
    "episodes": 20,
    "seed": 0
  },
  "metrics": {
    "mean_return": 123.0,
    "mean_action_latency_us": 42.0
  },
  "diagnostics": {
    "risk_coverage": []
  },
  "artifacts": {},
  "extra": {}
}
```

The common fields are intentionally small and stable:

- `task` and `protocol` identify what was run.
- `model` contains model/policy identity and resource-size fields.
- `evaluation` contains run configuration and split information needed to interpret metrics.
- `metrics` contains scalar numeric measurements that are comparable within a task/protocol.
- `diagnostics` contains structured curves, histories, class maps and other non-scalar analysis data.
- `artifacts` contains paths or identifiers for produced files.
- `extra` preserves task-specific non-numeric fields that do not belong to the common contract.

The adapter keeps legacy task dictionaries losslessly enough for the benchmark's existing fields: values are partitioned, not discarded. A future schema revision can change the partition only under a new schema identifier.

## Sweep result

Schema identifier:

```text
jev-benchmark.sweep/v1
```

A sweep has:

```json
{
  "schema_version": "jev-benchmark.sweep/v1",
  "task": "j03_candidate_selection",
  "protocol": "synthetic-candidate-size-sweep",
  "runs": ["...versioned result objects..."],
  "metadata": {
    "seed": 0
  },
  "pareto_front": ["...raw Pareto rows..."]
}
```

Every entry in `runs` is a `jev-benchmark.result/v1` object.

## Backward-compatible migration

Existing raw JSON can be converted without rerunning an experiment:

```bash
jev-bench normalize-result --input old-result.json --output result-v1.json
```

A protocol can be supplied when the legacy file did not contain one:

```bash
jev-bench normalize-result   --input old-result.json   --protocol sequential-train-evaluate   --output result-v1.json
```

## Scientific rule

The schema does not make unlike metrics automatically comparable. For example, J02 macro-F1, J03 selection accuracy and J04 game return remain task-specific metrics. The schema only gives them a common transport and metadata contract.

In particular, confidence-derived metrics must retain their evaluation context. J04 ECE is teacher-relative calibration diagnostic; it is not represented as a universal calibrated-probability claim.
