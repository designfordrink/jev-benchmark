from __future__ import annotations

import pytest

from jev_bench.core.results import (
    RESULT_SCHEMA_VERSION,
    SWEEP_SCHEMA_VERSION,
    BenchmarkResult,
    version_result,
    version_sweep,
)


def test_legacy_result_is_partitioned_into_shared_schema():
    raw = {
        "task": "j04_tetris",
        "model": "tiny_mlp",
        "hidden_units": 8,
        "episodes": 2,
        "seed": 7,
        "mean_return": 123.0,
        "mean_confidence": 0.81,
        "risk_coverage": [{"threshold": 0.9, "coverage": 0.2}],
        "history": {"train_mse": 0.5},
        "teacher": "local-heuristic",
    }
    result = version_result(raw, protocol="sequential-train-evaluate")
    assert result["schema_version"] == RESULT_SCHEMA_VERSION
    assert result["task"] == "j04_tetris"
    assert result["protocol"] == "sequential-train-evaluate"
    assert result["model"]["model"] == "tiny_mlp"
    assert result["model"]["hidden_units"] == 8
    assert result["evaluation"]["episodes"] == 2
    assert result["evaluation"]["seed"] == 7
    assert result["metrics"]["mean_return"] == 123.0
    assert result["metrics"]["mean_confidence"] == 0.81
    assert result["diagnostics"]["risk_coverage"][0]["coverage"] == 0.2
    assert result["diagnostics"]["history"]["train_mse"] == 0.5
    assert result["extra"]["teacher"] == "local-heuristic"


def test_result_round_trip_validates_version():
    raw = version_result(
        {"task": "j01_cartpole", "policy": "rule", "episodes": 1, "seed": 0, "mean_return": 10.0}
    )
    restored = BenchmarkResult.from_dict(raw)
    assert restored.to_dict() == raw


def test_unsupported_result_version_is_rejected():
    with pytest.raises(ValueError, match="unsupported result schema"):
        BenchmarkResult.from_dict({"schema_version": "jev-benchmark.result/v0"})


def test_sweep_contains_versioned_runs():
    sweep = version_sweep(
        task="j03_candidate_selection",
        protocol="synthetic-candidate-size-sweep",
        rows=[
            {"task": "j03_candidate_selection", "model": "linear", "seed": 0, "selection_accuracy": 0.5},
            {"task": "j03_candidate_selection", "model": "tiny_mlp", "seed": 0, "selection_accuracy": 0.7},
        ],
        metadata={"seed": 0},
        pareto_front=[{"model": "tiny_mlp"}],
    )
    assert sweep["schema_version"] == SWEEP_SCHEMA_VERSION
    assert len(sweep["runs"]) == 2
    assert all(r["schema_version"] == RESULT_SCHEMA_VERSION for r in sweep["runs"])
    assert sweep["runs"][1]["metrics"]["selection_accuracy"] == 0.7
    assert sweep["pareto_front"][0]["model"] == "tiny_mlp"
