from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

RESULT_SCHEMA_VERSION = "jev-benchmark.result/v1"
SWEEP_SCHEMA_VERSION = "jev-benchmark.sweep/v1"

_IDENTITY_KEYS = {"task", "protocol"}
_MODEL_KEYS = {
    "model", "policy", "hidden_units", "parameter_count",
    "model_size_bytes_fp32", "serialized_model_size_bytes",
}
_EVALUATION_KEYS = {
    "episodes", "seed", "train_episodes", "test_episodes",
    "train_problems", "test_problems", "candidate_count",
    "context_dim", "candidate_dim", "train_candidates",
    "train_examples", "train_windows", "test_windows",
    "test_subject", "train_subjects", "window_size", "stride",
    "subject_count",
}
_DIAGNOSTIC_KEYS = {
    "history", "risk_coverage", "confidence_bins", "classes",
    "test_class_counts", "aggregate", "rows", "pareto_front",
}


@dataclass
class BenchmarkResult:
    task: str
    protocol: str
    model: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RESULT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_task_output(
        cls,
        payload: Mapping[str, Any],
        *,
        protocol: str | None = None,
    ) -> "BenchmarkResult":
        if "task" not in payload:
            raise ValueError("benchmark result must contain task")

        model: dict[str, Any] = {}
        evaluation: dict[str, Any] = {}
        metrics: dict[str, float] = {}
        diagnostics: dict[str, Any] = {}
        artifacts: dict[str, Any] = {}
        extra: dict[str, Any] = {}

        for key, value in payload.items():
            if key in _IDENTITY_KEYS:
                continue
            if key in _MODEL_KEYS:
                model[key] = value
            elif key in _EVALUATION_KEYS:
                evaluation[key] = value
            elif key in _DIAGNOSTIC_KEYS:
                diagnostics[key] = value
            elif key == "model_output":
                artifacts[key] = value
            elif isinstance(value, bool):
                metrics[key] = float(value)
            elif isinstance(value, (int, float)):
                metrics[key] = float(value)
            else:
                extra[key] = value

        return cls(
            task=str(payload["task"]),
            protocol=str(protocol or payload.get("protocol") or "single-run"),
            model=model,
            evaluation=evaluation,
            metrics=metrics,
            diagnostics=diagnostics,
            artifacts=artifacts,
            extra=extra,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkResult":
        if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported result schema: {payload.get('schema_version')!r}"
            )
        return cls(
            task=str(payload["task"]),
            protocol=str(payload["protocol"]),
            model=dict(payload.get("model", {})),
            evaluation=dict(payload.get("evaluation", {})),
            metrics=dict(payload.get("metrics", {})),
            diagnostics=dict(payload.get("diagnostics", {})),
            artifacts=dict(payload.get("artifacts", {})),
            extra=dict(payload.get("extra", {})),
            schema_version=str(payload["schema_version"]),
        )


@dataclass
class BenchmarkSweep:
    task: str
    protocol: str
    runs: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SWEEP_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def version_result(payload: Mapping[str, Any], *, protocol: str | None = None) -> dict[str, Any]:
    """Convert a task's legacy dictionary output into the shared v1 result schema."""
    return BenchmarkResult.from_task_output(payload, protocol=protocol).to_dict()


def version_sweep(
    *,
    task: str,
    protocol: str,
    rows: list[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
    pareto_front: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a versioned sweep artifact while preserving every run's metrics."""
    payload = BenchmarkSweep(
        task=task,
        protocol=protocol,
        runs=[version_result(row, protocol=protocol) for row in rows],
        metadata=dict(metadata or {}),
    ).to_dict()
    if pareto_front is not None:
        payload["pareto_front"] = [dict(row) for row in pareto_front]
    return payload
