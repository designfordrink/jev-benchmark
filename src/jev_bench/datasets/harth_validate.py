from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from jev_bench.datasets.harth import FEATURE_COLUMNS, LABELS, REQUIRED_COLUMNS


def validate_subject_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    row_count = 0
    labels: set[int] = set()
    nonfinite_count = 0
    invalid_rows = 0
    timestamps: list[datetime] = []

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {path}")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing HARTH columns in {path}: {missing}")

        for row in reader:
            row_count += 1
            try:
                label = int(row["label"])
                values = np.asarray(
                    [float(row[column]) for column in FEATURE_COLUMNS],
                    dtype=np.float64,
                )
                timestamp = datetime.fromisoformat(row["timestamp"])
            except (TypeError, ValueError) as exc:
                invalid_rows += 1
                raise ValueError(f"Invalid HARTH row {row_count} in {path}") from exc

            labels.add(label)
            nonfinite_count += int(np.count_nonzero(~np.isfinite(values)))
            timestamps.append(timestamp)

    deltas = np.diff(np.asarray([ts.timestamp() for ts in timestamps], dtype=np.float64))
    positive_deltas = deltas[deltas > 0]
    median_interval = float(np.median(positive_deltas)) if len(positive_deltas) else None
    observed_hz = float(1.0 / median_interval) if median_interval and median_interval > 0 else None

    unknown_labels = sorted(labels - set(LABELS))
    return {
        "subject": path.stem,
        "path": str(path),
        "row_count": row_count,
        "labels": sorted(labels),
        "label_names": {str(label): LABELS.get(label, "unknown") for label in sorted(labels)},
        "unknown_labels": unknown_labels,
        "nonfinite_feature_values": nonfinite_count,
        "invalid_rows": invalid_rows,
        "timestamp_count": len(timestamps),
        "median_timestamp_interval_seconds": median_interval,
        "observed_sampling_hz": observed_hz,
        "expected_sampling_hz": 50.0,
        "sampling_hz_error": abs(observed_hz - 50.0) if observed_hz is not None else None,
    }


def validate_dataset(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    files = sorted(root.glob("S*.csv"))
    if not files:
        raise FileNotFoundError(f"No HARTH subject CSV files found in {root}")

    reports = [validate_subject_file(path) for path in files]
    return {
        "schema_version": "jev-benchmark.harth-validation/v1",
        "dataset_root": str(root),
        "subject_count": len(reports),
        "subjects": [report["subject"] for report in reports],
        "total_rows": sum(report["row_count"] for report in reports),
        "reports": reports,
    }
