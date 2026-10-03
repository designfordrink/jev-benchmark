import pytest

from jev_bench.datasets.harth_validate import validate_dataset, validate_subject_file


HEADER = "timestamp,back_x,back_y,back_z,thigh_x,thigh_y,thigh_z,label\n"


def _rows() -> str:
    return (
        HEADER
        + "2026-01-01T00:00:00,1,2,3,4,5,6,1\n"
        + "2026-01-01T00:00:00.020,1,2,3,4,5,6,1\n"
        + "2026-01-01T00:00:00.040,1,2,3,4,5,6,2\n"
    )


def test_validate_subject_reports_schema_rate_and_labels(tmp_path):
    path = tmp_path / "S001.csv"
    path.write_text(_rows(), encoding="utf-8")

    result = validate_subject_file(path)

    assert result["subject"] == "S001"
    assert result["row_count"] == 3
    assert result["labels"] == [1, 2]
    assert result["unknown_labels"] == []
    assert result["nonfinite_feature_values"] == 0
    assert result["observed_sampling_hz"] == pytest.approx(50.0, rel=1e-6)


def test_validate_dataset_aggregates_subjects(tmp_path):
    (tmp_path / "S001.csv").write_text(_rows(), encoding="utf-8")
    (tmp_path / "S002.csv").write_text(_rows(), encoding="utf-8")

    result = validate_dataset(tmp_path)

    assert result["schema_version"] == "jev-benchmark.harth-validation/v1"
    assert result["subject_count"] == 2
    assert result["subjects"] == ["S001", "S002"]
    assert result["total_rows"] == 6
