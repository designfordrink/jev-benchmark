from __future__ import annotations

from jev_bench.analysis import aggregate_multi_seed


def _run(seed: int, s1_f1: float, s2_f1: float, s1_acc: float, s2_acc: float):
    rows = [
        {
            "test_subject": "S001",
            "macro_f1": s1_f1,
            "accuracy": s1_acc,
            "mean_confidence": 0.8,
            "abstention_rate": 0.1,
            "mean_single_window_inference_latency_us": 10.0,
            "train_seconds": 1.0,
        },
        {
            "test_subject": "S002",
            "macro_f1": s2_f1,
            "accuracy": s2_acc,
            "mean_confidence": 0.7,
            "abstention_rate": 0.2,
            "mean_single_window_inference_latency_us": 11.0,
            "train_seconds": 1.2,
        },
    ]
    return {
        "task": "j02_harth",
        "protocol": "LOSO",
        "seed": seed,
        "rows": rows,
        "aggregate": {
            "subject_count": 2,
            "metrics": {
                "macro_f1": {"mean": (s1_f1 + s2_f1) / 2.0, "std": 0.0},
                "accuracy": {"mean": (s1_acc + s2_acc) / 2.0, "std": 0.0},
            },
        },
    }


def test_multi_seed_reports_seed_variance():
    result = aggregate_multi_seed([
        _run(0, 0.80, 0.60, 0.82, 0.62),
        _run(1, 0.90, 0.70, 0.92, 0.72),
        _run(2, 0.70, 0.50, 0.72, 0.52),
    ])

    assert result["seed_count"] == 3
    assert result["seeds"] == [0, 1, 2]
    assert result["seed_metrics"]["macro_f1"]["mean"] == 0.70
    assert result["seed_metrics"]["macro_f1"]["std"] > 0.0


def test_multi_seed_separates_subject_variance():
    result = aggregate_multi_seed([
        _run(0, 0.80, 0.60, 0.82, 0.62),
        _run(1, 0.90, 0.60, 0.92, 0.62),
        _run(2, 0.70, 0.60, 0.72, 0.62),
    ])

    assert result["subject_count"] == 2
    assert result["subjects"]["S001"]["macro_f1"]["std"] > 0.0
    assert result["subjects"]["S002"]["macro_f1"]["std"] == 0.0


def test_cli_exposes_multi_seed_loso():
    from jev_bench.cli import build_parser

    args = build_parser().parse_args([
        "harth-loso-multi-seed",
        "--dataset-root", "/tmp/harth",
        "--seeds", "0,2,4",
    ])
    assert args.command == "harth-loso-multi-seed"
    assert args.seeds == "0,2,4"
    assert args.model == "tiny_mlp"
