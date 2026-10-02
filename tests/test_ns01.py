import json
import subprocess
import sys
from pathlib import Path


def _result(seed, hidden, mean_return, size, latency, legal=1.0, permutation=1.0, fallback=0.0):
    return {
        "metrics": {
            "mean_return": mean_return,
            "mean_lines": 1.0,
            "model_size_bytes_fp32": size,
            "parameter_count": 10,
            "p95_action_latency_us": latency,
            "legal_action_rate": legal,
            "permutation_invariance_rate": permutation,
            "fallback_rate": fallback,
        },
        "model": {"hidden_units": hidden},
        "evaluation": {"seed": seed},
    }


def _write_fixture(root: Path):
    root.mkdir()
    for seed in (0, 1):
        (root / f"reference-seed-{seed}.json").write_text(json.dumps({
            "evaluation": {"seed": seed},
            "metrics": {"mean_return": 100.0},
        }))
    runs = []
    for hidden in (1, 8):
        runs.append(_result(0, hidden, 95.0 if hidden == 1 else 92.0,
                            3500 if hidden == 1 else 9000,
                            900 if hidden == 1 else 2000,
                            fallback=0.0 if hidden == 1 else 0.2))
        runs.append(_result(1, hidden, 95.0 if hidden == 1 else 92.0,
                            3500 if hidden == 1 else 9000,
                            900 if hidden == 1 else 2000,
                            fallback=0.0 if hidden == 1 else 0.2))
    (root / "sweep-seed-0.json").write_text(json.dumps({"runs": runs[:2]}))
    (root / "sweep-seed-1.json").write_text(json.dumps({"runs": runs[2:]}))


def test_ns01_analysis_profiles_and_gates(tmp_path):
    root = tmp_path / "ns01"
    _write_fixture(root)
    output = tmp_path / "analysis.json"
    subprocess.run([
        sys.executable, "scripts/analyze_ns01.py",
        "--root", str(root),
        "--output", str(output),
    ], check=True)
    data = json.loads(output.read_text())
    assert data["protocol"] == "ns01-operating-point-analysis/v2"
    assert data["profiles"]["ultra-small"]["feasible_operating_points"][0]["hidden_units"] == 1
    assert data["profiles"]["small-edge"]["feasible_operating_points"] == []


def test_ns01_workflow_triggers_on_analysis_and_tests():
    text = Path(".github/workflows/ns01-operating-point.yml").read_text(encoding="utf-8")
    assert '"scripts/**"' in text
    assert '"tests/**"' in text
    assert "--profiles ultra-small,small-edge" in text
