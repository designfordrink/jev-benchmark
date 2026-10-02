from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any


PROFILES = {
    "ultra-small": {
        "model_size_kb": 4.0,
        "p95_latency_ms": 1.0,
        "legal_action_rate_min": 1.0,
        "permutation_invariance_min": 1.0,
        "fallback_rate_max": None,
    },
    "small-edge": {
        "model_size_kb": 16.0,
        "p95_latency_ms": 5.0,
        "legal_action_rate_min": 1.0,
        "permutation_invariance_min": 1.0,
        "fallback_rate_max": 0.10,
    },
}


def _metric(run: dict[str, Any], key: str) -> float:
    try:
        return float(run["metrics"][key])
    except KeyError as exc:
        raise SystemExit(f"missing metric: {key}") from exc


def _load(root: str) -> tuple[list[dict], list[dict]]:
    refs, sweeps = [], []
    for path in glob.glob(str(Path(root) / "**/*.json"), recursive=True):
        name = Path(path).name
        if name == "operating-point-analysis.json":
            continue
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "sweep" in name:
            sweeps.append(data)
        elif "reference" in name:
            refs.append(data)
    if not refs or not sweeps:
        raise SystemExit("NS-01 results incomplete")
    return refs, sweeps


def _table(sweeps: list[dict], reference_mean: float) -> list[dict]:
    rows: dict[int, list[dict]] = {}
    for sweep in sweeps:
        for run in sweep.get("runs", []):
            hidden = int(run.get("model", {}).get("hidden_units", run.get("hidden_units", 0)))
            rows.setdefault(hidden, []).append(run)

    table = []
    for hidden in sorted(rows):
        rs = rows[hidden]

        def mean(key: str) -> float:
            return statistics.mean(_metric(r, key) for r in rs)

        def maximum(key: str) -> float:
            return max(_metric(r, key) for r in rs)

        def minimum(key: str) -> float:
            return min(_metric(r, key) for r in rs)

        utility = mean("mean_return")
        table.append({
            "hidden_units": hidden,
            "seed_count": len(rs),
            "mean_return": utility,
            "return_ratio": utility / reference_mean if reference_mean else 0.0,
            "mean_lines": mean("mean_lines"),
            "model_size_bytes_fp32_max": maximum("model_size_bytes_fp32"),
            "parameter_count_max": maximum("parameter_count"),
            "p95_action_latency_us_max": maximum("p95_action_latency_us"),
            "legal_action_rate_min": minimum("legal_action_rate"),
            "permutation_invariance_min": minimum("permutation_invariance_rate"),
            "fallback_rate_max": maximum("fallback_rate"),
            "fallback_rate_mean": mean("fallback_rate"),
        })
    return table


def _profile_result(table: list[dict], reference_mean: float, utility_ratio: float, profile: dict) -> dict:
    gate = reference_mean * utility_ratio
    feasible = []
    for row in table:
        checks = {
            "utility": row["mean_return"] >= gate,
            "model_size": row["model_size_bytes_fp32_max"] <= profile["model_size_kb"] * 1024,
            "p95_latency": row["p95_action_latency_us_max"] <= profile["p95_latency_ms"] * 1000,
            "legal_action_rate": row["legal_action_rate_min"] >= profile["legal_action_rate_min"],
            "permutation_invariance": row["permutation_invariance_min"] >= profile["permutation_invariance_min"],
        }
        if profile["fallback_rate_max"] is not None:
            checks["fallback_rate"] = row["fallback_rate_max"] <= profile["fallback_rate_max"]
        feasible.append({
            "hidden_units": row["hidden_units"],
            "feasible": all(checks.values()),
            "checks": checks,
        })
    passing = [x for x in feasible if x["feasible"]]
    return {
        "constraints": profile,
        "utility_gate": gate,
        "operating_points": table,
        "feasible_operating_points": passing,
        "interpretation": (
            "viable_tiny_operating_point"
            if passing else "no_operating_point_under_declared_constraints"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/ns01")
    parser.add_argument("--output", required=True)
    parser.add_argument("--utility-ratio", type=float, default=0.90)
    parser.add_argument("--profiles", default="ultra-small,small-edge")
    args = parser.parse_args()

    refs, sweeps = _load(args.root)
    reference_mean = statistics.mean(float(x["metrics"]["mean_return"]) for x in refs)
    table = _table(sweeps, reference_mean)

    names = [x.strip() for x in args.profiles.split(",") if x.strip()]
    unknown = [x for x in names if x not in PROFILES]
    if unknown:
        raise SystemExit(f"unknown NS-01 profile(s): {', '.join(unknown)}")

    profiles = {
        name: _profile_result(table, reference_mean, args.utility_ratio, PROFILES[name])
        for name in names
    }
    out = {
        "protocol": "ns01-operating-point-analysis/v2",
        "reference_mean_return": reference_mean,
        "utility_ratio": args.utility_ratio,
        "seed_count_reference": len(refs),
        "profiles": profiles,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
