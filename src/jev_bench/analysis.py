from __future__ import annotations
from math import sqrt
from typing import Any, Iterable

METRIC_KEYS=("accuracy","macro_f1","mean_confidence","abstention_rate","mean_single_window_inference_latency_us","train_seconds")

def mean_std(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values=[float(r[key]) for r in rows]
    if not values: return {"mean":0.0,"std":0.0}
    m=sum(values)/len(values)
    if len(values)<2: return {"mean":m,"std":0.0}
    return {"mean":m,"std":sqrt(sum((x-m)**2 for x in values)/(len(values)-1))}

def aggregate_loso(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows: raise ValueError("LOSO produced no successful subject results")
    metrics={key:mean_std(rows,key) for key in METRIC_KEYS if key in rows[0]}
    risk=[]
    if all("risk_coverage" in r for r in rows):
        for i,point in enumerate(rows[0]["risk_coverage"]):
            cover=sum(float(r["risk_coverage"][i]["coverage"]) for r in rows)/len(rows)
            risk_value=sum(float(r["risk_coverage"][i]["risk"]) for r in rows)/len(rows)
            risk.append({"threshold":float(point["threshold"]),"coverage_mean":cover,"risk_mean":risk_value})
    return {"subjects_evaluated":[r["test_subject"] for r in rows],"subject_count":len(rows),"metrics":metrics,"risk_coverage":risk}

def pareto_front(rows: Iterable[dict[str, Any]], *, maximize: tuple[str,...]=( "macro_f1",), minimize: tuple[str,...]=( "mean_single_window_inference_latency_us","model_size_bytes_fp32_mean")) -> list[dict[str, Any]]:
    data=list(rows); front=[]
    for candidate in data:
        dominated=False
        for other in data:
            if other is candidate: continue
            no_worse=all(float(other[k])>=float(candidate[k]) for k in maximize) and all(float(other[k])<=float(candidate[k]) for k in minimize)
            strictly_better=any(float(other[k])>float(candidate[k]) for k in maximize) or any(float(other[k])<float(candidate[k]) for k in minimize)
            if no_worse and strictly_better:
                dominated=True; break
        if not dominated: front.append(candidate)
    return front


def _mean_std_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    m = sum(values) / len(values)
    if len(values) < 2:
        return {"mean": m, "std": 0.0}
    return {"mean": m, "std": sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))}


def aggregate_multi_seed(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate training-seed variance from held-out-subject variance."""
    if not seed_results:
        raise ValueError("multi-seed LOSO produced no results")

    seed_summaries = []
    for result in seed_results:
        aggregate = result.get("aggregate", {})
        metrics = aggregate.get("metrics", {})
        seed_summaries.append({
            "seed": int(result["seed"]),
            "subject_count": int(aggregate.get("subject_count", 0)),
            "metrics": {
                key: float(value["mean"])
                for key, value in metrics.items()
                if isinstance(value, dict) and "mean" in value
            },
        })

    metric_keys = sorted({
        key
        for row in seed_summaries
        for key in row["metrics"]
    })
    seed_variance = {
        key: _mean_std_values([row["metrics"][key] for row in seed_summaries if key in row["metrics"]])
        for key in metric_keys
    }

    per_subject: dict[str, dict[str, dict[str, float]]] = {}
    for result in seed_results:
        seed = int(result["seed"])
        for row in result.get("rows", []):
            subject = str(row["test_subject"])
            per_subject.setdefault(subject, {})
            for key in METRIC_KEYS:
                if key in row:
                    per_subject[subject].setdefault(key, {})[str(seed)] = float(row[key])

    subject_summary = {}
    for subject, metrics in sorted(per_subject.items()):
        subject_summary[subject] = {
            key: mean_std(list(seed_values.values()))
            for key, seed_values in metrics.items()
        }

    return {
        "seed_count": len(seed_results),
        "seeds": [int(r["seed"]) for r in seed_results],
        "seed_metrics": seed_variance,
        "subject_count": len(subject_summary),
        "subjects": subject_summary,
    }
