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
