from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from jev_bench.policies.tiny_mlp import TinyMLPPolicy
from jev_bench.tasks.j01_cartpole import J01CartPole


def run_j01_size_sweep(hidden_units: Iterable[int], *, episodes: int = 20, seed: int = 0, output: str | None = None):
    task = J01CartPole()
    rows = []
    for units in hidden_units:
        policy = TinyMLPPolicy(hidden_units=int(units))
        row = task.evaluate(policy, episodes=episodes, seed=seed)
        row["hidden_units"] = int(units)
        rows.append(row)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def run_j02_size_sweep(dataset_root, *, test_subject: str, hidden_units: Iterable[int] = (1,2,4,8,16,32,64), window_size: int = 128, stride: int = 128, max_train_windows_per_subject: int | None = 500, max_test_windows: int | None = 2000, epochs: int = 10, lr: float = 0.01, batch_size: int = 128, seed: int = 0, abstain_threshold: float = 0.0, output: str | None = None):
    from jev_bench.tasks.j02_harth import J02Harth
    task=J02Harth(); rows=[]
    for units in hidden_units:
        row=task.train_and_evaluate(dataset_root,test_subject=test_subject,model="tiny_mlp",hidden_units=int(units),window_size=window_size,stride=stride,max_train_windows_per_subject=max_train_windows_per_subject,max_test_windows=max_test_windows,epochs=epochs,lr=lr,batch_size=batch_size,seed=seed,abstain_threshold=abstain_threshold)
        row["hidden_units"]=int(units); rows.append(row)
    if output:
        path=Path(output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(rows,indent=2),encoding="utf-8")
    return rows


def run_j02_loso(dataset_root, *, model: str = "tiny_mlp", hidden_units: int = 8, subjects: Iterable[str] | None = None, window_size: int = 128, stride: int = 128, max_train_windows_per_subject: int | None = 500, max_test_windows: int | None = 2000, epochs: int = 10, lr: float = 0.01, batch_size: int = 128, seed: int = 0, abstain_threshold: float = 0.0, output: str | None = None):
    from jev_bench.datasets.harth import HarthDataset
    from jev_bench.analysis import aggregate_loso
    from jev_bench.tasks.j02_harth import J02Harth
    dataset=HarthDataset(dataset_root)
    available=[dataset.subject_id(p) for p in dataset.files()]
    requested=available if subjects is None else list(subjects)
    unknown=sorted(set(requested)-set(available))
    if unknown: raise ValueError(f"Unknown HARTH subjects: {unknown}")
    if not requested: raise ValueError("No subjects selected")
    task=J02Harth(); rows=[]
    for i,subject in enumerate(requested):
        rows.append(task.train_and_evaluate(dataset_root,test_subject=subject,model=model,hidden_units=hidden_units,window_size=window_size,stride=stride,max_train_windows_per_subject=max_train_windows_per_subject,max_test_windows=max_test_windows,epochs=epochs,lr=lr,batch_size=batch_size,seed=seed+i,abstain_threshold=abstain_threshold))
    result={"task":"j02_harth","protocol":"LOSO","model":model,"hidden_units":hidden_units,"rows":rows,"aggregate":aggregate_loso(rows)}
    if output:
        path=Path(output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


def run_j02_loso_size_sweep(dataset_root, *, hidden_units: Iterable[int] = (1,2,4,8,16,32,64), subjects: Iterable[str] | None = None, window_size: int = 128, stride: int = 128, max_train_windows_per_subject: int | None = 500, max_test_windows: int | None = 2000, epochs: int = 10, lr: float = 0.01, batch_size: int = 128, seed: int = 0, abstain_threshold: float = 0.0, output: str | None = None):
    rows=[]
    for units in hidden_units:
        result=run_j02_loso(dataset_root,model="tiny_mlp",hidden_units=int(units),subjects=subjects,window_size=window_size,stride=stride,max_train_windows_per_subject=max_train_windows_per_subject,max_test_windows=max_test_windows,epochs=epochs,lr=lr,batch_size=batch_size,seed=seed,abstain_threshold=abstain_threshold)
        aggregate=result["aggregate"]; row={"hidden_units":int(units),"model":"tiny_mlp","macro_f1":aggregate["metrics"]["macro_f1"]["mean"],"accuracy":aggregate["metrics"]["accuracy"]["mean"],"mean_confidence":aggregate["metrics"]["mean_confidence"]["mean"],"abstention_rate":aggregate["metrics"]["abstention_rate"]["mean"],"mean_inference_latency_us":aggregate["metrics"]["mean_inference_latency_us"]["mean"],"model_size_bytes_fp32":int(result["rows"][0]["model_size_bytes_fp32"]),"subject_count":aggregate["subject_count"]}
        rows.append(row)
    from jev_bench.analysis import pareto_front
    result={"task":"j02_harth","protocol":"LOSO-size-sweep","rows":rows,"pareto_front":pareto_front(rows)}
    if output:
        path=Path(output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
