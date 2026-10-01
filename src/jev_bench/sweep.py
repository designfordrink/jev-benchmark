from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from jev_bench.core.results import version_result, version_sweep
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
        path.write_text(json.dumps(version_sweep(task="j01_cartpole",protocol="size-sweep",rows=rows,metadata={"episodes":episodes,"seed":seed}), indent=2), encoding="utf-8")
    return rows


def run_j02_size_sweep(dataset_root, *, test_subject: str, hidden_units: Iterable[int] = (1,2,4,8,16,32,64), window_size: int = 128, stride: int = 128, max_train_windows_per_subject: int | None = 500, max_test_windows: int | None = 2000, epochs: int = 10, lr: float = 0.01, batch_size: int = 128, seed: int = 0, abstain_threshold: float = 0.0, output: str | None = None):
    from jev_bench.tasks.j02_harth import J02Harth
    task=J02Harth(); rows=[]
    for units in hidden_units:
        row=task.train_and_evaluate(dataset_root,test_subject=test_subject,model="tiny_mlp",hidden_units=int(units),window_size=window_size,stride=stride,max_train_windows_per_subject=max_train_windows_per_subject,max_test_windows=max_test_windows,epochs=epochs,lr=lr,batch_size=batch_size,seed=seed,abstain_threshold=abstain_threshold)
        row["hidden_units"]=int(units); rows.append(row)
    if output:
        path=Path(output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(version_sweep(task="j02_harth",protocol="subject-size-sweep",rows=rows,metadata={"test_subject":test_subject,"seed":seed}),indent=2),encoding="utf-8")
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
        path=Path(output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(version_result(result,protocol="LOSO"),indent=2),encoding="utf-8")
    return result


def run_j02_loso_multi_seed(
    dataset_root,
    *,
    seeds: Iterable[int] = (0, 1, 2),
    model: str = "tiny_mlp",
    hidden_units: int = 8,
    subjects: Iterable[str] | None = None,
    window_size: int = 128,
    stride: int = 128,
    max_train_windows_per_subject: int | None = 500,
    max_test_windows: int | None = 2000,
    epochs: int = 10,
    lr: float = 0.01,
    batch_size: int = 128,
    abstain_threshold: float = 0.0,
    output: str | None = None,
):
    from jev_bench.analysis import aggregate_multi_seed

    seed_list = [int(s) for s in seeds]
    if not seed_list:
        raise ValueError("At least one seed is required")

    runs = [
        run_j02_loso(
            dataset_root,
            model=model,
            hidden_units=hidden_units,
            subjects=subjects,
            window_size=window_size,
            stride=stride,
            max_train_windows_per_subject=max_train_windows_per_subject,
            max_test_windows=max_test_windows,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=seed,
            abstain_threshold=abstain_threshold,
            output=None,
        )
        for seed in seed_list
    ]
    result = {
        "task": "j02_harth",
        "protocol": "LOSO-multi-seed",
        "model": model,
        "hidden_units": hidden_units,
        "seeds": seed_list,
        "runs": runs,
        "aggregate": aggregate_multi_seed(runs),
    }
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(version_result(result, protocol="LOSO-multi-seed"), indent=2),
            encoding="utf-8",
        )
    return result


def run_j02_loso_size_sweep(dataset_root, *, hidden_units: Iterable[int] = (1,2,4,8,16,32,64), subjects: Iterable[str] | None = None, window_size: int = 128, stride: int = 128, max_train_windows_per_subject: int | None = 500, max_test_windows: int | None = 2000, epochs: int = 10, lr: float = 0.01, batch_size: int = 128, seed: int = 0, abstain_threshold: float = 0.0, output: str | None = None):
    rows=[]
    for units in hidden_units:
        result=run_j02_loso(dataset_root,model="tiny_mlp",hidden_units=int(units),subjects=subjects,window_size=window_size,stride=stride,max_train_windows_per_subject=max_train_windows_per_subject,max_test_windows=max_test_windows,epochs=epochs,lr=lr,batch_size=batch_size,seed=seed,abstain_threshold=abstain_threshold)
        aggregate=result["aggregate"]; row={"hidden_units":int(units),"model":"tiny_mlp","macro_f1":aggregate["metrics"]["macro_f1"]["mean"],"accuracy":aggregate["metrics"]["accuracy"]["mean"],"mean_confidence":aggregate["metrics"]["mean_confidence"]["mean"],"abstention_rate":aggregate["metrics"]["abstention_rate"]["mean"],"mean_single_window_inference_latency_us":aggregate["metrics"]["mean_single_window_inference_latency_us"]["mean"],"model_size_bytes_fp32_mean":sum(float(r["model_size_bytes_fp32"]) for r in result["rows"])/len(result["rows"]),"subject_count":aggregate["subject_count"]}
        rows.append(row)
    from jev_bench.analysis import pareto_front
    result={"task":"j02_harth","protocol":"LOSO-size-sweep","rows":rows,"pareto_front":pareto_front(rows)}
    if output:
        path=Path(output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(version_sweep(task="j02_harth",protocol="LOSO-size-sweep",rows=rows,metadata={"seed":seed},pareto_front=pareto_front(rows)),indent=2),encoding="utf-8")
    return result

def run_j03_size_sweep(*, hidden_units=(1,2,4,8,16,32,64), train_problems=500, test_problems=300, epochs=20, lr=0.01, batch_size=128, seed=0, abstain_threshold=0.0, permutation_trials=5, output=None):
    from jev_bench.tasks.j03_candidate_selection import J03CandidateSelection
    from jev_bench.analysis import pareto_front
    task=J03CandidateSelection()
    rows=task.size_sweep(hidden_units=hidden_units,train_problems=train_problems,test_problems=test_problems,epochs=epochs,lr=lr,batch_size=batch_size,seed=seed,abstain_threshold=abstain_threshold,permutation_trials=permutation_trials)
    summary={"task":"j03_candidate_selection","protocol":"synthetic-candidate-size-sweep","rows":rows,"pareto_front":pareto_front(rows,maximize=("selection_accuracy",),minimize=("mean_action_latency_us","model_size_bytes_fp32"))}
    if output:
        path=Path(output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(version_sweep(task="j03_candidate_selection",protocol="synthetic-candidate-size-sweep",rows=rows,metadata={"seed":seed},pareto_front=summary["pareto_front"]),indent=2),encoding="utf-8")
    return summary
