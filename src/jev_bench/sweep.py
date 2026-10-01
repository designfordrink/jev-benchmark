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
