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
