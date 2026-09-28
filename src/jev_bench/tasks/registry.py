from __future__ import annotations
from jev_bench.tasks.j01_cartpole import J01CartPole


def get_task(name: str):
    if name == "j01_cartpole":
        return J01CartPole()
    raise ValueError(f"Unknown task: {name}")


def list_tasks() -> list[str]:
    return ["j01_cartpole"]
