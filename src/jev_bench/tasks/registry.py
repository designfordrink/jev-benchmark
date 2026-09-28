from __future__ import annotations
from jev_bench.tasks.j01_cartpole import J01CartPole
from jev_bench.tasks.j02_harth import J02Harth

def get_task(name: str):
    if name == "j01_cartpole": return J01CartPole()
    if name == "j02_harth": return J02Harth()
    raise ValueError(f"Unknown task: {name}")

def list_tasks() -> list[str]: return ["j01_cartpole", "j02_harth"]
