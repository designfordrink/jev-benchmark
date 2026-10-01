from __future__ import annotations
from jev_bench.tasks.j01_cartpole import J01CartPole
from jev_bench.tasks.j02_harth import J02Harth
from jev_bench.tasks.j03_candidate_selection import J03CandidateSelection
from jev_bench.tasks.j04_tetris import J04Tetris

def get_task(name:str):
    if name=="j01_cartpole": return J01CartPole()
    if name=="j02_harth": return J02Harth()
    if name=="j03_candidate_selection": return J03CandidateSelection()
    if name=="j04_tetris": return J04Tetris()
    raise ValueError(f"Unknown task: {name}")

def list_tasks()->list[str]: return ["j01_cartpole","j02_harth","j03_candidate_selection","j04_tetris"]
