from __future__ import annotations
from jev_bench.tasks.j01_cartpole import J01CartPole
from jev_bench.tasks.j02_harth import J02Harth
from jev_bench.tasks.j03_candidate_selection import J03CandidateSelection
from jev_bench.tasks.j04_tetris import J04Tetris
from jev_bench.tasks.j05_jev_rl import run_j05, evaluate_judge

def get_task(name:str):
    if name=="j01_cartpole": return J01CartPole()
    if name=="j02_harth": return J02Harth()
    if name=="j03_candidate_selection": return J03CandidateSelection()
    if name=="j04_tetris": return J04Tetris()
    if name=="j05_jev_rl": return J05Task()
    raise ValueError(f"Unknown task: {name}")

def list_tasks()->list[str]: return ["j01_cartpole","j02_harth","j03_candidate_selection","j04_tetris","j05_jev_rl"]

class J05Task:
    name = "j05_jev_rl"
    def make_environment(self, seed=None):
        from jev_bench.envs.key_quest import KeyQuestEnv
        return KeyQuestEnv()
    def evaluate(self, policy, *, episodes, seed):
        raise NotImplementedError("J05 is evaluated through provider-based commands, not a Policy")
