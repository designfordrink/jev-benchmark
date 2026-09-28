from __future__ import annotations
from jev_bench.policies.rule import CartPoleRulePolicy
from jev_bench.policies.tiny_mlp import TinyMLPPolicy

def get_policy(name: str):
    if name == "rule": return CartPoleRulePolicy()
    if name == "tiny_mlp": return TinyMLPPolicy()
    raise ValueError(f"Unknown policy: {name}")

def list_policies() -> list[str]: return ["rule", "tiny_mlp"]
