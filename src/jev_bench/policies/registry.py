from __future__ import annotations
from jev_bench.policies.rule import CartPoleRulePolicy


def get_policy(name: str):
    if name == "rule":
        return CartPoleRulePolicy()
    raise ValueError(f"Unknown policy: {name}")


def list_policies() -> list[str]:
    return ["rule"]
