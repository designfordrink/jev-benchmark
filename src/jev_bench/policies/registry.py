from __future__ import annotations
from jev_bench.policies.rule import CartPoleRulePolicy
from jev_bench.policies.tiny_mlp import TinyMLPPolicy
from jev_bench.policies.harth_tiny_mlp import HarthTinyMLP
from jev_bench.policies.harth_baseline import HarthNearestCentroid
from jev_bench.policies.j03_candidate import CandidateTinyMLP, CandidateLinearRegressor

def get_policy(name: str):
    if name == "rule": return CartPoleRulePolicy()
    if name == "tiny_mlp": return TinyMLPPolicy()
    if name == "harth_tiny_mlp": return HarthTinyMLP()
    if name == "harth_nearest_centroid": return HarthNearestCentroid()
    if name == "j03_tiny_mlp": return CandidateTinyMLP()
    if name == "j03_linear": return CandidateLinearRegressor()
    raise ValueError(f"Unknown policy: {name}")

def list_policies() -> list[str]: return ["rule", "tiny_mlp", "harth_tiny_mlp", "harth_nearest_centroid", "j03_tiny_mlp", "j03_linear"]
