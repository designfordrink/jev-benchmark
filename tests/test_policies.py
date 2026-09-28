from jev_bench.policies.tiny_mlp import TinyMLPPolicy
from jev_bench.policies.rule import CartPoleRulePolicy
from jev_bench.policies.fallback import FallbackPolicy

def test_tiny_mlp_size_and_confidence():
    p=TinyMLPPolicy(hidden_units=8); d=p.act([0,0,0,0])
    assert p.parameter_count == 58
    assert p.model_size_bytes_fp32 == 232
    assert 0 <= d.confidence <= 1

def test_fallback_routes_low_confidence():
    p=TinyMLPPolicy(hidden_units=1, abstain_threshold=1.0)
    w=FallbackPolicy(p, CartPoleRulePolicy(), threshold=1.0); w.reset(0)
    d=w.act([0,0,0,0])
    assert d.metadata["fallback"] is True
    assert w.fallback_count == 1
