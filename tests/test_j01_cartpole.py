from jev_bench.policies.rule import CartPoleRulePolicy
from jev_bench.tasks.j01_cartpole import J01CartPole


def test_j01_contract_and_determinism():
    task = J01CartPole()
    a = task.evaluate(CartPoleRulePolicy(), episodes=2, seed=7)
    b = task.evaluate(CartPoleRulePolicy(), episodes=2, seed=7)
    assert a["task"] == "j01_cartpole"
    assert a["policy"] == "rule"
    assert a["episodes"] == 2
    assert a["mean_return"] == b["mean_return"]
    assert a["mean_episode_length"] == b["mean_episode_length"]
    assert a["mean_action_latency_us"] >= 0
