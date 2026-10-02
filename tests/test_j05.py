from jev_bench.envs.key_quest import KeyQuestEnv
from jev_bench.tasks.j05_jev_rl import (
    JsonFileCache, NativeReward, RuleReward, build_transition_corpus,
    evaluate_judge, run_j05,
)


def test_key_quest_is_deterministic():
    a, b = KeyQuestEnv(), KeyQuestEnv()
    assert a.reset() == b.reset()
    assert [a.step(i).event for i in range(4)] == [b.step(i).event for i in range(4)]


def test_transition_corpus_is_reproducible():
    assert build_transition_corpus() == build_transition_corpus()


def test_native_and_rules_have_independent_ground_truth():
    for provider in (NativeReward(), RuleReward()):
        result = evaluate_judge(provider)
        assert result["reward_mae"] == 0.0
        assert result["exact_reward_rate"] == 1.0


def test_cache_roundtrip(tmp_path):
    cache = JsonFileCache(tmp_path / "j05-cache.json")
    assert cache.get("missing") is None
    cache.put("x", {"reward": 1.0})
    assert cache.get("x") == {"reward": 1.0}


def test_rl_runner_is_reproducible():
    a = run_j05(NativeReward(), episodes=30, seed=7)
    b = run_j05(NativeReward(), episodes=30, seed=7)
    assert a["success_rate"] == b["success_rate"]
    assert a["mean_return"] == b["mean_return"]


def test_train_holdout_split_is_disjoint_and_nonempty():
    train, holdout = split_transition_corpus()
    assert train and holdout
    assert set(train).isdisjoint(set(holdout))


def test_adversarial_cases_have_explicit_labels():
    cases = adversarial_transitions()
    assert len(cases) >= 5
    assert all(t.event for t in cases)


def test_representation_variants_preserve_semantics():
    for t in adversarial_transitions():
        assert all(v.event == t.event and v.next_state == t.next_state for v in representation_variants(t))
