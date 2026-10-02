from jev_bench.envs.key_quest import KeyQuestEnv, Transition
from jev_bench.tasks.j05_jev_rl import (
    ConfidenceFallback, JsonFileCache, JEVSystemOneReward, NativeReward,
    REWARD_LEVELS, RewardJudgment, RuleReward, adversarial_transitions,
    build_transition_corpus, evaluate_judge, representation_variants,
    run_j05, split_transition_corpus,
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


def test_confidence_fallback_replaces_abstention():
    class Abstaining:
        name = "mock_jev"
        def judge(self, transition):
            return RewardJudgment(0.0, 0.2, abstain=True, source=self.name)
    provider = ConfidenceFallback(Abstaining(), threshold=0.6)
    result = provider.judge(adversarial_transitions()[0])
    assert result.abstain is False
    assert result.reward == 0.0
    assert provider.fallback_count == 1

def test_systemone_state_does_not_contain_hidden_event():
    t = adversarial_transitions()[0]
    state = JEVSystemOneReward._state(t)
    assert "event" not in state
    assert state["state"] == t.state
    assert state["next_state"] == t.next_state


def test_systemone_questions_are_fixed_choice_contract():
    question = JEVSystemOneReward._questions()["event"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == set(REWARD_LEVELS)


def test_cache_key_does_not_depend_on_hidden_event():
    t = adversarial_transitions()[0]
    alternative = Transition(t.state, t.action, t.next_state, "move", t.terminated)
    assert JsonFileCache.key(t, "typesafe/jev-1.13", "j05-systemone-v1") == JsonFileCache.key(
        alternative, "typesafe/jev-1.13", "j05-systemone-v1"
    )
