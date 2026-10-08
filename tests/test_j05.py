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


def test_transition_corpus_contains_timeout():
    assert any(t.event == "timeout" and t.step == 40 for t in build_transition_corpus())


def test_native_and_rules_have_independent_ground_truth():
    for provider in (NativeReward(), RuleReward()):
        result = evaluate_judge(provider)
        assert result["reward_mae"] == 0.0
        assert result["exact_reward_rate"] == 1.0


def test_rules_do_not_depend_on_hidden_event_label():
    t = adversarial_transitions()[0]
    relabeled = Transition(t.state, t.action, t.next_state, "move", t.terminated, t.step)
    assert RuleReward().judge(t).reward == RuleReward().judge(relabeled).reward


def test_transition_records_observable_step():
    env = KeyQuestEnv()
    env.reset()
    t = env.step(1)
    assert t.step == 1


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
        assert all(
            v.event == t.event
            and v.next_state == t.next_state
            and v.step == t.step
            for v in representation_variants(t)
        )


def test_confidence_fallback_replaces_abstention():
    class Abstaining:
        name = "mock_jev"
        def judge(self, transition):
            return RewardJudgment(0.0, 0.2, abstain=True, source=self.name)
    provider = ConfidenceFallback(Abstaining(), threshold=0.6)
    result = provider.judge(adversarial_transitions()[0])
    assert result.abstain is False
    assert result.reward == REWARD_LEVELS["wall"]
    assert provider.fallback_count == 1

def test_systemone_state_does_not_contain_hidden_event():
    t = adversarial_transitions()[0]
    t = Transition(t.state, t.action, t.next_state, t.event, t.terminated, 12)
    state = JEVSystemOneReward._state(t)
    assert "event" not in state
    assert "reward" not in state
    assert "termination_reason" not in state
    assert state["environment"]["width"] == 5
    assert state["environment"]["height"] == 5
    assert state["environment"]["walls"] == [[1, 1], [1, 2], [3, 2], [3, 3]]
    assert state["environment"]["hazards"] == [[2, 3]]
    assert state["environment"]["key_location"] == [2, 2]
    assert state["environment"]["exit_location"] == [4, 4]
    assert state["transition"]["state"]["step"] == 11
    assert state["transition"]["next_state"]["step"] == 12
    assert state["transition"]["action"]["name"] == "RIGHT"
    assert state["transition"]["action"]["delta"] == (1, 0)
    assert state["transition"]["state"]["position"] == list(t.state[:2])
    assert state["transition"]["next_state"]["position"] == list(t.next_state[:2])


def test_systemone_questions_are_fixed_choice_contract():
    question = JEVSystemOneReward._questions()["event"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == set(REWARD_LEVELS)


def test_cache_key_does_not_depend_on_hidden_event():
    t = Transition(*adversarial_transitions()[0].__dict__.values())
    alternative = Transition(t.state, t.action, t.next_state, "move", t.terminated, t.step)
    assert JsonFileCache.key(t, "typesafe/jev-1.13", "j05-systemone-v2-state-aware") == JsonFileCache.key(
        alternative, "typesafe/jev-1.13", "j05-systemone-v2-state-aware"
    )


def test_cache_key_includes_observable_termination_and_step():
    t = adversarial_transitions()[0]
    terminated = Transition(t.state, t.action, t.next_state, t.event, not t.terminated, t.step)
    later = Transition(t.state, t.action, t.next_state, t.event, t.terminated, t.step + 1)
    assert JsonFileCache.key(t, "typesafe/jev-1.13", "j05-systemone-v2-state-aware") != JsonFileCache.key(
        terminated, "typesafe/jev-1.13", "j05-systemone-v2-state-aware"
    )
    assert JsonFileCache.key(t, "typesafe/jev-1.13", "j05-systemone-v2-state-aware") != JsonFileCache.key(
        later, "typesafe/jev-1.13", "j05-systemone-v2-state-aware"
    )
