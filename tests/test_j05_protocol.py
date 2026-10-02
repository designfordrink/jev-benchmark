from jev_bench.tasks.j05_jev_rl import NativeReward, split_transition_corpus
from jev_bench.tasks.j05_protocol import (
    evaluate_adversarial,
    evaluate_representation_consistency,
    read_label_corpus,
    run_multi_seed,
    write_label_corpus,
)


def test_label_corpus_is_versioned_and_reproducible(tmp_path):
    path = tmp_path / "labels.json"
    result = write_label_corpus(path)
    payload = read_label_corpus(path)
    assert result["schema_version"] == "jev-benchmark.j05-labels/v1"
    assert payload["schema_version"] == result["schema_version"]
    assert len(payload["examples"]) == len(split_transition_corpus()[0]) + len(split_transition_corpus()[1])


def test_native_adversarial_and_representation_controls_are_perfect():
    assert evaluate_adversarial(NativeReward())["reward_mae"] == 0.0
    assert evaluate_representation_consistency(NativeReward())["representation_consistency_rate"] == 1.0


def test_multi_seed_protocol_is_deterministic():
    factory = NativeReward
    a = run_multi_seed(factory, seeds=(0, 1), episodes=10)
    b = run_multi_seed(factory, seeds=(0, 1), episodes=10)
    assert a == b
