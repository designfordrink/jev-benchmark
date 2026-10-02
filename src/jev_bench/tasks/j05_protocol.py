from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jev_bench.tasks.j05_jev_rl import (
    RewardProvider,
    build_transition_corpus,
    evaluate_judge,
    independent_label,
    run_j05,
    split_transition_corpus,
)


def _transition_key(t) -> str:
    return json.dumps(
        {
            "state": t.state,
            "action": t.action,
            "next_state": t.next_state,
            "event": t.event,
            "terminated": t.terminated,
        },
        sort_keys=True,
    )


def write_label_corpus(path: str | Path) -> dict[str, Any]:
    """Materialize deterministic independent labels as a versioned artifact."""
    rows = build_transition_corpus()
    payload = {
        "schema_version": "jev-benchmark.j05-labels/v1",
        "task": "j05_jev_rl",
        "label_source": "KeyQuestEnv deterministic event",
        "examples": [
            {
                "state": list(t.state),
                "action": t.action,
                "next_state": list(t.next_state),
                "event": t.event,
                "terminated": t.terminated,
                "reward": independent_label(t),
                "key": _transition_key(t),
            }
            for t in rows
        ],
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"schema_version": payload["schema_version"], "examples": len(rows), "output": str(p)}


def read_label_corpus(path: str | Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "jev-benchmark.j05-labels/v1":
        raise ValueError("unsupported J05 label corpus schema")
    return payload


def evaluate_adversarial(provider: RewardProvider) -> dict[str, Any]:
    from jev_bench.tasks.j05_jev_rl import adversarial_transitions

    rows = adversarial_transitions()
    errors = [abs(provider.judge(t).reward - independent_label(t)) for t in rows]
    return {
        "provider": provider.name,
        "examples": len(rows),
        "reward_mae": sum(errors) / len(errors),
        "max_abs_error": max(errors),
        "exact_reward_rate": sum(e == 0.0 for e in errors) / len(errors),
    }


def evaluate_representation_consistency(provider: RewardProvider) -> dict[str, Any]:
    from jev_bench.tasks.j05_jev_rl import representation_variants

    rows = build_transition_corpus()
    checked = 0
    consistent = 0
    for t in rows:
        variants = representation_variants(t)
        judgments = [provider.judge(v) for v in variants]
        checked += 1
        if max(j.reward for j in judgments) - min(j.reward for j in judgments) <= 1e-9:
            consistent += 1
    return {
        "provider": provider.name,
        "examples": checked,
        "representation_consistency_rate": consistent / checked,
    }


def run_multi_seed(
    provider_factory,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    episodes: int = 100,
) -> dict[str, Any]:
    rows = [run_j05(provider_factory(), episodes=episodes, seed=seed) for seed in seeds]
    success = [float(r["success_rate"]) for r in rows]
    mean_return = [float(r["mean_return"]) for r in rows]

    def mean(xs):
        return sum(xs) / len(xs)

    def sample_std(xs):
        if len(xs) < 2:
            return 0.0
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    return {
        "schema_version": "jev-benchmark.j05-multi-seed/v1",
        "task": "j05_jev_rl",
        "seeds": list(seeds),
        "repeat_count": len(seeds),
        "episodes_per_seed": episodes,
        "runs": rows,
        "aggregate": {
            "success_rate_mean": mean(success),
            "success_rate_std": sample_std(success),
            "mean_return_mean": mean(mean_return),
            "mean_return_std": sample_std(mean_return),
        },
    }


def run_protocol(provider_factory, *, seeds=(0, 1, 2, 3, 4), episodes=100) -> dict[str, Any]:
    """One auditable J05 package: judge quality + adversarial + robustness + RL."""
    judge = provider_factory()
    train, holdout = split_transition_corpus()
    judge_result = evaluate_judge(judge, holdout=holdout)
    adversarial = evaluate_adversarial(provider_factory())
    robustness = evaluate_representation_consistency(provider_factory())
    multi = run_multi_seed(provider_factory, seeds=tuple(seeds), episodes=episodes)
    return {
        "schema_version": "jev-benchmark.j05-protocol/v1",
        "task": "j05_jev_rl",
        "protocol": "judge-quality-learning-quality-systems-cost",
        "corpus": {
            "total": len(train) + len(holdout),
            "train": len(train),
            "holdout": len(holdout),
            "split": "deterministic hash split",
        },
        "judge_quality": judge_result,
        "adversarial": adversarial,
        "representation_robustness": robustness,
        "learning_quality": multi,
        "systems_cost": {
            "calls": int(getattr(judge, "calls", 0)),
            "cache_hits": int(getattr(judge, "cache_hits", 0)),
        },
    }
