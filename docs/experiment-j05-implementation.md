# J05 — JEV Reward RL: implementation

J05 integrates the Bring-AI/jev-rl pattern:

**RL agent → action → environment transition → JEV judge → reward → RL update**

JEV does **not** choose the action.

## Components

- src/jev_bench/envs/key_quest.py — deterministic Key Quest environment.
- src/jev_bench/tasks/j05_jev_rl.py — reward providers, cache, judge evaluation and tabular Q-learning.
- tests/test_j05.py — deterministic/caching/regression tests.
- src/jev_bench/tasks/registry.py — task registration.
- src/jev_bench/cli.py — CLI entry points.

## Reward providers

| Provider | Purpose |
|---|---|
| native | deterministic environment reward; reference baseline |
| rules | hand-written rule judge; interpretable baseline |
| jev | OpenRouter-backed JEV-style probabilistic judge |

The jev adapter expects OPENROUTER_API_KEY. The model is supplied explicitly with --model.

## Judge evaluation

Run the deterministic reference first:

    jev-bench j05-judge --provider native
    jev-bench j05-judge --provider rules

This evaluates the reward provider against independent transition labels. It reports reward MAE, exact reward rate, mean confidence and abstention rate.

The current independent labels are generated from the deterministic environment event. They are benchmark ground truth for this synthetic task, not an external dataset.

## JEV evaluation

Use a persistent cache:

    jev-bench j05-judge --provider jev --model <OPENROUTER_MODEL> --cache .cache/j05-jev.json

Repeat the same command. Cache hits must produce the same judgment without another model call.

## RL comparison

Use the same learner and seed matrix for every provider:

    jev-bench j05-run --provider native --episodes 100 --seed 0
    jev-bench j05-run --provider rules --episodes 100 --seed 0
    jev-bench j05-run --provider jev --model <OPENROUTER_MODEL> --cache .cache/j05-jev.json --episodes 100 --seed 0

For the benchmark protocol, repeat at least seeds 0, 1, 2.

## Interpretation rule

Do not collapse these into one score.

1. Judge quality: does JEV assign useful rewards to held-out transitions?
2. Learning quality: does the same RL learner learn from those rewards?
3. Systems cost: how many calls, cache hits, milliseconds and tokens/dollars are required?

A high RL success rate alone does not establish that JEV is a better reward mechanism. The tested chain is:

**judge quality → reward signal quality → learning quality**, under the same environment and learner.

## Next research increment

- fixed train/holdout transition split;
- independent labels stored as versioned data;
- explicit confidence escalation/abstention fallback;
- adversarial reward-hacking cases;
- representation perturbation tests;
- multi-seed aggregate report;
- cache/live equivalence report;
- cost accounting compatible with jev-benchmark.result/v1.
