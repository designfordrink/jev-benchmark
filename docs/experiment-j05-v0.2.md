# J05 v0.2 — held-out and robustness protocol

J05 now separates the benchmark into four layers:

1. Reward judge quality on a deterministic held-out transition split.
2. Confidence behavior: abstention and rules fallback.
3. Adversarial robustness: transitions chosen to expose superficial state-only or event-confusion heuristics.
4. Downstream RL: identical tabular Q-learning under each reward provider.

## Held-out protocol

`split_transition_corpus()` deterministically hashes the full transition identity and creates a disjoint 75/25 train/holdout split.

The holdout is used by `j05-judge`. It must never be used to tune the JEV prompt/model.

The current ground truth is generated from the deterministic environment event. This is explicitly a synthetic benchmark label, not an external human-labeled dataset.

## Confidence fallback

The JEV provider emits a confidence value and abstains below 0.60. `ConfidenceFallback` converts this into an operational policy:

**JEV judgment → if low confidence, use rules reward.**

This lets us measure whether abstention reduces downstream damage at the cost of extra rule evaluation.

## Adversarial protocol

`adversarial_transitions()` contains explicit edge cases around walls, boundaries, lava and terminal behavior.

## Representation robustness

`representation_variants()` provides semantically identical transition representations. Future iterations should expand this to alternate textual renderings while keeping the underlying transition identical.

## Required result matrix

For seeds 0, 1, 2:

| Provider | Held-out MAE | Exact rate | Confidence | Abstention | Fallback | RL success | RL return | Calls | Cache hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native | | | | | | | | | |
| rules | | | | | | | | | |
| JEV | | | | | | | | | |
| JEV + fallback | | | | | | | | | |

Do not collapse these columns into a single score.

## Interpretation

The key question is whether JEV provides a useful, reliable and economical learning signal, not whether an LLM can solve Key Quest.

The evidence chain is:

**JEV judgment → reward fidelity → robust confidence behavior → downstream learning**, with cost and caching explicit.
