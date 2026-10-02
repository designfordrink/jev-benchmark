# JEV-RL integration into JEV Benchmark

## Purpose
Use Bring-AI/jev-rl as a reference RL reward task inside the benchmark, without replacing the existing J01–J04 suite.

Reference project: https://github.com/Bring-AI/jev-rl

The benchmark should test a stronger question than whether an LLM can judge a transition:
> Can a JEV-provided learning signal improve or preserve downstream learning under controlled, reproducible, held-out evaluation?

## What we import from jev-rl

| Feature | JEV-RL | JEV Benchmark decision |
|---|---|---|
| RL agent receives JEV reward | Yes | Keep |
| Native/rule/JEV reward comparison | Yes | Keep as mandatory baselines |
| Multiple random seeds | 3 | Keep; minimum 3 |
| Held-out evaluation | Yes | Strengthen with explicit held-out task/seed protocol |
| Probability distribution over reward criteria | Yes | Keep |
| Confidence | Returned by judge | Make operational |
| Cache | Central to cost control | Mandatory cache-first mode |
| Human-designed reward rubric | Yes | Keep for reference task, explicitly label it |
| Reward hacking tests | Limited | Add mandatory adversarial cases |
| Judge disagreement | Limited | Add |
| Representation robustness | Limited | Add |
| Independent ground-truth reward | Not central | Add where feasible |

## New benchmark task

### J05 — JEV Reward RL

A small deterministic environment modeled on the Key Quest idea from jev-rl.

State: `(x, y, has_key)`

Actions: `up`, `right`, `down`, `left`

The environment contains walls, a key, an exit and failure states.

The same fixed transition set is evaluated by native reward, deterministic rule reward and JEV reward. The RL learner is held constant across reward providers.

## Required protocols

### P1 — Baseline learning
Train the same RL algorithm under native, rules and JEV reward.

Report: final return, success rate, steps to success, training curve, seed variance, JEV call count, cache hit rate, estimated JEV cost and wall-clock time.

### P2 — Reward fidelity
Before training, evaluate JEV on a held-out transition set with independently generated ground-truth labels.

Report event classification accuracy, reward MAE, expected-reward calibration, confidence, abstention rate and confusion matrix.

This separates JEV understanding of the transition from downstream RL performance.

### P3 — Confidence escalation
Three modes: accept (always use JEV reward), abstain (low-confidence judgments are not accepted), escalate (low-confidence judgments are sent to a stronger or independent judge or deterministic verifier).

Record coverage and fallback rate rather than hiding rejected decisions.

### P4 — Reward hacking
Construct transitions where superficial textual cues conflict with actual environment progress.

Examples: movement toward a key followed by loss of key; apparently positive event with lower true objective value; repeated harmless action that appears productive; irrelevant salient event dominating the description.

Measure whether reward follows the actual task objective rather than salient wording.

### P5 — Representation robustness
Evaluate semantically identical transitions under reordered fields, paraphrased descriptions, irrelevant distractor text, renamed entities and equivalent state serialization.

Measure reward-distribution stability.

### P6 — Cached vs live
Run identical protocols with live JEV calls and with a fully populated cache.

Verify identical metrics while reporting differences in calls, latency and cost.

## Metrics

Keep three separate layers.

### Judge quality
- reward accuracy
- reward MAE
- rank correlation with ground truth
- calibration
- confidence
- abstention
- disagreement
- robustness delta

### Learning quality
- final success
- mean return
- sample efficiency
- steps to target success
- seed variance
- reward-to-performance correlation

### Systems cost
- JEV calls
- cache hit rate
- latency
- estimated token cost
- estimated dollar cost
- total wall-clock time

No single aggregate score should collapse these dimensions.

## Methodological rule

The benchmark must not claim that JEV is better RL merely because a JEV run reaches high final success.

A valid result must distinguish:

`JEV judge quality → reward quality → learning quality`

A failure at any layer remains visible.

## Relation to existing J01–J04

J01–J04 emphasize compact decision-making, confidence, abstention, robustness and resource-size trade-offs.

J05 adds a different axis: LLM/JEV as a learning signal rather than merely an action selector.

## Implementation sequence

1. Freeze a deterministic Key Quest environment and transition corpus.
2. Implement native and rule reward providers.
3. Implement JEV provider behind the existing Policy/Decision/result contracts.
4. Add persistent cache with exact input-key hashing.
5. Add held-out transition labels.
6. Add confidence/abstention/escalation modes.
7. Add reward-hacking and representation-robustness suites.
8. Train the same RL learner across all reward providers.
9. Emit `jev-benchmark.result/v1` artifacts.
10. Produce a J05 comparison report with judge, learning and cost metrics separated.

## Acceptance criteria

J05 is ready when all three reward providers run through one deterministic protocol; at least 3 seeds are reproducible; held-out transitions are never used for training; cache replay produces identical rewards to live calls; confidence and abstention are recorded; adversarial reward-hacking cases are included; representation robustness is measured; result artifacts conform to `jev-benchmark.result/v1`; cost and call counts are recorded; and no synthetic fixture is presented as external-dataset performance.

## Expected scientific value

This turns jev-rl from a standalone demonstration into a controlled benchmark component.

The central experiment becomes:
> When an RL system needs a reward function that is difficult to hand-code, how useful, reliable, robust and economical is JEV as a substitute learning signal?