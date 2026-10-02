# J05 — protocol v1: judge quality → learning quality → systems cost

The next J05 stage is not another environment. It is a stricter experiment around the existing Key Quest environment.

The protocol follows the main limitation identified by the JevRL reference experiment: successful RL training alone does not establish that a model judge is superior to a deterministic reward design. The reference experiment explicitly calls for stronger tuned baselines, more seeds, independently collected score tables, and tasks whose rubrics are less directly reproducible by a few rules.

## 1. Independent held-out reward labels

The deterministic Key Quest environment remains the source of benchmark ground truth.

We materialize the transition labels into:

`jev-benchmark.j05-labels/v1`

The artifact records the exact transition identity, event and reference reward. This prevents the evaluation code from silently changing the target labels while a judge is being tested.

The train/holdout split is deterministic and disjoint. Judge quality is measured only on the held-out partition.

## 2. Adversarial reward-hacking cases

J05 already contains transitions designed to distinguish the actual event from superficial state-only heuristics.

The protocol reports:
- reward MAE;
- maximum absolute error;
- exact reward rate.

These are a control against a judge that learns a convenient shortcut rather than evaluating the transition.

## 3. Representation robustness

Semantically identical representations are evaluated repeatedly.

The first implementation keeps the semantic content identical and establishes the protocol plumbing. A later increment should introduce controlled paraphrase / field-order / irrelevant-field perturbations without changing the ground-truth transition.

The key metric is representation consistency, not a single accuracy number.

## 4. Multi-seed learning

The same learner, environment and reward provider are evaluated on multiple training seeds.

Default protocol:

- seeds: 0, 1, 2, 3, 4;
- 100 episodes per seed.

Report mean and sample standard deviation of:
- success rate;
- mean learned return.

Do not select the best seed.

## 5. Systems cost

For live JEV providers, retain:
- model identity;
- cache hits;
- live calls;
- latency;
- token/cost information when supplied by the provider.

A frozen cache is a reproducibility artifact, not evidence that live scoring is free.

## 6. Required comparison

At minimum, run the same protocol for:

1. Native reward — reference control.
2. Rules reward — deterministic human-designed control.
3. JEV reward — model judge.

The same learner and seed matrix must be used.

Interpretation remains three-dimensional:

**judge quality → learning quality → systems cost**

No single aggregate score is used.

## CLI increments

The current implementation adds reusable protocol functions; the next CLI increment should expose them as one command once the local test suite is green.

Example intended artifact:

`results/j05/protocol-v1.json`

The protocol deliberately does not claim that JEV is better. It establishes the controls needed to test that question.
