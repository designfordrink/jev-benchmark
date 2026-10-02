# JEV Benchmark

**A reproducible research benchmark for tiny decision models.**

Русская версия: [README.ru.md](README.ru.md)

JEV Benchmark is an open-source framework for studying whether **very small models can make useful decisions when they operate on a compact state representation and are embedded in a deterministic decision loop**.

The project is deliberately broader than ordinary model-accuracy benchmarking. It measures not only whether a tiny model predicts correctly, but whether it can produce a **useful downstream action** under constraints such as model size, inference latency, confidence, abstention, fallback behavior and robustness to changes in candidate ordering.

> **Core idea**
>
> raw data → compact state → tiny model → decision/candidate → deterministic executor → reward/feedback

## Why this benchmark exists

Large models often solve tasks by spending more compute. JEV Benchmark studies the opposite question:

> **How far can we push the useful behavior of a very small decision model when the surrounding system performs perception, planning and constraint enforcement?**

A tiny model may only need to answer a narrow question:

- which action should be taken now?
- which legal candidate should be selected?
- is the model confident enough to act?
- should the decision be delegated to a fallback?
- how much performance is lost when the model is compressed?

The central architectural principle is:

**the tiny model is one component of a decision system, not the whole system.**

The benchmark therefore separates:

1. **Perception / state construction** — convert raw data into a compact state.
2. **Planning / candidate generation** — generate a finite set of legal actions when appropriate.
3. **Tiny decision model** — classify, score or rank the available choices.
4. **Execution** — apply the selected action deterministically.
5. **Feedback / evaluation** — measure the downstream result.
6. **Confidence and fallback** — allow the tiny model to abstain when uncertain.

This is especially important for candidate-selection tasks: **the planner generates legal options; the tiny model selects among them.**

## Research questions

### 1. How small can the model become?

Measure the relationship between:

- parameter count;
- model size;
- inference latency;
- task performance;
- downstream reward.

Capacity sweeps and Pareto analysis are preferred over choosing one arbitrary model size.

### 2. When does a tiny model stop being useful?

A model can often be reduced until a plateau or degradation appears. The benchmark records these transitions instead of assuming that larger models are always necessary.

### 3. Is accuracy the right metric?

Not necessarily.

Depending on the task, the benchmark measures:

- accuracy and macro-F1;
- reward and oracle reward;
- regret;
- game return;
- survival;
- legal-action rate;
- latency;
- confidence;
- abstention;
- fallback rate.

A wrong candidate can have almost no downstream cost, so accuracy and regret are intentionally reported separately.

### 4. Can confidence make tiny models more useful?

The intended pattern is:

    tiny model
        |
        +-- high confidence --> execute tiny-model decision
        |
        +-- low confidence --> abstain / fallback
                                  |
                                  +--> rule, teacher, larger model or human

This turns confidence into a system-level control variable rather than just a diagnostic.

### 5. Does the model learn the decision problem or an accidental representation?

Candidate-selection experiments include permutation tests. A useful selector should not change its chosen candidate merely because candidates were presented in a different order.

### 6. What happens when the model is used repeatedly?

A classifier can perform well on independent samples while producing poor sequential behavior. J04 therefore evaluates a tiny selector inside a changing environment where every decision affects the next state.

# Benchmark architecture

The common decision loop is:

    Raw data / state
            |
            v
    Compact state representation
            |
       +----+----+
       |         |
       v         v
    Perception  Planner
                |
                v
         Legal candidates
                |
                v
         Tiny scorer / policy
                |
                v
         Decision + confidence
             /       \
        confident   uncertain
           |           |
           v           v
        execute     fallback
             \       /
              \     /
                v
          reward / metrics

The key boundary is:

**planner generates legal candidates; tiny model selects among them.**

This prevents the benchmark from accidentally measuring the model's ability to invent invalid actions.

# Experimental tracks

| Track | Question | Example |
|---|---|---|
| Perception | Can a tiny model extract a useful state? | HARTH |
| Decision | Can it choose among explicit alternatives? | J03 Candidate Selection |
| Sequential Decision | Can repeated tiny decisions produce useful behavior? | J04 Tetris |
| Control | Can a tiny policy stabilize an environment? | J01 CartPole |
| Hierarchical Decision | Can tiny models participate inside larger systems? | Planned |

The goal is not to optimize one dataset. The goal is to discover **where tiny decision models work, where they fail, and what system architecture makes them useful**.

# Implemented tasks

## J01 — CartPole

J01 is the initial control vertical slice.

It provides:

- deterministic rule baseline;
- tiny fixed-weight MLP size probe;
- action latency measurement;
- confidence measurement;
- abstention / fallback instrumentation;
- parameter and FP32-size accounting.

J01 is primarily a systems and measurement baseline. Its tiny MLP is a fixed-weight synthetic inference-size probe, not a trained claim about CartPole performance.

    jev-bench run --task j01_cartpole --policy rule --episodes 20 --seed 0
    jev-bench sweep --task j01_cartpole --hidden-units 1,2,4,8,16,32,64 --episodes 20 --seed 0

## J02 — HARTH perception

J02 moves the benchmark to real sensor perception.

HARTH activity recordings are converted into fixed-size windows and evaluated with subject-disjoint leave-one-subject-out (LOSO) evaluation.

Default representation:

- 6 acceleration channels;
- 128 samples per window;
- stride 128;
- pure-label windows only;
- subject identity preserved;
- no subject leakage between train and test.

Implemented:

- streaming CSV loader;
- schema validation;
- manifest and inspection;
- trainable Tiny MLP;
- nearest-centroid baseline;
- LOSO;
- repeated multi-seed LOSO;
- risk-coverage;
- hidden-size sweep;
- Pareto analysis;
- single-window inference latency.

A critical reproducibility rule is that **HARTH performance is never fabricated**. Synthetic CSV fixtures are used only for testing the loader and contracts. Published performance must come from an actual run on a pinned dataset.

    jev-bench harth-manifest --dataset-root /path/to/harth
    jev-bench harth-inspect --dataset-root /path/to/harth --subject S015

    jev-bench harth-loso \
      --dataset-root /path/to/harth \
      --model tiny_mlp \
      --hidden-units 8 \
      --output results/j02-loso-h8.json

    jev-bench harth-loso-multi-seed \
      --dataset-root /path/to/harth \
      --model tiny_mlp \
      --hidden-units 8 \
      --seeds 0,1,2,3,4 \
      --output results/j02-loso-multi-seed-h8.json

See docs/experiment-j02-harth.md.

## J03 — Candidate Selection

J03 introduces the central planner → tiny selector → reward pattern.

The planner provides:

- compact context;
- finite legal candidate set;
- candidate-specific features.

The tiny model scores the candidates and selects one.

    state
      |
    planner
      |
    legal candidates
      |
    tiny scorer
      |
    selected candidate
      |
    reward

The current J03 problem is deliberately synthetic. Each instance contains:

- a 6-dimensional context;
- 8 candidates;
- 6-dimensional candidate features;
- a deterministic latent utility known to the evaluator but not to the policy.

Metrics include:

- selection accuracy;
- reward;
- oracle reward;
- regret and p95 regret;
- confidence;
- abstention;
- latency;
- parameter count;
- FP32 model bytes;
- permutation invariance.

The important principle is that **accuracy is not the only objective**.

    jev-bench j03-train --model tiny_mlp --hidden-units 8
    jev-bench j03-train --model linear
    jev-bench j03-sweep --hidden-units 1,2,4,8,16,32,64 --output results/j03-sweep.json

See docs/experiment-j03-candidate-selection.md.

## J04 — Sequential Tetris Candidate Selection

J04 moves candidate selection from synthetic tables into a repeated, changing environment.

    board state + current piece
              |
        legal placements
              |
          tiny scorer
              |
           placement
              |
         board update
              |
            reward
              |
         next decision

The research environment implements:

- deterministic 10×20 board;
- seven tetromino types;
- collision checks;
- gravity/drop placement;
- line clearing;
- legal-placement generation;
- deterministic seeding.

The tiny model never invents arbitrary coordinates. The environment generates legal placements and validates the selected placement before execution.

Metrics include:

- mean/std game return;
- lines cleared;
- pieces survived;
- teacher agreement;
- teacher-relative regret;
- permutation invariance;
- legal-action rate;
- mean/p95 decision latency;
- parameter count;
- FP32 model bytes;
- confidence;
- model coverage;
- fallback rate;
- expected calibration error where defined.

The selective-decision protocol allows the tiny model to abstain below a confidence threshold and delegate selection to the heuristic teacher.

    jev-bench j04-run --policy heuristic --episodes 20
    jev-bench j04-run --policy random --episodes 20

    jev-bench j04-train --model tiny_mlp --hidden-units 8

    jev-bench j04-sweep \
      --hidden-units 1,2,4,8,16,32,64 \
      --output results/j04-sweep.json

    jev-bench j04-risk-coverage \
      --model tiny_mlp \
      --hidden-units 8 \
      --output results/j04-risk-coverage.json

See docs/experiment-j04-tetris.md.

> J04 is a compact research environment, not evidence of performance on a third-party Tetris implementation. The next escalation should preserve the candidate-selection contract while moving toward an established benchmark or richer simulator.

## J05 — JEV Reward RL

J05 adds a second way to use JEV: **not as the action selector, but as a reward/judge that supplies a learning signal to an RL agent**.

The benchmark loop is:

    RL agent
        |
        v
      action
        |
        v
    environment transition
        |
        v
      JEV judge
        |
        v
      reward
        |
        v
     RL update

JEV does **not** choose the action in this experiment. The same learner and environment are compared under different reward providers.

### Current implementation

J05 currently contains:

- deterministic Key Quest environment;
- native environment reward;
- hand-written rule reward;
- OpenRouter-backed JEV-style probabilistic reward provider;
- persistent reward-judgment cache;
- judge evaluation CLI;
- tabular Q-learning runner;
- deterministic and cache/regression tests;
- J05 implementation and evaluation documentation.

The JEV adapter uses `OPENROUTER_API_KEY` and an explicitly selected model.

### Scientific separation

J05 deliberately separates three questions:

1. **Judge quality** — does JEV assign useful rewards to held-out transitions?
2. **Learning quality** — can the same RL learner learn from those rewards?
3. **Systems cost** — how many model calls, cache hits, milliseconds and tokens/dollars are required?

The benchmark does not collapse these dimensions into one score.

### Required next controls

The implementation is being extended toward:

- fixed train/holdout transition splits;
- independent versioned transition labels;
- confidence-based abstention/escalation;
- adversarial reward-hacking cases;
- representation-robustness tests;
- multi-seed aggregation;
- live-vs-cache equivalence;
- cost accounting in the common result schema.

J05 is therefore a **reward-learning benchmark layer**: it asks whether an LLM/JEV-generated learning signal is useful downstream, not merely whether an LLM can produce plausible judgments.

See `docs/experiment-j05-jev-rl.md` and `docs/experiment-j05-implementation.md`.

# Metrics that matter

The benchmark combines model-centric and system-centric measurements.

### Model

- parameter count;
- FP32 parameter bytes;
- serialized model size where available;
- hidden width / capacity.

### Inference

- single-decision latency;
- batch-amortized latency where relevant;
- p95 latency.

Single-window / single-decision latency is especially important because the project targets tiny models in frequent decision loops.

### Decision quality

Depending on the task:

- accuracy;
- macro-F1;
- reward;
- oracle reward;
- regret;
- game return;
- survival;
- lines cleared;
- teacher agreement;
- legal-action rate.

### Confidence and selective prediction

- confidence;
- abstention rate;
- coverage;
- fallback rate;
- risk-coverage curves;
- calibration diagnostics such as ECE where defined.

Confidence is currently score-derived in some tasks and **must not automatically be interpreted as a calibrated probability**.

### Robustness

- candidate permutation invariance;
- subject-disjoint evaluation;
- repeated training seeds;
- deterministic seeds;
- explicit class/schema validation.

# Pareto analysis

The benchmark does not assume that there is one universally optimal model.

Capacity sweeps produce a Pareto view of:

    task performance
          ^
          |        *
          |     *
          |   *
          | *
          +------------------> latency / model size

Typical objectives are:

- maximize task performance;
- minimize inference latency;
- minimize model size.

This makes it possible to identify useful operating points and performance plateaus rather than selecting a model only by accuracy or parameter count.

# Confidence, abstention and fallback

A central hypothesis is that **a tiny model does not have to be correct all the time to be useful**.

    confidence
         |
    +----+----+
    |         |
  above τ   below τ
    |         |
    v         v
 tiny-model  abstain
 action        |
                v
          fallback policy

Fallback may be:

- deterministic rule;
- teacher;
- larger model;
- another policy;
- eventually, a human.

This creates a measurable trade-off between coverage, accepted-decision risk, fallback frequency, downstream reward and compute/latency.

# Reproducibility

A published result should identify at least:

- task;
- protocol;
- dataset and version;
- exact train/test or held-out subject list;
- random seeds;
- windowing / sampling parameters;
- model architecture;
- hidden size;
- training hyperparameters;
- confidence / abstention threshold;
- result schema version.

Repeated experiments retain the individual runs rather than collapsing everything into one opaque number.

## Versioned result artifacts

The repository uses:

- jev-benchmark.result/v1 for individual benchmark results;
- jev-benchmark.sweep/v1 for capacity / parameter sweeps.

Legacy raw JSON can be normalized without rerunning the experiment:

    jev-bench normalize-result --input old-result.json --output result-v1.json

See docs/result-schema-v1.md.

# Data provenance

Datasets are external to the source repository unless explicitly stated otherwise.

For HARTH, the repository provides loaders, validation and download/materialization helpers but does not commit the full dataset.

The provenance question is:

> **Exactly which data produced this number?**

HARTH materialization manifests can record subject files, source information and SHA-256 information.

# What this project is — and is not

## It is

- a research benchmark;
- an executable experimental framework;
- a common contract for tiny policies and selectors;
- a way to compare model size, latency and downstream utility;
- a framework for confidence-aware fallback;
- a progressively harder sequence of tasks.

## It is not

- a leaderboard claiming that tiny models beat large models;
- a collection of fabricated benchmark numbers;
- a generic ML classification benchmark;
- a claim that one tiny architecture is universally optimal;
- a replacement for established real-world benchmarks.

The purpose is to make the question **measurable and reproducible**.

# Current status

**v0.1 — first vertical slices, now extended with reward-learning**

Implemented on `main`:

- common policy / decision contracts;
- versioned result schema;
- J01 CartPole;
- J02 HARTH data pipeline and LOSO evaluation;
- J02 multi-seed analysis;
- J02 capacity / Pareto analysis;
- HARTH Hugging Face archive/materialization helpers and provenance manifests;
- J03 candidate-selection mechanism;
- J04 sequential candidate selection in Tetris;
- confidence / abstention / fallback instrumentation;
- permutation-invariance tests;
- reproducibility-oriented validation;
- **J05 JEV Reward RL** with native/rules/JEV reward providers;
- J05 persistent cache, judge evaluation and tabular Q-learning;
- CLI runners, documentation and automated tests.

### Current integration status

- **PR #5 is open:** real HARTH validation and a Windows runner for the user's 22 local HARTH subject files.
- PR #5 is explicitly a **smoke-run/validation step**, not a published HARTH result.
- J05 implementation is already present on `main`; the full held-out/adversarial/multi-seed reward-learning protocol remains the next research increment.

The J02 code is ready to execute on real HARTH files, but **no HARTH performance number is considered a published benchmark result until it has been produced by the actual runner on the specified dataset**.

# Roadmap

## Near term

1. Finish the real HARTH validation/smoke-run PR and pin the actual dataset provenance.
2. Run J02 multi-seed LOSO on the pinned 22-subject dataset.
3. Run the full hidden-size sweep.
4. Compare size / latency / macro-F1 Pareto points.
5. Extend selective-prediction analysis.
6. Preserve all results as versioned artifacts.
7. Complete J05 held-out reward-fidelity evaluation.
8. Add J05 confidence/abstention, reward-hacking and representation-robustness controls.
9. Add J05 multi-seed and live-vs-cache/cost reports.

## Next benchmark layer

Move candidate selection beyond synthetic tables and the compact Tetris environment toward established environments such as:

- scheduling;
- routing;
- games;
- robotics simulators;
- resource allocation;
- browser / tool action selection.

The important requirement is to preserve the same contract:

    state → legal candidates → tiny selector → execution → reward

## Longer-term research questions

- Where is the useful lower bound on model size?
- Do small models benefit disproportionately from good state compression?
- Is candidate selection easier to compress than direct policy learning?
- How much can confidence + fallback compensate for model capacity?
- Does a tiny model become more useful when a planner guarantees legal actions?
- Which invariances are essential for robust tiny selectors?
- Where do performance plateaus appear as model capacity increases?
- How much energy / compute can be saved for a given downstream utility?
- Which task families genuinely favor tiny decision models?

# Development

Requires Python 3.10+.

    pip install -e '.[dev]'
    pytest
    jev-bench list-tasks

The reference implementations currently use NumPy so that the experimental logic remains transparent and reproducible.

# Repository structure

    src/jev_bench/
    ├── core/          # contracts, candidates, versioned results
    ├── datasets/      # dataset loaders, manifests and validation
    ├── policies/      # tiny models and baselines
    ├── envs/          # research environments
    ├── tasks/         # J01–J05 experiment implementations
    └── cli.py         # command-line benchmark runner

    docs/
    ├── experiment-j02-harth.md
    ├── experiment-j03-candidate-selection.md
    ├── experiment-j04-tetris.md
    ├── experiment-j05-jev-rl.md
    ├── experiment-j05-implementation.md
    └── result-schema-v1.md

    tests/             # deterministic unit / contract tests

# License / status

This repository is an active research project. The benchmark protocol and task suite are expected to evolve as experiments reveal which measurements and controls are actually necessary.
