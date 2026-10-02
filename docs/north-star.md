# North Star — JEV Benchmark

## The question

> **When does a very small model become a genuinely useful component of a decision-making system?**

The benchmark studies a specific systems architecture:

> **How much useful decision-making can be moved into an extremely small model when expensive intelligence — perception, planning, supervision and constraint checking — is kept outside the runtime model?**

This is the project's North Star.

The purpose is **not** to prove that tiny models are inherently better, and not to minimize parameters for their own sake.

The purpose is to find the boundary at which a tiny model becomes useful enough to deploy as a frequent local decision component.

## Why this matters

A large model or server-side system can perform expensive work:

- perception;
- interpretation;
- planning;
- candidate generation;
- constraint checking;
- supervision;
- reward/judging;
- retraining.

A constrained device may not be able to run that intelligence on every decision.

Instead, the external system can reduce the problem to a compact decision interface:

```
expensive system
      |
      | compact state / legal candidates
      v
tiny runtime model
      |
      | decision
      v
deterministic executor
      |
      v
observable outcome
```

This architecture is relevant to ESP32/STM32/RP2040-class devices, battery-powered systems, robotics controllers, industrial automation, IoT, wearables and other latency- or resource-constrained systems.

## What counts as "useful"

A tiny model is useful only in the context of a larger system.

The benchmark therefore evaluates four dimensions together:

1. **Decision quality**
   - task accuracy;
   - reward;
   - regret;
   - survival / task completion;
   - legal-action rate;
   - downstream utility.

2. **Runtime efficiency**
   - parameter count;
   - model/artifact size;
   - memory requirements where measurable;
   - single-decision latency;
   - p95 latency.

3. **Selective reliability**
   - confidence;
   - abstention;
   - coverage;
   - fallback frequency;
   - accepted-decision risk.

4. **System utility**
   - downstream sequential behavior;
   - robustness;
   - training/learning quality;
   - expensive-model calls;
   - cache hits;
   - latency/cost trade-offs.

The benchmark should answer:

> **What is the smallest/cheapest operating point that still provides enough downstream utility for the intended system?**

There is no universal answer and therefore no universal benchmark winner.

## The unit of research

The unit of research is not:

```
model -> score
```

It is:

```
system decomposition
      |
      v
compact decision interface
      |
      v
tiny model
      |
      v
runtime behavior
      |
      v
system utility
```

A result is interesting when shrinking the runtime component still leaves the overall system useful.

## The decomposition hypothesis

The benchmark tests whether expensive intelligence can be moved across the runtime boundary without destroying useful behavior.

### Candidate-selection form

```
perception / planner
        |
        v
legal candidates
        |
        v
tiny selector
        |
        v
executor
        |
        v
reward
```

The planner is responsible for producing legal candidates. The tiny model is responsible only for selecting among them.

### Reward-learning form

```
environment transition
        |
        v
JEV / LLM judge
        |
        v
reward signal
        |
        v
RL learner
        |
        v
runtime policy
```

Here JEV is a development/training-time intelligence source, not the runtime action selector.

## The benchmark progression

The task suite should progressively answer harder versions of the same systems question:

| Stage | Question |
|---|---|
| J01 | Can an extremely small decision policy act at all, and what is its runtime cost? |
| J02 | Can a tiny model extract a useful compact state from real sensor data? |
| J03 | If expensive planning produces legal candidates, can a tiny model select one? |
| J04 | Does tiny candidate selection remain useful over repeated sequential decisions? |
| J05 | Can expensive JEV supervision produce a learning signal that yields useful downstream behavior? |
| Future | Which real embedded/edge workloads retain enough utility to justify deployment? |

J01 is a measurement baseline. J02 tests compact perception. J03 establishes the planner → tiny selector decomposition. J04 tests that decomposition in a sequential environment. J05 tests a different boundary: expensive intelligence during learning, cheap behavior at runtime.

## Required comparison

For every serious task, the benchmark should compare at least:

- a conventional/non-tiny baseline;
- one or more tiny capacities;
- the same task with the external decomposition;
- fallback/escalation where applicable.

The result should show the trade-off rather than only the best tiny model.

## Operating-point report

A benchmark report should make it possible to identify operating points such as:

```
             downstream utility
                    ^
                    |
             A  *  |       larger model
                *  |
          B  *     |       tiny model
             *     |
       C  *       |
         +------------------------>
             latency / size / cost
```

The important output is the **frontier** and the constraints around it.

For an intended device, the user should be able to ask:

- Does this fit the memory budget?
- Is single-decision latency low enough?
- How often does it need fallback?
- What downstream utility remains?
- What happens under distribution/representation changes?
- What expensive work has been moved outside the device?
- What is the cost of that external work during development or operation?

## What would falsify the idea

The benchmark is successful scientifically even if it finds that a proposed decomposition is not useful.

Examples of negative evidence:

- tiny models require too much fallback;
- small models lose too much downstream utility;
- the compact representation discards information needed for decisions;
- candidate generation makes the task easy but hides unrealistic planner cost;
- confidence is not reliable enough for selective execution;
- sequential errors compound;
- JEV reward is plausible but does not produce useful learning;
- the external supervision cost exceeds the benefit of cheap runtime inference.

These are findings, not benchmark failures.

## North Star success criterion

The project has demonstrated practical value when it can produce reproducible cases where:

1. expensive intelligence is moved outside the runtime decision loop;
2. the remaining decision can be handled by a very small model;
3. the resulting system retains useful downstream behavior;
4. the runtime component has a measurable resource/latency advantage;
5. confidence/fallback can bound the cost of mistakes;
6. the result survives held-out, robustness and multi-seed evaluation.

The benchmark should also clearly identify cases where these conditions fail.

## Long-term destination

The eventual benchmark should move from synthetic demonstrations toward real decision workloads while preserving the same contract:

```
expensive intelligence
        ↓
compact decision interface
        ↓
tiny runtime component
        ↓
deterministic / controlled execution
        ↓
measurable system outcome
```

The end goal is a map of **where tiny decision components are actually useful**, not a leaderboard of tiny models.
