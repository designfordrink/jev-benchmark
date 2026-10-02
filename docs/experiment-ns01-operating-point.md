# NS-01 — First North Star Operating-Point Experiment

## Purpose

NS-01 is the first experiment designed specifically to answer the new North Star question with an existing benchmark task.

It uses **J04 Tetris** because J04 already contains the required decomposition:
- the environment generates legal placements;
- the teacher/heuristic evaluates candidates;
- the tiny selector chooses one candidate;
- the environment produces sequential downstream behavior;
- the implementation already measures model size, decision latency, fallback, confidence, regret and permutation robustness.

No new environment is needed for NS-01.

## Research question

> **How small can the runtime candidate selector become before the complete sequential system loses useful behavior?**

This is more important than asking which hidden size has the highest teacher agreement.

## System boundary

~~~text
Tetris environment
      |
      | legal placements + compact state
      v
tiny runtime selector
      |
      | selected legal placement
      v
environment
      |
      v
score / lines / pieces
~~~

The expensive teacher is a development-time supervisor used to create training targets. It is not part of the runtime loop.

## Experimental matrix

Run the tiny selector at:

~~~text
hidden_units = 1, 2, 4, 8, 16, 32, 64
seeds        = 0, 1, 2
~~~

For each operating point record:
- mean return;
- mean lines;
- mean pieces;
- teacher agreement;
- teacher regret;
- model parameter count;
- FP32 artifact size;
- mean decision latency;
- p95 decision latency;
- legal-action rate;
- permutation invariance;
- confidence;
- fallback rate when enabled.

Also run the non-tiny/heuristic system as the reference operating point.

## Primary output

Do **not** select a winner.

Construct an operating-point table:

| Capacity | Utility | Size | Latency | Fallback | Robustness |
|---|---:|---:|---:|---:|---:|
| heuristic/reference | reference | — | — | — | reference |
| 1 | ... | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... | ... |
| 4 | ... | ... | ... | ... | ... |
| 8 | ... | ... | ... | ... | ... |
| 16 | ... | ... | ... | ... | ... |
| 32 | ... | ... | ... | ... | ... |
| 64 | ... | ... | ... | ... | ... |

Then identify **feasible operating points** for explicit deployment constraints.

Example constraint sets:

### Ultra-small controller
- model artifact ≤ 4 KB;
- p95 decision latency ≤ 1 ms;
- legal-action rate = 100%;
- fallback allowed.

### Small edge controller
- model artifact ≤ 16 KB;
- p95 decision latency ≤ 5 ms;
- fallback rate ≤ 10%.

These are example analysis constraints, not universal requirements.

## North Star utility gate

A tiny operating point is considered **system-viable for a declared constraint set** only when all selected gates are satisfied:
1. downstream utility remains above the experiment's declared minimum;
2. runtime size fits the memory/model budget;
3. latency fits the declared latency budget;
4. legal-action rate remains acceptable;
5. robustness does not collapse;
6. fallback, if used, stays within the declared system budget.

The utility threshold must be declared **before** inspecting the result.

For NS-01, use a relative threshold rather than inventing an absolute Tetris score:

> retain at least **90% of the reference system's mean downstream return**, with the threshold reported explicitly and not treated as a universal definition of usefulness.

The 90% value is an experiment gate, not a claim that every deployment should use 90%.

## Required interpretation

NS-01 should produce one of three types of result for each declared constraint set:

### A. Viable tiny operating point
A tiny selector satisfies the declared resource and downstream-utility constraints.

This is positive evidence for the decomposition.

### B. No viable tiny operating point
Every sufficiently small model violates at least one constraint.

This identifies a boundary where the decomposition is not useful under those constraints.

### C. Selective viability
A tiny model is viable only with confidence-based fallback/escalation.

This is especially important: the result can show that **tiny model + selective fallback** is a useful architecture even when tiny model alone is not.

## Anti-cheating / scientific controls

NS-01 must not:
- rank models by teacher agreement alone;
- use test episodes to select hidden size;
- count training-time teacher calls as runtime inference;
- hide fallback calls from the runtime cost report;
- call a model useful solely because it is smaller;
- use a threshold chosen after seeing the results.

The reference heuristic and tiny selectors must use the same environment and evaluation episodes for comparable downstream measurements.

## Commands

Reference:

~~~bash
jev-bench j04-run --policy heuristic --episodes 50 --max-pieces 300 --seed 0
~~~

Tiny-model sweep:

~~~bash
jev-bench j04-sweep \
  --hidden-units 1,2,4,8,16,32,64 \
  --train-episodes 100 \
  --test-episodes 50 \
  --max-train-pieces 80 \
  --max-test-pieces 300 \
  --epochs 20 \
  --seed 0 \
  --output results/ns01-seed0.json
~~~

Repeat with seeds 1 and 2.

Risk/coverage:

~~~bash
jev-bench j04-risk-coverage \
  --model tiny_mlp \
  --hidden-units 8 \
  --train-episodes 100 \
  --test-episodes 50 \
  --seed 0 \
  --output results/ns01-risk-coverage.json
~~~

## What NS-01 should tell us

The important result is not:

> "8 hidden units is the best model."

The useful result is something like:

> "Under a 16 KB / 5 ms / ≥90% reference-utility constraint, a tiny selector can operate with X parameters and Y fallback rate."

Or:

> "No tiny operating point satisfies the constraint without unacceptable fallback."

That is the first result that directly answers the project's new North Star.