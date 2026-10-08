# Local AI Agent Runbook: J05 Live JEV System One

## Purpose

This document is an **execution contract for a local AI agent**.

The agent must run the real J05 JEV System One experiment locally, collect reproducible evidence, validate the outputs, and commit the results back to this GitHub repository.

The agent must **not** invent, edit, or manually manufacture experimental results.

The experiment is intended to answer a narrow question:

> Can the current J05 state-aware JEV System One integration run against the real OpenRouter/JEV service and produce a valid, reproducible reward-judgment signal without access to hidden event labels?

This is an integration and experiment-readiness check. **It is not evidence that JEV is better than the native or rule-based baseline.**

---

## 1. Repository contract

Repository:

`https://github.com/designfordrink/jev-benchmark`

Work from:

`main`

Experiment directory:

`experiments/j05-systemone-live/`

Expected result layout:

```text
experiments/
└── j05-systemone-live/
    ├── README.md
    ├── results/
    │   ├── raw/
    │   ├── reports/
    │   └── logs/
    └── manifests/
```

The local agent must create these directories/files as needed.

### What belongs in Git

Commit:

- raw experiment result JSON;
- a human-readable Markdown report;
- a run manifest containing commit SHA, timestamp, command, model and protocol information;
- relevant test output/log excerpts;
- failure reports when a run fails.

Do **not** commit:

- `.env`;
- `OPENROUTER_API_KEY`;
- API tokens;
- private credentials;
- local virtual environments;
- the JEV API cache unless explicitly requested;
- large temporary files.

The existing repository ignores `.env` and local build/cache files. Keep secrets local.

---

## 2. Preconditions

The local agent must have:

- Git;
- Python 3.10+ compatible with the repository;
- network access;
- a valid OpenRouter API key with access to the configured JEV model;
- permission to push to the GitHub repository.

The API key must already be available locally.

The agent must **never ask the user to paste the API key into GitHub, a Markdown file, a commit, or chat**.

---

## 3. Start from a clean repository

Run:

```powershell
git checkout main
git pull --ff-only
git status --short
```

If the working tree is not clean, stop and report the problem. Do not overwrite unrelated user changes.

Create an experiment branch:

```powershell
git checkout -b agent/j05-systemone-live-YYYYMMDD-HHMM
```

Use the actual local date/time in the branch name.

Record the starting commit:

```powershell
git rev-parse HEAD
```

---

## 4. Install and validate the benchmark

Create/use a local virtual environment if needed, then:

```powershell
python -m pip install -e ".[dev]"
pytest
```

The full test suite must pass before a live JEV call is attempted.

Record:

- Python version;
- package version if available;
- Git commit SHA;
- `pytest` result.

If tests fail, do **not** continue to the live experiment. Save a failure report under:

`experiments/j05-systemone-live/results/reports/`

and commit that report.

---

## 5. Configure credentials locally

Use the existing local `.env` mechanism.

Required:

```text
OPENROUTER_API_KEY=...
```

Do not put the real value into any tracked file.

The J05 live provider is:

```text
jev_systemone
```

The protocol model is currently:

```text
typesafe/jev-1.13
```

The local agent must not silently substitute another model.

If the configured model is unavailable, record the failure rather than changing the model.

---

## 6. First live smoke test

Run:

```powershell
jev-bench j05-judge `
  --provider jev_systemone `
  --model typesafe/jev-1.13 `
  --cache .cache/j05-jev-systemone-v2.json
```

The smoke test must establish that the real service can:

1. receive the state-aware transition representation;
2. return the expected typed choice response;
3. provide probabilities/reward levels;
4. return a valid event choice;
5. produce a usable confidence value;
6. operate without receiving the hidden event label;
7. work with the current cache contract.

If this fails, stop before the multi-episode run.

Save a failure report and the non-secret diagnostic output.

---

## 7. Ten-episode live smoke run

Only if the single judgment succeeds:

```powershell
jev-bench j05-run `
  --provider jev_systemone `
  --model typesafe/jev-1.13 `
  --cache .cache/j05-jev-systemone-v2.json `
  --episodes 10 `
  --seed 0
```

Check that the run completes and produces a structured result.

Record at minimum:

- provider;
- model;
- seed;
- number of episodes;
- reward/judgment statistics;
- confidence;
- abstention rate;
- latency;
- cache statistics if exposed;
- errors/timeouts;
- output artifact path.

The agent must inspect the actual generated JSON rather than only trusting the process exit code.

---

## 8. Required validation of anti-leakage

The live experiment must remain consistent with the J05 state-aware contract.

The JEV-facing transition may contain observable information such as:

- environment geometry;
- current position;
- whether the key is held;
- current step;
- action;
- next position;
- next `has_key` state;
- next step;
- termination flag.

It must **not** contain:

- the hidden event label;
- reference reward;
- termination reason.

The agent must not modify the benchmark implementation merely to make the live model succeed.

If the agent discovers possible leakage, stop and report it.

---

## 9. Escalation to the full protocol

Only run the larger experiment after:

- full `pytest` passes;
- single live judgment succeeds;
- 10-episode smoke run succeeds;
- generated output is structurally valid.

Then run seed 0 for 100 episodes:

```powershell
jev-bench j05-run `
  --provider jev_systemone `
  --model typesafe/jev-1.13 `
  --cache .cache/j05-jev-systemone-v2.json `
  --episodes 100 `
  --seed 0
```

If that succeeds, run the complete five-seed protocol:

- seeds: 0, 1, 2, 3, 4;
- 100 episodes per seed.

Use the repository's existing J05 protocol/runner rather than creating an alternative implementation.

Do not change the number of seeds, episodes, reward levels, abstention threshold, or model without recording an explicit protocol deviation.

---

## 10. Results directory

For each completed run, save a copy of the machine-readable result under:

```text
experiments/j05-systemone-live/results/raw/
```

Use descriptive names, for example:

```text
smoke-10eps-seed0.json
full-100eps-seed0.json
full-5x100.json
```

If the runner already creates a canonical result artifact, copy that artifact rather than reconstructing it.

Do not alter numeric values.

---

## 11. Human-readable report

Create:

```text
experiments/j05-systemone-live/results/reports/YYYY-MM-DD-j05-systemone-live.md
```

The report must contain:

### Run identity

- date/time;
- Git commit SHA;
- experiment branch;
- Python version;
- model;
- provider;
- protocol version;
- cache path/key policy.

### Execution

- exact commands executed;
- test-suite result;
- smoke-test result;
- episode counts;
- seeds;
- elapsed time.

### Results

Report the values produced by the runner, including where available:

- mean reward;
- reward distribution;
- mean confidence;
- abstention rate;
- latency;
- error/timeout count;
- cache hits/misses;
- model-call count;
- cost if the runner exposes a trustworthy value.

### Validation

Explicitly state:

- whether the state-aware request was accepted;
- whether typed choice output was returned;
- whether probabilities were valid;
- whether confidence was usable;
- whether hidden event/reference reward/termination reason were absent;
- whether the run completed without protocol changes.

### Interpretation

Separate these three statements:

1. **Integration result** — whether the live JEV System One path works.
2. **Judge result** — how JEV judgments compare with deterministic reference labels, if the protocol output supports that analysis.
3. **Downstream RL result** — whether the resulting reward signal produces useful learning behavior.

Do not collapse these into one score.

Do not claim that JEV is superior unless the experiment actually contains the required comparison and statistical evidence.

### Limitations

List:

- failed seeds;
- timeouts;
- abstentions;
- cache effects;
- protocol deviations;
- service/model availability issues;
- any other factor that limits interpretation.

---

## 12. Failure handling

A failed experiment is a valid result and must be committed.

Examples:

- API authentication failure;
- unavailable model;
- invalid response schema;
- timeout;
- malformed probability distribution;
- invalid choice;
- unexpected hidden-state dependency;
- runner failure.

For a failure create:

```text
experiments/j05-systemone-live/results/reports/YYYY-MM-DD-j05-systemone-failure.md
```

Include:

- exact command;
- exit code;
- timestamp;
- Git SHA;
- model/provider;
- sanitized error;
- what stage failed;
- whether the failure is reproducible.

Never include secrets or full authorization headers.

---

## 13. Reproducibility manifest

Create:

```text
experiments/j05-systemone-live/manifests/YYYY-MM-DD-run.json
```

Example structure:

```json
{
  "experiment": "j05-systemone-live",
  "protocol": "j05",
  "date_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "git_commit": "<sha>",
  "branch": "<branch>",
  "provider": "jev_systemone",
  "model": "typesafe/jev-1.13",
  "episodes": 10,
  "seeds": [0],
  "commands": [
    "pytest",
    "jev-bench j05-judge ...",
    "jev-bench j05-run ..."
  ],
  "secret_values_recorded": false
}
```

For the full protocol, record all five seeds and the exact commands actually executed.

Do not fabricate fields that the runner did not expose.

---

## 14. Commit results to GitHub

Before committing:

```powershell
git status --short
git diff --check
```

Inspect every tracked result file.

Verify:

```powershell
git grep -n "OPENROUTER_API_KEY=" -- experiments || exit 1
```

The command should find no real API key.

Also verify that no `.env` file is staged:

```powershell
git diff --cached --name-only
```

Then commit:

```powershell
git add experiments/j05-systemone-live
git commit -m "results(j05): record live JEV System One run"
```

Push the experiment branch:

```powershell
git push -u origin HEAD
```

---

## 15. Open a pull request

Create a pull request from the experiment branch into `main`.

Suggested title:

```text
results(j05): live JEV System One experiment
```

PR body must contain:

- what was executed;
- model/provider;
- seeds/episodes;
- whether tests passed;
- result artifact paths;
- whether the full protocol completed;
- any failure/deviation;
- explicit statement that no API secret was committed.

The PR should contain **evidence, not conclusions beyond the protocol**.

---

## 16. Final agent checklist

Before finishing, confirm all applicable items:

- [ ] Started from current `main`.
- [ ] Working tree was clean.
- [ ] Full `pytest` passed.
- [ ] Real JEV System One smoke test ran.
- [ ] 10-episode smoke run ran.
- [ ] 100-episode seed-0 run ran if smoke succeeded.
- [ ] Five-seed protocol ran if seed-0 run succeeded.
- [ ] Exact model remained `typesafe/jev-1.13`.
- [ ] No hidden event/reference reward/termination reason was supplied to JEV.
- [ ] Raw results were preserved.
- [ ] Human-readable report was created.
- [ ] Reproducibility manifest was created.
- [ ] No secrets were committed.
- [ ] `git diff --check` passed.
- [ ] Results were committed.
- [ ] Branch was pushed.
- [ ] Pull request was opened.

If a step was intentionally skipped because an earlier prerequisite failed, mark it **SKIPPED — prerequisite failed**, not PASS.

---

## 17. Important research rule

This run is part of a reproducible benchmark.

The local agent is an **experimental operator**, not a result editor.

It may:

- run commands;
- inspect source and generated artifacts;
- validate schemas;
- create reports;
- create manifests;
- commit and push evidence.

It may not:

- invent measurements;
- manually improve scores;
- change the model to obtain a better result without recording a protocol deviation;
- expose hidden labels to JEV;
- delete failed runs;
- silently retry with different parameters and report only the successful attempt.

Failed and negative results are valuable benchmark evidence.
