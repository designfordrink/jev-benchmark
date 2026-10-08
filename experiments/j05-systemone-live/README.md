# J05 Live JEV System One

This directory contains artifacts produced by the **local AI agent** when executing the real J05 System One experiment.

## Directory contract

- `results/raw/` — machine-readable result artifacts copied from the benchmark runner.
- `results/reports/` — human-readable run/failure reports.
- `results/logs/` — sanitized execution logs when useful.
- `manifests/` — reproducibility metadata for each run.

Secrets, API keys, `.env` files, virtual environments and private credentials must never be stored here.

See [Local AI Agent Runbook](../../docs/local-j05-systemone-agent.md) for the execution protocol.
