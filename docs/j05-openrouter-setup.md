# J05 — OpenRouter setup

## Local benchmark credentials

The J05 live adapter reads `OPENROUTER_API_KEY` from the process environment. A real key is never committed.

On Windows PowerShell:

    .\\scripts\\setup-openrouter.ps1

This creates a gitignored `.env` and reads the key without echoing it. Then:

    .\\scripts\\run-j05-openrouter.ps1

The runner loads `.env` into its own process and executes the five-seed J05 protocol.

Manual one-session alternative:

    $env:OPENROUTER_API_KEY = Read-Host 'OpenRouter API key'
    jev-bench j05-judge --provider jev --model <OPENROUTER_MODEL> --cache .cache/j05-jev.json

## Claude Code / Claude Desktop pattern

The separate `designfordrink/jev-plugin` keeps credentials outside the plugin in `~/.claude/jev.env` and resolves `OPENROUTER_API_KEY` from the environment or that file. This is the pattern to reuse for Claude Code: credentials belong to the user environment, not to the repository.

If Claude Code itself is routed through OpenRouter, OpenRouter documents `OPENROUTER_API_KEY`, `ANTHROPIC_BASE_URL=https://openrouter.ai/api`, `ANTHROPIC_AUTH_TOKEN=$OPENROUTER_API_KEY`, and an explicitly empty `ANTHROPIC_API_KEY`; native Claude Code does not automatically load a project `.env`. citeturn1search0

## Important J05 distinction

The current J05 `JEVOpenRouterReward` adapter calls OpenRouter's ordinary `/api/v1/chat/completions` endpoint and therefore expects a normal OpenRouter chat model slug. The `designfordrink/jev-plugin` uses Jev's System One endpoint (`/v1/systemone`) with structured questions. Therefore `jev-latest` in `.env.example` is a placeholder/default only; before claiming a JEV-specific benchmark result, the next implementation step is to add a dedicated System One adapter for J05 and compare it with the generic chat-judge adapter.

This distinction is deliberate: otherwise the benchmark would test a chat model pretending to be a Jev judge rather than the Jev decision API used by the plugin.

## Key safety

OpenRouter says newly created plaintext keys are shown only once; use a separate key with an appropriate spending limit for experiments and never put the key in Git, issues, PRs or chat. citeturn1search3
