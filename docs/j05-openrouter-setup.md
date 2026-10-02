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
    jev-bench j05-judge --provider jev_systemone --model typesafe/jev-1.13 --cache .cache/j05-jev.json

## Claude Code / Claude Desktop pattern

The separate `designfordrink/jev-plugin` keeps credentials outside the plugin in `~/.claude/jev.env` and resolves `OPENROUTER_API_KEY` from the environment or that file. This is the pattern to reuse for Claude Code: credentials belong to the user environment, not to the repository.

If Claude Code itself is routed through OpenRouter, OpenRouter documents `OPENROUTER_API_KEY`, `ANTHROPIC_BASE_URL=https://openrouter.ai/api`, `ANTHROPIC_AUTH_TOKEN=$OPENROUTER_API_KEY`, and an explicitly empty `ANTHROPIC_API_KEY`; native Claude Code does not automatically load a project `.env`. citeturn1search0

## J05 System One adapter

J05 now has a dedicated `jev_systemone` provider. It sends only the observable transition state plus a typed `choice` question whose seven criteria are `lava`, `timeout`, `wall`, `boundary`, `move`, `key`, and `exit`. The hidden environment event is never sent to Jev.

The adapter uses OpenRouter's Decisions API at `https://openrouter.ai/api/alpha/decisions` by default. OpenRouter documents this endpoint for Jev and returns a typed answer with the selected choice, probabilities for all choices, and confidence. citeturn1search0turn1search2

The older `jev` provider remains as a separate generic chat-judge baseline using `/api/v1/chat/completions`. This is intentional: it lets the benchmark distinguish a generative LLM judge from the actual System One decision interface.

The cache key is based only on observable transition fields, model and prompt version; the hidden event is excluded. The deterministic corpus split is likewise keyed on observable transition identity plus termination status.

## Key safety

OpenRouter says newly created plaintext keys are shown only once; use a separate key with an appropriate spending limit for experiments and never put the key in Git, issues, PRs or chat. citeturn1search3
