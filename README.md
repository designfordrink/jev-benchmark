# JEV Benchmark

A reproducible benchmark for evaluating tiny decision models across perception, decision, control, and hierarchical decision tasks.

## Status

**v0.1 — bootstrap**

The first implemented task is **J01 CartPole**.

## Development

```bash
pip install -e '.[dev]'
pytest
jev-bench list-tasks
```

## J01

```bash
jev-bench run --task j01_cartpole --policy rule --episodes 20 --seed 0
```

See `docs/jev-benchmark-runner-v0.1.md` for the implementation roadmap.
