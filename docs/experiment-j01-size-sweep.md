# J01 Model-Size Sweep

## Purpose

Measure the trade-off between tiny-model size and downstream CartPole behavior. This is the first experiment designed around the benchmark's JEV-like reporting dimensions.

## Command

```bash
jev-bench sweep --task j01_cartpole --hidden-units 1,2,4,8,16,32,64 --episodes 20 --seed 0 --output results/j01-size-sweep.json
```

## Output

Each row contains the task/policy identity plus:

- `hidden_units`
- `parameter_count`
- `model_size_bytes_fp32`
- `mean_return`
- `std_return`
- `mean_episode_length`
- `mean_action_latency_us`
- `p95_action_latency_us`
- `mean_confidence`
- `abstention_rate`

## Interpretation

The current fixed-weight MLP is an inference-size probe, not a trained benchmark champion. The sweep therefore measures infrastructure and scaling behavior first. Training and learned-policy comparisons must be added before making claims about task performance versus model size.
