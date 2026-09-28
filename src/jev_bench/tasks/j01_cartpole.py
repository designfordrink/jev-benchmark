from __future__ import annotations

import time
from statistics import mean
from typing import Any
import gymnasium as gym
from jev_bench.core.contracts import Policy


class J01CartPole:
    name = "j01_cartpole"

    def make_environment(self, seed: int | None = None):
        env = gym.make("CartPole-v1")
        if seed is not None:
            env.reset(seed=seed)
        return env

    def evaluate(self, policy: Policy, *, episodes: int = 20, seed: int = 0) -> dict[str, Any]:
        returns: list[float] = []
        lengths: list[int] = []
        latencies_us: list[float] = []
        for episode in range(episodes):
            episode_seed = seed + episode
            env = self.make_environment(episode_seed)
            observation, _ = env.reset(seed=episode_seed)
            policy.reset(episode_seed)
            total = 0.0
            steps = 0
            while True:
                started = time.perf_counter_ns()
                decision = policy.act(observation)
                latencies_us.append((time.perf_counter_ns() - started) / 1_000)
                observation, reward, terminated, truncated, _ = env.step(int(decision.action))
                total += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            env.close()
            returns.append(total)
            lengths.append(steps)
        return {
            "task": self.name,
            "policy": policy.name,
            "episodes": episodes,
            "seed": seed,
            "mean_return": mean(returns),
            "std_return": _std(returns),
            "mean_episode_length": mean(lengths),
            "mean_action_latency_us": mean(latencies_us),
            "p95_action_latency_us": _percentile(latencies_us, 95),
        }


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / (len(values) - 1)) ** 0.5


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
