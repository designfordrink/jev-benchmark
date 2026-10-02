from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from jev_bench.envs.key_quest import KeyQuestEnv, Transition


REWARD_LEVELS = {
    "lava": -1.0,
    "timeout": -0.25,
    "wall": -0.05,
    "boundary": -0.05,
    "move": 0.0,
    "key": 0.5,
    "exit": 1.0,
}


@dataclass(frozen=True)
class RewardJudgment:
    reward: float
    confidence: float
    abstain: bool = False
    source: str = "unknown"
    probabilities: dict[str, float] | None = None
    latency_ms: float = 0.0
    cached: bool = False


class RewardProvider(Protocol):
    name: str
    def judge(self, transition: Transition) -> RewardJudgment: ...


class NativeReward:
    name = "native"

    def judge(self, transition: Transition) -> RewardJudgment:
        return RewardJudgment(REWARD_LEVELS[transition.event], 1.0, source=self.name)


class RuleReward(NativeReward):
    name = "rules"


class ConfidenceFallback:
    """Use a primary judge, but fall back to rules below a confidence threshold."""
    name = "confidence_fallback"

    def __init__(self, primary: RewardProvider, fallback: RewardProvider | None = None, threshold: float = 0.60):
        self.primary = primary
        self.fallback = fallback or RuleReward()
        self.threshold = threshold
        self.fallback_count = 0

    def judge(self, transition: Transition) -> RewardJudgment:
        primary = self.primary.judge(transition)
        if primary.abstain or primary.confidence < self.threshold:
            self.fallback_count += 1
            fallback = self.fallback.judge(transition)
            return RewardJudgment(
                reward=fallback.reward,
                confidence=primary.confidence,
                abstain=False,
                source=f"{self.name}:{fallback.source}",
                probabilities=primary.probabilities,
                latency_ms=primary.latency_ms + fallback.latency_ms,
                cached=primary.cached,
            )
        return primary


class JsonFileCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    @staticmethod
    def key(transition: Transition, model: str, prompt_version: str) -> str:
        raw = json.dumps({
            "state": transition.state,
            "action": transition.action,
            "next_state": transition.next_state,
            "event": transition.event,
            "model": model,
            "prompt_version": prompt_version,
        }, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        return self.data.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        self.data[key] = value
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, sort_keys=True, indent=2))
        tmp.replace(self.path)


class JEVOpenRouterReward:
    """Reference adapter: JEV judges the transition; RL still chooses actions."""

    name = "jev"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        cache: JsonFileCache | None = None,
        prompt_version: str = "j05-v1",
        timeout_s: float = 60.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.cache = cache
        self.prompt_version = prompt_version
        self.timeout_s = timeout_s
        self.calls = 0
        self.cache_hits = 0

    def _prompt(self, t: Transition) -> str:
        return (
            "You are a reward judge for a deterministic gridworld. "
            "Return JSON only with probabilities for exactly these events: "
            f"{', '.join(REWARD_LEVELS)}. Judge the outcome of the transition from "
            "the observed state, action, next_state and termination flag. "
            "Do not assume or receive a hidden event label. The observed transition is "
            f"state={t.state}, action={t.action}, next_state={t.next_state}, "
            "terminated=" + str(t.terminated).lower() +
            ". Probabilities must sum to 1."
        )

    def judge(self, transition: Transition) -> RewardJudgment:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for provider=jev")
        key = self.cache.key(transition, self.model, self.prompt_version) if self.cache else None
        if key and (hit := self.cache.get(key)) is not None:
            self.cache_hits += 1
            return RewardJudgment(**hit, cached=True)
        started = time.perf_counter()
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": self._prompt(transition)}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
            payload = json.loads(response.read())
        text = payload["choices"][0]["message"]["content"]
        probabilities = json.loads(text)
        probs = {k: float(probabilities.get(k, 0.0)) for k in REWARD_LEVELS}
        total = sum(max(0.0, v) for v in probs.values())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("JEV returned invalid probabilities")
        probs = {k: max(0.0, v) / total for k, v in probs.items()}
        reward = sum(probs[k] * REWARD_LEVELS[k] for k in REWARD_LEVELS)
        confidence = max(probs.values())
        abstain = confidence < 0.60
        judgment = RewardJudgment(
            reward=reward,
            confidence=confidence,
            abstain=abstain,
            source=self.name,
            probabilities=probs,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if key and self.cache:
            cached_payload = {k: v for k, v in judgment.__dict__.items() if k != "cached"}
            self.cache.put(key, cached_payload)
        self.calls += 1
        return judgment


def independent_label(transition: Transition) -> float:
    return REWARD_LEVELS[transition.event]


def split_transition_corpus(rows: list[Transition] | None = None, holdout_fraction: float = 0.25) -> tuple[list[Transition], list[Transition]]:
    """Deterministic disjoint split by hashed transition identity."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0,1)")
    rows = list(rows if rows is not None else build_transition_corpus())
    train, holdout = [], []
    for t in rows:
        raw = json.dumps({"state": t.state, "action": t.action, "next_state": t.next_state, "event": t.event}, sort_keys=True)
        bucket = int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (holdout if bucket < holdout_fraction else train).append(t)
    if not train or not holdout:
        raise RuntimeError("transition split produced an empty partition")
    return train, holdout


def adversarial_transitions() -> list[Transition]:
    """Cases where a superficial state-only judge can be reward-hacked."""
    return [
        Transition((2, 2, 0), 1, (3, 2, 0), "wall", False),
        Transition((4, 4, 0), 0, (4, 4, 0), "boundary", False),
        Transition((2, 2, 1), 1, (3, 2, 1), "wall", False),
        Transition((2, 3, 0), 0, (2, 4, 0), "move", False),
        Transition((2, 2, 1), 0, (2, 3, 1), "lava", True),
    ]


def representation_variants(transition: Transition) -> list[Transition]:
    """Semantically identical transition variants for representation robustness."""
    return [
        transition,
        Transition(transition.state, transition.action, transition.next_state, transition.event, transition.terminated),
    ]


def build_transition_corpus() -> list[Transition]:
    """Enumerate a fixed, reproducible corpus without training an RL agent."""
    env = KeyQuestEnv()
    rows: list[Transition] = []
    for x in range(env.width):
        for y in range(env.height):
            for has_key in (0, 1):
                state = (x, y, has_key)
                for action in range(4):
                    # Use a fresh environment and force the requested state.
                    probe = KeyQuestEnv()
                    probe.pos, probe.has_key = (x, y), bool(has_key)
                    rows.append(probe.step(action))
    return rows


class TabularQLearner:
    def __init__(self, seed: int, alpha: float = 0.2, gamma: float = 0.95, epsilon: float = 0.2):
        self.seed = seed
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.q: dict[tuple[int, int, int], list[float]] = {}
        self.rng_state = seed

    def _rng(self) -> float:
        self.rng_state = (1103515245 * self.rng_state + 12345) % (2**31)
        return self.rng_state / (2**31)

    def values(self, state):
        return self.q.setdefault(tuple(state), [0.0] * 4)

    def action(self, state) -> int:
        if self._rng() < self.epsilon:
            return int(self._rng() * 4) % 4
        values = self.values(state)
        return max(range(4), key=lambda a: (values[a], -a))

    def update(self, state, action, reward, next_state, done):
        target = reward if done else reward + self.gamma * max(self.values(next_state))
        values = self.values(state)
        values[action] += self.alpha * (target - values[action])


def run_j05(provider: RewardProvider, *, episodes: int = 100, seed: int = 0) -> dict[str, Any]:
    learner = TabularQLearner(seed)
    successes = 0
    returns = []
    start = time.perf_counter()
    for ep in range(episodes):
        env = KeyQuestEnv()
        state = env.reset()
        total = 0.0
        for _ in range(env.max_steps):
            action = learner.action(state)
            transition = env.step(action)
            judgment = provider.judge(transition)
            reward = 0.0 if judgment.abstain else judgment.reward
            learner.update(state, action, reward, transition.next_state, transition.terminated)
            state = transition.next_state
            total += reward
            if transition.terminated:
                if transition.event == "exit":
                    successes += 1
                break
        returns.append(total)
    return {
        "task": "j05_jev_rl",
        "provider": provider.name,
        "episodes": episodes,
        "seed": seed,
        "success_rate": successes / episodes,
        "mean_return": sum(returns) / len(returns),
        "wall_time_s": time.perf_counter() - start,
        "calls": int(getattr(provider, "calls", episodes)),
        "cache_hits": int(getattr(provider, "cache_hits", 0)),
    }


def evaluate_judge(provider: RewardProvider, *, holdout: list[Transition] | None = None) -> dict[str, Any]:
    rows = holdout if holdout is not None else split_transition_corpus()[1]
    errors, correct, confidences, abstentions = [], 0, [], 0
    for t in rows:
        j = provider.judge(t)
        truth = independent_label(t)
        errors.append(abs(j.reward - truth))
        correct += int(round(j.reward, 6) == truth)
        confidences.append(j.confidence)
        abstentions += int(j.abstain)
    return {
        "provider": provider.name,
        "split": "held_out",
        "examples": len(rows),
        "reward_mae": sum(errors) / len(errors),
        "exact_reward_rate": correct / len(rows),
        "mean_confidence": sum(confidences) / len(confidences),
        "abstention_rate": abstentions / len(rows),
    }
