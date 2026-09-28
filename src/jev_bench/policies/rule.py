from __future__ import annotations

from typing import Any
from jev_bench.core.contracts import Decision


class CartPoleRulePolicy:
    name = "rule"

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: Any) -> Decision:
        x, x_dot, theta, theta_dot = observation
        score = theta + 0.25 * theta_dot + 0.05 * x + 0.01 * x_dot
        return Decision(action=1 if score > 0 else 0, confidence=1.0)
