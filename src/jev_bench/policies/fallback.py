from __future__ import annotations
from typing import Any
from jev_bench.core.contracts import Decision

class FallbackPolicy:
    def __init__(self, primary: Any, fallback: Any, threshold: float = 0.5):
        if not 0.0 <= threshold <= 1.0: raise ValueError("threshold must be in [0, 1]")
        self.primary, self.fallback, self.threshold = primary, fallback, threshold
        self.name = f"{primary.name}_fallback_{fallback.name}"
        self.fallback_count = 0
    def reset(self, seed: int | None = None) -> None:
        self.fallback_count = 0; self.primary.reset(seed); self.fallback.reset(seed)
    def act(self, observation: Any) -> Decision:
        d = self.primary.act(observation)
        if d.abstain or d.confidence < self.threshold:
            self.fallback_count += 1; f = self.fallback.act(observation)
            return Decision(action=f.action, confidence=f.confidence, metadata={"fallback": True, "primary_confidence": d.confidence})
        return Decision(action=d.action, confidence=d.confidence, metadata={"fallback": False})
