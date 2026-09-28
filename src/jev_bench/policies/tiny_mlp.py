from __future__ import annotations
from typing import Any
import numpy as np
from jev_bench.core.contracts import Decision

class TinyMLPPolicy:
    name = "tiny_mlp"
    def __init__(self, hidden_units: int = 8, abstain_threshold: float = 0.0):
        if hidden_units < 1: raise ValueError("hidden_units must be positive")
        self.hidden_units = hidden_units
        self.abstain_threshold = abstain_threshold
        self._w1 = self._make_weights(4, hidden_units)
        self._b1 = np.zeros(hidden_units, dtype=np.float32)
        self._w2 = self._make_weights(hidden_units, 2)
        self._b2 = np.zeros(2, dtype=np.float32)
    @staticmethod
    def _make_weights(rows: int, cols: int) -> np.ndarray:
        values = np.arange(1, rows * cols + 1, dtype=np.float32)
        return (((values % 7) - 3) / 20.0).reshape(rows, cols)
    def reset(self, seed: int | None = None) -> None: pass
    def act(self, observation: Any) -> Decision:
        x = np.asarray(observation, dtype=np.float32)
        hidden = np.tanh(x @ self._w1 + self._b1)
        logits = hidden @ self._w2 + self._b2
        shifted = logits - np.max(logits)
        probs = np.exp(shifted) / np.exp(shifted).sum()
        confidence = float(np.max(probs))
        return Decision(action=int(np.argmax(probs)), confidence=confidence, abstain=confidence < self.abstain_threshold, metadata={"hidden_units": self.hidden_units})
    @property
    def parameter_count(self) -> int: return 4*self.hidden_units + self.hidden_units + self.hidden_units*2 + 2
    @property
    def model_size_bytes_fp32(self) -> int: return self.parameter_count * 4
