from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    features: np.ndarray
    legal: bool = True

@dataclass(frozen=True)
class CandidateProblem:
    problem_id: int
    context: np.ndarray
    candidates: tuple[Candidate, ...]
    metadata: dict[str, Any]

    def legal_candidates(self) -> tuple[Candidate, ...]:
        legal = tuple(c for c in self.candidates if c.legal)
        if not legal:
            raise ValueError("candidate problem contains no legal candidates")
        return legal
