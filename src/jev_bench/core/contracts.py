from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

@dataclass(frozen=True)
class Decision:
    action: Any
    confidence: float = 1.0
    abstain: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

class Policy(Protocol):
    name: str
    def reset(self, seed: int | None = None) -> None: ...
    def act(self, observation: Any) -> Decision: ...

class Task(Protocol):
    name: str
    def make_environment(self, seed: int | None = None) -> Any: ...
    def evaluate(self, policy: Policy, *, episodes: int, seed: int) -> dict[str, Any]: ...
