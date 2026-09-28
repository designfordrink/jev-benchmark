from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass
class BenchmarkResult:
    task: str
    policy: str
    episodes: int
    seed: int
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
