from __future__ import annotations
from typing import Any
from jev_bench.core.contracts import Decision
class FallbackPolicy:
    def __init__(self, primary: Any, fallback: Any, threshold: float = 0.5):
        if not 0.0 <= threshold <= 1.0: raise ValueError("threshold must be in [0, 1]")
        self.primary,self.fallback,self.threshold=primary,fallback,threshold; self.name=f"{primary.name}_fallback_{fallback.name}"; self.fallback_count=0; self.decision_count=0
    def reset(self, seed: int|None=None)->None: self.primary.reset(seed); self.fallback.reset(seed)
    def start_evaluation(self)->None: self.fallback_count=0; self.decision_count=0
    def act(self, observation: Any)->Decision:
        self.decision_count+=1; d=self.primary.act(observation)
        if d.abstain or d.confidence<self.threshold:
            self.fallback_count+=1; f=self.fallback.act(observation)
            return Decision(action=f.action,confidence=f.confidence,metadata={"fallback":True,"primary_confidence":d.confidence})
        return Decision(action=d.action,confidence=d.confidence,metadata={"fallback":False})
    @property
    def fallback_rate(self)->float:return self.fallback_count/max(1,self.decision_count)
