from __future__ import annotations
from typing import Any
import numpy as np

class J04HeuristicSelector:
    name="j04_heuristic"
    def __init__(self,score_fn): self.score_fn=score_fn
    @property
    def parameter_count(self)->int:return 0
    @property
    def model_size_bytes_fp32(self)->int:return 0
    def reset(self,seed:int|None=None)->None: pass
    def rank(self,context:np.ndarray,candidates:list[Any]|tuple[Any,...]):
        scores=np.asarray([self.score_fn(c) for c in candidates],dtype=np.float32)
        ids=np.asarray([int(c.candidate_id) for c in candidates],dtype=np.int64); idx=int(np.lexsort((ids,-scores))[0]); shifted=scores-scores.max(); p=np.exp(shifted); p/=p.sum()
        return idx,float(p[idx]),p

class J04RandomSelector:
    name="j04_random"
    parameter_count=0
    model_size_bytes_fp32=0
    def __init__(self,seed:int=0): self.rng=np.random.default_rng(seed)
    def reset(self,seed:int|None=None): pass
    def rank(self,context,candidates):
        idx=int(self.rng.integers(len(candidates))); p=np.ones(len(candidates),dtype=np.float32)/len(candidates)
        return idx,float(p[idx]),p
