from __future__ import annotations
from typing import Any
import numpy as np
from jev_bench.core.contracts import Decision
class HarthNearestCentroid:
    name="harth_nearest_centroid"
    def __init__(self,n_classes:int=12): self.n_classes=n_classes; self.centroids=None
    def reset(self,seed:int|None=None)->None: pass
    @property
    def parameter_count(self)->int: return 0 if self.centroids is None else int(self.centroids.size)
    @property
    def model_size_bytes_fp32(self)->int: return self.parameter_count*4
    def fit(self,X:np.ndarray,y:np.ndarray)->None:
        X=np.asarray(X,dtype=np.float32); y=np.asarray(y,dtype=np.int64); self.centroids=np.zeros((self.n_classes,X.shape[1]),dtype=np.float32)
        for c in range(self.n_classes):
            rows=X[y==c]
            if len(rows): self.centroids[c]=rows.mean(axis=0)
    def predict_proba(self,X:np.ndarray)->np.ndarray:
        if self.centroids is None: raise RuntimeError("model is not fitted")
        x=np.asarray(X,dtype=np.float32); d=((x[:,None,:]-self.centroids[None,:,:])**2).mean(axis=2); logits=-d
        logits-=logits.max(axis=1,keepdims=True); p=np.exp(logits); return p/p.sum(axis=1,keepdims=True)
    def predict(self,X:np.ndarray)->np.ndarray: return self.predict_proba(X).argmax(1).astype(np.int64)
    def act(self,observation:Any)->Decision:
        p=self.predict_proba(np.asarray(observation,dtype=np.float32).reshape(1,-1))[0]; return Decision(action=int(p.argmax()),confidence=float(p.max()))
