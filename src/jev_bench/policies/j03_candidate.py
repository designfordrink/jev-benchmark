from __future__ import annotations
from typing import Any
import numpy as np
from jev_bench.core.contracts import Decision

class CandidateTinyMLP:
    """Tiny per-candidate scorer; the planner supplies candidates, the model only ranks them."""
    name = "j03_tiny_mlp"
    def __init__(self, input_dim: int = 12, hidden_units: int = 8, seed: int = 0, abstain_threshold: float = 0.0):
        if input_dim < 1 or hidden_units < 1: raise ValueError("model dimensions must be positive")
        if not 0.0 <= abstain_threshold <= 1.0: raise ValueError("abstain_threshold must be in [0, 1]")
        self.input_dim = input_dim; self.hidden_units = hidden_units; self.seed = seed; self.abstain_threshold = abstain_threshold
        self.rng = np.random.default_rng(seed)
        self.w1 = (self.rng.standard_normal((input_dim, hidden_units)) * np.sqrt(2.0 / input_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_units, dtype=np.float32)
        self.w2 = (self.rng.standard_normal((hidden_units, 1)) * np.sqrt(2.0 / hidden_units)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
        self.mean_ = np.zeros(input_dim, dtype=np.float32); self.std_ = np.ones(input_dim, dtype=np.float32)
        self.target_mean_ = 0.0; self.target_std_ = 1.0

    @property
    def parameter_count(self) -> int: return int(self.w1.size + self.b1.size + self.w2.size + self.b2.size)
    @property
    def model_size_bytes_fp32(self) -> int: return self.parameter_count * 4
    def reset(self, seed: int | None = None) -> None: pass

    def _normalize(self, X: np.ndarray) -> np.ndarray: return ((X - self.mean_) / self.std_).astype(np.float32)
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32); hidden = np.tanh(self._normalize(X) @ self.w1 + self.b1); return (hidden @ self.w2 + self.b2).reshape(-1)
    def rank(self, context: np.ndarray, candidates: list[Any] | tuple[Any, ...]) -> tuple[int, float, np.ndarray]:
        X = np.stack([np.concatenate([np.asarray(context, dtype=np.float32), np.asarray(c.features, dtype=np.float32)]) for c in candidates])
        raw = self.predict_scores(X); shifted = raw - raw.max(); probs = np.exp(shifted); probs /= probs.sum()
        index = int(np.argmax(raw)); return index, float(probs[index]), probs
    def act(self, observation: Any) -> Decision:
        index, confidence, _ = self.rank(observation["context"], observation["candidates"])
        return Decision(action=int(observation["candidates"][index].candidate_id), confidence=confidence, abstain=confidence < self.abstain_threshold, metadata={"candidate_index": index})

    def fit(self, X: np.ndarray, y: np.ndarray, *, epochs: int = 20, lr: float = 0.01, batch_size: int = 128, weight_decay: float = 0.0) -> dict[str, float]:
        if epochs < 1 or lr <= 0 or batch_size < 1: raise ValueError("invalid training parameters")
        X=np.asarray(X,dtype=np.float32); y=np.asarray(y,dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.input_dim or y.shape != (len(X),): raise ValueError("X/y shapes do not match model")
        self.mean_=X.mean(axis=0).astype(np.float32); self.std_=X.std(axis=0).astype(np.float32); self.std_[self.std_<1e-6]=1.0
        self.target_mean_=float(y.mean()); self.target_std_=float(y.std()) or 1.0; yn=((y-self.target_mean_)/self.target_std_).astype(np.float32); Xn=self._normalize(X)
        history={"train_mse":0.0}
        for _ in range(epochs):
            order=self.rng.permutation(len(X)); total=0.0; seen=0
            for start in range(0,len(X),batch_size):
                idx=order[start:start+batch_size]; xb=Xn[idx]; yb=yn[idx]
                h=np.tanh(xb@self.w1+self.b1); pred=(h@self.w2+self.b2).reshape(-1); err=pred-yb; total+=float((err*err).mean()); seen+=1
                gp=(2.0/len(yb))*err[:,None]; gw2=h.T@gp+weight_decay*self.w2; gb2=gp.sum(axis=0); gh=(gp@self.w2.T)*(1-h*h); gw1=xb.T@gh+weight_decay*self.w1; gb1=gh.sum(axis=0)
                self.w1-=lr*gw1.astype(np.float32); self.b1-=lr*gb1.astype(np.float32); self.w2-=lr*gw2.astype(np.float32); self.b2-=lr*gb2.astype(np.float32)
            history={"train_mse":total/max(1,seen)}
        return history

class CandidateLinearRegressor:
    name="j03_linear"
    def __init__(self, input_dim: int = 12, abstain_threshold: float = 0.0):
        self.input_dim=input_dim; self.abstain_threshold=abstain_threshold; self.w=np.zeros(input_dim,dtype=np.float32); self.b=0.0
    @property
    def parameter_count(self)->int:return self.input_dim+1
    @property
    def model_size_bytes_fp32(self)->int:return self.parameter_count*4
    def reset(self,seed:int|None=None)->None:pass
    def fit(self,X:np.ndarray,y:np.ndarray)->dict[str,float]:
        X=np.asarray(X,dtype=np.float64); y=np.asarray(y,dtype=np.float64); design=np.column_stack([X,np.ones(len(X))]); coef=np.linalg.lstsq(design,y,rcond=None)[0]; self.w=coef[:-1].astype(np.float32); self.b=float(coef[-1]); pred=design@coef; return {"train_mse":float(np.mean((pred-y)**2))}
    def predict_scores(self,X:np.ndarray)->np.ndarray:return (np.asarray(X,dtype=np.float32)@self.w+self.b).reshape(-1)
    def rank(self,context:np.ndarray,candidates:list[Any]|tuple[Any,...])->tuple[int,float,np.ndarray]:
        X=np.stack([np.concatenate([np.asarray(context,dtype=np.float32),np.asarray(c.features,dtype=np.float32)]) for c in candidates]); raw=self.predict_scores(X); shifted=raw-raw.max(); probs=np.exp(shifted); probs/=probs.sum(); index=int(np.argmax(raw)); return index,float(probs[index]),probs
    def act(self,observation:Any)->Decision:
        index,confidence,_=self.rank(observation["context"],observation["candidates"]); return Decision(action=int(observation["candidates"][index].candidate_id),confidence=confidence,abstain=confidence<self.abstain_threshold,metadata={"candidate_index":index})
