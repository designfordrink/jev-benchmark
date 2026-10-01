from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from jev_bench.core.contracts import Decision

class HarthTinyMLP:
    """Trainable NumPy MLP for flattened HARTH windows."""
    name = "harth_tiny_mlp"
    def __init__(self, input_dim:int=768, hidden_units:int=8, n_classes:int=12, seed:int=0, abstain_threshold:float=0.0):
        if min(input_dim,hidden_units,n_classes)<1: raise ValueError("model dimensions must be positive")
        if not 0.0<=abstain_threshold<=1.0: raise ValueError("abstain_threshold must be in [0, 1]")
        self.input_dim,self.hidden_units,self.n_classes=input_dim,hidden_units,n_classes
        self.seed=seed; self.abstain_threshold=abstain_threshold; self.rng=np.random.default_rng(seed)
        scale1=np.sqrt(2.0/input_dim); scale2=np.sqrt(2.0/hidden_units)
        self.w1=(self.rng.standard_normal((input_dim,hidden_units))*scale1).astype(np.float32)
        self.b1=np.zeros(hidden_units,dtype=np.float32)
        self.w2=(self.rng.standard_normal((hidden_units,n_classes))*scale2).astype(np.float32)
        self.b2=np.zeros(n_classes,dtype=np.float32)
        self.mean_=np.zeros(input_dim,dtype=np.float32); self.std_=np.ones(input_dim,dtype=np.float32)
    @property
    def parameter_count(self)->int: return int(self.w1.size+self.b1.size+self.w2.size+self.b2.size)
    @property
    def model_size_bytes_fp32(self)->int: return self.parameter_count*4
    def reset(self,seed:int|None=None)->None: pass
    def _normalize(self,X:np.ndarray)->np.ndarray: return ((X-self.mean_)/self.std_).astype(np.float32)
    def _proba_matrix(self,X:np.ndarray)->np.ndarray:
        z1=np.tanh(X@self.w1+self.b1); logits=z1@self.w2+self.b2; logits=logits-logits.max(axis=1,keepdims=True)
        exp=np.exp(logits); return exp/exp.sum(axis=1,keepdims=True)
    def predict_proba(self,X:np.ndarray)->np.ndarray: return self._proba_matrix(self._normalize(np.asarray(X,dtype=np.float32)))
    def predict(self,X:np.ndarray)->np.ndarray: return np.argmax(self.predict_proba(X),axis=1).astype(np.int64)
    def act(self,observation:Any)->Decision:
        x=np.asarray(observation,dtype=np.float32).reshape(1,-1); p=self.predict_proba(x)[0]; conf=float(p.max()); return Decision(action=int(p.argmax()),confidence=conf,abstain=conf<self.abstain_threshold,metadata={"hidden_units":self.hidden_units})
    def fit(self,X:np.ndarray,y:np.ndarray,*,epochs:int=10,lr:float=0.01,batch_size:int=128,weight_decay:float=0.0)->dict[str,float]:
        if epochs<1 or lr<=0 or batch_size<1: raise ValueError("epochs, lr and batch_size are invalid")
        X=np.asarray(X,dtype=np.float32); y=np.asarray(y,dtype=np.int64)
        if X.ndim!=2 or X.shape[1]!=self.input_dim or y.shape!=(X.shape[0],): raise ValueError("X/y shapes do not match model")
        self.mean_=X.mean(axis=0).astype(np.float32); self.std_=X.std(axis=0).astype(np.float32); self.std_[self.std_<1e-6]=1.0
        Xn=self._normalize(X)
        counts=np.bincount(y,minlength=self.n_classes).astype(np.float32); weights=np.zeros(self.n_classes,dtype=np.float32); present=counts>0; weights[present]=len(y)/(self.n_classes*counts[present])
        history={"train_loss":0.0,"train_accuracy":0.0}
        for _ in range(epochs):
            order=self.rng.permutation(len(y)); total_loss=0.0; correct=0
            for start in range(0,len(y),batch_size):
                idx=order[start:start+batch_size]; xb=Xn[idx]; yb=y[idx]
                h=np.tanh(xb@self.w1+self.b1); logits=h@self.w2+self.b2; logits-=logits.max(axis=1,keepdims=True)
                exp=np.exp(logits); probs=exp/exp.sum(axis=1,keepdims=True); row=np.arange(len(yb)); sample_w=weights[yb]
                total_loss+=float((-np.log(probs[row,yb]+1e-12)*sample_w).mean()); correct+=int((probs.argmax(1)==yb).sum())
                grad=probs; grad[row,yb]-=1.0; grad*=sample_w[:,None]/len(yb)
                gw2=h.T@grad+weight_decay*self.w2; gb2=grad.sum(axis=0); gh=(grad@self.w2.T)*(1-h*h); gw1=xb.T@gh+weight_decay*self.w1; gb1=gh.sum(axis=0)
                self.w1-=lr*gw1.astype(np.float32); self.b1-=lr*gb1.astype(np.float32); self.w2-=lr*gw2.astype(np.float32); self.b2-=lr*gb2.astype(np.float32)
            history={"train_loss":total_loss/max(1,(len(y)+batch_size-1)//batch_size),"train_accuracy":correct/max(1,len(y))}
        return history
    def save(self,path:str|Path)->None:
        np.savez_compressed(path,w1=self.w1,b1=self.b1,w2=self.w2,b2=self.b2,mean=self.mean_,std=self.std_,input_dim=self.input_dim,hidden_units=self.hidden_units,n_classes=self.n_classes,seed=self.seed)
    @classmethod
    def load(cls,path:str|Path)->"HarthTinyMLP":
        d=np.load(path,allow_pickle=False); obj=cls(int(d["input_dim"]),int(d["hidden_units"]),int(d["n_classes"]),int(d["seed"])); obj.w1=d["w1"].astype(np.float32); obj.b1=d["b1"].astype(np.float32); obj.w2=d["w2"].astype(np.float32); obj.b2=d["b2"].astype(np.float32); obj.mean_=d["mean"].astype(np.float32); obj.std_=d["std"].astype(np.float32); return obj
