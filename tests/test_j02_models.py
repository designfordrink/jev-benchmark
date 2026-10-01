from __future__ import annotations
import numpy as np
from jev_bench.policies.harth_tiny_mlp import HarthTinyMLP

def test_harth_tiny_mlp_train_predict_and_size():
    rng=np.random.default_rng(0); X=rng.normal(size=(40,768)).astype(np.float32); y=np.repeat(np.arange(4),10)
    model=HarthTinyMLP(input_dim=768,hidden_units=2,n_classes=4,seed=0)
    history=model.fit(X,y,epochs=2,lr=0.01,batch_size=8)
    p=model.predict_proba(X[:5])
    assert p.shape==(5,4)
    assert np.allclose(p.sum(axis=1),1.0,atol=1e-5)
    assert model.parameter_count==2*768+2+2*4+4
    assert model.model_size_bytes_fp32==model.parameter_count*4
    assert set(history)=={"train_loss","train_accuracy"}

def test_harth_tiny_mlp_save_load(tmp_path):
    model=HarthTinyMLP(input_dim=6,hidden_units=2,n_classes=2,seed=3)
    X=np.ones((4,6),dtype=np.float32); y=np.array([0,1,0,1])
    model.fit(X,y,epochs=1,lr=0.01)
    path=tmp_path/"model.npz"; model.save(path); loaded=HarthTinyMLP.load(path)
    assert np.allclose(model.predict_proba(X),loaded.predict_proba(X))
