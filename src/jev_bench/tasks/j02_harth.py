from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any
import numpy as np
from jev_bench.datasets.harth import LABELS, load_subject_split
from jev_bench.policies.harth_baseline import HarthNearestCentroid
from jev_bench.policies.harth_tiny_mlp import HarthTinyMLP

class J02Harth:
    name = "j02_harth"

    def evaluate_manifest(self, dataset_root: str | Path, *, window_size: int = 128, stride: int = 128) -> dict[str, Any]:
        from jev_bench.datasets.harth import HarthDataset
        dataset = HarthDataset(dataset_root)
        subjects = [dataset.subject_id(p) for p in dataset.files()]
        return {"task": self.name, "dataset_root": str(dataset.root), "subjects": subjects, "subject_count": len(subjects), "window_size": window_size, "stride": stride, "sampling_hz": 50, "input_shape": [window_size, 6], "label_map": LABELS, "evaluation_protocol": "subject-disjoint; no subject may occur in both train and test"}

    def inspect_windows(self, dataset_root: str | Path, *, test_subject: str, window_size: int = 128, stride: int = 128, max_windows: int = 1000) -> dict[str, Any]:
        from collections import Counter
        from jev_bench.datasets.harth import HarthDataset
        dataset = HarthDataset(dataset_root); path = dataset.root / f"{test_subject}.csv"
        if not path.exists(): raise FileNotFoundError(path)
        counts=Counter(); total=0
        for window in dataset.iter_windows(path, window_size=window_size, stride=stride):
            counts[window.label]+=1; total+=1
            if total>=max_windows: break
        return {"task": self.name, "subject": test_subject, "windows_scanned": total, "class_counts": dict(sorted(counts.items())), "window_size": window_size, "stride": stride}

    def train_and_evaluate(self, dataset_root: str | Path, *, test_subject: str, model: str = "tiny_mlp", hidden_units: int = 8, window_size: int = 128, stride: int = 128, max_train_windows_per_subject: int | None = 500, max_test_windows: int | None = 2000, epochs: int = 10, lr: float = 0.01, batch_size: int = 128, seed: int = 0, abstain_threshold: float = 0.0, model_output: str | Path | None = None) -> dict[str, Any]:
        train_windows,test_windows,train_subjects=load_subject_split(dataset_root,test_subject=test_subject,window_size=window_size,stride=stride,max_train_windows_per_subject=max_train_windows_per_subject,max_test_windows=max_test_windows,seed=seed)
        X_train=np.stack([w.features.reshape(-1) for w in train_windows]).astype(np.float32); X_test=np.stack([w.features.reshape(-1) for w in test_windows]).astype(np.float32)
        raw_classes=sorted(set(w.label for w in train_windows))
        test_classes=sorted(set(w.label for w in test_windows))
        unseen=sorted(set(test_classes)-set(raw_classes))
        if unseen: raise ValueError(f"Test subject contains labels absent from training subjects: {unseen}")
        class_to_index={label:i for i,label in enumerate(raw_classes)}
        y_train=np.asarray([class_to_index[w.label] for w in train_windows],dtype=np.int64); y_test=np.asarray([class_to_index[w.label] for w in test_windows],dtype=np.int64)
        if model=="tiny_mlp": estimator=HarthTinyMLP(input_dim=X_train.shape[1],hidden_units=hidden_units,n_classes=len(raw_classes),seed=seed,abstain_threshold=abstain_threshold); fit_started=time.perf_counter(); history=estimator.fit(X_train,y_train,epochs=epochs,lr=lr,batch_size=batch_size); train_seconds=time.perf_counter()-fit_started
        elif model=="nearest_centroid": estimator=HarthNearestCentroid(n_classes=len(raw_classes)); fit_started=time.perf_counter(); estimator.fit(X_train,y_train); history={}; train_seconds=time.perf_counter()-fit_started
        else: raise ValueError(f"Unknown J02 model: {model}")
        started=time.perf_counter_ns(); probabilities=estimator.predict_proba(X_test); total_ns=time.perf_counter_ns()-started
        probe_count=min(len(X_test),256); single_started=time.perf_counter_ns()
        for row in X_test[:probe_count]: estimator.predict_proba(row.reshape(1,-1))
        single_total_ns=time.perf_counter_ns()-single_started
        predicted=probabilities.argmax(axis=1); confidence=probabilities.max(axis=1); abstained=confidence<abstain_threshold
        batch_latency_us=(total_ns/max(1,len(X_test)))/1000.0
        single_latency_us=(single_total_ns/max(1,probe_count))/1000.0
        result={"task":self.name,"model":model,"test_subject":test_subject,"train_subjects":train_subjects,"seed":seed,"window_size":window_size,"stride":stride,"train_windows":len(X_train),"test_windows":len(X_test),"classes":{str(k):LABELS.get(k,str(k)) for k in raw_classes},"accuracy":float((predicted==y_test).mean()),"macro_f1":_macro_f1(y_test,predicted,len(raw_classes)),"mean_confidence":float(confidence.mean()),"abstention_rate":float(abstained.mean()),"mean_batch_inference_latency_us":float(batch_latency_us),"mean_single_window_inference_latency_us":float(single_latency_us),"parameter_count":int(estimator.parameter_count),"model_size_bytes_fp32":int(estimator.model_size_bytes_fp32),"train_seconds":float(train_seconds),"history":history,"test_class_counts":{str(k):int((np.asarray([w.label for w in test_windows])==k).sum()) for k in test_classes},"risk_coverage":_risk_coverage(y_test,predicted,confidence)}
        if model_output is not None and hasattr(estimator,"save"):
            path=Path(model_output); path.parent.mkdir(parents=True,exist_ok=True); estimator.save(path); result["model_output"]=str(path); result["serialized_model_size_bytes"]=path.stat().st_size
        return result

def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    values=[]
    for c in range(n_classes):
        tp=int(((y_true==c)&(y_pred==c)).sum()); fp=int(((y_true!=c)&(y_pred==c)).sum()); fn=int(((y_true==c)&(y_pred!=c)).sum()); denom=2*tp+fp+fn
        values.append(0.0 if denom==0 else 2*tp/denom)
    return float(np.mean(values))
