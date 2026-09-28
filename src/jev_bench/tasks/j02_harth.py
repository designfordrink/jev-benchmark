from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Any
from jev_bench.datasets.harth import HarthDataset,LABELS

class J02Harth:
    name="j02_harth"
    def evaluate_manifest(self,dataset_root:str|Path,*,window_size:int=128,stride:int=128)->dict[str,Any]:
        dataset=HarthDataset(dataset_root); subjects=[dataset.subject_id(p) for p in dataset.files()]
        return {"task":self.name,"dataset_root":str(dataset.root),"subjects":subjects,"subject_count":len(subjects),"window_size":window_size,"stride":stride,"sampling_hz":50,"input_shape":[window_size,6],"label_map":LABELS,"evaluation_protocol":"subject-disjoint; no subject may occur in both train and test"}
    def inspect_windows(self,dataset_root:str|Path,*,test_subject:str,window_size:int=128,stride:int=128,max_windows:int=1000)->dict[str,Any]:
        dataset=HarthDataset(dataset_root); path=dataset.root/f"{test_subject}.csv"
        if not path.exists(): raise FileNotFoundError(path)
        counts=Counter(); total=0
        for window in dataset.iter_windows(path,window_size=window_size,stride=stride):
            counts[window.label]+=1; total+=1
            if total>=max_windows: break
        return {"task":self.name,"subject":test_subject,"windows_scanned":total,"class_counts":dict(sorted(counts.items())),"window_size":window_size,"stride":stride}
