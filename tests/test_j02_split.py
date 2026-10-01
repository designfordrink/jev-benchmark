from __future__ import annotations
import csv
from pathlib import Path
from jev_bench.datasets.harth import load_subject_split

def write_subject(path: Path, label: int):
    fields=["timestamp","back_x","back_y","back_z","thigh_x","thigh_y","thigh_z","label"]
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i in range(128): w.writerow({"timestamp":i,"back_x":0,"back_y":0,"back_z":0,"thigh_x":1,"thigh_y":0,"thigh_z":0,"label":label})

def test_subject_split_excludes_held_out_subject(tmp_path):
    write_subject(tmp_path/"S001.csv",1); write_subject(tmp_path/"S002.csv",1); write_subject(tmp_path/"S003.csv",1)
    train,test,train_subjects=load_subject_split(tmp_path,test_subject="S003",window_size=128,stride=128,max_train_windows_per_subject=None,max_test_windows=None,seed=0)
    assert train_subjects==["S001","S002"]
    assert all(w.subject!="S003" for w in train)
    assert all(w.subject=="S003" for w in test)
