from __future__ import annotations
import csv
from pathlib import Path
from jev_bench.datasets.harth import HarthDataset

def make_dataset(root: Path):
    path=root/"S001.csv"; fields=["timestamp","back_x","back_y","back_z","thigh_x","thigh_y","thigh_z","label"]
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i in range(256):
            w.writerow({"timestamp":f"00:00:{i:02d}","back_x":0,"back_y":0,"back_z":0,"thigh_x":1,"thigh_y":0,"thigh_z":0,"label":1 if i<128 else 6})
    return path

def test_harth_subject_windows_are_pure(tmp_path):
    path=make_dataset(tmp_path); windows=list(HarthDataset(tmp_path).iter_windows(path,window_size=128,stride=128))
    assert len(windows)==2
    assert all(w.features.shape==(128,6) for w in windows)
    assert [w.label for w in windows]==[1,6]

def test_harth_schema_is_validated(tmp_path):
    path=tmp_path/"S001.csv"; path.write_text("timestamp,back_x\n",encoding="utf-8")
    try: list(HarthDataset(tmp_path).iter_rows(path))
    except ValueError as exc: assert "Unexpected HARTH schema" in str(exc)
    else: raise AssertionError("schema validation did not fail")
