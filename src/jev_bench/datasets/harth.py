from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import numpy as np

FEATURE_COLUMNS=("back_x","back_y","back_z","thigh_x","thigh_y","thigh_z")
REQUIRED_COLUMNS=("timestamp",*FEATURE_COLUMNS,"label")
LABELS={1:"walking",2:"running",3:"shuffling",4:"stairs_up",5:"stairs_down",6:"standing",7:"sitting",8:"lying",13:"cycling_sit",14:"cycling_stand",130:"cycling_sit_inactive",140:"cycling_stand_inactive"}

@dataclass(frozen=True)
class HarthWindow:
    subject: str
    features: np.ndarray
    label: int

class HarthDataset:
    """Streaming reader for public HARTH v1/v2 subject CSV files."""
    def __init__(self, root: str|Path):
        self.root=Path(root)
        if not self.root.exists(): raise FileNotFoundError(f"HARTH dataset root does not exist: {self.root}")
    def files(self)->list[Path]:
        files=sorted(self.root.glob("S*.csv"))
        if not files: raise FileNotFoundError(f"No HARTH subject CSV files found in {self.root}")
        return files
    @staticmethod
    def subject_id(path:Path)->str: return path.stem
    def iter_rows(self,path:Path)->Iterator[tuple[np.ndarray,int]]:
        with path.open("r",newline="",encoding="utf-8") as handle:
            reader=csv.DictReader(handle)
            if reader.fieldnames is None or any(c not in reader.fieldnames for c in REQUIRED_COLUMNS):
                raise ValueError(f"Unexpected HARTH schema in {path}; expected {REQUIRED_COLUMNS}")
            for row in reader:
                try: label=int(row["label"]); values=np.asarray([float(row[c]) for c in FEATURE_COLUMNS],dtype=np.float32)
                except (TypeError,ValueError) as exc: raise ValueError(f"Invalid HARTH row in {path}") from exc
                yield values,label
    def iter_windows(self,path:Path,*,window_size:int=128,stride:int=128)->Iterator[HarthWindow]:
        if window_size<1 or stride<1: raise ValueError("window_size and stride must be positive")
        buffer=[]; labels=[]
        for values,label in self.iter_rows(path):
            buffer.append(values); labels.append(label)
            if len(buffer)<window_size: continue
            if len(set(labels[:window_size]))==1:
                yield HarthWindow(self.subject_id(path),np.stack(buffer[:window_size]),labels[0])
            del buffer[:stride]; del labels[:stride]
