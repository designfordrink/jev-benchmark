from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import statistics

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="results/ns01")
    p.add_argument("--output",required=True)
    p.add_argument("--utility-ratio",type=float,default=0.90)
    p.add_argument("--size-kb",type=float,default=16.0)
    p.add_argument("--latency-ms",type=float,default=5.0)
    args=p.parse_args()
    refs=[]; sweeps=[]
    for path in glob.glob(str(Path(args.root)/"**/*.json"),recursive=True):
        data=json.loads(Path(path).read_text())
        if "sweep" in Path(path).name: sweeps.append(data)
        elif "reference" in Path(path).name: refs.append(data)
    if not refs or not sweeps: raise SystemExit("NS-01 results incomplete")
    ref_by_seed={int(x["evaluation"]["seed"]):x for x in refs}
    rows={}
    for s in sweeps:
        seed=int(s["metadata"]["seed"]) if "metadata" in s else int(s.get("evaluation",{}).get("seed",0))
        for run in s.get("runs",[]):
            h=int(run["model"].get("hidden_units",run["extra"].get("hidden_units",0)))
            rows.setdefault(h,[]).append(run)
    reference_mean=statistics.mean(float(x["metrics"]["mean_return"]) for x in refs)
    gate=reference_mean*args.utility_ratio
    table=[]
    for h in sorted(rows):
        rs=rows[h]
        def mean(k): return statistics.mean(float(r["metrics"][k]) for r in rs)
        def maxv(k): return max(float(r["metrics"][k]) for r in rs)
        size=maxv("model_size_bytes_fp32")
        p95=maxv("p95_action_latency_us")
        utility=mean("mean_return")
        table.append({"hidden_units":h,"mean_return":utility,"return_ratio":utility/reference_mean if reference_mean else 0,
                      "mean_lines":mean("mean_lines"),"model_size_bytes_fp32_max":size,
                      "p95_action_latency_us_max":p95,"legal_action_rate_min":min(float(r["metrics"]["legal_action_rate"]) for r in rs),
                      "permutation_invariance_min":min(float(r["metrics"]["permutation_invariance_rate"]) for r in rs),
                      "fallback_rate_mean":mean("fallback_rate"),
                      "seed_count":len(rs),
                      "passes_90pct_utility":utility>=gate})
    feasible=[r for r in table if r["passes_90pct_utility"] and r["model_size_bytes_fp32_max"]<=args.size_kb*1024 and r["p95_action_latency_us_max"]<=args.latency_ms*1000 and r["legal_action_rate_min"]>=1.0]
    out={"protocol":"ns01-operating-point-analysis/v1","reference_mean_return":reference_mean,
         "utility_gate":gate,"constraints":{"model_size_kb":args.size_kb,"p95_latency_ms":args.latency_ms,"legal_action_rate_min":1.0},
         "operating_points":table,"feasible_operating_points":feasible,
         "interpretation":"viable_tiny_operating_point" if feasible else "no_operating_point_under_declared_constraints"}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
