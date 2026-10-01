from __future__ import annotations
import argparse
import json
from jev_bench.policies.registry import get_policy, list_policies
from jev_bench.tasks.registry import get_task, list_tasks
from jev_bench.sweep import run_j01_size_sweep

def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(prog="jev-bench"); sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("list-tasks"); sub.add_parser("list-policies")
    run=sub.add_parser("run"); run.add_argument("--task",required=True); run.add_argument("--policy",required=True); run.add_argument("--episodes",type=int,default=20); run.add_argument("--seed",type=int,default=0)
    sweep=sub.add_parser("sweep"); sweep.add_argument("--task",default="j01_cartpole"); sweep.add_argument("--hidden-units",default="1,2,4,8,16,32,64"); sweep.add_argument("--episodes",type=int,default=20); sweep.add_argument("--seed",type=int,default=0); sweep.add_argument("--output")
    manifest=sub.add_parser("harth-manifest"); manifest.add_argument("--dataset-root",required=True); manifest.add_argument("--window-size",type=int,default=128); manifest.add_argument("--stride",type=int,default=128)
    inspect=sub.add_parser("harth-inspect"); inspect.add_argument("--dataset-root",required=True); inspect.add_argument("--subject",required=True); inspect.add_argument("--window-size",type=int,default=128); inspect.add_argument("--stride",type=int,default=128); inspect.add_argument("--max-windows",type=int,default=1000)
    train=sub.add_parser("harth-train"); train.add_argument("--dataset-root",required=True); train.add_argument("--test-subject",required=True); train.add_argument("--model",choices=["tiny_mlp","nearest_centroid"],default="tiny_mlp"); train.add_argument("--hidden-units",type=int,default=8); train.add_argument("--window-size",type=int,default=128); train.add_argument("--stride",type=int,default=128); train.add_argument("--max-train-windows-per-subject",type=int,default=500); train.add_argument("--max-test-windows",type=int,default=2000); train.add_argument("--epochs",type=int,default=10); train.add_argument("--lr",type=float,default=0.01); train.add_argument("--batch-size",type=int,default=128); train.add_argument("--seed",type=int,default=0); train.add_argument("--abstain-threshold",type=float,default=0.0); train.add_argument("--model-output"); train.add_argument("--output")
    return parser

def main()->int:
    args=build_parser().parse_args()
    if args.command=="list-tasks": print("\\n".join(list_tasks())); return 0
    if args.command=="list-policies": print("\\n".join(list_policies())); return 0
    if args.command=="run":
        result=get_task(args.task).evaluate(get_policy(args.policy),episodes=args.episodes,seed=args.seed); print(json.dumps(result,indent=2)); return 0
    if args.command=="sweep":
        if args.task!="j01_cartpole": raise SystemExit("Only j01_cartpole is supported by the current sweep runner")
        units=[int(x.strip()) for x in args.hidden_units.split(",") if x.strip()]
        if not units or any(u<1 for u in units): raise SystemExit("--hidden-units must contain positive integers")
        print(json.dumps(run_j01_size_sweep(units,episodes=args.episodes,seed=args.seed,output=args.output),indent=2)); return 0
    if args.command=="harth-manifest":
        print(json.dumps(get_task("j02_harth").evaluate_manifest(args.dataset_root,window_size=args.window_size,stride=args.stride),indent=2)); return 0
    if args.command=="harth-inspect":
        print(json.dumps(get_task("j02_harth").inspect_windows(args.dataset_root,test_subject=args.subject,window_size=args.window_size,stride=args.stride,max_windows=args.max_windows),indent=2)); return 0
    if args.command=="harth-train":
        result=get_task("j02_harth").train_and_evaluate(args.dataset_root,test_subject=args.test_subject,model=args.model,hidden_units=args.hidden_units,window_size=args.window_size,stride=args.stride,max_train_windows_per_subject=args.max_train_windows_per_subject,max_test_windows=args.max_test_windows,epochs=args.epochs,lr=args.lr,batch_size=args.batch_size,seed=args.seed,abstain_threshold=args.abstain_threshold,model_output=args.model_output)
        payload=json.dumps(result,indent=2)
        if args.output:
            from pathlib import Path
            p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(payload,encoding="utf-8")
        print(payload); return 0
    return 1
