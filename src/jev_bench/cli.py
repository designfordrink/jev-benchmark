from __future__ import annotations
import argparse
import json
from jev_bench.policies.registry import get_policy, list_policies
from jev_bench.tasks.registry import get_task, list_tasks
from jev_bench.sweep import run_j01_size_sweep, run_j02_size_sweep, run_j02_loso, run_j02_loso_size_sweep, run_j03_size_sweep

def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(prog="jev-bench"); sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("list-tasks"); sub.add_parser("list-policies")
    run=sub.add_parser("run"); run.add_argument("--task",required=True); run.add_argument("--policy",required=True); run.add_argument("--episodes",type=int,default=20); run.add_argument("--seed",type=int,default=0)
    sweep=sub.add_parser("sweep"); sweep.add_argument("--task",default="j01_cartpole"); sweep.add_argument("--hidden-units",default="1,2,4,8,16,32,64"); sweep.add_argument("--episodes",type=int,default=20); sweep.add_argument("--seed",type=int,default=0); sweep.add_argument("--output")
    manifest=sub.add_parser("harth-manifest"); manifest.add_argument("--dataset-root",required=True); manifest.add_argument("--window-size",type=int,default=128); manifest.add_argument("--stride",type=int,default=128)
    inspect=sub.add_parser("harth-inspect"); inspect.add_argument("--dataset-root",required=True); inspect.add_argument("--subject",required=True); inspect.add_argument("--window-size",type=int,default=128); inspect.add_argument("--stride",type=int,default=128); inspect.add_argument("--max-windows",type=int,default=1000)
    train=sub.add_parser("harth-train"); train.add_argument("--dataset-root",required=True); train.add_argument("--test-subject",required=True); train.add_argument("--model",choices=["tiny_mlp","nearest_centroid"],default="tiny_mlp"); train.add_argument("--hidden-units",type=int,default=8); train.add_argument("--window-size",type=int,default=128); train.add_argument("--stride",type=int,default=128); train.add_argument("--max-train-windows-per-subject",type=int,default=500); train.add_argument("--max-test-windows",type=int,default=2000); train.add_argument("--epochs",type=int,default=10); train.add_argument("--lr",type=float,default=0.01); train.add_argument("--batch-size",type=int,default=128); train.add_argument("--seed",type=int,default=0); train.add_argument("--abstain-threshold",type=float,default=0.0); train.add_argument("--model-output"); train.add_argument("--output")
    hsweep=sub.add_parser("harth-sweep"); hsweep.add_argument("--dataset-root",required=True); hsweep.add_argument("--test-subject",required=True); hsweep.add_argument("--hidden-units",default="1,2,4,8,16,32,64"); hsweep.add_argument("--window-size",type=int,default=128); hsweep.add_argument("--stride",type=int,default=128); hsweep.add_argument("--max-train-windows-per-subject",type=int,default=500); hsweep.add_argument("--max-test-windows",type=int,default=2000); hsweep.add_argument("--epochs",type=int,default=10); hsweep.add_argument("--lr",type=float,default=0.01); hsweep.add_argument("--batch-size",type=int,default=128); hsweep.add_argument("--seed",type=int,default=0); hsweep.add_argument("--abstain-threshold",type=float,default=0.0); hsweep.add_argument("--output")
    loso=sub.add_parser("harth-loso"); loso.add_argument("--dataset-root",required=True); loso.add_argument("--model",choices=["tiny_mlp","nearest_centroid"],default="tiny_mlp"); loso.add_argument("--hidden-units",type=int,default=8); loso.add_argument("--subjects"); loso.add_argument("--window-size",type=int,default=128); loso.add_argument("--stride",type=int,default=128); loso.add_argument("--max-train-windows-per-subject",type=int,default=500); loso.add_argument("--max-test-windows",type=int,default=2000); loso.add_argument("--epochs",type=int,default=10); loso.add_argument("--lr",type=float,default=0.01); loso.add_argument("--batch-size",type=int,default=128); loso.add_argument("--seed",type=int,default=0); loso.add_argument("--abstain-threshold",type=float,default=0.0); loso.add_argument("--output")
    lsw=sub.add_parser("harth-loso-sweep"); lsw.add_argument("--dataset-root",required=True); lsw.add_argument("--hidden-units",default="1,2,4,8,16,32,64"); lsw.add_argument("--subjects"); lsw.add_argument("--window-size",type=int,default=128); lsw.add_argument("--stride",type=int,default=128); lsw.add_argument("--max-train-windows-per-subject",type=int,default=500); lsw.add_argument("--max-test-windows",type=int,default=2000); lsw.add_argument("--epochs",type=int,default=10); lsw.add_argument("--lr",type=float,default=0.01); lsw.add_argument("--batch-size",type=int,default=128); lsw.add_argument("--seed",type=int,default=0); lsw.add_argument("--abstain-threshold",type=float,default=0.0); lsw.add_argument("--output")
    pareto=sub.add_parser("harth-pareto"); pareto.add_argument("--input",required=True);
    j03=sub.add_parser("j03-train"); j03.add_argument("--model",choices=["tiny_mlp","linear"],default="tiny_mlp"); j03.add_argument("--hidden-units",type=int,default=8); j03.add_argument("--train-problems",type=int,default=500); j03.add_argument("--test-problems",type=int,default=300); j03.add_argument("--epochs",type=int,default=20); j03.add_argument("--lr",type=float,default=0.01); j03.add_argument("--batch-size",type=int,default=128); j03.add_argument("--seed",type=int,default=0); j03.add_argument("--abstain-threshold",type=float,default=0.0); j03.add_argument("--permutation-trials",type=int,default=5); j03.add_argument("--output");
    j03s=sub.add_parser("j03-sweep"); j03s.add_argument("--hidden-units",default="1,2,4,8,16,32,64"); j03s.add_argument("--train-problems",type=int,default=500); j03s.add_argument("--test-problems",type=int,default=300); j03s.add_argument("--epochs",type=int,default=20); j03s.add_argument("--lr",type=float,default=0.01); j03s.add_argument("--batch-size",type=int,default=128); j03s.add_argument("--seed",type=int,default=0); j03s.add_argument("--abstain-threshold",type=float,default=0.0); j03s.add_argument("--permutation-trials",type=int,default=5); j03s.add_argument("--output");
    j04=sub.add_parser("j04-run"); j04.add_argument("--policy",choices=["heuristic","random"],default="heuristic"); j04.add_argument("--episodes",type=int,default=20); j04.add_argument("--max-pieces",type=int,default=300); j04.add_argument("--seed",type=int,default=0); j04.add_argument("--permutation-trials",type=int,default=3);
    j04t=sub.add_parser("j04-train"); j04t.add_argument("--model",choices=["tiny_mlp","linear"],default="tiny_mlp"); j04t.add_argument("--hidden-units",type=int,default=8); j04t.add_argument("--train-episodes",type=int,default=100); j04t.add_argument("--test-episodes",type=int,default=20); j04t.add_argument("--max-train-pieces",type=int,default=80); j04t.add_argument("--max-test-pieces",type=int,default=300); j04t.add_argument("--epochs",type=int,default=20); j04t.add_argument("--lr",type=float,default=0.01); j04t.add_argument("--batch-size",type=int,default=128); j04t.add_argument("--seed",type=int,default=0); j04t.add_argument("--abstain-threshold",type=float,default=0.0); j04t.add_argument("--permutation-trials",type=int,default=3); j04t.add_argument("--output");
    j04s=sub.add_parser("j04-sweep"); j04s.add_argument("--hidden-units",default="1,2,4,8,16,32,64"); j04s.add_argument("--train-episodes",type=int,default=100); j04s.add_argument("--test-episodes",type=int,default=20); j04s.add_argument("--max-train-pieces",type=int,default=80); j04s.add_argument("--max-test-pieces",type=int,default=300); j04s.add_argument("--epochs",type=int,default=20); j04s.add_argument("--lr",type=float,default=0.01); j04s.add_argument("--batch-size",type=int,default=128); j04s.add_argument("--seed",type=int,default=0); j04s.add_argument("--abstain-threshold",type=float,default=0.0); j04s.add_argument("--permutation-trials",type=int,default=3); j04s.add_argument("--output");
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
    if args.command=="harth-sweep":
        units=[int(x.strip()) for x in args.hidden_units.split(",") if x.strip()]
        if not units or any(u<1 for u in units): raise SystemExit("--hidden-units must contain positive integers")
        rows=run_j02_size_sweep(args.dataset_root,test_subject=args.test_subject,hidden_units=units,window_size=args.window_size,stride=args.stride,max_train_windows_per_subject=args.max_train_windows_per_subject,max_test_windows=args.max_test_windows,epochs=args.epochs,lr=args.lr,batch_size=args.batch_size,seed=args.seed,abstain_threshold=args.abstain_threshold,output=args.output)
        print(json.dumps(rows,indent=2)); return 0
    if args.command=="harth-loso":
        subjects=[x.strip() for x in args.subjects.split(",") if x.strip()] if args.subjects else None
        result=run_j02_loso(args.dataset_root,model=args.model,hidden_units=args.hidden_units,subjects=subjects,window_size=args.window_size,stride=args.stride,max_train_windows_per_subject=args.max_train_windows_per_subject,max_test_windows=args.max_test_windows,epochs=args.epochs,lr=args.lr,batch_size=args.batch_size,seed=args.seed,abstain_threshold=args.abstain_threshold,output=args.output)
        print(json.dumps(result,indent=2)); return 0
    if args.command=="harth-loso-sweep":
        subjects=[x.strip() for x in args.subjects.split(",") if x.strip()] if args.subjects else None
        units=[int(x.strip()) for x in args.hidden_units.split(",") if x.strip()]
        if not units or any(u<1 for u in units): raise SystemExit("--hidden-units must contain positive integers")
        result=run_j02_loso_size_sweep(args.dataset_root,hidden_units=units,subjects=subjects,window_size=args.window_size,stride=args.stride,max_train_windows_per_subject=args.max_train_windows_per_subject,max_test_windows=args.max_test_windows,epochs=args.epochs,lr=args.lr,batch_size=args.batch_size,seed=args.seed,abstain_threshold=args.abstain_threshold,output=args.output)
        print(json.dumps(result,indent=2)); return 0
    if args.command=="harth-pareto":
        from pathlib import Path
        from jev_bench.analysis import pareto_front
        payload=json.loads(Path(args.input).read_text(encoding="utf-8"))
        rows=payload.get("rows",payload if isinstance(payload,list) else [])
        print(json.dumps(pareto_front(rows),indent=2)); return 0
    if args.command=="j03-train":
        from pathlib import Path
        from jev_bench.tasks.j03_candidate_selection import J03CandidateSelection
        result=J03CandidateSelection().train_and_evaluate(train_problems=args.train_problems,test_problems=args.test_problems,model=args.model,hidden_units=args.hidden_units,epochs=args.epochs,lr=args.lr,batch_size=args.batch_size,seed=args.seed,abstain_threshold=args.abstain_threshold,permutation_trials=args.permutation_trials)
        payload=json.dumps(result,indent=2)
        if args.output:
            p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(payload,encoding="utf-8")
        print(payload); return 0
    if args.command=="j03-sweep":
        units=[int(x.strip()) for x in args.hidden_units.split(",") if x.strip()]
        if not units or any(u<1 for u in units): raise SystemExit("--hidden-units must contain positive integers")
        result=run_j03_size_sweep(hidden_units=units,train_problems=args.train_problems,test_problems=args.test_problems,epochs=args.epochs,lr=args.lr,batch_size=args.batch_size,seed=args.seed,abstain_threshold=args.abstain_threshold,permutation_trials=args.permutation_trials,output=args.output)
        print(json.dumps(result,indent=2)); return 0
    if args.command=="j04-run":
        from jev_bench.envs.tetris import TetrisEnv
        from jev_bench.policies.j04_tetris import J04HeuristicSelector, J04RandomSelector
        from jev_bench.tasks.j04_tetris import J04Tetris
        task=J04Tetris(); selector=J04HeuristicSelector(task.teacher_score) if args.policy=="heuristic" else J04RandomSelector(args.seed)
        rows=[TetrisEnv(args.seed+i).run_episode(selector,seed=args.seed+i,max_pieces=args.max_pieces) for i in range(args.episodes)]
        print(json.dumps({"task":"j04_tetris","policy":selector.name,"episodes":args.episodes,"seed":args.seed,"mean_return":sum(r["return"] for r in rows)/len(rows),"mean_lines":sum(r["lines"] for r in rows)/len(rows),"mean_pieces":sum(r["pieces"] for r in rows)/len(rows),"rows":rows},indent=2)); return 0
    if args.command=="j04-train":
        from pathlib import Path
        from jev_bench.tasks.j04_tetris import J04Tetris
        result=J04Tetris().train_and_evaluate(model=args.model,hidden_units=args.hidden_units,train_episodes=args.train_episodes,test_episodes=args.test_episodes,max_train_pieces=args.max_train_pieces,max_test_pieces=args.max_test_pieces,epochs=args.epochs,lr=args.lr,batch_size=args.batch_size,seed=args.seed,abstain_threshold=args.abstain_threshold,permutation_trials=args.permutation_trials)
        payload=json.dumps(result,indent=2)
        if args.output:
            p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(payload,encoding="utf-8")
        print(payload); return 0
    if args.command=="j04-sweep":
        from pathlib import Path
        from jev_bench.tasks.j04_tetris import J04Tetris
        units=[int(x.strip()) for x in args.hidden_units.split(",") if x.strip()]
        if not units or any(u<1 for u in units): raise SystemExit("--hidden-units must contain positive integers")
        rows=J04Tetris().size_sweep(hidden_units=units,train_episodes=args.train_episodes,test_episodes=args.test_episodes,max_train_pieces=args.max_train_pieces,max_test_pieces=args.max_test_pieces,epochs=args.epochs,lr=args.lr,batch_size=args.batch_size,seed=args.seed,abstain_threshold=args.abstain_threshold,permutation_trials=args.permutation_trials)
        payload=json.dumps({"task":"j04_tetris","protocol":"real-environment-size-sweep","rows":rows},indent=2)
        if args.output:
            p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(payload,encoding="utf-8")
        print(payload); return 0
    if args.command=="harth-train":
        result=get_task("j02_harth").train_and_evaluate(args.dataset_root,test_subject=args.test_subject,model=args.model,hidden_units=args.hidden_units,window_size=args.window_size,stride=args.stride,max_train_windows_per_subject=args.max_train_windows_per_subject,max_test_windows=args.max_test_windows,epochs=args.epochs,lr=args.lr,batch_size=args.batch_size,seed=args.seed,abstain_threshold=args.abstain_threshold,model_output=args.model_output)
        payload=json.dumps(result,indent=2)
        if args.output:
            from pathlib import Path
            p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(payload,encoding="utf-8")
        print(payload); return 0
    return 1
