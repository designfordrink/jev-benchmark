from __future__ import annotations
import argparse
import json
from jev_bench.policies.registry import get_policy, list_policies
from jev_bench.tasks.registry import get_task, list_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jev-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-tasks")
    sub.add_parser("list-policies")
    run = sub.add_parser("run")
    run.add_argument("--task", required=True)
    run.add_argument("--policy", required=True)
    run.add_argument("--episodes", type=int, default=20)
    run.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "list-tasks":
        print("\n".join(list_tasks()))
        return 0
    if args.command == "list-policies":
        print("\n".join(list_policies()))
        return 0
    if args.command == "run":
        task = get_task(args.task)
        policy = get_policy(args.policy)
        print(json.dumps(task.evaluate(policy, episodes=args.episodes, seed=args.seed), indent=2))
        return 0
    return 1
