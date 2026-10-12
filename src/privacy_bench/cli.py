"""pbench: run conditions, cache their models, and rescore saved runs."""

import argparse
import json
import os
from pathlib import Path
import sys

from .core.conditions import load_condition
from .core.harness import evaluate_run, execute
from .core.files import read_json
from .core.runs import run_directory
from .tasks import get_task


def run(args):
    condition = load_condition(args.condition)
    task = get_task(condition.task)
    output = args.output or run_directory(task.name, condition.name)
    metrics = execute(task, condition, output)
    print(task.summary_line(metrics))


def download(args):
    condition = load_condition(args.condition)
    strategy = get_task(condition.task).build_strategy(condition)
    print(json.dumps(strategy.download(), indent=2))


def evaluate(args):
    for path in args.runs:
        task = get_task(read_json(Path(path) / "run.json")["task"])
        print(f"{path}: {task.summary_line(evaluate_run(task, Path(path)))}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pbench", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run", help="Run one condition file into a new run directory")
    command.add_argument("condition", type=Path)
    command.add_argument("--output", type=Path, help="New run directory (default: results/runs/<task>/<condition>/<UTC start>)")
    command.set_defaults(handler=run)
    command = commands.add_parser("download", help="Cache a condition's pinned model while online")
    command.add_argument("condition", type=Path)
    command.set_defaults(handler=download)
    command = commands.add_parser("evaluate", help="Rescore saved runs, rewriting their evaluation/ folders")
    command.add_argument("runs", nargs="+", type=Path)
    command.set_defaults(handler=evaluate)
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        args.handler(args)
    except (ValueError, OSError, RuntimeError) as exc:
        sys.exit(f"Error: {exc}")
    except KeyboardInterrupt:
        sys.exit("Interrupted. Completed responses and the run record are saved.")


if __name__ == "__main__":
    main()
