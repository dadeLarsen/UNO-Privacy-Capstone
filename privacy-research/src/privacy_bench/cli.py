"""pbench: run conditions, fit heads, cache models, rescore saved runs, compare them, and judge rewrites."""

import argparse
import json
from pathlib import Path
import sys

from .core.conditions import load_condition
from .core.fits import execute_fit, fit_directory, load_fit_spec
from .core.harness import evaluate_run, execute
from .core.files import read_json
from .core.paths import COMPARISONS_DIR, REPO_ROOT
from .core.runs import run_directory
from .report.comparisons import load_comparison
from .report.plot import bar_label, common_footnotes, draw_matrix
from .report.summary import COST_PANELS, summarize, write_summary
from .tasks import get_task


def run(args):
    condition = load_condition(args.condition)
    task = get_task(condition.task)
    output = args.output or run_directory(task.name, condition.name)
    metrics = execute(task, condition, output)
    print(task.summary_line(metrics))


def fit(args):
    spec = load_fit_spec(args.fit)
    task = get_task(spec.task)
    record = execute_fit(task, spec, args.output or fit_directory(task.name, spec.name))
    print(f"Fitted {', '.join(record['tensors'])}")


def download(args):
    condition = load_condition(args.condition)
    strategy = get_task(condition.task).build_strategy(condition)
    print(json.dumps(strategy.download(), indent=2))


def evaluate(args):
    for path in args.runs:
        task = get_task(read_json(Path(path) / "run.json")["task"])
        print(f"{path}: {task.summary_line(evaluate_run(task, Path(path)))}")


def compare(args):
    comparison = load_comparison(args.comparison)
    task = get_task(comparison.task)
    rows = summarize(task, comparison)
    output = (args.output_dir or COMPARISONS_DIR / comparison.name).resolve()
    summary = output / "summary.csv"
    write_summary(rows, summary)
    (output / "comparison.toml").write_text(comparison.source_text, encoding="utf-8")
    described = task.describe_comparison(rows, comparison.scoring)
    source = summary.relative_to(REPO_ROOT) if summary.is_relative_to(REPO_ROOT) else summary
    footnotes = [*common_footnotes(rows), *described["footnotes"], f"Source: {source.as_posix()}"]
    quality = task.quality_panels(comparison.scoring)
    paths = draw_matrix(rows, [*quality, *COST_PANELS], output,
                        title=comparison.title or comparison.name,
                        context=f"{described['context']}  |  {len(rows)} runs", footnotes=footnotes,
                        group_by=comparison.group_by)
    paths = [*paths, *task.write_comparison_files(rows, comparison.scoring, output)]
    headline = quality[0]
    for row in rows:
        value = row[headline.field]
        shown = "N/A" if value is None else bar_label(value * headline.scale, headline.style)
        print(f"{row['run']}: {headline.field}={shown} mean_seconds={row['mean_seconds']:.4f}")
    for path in (summary, *paths):
        print(f"Saved: {path}")


def judging_task(name):
    task = get_task(name)
    if not hasattr(task, "prepare_judgment"):
        raise ValueError(f"The {name} task has no judging")
    return task


def judge(args):
    if args.action == "prepare":
        comparison = load_comparison(args.comparison)
        directory = judging_task(comparison.task).prepare_judgment(comparison, args.instructions, args.judge,
                                                                   output=args.output)
        print(f"Prompts: {directory / 'prompts'}\nWrite each reply to {directory / 'replies'}/<passage>.json, "
              f"then run: pbench judge record {directory}")
        return
    record = judging_task(read_json(args.judgment / "judgment.json")["task"]).record_judgment(args.judgment)
    judged = sum(len(entry["labels"]) for entry in record["passages"].values())
    print(f"Recorded {judged} judged rewrites across {len(record['passages'])} passages: {args.judgment}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pbench", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run", help="Run one condition file into a new run directory")
    command.add_argument("condition", type=Path)
    command.add_argument("--output", type=Path, help="New run directory (default: results/runs/<task>/<condition>/<UTC start>)")
    command.set_defaults(handler=run)
    command = commands.add_parser("fit", help="Fit what a fit file describes into a new fit record")
    command.add_argument("fit", type=Path)
    command.add_argument("--output", type=Path, help="New fit directory (default: results/fits/<task>/<fit>/<UTC start>)")
    command.set_defaults(handler=fit)
    command = commands.add_parser("download", help="Cache a condition's pinned model while online")
    command.add_argument("condition", type=Path)
    command.set_defaults(handler=download)
    command = commands.add_parser("evaluate", help="Rescore saved runs, rewriting their evaluation/ folders")
    command.add_argument("runs", nargs="+", type=Path)
    command.set_defaults(handler=evaluate)
    command = commands.add_parser("compare", help="Summarize and plot the runs named by a comparison file")
    command.add_argument("comparison", type=Path)
    command.add_argument("--output-dir", type=Path, help="Default: results/comparisons/<comparison name>/")
    command.set_defaults(handler=compare)
    command = commands.add_parser("judge", help="Prepare blind prompts for a judge, or record its checked replies")
    actions = command.add_subparsers(dest="action", required=True)
    action = actions.add_parser("prepare", help="Write one prompt per passage for a comparison's runs")
    action.add_argument("comparison", type=Path)
    action.add_argument("--judge", required=True, help="Who judges, for the record: the model and how it ran")
    action.add_argument("--instructions", default="keypoints", help="Judge instructions under prompts/<task>/judge/")
    action.add_argument("--output", type=Path, help="New judgment directory (default: results/judgments/<task>/<comparison name>/)")
    action = actions.add_parser("record", help="Check every reply and freeze the judgment")
    action.add_argument("judgment", type=Path)
    command.set_defaults(handler=judge)
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (ValueError, OSError, RuntimeError) as exc:
        sys.exit(f"Error: {exc}")
    except KeyboardInterrupt:
        sys.exit("Interrupted. Completed responses and the run record are saved.")


if __name__ == "__main__":
    main()
