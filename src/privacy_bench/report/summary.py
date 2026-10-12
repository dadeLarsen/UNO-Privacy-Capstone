"""Score every run in a comparison with the current task code and tabulate quality and cost."""

import csv
from pathlib import Path

from ..core.files import digest_json
from ..core.paths import RUNS_DIR
from ..core.protocols import Panel, Task
from ..core.runs import cost_summary, load_run
from .comparisons import Comparison

COST_PANELS = (
    Panel("mean_seconds", "Inference time", "Mean seconds / item", 1, "seconds", "lower"),
    Panel("max_peak_allocated_bytes", "Peak GPU allocation", "GiB · PyTorch allocated", 1 / 2**30, "memory", "lower"),
    Panel("mean_input_tokens", "Input context", "Mean tokens / item", 1, "tokens", "neutral"),
)
COST_FIELDS = ("mean_seconds", "max_peak_allocated_bytes", "max_peak_reserved_bytes", "mean_input_tokens",
               "mean_output_tokens")


def summarize(task: Task, comparison: Comparison, runs_root=RUNS_DIR) -> list:
    """One row per run; every run must be scored on the same items and references."""
    if comparison.task != task.name:
        raise ValueError(f"Comparison {comparison.name!r} is for task {comparison.task!r}, not {task.name!r}")
    rows = []
    for path in comparison.runs:
        run = load_run(Path(runs_root) / path)
        if run.record["task"] != task.name:
            raise ValueError(f"Run {path} belongs to task {run.record['task']!r}, not {task.name!r}")
        scored = task.score_for_comparison(run, comparison.scoring)
        model = run.record["condition"]["model"]
        scored_ids = set(scored["item_ids"])
        costs = cost_summary([response for response in run.responses if response["id"] in scored_ids])
        rows.append({
            "run": path, "condition": run.record["condition"]["name"], "strategy": run.record["strategy"],
            "condition_label": task.condition_label(run), "legend_label": comparison.labels.get(path, ""),
            "model": model["id"], "revision": model["revision"], "precision": model["precision"],
            "timing_scope": run.record["timing_scope"], "cohort_sha256": digest_json(scored["cohort"]),
            **scored["metrics"],
            **{field: costs[field] for field in COST_FIELDS},
            "load_seconds": run.record.get("backend", {}).get("load_seconds"),
            "transformers": run.record["environment"]["packages"].get("transformers"),
            "condition_sha256": run.record["condition_sha256"], "input_spec_sha256": run.record["input_spec_sha256"],
        })
    if len({row["cohort_sha256"] for row in rows}) != 1:
        raise ValueError("Runs in one comparison must be scored on the same items and reference answers")
    seen = set()
    for row in rows:
        cell = (row["model"], row["revision"], row["precision"], row["condition_label"])
        if cell in seen:
            raise ValueError(f"Two runs share model, revision, precision and condition {cell}; "
                             "compare one repetition at a time instead of averaging")
        seen.add(cell)
    return rows


def write_summary(rows: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
