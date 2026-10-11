"""The one run loop: select items, preflight every input, warm up, predict, record, evaluate."""

from collections import Counter
import json
from pathlib import Path

from .conditions import Condition
from .files import digest_json, write_json
from .protocols import Task
from .provenance import environment, utc_now
from .runs import SCHEMA_VERSION, cost_summary, load_run, run_condition


def _preflight(strategy, items, output: Path):
    """Prepare every input before any model call; one unusable input stops the run."""
    prepared, records, errors = [], [], []
    for item in items:
        try:
            value = strategy.prepare(item)
        except ValueError as exc:
            records.append({"id": item.id, "error": str(exc)})
            errors.append(f"ID {item.id}: {exc}")
            continue
        prepared.append(value)
        records.append({"id": item.id, **strategy.describe(value)})
    counts = [record["input_tokens"] for record in records if "input_tokens" in record]
    audit = {
        "n": len(items), "validated": len(items) - len(errors), "failed": len(errors),
        "min_input_tokens": min(counts) if counts else None,
        "max_input_tokens": max(counts) if counts else None,
        "token_count_histogram": dict(sorted(Counter(counts).items())),
    }
    write_json(output / "input-audit.json", {**audit, "rows": records})
    if errors:
        raise ValueError("Input preflight failed:\n" + "\n".join(errors))
    return prepared, audit


def execute(task: Task, condition: Condition, output: Path) -> dict:
    """Run a condition into a new directory and return the evaluation summary."""
    if condition.task != task.name:
        raise ValueError(f"Condition {condition.name!r} is for task {condition.task!r}, not {task.name!r}")
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output already exists; choose a new run directory: {output}")
    selection = task.select_items(condition)
    strategy = task.build_strategy(condition)
    spec = strategy.input_spec
    item_ids = [item.id for item in selection.items]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Selected items must have unique IDs")
    record = {
        "schema_version": SCHEMA_VERSION,
        "task": task.name, "strategy": strategy.name,
        "condition": condition.as_dict(), "condition_sha256": condition.sha256,
        "items": selection.identity, "item_ids": item_ids,
        "input_spec_sha256": digest_json(spec), "timing_scope": strategy.timing_scope,
        "warmup_passes": condition.warmup,
        "environment": environment(),
        "status": "initializing", "started_utc": utc_now(),
    }
    output.mkdir(parents=True)
    (output / "condition.toml").write_text(condition.source_text, encoding="utf-8")
    write_json(output / "input.json", spec)
    write_json(output / "run.json", record)
    responses, warmups = [], []
    try:
        record["backend"] = strategy.load()
        write_json(output / "run.json", record)
        prepared, record["input_audit"] = _preflight(strategy, selection.items, output)
        print(f"Validated all {len(item_ids)} inputs: {record['input_audit']['min_input_tokens']} to "
              f"{record['input_audit']['max_input_tokens']} tokens", flush=True)
        write_json(output / "warmup.json", warmups)
        for _ in range(condition.warmup):
            try:
                warmups.append(strategy.predict(prepared[0]))
            except BaseException as exc:
                warmups.append({"error": f"{type(exc).__name__}: {exc}"})
                raise
            finally:
                write_json(output / "warmup.json", warmups)
        record["status"] = "running"
        write_json(output / "run.json", record)
        progress_every = max(10, len(item_ids) // 20)
        with (output / "responses.jsonl").open("x", encoding="utf-8") as log:
            for index, (item, value) in enumerate(zip(selection.items, prepared, strict=True), 1):
                response = {"id": item.id}
                try:
                    response.update(strategy.predict(value))
                except BaseException as exc:
                    response["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    responses.append(response)
                    log.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
                    log.flush()
                if index == 1 or index % progress_every == 0 or index == len(item_ids):
                    print(f"Processed {index}/{len(item_ids)}", flush=True)
        record["status"] = "complete"
    except BaseException as exc:
        record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["recorded_responses"] = len(responses)
        record["finished_utc"] = utc_now()
        record["costs"] = cost_summary(responses)
        write_json(output / "run.json", record)
        evaluation = output / "evaluation"
        evaluation.mkdir()
        summary = task.evaluate(selection, responses, evaluation)
        print(f"Saved: {output.resolve()}", flush=True)
    return summary


def evaluate_run(task: Task, path: Path) -> dict:
    """Score a saved run again with the current scorer, rewriting its evaluation/ folder."""
    run = load_run(path, require_complete=False)
    condition = run_condition(run)
    if condition.task != task.name:
        raise ValueError(f"Run {path} belongs to task {condition.task!r}, not {task.name!r}")
    selection = task.select_items(condition)
    if selection.identity != run.record["items"]:
        raise ValueError(f"The data named by this run no longer matches its recorded identity: {path}")
    evaluation = run.path / "evaluation"
    evaluation.mkdir(exist_ok=True)
    return task.evaluate(selection, run.responses, evaluation)
