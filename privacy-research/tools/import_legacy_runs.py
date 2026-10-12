"""Import completed runs from the AirLLM privacy-classification workspace into results/runs/.

Usage::

    uv run python tools/import_legacy_runs.py OLD_RUN_DIR=CONDITION.toml [...]

Each old run must agree with its condition file: model, revision, precision, prompt, input
specification, dataset, sample, and every per-sentence input hash. Responses and warmups are
copied byte for byte, the original run.json is kept as legacy-run.json, and the copied
responses are rescored and must reproduce the original metrics exactly.
"""

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys

from privacy_bench.core.conditions import load_condition
from privacy_bench.core.files import digest_bytes, digest_json, read_json, read_jsonl, write_json
from privacy_bench.core.paths import RUNS_DIR
from privacy_bench.core.provenance import code_revision, utc_now
from privacy_bench.core.runs import SCHEMA_VERSION, cost_summary, load_run
from privacy_bench.tasks import get_task
from privacy_bench.tasks.classification.prompts import build_prompt

# Metric fields whose wording changed in the port; every number must still match.
DESCRIPTION_FIELDS = {"scoring", "accepted_scoring"}


def check(condition, failures, message):
    if not condition:
        failures.append(message)


def input_audit(responses, items) -> dict:
    counts = [response["input_tokens"] for response in responses]
    histogram = {}
    for count in sorted(counts):
        histogram[count] = histogram.get(count, 0) + 1
    return {"n": len(items), "validated": len(items), "failed": 0, "min_input_tokens": min(counts),
            "max_input_tokens": max(counts), "token_count_histogram": histogram,
            "source": "Recomputed from the imported responses' input_tokens"}


def import_run(old: Path, condition_path: Path, root: Path) -> Path:
    old_record = read_json(old / "run.json")
    condition = load_condition(condition_path)
    task = get_task(condition.task)
    strategy = task.build_strategy(condition)
    selection = task.select_items(condition)
    responses = read_jsonl(old / "responses.jsonl")
    old_config, backend = old_record["model_config"], old_record["backend"]
    failures = []

    check(old_record["status"] == "complete", failures, f"status is {old_record['status']!r}")
    check(old_config["model"] == condition.model["id"], failures, "model id differs")
    check(old_config["revision"] == condition.model["revision"] == backend["resolved_revision"], failures,
          "pinned or resolved revision differs")
    check(old_config["precision"] == condition.model["precision"], failures, "precision differs")
    for key in ("device", "min_free_vram_gib", "max_input_tokens"):
        if key in old_config:
            expected = condition.model.get(key, condition.options.get(key))
            check(old_config[key] == expected, failures, f"{key} differs")
    for key in ("max_new_tokens", "seed"):
        if key in old_config:
            check(old_config[key] == condition.options[key], failures, f"{key} differs")
    spec = strategy.input_spec
    if "input_spec" in old_record:
        check(old_record["input_spec"] == spec, failures, "input specification differs")
        check(old_record["input_spec_sha256"] == digest_json(spec), failures, "input specification hash differs")
    if "prompt_spec_sha256" in old_record:
        check(old_record["prompt_spec_sha256"] == digest_json(spec.get("prompt_spec")), failures, "prompt differs")
    identity = selection.identity
    check(old_record["dataset_sha256"] == identity["dataset"]["sha256"], failures, "dataset hash differs")
    check(old_record["manifest_sha256"] == identity["sample"]["manifest_sha256"], failures, "sample differs")
    item_ids = [item.id for item in selection.items]
    check(old_record["ids"] == item_ids, failures, "item IDs or order differ")
    check([response["id"] for response in responses] == item_ids, failures, "responses are not in item order")
    check(not any(response.get("error") for response in responses), failures, "responses contain errors")
    for item, response in zip(selection.items, responses, strict=True):
        if "prompt_sha256" in response:
            prompt = build_prompt(item.text, spec["prompt_spec"])
            check(response["prompt_sha256"] == digest_bytes(prompt.encode("utf-8")), failures,
                  f"prompt hash differs for {item.id}")
        else:
            check(response["input_text_sha256"] == digest_bytes(item.text.encode("utf-8")), failures,
                  f"input text hash differs for {item.id}")
    if failures:
        raise ValueError(f"{old} does not match {condition_path}:\n  " + "\n  ".join(failures))

    started = datetime.fromisoformat(old_record["started_utc"])
    output = root / task.name / condition.name / started.strftime("%Y%m%d-%H%M%SZ")
    output.mkdir(parents=True)
    (output / "condition.toml").write_text(condition.source_text, encoding="utf-8")
    write_json(output / "input.json", spec)
    shutil.copyfile(old / "responses.jsonl", output / "responses.jsonl")
    shutil.copyfile(old / "warmup.json", output / "warmup.json")
    shutil.copyfile(old / "run.json", output / "legacy-run.json")
    audit = input_audit(responses, selection.items)
    write_json(output / "input-audit.json", {**audit, "rows": [{"id": r["id"], "input_tokens": r["input_tokens"]}
                                                               for r in responses]})
    versions = dict(backend.get("versions", {}))
    record = {
        "schema_version": SCHEMA_VERSION, "task": task.name, "strategy": strategy.name,
        "condition": condition.as_dict(), "condition_sha256": condition.sha256,
        "items": identity, "item_ids": item_ids,
        "input_spec_sha256": digest_json(spec), "timing_scope": strategy.timing_scope,
        "warmup_passes": old_record.get("warmup_generations", old_record.get("warmup_forward_passes")),
        "environment": {
            "python": versions.pop("python", old_record["python"]), "platform": backend.get("platform"),
            "packages": versions, "packages_scope": "Only the packages the original runner recorded",
            "code": {"legacy_source_sha256": old_record["source_sha256"]},
        },
        "status": "complete", "started_utc": old_record["started_utc"], "finished_utc": old_record["finished_utc"],
        "backend": backend, "input_audit": audit,
        "recorded_responses": len(responses), "costs": cost_summary(responses),
        "imported_from": {"path": str(old.resolve()), "run_json_sha256": digest_bytes((old / "run.json").read_bytes()),
                          "responses_sha256": digest_bytes((old / "responses.jsonl").read_bytes()),
                          "imported_utc": utc_now(), "importer_code": code_revision()},
    }
    write_json(output / "run.json", record)
    run = load_run(output)
    evaluation = output / "evaluation"
    evaluation.mkdir()
    metrics = task.evaluate(selection, run.responses, evaluation)
    original = read_json(old / "metrics.json")
    differing = sorted(key for key in set(metrics) - DESCRIPTION_FIELDS - {"per_category"}
                       if key in original and original[key] != metrics[key])
    # Per-category entries may gain fields; every field the old run recorded must still match.
    if [{key: new[key] for key in old_entry} for new, old_entry in
            zip(metrics["per_category"], original["per_category"], strict=True)] != original["per_category"]:
        differing.append("per_category")
    if differing or original.get("scoring_mode") != metrics["scoring_mode"]:
        raise ValueError(f"Rescored metrics differ from {old / 'metrics.json'}: {differing}")
    for key, value in record["costs"].items():
        if key in original and original[key] != value:
            raise ValueError(f"Cost summary {key} differs from {old / 'metrics.json'}")
    print(f"Imported {old.name} -> {output}: {task.summary_line(metrics)}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pairs", nargs="+", metavar="OLD_RUN_DIR=CONDITION.toml")
    parser.add_argument("--root", type=Path, default=RUNS_DIR)
    args = parser.parse_args(argv)
    pairs = []
    for pair in args.pairs:
        old, separator, condition = pair.partition("=")
        if not separator:
            parser.error(f"Expected OLD_RUN_DIR=CONDITION.toml, got {pair!r}")
        pairs.append((Path(old), Path(condition)))
    try:
        for old, condition in pairs:
            import_run(old, condition, args.root)
    except (ValueError, OSError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
