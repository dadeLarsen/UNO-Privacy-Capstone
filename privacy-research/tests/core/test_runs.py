from datetime import datetime, timezone
import json

import pytest

from privacy_bench.core.files import digest_json, write_json
from privacy_bench.core.runs import SCHEMA_VERSION, cost_summary, load_run, run_directory

SPEC = {"name": "spec"}


def write_run(path, *, responses, status="complete", ids=("a", "b"), **changes):
    path.mkdir()
    record = {"schema_version": SCHEMA_VERSION, "status": status, "condition": {"name": "cond"},
              "item_ids": list(ids), "input_spec_sha256": digest_json(SPEC),
              "recorded_responses": len(responses), **changes}
    write_json(path / "run.json", record)
    write_json(path / "input.json", SPEC)
    (path / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in responses), encoding="utf-8")
    return path


GOOD = [{"id": "a", "raw_output": "x"}, {"id": "b", "raw_output": "y"}]


def test_run_directory_names_task_condition_and_utc_start(tmp_path):
    started = datetime(2026, 10, 11, 9, 5, 7, tzinfo=timezone.utc)
    assert run_directory("classification", "qwen", root=tmp_path, started=started) \
        == tmp_path / "classification" / "qwen" / "20261011-090507Z"


def test_complete_run_loads(tmp_path):
    run = load_run(write_run(tmp_path / "run", responses=GOOD))
    assert run.responses == GOOD
    assert run.name == "cond/run"


@pytest.mark.parametrize("responses, changes, message", [
    (GOOD, {"schema_version": 2}, "schema 3"),
    (GOOD, {"status": "failed"}, "not complete"),
    (GOOD, {"input_spec_sha256": "0" * 64}, "input.json differs"),
    (GOOD, {"item_ids": ["a", "a"]}, "unique item_ids"),
    (GOOD + [{"id": "a"}], {}, "duplicate or unknown"),
    ([{"id": "z"}], {}, "duplicate or unknown"),
    (GOOD, {"recorded_responses": 3}, "response count"),
    (GOOD[::-1], {}, "in item order"),
    (GOOD[:1], {}, "one response per item"),
    ([GOOD[0], {"id": "b", "error": "boom"}], {}, "failed responses"),
])
def test_inconsistent_runs_are_rejected(tmp_path, responses, changes, message):
    path = write_run(tmp_path / "run", responses=responses)
    record = json.loads((path / "run.json").read_text(encoding="utf-8"))
    record.update(changes)
    write_json(path / "run.json", record)
    with pytest.raises(ValueError, match=message):
        load_run(path)


def test_cost_summary_ignores_missing_and_boolean_values():
    summary = cost_summary([
        {"seconds": 1.0, "input_tokens": 10, "peak_allocated_bytes": 5},
        {"seconds": 3.0, "input_tokens": True, "peak_allocated_bytes": 9},
        {"error": "failed"},
    ])
    assert summary["mean_seconds"] == 2.0 and summary["seconds_count"] == 2
    assert summary["mean_input_tokens"] == 10 and summary["input_tokens_count"] == 1
    assert summary["max_peak_allocated_bytes"] == 9
    assert summary["mean_output_tokens"] is None and summary["output_tokens_count"] == 0
    assert summary["max_peak_reserved_bytes"] is None
