"""The run loop with a stand-in task and strategy; no model or task data involved."""

from dataclasses import dataclass
import json

import pytest

from privacy_bench.core.conditions import parse_condition
from privacy_bench.core.files import digest_json, read_json, read_jsonl
from privacy_bench.core.harness import execute
from privacy_bench.core.protocols import Selection
from privacy_bench.core.runs import SCHEMA_VERSION, load_run

CONDITION = parse_condition("fake-condition", """
task = "fake"
strategy = "echo"
warmup = 1
[items]
count = 3
[options]
limit = 10
""")


@dataclass(frozen=True)
class FakeItem:
    id: str
    text: str


class FakeStrategy:
    name = "echo"
    timing_scope = "fake_forward"

    def __init__(self, limit, fail_at=None, interrupt_at=None, warmup_fails=False, load_fails=False):
        self.limit, self.fail_at, self.interrupt_at = limit, fail_at, interrupt_at
        self.warmup_fails, self.load_fails = warmup_fails, load_fails
        self.calls, self.loaded = [], False

    @property
    def input_spec(self):
        return {"name": "echo", "limit": self.limit}

    def load(self):
        if self.load_fails:
            raise RuntimeError("weights missing")
        self.loaded = True
        return {"resolved_revision": "a" * 40, "load_seconds": 0.5}

    def prepare(self, item):
        if len(item.text) > self.limit:
            raise ValueError(f"{len(item.text)} characters exceed {self.limit}")
        return item

    def describe(self, prepared):
        return {"input_tokens": len(prepared.text)}

    def predict(self, prepared):
        assert self.loaded
        if self.warmup_fails and not self.calls:
            self.calls.append(prepared.id)
            raise RuntimeError("warmup broke")
        self.calls.append(prepared.id)
        if prepared.id == self.fail_at and len(self.calls) > 1:
            raise RuntimeError("CUDA out of memory")
        if prepared.id == self.interrupt_at and len(self.calls) > 1:
            raise KeyboardInterrupt
        return {"raw_output": prepared.text.upper(), "seconds": 0.25, "input_tokens": len(prepared.text)}


class FakeTask:
    name = "fake"

    def __init__(self, texts=("a", "bb", "ccc"), **strategy_options):
        self.items = [FakeItem(f"id-{index}", text) for index, text in enumerate(texts)]
        self.strategy_options = strategy_options
        self.strategy = None
        self.evaluated = None

    def select_items(self, condition):
        return Selection(items=self.items, identity={"sample": "fake", "n": len(self.items)})

    def build_strategy(self, condition, selection):
        self.strategy = FakeStrategy(condition.options["limit"], **self.strategy_options)
        return self.strategy

    def evaluate(self, selection, responses, output):
        self.evaluated = [response["id"] for response in responses]
        (output / "scores.json").write_text(json.dumps({"n": len(responses)}), encoding="utf-8")
        return {"n": len(responses)}

    def summary_line(self, metrics):
        return f"n={metrics['n']}"


def test_complete_run_records_identity_inputs_and_responses_in_item_order(tmp_path):
    task, output = FakeTask(), tmp_path / "run"
    assert execute(task, CONDITION, output) == {"n": 3}
    record = read_json(output / "run.json")
    assert record["schema_version"] == SCHEMA_VERSION
    assert (record["task"], record["strategy"], record["status"]) == ("fake", "echo", "complete")
    assert record["condition"] == CONDITION.as_dict()
    assert record["condition_sha256"] == CONDITION.sha256
    assert record["items"] == {"sample": "fake", "n": 3}
    assert record["item_ids"] == ["id-0", "id-1", "id-2"]
    assert record["input_spec_sha256"] == digest_json({"name": "echo", "limit": 10})
    assert record["timing_scope"] == "fake_forward"
    assert record["backend"] == {"resolved_revision": "a" * 40, "load_seconds": 0.5}
    assert record["recorded_responses"] == 3
    assert record["costs"]["mean_seconds"] == 0.25
    assert record["costs"]["mean_input_tokens"] == 2
    assert record["input_audit"]["token_count_histogram"] == {"1": 1, "2": 1, "3": 1}
    assert record["environment"]["code"]["commit"]
    assert (output / "condition.toml").read_text(encoding="utf-8") == CONDITION.source_text
    assert read_json(output / "input.json") == {"name": "echo", "limit": 10}
    assert [response["raw_output"] for response in read_jsonl(output / "responses.jsonl")] == ["A", "BB", "CCC"]
    # The warmup uses the first item, is saved, and is never scored.
    assert read_json(output / "warmup.json") == [{"raw_output": "A", "seconds": 0.25, "input_tokens": 1}]
    assert task.strategy.calls == ["id-0", "id-0", "id-1", "id-2"]
    assert task.evaluated == ["id-0", "id-1", "id-2"]
    assert (output / "evaluation" / "scores.json").is_file()
    run = load_run(output)
    assert run.name == "fake-condition/run"


def test_preflight_checks_every_input_and_blocks_all_inference(tmp_path):
    task, output = FakeTask(texts=("ok", "far too long text", "also much too long")), tmp_path / "run"
    with pytest.raises(ValueError, match="Input preflight failed") as caught:
        execute(task, CONDITION, output)
    assert "id-1" in str(caught.value) and "id-2" in str(caught.value)
    audit = read_json(output / "input-audit.json")
    assert (audit["validated"], audit["failed"]) == (1, 2)
    assert [row.get("error") is None for row in audit["rows"]] == [True, False, False]
    assert task.strategy.calls == []
    record = read_json(output / "run.json")
    assert record["status"] == "failed" and record["recorded_responses"] == 0
    assert task.evaluated == []


def test_inference_failure_keeps_earlier_responses_and_the_failed_record(tmp_path):
    task, output = FakeTask(fail_at="id-1"), tmp_path / "run"
    with pytest.raises(RuntimeError, match="out of memory"):
        execute(task, CONDITION, output)
    responses = read_jsonl(output / "responses.jsonl")
    assert [response["id"] for response in responses] == ["id-0", "id-1"]
    assert responses[1] == {"id": "id-1", "error": "RuntimeError: CUDA out of memory"}
    record = read_json(output / "run.json")
    assert record["status"] == "failed"
    assert record["error"] == "RuntimeError: CUDA out of memory"
    assert task.evaluated == ["id-0", "id-1"]
    with pytest.raises(ValueError, match="not complete"):
        load_run(output)
    assert load_run(output, require_complete=False).record["recorded_responses"] == 2


def test_interruption_marks_the_run_interrupted_and_keeps_the_attempted_item(tmp_path):
    task, output = FakeTask(interrupt_at="id-2"), tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        execute(task, CONDITION, output)
    record = read_json(output / "run.json")
    assert record["status"] == "interrupted"
    assert read_jsonl(output / "responses.jsonl")[-1] == {"id": "id-2", "error": "KeyboardInterrupt: "}


def test_warmup_failure_is_saved_and_never_scored(tmp_path):
    task, output = FakeTask(warmup_fails=True), tmp_path / "run"
    with pytest.raises(RuntimeError, match="warmup broke"):
        execute(task, CONDITION, output)
    assert read_json(output / "warmup.json") == [{"error": "RuntimeError: warmup broke"}]
    assert not (output / "responses.jsonl").exists()
    assert task.evaluated == []


def test_load_failure_records_the_error_and_evaluates_with_no_responses(tmp_path):
    task, output = FakeTask(load_fails=True), tmp_path / "run"
    with pytest.raises(RuntimeError, match="weights missing"):
        execute(task, CONDITION, output)
    record = read_json(output / "run.json")
    assert record["status"] == "failed" and "backend" not in record
    assert read_json(output / "input.json") == {"name": "echo", "limit": 10}
    assert task.evaluated == []


def test_existing_output_is_refused_before_selecting_items(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "keep.txt").write_text("unchanged", encoding="utf-8")
    task = FakeTask()
    with pytest.raises(ValueError, match="already exists"):
        execute(task, CONDITION, output)
    assert task.strategy is None
    assert [path.name for path in output.iterdir()] == ["keep.txt"]


def test_condition_for_another_task_is_refused(tmp_path):
    other = parse_condition("other", 'task = "classification"\nstrategy = "echo"\n')
    with pytest.raises(ValueError, match="for task 'classification'"):
        execute(FakeTask(), other, tmp_path / "run")
    assert not (tmp_path / "run").exists()
