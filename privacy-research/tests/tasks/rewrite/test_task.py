"""The rewrite task end to end through the real harness and comparison summary, with a stand-in chat model."""

import pytest

from privacy_bench.core.conditions import parse_condition
from privacy_bench.core.files import read_json
from privacy_bench.core.harness import execute
from privacy_bench.report.comparisons import parse_comparison
from privacy_bench.report.summary import summarize
from privacy_bench.tasks import get_task
from privacy_bench.tasks.rewrite.data import load_prompt
from privacy_bench.tasks.rewrite.strategy import PromptedRewriteStrategy
from privacy_bench.tasks.rewrite.task import RewriteTask

COMMIT = "a" * 40


def condition(name, precision):
    return parse_condition(name, f"""
task = "rewrite"
strategy = "prompted"
[items]
passages = "appsi139-sections-20"
[model]
id = "o/m"
revision = "{COMMIT}"
precision = "{precision}"
[options]
prompt = "summary-baseline"
""")


class FakeChat:
    """Echoes its whole prompt, passage included, in bf16 and answers with one word in int8."""

    def __init__(self, precision):
        self.precision, self.calls = precision, []

    def encode(self, prompt, system=None):
        self.calls.append((prompt, system))
        return {"rendered": f"{system}|{prompt}", "encoded": {"input_ids": [1, 2, 3]}}

    def generate(self, chat):
        text = chat["rendered"] if self.precision == "bf16" else "Summary."
        return {"raw_output": text, "reached_output_token_limit": False, "seconds": 0.25,
                "input_tokens": 3, "output_tokens": len(text.split())}


@pytest.fixture
def fake_load(monkeypatch):
    def load(self):
        self._backend = FakeChat(self.model["precision"])
        return {"load_seconds": 1.0}
    monkeypatch.setattr(PromptedRewriteStrategy, "load", load)


def test_registry_and_strategy_settings():
    assert isinstance(get_task("rewrite"), RewriteTask)
    strategy = RewriteTask().build_strategy(condition("c", "bf16"))
    spec = strategy.input_spec
    assert spec["prompt"]["name"] == "summary-baseline" and spec["max_new_tokens"] == 1024
    assert spec["chat_template_kwargs"]["enable_thinking"] is False
    with pytest.raises(ValueError, match="Unknown rewrite strategy"):
        RewriteTask().build_strategy(parse_condition("c", 'task = "rewrite"\nstrategy = "kids"\n'))


def test_runs_compare_side_by_side(tmp_path, fake_load):
    task = RewriteTask()
    for name, precision in (("echo", "bf16"), ("short", "int8")):
        metrics = execute(task, condition(name, precision), tmp_path / "runs" / "rewrite" / name / "1")
        assert metrics["n"] == 20 and metrics["valid_output_rate"] == 1
    run = tmp_path / "runs" / "rewrite" / "echo" / "1"
    assert read_json(run / "input.json")["prompt"]["system"] == load_prompt("summary-baseline")["system"]
    assert (run / "evaluation" / "review.md").is_file() and (run / "evaluation" / "details.csv").is_file()

    comparison = parse_comparison("c", 'task = "rewrite"\nruns = ["rewrite/echo/1", "rewrite/short/1"]\n'
                                       '[labels]\n"rewrite/short/1" = "Short"\n')
    rows = summarize(task, comparison, runs_root=tmp_path / "runs")
    echo, short = rows
    assert echo["numbers_kept_rate"] == 1 and echo["qualifiers_kept_rate"] == 1
    assert short["numbers_kept_rate"] == 0 and short["condition_label"] == "summary-baseline"
    [path] = task.write_comparison_files(rows, {}, tmp_path, runs_root=tmp_path / "runs")
    text = path.read_text(encoding="utf-8")
    assert text.count("### Short") == 20 and text.count("### o/m bf16 · summary-baseline") == 20

    with pytest.raises(ValueError, match="unknown keys"):
        task.score_for_comparison(None, {"gold": "x"})
