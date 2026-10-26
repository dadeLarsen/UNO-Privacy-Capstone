"""Blind key-point judging end to end: prompts, checked replies, the frozen record, and comparison scores."""

import json

import pytest

from privacy_bench.core.conditions import parse_condition
from privacy_bench.core.files import read_json
from privacy_bench.core.harness import execute
from privacy_bench.report.comparisons import parse_comparison
from privacy_bench.report.summary import summarize
from privacy_bench.tasks.rewrite import task as task_module
from privacy_bench.tasks.rewrite.data import load_passages
from privacy_bench.tasks.rewrite.judging import load_checklist, parse_reply, unquoted
from privacy_bench.tasks.rewrite.strategy import PromptedRewriteStrategy
from privacy_bench.tasks.rewrite.task import JUDGE_PANELS, RewriteTask

RUNS = ["rewrite/echo/1", "rewrite/short/1", "rewrite/broken/1"]


class FakeChat:
    """Echoes its prompt in bf16, answers one word in int8, and shows reasoning tags in nf4."""

    def __init__(self, precision):
        self.precision = precision

    def encode(self, prompt, system=None):
        return {"rendered": prompt, "encoded": {"input_ids": [1, 2, 3]}}

    def generate(self, chat):
        text = {"bf16": chat["rendered"], "int8": "Summary.", "nf4": "<think>x</think> Summary."}[self.precision]
        return {"raw_output": text, "reached_output_token_limit": False, "seconds": 0.25, "input_tokens": 3,
                "output_tokens": 2}


@pytest.fixture
def judged(tmp_path, monkeypatch):
    """Three runs on the 20 passages and a pending judgment of them."""
    def load(self):
        self._backend = FakeChat(self.model["precision"])
        return {"load_seconds": 1.0}
    monkeypatch.setattr(PromptedRewriteStrategy, "load", load)
    monkeypatch.setattr(task_module, "JUDGMENTS_DIR", tmp_path / "judgments")
    task = RewriteTask()
    for path, precision in zip(RUNS, ("bf16", "int8", "nf4")):
        name = path.split("/")[1]
        execute(task, parse_condition(name, f'task = "rewrite"\nstrategy = "prompted"\n[items]\n'
                                            f'passages = "appsi139-sections-20"\n[model]\nid = "o/{name}"\n'
                                            f'revision = "{"a" * 40}"\nprecision = "{precision}"\n'
                                            '[options]\nprompt = "summary-v3"\n'), tmp_path / "runs" / path)
    comparison = parse_comparison("judged", f'task = "rewrite"\nruns = {json.dumps(RUNS)}\n')
    directory = task.prepare_judgment(comparison, "keypoints", "a test judge", runs_root=tmp_path / "runs")
    return task, directory, tmp_path / "runs"


def write_replies(directory, contradicted_quote="Summary."):
    """The echo keeps every point; the one-word summary contradicts the first and invents a major claim."""
    passages, identity = load_passages("appsi139-sections-20")
    checklist, _ = load_checklist(passages, identity)
    record = read_json(directory / "judgment.json")
    for passage_id, entry in record["passages"].items():
        ids = [item["id"] for item in checklist[passage_id]]
        rewrites = []
        for label, item in entry["labels"].items():
            if item["run"] == "rewrite/echo/1":
                rewrites.append({"label": label, "verdicts": dict.fromkeys(ids, "kept")})
            else:
                rewrites.append({"label": label, "verdicts": {**dict.fromkeys(ids, "missing"), ids[0]: "contradicted"},
                                 "evidence": {ids[0]: contradicted_quote},
                                 "unsupported": [{"quote": "summary", "claim": "x", "severity": "major"}]})
        (directory / "replies" / f"{passage_id}.json").write_text(json.dumps({"rewrites": rewrites}), encoding="utf-8")
    return checklist


def test_prompts_are_blind_and_skip_invalid_rewrites(judged):
    _, directory, _ = judged
    record = read_json(directory / "judgment.json")
    assert record["status"] == "pending" and record["judge"] == "a test judge"
    first = record["passages"]["128683-128686"]
    assert sorted(item["run"] for item in first["labels"].values()) == ["rewrite/echo/1", "rewrite/short/1"]
    assert first["not_judged"] == {"rewrite/broken/1": "reasoning"}
    prompt = (directory / "prompts" / "128683-128686.txt").read_text(encoding="utf-8")
    assert "Rewrite A:" in prompt and "Rewrite B:" in prompt and "Rewrite C:" not in prompt
    assert "(critical)" in prompt and "rewrite/" not in prompt and "o/echo" not in prompt
    orders = {tuple(item["run"] for item in entry["labels"].values()) for entry in record["passages"].values()}
    assert len(orders) == 2


def test_recorded_judgments_score_each_run_and_draw_the_chart(judged, tmp_path):
    task, directory, runs_root = judged
    with pytest.raises(ValueError, match="128683-128686: .*No such file|128683-128686: .*cannot find"):
        task.record_judgment(directory, runs_root=runs_root)
    checklist = write_replies(directory)
    task.record_judgment(directory, runs_root=runs_root)
    assert read_json(directory / "judgment.json")["status"] == "recorded"

    comparison = parse_comparison("judged", f'task = "rewrite"\nruns = {json.dumps(RUNS)}\n'
                                            '[scoring]\njudgments = "judged"\n')
    echo, short, broken = summarize(task, comparison, runs_root=runs_root)
    points = sum(len(items) for items in checklist.values())
    assert echo["judged_rewrites"] == 20 and echo["keypoints_kept_share"] == 1 and echo["misleading_rate"] == 0
    assert short["keypoints_contradicted_share"] == pytest.approx(20 / points) and short["misleading_rate"] == 1
    assert short["major_unsupported_rate"] == 1
    assert broken["judged_rewrites"] == 0 and broken["keypoints_kept_share"] is None
    assert JUDGE_PANELS[0] in task.quality_panels({"judgments": "judged"})
    (tmp_path / "out").mkdir()
    names = [path.name for path in task.write_comparison_files([echo, short, broken], {"judgments": "judged"},
                                                               tmp_path / "out", runs_root=runs_root)]
    assert names == ["side-by-side.md", "keypoints.png", "keypoints.svg"]
    assert "Judge: a test judge" in (tmp_path / "out" / "keypoints.svg").read_text(encoding="utf-8")

    reply = directory / "replies" / "128683-128686.json"
    reply.write_text(reply.read_text(encoding="utf-8").replace("missing", "partial", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the hash"):
        summarize(task, comparison, runs_root=runs_root)


def test_record_refuses_quotes_the_rewrite_does_not_contain(judged):
    task, directory, runs_root = judged
    write_replies(directory, contradicted_quote="we will sell your data")
    with pytest.raises(ValueError, match="point 1: a contradicted verdict needs a quote"):
        task.record_judgment(directory, runs_root=runs_root)
    comparison = parse_comparison("judged", f'task = "rewrite"\nruns = {json.dumps(RUNS)}\n')
    with pytest.raises(ValueError, match="never overwritten"):
        task.prepare_judgment(comparison, "keypoints", "a test judge", runs_root=runs_root)


@pytest.mark.parametrize("reply, message", [
    ("not json", "not valid JSON"),
    ('{"rewrites": [], "notes": 1}', "exactly"),
    ('{"rewrites": [{"label": "A", "verdicts": {"1": "kept", "2": "kept"}}]}', "no grades for rewrites B"),
    ('{"rewrites": [{"label": "C", "verdicts": {}}]}', "unexpected or repeated label 'C'"),
    ('{"rewrites": [{"label": "A", "verdicts": {"1": "kept"}}]}', "for each point"),
    ('{"rewrites": [{"label": "A", "verdicts": {"1": "kept", "2": "fine"}}]}', "for each point"),
    ('{"rewrites": [{"label": "A", "verdicts": {"1": "kept", "2": "kept"}, "unsupported": [{"quote": "x"}]}]}',
     "severity"),
])
def test_malformed_replies_are_refused(reply, message):
    with pytest.raises(ValueError, match=message):
        parse_reply(reply, {"A": "We may share data.", "B": "We share data."}, ["1", "2"])


def test_quotes_match_across_case_spacing_curly_quotes_and_ellipses():
    rewrites = {"A": "We  may share your “personal” data,\nunless you opt out."}
    grade = {"verdicts": {"1": "partial", "2": "contradicted"}, "unsupported": [],
             "evidence": {"1": '"we may share your "personal" data"', "2": "may share … unless you opt out"}}
    assert unquoted({"A": grade}, rewrites) == []
    grade["evidence"]["2"] = "we will share"
    assert unquoted({"A": grade}, rewrites) == ["rewrite A, point 2: a contradicted verdict needs a quote from the rewrite"]
