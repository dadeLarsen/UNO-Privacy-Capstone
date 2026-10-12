"""Classification runs in comparisons, using the imported pilot runs in results/runs/."""

import pytest

from privacy_bench.core.paths import RUNS_DIR
from privacy_bench.core.runs import load_run
from privacy_bench.tasks.classification.task import ClassificationTask

QWEN_INT8 = RUNS_DIR / "classification" / "qwen3-14b-int8-definitions" / "20260918-220631Z"
PRIVACY_MODEL = RUNS_DIR / "classification" / "privacy-model-fp32" / "20261002-213049Z"
JEV_NF4 = RUNS_DIR / "classification" / "jev-9b-nf4-definitions" / "20261002-205049Z"


@pytest.fixture(scope="module")
def task():
    return ClassificationTask()


def test_reviewed_gold_scores_accepted_accuracy_on_the_run_sample(task):
    scored = task.score_for_comparison(load_run(QWEN_INT8), {"gold": "pilot-200-10cat-reviewed"})
    metrics = scored["metrics"]
    assert (metrics["n"], metrics["accuracy"], metrics["macro_accuracy"]) == (200, 0.89, 0.89)
    assert metrics["original_accuracy"] == 0.765 and metrics["scoring_mode"] == "any_accepted_gold_label"
    assert (metrics["sample"], metrics["gold"]) == ("pilot-200-10cat", "pilot-200-10cat-reviewed")
    assert scored["cohort"]["gold"]["accepted_gold_sha256"].startswith("2b5d2860")


def test_without_gold_the_original_single_labels_are_scored(task):
    metrics = task.score_for_comparison(load_run(QWEN_INT8), {})["metrics"]
    assert metrics["accuracy"] == metrics["original_accuracy"] == 0.765
    assert metrics["macro_accuracy"] is None and metrics["gold"] == ""


def test_unknown_scoring_keys_are_rejected(task):
    with pytest.raises(ValueError, match="unknown keys: weights"):
        task.score_for_comparison(load_run(QWEN_INT8), {"weights": "x"})


def test_condition_labels_come_from_each_strategy(task):
    assert task.condition_label(load_run(QWEN_INT8)) == "Definitions"
    assert task.condition_label(load_run(PRIVACY_MODEL)) == "Raw sentence\n(classifier)"
    assert task.condition_label(load_run(JEV_NF4)) == "JEV choice\ndefinitions"


def test_panels_follow_the_gold_choice(task):
    assert [panel.field for panel in task.quality_panels({"gold": "pilot-200-10cat-reviewed"})] == \
        ["accuracy", "macro_accuracy", "valid_output_rate"]
    assert [panel.field for panel in task.quality_panels({})] == ["accuracy", "macro_f1", "valid_output_rate"]


def test_notes_explain_timing_validity_and_scoring_for_the_strategies_present(task):
    row = {"n": 200, "sample": "pilot-200-10cat", "gold": "pilot-200-10cat-reviewed",
           "macro_accuracy_groups": "sample strata"}
    rows = [{**row, "strategy": "prompted"}, {**row, "strategy": "classifier-head"}]
    described = task.describe_comparison(rows, {"gold": "pilot-200-10cat-reviewed"})
    assert "200 sentences" in described["context"]
    assert "accepted labels pilot-200-10cat-reviewed" in described["context"]
    notes = " ".join(described["footnotes"])
    assert "prompt processing + generation" in notes and "classifier forward pass" in notes
    assert "JEV" not in notes and "always valid" in notes
    assert "marked in pilot-200-10cat-reviewed" in notes and "the category the sample drew them for" in notes
    unstratified = task.describe_comparison([{**rows[0], "macro_accuracy_groups": "original labels"}],
                                            {"gold": "pilot-200-10cat-reviewed"})
    assert "grouping sentences by their original labels" in " ".join(unstratified["footnotes"])
    without_gold = task.describe_comparison([{**rows[0], "gold": "", "macro_accuracy_groups": None}], {})
    assert "original labels" in without_gold["context"]
    assert "macro-F1 uses the sentences with one original label" in " ".join(without_gold["footnotes"])
