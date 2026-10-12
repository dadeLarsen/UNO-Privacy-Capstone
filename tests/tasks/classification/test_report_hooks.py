"""Classification runs in comparisons, using saved pilot and APPSI-139 runs in results/runs/."""

import pytest

from privacy_bench.core.files import read_json
from privacy_bench.core.paths import RUNS_DIR, data_file
from privacy_bench.core.runs import load_run
from privacy_bench.tasks.classification.task import ClassificationTask

QWEN_INT8 = RUNS_DIR / "classification" / "qwen3-14b-int8-definitions" / "20260918-220631Z"
PRIVACY_MODEL = RUNS_DIR / "classification" / "privacy-model-fp32" / "20261002-213049Z"
JEV_NF4 = RUNS_DIR / "classification" / "jev-9b-nf4-definitions" / "20261002-205049Z"
PRIVACY_MODEL_APPSI = RUNS_DIR / "classification" / "privacy-model-fp32-appsi139-all" / "20261012-014711Z"
PRIVACY_MODEL_APPSI_RANDOM = (RUNS_DIR / "classification" / "privacy-model-fp32-appsi139-random-1000"
                              / "20261012-013138Z")


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


def test_a_comparison_sample_scores_only_its_sentences_of_a_larger_run(task):
    run = load_run(PRIVACY_MODEL_APPSI)
    whole = task.score_for_comparison(run, {"gold": "appsi139-astra"})
    halves = {half: task.score_for_comparison(run, {"gold": "appsi139-astra", "sample": f"appsi139-{half}-half"})
              for half in ("tuning", "test")}
    test = halves["test"]
    saved = read_json(data_file("classification", "samples", "appsi139-test-half", suffix=".json"))
    assert test["item_ids"] == saved["ids"] and whole["item_ids"] == run.record["item_ids"]
    assert (test["metrics"]["n"], test["metrics"]["sample"]) == (6058, "appsi139-test-half")
    assert test["cohort"]["sample"] == {"name": "appsi139-test-half", "manifest_sha256": saved["manifest_sha256"],
                                        "n": 6058}
    assert test["cohort"]["gold"]["accepted_gold_sha256"] != whole["cohort"]["gold"]["accepted_gold_sha256"]
    # The halves split every sentence once, so their correct counts add up to the whole run's.
    correct = sum(scored["metrics"]["accuracy"] * scored["metrics"]["n"] for scored in halves.values())
    assert correct == pytest.approx(whole["metrics"]["accuracy"] * 13291)


def test_a_comparison_sample_must_lie_inside_every_run(task):
    with pytest.raises(ValueError, match=r"no response for \d+ of the 6058 sentences in sample 'appsi139-test-half'"):
        task.score_for_comparison(load_run(PRIVACY_MODEL_APPSI_RANDOM),
                                  {"gold": "appsi139-astra", "sample": "appsi139-test-half"})


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
    assert "alone" not in notes
    subset = task.describe_comparison(rows, {"gold": "pilot-200-10cat-reviewed", "sample": "pilot-200-10cat"})
    assert "costs included, on the sentences of pilot-200-10cat alone" in " ".join(subset["footnotes"])
