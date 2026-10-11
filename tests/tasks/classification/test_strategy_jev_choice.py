"""The jev-choice strategy: the recorded input spec, prompt layout, and answers from a stand-in judge."""

import json
from unittest.mock import patch

import pytest

from privacy_bench.core.files import digest_json
from privacy_bench.models.jev.judge import choice_prompt
from privacy_bench.tasks.classification.data import Row
from privacy_bench.tasks.classification.strategies import STRATEGIES, jev_choice
from privacy_bench.tasks.classification.strategies.jev_choice import JEVChoiceStrategy
from privacy_bench.tasks.classification.taxonomy import LABELS

COMMIT = "a" * 40
MODEL = {"id": "o/m", "revision": COMMIT, "precision": "nf4"}
OPTIONS = {"prompt": "definitions", "max_input_tokens": 1024}


def test_registered_under_its_condition_name():
    assert STRATEGIES["jev-choice"] is JEVChoiceStrategy


@pytest.mark.parametrize("model, options, message", [
    (MODEL, {"prompt": "definitions"}, "missing: max_input_tokens"),
    (MODEL, {"prompt": "definitions", "max_input_tokens": 0}, "positive integer"),
    ({**MODEL, "revision": "main"}, OPTIONS, "40-character"),
    ({**MODEL, "reserve": 1}, OPTIONS, "unknown keys: reserve"),
])
def test_invalid_conditions_fail_before_any_model_loads(make_condition, model, options, message):
    with pytest.raises(ValueError, match=message):
        JEVChoiceStrategy(make_condition("jev-choice", model, options))


def test_input_spec_and_prompt_match_the_recorded_runs(make_condition):
    spec = JEVChoiceStrategy(make_condition("jev-choice", MODEL, OPTIONS)).input_spec
    # The input-spec hash the October JEV-9B runs recorded.
    assert digest_json(spec) == "7dbcd3cdb4f8ee76ad35bbe050124c06757740987e34b12ac0f88b04d93415e9"
    assert spec["option_letters"] == list("ABCDEFGHIJ")
    text = 'Sentence with "quotes", unicode café and\na newline.'
    prompt = choice_prompt(text, spec["question"], spec["options"])
    assert prompt.startswith(f"[kind] choice\n[state] {text}\n[question] Classify the privacy-policy sentence")
    assert "\nA) Policy Change: " in prompt and "\nJ) Other: " in prompt and prompt.endswith("\n[decision]:")


class FakeJudge:
    def __init__(self, logits):
        self.logits, self.prompts = logits, []

    def encode(self, prompt):
        self.prompts.append(prompt)
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    def choose(self, encoded, count):
        assert count == 10
        return {"logits": self.logits, "probabilities": [0.1] * 10, "input_tokens": 2, "seconds": 0.1}


@pytest.mark.parametrize("logits, expected", [([0.0] * 7 + [5.0] + [0.0] * 2, 7), ([1.0] * 10, 0)])
def test_answers_with_the_highest_choice_and_the_first_on_ties(make_condition, logits, expected):
    strategy = JEVChoiceStrategy(make_condition("jev-choice", MODEL, OPTIONS))
    strategy._judge = FakeJudge(logits)
    response = strategy.predict(strategy.prepare(Row("x", "Sentence", (0,))))
    assert json.loads(response["raw_output"]) == {"label_id": expected}
    assert (response["model_label"], response["option_letter"]) == (LABELS[expected], "ABCDEFGHIJ"[expected])
    assert strategy._judge.prompts[0].startswith("[kind] choice\n[state] Sentence\n")


def test_download_fetches_the_complete_bundle(make_condition, tmp_path):
    snapshot = tmp_path / "models--o--m" / "snapshots" / COMMIT
    strategy = JEVChoiceStrategy(make_condition("jev-choice", MODEL, OPTIONS))
    with patch.object(jev_choice, "download_bundle", return_value=snapshot) as download:
        assert strategy.download()["resolved_revision"] == COMMIT
    download.assert_called_once_with("o/m", COMMIT)
