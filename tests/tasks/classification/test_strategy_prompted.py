"""The prompted strategy: condition parsing, input spec, and responses from a stand-in chat model."""

from unittest.mock import patch

import pytest

from privacy_bench.tasks.classification.strategies import STRATEGIES, prompted
from privacy_bench.tasks.classification.strategies.prompted import PromptedStrategy

COMMIT = "a" * 40
MODEL = {"id": "o/m", "revision": COMMIT, "precision": "int8"}


def test_registered_under_its_condition_name():
    assert STRATEGIES["prompted"] is PromptedStrategy


@pytest.mark.parametrize("model, options, message", [
    ({**MODEL, "revision": "main"}, {"prompt": "definitions"}, "40-character"),
    ({"id": "o/m", "revision": COMMIT}, {"prompt": "definitions"}, "missing: precision"),
    ({**MODEL, "gpu": 1}, {"prompt": "definitions"}, "unknown keys: gpu"),
    (MODEL, {}, "missing: prompt"),
    (MODEL, {"prompt": "nope"}, "does not exist"),
    (MODEL, {"prompt": "definitions", "temperature": 0.7}, "unknown keys: temperature"),
])
def test_invalid_conditions_fail_before_any_model_loads(make_condition, model, options, message):
    with pytest.raises(ValueError, match=message):
        PromptedStrategy(make_condition("prompted", model, options))


class FakeChat:
    def __init__(self):
        self.prompts = []

    def encode(self, prompt):
        self.prompts.append(prompt)
        return {"rendered": "<r>" + prompt, "encoded": {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}}

    def generate(self, chat):
        return {"raw_output": '{"label_id": 3}', "input_tokens": 3, "seconds": 0.5}


def test_sends_the_recorded_definitions_prompt_and_keeps_the_raw_text(make_condition, pilot_rows):
    strategy = PromptedStrategy(make_condition("prompted", MODEL, {"prompt": "definitions", "max_new_tokens": 32,
                                                                   "seed": 42}))
    spec = strategy.input_spec
    assert spec["prompt_spec"]["name"] == "definitions" and (spec["max_new_tokens"], spec["seed"]) == (32, 42)
    assert spec["chat_template_kwargs"]["enable_thinking"] is False
    strategy._backend = FakeChat()
    prepared = strategy.prepare(pilot_rows["8396"])
    assert strategy.describe(prepared) == {"input_tokens": 3}
    assert strategy._backend.prompts[0].startswith("Classify the privacy-policy sentence into exactly ONE category")
    response = strategy.predict(prepared)
    # The hash the September Qwen3-14B runs recorded for this sentence.
    assert response["prompt_sha256"] == "aebb52a3885c2e9c1f0ae07734f4d75c09251c424cf3924ec58930569e947fa5"
    assert response["raw_output"] == '{"label_id": 3}' and response["seconds"] == 0.5


def test_options_default_to_the_recorded_generation_settings(make_condition):
    strategy = PromptedStrategy(make_condition("prompted", MODEL, {"prompt": "definitions"}))
    assert (strategy.input_spec["max_new_tokens"], strategy.input_spec["seed"]) == (32, 42)


def test_download_caches_the_standard_files_of_the_pinned_commit(make_condition, tmp_path):
    snapshot = tmp_path / "models--o--m" / "snapshots" / COMMIT
    strategy = PromptedStrategy(make_condition("prompted", MODEL, {"prompt": "definitions"}))
    with patch.object(prompted, "download_snapshot", return_value=snapshot) as download:
        assert strategy.download()["resolved_revision"] == COMMIT
    download.assert_called_once_with("o/m", COMMIT)
