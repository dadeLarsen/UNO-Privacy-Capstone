"""The retired jev-choice strategy: registered for the comparisons of its recorded runs, and refused by name."""

import pytest

from privacy_bench.core.harness import execute
from privacy_bench.tasks.classification.strategies import STRATEGIES
from privacy_bench.tasks.classification.strategies.jev_choice import JEVChoiceStrategy
from privacy_bench.tasks.classification.task import ClassificationTask

MODEL = {"id": "o/m", "revision": "a" * 40, "precision": "nf4"}
OPTIONS = {"prompt": "definitions", "max_input_tokens": 1024}


def test_registered_under_its_condition_name():
    assert STRATEGIES["jev-choice"] is JEVChoiceStrategy
    assert JEVChoiceStrategy.condition_label({"prompt_spec": {"name": "definitions"}}) == "JEV choice\ndefinitions"
    assert JEVChoiceStrategy.generates_text is False


def test_a_run_of_it_is_refused_by_name_and_writes_nothing(make_condition, tmp_path):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="'test' names the retired strategy 'jev-choice'.*README.md"):
        execute(ClassificationTask(), make_condition("jev-choice", MODEL, OPTIONS), output)
    assert not output.exists()
