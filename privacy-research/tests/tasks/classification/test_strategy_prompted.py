"""The retired prompted strategy: registered for the comparisons of its recorded runs, and refused by name."""

import pytest

from privacy_bench.core.harness import execute
from privacy_bench.tasks.classification.strategies import STRATEGIES
from privacy_bench.tasks.classification.strategies.prompted import PromptedStrategy
from privacy_bench.tasks.classification.task import ClassificationTask

MODEL = {"id": "o/m", "revision": "a" * 40, "precision": "int8"}


def test_registered_under_its_condition_name():
    assert STRATEGIES["prompted"] is PromptedStrategy
    assert PromptedStrategy.condition_label({"prompt_spec": {"name": "definitions"}}) == "Definitions"


def test_a_run_of_it_is_refused_by_name_and_writes_nothing(make_condition, tmp_path):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="'test' names the retired strategy 'prompted'.*README.md"):
        execute(ClassificationTask(), make_condition("prompted", MODEL, {"prompt": "definitions"}), output)
    assert not output.exists()
