"""The retired classifier-head strategy: registered for the comparisons of its recorded runs, and refused by name."""

import pytest

from privacy_bench.core.harness import execute
from privacy_bench.tasks.classification.strategies import STRATEGIES
from privacy_bench.tasks.classification.strategies.classifier_head import ClassifierHeadStrategy
from privacy_bench.tasks.classification.task import ClassificationTask

MODEL = {"id": "o/m", "revision": "a" * 40, "precision": "fp32"}
OPTIONS = {"max_input_tokens": 512}


def test_registered_under_its_condition_name():
    assert STRATEGIES["classifier-head"] is ClassifierHeadStrategy
    assert ClassifierHeadStrategy.condition_label({"name": "raw-text"}) == "Raw sentence\n(classifier)"
    assert ClassifierHeadStrategy.generates_text is False


def test_a_run_of_it_is_refused_by_name_and_writes_nothing(make_condition, tmp_path):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="'test' names the retired strategy 'classifier-head'.*README.md"):
        execute(ClassificationTask(), make_condition("classifier-head", MODEL, OPTIONS), output)
    assert not output.exists()
