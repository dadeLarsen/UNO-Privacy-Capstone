import pytest

from privacy_bench.core.conditions import parse_condition
from privacy_bench.tasks import get_task
from privacy_bench.tasks.classification.task import ClassificationTask

PILOT = """
task = "classification"
strategy = "prompted"
[items]
dataset = "opp115-sentences"
sample = "pilot-200-10cat"
gold = "pilot-200-10cat-reviewed"
"""


def test_task_registry_resolves_classification_and_rejects_unknown_tasks():
    assert isinstance(get_task("classification"), ClassificationTask)
    with pytest.raises(ValueError, match="Unknown task 'summaries'"):
        get_task("summaries")


def test_selection_records_dataset_sample_and_gold_identity():
    selection = ClassificationTask().select_items(parse_condition("pilot", PILOT))
    assert len(selection.items) == 200
    identity = selection.identity
    assert identity["dataset"] == {"name": "opp115-sentences",
                                   "sha256": "34dc5a59736e66caa570572f9cde340a1e3b1dcf02bedec726db9487b5542c8e"}
    assert identity["sample"] == {"name": "pilot-200-10cat", "n": 200,
                                  "manifest_sha256": "46702b141fee4aa75dba9dc8157a81661394790eeac386b49abfb06c9c5f1658"}
    assert identity["gold"] == {"name": "pilot-200-10cat-reviewed",
                                "accepted_gold_sha256": "2b5d2860f9af75f144daf1517c4c18747a123d01695dfc22abf287b75fc6a03d"}
    assert identity["taxonomy_id"] == "opp115-10-other" and len(identity["labels"]) == 10
    assert set(selection.references) == {row.id for row in selection.items}


def test_selection_without_gold_scores_original_labels(tmp_path):
    condition = parse_condition("pilot", PILOT.replace('gold = "pilot-200-10cat-reviewed"\n', ""))
    task = ClassificationTask()
    selection = task.select_items(condition)
    assert selection.identity["gold"] is None and selection.references is None
    metrics = task.evaluate(selection, [], tmp_path)
    assert metrics["scoring_mode"] == "original_single_label"
    assert metrics["status_counts"]["missing"] == 200
    assert "accepted accuracy" not in task.summary_line(metrics)


@pytest.mark.parametrize("items, message", [
    ('dataset = "opp115-sentences"\n', "missing: sample"),
    ('dataset = "opp115-sentences"\nsample = "pilot-200-10cat"\nsplit = "x"\n', "unknown keys: split"),
    ('dataset = "nope"\nsample = "pilot-200-10cat"\n', "does not exist"),
    ('dataset = "opp115-sentences"\nsample = "pilot-200-10cat"\ngold = "nope"\n', "does not exist"),
])
def test_invalid_item_settings_are_rejected(items, message):
    condition = parse_condition("bad", 'task = "classification"\nstrategy = "prompted"\n[items]\n' + items)
    with pytest.raises(ValueError, match=message):
        ClassificationTask().select_items(condition)


def test_unknown_strategy_is_rejected_before_any_model_loads():
    task = ClassificationTask()
    condition = parse_condition("pilot", PILOT.replace('"prompted"', '"telepathy"'))
    with pytest.raises(ValueError, match="Unknown classification strategy 'telepathy'"):
        task.build_strategy(condition)
