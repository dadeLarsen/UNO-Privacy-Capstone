"""The classifier-head strategy: label mapping and answers from a real tiny CPU checkpoint."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from privacy_bench.core.files import read_json
from privacy_bench.tasks.classification.data import Row
from privacy_bench.tasks.classification.strategies import STRATEGIES, classifier_head
from privacy_bench.tasks.classification.strategies.classifier_head import ClassifierHeadStrategy, label_mapping
from privacy_bench.tasks.classification.taxonomy import LABELS

COMMIT = "a" * 40
MODEL = {"id": "o/m", "revision": COMMIT, "precision": "fp32", "device": "cpu"}
PUBLISHED_LABELS = ["Data Retention", "Data Security", "Do Not Track", "First Party Collection/Use",
                    "International and Specific Audiences", "Other", "Policy Change", "Third Party Sharing/Collection",
                    "User Access, Edit and Deletion", "User Choice/Control"]


def test_registered_under_its_condition_name():
    assert STRATEGIES["classifier-head"] is ClassifierHeadStrategy


@pytest.mark.parametrize("model, options, message", [
    ({**MODEL, "precision": "bf16"}, {"max_input_tokens": 512}, "fp32"),
    (MODEL, {}, "missing: max_input_tokens"),
    ({**MODEL, "revision": "main"}, {"max_input_tokens": 512}, "40-character"),
])
def test_invalid_conditions_fail_before_any_model_loads(make_condition, model, options, message):
    with pytest.raises(ValueError, match=message):
        ClassifierHeadStrategy(make_condition("classifier-head", model, options))


def test_maps_permuted_checkpoint_labels_onto_the_benchmark(make_condition, tiny_classifier):
    strategy = ClassifierHeadStrategy(make_condition("classifier-head", MODEL, {"max_input_tokens": 64}))
    assert strategy.input_spec == {"name": "raw-text", "format": "sentence_text", "add_special_tokens": True,
                                   "truncation": False, "padding": False, "batch_size": 1, "max_input_tokens": 64}
    identity = {"model": "o/m", "revision": COMMIT, "resolved_revision": COMMIT, "local_snapshot": "x"}
    with patch.object(classifier_head, "resolve_snapshot", return_value=tiny_classifier), \
            patch.object(classifier_head, "snapshot_identity", return_value=identity):
        metadata = strategy.load()
    assert metadata["resolved_revision"] == COMMIT and len(metadata["label_mapping"]) == 10
    prepared = strategy.prepare(Row("x", "We retain café data.", (3,)))
    assert strategy.describe(prepared) == {"input_tokens": 7}
    response = strategy.predict(prepared)
    model_label = max(range(10), key=response["logits"].__getitem__)
    names = read_json(tiny_classifier / "config.json")["id2label"]
    expected = LABELS.index(names[str(model_label)].replace("Access, Edit", "Access Edit"))
    assert json.loads(response["raw_output"]) == {"label_id": expected}
    assert response["model_label_id"] == model_label and response["model_label"] == names[str(model_label)]
    assert len(response["input_text_sha256"]) == 64


def test_exact_ties_select_the_first_checkpoint_index(make_condition):
    strategy = ClassifierHeadStrategy(make_condition("classifier-head", MODEL, {"max_input_tokens": 64}))
    strategy._mapping = label_mapping(classifier_config())
    strategy._backend = SimpleNamespace(classify=lambda encoded: {"logits": [0.0] * 10, "probabilities": [0.1] * 10})
    response = strategy.predict({"text_sha256": "x", "encoded": {}})
    assert response["model_label_id"] == 0 and json.loads(response["raw_output"]) == {"label_id": 3}


def classifier_config(**changes):
    config = SimpleNamespace(problem_type="single_label_classification", num_labels=10,
                             id2label=dict(enumerate(PUBLISHED_LABELS)),
                             label2id={name: i for i, name in enumerate(PUBLISHED_LABELS)})
    for key, value in changes.items():
        setattr(config, key, value)
    return config


def test_label_mapping_covers_all_ten_categories_once():
    mapping = label_mapping(classifier_config())
    assert [entry["label_id"] for entry in mapping] == [3, 6, 7, 8, 5, 9, 0, 1, 4, 2]
    assert {entry["label"] for entry in mapping} == set(LABELS)


@pytest.mark.parametrize("changes", [
    {"id2label": {**dict(enumerate(LABELS)), 0: "LABEL_0"}, "label2id": {}},
    {"id2label": {**dict(enumerate(LABELS)), 0: LABELS[1]}},
    {"id2label": {i: label for i, label in enumerate(LABELS) if i}},
    {"id2label": dict(enumerate(LABELS)), "label2id": {**{label: i for i, label in enumerate(LABELS)}, LABELS[0]: 1}},
    {"problem_type": "multi_label_classification"},
    {"problem_type": None},
    {"num_labels": 9},
])
def test_label_mapping_rejects_unknown_duplicate_missing_and_inconsistent_labels(changes):
    with pytest.raises(ValueError):
        label_mapping(classifier_config(**changes))
