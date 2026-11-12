import pytest

from model_service.taxonomy import LABELS, TAXONOMY


def test_category_ids_are_in_order():
    assert [category["id"] for category in TAXONOMY["categories"]] == list(range(len(LABELS)))


def test_names_match_the_benchmark():
    pytest.importorskip("privacy_bench")
    from privacy_bench.tasks.classification.taxonomy import LABELS as MODEL_LABELS

    assert tuple(MODEL_LABELS) == LABELS
