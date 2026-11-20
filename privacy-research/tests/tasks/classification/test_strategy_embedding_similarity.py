"""The retired embedding-similarity strategy: registered for the comparisons of its recorded runs, and refused."""

import pytest

from privacy_bench.core.harness import execute
from privacy_bench.tasks.classification.strategies import STRATEGIES, embedding_similarity
from privacy_bench.tasks.classification.strategies.embedding_similarity import EmbeddingSimilarityStrategy
from privacy_bench.tasks.classification.task import ClassificationTask

MODEL = {"id": "o/m", "revision": "a" * 40, "precision": "bf16", "text_only": True}
CLASSIFY = "task: classification | query: "
SEARCH_QUERY, SEARCH_DOCUMENT = "task: search result | query: ", "title: none | text: "
OPTIONS = {"prompt": "definitions", "sentence_prefix": CLASSIFY, "category_prefix": CLASSIFY,
           "max_input_tokens": 8192}


def recorded_spec(sentence_prefix, category_prefix):
    """The part of a recorded run's input.json that its chart label reads."""
    return {"prompt_spec": {"name": "definitions"}, "sentence_prefix": sentence_prefix,
            "category_prefix": category_prefix}


def test_registered_under_its_condition_name():
    assert STRATEGIES["embedding-similarity"] is EmbeddingSimilarityStrategy
    assert EmbeddingSimilarityStrategy.generates_text is False


def test_the_label_names_a_shared_prefix_once_and_keeps_different_prefixes_apart():
    assert EmbeddingSimilarityStrategy.condition_label(recorded_spec(CLASSIFY, CLASSIFY)) == (
        "Embedding\nsimilarity\ndefinitions\nclassification")
    assert EmbeddingSimilarityStrategy.condition_label(recorded_spec(SEARCH_QUERY, SEARCH_DOCUMENT)) == (
        "Embedding\nsimilarity\ndefinitions\nsentence: search result\ncategories: document")
    assert EmbeddingSimilarityStrategy.condition_label(recorded_spec(SEARCH_DOCUMENT, SEARCH_QUERY)).endswith(
        "\nsentence: document\ncategories: search result")


@pytest.mark.parametrize("prefix, name", [
    ("task: sentence similarity | query: ", "sentence similarity"),
    ("title: none | text: ", "document"),
    ("title: Data Retention | text: ", "document"),
    ("query: ", "query:"),
])
def test_instruction_names_come_from_the_prefix(prefix, name):
    assert embedding_similarity.instruction_name(prefix) == name


def test_a_run_of_it_is_refused_by_name_and_writes_nothing(make_condition, tmp_path):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="'test' names the retired strategy 'embedding-similarity'.*README.md"):
        execute(ClassificationTask(), make_condition("embedding-similarity", MODEL, OPTIONS), output)
    assert not output.exists()
