"""The embedding-similarity strategy: input spec, category embeddings, and the most similar category."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from privacy_bench.tasks.classification.data import Row
from privacy_bench.tasks.classification.strategies import STRATEGIES, embedding_similarity
from privacy_bench.tasks.classification.strategies.embedding_similarity import EmbeddingSimilarityStrategy
from privacy_bench.tasks.classification.taxonomy import LABELS

COMMIT = "a" * 40
MODEL = {"id": "o/m", "revision": COMMIT, "precision": "bf16", "text_only": True}
CLASSIFY = "task: classification | query: "
SEARCH_QUERY, SEARCH_DOCUMENT = "task: search result | query: ", "title: none | text: "
OPTIONS = {"prompt": "definitions", "sentence_prefix": CLASSIFY, "category_prefix": CLASSIFY,
           "max_input_tokens": 8192}
SEARCH = {**OPTIONS, "sentence_prefix": SEARCH_QUERY, "category_prefix": SEARCH_DOCUMENT}


def test_registered_under_its_condition_name():
    assert STRATEGIES["embedding-similarity"] is EmbeddingSimilarityStrategy


@pytest.mark.parametrize("model, options, message", [
    (MODEL, {k: v for k, v in OPTIONS.items() if k != "sentence_prefix"}, "missing: sentence_prefix"),
    (MODEL, {k: v for k, v in OPTIONS.items() if k != "category_prefix"}, "missing: category_prefix"),
    (MODEL, {**OPTIONS, "sentence_prefix": " "}, "sentence_prefix"),
    (MODEL, {**OPTIONS, "category_prefix": 3}, "category_prefix"),
    (MODEL, {**OPTIONS, "task_prefix": CLASSIFY}, "unknown keys: task_prefix"),
    (MODEL, {k: v for k, v in OPTIONS.items() if k != "max_input_tokens"}, "missing: max_input_tokens"),
    ({**MODEL, "revision": "main"}, OPTIONS, "40-character"),
    ({**MODEL, "pooling": "cls"}, OPTIONS, "unknown keys: pooling"),
])
def test_invalid_conditions_fail_before_any_model_loads(make_condition, model, options, message):
    with pytest.raises(ValueError, match=message):
        EmbeddingSimilarityStrategy(make_condition("embedding-similarity", model, options))


def test_input_spec_names_the_category_texts_and_a_shared_prefix(make_condition):
    spec = EmbeddingSimilarityStrategy(make_condition("embedding-similarity", MODEL, OPTIONS)).input_spec
    assert spec["category_texts"][0].startswith("Policy Change: Whether or how the privacy policy may change")
    assert spec["category_texts"][9].startswith("Other: Introductions")
    assert len(spec["category_texts"]) == 10 and spec["labels"] == list(LABELS)
    assert spec["sentence_prefix"] == spec["category_prefix"] == CLASSIFY
    assert spec["truncation"] is False and spec["max_input_tokens"] == 8192
    assert EmbeddingSimilarityStrategy.condition_label(spec) == "Embedding\nsimilarity\ndefinitions\nclassification"


def test_input_spec_and_label_keep_different_prefixes_apart(make_condition):
    spec = EmbeddingSimilarityStrategy(make_condition("embedding-similarity", MODEL, SEARCH)).input_spec
    assert (spec["sentence_prefix"], spec["category_prefix"]) == (SEARCH_QUERY, SEARCH_DOCUMENT)
    assert EmbeddingSimilarityStrategy.condition_label(spec) == (
        "Embedding\nsimilarity\ndefinitions\nsentence: search result\ncategories: document")
    swapped = {**spec, "sentence_prefix": SEARCH_DOCUMENT, "category_prefix": SEARCH_QUERY}
    assert EmbeddingSimilarityStrategy.condition_label(swapped).endswith(
        "\nsentence: document\ncategories: search result")


@pytest.mark.parametrize("prefix, name", [
    ("task: sentence similarity | query: ", "sentence similarity"),
    ("title: none | text: ", "document"),
    ("title: Data Retention | text: ", "document"),
    ("query: ", "query:"),
])
def test_instruction_names_come_from_the_prefix(prefix, name):
    assert embedding_similarity.instruction_name(prefix) == name


class FakeEmbedder:
    """Embeds each category as a one-hot vector and each sentence by a fixed lookup."""

    def __init__(self, sentence_vector):
        self.sentence_vector, self.calls = sentence_vector, []

    def load(self):
        return {"backend": "fake", "load_seconds": 0.1}

    def encode(self, text, prompt):
        self.calls.append((text, prompt))
        return {"text": text, "input_ids": [1, 2, 3]}

    def embed(self, encoded):
        text = encoded["text"]
        if text.split(":")[0] in LABELS:
            vector = torch.zeros(10)
            vector[LABELS.index(text.split(":")[0])] = 2.0
        else:
            vector = self.sentence_vector
        return {"embedding": vector, "input_tokens": 3, "input_token_ids": [1, 2, 3], "seconds": 0.01}


@pytest.mark.parametrize("vector, expected", [
    (torch.tensor([0.1, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 3),
    (torch.ones(10), 0),
])
def test_the_most_similar_category_wins_and_ties_go_to_the_first(make_condition, vector, expected):
    strategy = EmbeddingSimilarityStrategy(make_condition("embedding-similarity", MODEL, SEARCH))
    fake = FakeEmbedder(vector)
    identity = {"model": "o/m", "revision": COMMIT, "resolved_revision": COMMIT, "local_snapshot": "x"}
    with patch.object(embedding_similarity, "resolve_snapshot", return_value="snapshot"), \
            patch.object(embedding_similarity, "snapshot_identity", return_value=identity), \
            patch.object(embedding_similarity, "SentenceEmbedder", return_value=fake) as constructor:
        metadata = strategy.load()
    assert constructor.call_args.kwargs == {"precision": "bf16", "device": "cuda", "max_input_tokens": 8192,
                                            "text_only": True}
    assert [call[1] for call in fake.calls] == [SEARCH_DOCUMENT] * 10
    assert metadata["category_embeddings"]["input_tokens"] == [3] * 10
    response = strategy.predict(strategy.prepare(Row("x", "We keep data.", (3,))))
    assert fake.calls[-1] == ("We keep data.", SEARCH_QUERY)
    assert json.loads(response["raw_output"]) == {"label_id": expected}
    assert response["model_label"] == LABELS[expected] and len(response["similarities"]) == 10
    assert max(response["similarities"]) == pytest.approx(response["similarities"][expected])
    assert "embedding" not in response and response["input_tokens"] == 3


def test_guidance_has_no_place_in_category_embeddings():
    with pytest.raises(ValueError, match="guidance"):
        embedding_similarity.category_texts({"name": "g", "definitions": {}, "guidance": ["Prefer X."]})
    assert embedding_similarity.category_texts({"name": "l", "definitions": {}, "guidance": []}) == list(LABELS)


def test_download_caches_the_pinned_commit(make_condition, tmp_path):
    strategy = EmbeddingSimilarityStrategy(make_condition("embedding-similarity", MODEL, OPTIONS))
    snapshot = tmp_path / "models--o--m" / "snapshots" / COMMIT
    with patch.object(embedding_similarity, "download_snapshot", return_value=snapshot) as download:
        assert strategy.download()["resolved_revision"] == COMMIT
    download.assert_called_once_with("o/m", COMMIT)


def test_similarities_are_cosines_of_renormalized_float32_vectors(make_condition):
    strategy = EmbeddingSimilarityStrategy(make_condition("embedding-similarity", MODEL, OPTIONS))
    strategy._categories = torch.nn.functional.normalize(torch.eye(10), dim=1)
    strategy._backend = SimpleNamespace(embed=lambda encoded: {"embedding": torch.tensor([3.0, 4.0] + [0.0] * 8),
                                                               "seconds": 0.0})
    response = strategy.predict({"text_sha256": "x", "encoded": {}})
    assert response["similarities"][:2] == pytest.approx([0.6, 0.8])
