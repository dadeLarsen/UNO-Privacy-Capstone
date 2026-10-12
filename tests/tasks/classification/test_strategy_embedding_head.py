"""The embedding-head strategy: fit record checks, input spec, and scores from a fitted linear head."""

import functools
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from privacy_bench.core.files import digest_bytes
from privacy_bench.core.fits import execute_fit, parse_fit_spec, resolve_fit
from privacy_bench.tasks.classification.data import Row
from privacy_bench.tasks.classification.strategies import STRATEGIES, embedding_head
from privacy_bench.tasks.classification.strategies.embedding_head import EmbeddingHeadStrategy
from privacy_bench.tasks.classification.taxonomy import LABELS

COMMIT = "a" * 40
PREFIX = "task: classification | query: "
MODEL = {"id": "o/m", "revision": COMMIT, "precision": "bf16", "text_only": True}
FIT_SPEC = (f'task = "classification"\nkind = "embedding-heads"\n[model]\nid = "o/m"\nrevision = "{COMMIT}"\n'
            'precision = "bf16"\ntext_only = true\n')
REFERENCE = "classification/toy-heads/20261012-000000Z"


class StubHeads:
    """A fit whose head follows embedding dimension 0 for category 3 and dimension 1 for category 5."""

    name = "classification"

    def fit(self, spec):
        weight = np.zeros((len(LABELS), 4), dtype=np.float32)
        weight[3, 0] = weight[5, 1] = 1.0
        bias = np.zeros(len(LABELS), dtype=np.float32)
        bias[9] = 0.1
        return {"report": {"prefix": PREFIX, "examples": {"examples": {"name": "toy", "manifest_sha256": "e" * 64,
                                                                         "n": 30}},
                           "heads": {"linear": {"method": "logistic", "class_weight": "balanced", "c": 1.0,
                                                "cv_balanced_accuracy": 0.5}}},
                "tensors": {"linear.weight": weight, "linear.bias": bias}}


@pytest.fixture
def fits_root(tmp_path, monkeypatch):
    """Write a stub fit record under a temporary results/fits/ and point the strategy at it."""
    def write(spec_text=FIT_SPEC, name="toy-heads"):
        execute_fit(StubHeads(), parse_fit_spec(name, spec_text), tmp_path / "classification" / name / "20261012-000000Z")
    monkeypatch.setattr(embedding_head, "resolve_fit", functools.partial(resolve_fit, root=tmp_path))
    write()
    return write


def options(**changes):
    return {"fit": REFERENCE, "head": "linear", "max_input_tokens": 8192, **changes}


def test_registered_under_its_condition_name():
    assert STRATEGIES["embedding-head"] is EmbeddingHeadStrategy


def test_input_spec_takes_the_prefix_and_head_from_the_fit(make_condition, fits_root):
    spec = EmbeddingHeadStrategy(make_condition("embedding-head", MODEL, options())).input_spec
    assert spec["prefix"] == PREFIX and spec["truncation"] is False and spec["labels"] == list(LABELS)
    assert spec["fit"]["reference"] == REFERENCE and spec["fit"]["examples"]["n"] == 30
    assert {key: spec["head"][key] for key in ("name", "method", "class_weight", "c")} == {
        "name": "linear", "method": "logistic", "class_weight": "balanced", "c": 1.0}
    weight = np.zeros((len(LABELS), 4), dtype=np.float32)
    weight[3, 0] = weight[5, 1] = 1.0
    assert spec["head"]["weight_sha256"] == digest_bytes(weight.tobytes())
    assert EmbeddingHeadStrategy.condition_label(spec) == "Embedding\nhead\nlinear"


@pytest.mark.parametrize("model, settings, message", [
    (MODEL, {key: value for key, value in options().items() if key != "head"}, "missing: head"),
    (MODEL, {key: value for key, value in options().items() if key != "fit"}, "missing: fit"),
    (MODEL, options(fit="toy-heads"), "fit reference"),
    (MODEL, options(fit="classification/toy-heads/20991231-000000Z"), "No fit record"),
    (MODEL, options(head="nope"), "has no head 'nope'; it has linear"),
    ({**MODEL, "revision": "b" * 40}, options(), "differs from the model .* used: revision"),
    ({**MODEL, "precision": "fp32"}, options(), "used: precision"),
    ({**MODEL, "text_only": False}, options(), "used: text_only"),
    ({**MODEL, "id": "o/other"}, options(), "used: id"),
])
def test_invalid_conditions_fail_before_any_model_loads(make_condition, fits_root, model, settings, message):
    with pytest.raises(ValueError, match=message):
        EmbeddingHeadStrategy(make_condition("embedding-head", model, settings))


def test_fits_of_another_kind_are_refused(make_condition, fits_root):
    fits_root(FIT_SPEC.replace("embedding-heads", "other-kind"), name="other")
    with pytest.raises(ValueError, match="'other-kind' fit, not 'embedding-heads' or 'prior-shift'"):
        EmbeddingHeadStrategy(make_condition("embedding-head", MODEL,
                                             options(fit="classification/other/20261012-000000Z")))


class FakeEmbedder:
    VECTORS = {"third": [2.0, 0, 0, 0], "neither": [0, 0, 3.0, 0], "tie": [1.0, 1.0, 0, 0]}

    def __init__(self):
        self.prompts = []

    def load(self):
        return {"backend": "fake", "load_seconds": 0.1}

    def encode(self, text, prompt):
        self.prompts.append(prompt)
        return {"text": text, "input_ids": [1, 2, 3]}

    def embed(self, encoded):
        return {"embedding": torch.tensor(self.VECTORS[encoded["text"]]), "input_tokens": 3, "seconds": 0.01}


@pytest.mark.parametrize("text, expected", [("third", 3), ("neither", 9), ("tie", 3)])
def test_scores_are_the_head_applied_to_the_normalized_embedding(make_condition, fits_root, text, expected):
    strategy = EmbeddingHeadStrategy(make_condition("embedding-head", MODEL, options()))
    fake = FakeEmbedder()
    identity = {"model": "o/m", "revision": COMMIT, "resolved_revision": COMMIT, "local_snapshot": "x"}
    with patch.object(embedding_head, "resolve_snapshot", return_value="snapshot"), \
            patch.object(embedding_head, "snapshot_identity", return_value=identity), \
            patch.object(embedding_head, "SentenceEmbedder", return_value=fake) as constructor:
        metadata = strategy.load()
    assert constructor.call_args.kwargs == {"precision": "bf16", "device": "cuda", "max_input_tokens": 8192,
                                            "text_only": True}
    assert metadata["fit"] == REFERENCE and metadata["head"] == "linear"
    response = strategy.predict(strategy.prepare(Row("x", text, (3,))))
    assert fake.prompts == [PREFIX]
    assert json.loads(response["raw_output"]) == {"label_id": expected} and response["model_label"] == LABELS[expected]
    vector = torch.nn.functional.normalize(torch.tensor(FakeEmbedder.VECTORS[text]), dim=0)
    assert response["scores"][3] == pytest.approx(float(vector[0])) and response["scores"][9] == pytest.approx(0.1)
    assert len(response["scores"]) == 10 and "embedding" not in response and response["input_tokens"] == 3


def test_download_caches_the_pinned_commit(make_condition, fits_root, tmp_path):
    strategy = EmbeddingHeadStrategy(make_condition("embedding-head", MODEL, options()))
    snapshot = tmp_path / "models--o--m" / "snapshots" / COMMIT
    with patch.object(embedding_head, "download_snapshot", return_value=snapshot) as download:
        assert strategy.download()["resolved_revision"] == COMMIT
    download.assert_called_once_with("o/m", COMMIT)
