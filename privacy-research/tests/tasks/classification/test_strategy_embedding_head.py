"""The embedding-head strategy: fit record checks, input spec, and scores from a fitted linear head."""

import functools
import json
from unittest.mock import patch

import numpy as np
import pytest

from privacy_bench.core.files import digest_bytes, read_json
from privacy_bench.core.fits import resolve_fit
from privacy_bench.tasks.classification.data import Row
from privacy_bench.tasks.classification.strategies import STRATEGIES, embedding_head
from privacy_bench.tasks.classification.strategies.embedding_head import EmbeddingHeadStrategy
from privacy_bench.tasks.classification.taxonomy import LABELS

COMMIT = "a" * 40
PREFIX = "task: classification | query: "
GGUF_COMMIT = "c" * 40
MODEL = {"id": "o/m-GGUF", "revision": GGUF_COMMIT, "precision": "q8_0", "file": "m-Q8_0.gguf", "device": "cpu",
         "converted_from": "o/m", "converted_from_revision": COMMIT}
FIT_SPEC = (f'task = "classification"\nkind = "embedding-heads"\n[model]\nid = "o/m"\nrevision = "{COMMIT}"\n'
            'precision = "bf16"\ntext_only = true\n')
REFERENCE = "classification/toy-heads/20261012-000000Z"
SHIFT_REFERENCE = "classification/toy-shift/20261012-000000Z"


def head_tensors(name="linear"):
    """A head that follows embedding dimension 0 for category 3 and dimension 1 for category 5."""
    weight = np.zeros((len(LABELS), 4), dtype=np.float32)
    weight[3, 0] = weight[5, 1] = 1.0
    bias = np.zeros(len(LABELS), dtype=np.float32)
    bias[9] = 0.1
    return {f"{name}.weight": weight, f"{name}.bias": bias}


@pytest.fixture
def fits_root(tmp_path, monkeypatch, write_fit):
    """Write a stub fit record under a temporary results/fits/ and point the strategy at it."""
    def write(spec_text=FIT_SPEC, name="toy-heads", report=None, tensors=None):
        report = report or {"prefix": PREFIX,
                            "examples": {"examples": {"name": "toy", "manifest_sha256": "e" * 64, "n": 30}},
                            "heads": {"linear": {"method": "logistic", "class_weight": "balanced", "c": 1.0,
                                                 "cv_balanced_accuracy": 0.5}}}
        return write_fit(tmp_path / "classification" / name / "20261012-000000Z", name, spec_text, report,
                         tensors or head_tensors())
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
    tensors = head_tensors()
    assert spec["head"]["weight_sha256"] == digest_bytes(tensors["linear.weight"].tobytes())
    assert spec["head"]["bias_sha256"] == digest_bytes(tensors["linear.bias"].tobytes())
    assert EmbeddingHeadStrategy.condition_label(spec) == "Embedding\nhead\nlinear"


@pytest.mark.parametrize("model, settings, message", [
    (MODEL, {key: value for key, value in options().items() if key != "head"}, "missing: head"),
    (MODEL, {key: value for key, value in options().items() if key != "fit"}, "missing: fit"),
    (MODEL, options(fit="toy-heads"), "fit reference"),
    (MODEL, options(fit="classification/toy-heads/20991231-000000Z"), "No fit record"),
    (MODEL, options(head="nope"), "has no head 'nope'; it has linear"),
    ({**MODEL, "converted_from": "o/other"}, options(), "differs from the model .* used: converted_from$"),
    ({**MODEL, "converted_from_revision": "b" * 40}, options(), "used: converted_from_revision$"),
    ({**MODEL, "precision": "q4_k_m"}, options(), "used: precision"),
    ({**MODEL, "converted_from_revision": "main"}, options(), "full 40-character commit"),
    ({key: value for key, value in MODEL.items() if key != "converted_from"}, options(), "missing: converted_from"),
    ({key: value for key, value in MODEL.items() if key != "device"}, options(), "missing: device"),
    ({**MODEL, "text_only": True}, options(), "unknown keys: text_only"),
])
def test_invalid_conditions_fail_before_any_model_loads(make_condition, fits_root, model, settings, message):
    with pytest.raises(ValueError, match=message):
        EmbeddingHeadStrategy(make_condition("embedding-head", model, settings))


def test_a_condition_of_the_retired_sentence_transformers_backend_is_refused(make_condition, fits_root):
    checkpoint = {"id": "o/m", "revision": COMMIT, "precision": "bf16", "text_only": True}
    with pytest.raises(ValueError, match="'test' .model. names no GGUF file.*retired.*README.md"):
        EmbeddingHeadStrategy(make_condition("embedding-head", checkpoint, options()))


def test_a_file_on_an_fp32_fit_is_refused(make_condition, fits_root):
    fits_root(FIT_SPEC.replace('"bf16"', '"fp32"'), name="fp32-heads")
    with pytest.raises(ValueError, match="used: precision"):
        EmbeddingHeadStrategy(make_condition("embedding-head", MODEL,
                                             options(fit="classification/fp32-heads/20261012-000000Z")))


def test_both_file_precisions_use_the_bf16_fit_and_record_the_same_input(make_condition, fits_root):
    q8, bf16 = (EmbeddingHeadStrategy(make_condition("embedding-head", {**MODEL, "precision": precision}, options()))
                for precision in ("q8_0", "bf16"))
    assert (q8.model["precision"], bf16.model["precision"]) == ("q8_0", "bf16")
    # Only [model] differs, so both runs record the same input.
    assert q8.input_spec == bf16.input_spec


def test_fits_of_another_kind_are_refused(make_condition, fits_root):
    fits_root(FIT_SPEC.replace("embedding-heads", "other-kind"), name="other")
    with pytest.raises(ValueError, match="'other-kind' fit, not 'embedding-heads' or 'prior-shift'"):
        EmbeddingHeadStrategy(make_condition("embedding-head", MODEL,
                                             options(fit="classification/other/20261012-000000Z")))


def test_a_shifted_head_takes_its_model_and_prefix_from_its_verified_base(make_condition, fits_root, tmp_path):
    base = read_json(tmp_path / "classification" / "toy-heads" / "20261012-000000Z" / "fit.json")
    report = {"base": {"reference": REFERENCE, "fit_sha256": base["fit_sha256"],
                       "weights_sha256": base["weights_sha256"]},
              "tuning": {"sample": {"name": "toy-tuning", "n": 12}},
              "heads": {"linear-tuned": {"method": "logistic", "class_weight": "balanced", "c": 1.0,
                                         "base_head": "linear", "strength": 0.5, "strength_choice": "tuned"}}}
    shift_spec = f'task = "classification"\nkind = "prior-shift"\n[base]\nfit = "{REFERENCE}"\n'
    fits_root(shift_spec, name="toy-shift", report=report, tensors=head_tensors("linear-tuned"))
    shifted = options(fit=SHIFT_REFERENCE, head="linear-tuned")
    spec = EmbeddingHeadStrategy(make_condition("embedding-head", MODEL, shifted)).input_spec
    assert spec["prefix"] == PREFIX and spec["fit"]["examples"]["n"] == 30
    assert spec["fit"]["base"] == report["base"] and spec["fit"]["tuning"] == report["tuning"]
    assert EmbeddingHeadStrategy.condition_label(spec) == "Embedding\nhead\nlinear\nprior shift 0.5"
    # The base is the fit the shift recorded, and the file must come from the checkpoint the base embedded with.
    with pytest.raises(ValueError, match="used: converted_from$"):
        EmbeddingHeadStrategy(make_condition("embedding-head", {**MODEL, "converted_from": "o/other"}, shifted))
    fits_root(shift_spec, name="toy-moved", report={**report, "base": {**report["base"], "weights_sha256": "0" * 64}},
              tensors=head_tensors("linear-tuned"))
    with pytest.raises(ValueError, match="differs from the one .* shifted: weights_sha256"):
        EmbeddingHeadStrategy(make_condition("embedding-head", MODEL,
                                             options(fit="classification/toy-moved/20261012-000000Z",
                                                     head="linear-tuned")))


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
        return {"embedding": np.array(self.VECTORS[encoded["text"]], dtype=np.float32), "input_tokens": 3,
                "seconds": 0.01}


@pytest.mark.parametrize("text, expected", [("third", 3), ("neither", 9), ("tie", 3)])
def test_the_one_file_loads_through_llama_cpp_and_the_head_scores_its_normalized_embedding(
        make_condition, fits_root, tmp_path, text, expected):
    strategy = EmbeddingHeadStrategy(make_condition("embedding-head", MODEL, options()))
    fake = FakeEmbedder()
    snapshot = tmp_path / "models--o--m-GGUF" / "snapshots" / GGUF_COMMIT
    with patch.object(embedding_head, "resolve_snapshot", return_value=snapshot) as located, \
            patch.object(embedding_head, "GgufSentenceEmbedder", return_value=fake) as constructor:
        metadata = strategy.load()
    located.assert_called_once_with("o/m-GGUF", GGUF_COMMIT, "m-Q8_0.gguf")
    constructor.assert_called_once_with(snapshot / "m-Q8_0.gguf", precision="q8_0", device="cpu", max_input_tokens=8192)
    assert (metadata["model"], metadata["backend"]) == ("o/m-GGUF", "fake")
    assert metadata["fit"] == REFERENCE and metadata["head"] == "linear"
    response = strategy.predict(strategy.prepare(Row("x", text, (3,))))
    assert fake.prompts == [PREFIX]
    assert json.loads(response["raw_output"]) == {"label_id": expected} and response["model_label"] == LABELS[expected]
    vector = np.array(FakeEmbedder.VECTORS[text], dtype=np.float32)
    assert response["scores"][3] == pytest.approx(float(vector[0] / np.linalg.norm(vector)))
    assert response["scores"][9] == pytest.approx(0.1) and all(type(score) is float for score in response["scores"])
    assert len(response["scores"]) == 10 and "embedding" not in response and response["input_tokens"] == 3


def test_download_caches_the_one_file_of_the_pinned_commit(make_condition, fits_root, tmp_path):
    strategy = EmbeddingHeadStrategy(make_condition("embedding-head", MODEL, options()))
    snapshot = tmp_path / "models--o--m-GGUF" / "snapshots" / GGUF_COMMIT
    with patch.object(embedding_head, "download_snapshot", return_value=snapshot) as download:
        assert strategy.download()["resolved_revision"] == GGUF_COMMIT
    download.assert_called_once_with("o/m-GGUF", GGUF_COMMIT, only="m-Q8_0.gguf")
