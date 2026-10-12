"""Embedding heads fitted on a tiny CPU sentence embedder and synthetic example sentences."""

import csv
import functools

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
import torch

from privacy_bench.core.files import digest_bytes, write_json
from privacy_bench.core.fits import execute_fit, load_fit, parse_fit_spec, resolve_fit
from privacy_bench.models.sentence_embedder import SentenceEmbedder
from privacy_bench.tasks.classification import fitting
from privacy_bench.tasks.classification.data import make_example_manifest, read_dataset
from privacy_bench.tasks.classification.fitting import EmbeddingHeadsFit, centroid_weights
from privacy_bench.tasks.classification.strategies import embedding_head
from privacy_bench.tasks.classification.strategies.embedding_head import EmbeddingHeadStrategy
from privacy_bench.tasks.classification.task import ClassificationTask
from privacy_bench.tasks.classification.taxonomy import HEADER, LABELS

WORDS = ("we", "keep", "data", "for", "years", ".")
PREFIX = "task: classification | query: "
SPEC = f"""task = "classification"
kind = "embedding-heads"
[examples]
dataset = "toy-sentences"
examples = "toy-examples"
[model]
id = "o/m"
revision = "{'a' * 40}"
precision = "fp32"
device = "cpu"
[options]
prefix = "{PREFIX}"
max_input_tokens = 32
folds = 2
c_values = [0.1, 1.0, 10.0]
[heads.centroid]
method = "centroid"
[heads.linear-balanced]
method = "logistic"
class_weight = "balanced"
[heads.linear-frequencies]
method = "logistic"
class_weight = "none"
"""


def sentence(label, example):
    """Thirty distinct sentences; a repeat across categories would tie their centroids exactly."""
    stride = 1 + label // len(WORDS)
    return " ".join(WORDS[(label + stride * step) % len(WORDS)] for step in range(example + 2 + label % 3))


@pytest.fixture
def toy_examples(tmp_path, tiny_embedder, monkeypatch):
    """Write three examples per category and point the fitter's data and model lookups at them."""
    def install(categories=range(len(LABELS))):
        path = tmp_path / "toy-sentences.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            for label in categories:
                for example in range(3):
                    writer.writerow([f"{label}-{example}", sentence(label, example),
                                     *[int(i == label) for i in range(len(LABELS))]])
        dataset = read_dataset(path, allow_target=True)
        examples = tmp_path / "toy-examples.json"
        write_json(examples, make_example_manifest(dataset, []))
        files = {("datasets", "toy-sentences"): path, ("classification", "examples", "toy-examples"): examples}
        monkeypatch.setattr(fitting, "data_file", lambda *parts, suffix: files[parts])
        monkeypatch.setattr(fitting, "resolve_snapshot", lambda model_id, revision: tiny_embedder)
        monkeypatch.setattr(fitting, "snapshot_identity",
                            lambda model_id, revision, snapshot: {"model": model_id, "revision": revision})
        return dataset
    return install


def reference_embeddings(snapshot, rows):
    embedder = SentenceEmbedder(snapshot, precision="fp32", device="cpu", max_input_tokens=32, text_only=False)
    embedder.load()
    vectors = [embedder.embed(embedder.encode(row.text, PREFIX))["embedding"] for row in rows]
    return torch.nn.functional.normalize(torch.stack(vectors), dim=1).numpy()


def test_heads_match_their_definitions_and_refit_identically(toy_examples, tiny_embedder):
    dataset = toy_examples()
    fitted = EmbeddingHeadsFit(parse_fit_spec("toy-heads", SPEC)).run()
    again = EmbeddingHeadsFit(parse_fit_spec("toy-heads", SPEC)).run()
    assert sorted(fitted["tensors"]) == sorted(again["tensors"]) == sorted(
        f"{head}.{part}" for head in ("centroid", "linear-balanced", "linear-frequencies") for part in ("weight", "bias"))
    for name, value in fitted["tensors"].items():
        assert value.dtype == np.float32 and np.array_equal(value, again["tensors"][name]), name
    embeddings = reference_embeddings(tiny_embedder, dataset.rows)
    labels = np.array([row.label_id for row in dataset.rows])
    report = fitted["report"]
    assert report["embeddings"]["sha256"] == digest_bytes(embeddings.tobytes())
    np.testing.assert_allclose(fitted["tensors"]["centroid.weight"], centroid_weights(embeddings, labels))
    assert not fitted["tensors"]["centroid.bias"].any()
    for name, weights in (("linear-balanced", "balanced"), ("linear-frequencies", None)):
        head = report["heads"][name]
        assert head["c"] in (0.1, 1.0, 10.0) and [entry["c"] for entry in head["cv_by_c"]] == [0.1, 1.0, 10.0]
        expected = LogisticRegression(C=head["c"], class_weight=weights, max_iter=5000).fit(embeddings, labels)
        np.testing.assert_allclose(fitted["tensors"][f"{name}.weight"], expected.coef_, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(fitted["tensors"][f"{name}.bias"], expected.intercept_, rtol=1e-5, atol=1e-6)
    assert report["examples"]["label_counts"] == {label: 3 for label in LABELS}
    assert report["examples"]["examples"]["n"] == 30 and report["prefix"] == PREFIX
    assert report["cross_validation"]["folds"] == 2 and 0 <= report["heads"]["centroid"]["cv_balanced_accuracy"] <= 1


def test_the_task_fits_through_the_shared_fit_record(toy_examples, tmp_path):
    toy_examples()
    output = tmp_path / "record"
    execute_fit(ClassificationTask(), parse_fit_spec("toy-heads", SPEC), output)
    fit = load_fit(output)
    assert fit.record["kind"] == "embedding-heads" and fit.tensors["linear-balanced.weight"].shape == (10, 16)


def test_the_strategy_reproduces_each_fitted_head_on_its_examples(toy_examples, tiny_embedder, tmp_path,
                                                                 make_condition, monkeypatch):
    dataset = toy_examples()
    reference = "classification/toy-heads/20261012-000000Z"
    execute_fit(ClassificationTask(), parse_fit_spec("toy-heads", SPEC), tmp_path.joinpath("fits", *reference.split("/")))
    monkeypatch.setattr(embedding_head, "resolve_fit", functools.partial(resolve_fit, root=tmp_path / "fits"))
    monkeypatch.setattr(embedding_head, "resolve_snapshot", lambda model_id, revision: tiny_embedder)
    monkeypatch.setattr(embedding_head, "snapshot_identity",
                        lambda model_id, revision, snapshot: {"model": model_id, "revision": revision})
    embeddings = reference_embeddings(tiny_embedder, dataset.rows)
    labels = np.array([row.label_id for row in dataset.rows])
    chosen_c = load_fit(tmp_path.joinpath("fits", *reference.split("/"))).record["report"]["heads"]
    expected = {
        "centroid": (embeddings @ centroid_weights(embeddings, labels).T).argmax(axis=1),
        "linear-balanced": LogisticRegression(C=chosen_c["linear-balanced"]["c"], class_weight="balanced",
                                              max_iter=5000).fit(embeddings, labels).predict(embeddings),
    }
    model = {"id": "o/m", "revision": "a" * 40, "precision": "fp32", "device": "cpu"}
    for head, predictions in expected.items():
        strategy = EmbeddingHeadStrategy(make_condition("embedding-head", model,
                                                        {"fit": reference, "head": head, "max_input_tokens": 32}))
        strategy.load()
        assert [strategy.predict(strategy.prepare(row))["model_label_id"] for row in dataset.rows] == \
            predictions.tolist(), head


def test_examples_over_the_input_limit_stop_the_fit_before_embedding(toy_examples):
    toy_examples()
    with pytest.raises(ValueError, match="Example preflight failed:\nID 1-2: Input has 13 tokens") as raised:
        EmbeddingHeadsFit(parse_fit_spec("toy-heads", SPEC.replace("max_input_tokens = 32",
                                                                    "max_input_tokens = 12"))).run()
    # Every example over the limit is listed, not only the first.
    assert str(raised.value).count("exceeds the 12-token limit") == 9


def test_every_category_needs_examples(toy_examples):
    toy_examples(categories=range(9))
    with pytest.raises(ValueError, match="none for: Other"):
        EmbeddingHeadsFit(parse_fit_spec("toy-heads", SPEC)).run()


@pytest.mark.parametrize("old, new, message", [
    ("[examples]", "[extra]\nx = 1\n[examples]", "unknown tables: extra"),
    ('examples = "toy-examples"\n', "", "missing: examples"),
    (f'revision = "{"a" * 40}"', 'revision = "main"', "40-character"),
    (f'prefix = "{PREFIX}"', 'prefix = " "', "prefix"),
    ("folds = 2", "folds = 1", "folds"),
    ("folds = 2", "folds = 2\nseed = 1.5", "seed"),
    ("c_values = [0.1, 1.0, 10.0]", "c_values = []", "c_values"),
    ("c_values = [0.1, 1.0, 10.0]", "c_values = [0.1, -1.0]", "c_values"),
    ("c_values = [0.1, 1.0, 10.0]", "c_values = [true]", "c_values"),
    ('method = "centroid"', 'method = "centroid"\nclass_weight = "balanced"', "takes no class_weight"),
    ('class_weight = "none"', 'class_weight = "inverse"', "needs class_weight"),
    ('[heads.linear-frequencies]\nmethod = "logistic"\nclass_weight = "none"\n',
     '[heads.linear-frequencies]\nmethod = "svm"\n', "'centroid' or 'logistic'"),
    ("[heads.centroid]", "[heads.Centroid]", "lowercase name"),
    ("[heads.centroid]", "[heads.centroid.x]", "unknown keys"),
])
def test_invalid_fit_files_fail_before_any_model_loads(old, new, message):
    assert old in SPEC
    with pytest.raises(ValueError, match=message):
        EmbeddingHeadsFit(parse_fit_spec("toy-heads", SPEC.replace(old, new, 1)))


def test_fit_files_need_heads_and_a_known_kind():
    without_heads = SPEC.split("[heads.centroid]")[0]
    with pytest.raises(ValueError, match="at least one"):
        EmbeddingHeadsFit(parse_fit_spec("toy-heads", without_heads))
    with pytest.raises(ValueError, match="Unknown classification fit kind 'magic'"):
        ClassificationTask().fit(parse_fit_spec("toy-heads", SPEC.replace("embedding-heads", "magic")))
