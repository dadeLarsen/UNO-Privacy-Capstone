"""Embedding heads, and prior shifts of them, fitted on a tiny CPU sentence embedder and synthetic sentences."""

import csv
import functools

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
import torch

from privacy_bench.core.files import digest_bytes, read_json, write_json
from privacy_bench.core.fits import execute_fit, load_fit, parse_fit_spec, resolve_fit
from privacy_bench.models.sentence_embedder import SentenceEmbedder
from privacy_bench.tasks.classification import fitting
from privacy_bench.tasks.classification.data import (
    make_example_manifest, make_half_manifests, read_dataset, select_rows,
)
from privacy_bench.tasks.classification.fitting import (
    EmbeddingHeadsFit, PriorShiftFit, accepted_shares, centroid_weights, training_shares,
)
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


BASE_REFERENCE = "classification/toy-heads/20261012-000000Z"
SHIFT_REFERENCE = "classification/toy-shift/20261012-000000Z"
STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
SHIFT_SPEC = f"""task = "classification"
kind = "prior-shift"
[base]
fit = "{BASE_REFERENCE}"
[tuning]
dataset = "toy-tuning"
sample = "toy-tuning-half"
gold = "toy-gold"
[options]
strengths = {STRENGTHS}
[heads.balanced-tuned]
base = "linear-balanced"
strength = "tuned"
[heads.frequencies-tuned]
base = "linear-frequencies"
strength = "tuned"
[heads.balanced-half]
base = "linear-balanced"
strength = 0.5
"""


def write_target(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for row_id, text, labels in rows:
            writer.writerow([row_id, text, *[int(i in labels) for i in range(len(LABELS))]])
    return path


@pytest.fixture
def toy_tuning(toy_examples, tmp_path, monkeypatch):
    """Fit the toy heads as a base, then write a tuning half whose mix leans on First Party Collection/Use.

    Sixty tuning sentences repeat an Other example's text but are labeled First Party, so shifting toward
    First Party gains accuracy. The accepted labels of the first tuning sentence add Other.
    """
    toy_examples()
    execute_fit(ClassificationTask(), parse_fit_spec("toy-heads", SPEC),
                tmp_path.joinpath("fits", *BASE_REFERENCE.split("/")))
    monkeypatch.setattr(fitting, "resolve_fit", functools.partial(resolve_fit, root=tmp_path / "fits"))

    def install(skip=()):
        drawn = [(sentence(label, example), label) for label in range(len(LABELS)) if label not in skip
                 for example in range(3)]
        drawn += [(sentence(9, example % 3), 8) for example in range(60)]
        rows = [(str(index), text, (label,)) for index, (text, label) in enumerate(drawn, 2)]
        # ID 1 restarts the rising IDs, so it alone forms the test half.
        rows.append(("1", sentence(0, 0), (0,)))
        dataset_path = write_target(tmp_path / "toy-tuning.csv", rows)
        gold_path = write_target(tmp_path / "toy-gold.csv", [(row_id, text, (*labels, 9) if row_id == "2" else labels)
                                                             for row_id, text, labels in rows])
        dataset = read_dataset(dataset_path, allow_target=True)
        files = {("datasets", "toy-tuning"): dataset_path, ("classification", "gold", "toy-gold"): gold_path}
        for half, manifest in make_half_manifests(dataset).items():
            files[("classification", "samples", f"toy-{half}-half")] = tmp_path / f"toy-{half}-half.json"
            write_json(files[("classification", "samples", f"toy-{half}-half")], manifest)
        monkeypatch.setattr(fitting, "data_file", lambda *parts, suffix: files[parts])
        return dataset
    return install


def tuning_rows(dataset):
    rows = select_rows(dataset, make_half_manifests(dataset)["tuning"])
    accepted = {row.id: frozenset(row.labels) | ({9} if row.id == "2" else set()) for row in rows}
    return rows, accepted


def test_shifted_heads_add_the_scaled_log_share_ratio_and_tune_the_strength_on_tuning_accuracy(
        toy_tuning, tiny_embedder, tmp_path):
    dataset = toy_tuning()
    fitted = PriorShiftFit(parse_fit_spec("toy-shift", SHIFT_SPEC)).run()
    base = load_fit(tmp_path.joinpath("fits", *BASE_REFERENCE.split("/")))
    rows, accepted = tuning_rows(dataset)
    report = fitted["report"]
    # 90 tuning sentences: three per category plus sixty more for First Party. The first one splits its
    # count between Policy Change and Other.
    counts = np.full(len(LABELS), 3.0)
    counts[8] += 60
    counts[0] -= 0.5
    counts[9] += 0.5
    target = counts / counts.sum()
    np.testing.assert_allclose(accepted_shares(rows, accepted), target)
    np.testing.assert_allclose(list(report["target_shares"].values()), target)
    assert report["tuning"]["sample"]["n"] == 90 and report["base"]["reference"] == BASE_REFERENCE
    embeddings = reference_embeddings(tiny_embedder, rows)
    for name, base_name in (("balanced-tuned", "linear-balanced"), ("frequencies-tuned", "linear-frequencies"),
                            ("balanced-half", "linear-balanced")):
        weight, bias = base.tensors[f"{base_name}.weight"], base.tensors[f"{base_name}.bias"]
        # The toy examples hold three sentences per category, so both heads were trained on an even mix.
        shift = np.log(target) - np.log(np.full(len(LABELS), 0.1))
        accuracies = []
        for strength in STRENGTHS:
            predictions = (embeddings @ weight.T + (bias + strength * shift).astype(np.float32)).argmax(axis=1)
            accuracies.append(np.mean([label in accepted[row.id] for row, label in zip(rows, predictions)]))
        assert [entry["accuracy"] for entry in report["tuning_scores"][base_name]] == pytest.approx(accuracies)
        head = report["heads"][name]
        expected = 0.5 if name == "balanced-half" else STRENGTHS[int(np.argmax(accuracies))]
        assert (head["base_head"], head["strength"]) == (base_name, expected)
        assert head["strength_choice"] == ("fixed" if name == "balanced-half" else "tuned")
        np.testing.assert_array_equal(fitted["tensors"][f"{name}.weight"], weight)
        np.testing.assert_allclose(fitted["tensors"][f"{name}.bias"], bias + expected * shift, rtol=1e-6, atol=1e-6)
    # The toy data must make the shift matter, or this test could not tell a tuned strength from zero.
    assert report["heads"]["balanced-tuned"]["strength"] > 0


def test_training_shares_are_even_for_balanced_heads_and_the_example_mix_otherwise():
    counts = {label: 2 * (i + 1) for i, label in enumerate(LABELS)}
    np.testing.assert_allclose(training_shares({"class_weight": "balanced"}, counts), np.full(len(LABELS), 0.1))
    np.testing.assert_allclose(training_shares({"class_weight": "none"}, counts), np.arange(1, 11) / 55)


def test_the_strategy_applies_a_shifted_head_from_its_verified_base(toy_tuning, tiny_embedder, tmp_path,
                                                                    make_condition, monkeypatch):
    dataset = toy_tuning()
    output = tmp_path.joinpath("fits", *SHIFT_REFERENCE.split("/"))
    execute_fit(ClassificationTask(), parse_fit_spec("toy-shift", SHIFT_SPEC), output)
    monkeypatch.setattr(embedding_head, "resolve_fit", functools.partial(resolve_fit, root=tmp_path / "fits"))
    monkeypatch.setattr(embedding_head, "resolve_snapshot", lambda model_id, revision: tiny_embedder)
    monkeypatch.setattr(embedding_head, "snapshot_identity",
                        lambda model_id, revision, snapshot: {"model": model_id, "revision": revision})
    shifted = load_fit(output)
    rows, accepted = tuning_rows(dataset)
    model = {"id": "o/m", "revision": "a" * 40, "precision": "fp32", "device": "cpu"}
    settings = {"fit": SHIFT_REFERENCE, "head": "balanced-tuned", "max_input_tokens": 32}
    strategy = EmbeddingHeadStrategy(make_condition("embedding-head", model, settings))
    strategy.load()
    predictions = [strategy.predict(strategy.prepare(row))["model_label_id"] for row in rows]
    head = shifted.record["report"]["heads"]["balanced-tuned"]
    chosen = next(entry for entry in shifted.record["report"]["tuning_scores"]["linear-balanced"]
                  if entry["strength"] == head["strength"])
    assert np.mean([label in accepted[row.id] for row, label in zip(rows, predictions)]) == \
        pytest.approx(chosen["accuracy"])
    spec = strategy.input_spec
    assert spec["prefix"] == PREFIX and spec["fit"]["base"]["reference"] == BASE_REFERENCE
    assert spec["fit"]["tuning"]["sample"]["name"] == "toy-tuning-half" and spec["fit"]["examples"]["n"] == 30
    assert EmbeddingHeadStrategy.condition_label(spec) == \
        f"Embedding\nhead\nlinear-balanced\nprior shift {head['strength']:g}"
    # A base fit that no longer matches the hashes the shift recorded is refused.
    record = read_json(output / "fit.json")
    record["report"]["base"]["weights_sha256"] = "0" * 64
    write_json(output / "fit.json", record)
    with pytest.raises(ValueError, match="differs from the one .* shifted: weights_sha256"):
        EmbeddingHeadStrategy(make_condition("embedding-head", model, settings))


@pytest.mark.parametrize("old, new, message", [
    ('base = "linear-balanced"\nstrength = 0.5', 'base = "centroid"\nstrength = 0.5', "needs log-odds"),
    ('base = "linear-balanced"\nstrength = 0.5', 'base = "nope"\nstrength = 0.5', "has no head 'nope'"),
    ('sample = "toy-tuning-half"', 'sample = "toy-test-half"', "never tunes on a sample that results are reported"),
])
def test_shifts_need_logistic_base_heads_and_a_tuning_sample(toy_tuning, old, new, message):
    toy_tuning()
    assert old in SHIFT_SPEC
    with pytest.raises(ValueError, match=message):
        PriorShiftFit(parse_fit_spec("toy-shift", SHIFT_SPEC.replace(old, new))).run()


def test_every_category_needs_a_tuning_sentence(toy_tuning):
    toy_tuning(skip=(7,))
    with pytest.raises(ValueError, match="none for: Do Not Track"):
        PriorShiftFit(parse_fit_spec("toy-shift", SHIFT_SPEC)).run()


@pytest.mark.parametrize("old, new, message", [
    ("[tuning]", "[extra]\nx = 1\n[tuning]", "unknown tables: extra"),
    ('gold = "toy-gold"\n', "", "missing: gold"),
    (f'fit = "{BASE_REFERENCE}"', 'fit = "toy-heads"', "fit reference"),
    (f"strengths = {STRENGTHS}", "strengths = []", "strengths"),
    (f"strengths = {STRENGTHS}", "strengths = [0.5, 0.25]", "strengths"),
    (f"strengths = {STRENGTHS}", "strengths = [0.5, 0.5]", "strengths"),
    (f"strengths = {STRENGTHS}", "strengths = [0.0, 1.5]", "strengths"),
    (f"strengths = {STRENGTHS}", "strengths = [true]", "strengths"),
    ("strength = 0.5", "strength = 2.0", "'tuned' or a number from 0 through 1"),
    ("strength = 0.5", 'strength = "best"', "'tuned' or a number from 0 through 1"),
    ('base = "linear-balanced"\nstrength = 0.5', "base = 3\nstrength = 0.5", "must name a head"),
    ("[heads.balanced-half]", "[heads.Balanced]", "lowercase name"),
])
def test_invalid_shift_files_fail_before_any_model_loads(old, new, message):
    assert old in SHIFT_SPEC
    with pytest.raises(ValueError, match=message):
        PriorShiftFit(parse_fit_spec("toy-shift", SHIFT_SPEC.replace(old, new, 1)))
