"""Fit classification heads on frozen sentence embeddings of labeled example sentences.

Fit kind ``embedding-heads``::

    task = "classification"
    kind = "embedding-heads"

    [examples]
    dataset = "opp115-sentences"
    examples = "opp115-examples"

    [model]
    id = "google/embeddinggemma-2"
    revision = "914f7f89142e33e77833254d9c9b90c3cef7303b"
    precision = "bf16"
    text_only = true

    [options]
    prefix = "task: classification | query: "
    max_input_tokens = 8192
    c_values = [0.1, 1.0, 10.0]

    [heads.centroid]
    method = "centroid"

    [heads.linear-balanced]
    method = "logistic"
    class_weight = "balanced"

Every head is linear: a category's score is ``weight @ embedding + bias`` on the normalized embedding,
and the highest score wins. A centroid head's weights are the normalized mean embedding of each
category's examples with zero bias, so its scores are cosine similarities. A logistic head is a
multinomial logistic regression; stratified cross-validation on the examples, scored by balanced
accuracy, chooses its inverse regularization strength C from ``c_values``.

Fit kind ``prior-shift`` moves logistic heads of an ``embedding-heads`` fit toward the category mix of
a tuning sample, which no reported result may use::

    task = "classification"
    kind = "prior-shift"

    [base]
    fit = "classification/embeddinggemma2-opp115-heads/20261012-011902Z"

    [tuning]
    dataset = "appsi139-sentences"
    sample = "appsi139-tuning-half"
    gold = "appsi139-astra"

    [options]
    strengths = [0.0, 0.5, 1.0]

    [heads.linear-balanced-tuned]
    base = "linear-balanced"
    strength = "tuned"

Each shifted head keeps its base head's weights and adds ``strength * (log target share - log training
share)`` to its bias. A target share is a category's share of the tuning sentences, each split evenly
among its accepted labels; the training share is even for a balanced head and the examples' own mix
otherwise. A ``tuned`` strength is
the one in ``strengths`` with the highest accepted-label accuracy on the tuning sentences.
"""

import json
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import torch

from ...core.conditions import take_settings
from ...core.files import digest_bytes, read_json
from ...core.fits import load_fit, parse_fit_spec, resolve_fit
from ...core.paths import data_file
from ...models.hub import require_commit, resolve_snapshot, snapshot_identity
from ...models.sentence_embedder import SentenceEmbedder
from .data import (
    TUNING_PURPOSE, accepted_gold_digest, load_accepted_gold, read_dataset, select_examples, select_rows,
)
from .scoring import evaluate
from .taxonomy import LABELS

HEAD_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")
CLASS_WEIGHTS = {"balanced": "balanced", "none": None}
MAX_ITERATIONS = 5000


def centroid_weights(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The normalized mean embedding of each category, one row per category."""
    means = np.stack([embeddings[labels == i].mean(axis=0) for i in range(len(LABELS))])
    return (means / np.linalg.norm(means, axis=1, keepdims=True)).astype(np.float32)


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def embed_sentences(model: dict, prefix: str, max_input_tokens: int, rows, noun: str):
    """Embed each row once, after the prefix; returns normalized float32 embeddings, a summary, and the model.

    Every row is checked against the input limit before the first forward pass.
    """
    snapshot = resolve_snapshot(model["id"], model["revision"])
    embedder = SentenceEmbedder(snapshot, precision=model["precision"], device=model["device"],
                                max_input_tokens=max_input_tokens, text_only=model["text_only"])
    backend = embedder.load()
    encoded, errors = [], []
    for row in rows:
        try:
            encoded.append(embedder.encode(row.text, prefix))
        except ValueError as exc:
            errors.append(f"ID {row.id}: {exc}")
    if errors:
        raise ValueError(f"{noun.capitalize()} preflight failed:\n" + "\n".join(errors))
    vectors, seconds = [], 0.0
    for index, item in enumerate(encoded, 1):
        output = embedder.embed(item)
        vectors.append(output["embedding"])
        seconds += output["seconds"]
        if index % 1000 == 0 or index == len(encoded):
            print(f"Embedded {index}/{len(encoded)} {noun}s", flush=True)
    embeddings = torch.nn.functional.normalize(torch.stack(vectors), dim=1).numpy()
    tokens = [len(item["input_ids"]) for item in encoded]
    summary = {"n": len(rows), "dimension": int(embeddings.shape[1]),
               "sha256": digest_bytes(embeddings.tobytes()), "seconds": seconds,
               "min_input_tokens": min(tokens), "max_input_tokens": max(tokens)}
    return embeddings, summary, {**snapshot_identity(model["id"], model["revision"], snapshot), **backend}


class EmbeddingHeadsFit:
    kind = "embedding-heads"

    def __init__(self, spec):
        context = f"Fit {spec.name!r}"
        unknown = set(spec.tables) - {"examples", "model", "options", "heads"}
        if unknown:
            raise ValueError(f"{context} has unknown tables: {', '.join(sorted(unknown))}")
        self.name = spec.name
        self.examples = take_settings(spec.tables.get("examples", {}), f"{context} [examples]",
                                      required=("dataset", "examples"))
        self.model = take_settings(spec.tables.get("model", {}), f"{context} [model]",
                                   required=("id", "revision", "precision"),
                                   optional={"device": "cuda", "text_only": False})
        require_commit(self.model["id"], self.model["revision"])
        self.options = take_settings(spec.tables.get("options", {}), f"{context} [options]",
                                     required=("prefix", "max_input_tokens", "c_values"),
                                     optional={"folds": 5, "seed": 0})
        if not isinstance(self.options["prefix"], str) or not self.options["prefix"].strip():
            raise ValueError(f"{context}: prefix must be the model's nonempty instruction prefix")
        if type(self.options["folds"]) is not int or self.options["folds"] < 2:
            raise ValueError(f"{context}: folds must be an integer of at least 2")
        if type(self.options["seed"]) is not int:
            raise ValueError(f"{context}: seed must be an integer")
        c_values = self.options["c_values"]
        if not isinstance(c_values, list) or not c_values or any(not _number(c) or c <= 0 for c in c_values):
            raise ValueError(f"{context}: c_values must be a nonempty list of positive numbers")
        heads = spec.tables.get("heads", {})
        if not isinstance(heads, dict) or not heads:
            raise ValueError(f"{context} needs at least one [heads.<name>] table")
        self.heads = {}
        for name, table in heads.items():
            if not HEAD_NAME.fullmatch(name) or not isinstance(table, dict):
                raise ValueError(f"{context}: head {name!r} needs a lowercase name of letters, digits and '-', "
                                 "and a table")
            head = take_settings(table, f"{context} [heads.{name}]", required=("method",),
                                 optional={"class_weight": None})
            if head["method"] not in ("centroid", "logistic"):
                raise ValueError(f"{context}: head {name!r} method must be 'centroid' or 'logistic'")
            if head["method"] == "centroid" and head["class_weight"] is not None:
                raise ValueError(f"{context}: centroid head {name!r} takes no class_weight")
            if head["method"] == "logistic" and head["class_weight"] not in CLASS_WEIGHTS:
                raise ValueError(f"{context}: logistic head {name!r} needs class_weight "
                                 f"{' or '.join(map(repr, CLASS_WEIGHTS))}")
            self.heads[name] = head

    def run(self) -> dict:
        dataset = read_dataset(data_file("datasets", self.examples["dataset"], suffix=".csv"), allow_target=True)
        manifest = read_json(data_file("classification", "examples", self.examples["examples"], suffix=".json"))
        rows = select_examples(dataset, manifest)
        labels = np.array([row.label_id for row in rows])
        missing = [LABELS[i] for i in range(len(LABELS)) if not (labels == i).any()]
        if missing:
            raise ValueError(f"Every category needs examples; none for: {', '.join(missing)}")
        embeddings, embedding_summary, model = embed_sentences(self.model, self.options["prefix"],
                                                               self.options["max_input_tokens"], rows, "example")
        splits = list(StratifiedKFold(n_splits=self.options["folds"], shuffle=True,
                                      random_state=self.options["seed"]).split(embeddings, labels))
        tensors, heads = {}, {}
        for name, head in self.heads.items():
            if head["method"] == "centroid":
                weight = centroid_weights(embeddings, labels)
                bias = np.zeros(len(LABELS), dtype=np.float32)
                scores = [balanced_accuracy_score(labels[test], (embeddings[test] @ centroid_weights(
                    embeddings[train], labels[train]).T).argmax(axis=1)) for train, test in splits]
                heads[name] = {"method": "centroid", "cv_balanced_accuracy": float(np.mean(scores))}
            else:
                search = GridSearchCV(
                    LogisticRegression(class_weight=CLASS_WEIGHTS[head["class_weight"]], max_iter=MAX_ITERATIONS),
                    {"C": self.options["c_values"]}, cv=splits, scoring="balanced_accuracy").fit(embeddings, labels)
                estimator = search.best_estimator_
                if list(estimator.classes_) != list(range(len(LABELS))):
                    raise ValueError(f"Head {name!r} learned classes {list(estimator.classes_)}, not all ten")
                iterations = int(np.max(estimator.n_iter_))
                if iterations >= MAX_ITERATIONS:
                    raise ValueError(f"Head {name!r} did not converge in {MAX_ITERATIONS} iterations")
                weight, bias = estimator.coef_.astype(np.float32), estimator.intercept_.astype(np.float32)
                heads[name] = {
                    "method": "logistic", "class_weight": head["class_weight"], "c": float(search.best_params_["C"]),
                    "iterations": iterations, "cv_balanced_accuracy": float(search.best_score_),
                    "cv_by_c": [{"c": float(params["C"]), "balanced_accuracy": float(score)} for params, score in
                                zip(search.cv_results_["params"], search.cv_results_["mean_test_score"], strict=True)],
                }
            tensors[f"{name}.weight"] = weight
            tensors[f"{name}.bias"] = bias
            print(f"Head {name}: cross-validated balanced accuracy {heads[name]['cv_balanced_accuracy']:.4f}",
                  flush=True)
        report = {
            "model": model, "prefix": self.options["prefix"], "max_input_tokens": self.options["max_input_tokens"],
            "examples": {
                "dataset": {"name": self.examples["dataset"], "sha256": dataset.sha256},
                "examples": {"name": self.examples["examples"], "manifest_sha256": manifest["manifest_sha256"],
                             "n": len(rows)},
                "label_counts": {label: int((labels == i).sum()) for i, label in enumerate(LABELS)},
            },
            "embeddings": embedding_summary,
            "cross_validation": {"folds": self.options["folds"], "seed": self.options["seed"], "shuffle": True,
                                 "stratified": True, "scoring": "balanced_accuracy",
                                 "c_values": self.options["c_values"]},
            "heads": heads,
            "scoring": "Category scores are weight @ normalized embedding + bias; the highest score wins.",
        }
        return {"report": report, "tensors": tensors}


def training_shares(head: dict, label_counts: dict) -> np.ndarray:
    """The category mix a logistic head was trained to expect: even under balanced weights, else the examples' mix."""
    if head["class_weight"] == "balanced":
        return np.full(len(LABELS), 1 / len(LABELS))
    counts = np.array([label_counts[label] for label in LABELS], dtype=np.float64)
    return counts / counts.sum()


def accepted_shares(rows, accepted: dict) -> np.ndarray:
    """Each category's share of the sentences, counting a sentence once, split evenly among its accepted labels."""
    totals = np.zeros(len(LABELS))
    for row in rows:
        for label in accepted[row.id]:
            totals[label] += 1 / len(accepted[row.id])
    return totals / len(rows)


class PriorShiftFit:
    """Shift logistic heads from the category mix they were trained on toward a tuning sample's mix.

    A head's scores are log-odds, so a different category mix adds ``log target share - log training share``
    to each category's bias. ``strength`` scales that shift from 0 (the base head) to 1 (the tuning mix).
    """

    kind = "prior-shift"

    def __init__(self, spec):
        context = f"Fit {spec.name!r}"
        unknown = set(spec.tables) - {"base", "tuning", "options", "heads"}
        if unknown:
            raise ValueError(f"{context} has unknown tables: {', '.join(sorted(unknown))}")
        self.name = spec.name
        self.base = take_settings(spec.tables.get("base", {}), f"{context} [base]", required=("fit",))
        resolve_fit(self.base["fit"])
        self.tuning = take_settings(spec.tables.get("tuning", {}), f"{context} [tuning]",
                                    required=("dataset", "sample", "gold"))
        self.options = take_settings(spec.tables.get("options", {}), f"{context} [options]", required=("strengths",))
        strengths = self.options["strengths"]
        if (not isinstance(strengths, list) or not strengths
                or any(not _number(s) or not 0 <= s <= 1 for s in strengths) or strengths != sorted(set(strengths))):
            raise ValueError(f"{context}: strengths must be a nonempty rising list of distinct numbers "
                             "from 0 through 1")
        heads = spec.tables.get("heads", {})
        if not isinstance(heads, dict) or not heads:
            raise ValueError(f"{context} needs at least one [heads.<name>] table")
        self.heads = {}
        for name, table in heads.items():
            if not HEAD_NAME.fullmatch(name) or not isinstance(table, dict):
                raise ValueError(f"{context}: head {name!r} needs a lowercase name of letters, digits and '-', "
                                 "and a table")
            head = take_settings(table, f"{context} [heads.{name}]", required=("base", "strength"))
            if not isinstance(head["base"], str):
                raise ValueError(f"{context}: head {name!r} base must name a head of the base fit")
            if head["strength"] != "tuned" and (not _number(head["strength"]) or not 0 <= head["strength"] <= 1):
                raise ValueError(f"{context}: head {name!r} strength must be 'tuned' or a number from 0 through 1")
            self.heads[name] = head

    def _load_base(self):
        base = load_fit(resolve_fit(self.base["fit"]))
        if base.record["kind"] != EmbeddingHeadsFit.kind:
            raise ValueError(f"Base fit {self.base['fit']!r} is a {base.record['kind']!r} fit, "
                             f"not {EmbeddingHeadsFit.kind!r}")
        heads = base.record["report"]["heads"]
        for name, head in self.heads.items():
            if head["base"] not in heads:
                raise ValueError(f"Base fit {self.base['fit']!r} has no head {head['base']!r}; "
                                 f"it has {', '.join(sorted(heads))}")
            if heads[head["base"]]["method"] != "logistic":
                raise ValueError(f"Head {name!r}: a prior shift needs log-odds scores, from a logistic head; "
                                 f"{head['base']!r} is a {heads[head['base']]['method']!r} head")
        # The base fit file gives the model and prefix its heads read embeddings from.
        settings = EmbeddingHeadsFit(parse_fit_spec(base.record["fit"]["name"],
                                                    (base.path / "fit.toml").read_text(encoding="utf-8")))
        return base, settings

    def run(self) -> dict:
        base, settings = self._load_base()
        dataset = read_dataset(data_file("datasets", self.tuning["dataset"], suffix=".csv"), allow_target=True)
        manifest = read_json(data_file("classification", "samples", self.tuning["sample"], suffix=".json"))
        if manifest.get("purpose") != TUNING_PURPOSE:
            raise ValueError(f"Sample {self.tuning['sample']!r} has purpose {manifest.get('purpose')!r}, not "
                             f"{TUNING_PURPOSE!r}; a fit never tunes on a sample that results are reported on")
        rows = select_rows(dataset, manifest)
        accepted = load_accepted_gold(data_file("classification", "gold", self.tuning["gold"], suffix=".csv"), rows)
        target = accepted_shares(rows, accepted)
        missing = [LABELS[i] for i in range(len(LABELS)) if target[i] == 0]
        if missing:
            raise ValueError(f"Every category needs a tuning sentence to set its share; none for: {', '.join(missing)}")
        embeddings, embedding_summary, model = embed_sentences(
            settings.model, settings.options["prefix"], settings.options["max_input_tokens"], rows, "tuning sentence")
        base_heads = base.record["report"]["heads"]
        label_counts = base.record["report"]["examples"]["label_counts"]
        strengths = [float(strength) for strength in self.options["strengths"]]
        shifts, assumed, curves, tuned = {}, {}, {}, {}
        for base_name in dict.fromkeys(head["base"] for head in self.heads.values()):
            assumed[base_name] = training_shares(base_heads[base_name], label_counts)
            shifts[base_name] = np.log(target) - np.log(assumed[base_name])
            weight, bias = base.tensors[f"{base_name}.weight"], base.tensors[f"{base_name}.bias"]
            curves[base_name] = []
            for strength in strengths:
                # The same float32 bias the shifted head stores, so these predictions are the head's own.
                shifted = (bias + strength * shifts[base_name]).astype(np.float32)
                predictions = (embeddings @ weight.T + shifted).argmax(axis=1)
                responses = [{"id": row.id, "raw_output": json.dumps({"label_id": int(label)})}
                             for row, label in zip(rows, predictions, strict=True)]
                metrics, _, _ = evaluate(rows, responses, accepted)
                curves[base_name].append({"strength": strength, "accuracy": metrics["accepted_accuracy"],
                                          "macro_accuracy": metrics["macro_accuracy"]})
            # max keeps the first of equal scores, so the weakest shift wins a tie.
            tuned[base_name] = max(curves[base_name], key=lambda entry: entry["accuracy"])["strength"]
            print(f"Base head {base_name}: tuned strength {tuned[base_name]:g}", flush=True)
        tensors, heads = {}, {}
        for name, head in self.heads.items():
            base_name = head["base"]
            strength = tuned[base_name] if head["strength"] == "tuned" else float(head["strength"])
            shift = strength * shifts[base_name]
            tensors[f"{name}.weight"] = base.tensors[f"{base_name}.weight"].copy()
            tensors[f"{name}.bias"] = (base.tensors[f"{base_name}.bias"] + shift).astype(np.float32)
            heads[name] = {**{key: base_heads[base_name][key] for key in ("method", "class_weight", "c")},
                           "base_head": base_name, "strength": strength,
                           "strength_choice": "tuned" if head["strength"] == "tuned" else "fixed",
                           "shift": shift.tolist()}
        report = {
            "base": {"reference": base.reference, "fit_sha256": base.record["fit_sha256"],
                     "weights_sha256": base.record["weights_sha256"]},
            "model": model, "prefix": settings.options["prefix"],
            "max_input_tokens": settings.options["max_input_tokens"],
            "tuning": {
                "dataset": {"name": self.tuning["dataset"], "sha256": dataset.sha256},
                "sample": {"name": self.tuning["sample"], "manifest_sha256": manifest["manifest_sha256"],
                           "n": len(rows)},
                "gold": {"name": self.tuning["gold"], "accepted_gold_sha256": accepted_gold_digest(rows, accepted)},
            },
            "target_shares": dict(zip(LABELS, target.tolist())),
            "target_counting": "Each tuning sentence counts once, split evenly among its accepted labels.",
            "training_shares": {name: dict(zip(LABELS, shares.tolist())) for name, shares in assumed.items()},
            "embeddings": embedding_summary,
            "tuning_scores": curves,
            "heads": heads,
            "scoring": "Category scores are weight @ normalized embedding + bias; the highest score wins. Each bias "
                       "is the base head's plus strength * (log target share - log training share). A tuned "
                       "strength has the highest accepted-label accuracy on the tuning sentences; the smallest "
                       "wins a tie.",
        }
        return {"report": report, "tensors": tensors}


FITTERS = {fitter.kind: fitter for fitter in (EmbeddingHeadsFit, PriorShiftFit)}
