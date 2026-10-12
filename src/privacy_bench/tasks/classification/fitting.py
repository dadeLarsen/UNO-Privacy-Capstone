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
"""

import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import torch

from ...core.conditions import take_settings
from ...core.files import digest_bytes, read_json
from ...core.paths import data_file
from ...models.hub import require_commit, resolve_snapshot, snapshot_identity
from ...models.sentence_embedder import SentenceEmbedder
from .data import read_dataset, select_examples
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

    def _embed(self, rows):
        snapshot = resolve_snapshot(self.model["id"], self.model["revision"])
        embedder = SentenceEmbedder(snapshot, precision=self.model["precision"], device=self.model["device"],
                                    max_input_tokens=self.options["max_input_tokens"],
                                    text_only=self.model["text_only"])
        backend = embedder.load()
        # Check every example against the input limit before the first forward pass.
        encoded, errors = [], []
        for row in rows:
            try:
                encoded.append(embedder.encode(row.text, self.options["prefix"]))
            except ValueError as exc:
                errors.append(f"ID {row.id}: {exc}")
        if errors:
            raise ValueError("Example preflight failed:\n" + "\n".join(errors))
        vectors, seconds = [], 0.0
        for index, item in enumerate(encoded, 1):
            output = embedder.embed(item)
            vectors.append(output["embedding"])
            seconds += output["seconds"]
            if index % 1000 == 0 or index == len(encoded):
                print(f"Embedded {index}/{len(encoded)} examples", flush=True)
        embeddings = torch.nn.functional.normalize(torch.stack(vectors), dim=1).numpy()
        tokens = [len(item["input_ids"]) for item in encoded]
        summary = {"n": len(rows), "dimension": int(embeddings.shape[1]),
                   "sha256": digest_bytes(embeddings.tobytes()), "seconds": seconds,
                   "min_input_tokens": min(tokens), "max_input_tokens": max(tokens)}
        model = {**snapshot_identity(self.model["id"], self.model["revision"], snapshot), **backend}
        return embeddings, summary, model

    def run(self) -> dict:
        dataset = read_dataset(data_file("datasets", self.examples["dataset"], suffix=".csv"), allow_target=True)
        manifest = read_json(data_file("classification", "examples", self.examples["examples"], suffix=".json"))
        rows = select_examples(dataset, manifest)
        labels = np.array([row.label_id for row in rows])
        missing = [LABELS[i] for i in range(len(LABELS)) if not (labels == i).any()]
        if missing:
            raise ValueError(f"Every category needs examples; none for: {', '.join(missing)}")
        embeddings, embedding_summary, model = self._embed(rows)
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


FITTERS = {fitter.kind: fitter for fitter in (EmbeddingHeadsFit,)}
