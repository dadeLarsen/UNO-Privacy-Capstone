"""Classify the raw sentence with a trained ten-category sequence classifier."""

import json

from ....core.conditions import Condition
from ....models.hub import download_snapshot, resolve_snapshot, snapshot_identity
from ....models.sequence_classifier import SequenceClassifier
from ..taxonomy import LABELS
from .settings import first_argmax, model_settings, option_settings, text_sha256

# Checkpoint label names that differ from the benchmark's only in punctuation.
LABEL_ALIASES = {"User Access, Edit and Deletion": "User Access Edit and Deletion"}


def label_mapping(config) -> list:
    """Map every checkpoint label index one-to-one onto the ten benchmark categories."""
    if config.problem_type != "single_label_classification":
        raise ValueError("Checkpoint must declare problem_type=single_label_classification")
    if config.num_labels != len(LABELS):
        raise ValueError(f"Checkpoint must have exactly {len(LABELS)} labels")
    names = config.id2label
    if not isinstance(names, dict) or set(names) != set(range(len(LABELS))):
        raise ValueError("Checkpoint id2label must cover every model label index exactly once")
    mapping = []
    for index in range(len(LABELS)):
        name = names[index]
        if not isinstance(name, str):
            raise ValueError(f"Checkpoint label {index} must be a string")
        target = LABEL_ALIASES.get(name, name)
        if target not in LABELS:
            raise ValueError(f"Unknown checkpoint category at index {index}: {name!r}")
        mapping.append({"model_label_id": index, "model_label": name,
                        "label_id": LABELS.index(target), "label": target})
    if len({item["label_id"] for item in mapping}) != len(LABELS):
        raise ValueError("Checkpoint labels must map one-to-one onto all ten benchmark categories")
    reverse = config.label2id
    if not isinstance(reverse, dict) or reverse != {name: index for index, name in names.items()}:
        raise ValueError("Checkpoint label2id must be the exact inverse of id2label")
    return mapping


class ClassifierHeadStrategy:
    name = "classifier-head"
    timing_scope = "model_forward"

    def __init__(self, condition: Condition):
        self.model = model_settings(condition, optional={"device": "cuda"})
        if self.model["precision"] != "fp32":
            raise ValueError("The classifier-head strategy runs the checkpoint in fp32")
        self.options = option_settings(condition, required=("max_input_tokens",))
        self._backend = self._mapping = None

    @property
    def input_spec(self) -> dict:
        return {"name": "raw-text", "format": "sentence_text", "add_special_tokens": True,
                "truncation": False, "padding": False, "batch_size": 1,
                "max_input_tokens": self.options["max_input_tokens"]}

    def download(self) -> dict:
        snapshot = download_snapshot(self.model["id"], self.model["revision"])
        return snapshot_identity(self.model["id"], self.model["revision"], snapshot)

    def load(self) -> dict:
        snapshot = resolve_snapshot(self.model["id"], self.model["revision"])
        backend = SequenceClassifier(snapshot, device=self.model["device"],
                                     max_input_tokens=self.options["max_input_tokens"])
        # Check the label names before any weights are loaded.
        self._mapping = label_mapping(backend.config)
        metadata = backend.load()
        self._backend = backend
        return {**snapshot_identity(self.model["id"], self.model["revision"], snapshot), **metadata,
                "label_mapping": self._mapping, "selection": "argmax logits; first model index wins exact ties"}

    def prepare(self, row) -> dict:
        return {"text_sha256": text_sha256(row.text), "encoded": self._backend.encode(row.text)}

    def describe(self, prepared) -> dict:
        return {"input_tokens": len(prepared["encoded"]["input_ids"])}

    def predict(self, prepared) -> dict:
        output = self._backend.classify(prepared["encoded"])
        selected = self._mapping[first_argmax(output["logits"])]
        return {"input_text_sha256": prepared["text_sha256"],
                "raw_output": json.dumps({"label_id": selected["label_id"]}, ensure_ascii=False),
                "model_label_id": selected["model_label_id"], "model_label": selected["model_label"], **output}
