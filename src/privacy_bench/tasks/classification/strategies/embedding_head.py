"""Classify with a fitted linear head on a sentence embedder's frozen embedding of the sentence.

The condition names a fit record and one of its heads::

    [options]
    fit = "classification/embeddinggemma2-opp115-heads/20261012-010203Z"
    head = "linear-balanced"
    max_input_tokens = 8192

The instruction prefix comes from the fit, so sentences are embedded exactly as its examples were.
"""

import json

import torch

from ....core.conditions import Condition
from ....core.files import digest_bytes
from ....core.fits import load_fit, resolve_fit
from ....models.hub import download_snapshot, resolve_snapshot, snapshot_identity
from ....models.sentence_embedder import SentenceEmbedder
from ..taxonomy import LABELS
from .settings import first_argmax, model_settings, option_settings, text_sha256

FIT_KIND = "embedding-heads"
# The fit's embeddings are only valid for the same checkpoint at the same precision and module set.
MATCHED_MODEL_KEYS = {"id": None, "revision": None, "precision": None, "text_only": False}


class EmbeddingHeadStrategy:
    name = "embedding-head"
    timing_scope = "embedding_forward"
    timing_description = "embedding forward pass (fitted head not timed)"
    generates_text = False

    @staticmethod
    def condition_label(spec: dict) -> str:
        return "Embedding\nhead\n" + spec["head"]["name"]

    def __init__(self, condition: Condition):
        self.model = model_settings(condition, optional={"device": "cuda", "text_only": False})
        self.options = option_settings(condition, required=("fit", "head", "max_input_tokens"))
        fit = load_fit(resolve_fit(self.options["fit"]))
        if fit.record["kind"] != FIT_KIND:
            raise ValueError(f"Fit {self.options['fit']!r} is a {fit.record['kind']!r} fit, not {FIT_KIND!r}")
        fitted_model = fit.record["fit"]["model"]
        differing = [key for key, default in MATCHED_MODEL_KEYS.items()
                     if fitted_model.get(key, default) != self.model[key]]
        if differing:
            raise ValueError(f"Condition [model] differs from the model fit {self.options['fit']!r} used: "
                             f"{', '.join(differing)}")
        head = self.options["head"]
        heads = fit.record["report"]["heads"]
        if head not in heads:
            raise ValueError(f"Fit {self.options['fit']!r} has no head {head!r}; it has {', '.join(sorted(heads))}")
        self._fit = fit
        self._head = {"name": head, **{key: heads[head][key] for key in ("method", "class_weight", "c")
                                       if key in heads[head]}}
        self._weight = torch.from_numpy(fit.tensors[f"{head}.weight"])
        self._bias = torch.from_numpy(fit.tensors[f"{head}.bias"])
        if tuple(self._weight.shape) != (len(LABELS), self._weight.shape[1]) or tuple(self._bias.shape) != (len(LABELS),):
            raise ValueError(f"Head {head!r} must score exactly {len(LABELS)} categories")
        self._backend = None

    @property
    def input_spec(self) -> dict:
        report = self._fit.record["report"]
        return {
            "name": "embedding-head", "labels": list(LABELS),
            "fit": {"reference": self._fit.reference, "fit_sha256": self._fit.record["fit_sha256"],
                    "weights_sha256": self._fit.record["weights_sha256"], "examples": report["examples"]["examples"]},
            "head": {**self._head, "weight_sha256": digest_bytes(self._weight.numpy().tobytes()),
                     "bias_sha256": digest_bytes(self._bias.numpy().tobytes())},
            "prefix": report["prefix"], "scores": "weight @ normalized float32 embedding + bias",
            "max_input_tokens": self.options["max_input_tokens"], "truncation": False, "batch_size": 1,
        }

    def download(self) -> dict:
        snapshot = download_snapshot(self.model["id"], self.model["revision"])
        return snapshot_identity(self.model["id"], self.model["revision"], snapshot)

    def load(self) -> dict:
        snapshot = resolve_snapshot(self.model["id"], self.model["revision"])
        backend = SentenceEmbedder(snapshot, precision=self.model["precision"], device=self.model["device"],
                                   max_input_tokens=self.options["max_input_tokens"], text_only=self.model["text_only"])
        metadata = backend.load()
        self._backend = backend
        return {**snapshot_identity(self.model["id"], self.model["revision"], snapshot), **metadata,
                "fit": self._fit.reference, "head": self._head["name"],
                "selection": "argmax of head scores; first category wins exact ties"}

    def prepare(self, row) -> dict:
        return {"text_sha256": text_sha256(row.text),
                "encoded": self._backend.encode(row.text, self._fit.record["report"]["prefix"])}

    def describe(self, prepared) -> dict:
        return {"input_tokens": len(prepared["encoded"]["input_ids"])}

    def predict(self, prepared) -> dict:
        output = self._backend.embed(prepared["encoded"])
        embedding = torch.nn.functional.normalize(output.pop("embedding"), dim=0)
        scores = (self._weight @ embedding + self._bias).tolist()
        label_id = first_argmax(scores)
        return {"input_text_sha256": prepared["text_sha256"],
                "raw_output": json.dumps({"label_id": label_id}, ensure_ascii=False),
                "model_label_id": label_id, "model_label": LABELS[label_id], "scores": scores, **output}
