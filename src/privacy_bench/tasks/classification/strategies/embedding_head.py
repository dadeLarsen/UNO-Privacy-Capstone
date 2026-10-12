"""Classify with a fitted linear head on a sentence embedder's frozen embedding of the sentence.

The condition names a fit record and one of its heads::

    [options]
    fit = "classification/embeddinggemma2-opp115-heads/20261012-010203Z"
    head = "linear-balanced"
    max_input_tokens = 8192

The instruction prefix comes from the fit, so sentences are embedded exactly as its examples were. A
``prior-shift`` fit reshapes the heads of an ``embedding-heads`` fit, its base; the model and prefix then
come from the base fit, whose hashes must match the ones the prior-shift fit recorded.
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

EMBEDDING_FIT_KIND = "embedding-heads"
PRIOR_SHIFT_KIND = "prior-shift"
FIT_KINDS = (EMBEDDING_FIT_KIND, PRIOR_SHIFT_KIND)
# The fit's embeddings are only valid for the same checkpoint at the same precision and module set.
MATCHED_MODEL_KEYS = {"id": None, "revision": None, "precision": None, "text_only": False}
HEAD_KEYS = ("method", "class_weight", "c", "base_head", "strength")


def base_fit(fit):
    """The embedding-heads fit a prior-shift fit was made from, checked against the hashes it recorded."""
    recorded = fit.record["report"]["base"]
    base = load_fit(resolve_fit(recorded["reference"]))
    if base.record["kind"] != EMBEDDING_FIT_KIND:
        raise ValueError(f"Base fit {recorded['reference']!r} is a {base.record['kind']!r} fit, "
                         f"not {EMBEDDING_FIT_KIND!r}")
    changed = [key for key in ("fit_sha256", "weights_sha256") if base.record[key] != recorded[key]]
    if changed:
        raise ValueError(f"Base fit {recorded['reference']!r} differs from the one {fit.reference!r} shifted: "
                         f"{', '.join(changed)}")
    return base


class EmbeddingHeadStrategy:
    name = "embedding-head"
    timing_scope = "embedding_forward"
    timing_description = "embedding forward pass (fitted head not timed)"
    generates_text = False

    @staticmethod
    def condition_label(spec: dict) -> str:
        head = spec["head"]
        if "strength" in head:
            return f"Embedding\nhead\n{head['base_head']}\nprior shift {head['strength']:g}"
        return "Embedding\nhead\n" + head["name"]

    def __init__(self, condition: Condition):
        self.model = model_settings(condition, optional={"device": "cuda", "text_only": False})
        self.options = option_settings(condition, required=("fit", "head", "max_input_tokens"))
        fit = load_fit(resolve_fit(self.options["fit"]))
        if fit.record["kind"] not in FIT_KINDS:
            raise ValueError(f"Fit {self.options['fit']!r} is a {fit.record['kind']!r} fit, "
                             f"not {' or '.join(map(repr, FIT_KINDS))}")
        embedding_fit = fit if fit.record["kind"] == EMBEDDING_FIT_KIND else base_fit(fit)
        fitted_model = embedding_fit.record["fit"]["model"]
        differing = [key for key, default in MATCHED_MODEL_KEYS.items()
                     if fitted_model.get(key, default) != self.model[key]]
        if differing:
            raise ValueError(f"Condition [model] differs from the model fit {embedding_fit.reference!r} used: "
                             f"{', '.join(differing)}")
        head = self.options["head"]
        heads = fit.record["report"]["heads"]
        if head not in heads:
            raise ValueError(f"Fit {self.options['fit']!r} has no head {head!r}; it has {', '.join(sorted(heads))}")
        self._fit = fit
        self._embedding_fit = embedding_fit
        self._head = {"name": head, **{key: heads[head][key] for key in HEAD_KEYS if key in heads[head]}}
        self._weight = torch.from_numpy(fit.tensors[f"{head}.weight"])
        self._bias = torch.from_numpy(fit.tensors[f"{head}.bias"])
        if tuple(self._weight.shape) != (len(LABELS), self._weight.shape[1]) or tuple(self._bias.shape) != (len(LABELS),):
            raise ValueError(f"Head {head!r} must score exactly {len(LABELS)} categories")
        self._backend = None

    @property
    def prefix(self) -> str:
        return self._embedding_fit.record["report"]["prefix"]

    @property
    def input_spec(self) -> dict:
        fit = {"reference": self._fit.reference, "fit_sha256": self._fit.record["fit_sha256"],
               "weights_sha256": self._fit.record["weights_sha256"],
               "examples": self._embedding_fit.record["report"]["examples"]["examples"]}
        if self._fit is not self._embedding_fit:
            fit.update({key: self._fit.record["report"][key] for key in ("base", "tuning")})
        return {
            "name": "embedding-head", "labels": list(LABELS), "fit": fit,
            "head": {**self._head, "weight_sha256": digest_bytes(self._weight.numpy().tobytes()),
                     "bias_sha256": digest_bytes(self._bias.numpy().tobytes())},
            "prefix": self.prefix, "scores": "weight @ normalized float32 embedding + bias",
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
                "encoded": self._backend.encode(row.text, self.prefix)}

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
