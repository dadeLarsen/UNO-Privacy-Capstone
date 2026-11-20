"""Classify with a fitted linear head on a sentence embedder's frozen embedding of the sentence.

The condition names a fit record and one of its heads::

    [options]
    fit = "classification/embeddinggemma2-opp115-heads/20261012-010203Z"
    head = "linear-balanced"
    max_input_tokens = 8192

The instruction prefix comes from the fit, so sentences are embedded exactly as its examples were. A
``prior-shift`` fit reshapes the heads of an ``embedding-heads`` fit, its base; the model and prefix then
come from the base fit, whose hashes must match the ones the prior-shift fit recorded.

The condition's ``[model]`` names the GGUF file that embeds through llama.cpp, and the checkpoint the file
was converted from, which must be the one the fit embedded with::

    [model]
    id = "ggml-org/embeddinggemma-2-GGUF"
    revision = "6ea3e3a309d12a4779cfda98b8c19800d2d5da71"
    precision = "q8_0"
    file = "embeddinggemma-2-Q8_0.gguf"
    device = "cpu"
    converted_from = "google/embeddinggemma-2"
    converted_from_revision = "914f7f89142e33e77833254d9c9b90c3cef7303b"

The sentence-transformers backend, which ran that checkpoint itself, is retired. A condition that names no
file is refused; the recorded runs of that backend still compare.
"""

import json

import numpy as np

from ....core.conditions import Condition
from ....core.files import digest_bytes, digest_text
from ....core.fits import load_fit, resolve_fit
from ....models.gguf_sentence_embedder import GgufSentenceEmbedder
from ....models.hub import download_snapshot, require_commit, resolve_snapshot, snapshot_identity
from ..taxonomy import LABELS
from .settings import first_argmax, model_settings, option_settings

EMBEDDING_FIT_KIND = "embedding-heads"
PRIOR_SHIFT_KIND = "prior-shift"
FIT_KINDS = (EMBEDDING_FIT_KIND, PRIOR_SHIFT_KIND)
# The one fit a GGUF file may use, as fit precision: file precisions. The head was fitted on the bf16
# embeddings of the torch checkpoint the file was converted from; llama.cpp computes them again from the
# file. The file holds the text encoder alone, so the fit's text_only is not compared.
TORCH_FIT_FOR_GGUF = {"bf16": ("bf16", "q8_0")}
HEAD_KEYS = ("method", "class_weight", "c", "base_head", "strength")


def differing_settings(fitted: dict, model: dict) -> list:
    """The [model] keys of a condition that keep it from using the heads of a fit made with `fitted`."""
    differing = [key for key, fitted_key in (("converted_from", "id"), ("converted_from_revision", "revision"))
                 if fitted.get(fitted_key) != model[key]]
    if model["precision"] not in TORCH_FIT_FOR_GGUF.get(fitted.get("precision"), ()):
        differing.append("precision")
    return differing


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
        if "file" not in condition.model:
            raise ValueError(f"Condition {condition.name!r} [model] names no GGUF file. The embedding-head strategy "
                             "embeds with a GGUF file through llama.cpp; its sentence-transformers backend is "
                             "retired. Recorded sentence-transformers runs still compare, and History in "
                             "README.md names the last commit that runs one")
        # llama.cpp has no default device, and a GGUF file is not the checkpoint a fit names.
        self.model = model_settings(condition, required=("file", "device", "converted_from",
                                                         "converted_from_revision"))
        require_commit(self.model["converted_from"], self.model["converted_from_revision"])
        self.options = option_settings(condition, required=("fit", "head", "max_input_tokens"))
        fit = load_fit(resolve_fit(self.options["fit"]))
        if fit.record["kind"] not in FIT_KINDS:
            raise ValueError(f"Fit {self.options['fit']!r} is a {fit.record['kind']!r} fit, "
                             f"not {' or '.join(map(repr, FIT_KINDS))}")
        embedding_fit = fit if fit.record["kind"] == EMBEDDING_FIT_KIND else base_fit(fit)
        differing = differing_settings(embedding_fit.record["fit"]["model"], self.model)
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
        self._weight = fit.tensors[f"{head}.weight"]
        self._bias = fit.tensors[f"{head}.bias"]
        if self._weight.ndim != 2 or len(self._weight) != len(LABELS) or self._bias.shape != (len(LABELS),):
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
            "head": {**self._head, "weight_sha256": digest_bytes(self._weight.tobytes()),
                     "bias_sha256": digest_bytes(self._bias.tobytes())},
            "prefix": self.prefix, "scores": "weight @ normalized float32 embedding + bias",
            "max_input_tokens": self.options["max_input_tokens"], "truncation": False, "batch_size": 1,
        }

    def download(self) -> dict:
        snapshot = download_snapshot(self.model["id"], self.model["revision"], only=self.model["file"])
        return snapshot_identity(self.model["id"], self.model["revision"], snapshot)

    def load(self) -> dict:
        snapshot = resolve_snapshot(self.model["id"], self.model["revision"], self.model["file"])
        backend = GgufSentenceEmbedder(snapshot / self.model["file"], precision=self.model["precision"],
                                       device=self.model["device"],
                                       max_input_tokens=self.options["max_input_tokens"])
        metadata = backend.load()
        self._backend = backend
        return {**snapshot_identity(self.model["id"], self.model["revision"], snapshot), **metadata,
                "fit": self._fit.reference, "head": self._head["name"],
                "selection": "argmax of head scores; first category wins exact ties"}

    def prepare(self, row) -> dict:
        return {"text_sha256": digest_text(row.text),
                "encoded": self._backend.encode(row.text, self.prefix)}

    def describe(self, prepared) -> dict:
        return {"input_tokens": len(prepared["encoded"]["input_ids"])}

    def predict(self, prepared) -> dict:
        output = self._backend.embed(prepared["encoded"])
        embedding = output.pop("embedding")
        scores = (self._weight @ (embedding / np.linalg.norm(embedding)) + self._bias).tolist()
        label_id = first_argmax(scores)
        return {"input_text_sha256": prepared["text_sha256"],
                "raw_output": json.dumps({"label_id": label_id}, ensure_ascii=False),
                "model_label_id": label_id, "model_label": LABELS[label_id], "scores": scores, **output}
