"""Classify without training: embed the sentence and each category's description, pick the most similar."""

import json

import torch

from ....core.conditions import Condition
from ....core.files import digest_json
from ....models.hub import download_snapshot, resolve_snapshot, snapshot_identity
from ....models.sentence_embedder import SentenceEmbedder
from ..prompts import display_name, load_prompt
from ..taxonomy import LABELS
from .settings import first_argmax, model_settings, option_settings, text_sha256


def category_texts(prompt_spec: dict) -> list:
    """Each category as its name, followed by its definition when the prompt supplies one."""
    definitions = prompt_spec["definitions"]
    if prompt_spec["guidance"]:
        raise ValueError("Embedding similarity has no place for prompt guidance; use a prompt without guidance")
    return [label + (": " + definitions[label] if definitions else "") for label in LABELS]


class EmbeddingSimilarityStrategy:
    name = "embedding-similarity"
    timing_scope = "embedding_forward"
    timing_description = "embedding forward pass"
    generates_text = False

    @staticmethod
    def condition_label(spec: dict) -> str:
        return "Embedding similarity\n" + display_name(spec["prompt_spec"]["name"]).lower()

    def __init__(self, condition: Condition):
        self.model = model_settings(condition, optional={"device": "cuda", "text_only": False})
        self.options = option_settings(condition, required=("prompt", "task_prefix", "max_input_tokens"))
        if not isinstance(self.options["task_prefix"], str) or not self.options["task_prefix"].strip():
            raise ValueError("task_prefix must be the model's nonempty instruction prefix")
        self.prompt_spec = load_prompt(self.options["prompt"])
        self._texts = category_texts(self.prompt_spec)
        self._backend = self._categories = None

    @property
    def input_spec(self) -> dict:
        prefix = self.options["task_prefix"]
        return {"name": "embedding-similarity", "prompt_spec": self.prompt_spec, "labels": list(LABELS),
                "category_texts": self._texts, "sentence_prefix": prefix, "category_prefix": prefix,
                "similarity": "cosine of float32 embeddings", "max_input_tokens": self.options["max_input_tokens"],
                "truncation": False, "batch_size": 1}

    def download(self) -> dict:
        snapshot = download_snapshot(self.model["id"], self.model["revision"])
        return snapshot_identity(self.model["id"], self.model["revision"], snapshot)

    def load(self) -> dict:
        snapshot = resolve_snapshot(self.model["id"], self.model["revision"])
        backend = SentenceEmbedder(snapshot, precision=self.model["precision"], device=self.model["device"],
                                   max_input_tokens=self.options["max_input_tokens"], text_only=self.model["text_only"])
        metadata = backend.load()
        # The category embeddings are computed once, at load, with the same prefix as every sentence.
        outputs = [backend.embed(backend.encode(text, self.options["task_prefix"])) for text in self._texts]
        self._categories = torch.nn.functional.normalize(torch.stack([out["embedding"] for out in outputs]), dim=1)
        self._backend = backend
        return {**snapshot_identity(self.model["id"], self.model["revision"], snapshot), **metadata,
                "category_embeddings": {
                    "input_tokens": [out["input_tokens"] for out in outputs],
                    "input_token_ids": [out["input_token_ids"] for out in outputs],
                    "seconds": sum(out["seconds"] for out in outputs),
                    "sha256": digest_json(self._categories.tolist()),
                },
                "selection": "argmax cosine similarity; first category wins exact ties"}

    def prepare(self, row) -> dict:
        return {"text_sha256": text_sha256(row.text), "encoded": self._backend.encode(row.text, self.options["task_prefix"])}

    def describe(self, prepared) -> dict:
        return {"input_tokens": len(prepared["encoded"]["input_ids"])}

    def predict(self, prepared) -> dict:
        output = self._backend.embed(prepared["encoded"])
        sentence = torch.nn.functional.normalize(output.pop("embedding"), dim=0)
        similarities = (self._categories @ sentence).tolist()
        label_id = first_argmax(similarities)
        return {"input_text_sha256": prepared["text_sha256"],
                "raw_output": json.dumps({"label_id": label_id}, ensure_ascii=False),
                "model_label_id": label_id, "model_label": LABELS[label_id], "similarities": similarities, **output}
