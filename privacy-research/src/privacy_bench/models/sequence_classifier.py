"""FP32 sequence classification of one raw text at a time, never truncating."""

import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from ..core.files import digest_json, digest_text
from .hub import context_limit, reject_prequantized, require_exact_load, validate_weights
from .measure import gpu_metadata, measured, require_cuda, timed_load, use_exact_fp32_matmul

LOCAL = {"local_files_only": True, "trust_remote_code": False}


def validate_encoding(encoded, limit: int) -> list:
    if not isinstance(encoded, dict) or "input_ids" not in encoded:
        raise ValueError("Encoded input must be a dictionary containing input_ids")
    ids = encoded["input_ids"]
    if not isinstance(ids, list) or not ids:
        raise ValueError("input_ids must be a nonempty list")
    for name, values in encoded.items():
        if (not isinstance(name, str) or not isinstance(values, list) or len(values) != len(ids)
                or any(type(value) is not int or value < 0 for value in values)):
            raise ValueError(f"Encoded field {name!r} must contain one nonnegative integer per input token")
    if len(ids) > limit:
        raise ValueError(f"Input has {len(ids)} tokens including special tokens; exceeds the "
                         f"{limit}-token limit. Input was not truncated.")
    return ids


class SequenceClassifier:
    def __init__(self, snapshot: Path, *, device: str, max_input_tokens: int):
        if device not in ("cuda", "cpu"):
            raise ValueError("device must be cuda or cpu")
        if type(max_input_tokens) is not int or max_input_tokens < 1:
            raise ValueError("max_input_tokens must be a positive integer")
        self.snapshot, self.max_input_tokens = Path(snapshot), max_input_tokens
        self._cuda = device == "cuda"
        self._device = "cuda:0" if self._cuda else "cpu"
        self._model = self._tokenizer = self._config = None

    @property
    def config(self):
        """The checkpoint's configuration, read without loading weights."""
        if self._config is None:
            self._config = AutoConfig.from_pretrained(str(self.snapshot), **LOCAL)
        return self._config

    def load(self) -> dict:
        if self._cuda:
            require_cuda()
        precision_settings = use_exact_fp32_matmul()
        reject_prequantized(self.snapshot)
        validate_weights(self.snapshot)
        tokenizer = AutoTokenizer.from_pretrained(str(self.snapshot), **LOCAL)
        model_limit, candidates = context_limit(self.config, tokenizer)
        if self.max_input_tokens > model_limit:
            raise ValueError(f"Requested input limit {self.max_input_tokens} exceeds the model limit {model_limit}")

        def load():
            model, loading = AutoModelForSequenceClassification.from_pretrained(
                str(self.snapshot), **LOCAL, dtype=torch.float32, output_loading_info=True)
            loading = require_exact_load(loading)
            model.to(self._device)
            model.eval()
            return model, loading

        (model, loading), load_seconds = timed_load(load, cuda=self._cuda)
        if any(parameter.is_floating_point() and parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise RuntimeError("The loaded classifier has floating-point weights that are not FP32")
        self._model, self._tokenizer = model, tokenizer
        metadata = {
            "backend": "transformers-sequence-classification", "load_seconds": load_seconds,
            "model_class": type(model).__name__, "model_type": self.config.model_type,
            "problem_type": self.config.problem_type, "num_labels": self.config.num_labels,
            "id2label": {str(key): value for key, value in self.config.id2label.items()},
            "precision": "fp32", "dtype": "torch.float32", "device": self._device, "device_map": {"": self._device},
            "context_limit": self.max_input_tokens, "model_context_limit": model_limit,
            "context_limit_candidates": candidates,
            "model_config_sha256": digest_text((self.snapshot / "config.json").read_text(encoding="utf-8")),
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_settings": {"add_special_tokens": True, "truncation": False, "padding": False},
            "attention_implementation": model.config._attn_implementation,
            **precision_settings, "batch_size": 1,
            "timing_scope": "model forward pass only; CUDA synchronized when used; excludes tokenization, "
                            "input transfer, probability calculation and model load",
            "memory_scope": "CUDA device 0 peak allocated/reserved bytes including resident model"
                            if self._cuda else "CUDA memory metrics not applicable on CPU",
            "loading_info": loading,
        }
        if self._cuda:
            metadata.update(gpu_metadata())
        return json.loads(json.dumps(metadata, default=str))

    def encode(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty raw sentence string")
        encoded = dict(self._tokenizer(text, add_special_tokens=True, truncation=False,
                                       padding=False, return_attention_mask=True))
        validate_encoding(encoded, self.max_input_tokens)
        return encoded

    def classify(self, encoded: dict) -> dict:
        """Return the logits and softmax probabilities in the checkpoint's own label order."""
        ids = validate_encoding(encoded, self.max_input_tokens)
        tensors = {name: torch.tensor([values], dtype=torch.long, device=self._device)
                   for name, values in encoded.items()}
        output, measurements = measured(lambda: self._model(**tensors), cuda=self._cuda)
        if not hasattr(output, "logits"):
            raise ValueError("Classifier output has no logits")
        logits = output.logits
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, self.config.num_labels):
            raise ValueError(f"Classifier must return logits with shape (1, {self.config.num_labels})")
        if not logits.is_floating_point():
            raise ValueError("Classifier logits must be floating-point values")
        if not torch.isfinite(logits).all().item():
            raise ValueError("Classifier returned nonfinite logits")
        values = logits.detach().to(device="cpu", dtype=torch.float32)[0]
        return {
            "logits": values.tolist(), "probabilities": torch.softmax(values, dim=0).tolist(),
            "input_tokens": len(ids), "input_token_ids": list(ids),
            "input_token_sha256": digest_json(ids),
            **measurements,
        }
