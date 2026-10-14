"""Sentence embeddings from a local sentence-transformers checkpoint, one text at a time, never truncating."""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
import torch

from ..core.files import digest_json, digest_text
from .hub import context_limit, reject_prequantized, validate_weights
from .measure import gpu_metadata, measured, require_cuda, timed_load, use_exact_fp32_matmul

MODES = (("cuda", "bf16"), ("cuda", "fp32"), ("cpu", "fp32"))
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
# A normalized, pooled sentence embedding needs exactly these three modules, in this order.
PIPELINE = ("Transformer", "Pooling", "Normalize")
NON_TEXT_CONFIGS = ("vision_config", "audio_config")


class SentenceEmbedder:
    def __init__(self, snapshot: Path, *, precision: str, device: str, max_input_tokens: int, text_only: bool):
        if (device, precision) not in MODES:
            raise ValueError("Embeddings run on CUDA in bf16 or fp32, or on CPU in fp32; fp16 overflows "
                             "the activations of some embedding models")
        if type(max_input_tokens) is not int or max_input_tokens < 1:
            raise ValueError("max_input_tokens must be a positive integer")
        if type(text_only) is not bool:
            raise ValueError("text_only must be true or false")
        self.snapshot, self.precision, self.max_input_tokens = Path(snapshot), precision, max_input_tokens
        self.text_only = text_only
        self._cuda = device == "cuda"
        self._device = "cuda:0" if self._cuda else "cpu"
        self._model = None

    def _config_overrides(self, config: dict) -> dict:
        """Drop the non-text encoders of a multimodal checkpoint so they are never loaded."""
        if not self.text_only:
            return {}
        present = [key for key in NON_TEXT_CONFIGS if config.get(key) is not None]
        if not present:
            raise ValueError("text_only needs a multimodal checkpoint with vision or audio encoder configs")
        return {key: None for key in present}

    def load(self) -> dict:
        if self._cuda:
            require_cuda(bf16=self.precision == "bf16")
        precision_settings = use_exact_fp32_matmul() if self.precision == "fp32" else {}
        config = reject_prequantized(self.snapshot)
        validate_weights(self.snapshot)
        modules = json.loads((self.snapshot / "modules.json").read_text(encoding="utf-8"))
        kinds = tuple(module["type"].rsplit(".", 1)[-1] for module in sorted(modules, key=lambda item: item["idx"]))
        if kinds != PIPELINE:
            raise ValueError(f"Expected a {' -> '.join(PIPELINE)} sentence-transformers pipeline, got {kinds}")
        overrides = self._config_overrides(config)
        dtype = DTYPES[self.precision]

        def load():
            return SentenceTransformer(str(self.snapshot), device=self._device, local_files_only=True,
                                       trust_remote_code=False, model_kwargs={"dtype": dtype},
                                       config_kwargs=overrides).eval()

        model, load_seconds = timed_load(load, cuda=self._cuda)
        dtypes = {parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()}
        if dtypes != {dtype}:
            raise RuntimeError(f"Loaded embedding weights are {sorted(map(str, dtypes))}, not {dtype}")
        backbone = model[0].model
        model_limit, candidates = context_limit(backbone.config, model.tokenizer)
        if self.max_input_tokens > model_limit:
            raise ValueError(f"Requested input limit {self.max_input_tokens} exceeds the model limit {model_limit}")
        self._model = model
        metadata = {
            "backend": "sentence-transformers-embedding", "load_seconds": load_seconds,
            "pipeline": list(kinds), "model_class": type(backbone).__name__,
            "loaded_submodules": [name for name, _ in backbone.named_children()],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "text_only": self.text_only, "config_overrides": sorted(overrides),
            "pooling": model[1].get_config_dict(), "embedding_dimension": model.get_embedding_dimension(),
            "similarity_fn_name": model.similarity_fn_name,
            "precision": self.precision, "dtype": str(dtype), "device": self._device, "device_map": {"": self._device},
            "context_limit": self.max_input_tokens, "model_context_limit": model_limit,
            "context_limit_candidates": candidates, **precision_settings,
            "model_config_sha256": digest_text((self.snapshot / "config.json").read_text(encoding="utf-8")),
            "tokenizer_settings": {"processing": "sentence-transformers preprocess", "truncation": False},
            "batch_size": 1,
            "timing_scope": "synchronized embedding forward pass, including pooling and normalization; excludes "
                            "tokenization, input transfer, similarity and model load",
            "memory_scope": "PyTorch CUDA device 0 peak allocated/reserved bytes including resident weights"
                            if self._cuda else "CUDA memory metrics not applicable on CPU",
        }
        if self._cuda:
            metadata.update(gpu_metadata())
        return json.loads(json.dumps(metadata, default=str))

    def encode(self, text: str, prompt: str) -> dict:
        """Apply the prompt the way the checkpoint expects and tokenize without truncation."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")
        features = self._model.preprocess([text], prompt=prompt, processing_kwargs={"text": {"truncation": False}})
        ids = features["input_ids"][0].tolist()
        if len(ids) > self.max_input_tokens:
            raise ValueError(f"Input has {len(ids)} tokens including the prompt and special tokens; exceeds the "
                             f"{self.max_input_tokens}-token limit. Input was not truncated.")
        return {"features": features, "input_ids": ids}

    def embed(self, encoded: dict) -> dict:
        """Return the normalized sentence embedding (float32 on the CPU) and its measurements."""
        # batch_to_device and the model both write into the dict they receive. A copy keeps the caller's
        # encoded input on the CPU, so the model's outputs are freed after this call.
        features = batch_to_device(dict(encoded["features"]), self._device)
        embedding, measurements = measured(lambda: self._model(features)["sentence_embedding"][0], cuda=self._cuda)
        ids = encoded["input_ids"]
        return {"embedding": embedding.float().cpu(), "input_tokens": len(ids), "input_token_ids": list(ids),
                "input_token_sha256": digest_json(ids), **measurements}
