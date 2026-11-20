"""Sentence embeddings from a local GGUF embedding model through llama.cpp, one text at a time, never truncating."""

import json
from pathlib import Path
import platform
import time

import llama_cpp
import numpy as np

from ..core.files import digest_json
from .gguf_chat_model import placement

# Device and precision. gpu puts every layer on the GPU backend of the installed llama.cpp build.
MODES = tuple((device, precision) for device in ("cpu", "gpu") for precision in ("bf16", "q8_0"))
# The general.file_type that a GGUF file of each precision declares.
FILE_TYPES = {"bf16": "32", "q8_0": "7"}


class GgufSentenceEmbedder:
    def __init__(self, path: Path, *, precision: str, device: str, max_input_tokens: int):
        if (device, precision) not in MODES:
            raise ValueError("GGUF embedders run on cpu or gpu at precision bf16 or q8_0; got device "
                             f"{device!r}, precision {precision!r}")
        if type(max_input_tokens) is not int or max_input_tokens < 1:
            raise ValueError("max_input_tokens must be a positive integer")
        self.path, self.precision, self.device = Path(path), precision, device
        self.max_input_tokens = max_input_tokens
        self._llm = self._size = None

    def load(self) -> dict:
        placed = placement(self.device)
        if not self.path.is_file():
            raise ValueError(f"No GGUF file at {self.path}. Run `pbench download` to cache it.")
        limit = self.max_input_tokens
        started = time.perf_counter()
        # An embedding model reads its whole input in one batch, so the batch gets room for the limit too.
        llm = llama_cpp.Llama(model_path=str(self.path), embedding=True, n_ctx=limit, n_batch=limit, n_ubatch=limit,
                              verbose=False, **placed)
        load_seconds = time.perf_counter() - started
        # llama-cpp-python allocates a score for every vocabulary entry of every token in the batch, which is
        # 1 MiB per token of the limit for EmbeddingGemma 2. Embedding reads no score, and Windows counts the
        # whole array as committed memory, so the array goes.
        llm.scores = np.ndarray((0, llm.scores.shape[1]), dtype=np.single)
        described = dict(llm.metadata)
        # Sentences are embedded as plain text behind a prefix; a chat template in the file is not used.
        described.pop("tokenizer.chat_template", None)
        if described.get("general.file_type") != FILE_TYPES[self.precision]:
            raise ValueError(f"{self.path.name} declares file type {described.get('general.file_type')}, not the "
                             f"{FILE_TYPES[self.precision]} of precision {self.precision}")
        architecture = described["general.architecture"]
        model_limit = int(described[f"{architecture}.context_length"])
        if limit > model_limit:
            raise ValueError(f"Requested input limit {limit} exceeds the model limit {model_limit}")
        # The context takes its pooling from the file. A fitted head is only valid for the pooling its
        # examples were embedded with, and every recorded fit pooled by the mean.
        pooling = llm.pooling_type()
        if pooling != llama_cpp.LLAMA_POOLING_TYPE_MEAN:
            raise ValueError(f"{self.path.name} pools with llama.cpp pooling type {pooling}, not the mean "
                             f"({llama_cpp.LLAMA_POOLING_TYPE_MEAN})")
        self._llm, self._size = llm, llama_cpp.llama_model_n_embd_out(llm.model)
        metadata = {
            "backend": "llama-cpp-embedding", "load_seconds": load_seconds,
            "precision": self.precision, "file": self.path.name, "file_bytes": self.path.stat().st_size,
            "gguf_metadata": described, "architecture": architecture, "pooling": "mean",
            "embedding_dimension": self._size, "device": self.device,
            "gpu_offload_supported": llama_cpp.llama_supports_gpu_offload(),
            "llama_cpp_python_version": llama_cpp.__version__,
            "llama_cpp_system_info": llama_cpp.llama_print_system_info().decode("utf-8").strip(),
            "threads": llm.n_threads, "threads_batch": llm.n_threads_batch,
            "context_limit": limit, "model_context_limit": model_limit,
            "tokenizer_settings": {"processing": "llama.cpp tokenizer; adds the special tokens the file asks for "
                                                 "and reads special tokens written in the text",
                                   "truncation": False},
            "batch_size": 1,
            "timing_scope": "llama.cpp evaluation of the whole input, including pooling and normalization; "
                            "excludes tokenization, similarity and model load",
            "memory_scope": "no memory metrics",
            "cpu": platform.processor(),
        }
        return json.loads(json.dumps(metadata, default=str))

    def encode(self, text: str, prompt: str) -> dict:
        """Put the prompt before the text and tokenize both at once, without truncation."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")
        ids = self._llm.tokenize((prompt + text).encode("utf-8"), add_bos=True, special=True)
        if len(ids) > self.max_input_tokens:
            raise ValueError(f"Input has {len(ids)} tokens including the prompt and special tokens; exceeds the "
                             f"{self.max_input_tokens}-token limit. Input was not truncated.")
        return {"input_ids": ids}

    def embed(self, encoded: dict) -> dict:
        """Return the normalized sentence embedding as a float32 array, and its measurements."""
        ids = encoded["input_ids"]
        context, batch = self._llm._ctx, self._llm._batch
        # llama.cpp keeps what it last evaluated; every sentence starts from an empty context.
        context.kv_cache_clear()
        batch.reset()
        # Pooling needs an output for every token.
        batch.add_sequence(ids, 0, True)
        started = time.perf_counter()
        context.decode(batch)
        # Reading the pooled embedding waits for a GPU backend to finish.
        pooled = llama_cpp.llama_get_embeddings_seq(context.ctx, 0)
        embedding = np.array(pooled[:self._size], dtype=np.float32)
        embedding /= np.linalg.norm(embedding)
        seconds = time.perf_counter() - started
        return {"embedding": embedding, "input_tokens": len(ids), "input_token_ids": list(ids),
                "input_token_sha256": digest_json(ids), "seconds": seconds}
