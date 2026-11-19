"""Greedy generation from a local GGUF chat model through llama.cpp, one prompt at a time, never truncating."""

import ctypes
import json
from pathlib import Path
import platform
import time

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
import llama_cpp

from ..core.files import digest_json, digest_text
from .chat_model import CHAT_TEMPLATE_KWARGS, chat_messages

# Device and precision. gpu puts every layer on the GPU backend of the installed llama.cpp build.
MODES = (("cpu", "q4_k_m"), ("gpu", "q4_k_m"))
# The general.file_type that a GGUF file of each precision declares.
FILE_TYPES = {"q4_k_m": "15"}
# llama.cpp allocates its context at load. This many tokens hold the longest rewrite prompt and its output;
# a model trained on a shorter context keeps its own limit.
CONTEXT_TOKENS = 4096


def placement(device: str) -> dict:
    """The model parameters that put every layer on the GPU backend, or the whole model on the CPU."""
    if device == "gpu":
        if not llama_cpp.llama_supports_gpu_offload():
            raise RuntimeError("The installed llama.cpp build has no GPU backend; run a condition with device cpu")
        return {"n_gpu_layers": -1}
    # With zero layers on the GPU, a build that has a GPU backend still sends large prompt batches to it.
    # Naming no GPU device takes that backend out of the model, so a cpu run computes only on the CPU.
    return {"n_gpu_layers": 0, "split_mode": llama_cpp.LLAMA_SPLIT_MODE_NONE, "main_gpu": -1}


class GgufChatModel:
    def __init__(self, path: Path, *, precision: str, device: str, max_new_tokens: int, seed: int):
        if (device, precision) not in MODES:
            raise ValueError("GGUF chat models run on cpu or gpu at precision q4_k_m; got device "
                             f"{device!r}, precision {precision!r}")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.path, self.precision, self.device = Path(path), precision, device
        self.max_new_tokens, self.seed = max_new_tokens, seed
        self._llm = self._vocab = self._template = self._context = None

    def load(self) -> dict:
        placed = placement(self.device)
        if not self.path.is_file():
            raise ValueError(f"No GGUF file at {self.path}. Run `pbench download` to cache it.")
        started = time.perf_counter()
        llm = llama_cpp.Llama(model_path=str(self.path), n_ctx=CONTEXT_TOKENS, seed=self.seed, verbose=False,
                              **placed)
        load_seconds = time.perf_counter() - started
        described = dict(llm.metadata)
        template = described.pop("tokenizer.chat_template", None)
        if not template:
            raise ValueError("This GGUF file has no chat template; choose an instruction model")
        if described.get("general.file_type") != FILE_TYPES[self.precision]:
            raise ValueError(f"{self.path.name} declares file type {described.get('general.file_type')}, not the "
                             f"{FILE_TYPES[self.precision]} of precision {self.precision}")
        # Strict, so a template that needs a variable this backend does not pass fails instead of dropping it.
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                    extensions=["jinja2.ext.loopcontrols"],
                                                    undefined=jinja2.StrictUndefined)
        self._llm, self._template = llm, environment.from_string(template)
        self._vocab = llama_cpp.llama_model_get_vocab(llm.model)
        trained = int(described[f"{described['general.architecture']}.context_length"])
        self._context = min(llm.n_ctx(), trained)
        metadata = {
            "backend": "llama-cpp-chat-generation", "load_seconds": load_seconds,
            "precision": self.precision, "file": self.path.name, "file_bytes": self.path.stat().st_size,
            "gguf_metadata": described, "device": self.device,
            "gpu_offload_supported": llama_cpp.llama_supports_gpu_offload(),
            "llama_cpp_python_version": llama_cpp.__version__,
            "llama_cpp_system_info": llama_cpp.llama_print_system_info().decode("utf-8").strip(),
            "threads": llm.n_threads, "threads_batch": llm.n_threads_batch,
            "chat_template_sha256": digest_text(template), "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            "context_tokens": self._context,
            "generation": "greedy; stops at an end-of-generation token or at max_new_tokens",
            "max_new_tokens": self.max_new_tokens, "seed": self.seed, "batch_size": 1,
            "timing_scope": "prompt processing and generation from an empty context; excludes tokenization, "
                            "decoding, and model load",
            "memory_scope": "no memory metrics",
            "output_decoding": "special tokens kept as text; only a final end-of-generation token removed "
                               "from raw_output",
            "warnings": ["enable_thinking=False is passed to the chat template; only templates that implement it "
                         "disable reasoning."],
            "cpu": platform.processor(),
        }
        return json.loads(json.dumps(metadata, default=str))

    def encode(self, prompt: str, system: str | None = None) -> dict:
        """Render one user turn, after an optional system turn, and tokenize it without truncation."""
        variables = {name: value for name, value in CHAT_TEMPLATE_KWARGS.items() if name != "tokenize"}
        rendered = self._template.render(messages=chat_messages(prompt, system), tools=None, documents=None,
                                         **variables)
        ids = self._llm.tokenize(rendered.encode("utf-8"), add_bos=False, special=True)
        if len(ids) + self.max_new_tokens > self._context:
            raise ValueError(f"Input has {len(ids)} tokens; plus {self.max_new_tokens} requested output tokens "
                             f"exceeds the {self._context}-token context. Input was not truncated.")
        return {"rendered": rendered, "encoded": {"input_ids": ids}}

    def generate(self, chat_input: dict) -> dict:
        input_ids = chat_input["encoded"]["input_ids"]
        # llama.cpp keeps the last prompt and skips the part a new prompt repeats. Starting empty makes
        # every call process its whole prompt, as the transformers backend does.
        self._llm.reset()
        ids = []
        started = time.perf_counter()
        for token in self._llm.generate(input_ids, temp=0.0):
            ids.append(token)
            if self._ends(token) or len(ids) >= self.max_new_tokens:
                break
        seconds = time.perf_counter() - started
        return {
            "raw_output": self._decode(ids[:-1] if self._ends(ids[-1]) else ids),
            "raw_output_with_special_tokens": self._decode(ids),
            "input_tokens": len(input_ids), "output_tokens": len(ids), "output_token_ids": ids,
            "input_token_sha256": digest_json(input_ids),
            "rendered_prompt_sha256": digest_text(chat_input["rendered"]),
            "seconds": seconds,
            "reached_output_token_limit": len(ids) >= self.max_new_tokens,
        }

    def _ends(self, token: int) -> bool:
        return llama_cpp.llama_vocab_is_eog(self._vocab, token)

    def _decode(self, ids: list) -> str:
        # Llama.detokenize drops every token whose text is longer than its 32-byte buffer. llama.cpp answers
        # a buffer that is too small with the negative of the size the token needs.
        text = b""
        for token in ids:
            buffer = ctypes.create_string_buffer(32)
            length = llama_cpp.llama_token_to_piece(self._vocab, token, buffer, len(buffer), 0, True)
            if length < 0:
                buffer = ctypes.create_string_buffer(-length)
                length = llama_cpp.llama_token_to_piece(self._vocab, token, buffer, len(buffer), 0, True)
            if length < 0:
                raise RuntimeError(f"llama.cpp could not give the text of token {token}")
            text += buffer.raw[:length]
        # A token limit can cut a character in half; show that as U+FFFD, as transformers does.
        return text.decode("utf-8", errors="replace")
