"""Greedy generation from a local chat model on one GPU, one prompt at a time, never truncating."""

from copy import deepcopy
import json
from pathlib import Path

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer,
                          GenerationConfig, set_seed)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from ..core.files import digest_json, digest_text
from .hub import context_limit, reject_prequantized, require_exact_load, validate_weights
from .measure import gpu_metadata, measured, require_cuda, timed_load
from .quantization import bitsandbytes_config, install_int8_cast_log_filter

PRECISIONS = ("bf16", "int8", "nf4")
CHAT_TEMPLATE_KWARGS = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
LOCAL = {"local_files_only": True, "trust_remote_code": False}
# transformers flags these tokenizers' pre-tokenizer regex from the config's transformers_version alone.
# Ministral 3 already ships the corrected regex (identical IDs on every OPP-115 sentence either way), so
# the flag only applies the fix to checkpoints that still need it and silences the warning on the rest.
MISTRAL_TOKENIZER_TYPES = ("mistral", "mistral3", "ministral", "ministral3", "pixtral", "voxtral")


def model_class(model_type: str):
    """Text-only causal LMs and multimodal chat models such as Ministral 3 both generate from text alone."""
    return AutoModelForCausalLM if model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES else AutoModelForImageTextToText


def chat_messages(prompt: str, system: str | None) -> list:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if system is not None and (not isinstance(system, str) or not system.strip()):
        raise ValueError("system must be None or a nonempty string")
    turns = [{"role": "user", "content": prompt}]
    return turns if system is None else [{"role": "system", "content": system}, *turns]


def greedy_config(original, defaults, max_new_tokens):
    """Keep the checkpoint's special token IDs, but reset its decoding policy to library defaults."""
    generation = deepcopy(original)
    preserved = {"bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id"}
    for name, value in defaults.to_dict().items():
        if name not in preserved and not name.startswith("_") and name != "transformers_version":
            setattr(generation, name, deepcopy(value))
    generation.do_sample = False
    generation.num_beams = 1
    generation.num_beam_groups = 1
    generation.num_return_sequences = 1
    generation.max_new_tokens = max_new_tokens
    generation.min_new_tokens = 0
    generation.use_cache = True
    generation.return_dict_in_generate = False
    if generation.pad_token_id is None:
        eos = generation.eos_token_id
        generation.pad_token_id = eos[0] if isinstance(eos, (list, tuple)) and eos else eos
    generation.validate()
    return generation


def visible_token_ids(ids, generation):
    """Remove only trailing end-of-sequence and padding IDs; keep any reasoning or control text."""
    eos = generation.eos_token_id
    terminal = set(eos if isinstance(eos, (list, tuple)) else [eos])
    terminal.add(generation.pad_token_id)
    end = len(ids)
    while end and ids[end - 1] in terminal:
        end -= 1
    return ids[:end]


class ChatModel:
    def __init__(self, snapshot: Path, *, precision: str, max_new_tokens: int, seed: int):
        if precision not in PRECISIONS:
            raise ValueError(f"Chat model precision must be one of {', '.join(PRECISIONS)}, not {precision!r}")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.snapshot, self.precision = Path(snapshot), precision
        self.max_new_tokens, self.seed = max_new_tokens, seed
        self._cuda, self._device = True, "cuda:0"
        self._model = self._tokenizer = self._generation = self._context_limit = None

    def load(self) -> dict:
        require_cuda(bf16=True)
        dtype = torch.bfloat16
        set_seed(self.seed)
        reject_prequantized(self.snapshot)
        validate_weights(self.snapshot)
        model_config = AutoConfig.from_pretrained(str(self.snapshot), **LOCAL)
        if getattr(model_config, "is_encoder_decoder", False):
            raise ValueError("This backend supports decoder-only causal language models")
        tokenizer_kwargs = {"fix_mistral_regex": True} if model_config.model_type in MISTRAL_TOKENIZER_TYPES else {}
        tokenizer = AutoTokenizer.from_pretrained(str(self.snapshot), **LOCAL, **tokenizer_kwargs)
        try:
            chat_template = tokenizer.get_chat_template()
        except (ValueError, AttributeError) as exc:
            raise ValueError("This checkpoint has no usable chat template; choose an instruction model") from exc
        if not chat_template:
            raise ValueError("This checkpoint has an empty chat template; choose an instruction model")
        limit, candidates = context_limit(model_config, tokenizer)
        free_gpu, _ = torch.cuda.mem_get_info(0)
        kwargs = {**LOCAL, "dtype": dtype, "attn_implementation": "sdpa", "device_map": {"": 0}}
        quantization = None
        if self.precision in ("int8", "nf4"):
            quantization = bitsandbytes_config(self.precision, dtype)
            kwargs["quantization_config"] = quantization
        if self.precision == "int8":
            install_int8_cast_log_filter()
        auto_class = model_class(model_config.model_type)
        try:
            (model, loading), load_seconds = timed_load(
                lambda: auto_class.from_pretrained(str(self.snapshot), **kwargs, output_loading_info=True), cuda=True)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("The model does not fit on CUDA device 0; choose int8 or nf4, or a smaller model") from exc
        model.eval()
        require_exact_load(loading)
        device_map = getattr(model, "hf_device_map", kwargs["device_map"])
        if any(str(device) not in ("0", "cuda", "cuda:0") for device in device_map.values()):
            raise RuntimeError(f"Expected the whole model on CUDA device 0, got {device_map!r}")
        self._generation = greedy_config(model.generation_config, GenerationConfig(), self.max_new_tokens)
        # generate() fills every unset field from model.generation_config before the library defaults.
        # Making the greedy config the model's own keeps checkpoint sampling settings out entirely.
        model.generation_config = self._generation
        self._model, self._tokenizer, self._context_limit = model, tokenizer, limit
        warnings = ["enable_thinking=False is passed to the chat template; only templates that implement it "
                    "disable reasoning."]
        if self.precision == "int8":
            warnings.append("bitsandbytes casts BF16 activations to FP16 before INT8 quantization; that log "
                            "message is shown once per process.")
        metadata = {
            "backend": "transformers-chat-generation", **gpu_metadata(),
            "gpu_free_before_load_bytes": free_gpu, "load_seconds": load_seconds,
            "precision": self.precision, "nonquantized_dtype": str(dtype),
            "quantization_config": quantization.to_dict() if quantization else None,
            "device_map": device_map, "auto_class": auto_class.__name__, "model_class": type(model).__name__,
            "model_type": model_config.model_type, "tokenizer_kwargs": tokenizer_kwargs,
            "model_config_sha256": digest_text((self.snapshot / "config.json").read_text(encoding="utf-8")),
            "chat_template_sha256": digest_text(chat_template), "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            "context_limit": limit, "context_limit_candidates": candidates,
            "attention_implementation": getattr(model.config, "_attn_implementation", None),
            "generation_config": self._generation.to_dict(),
            "generation_defaults": "unset fields take library global defaults; checkpoint sampling settings unused",
            "seed": self.seed, "batch_size": 1,
            "timing_scope": "synchronized model.generate only; excludes tokenization, input transfer, "
                            "decoding, and model load",
            "memory_scope": "PyTorch CUDA device 0 peak allocated/reserved bytes; includes resident model "
                            "and KV cache, excludes CPU RAM",
            "output_decoding": "skip_special_tokens=False; only trailing EOS/PAD IDs removed from raw_output",
            "warnings": warnings,
        }
        return json.loads(json.dumps(metadata, default=str))

    def encode(self, prompt: str, system: str | None = None) -> dict:
        """Render one user turn, after an optional system turn, and tokenize it without truncation."""
        rendered = self._tokenizer.apply_chat_template(chat_messages(prompt, system), **CHAT_TEMPLATE_KWARGS)
        encoded = dict(self._tokenizer(rendered, add_special_tokens=False, truncation=False,
                                       return_attention_mask=True))
        count = len(encoded["input_ids"])
        if count + self.max_new_tokens > self._context_limit:
            raise ValueError(f"Input has {count} tokens; plus {self.max_new_tokens} requested output tokens "
                             f"exceeds the {self._context_limit}-token context limit. Input was not truncated.")
        return {"rendered": rendered, "encoded": encoded}

    def generate(self, chat_input: dict) -> dict:
        input_ids = chat_input["encoded"]["input_ids"]
        tensors = {name: torch.tensor([values], dtype=torch.long, device=self._device)
                   for name, values in chat_input["encoded"].items()}
        output, measurements = measured(lambda: self._model.generate(
            **tensors, generation_config=self._generation), cuda=self._cuda)
        ids = output[0, len(input_ids):].tolist()
        decode = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return {
            "raw_output": self._tokenizer.decode(visible_token_ids(ids, self._generation), **decode),
            "raw_output_with_special_tokens": self._tokenizer.decode(ids, **decode),
            "input_tokens": len(input_ids), "output_tokens": len(ids), "output_token_ids": ids,
            "input_token_sha256": digest_json(input_ids),
            "rendered_prompt_sha256": digest_text(chat_input["rendered"]),
            **measurements,
            "reached_output_token_limit": len(ids) >= self._generation.max_new_tokens,
        }
