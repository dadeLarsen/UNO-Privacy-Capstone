"""Offline JEV choice decisions from its trained LoRA adapter and calibrated decision head."""

import json
import math
from pathlib import Path

from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ...core.files import digest_bytes, digest_json
from ..hub import context_limit, reject_prequantized, require_exact_load
from ..measure import GIB, gpu_metadata, measured, require_cuda, timed_load, use_exact_fp32_matmul
from ..quantization import install_int8_cast_log_filter
from .bundle import validate_bundle
from .quantization import plan_quantization, validate_quantization

SLOT_RANGES = {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]}
VERBALIZERS = ["false", "true", *map(str, range(6)), *"ABCDEFGHIJKLMNOP"]
CHOICE_LETTERS = "ABCDEFGHIJKLMNOP"
MODES = (("cuda", "bf16"), ("cuda", "int8"), ("cuda", "nf4"), ("cpu", "fp32"))
LOCAL = {"local_files_only": True, "trust_remote_code": False}
ARTIFACTS = ("config.json", "judge_config.json", "calibration.json", "head.safetensors",
             "adapter/adapter_config.json", "adapter/adapter_model.safetensors",
             "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json")


def choice_prompt(state: str, question: str, options: list) -> str:
    """JEV's bare-v1 choice template, with options lettered from A."""
    if not isinstance(state, str) or not state.strip():
        raise ValueError("JEV requires a nonempty state text")
    if not 2 <= len(options) <= len(CHOICE_LETTERS):
        raise ValueError(f"JEV choice decisions take 2 to {len(CHOICE_LETTERS)} options")
    lines = "\n".join(f"{letter}) {option}" for letter, option in zip(CHOICE_LETTERS, options))
    return f"[kind] choice\n[state] {state}\n[question] {question}\n[options]\n{lines}\n[decision]:"


def validate_decision_artifacts(judge, calibration, adapter, head, hidden_size):
    """Reject incomplete or incompatible decision weights before allocating the base model."""
    if (judge.get("hidden_size") != hidden_size or judge.get("weights_mode") != "unmerged"
            or judge.get("adapter_subfolder") != "adapter" or judge.get("softcap") is not None):
        raise ValueError("Unsupported JEV hidden size, adapter layout, weights mode, or softcap")
    slots = judge.get("slots", {})
    if (not isinstance(slots, dict) or slots.get("num_slots") != 24 or slots.get("ranges") != SLOT_RANGES
            or slots.get("verbalizers") != VERBALIZERS or slots.get("template_version") != "bare-v1"):
        raise ValueError("JEV requires the published bare-v1 24-slot layout")
    ids = judge.get("verbalizer_ids")
    if (not isinstance(ids, list) or len(ids) != 24
            or any(type(value) is not int or value < 0 for value in ids) or len(set(ids)) != 24):
        raise ValueError("JEV verbalizer token IDs must be 24 distinct nonnegative integers")
    if set(head) != {"proj.weight", "proj.bias"}:
        raise ValueError("JEV head must contain exactly proj.weight and proj.bias")
    for key, shape in (("proj.weight", (24, hidden_size)), ("proj.bias", (24,))):
        tensor = head[key]
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != shape or not torch.isfinite(tensor).all().item():
            raise ValueError(f"Invalid JEV {key}: expected finite FP32 tensor of shape {shape}")
    temperatures = calibration.get("per_kind")
    if not isinstance(temperatures, dict) or set(temperatures) != set(SLOT_RANGES):
        raise ValueError("Calibration must provide temperatures for noul, choice, and score")
    for kind, value in temperatures.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid calibration temperature for {kind}: {value!r}")
    lora = judge.get("lora", {})
    mapping = adapter.get("auto_mapping")
    if (not isinstance(lora, dict) or not isinstance(mapping, dict)
            or adapter.get("peft_type") != "LORA" or adapter.get("task_type") is not None
            or mapping.get("base_model_class") != "Qwen3_5ForCausalLM"
            or adapter.get("r") != lora.get("r") or adapter.get("lora_alpha") != lora.get("alpha")
            or not isinstance(adapter.get("target_modules"), list)
            or set(adapter["target_modules"]) != set(lora.get("target_modules", []))
            or not adapter["target_modules"]):
        raise ValueError("LoRA configuration differs from the JEV judge configuration")
    return float(temperatures["choice"])


def validate_encoding(encoded, limit):
    if not isinstance(encoded, dict) or set(encoded) != {"input_ids", "attention_mask"}:
        raise ValueError("JEV encoding requires input_ids and attention_mask")
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    if (not isinstance(ids, list) or not ids or any(type(value) is not int or value < 0 for value in ids)
            or not isinstance(mask, list) or len(mask) != len(ids)
            or any(type(value) is not int or value != 1 for value in mask)):
        raise ValueError("JEV encoding must be one unpadded sequence with an all-one attention mask")
    if len(ids) > limit:
        raise ValueError(f"Full decision prompt has {len(ids)} tokens; limit is {limit}. "
                         "Input was not truncated. Raise max_input_tokens explicitly if appropriate.")
    return ids


def choice_distribution(hidden, weight, bias, temperature, option_count):
    """Read only the active choice slots, in option order."""
    if (hidden.ndim != 1 or weight.ndim != 2 or tuple(weight.shape) != (24, hidden.numel())
            or tuple(bias.shape) != (24,)):
        raise ValueError("Decision hidden state and head dimensions do not match")
    if not 2 <= option_count <= len(CHOICE_LETTERS):
        raise ValueError(f"JEV choice decisions take 2 to {len(CHOICE_LETTERS)} options")
    if not torch.isfinite(hidden).all().item():
        raise ValueError("JEV returned a nonfinite final hidden state")
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Choice calibration temperature must be finite and positive")
    start = SLOT_RANGES["choice"][0]
    logits = (weight[start:start + option_count] @ hidden.float() + bias[start:start + option_count]) / temperature
    if not torch.isfinite(logits).all().item():
        raise ValueError("JEV produced nonfinite decision logits")
    return logits, torch.softmax(logits, dim=0)


class JEVJudge:
    def __init__(self, snapshot: Path, *, precision: str, device: str, max_input_tokens: int,
                 min_free_vram_gib: float):
        if (device, precision) not in MODES:
            raise ValueError("JEV runs on CUDA with bf16, int8, or nf4, or on CPU with fp32")
        if type(min_free_vram_gib) not in (int, float) or not math.isfinite(min_free_vram_gib) or min_free_vram_gib < 0:
            raise ValueError("min_free_vram_gib must be finite and nonnegative")
        if type(max_input_tokens) is not int or max_input_tokens < 1:
            raise ValueError("max_input_tokens must be a positive integer")
        self.snapshot, self.precision, self.max_input_tokens = Path(snapshot), precision, max_input_tokens
        self._cuda = device == "cuda"
        self._device = "cuda:0" if self._cuda else "cpu"
        self._reserve_bytes = int(min_free_vram_gib * GIB)
        self._dtype = torch.bfloat16 if self._cuda else torch.float32
        self._model = self._backbone = self._tokenizer = self._weight = self._bias = self._temperature = None

    def _check_headroom(self):
        free, _ = torch.cuda.mem_get_info(0)
        if free < self._reserve_bytes:
            raise RuntimeError(f"Free VRAM {free / GIB:.2f} GiB is below the configured "
                               f"{self._reserve_bytes / GIB:.2f} GiB reserve")

    def load(self) -> dict:
        if self._cuda:
            require_cuda(bf16=True)
        precision_settings = use_exact_fp32_matmul()
        snapshot = validate_bundle(self.snapshot)
        raw = reject_prequantized(snapshot)
        if raw.get("model_type") != "qwen3_5_text":
            raise ValueError("Expected the qwen3_5_text JEV checkpoint")
        judge = json.loads((snapshot / "judge_config.json").read_text(encoding="utf-8"))
        calibration = json.loads((snapshot / "calibration.json").read_text(encoding="utf-8"))
        adapter = json.loads((snapshot / "adapter/adapter_config.json").read_text(encoding="utf-8"))
        head = load_file(str(snapshot / "head.safetensors"), device="cpu")
        temperature = validate_decision_artifacts(judge, calibration, adapter, head, raw["hidden_size"])
        model_config = AutoConfig.from_pretrained(str(snapshot), **LOCAL)
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), **LOCAL)
        model_limit, candidates = context_limit(model_config, tokenizer)
        if self.max_input_tokens > model_limit:
            raise ValueError(f"Requested prompt limit {self.max_input_tokens} exceeds model limit {model_limit}")
        for token, token_id in zip(VERBALIZERS, judge["verbalizer_ids"], strict=True):
            if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
                raise ValueError(f"Tokenizer identity differs for JEV decision token {token!r}")
        index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
        weight_bytes = index.get("metadata", {}).get("total_size")
        if type(weight_bytes) is not int or weight_bytes <= 0:
            raise ValueError("Base checkpoint index must record a positive total_size")
        quantized = self.precision in ("int8", "nf4")
        quantization, plan = (plan_quantization(model_config, self.precision, self._dtype)
                              if quantized else (None, None))
        estimated_base = plan["estimated_base_weight_bytes"] if quantized else weight_bytes
        staging = plan["loading_staging_bytes"] if quantized else 0
        # The adapter and decision head keep their floating-point weights in every mode.
        estimated_weights = (estimated_base + (snapshot / "adapter/adapter_model.safetensors").stat().st_size
                             + sum(tensor.numel() * tensor.element_size() for tensor in head.values()))
        headroom = self._reserve_bytes + staging
        free_before = None
        if self._cuda:
            free_before, _ = torch.cuda.mem_get_info(0)
            if free_before < estimated_weights + headroom:
                raise RuntimeError(f"Insufficient free VRAM before loading: {free_before / GIB:.2f} GiB free; "
                                   f"estimated weights {estimated_weights / GIB:.2f} GiB plus "
                                   f"{headroom / GIB:.2f} GiB loading headroom")
        if self.precision == "int8":
            install_int8_cast_log_filter()
        quantization_validation = None

        def load():
            nonlocal quantization_validation
            base, loading = AutoModelForCausalLM.from_pretrained(
                str(snapshot), **LOCAL, dtype=self._dtype, device_map={"": self._device},
                attn_implementation="sdpa", output_loading_info=True,
                **({"quantization_config": quantization} if quantized else {}))
            require_exact_load(loading)
            if quantized:
                quantization_validation = validate_quantization(base, self.precision, plan)
            model = PeftModel.from_pretrained(base, str(snapshot / "adapter"), is_trainable=False,
                                              local_files_only=True, torch_device=self._device)
            saved = load_file(str(snapshot / "adapter/adapter_model.safetensors"), device="cpu")
            loaded = get_peft_model_state_dict(model, save_embedding_layers=False)
            if set(saved) != set(loaded):
                raise RuntimeError(f"Loaded adapter keys differ: missing={sorted(set(saved) - set(loaded))}, "
                                   f"unexpected={sorted(set(loaded) - set(saved))}")
            for name, expected in saved.items():
                actual = loaded[name].detach().cpu().to(expected.dtype)
                if not torch.isfinite(expected).all().item() or not torch.equal(actual, expected):
                    raise RuntimeError(f"LoRA tensor did not load exactly: {name}")
            model.eval()
            return base, model

        (base, model), load_seconds = timed_load(load, cuda=self._cuda)
        # Keep the whole model resident, including the language-model head that decisions do not
        # read, so memory matches a deployment that also generates text with the same weights.
        self._model = model
        self._backbone = model.get_base_model().base_model
        self._weight, self._bias = head["proj.weight"].to(self._device), head["proj.bias"].to(self._device)
        self._tokenizer, self._temperature = tokenizer, temperature
        if self._cuda:
            self._check_headroom()
        metadata = {
            "backend": "transformers-jev-choice", "model_class": type(base).__name__,
            "model_type": model_config.model_type, "precision": self.precision, "dtype": str(self._dtype),
            "device": self._device, "device_map": {"": self._device}, "load_seconds": load_seconds,
            "context_limit": self.max_input_tokens, "context_limit_candidates": candidates,
            "base_context_limit": model_limit, "adapter_merged": False, "decision_head_dtype": "torch.float32",
            "quantization_config": quantization.to_dict() if quantized else None,
            "quantization_plan": plan, "quantization_validation": quantization_validation,
            "int8_activation_cast": "BF16 activations cast to FP16 by bitsandbytes; cast message logged once"
                                    if self.precision == "int8" else None,
            "estimated_resident_weight_bytes": estimated_weights if self._cuda else None,
            "loading_headroom_bytes": headroom if self._cuda else None,
            "choice_temperature": temperature, "batch_size": 1, **precision_settings,
            "selection": "argmax calibrated choice logits; first option wins exact ties",
            "artifact_sha256": {name: digest_bytes((snapshot / name).read_bytes()) for name in ARTIFACTS},
            "timing_scope": "synchronized backbone forward, FP32 decision head and softmax; "
                            "excludes tokenization, input transfer, output transfer, and model loading",
            "memory_scope": "PyTorch CUDA device 0 peak allocated/reserved bytes including all resident weights, "
                            "with the unused language-model head",
            "tokenizer_settings": {"add_special_tokens": False, "truncation": False, "padding": False},
            "gpu_free_before_load_bytes": free_before, "min_free_vram_bytes": self._reserve_bytes,
        }
        if self._cuda:
            metadata.update(gpu_metadata())
        return json.loads(json.dumps(metadata, default=str))

    def encode(self, prompt: str) -> dict:
        encoded = dict(self._tokenizer(prompt, add_special_tokens=False, truncation=False,
                                       padding=False, return_attention_mask=True))
        validate_encoding(encoded, self.max_input_tokens)
        return encoded

    def choose(self, encoded: dict, option_count: int) -> dict:
        """Return calibrated logits and probabilities over the first option_count choice slots."""
        ids = validate_encoding(encoded, self.max_input_tokens)
        if self._cuda:
            self._check_headroom()
        tensors = {key: torch.tensor([value], dtype=torch.long, device=self._device) for key, value in encoded.items()}

        def forward():
            hidden = self._backbone(**tensors, use_cache=False, return_dict=True).last_hidden_state
            if tuple(hidden.shape) != (1, len(ids), self._weight.shape[1]):
                raise ValueError(f"Unexpected JEV hidden-state shape: {tuple(hidden.shape)}")
            return choice_distribution(hidden[0, -1], self._weight, self._bias, self._temperature, option_count)

        (logits, probabilities), measurements = measured(forward, cuda=self._cuda)
        return {"logits": logits.cpu().tolist(), "probabilities": probabilities.cpu().tolist(),
                "input_tokens": len(ids), "input_token_ids": list(ids), "input_token_sha256": digest_json(ids),
                **measurements}
