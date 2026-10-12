"""Plan and verify JEV base-model quantization without materializing base weights."""

from collections import Counter

from accelerate import init_empty_weights
from bitsandbytes.nn import Linear4bit, Linear8bitLt
import torch
from transformers import AutoModelForCausalLM
from transformers.core_model_loading import build_glob_alternation
from transformers.quantizers.base import HfQuantizer
from transformers.quantizers.quantizers_utils import should_convert_module

from ..quantization import bitsandbytes_config

NF4_BLOCK_SIZE = 64


def plan_quantization(model_config, precision, dtype):
    """Estimate tensor storage using the exact modules Transformers will convert."""
    if precision not in ("int8", "nf4"):
        raise ValueError("JEV quantization precision must be int8 or nf4")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Quantized JEV requires float16, bfloat16, or float32 compute weights")
    quantization = bitsandbytes_config(precision, dtype)
    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(
            model_config, dtype=dtype, trust_remote_code=False, attn_implementation="sdpa")
    model.tie_weights()
    if any(tensor.device.type != "meta" for tensor in (*model.parameters(), *model.buffers())):
        raise RuntimeError("Quantization planning unexpectedly materialized model tensors")
    exclusions = sorted(HfQuantizer.get_modules_to_not_convert(
        model, quantization.llm_int8_skip_modules, model._keep_in_fp32_modules))
    quantization.llm_int8_skip_modules = exclusions
    modules = {name: module for name, module in model.named_modules()
               if type(module) is torch.nn.Linear and should_convert_module(name, exclusions)}
    if not modules:
        raise ValueError("No supported linear modules remain for JEV quantization")
    dtype_plan = model._get_dtype_plan(dtype)
    matcher, groups, _ = build_glob_alternation(list(dtype_plan)) if dtype_plan else (None, {}, {})
    parameters = dict(model.named_parameters())
    shapes = {}
    quantized_weight_bytes = quantization_state_bytes = largest_linear = quantized_parameters = 0
    for name, module in modules.items():
        key = name + ".weight"
        if key not in parameters or parameters[key] is not module.weight:
            raise ValueError(f"Quantized module has an unsupported shared or missing weight: {name}")
        shape = tuple(module.weight.shape)
        if shape != (module.out_features, module.in_features) or min(shape) < 1:
            raise ValueError(f"Quantized module has an invalid linear weight shape: {name}")
        count = module.weight.numel()
        shapes[name] = list(shape)
        quantized_parameters += count
        match = matcher.search(key) if matcher is not None else None
        loading_dtype = dtype_plan[groups[match.lastgroup]] if match is not None else module.weight.dtype
        largest_linear = max(largest_linear, count * torch.empty((), device="meta", dtype=loading_dtype).element_size())
        if precision == "int8":
            quantized_weight_bytes += count
            quantization_state_bytes += 4 * module.out_features
        else:
            quantized_weight_bytes += (count + 1) // 2
            quantization_state_bytes += 4 * ((count + NF4_BLOCK_SIZE - 1) // NF4_BLOCK_SIZE) + 16 * 4
    unquantized_bytes = 0
    dtype_counts = Counter()
    quantized_keys = {name + ".weight" for name in modules}
    for name, parameter in parameters.items():
        if name in quantized_keys:
            continue
        match = matcher.search(name) if matcher is not None else None
        target_dtype = dtype_plan[groups[match.lastgroup]] if match is not None else parameter.dtype
        size = parameter.numel() * torch.empty((), device="meta", dtype=target_dtype).element_size()
        unquantized_bytes += size
        dtype_counts[str(target_dtype)] += size
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    total = quantized_weight_bytes + quantization_state_bytes + unquantized_bytes + buffer_bytes
    plan = {
        "precision": precision, "compute_dtype": str(dtype),
        "quantized_module_names": sorted(modules), "quantized_module_shapes": shapes,
        "excluded_module_names": exclusions, "quantized_parameter_count": quantized_parameters,
        "quantized_weight_bytes": quantized_weight_bytes, "quantization_state_bytes": quantization_state_bytes,
        "unquantized_parameter_bytes": unquantized_bytes, "unquantized_bytes_by_dtype": dict(dtype_counts),
        "model_buffer_bytes": buffer_bytes, "estimated_base_weight_bytes": total,
        "largest_unquantized_linear_bytes": largest_linear,
        "loading_staging_bytes": largest_linear * (2 if precision == "int8" else 1),
        "details": {
            "method": "Meta-model module selection matches Transformers bitsandbytes conversion; "
                      "unique parameters count tied weights once",
            "int8": "One byte per quantized weight plus one FP32 row scale per output feature",
            "nf4": "Two packed weights per uint8, one FP32 scale per 64 weights, and a 16-entry FP32 codebook "
                   "per converted module; double quantization disabled",
            "unquantized": "Embeddings, skipped linear modules, biases and nonlinear parameters keep their "
                           "configured loading dtype; model buffers retain their construction dtype",
            "loading": "Largest original converted linear tensor; INT8 allows both the staged input and its FP16 "
                       "quantizer copy, NF4 allows the staged input. Transformers loads on-the-fly quantization "
                       "synchronously",
            "scope": "Base tensor storage estimate only; excludes adapter, decision head, allocator overhead, "
                     "CUDA context, quantizer workspace and inference activations",
        },
    }
    return quantization, plan


def validate_quantization(model, precision, plan):
    """Require the planned bitsandbytes conversion before attaching the LoRA adapter."""
    if precision not in ("int8", "nf4") or plan.get("precision") != precision:
        raise ValueError("Quantization plan precision does not match the requested mode")
    names, shapes = plan.get("quantized_module_names"), plan.get("quantized_module_shapes")
    if (not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or not isinstance(shapes, dict) or set(shapes) != set(names)):
        raise ValueError("Quantization plan must identify every converted module and weight shape")
    found = {name: module for name, module in model.named_modules()
             if isinstance(module, (Linear8bitLt, Linear4bit))}
    if set(found) != set(names):
        raise RuntimeError(f"Quantized modules differ from plan: missing={sorted(set(names) - set(found))}, "
                           f"unexpected={sorted(set(found) - set(names))}")
    state_bytes = weight_bytes = 0
    for name, module in found.items():
        shape = shapes[name]
        if (not isinstance(shape, list) or len(shape) != 2 or any(type(value) is not int or value < 1 for value in shape)
                or (module.out_features, module.in_features) != tuple(shape)):
            raise RuntimeError(f"Quantized linear dimensions differ from plan: {name}")
        weight = module.weight
        if weight.device.type == "meta":
            raise RuntimeError(f"Quantized weight was not materialized: {name}")
        if precision == "int8":
            if (not isinstance(module, Linear8bitLt) or weight.dtype != torch.int8
                    or tuple(weight.shape) != tuple(shape) or weight.has_fp16_weights):
                raise RuntimeError(f"Expected materialized INT8 linear weights: {name}")
            scale = weight.SCB
            if (not isinstance(scale, torch.Tensor) or scale.dtype != torch.float32
                    or tuple(scale.shape) != (shape[0],) or scale.device != weight.device
                    or not torch.isfinite(scale).all().item()):
                raise RuntimeError(f"Invalid INT8 row-scale state: {name}")
            state_bytes += scale.numel() * scale.element_size()
        else:
            if not isinstance(module, Linear4bit) or weight.dtype != torch.uint8 or not weight.bnb_quantized:
                raise RuntimeError(f"Expected materialized packed NF4 linear weights: {name}")
            state = weight.quant_state
            count = shape[0] * shape[1]
            if (state is None or state.quant_type != "nf4" or state.blocksize != NF4_BLOCK_SIZE
                    or state.nested or tuple(state.shape) != tuple(shape)
                    or str(state.dtype) != plan.get("compute_dtype")
                    or str(module.compute_dtype) != plan.get("compute_dtype")
                    or weight.numel() != (count + 1) // 2):
                raise RuntimeError(f"NF4 quantization state differs from plan: {name}")
            for tensor, expected_count in ((state.absmax, (count + NF4_BLOCK_SIZE - 1) // NF4_BLOCK_SIZE),
                                           (state.code, 16)):
                if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                        or tensor.numel() != expected_count or tensor.device != weight.device
                        or not torch.isfinite(tensor).all().item()):
                    raise RuntimeError(f"Invalid NF4 block-scale or codebook state: {name}")
                state_bytes += tensor.numel() * tensor.element_size()
        weight_bytes += weight.numel() * weight.element_size()
    if weight_bytes != plan.get("quantized_weight_bytes") or state_bytes != plan.get("quantization_state_bytes"):
        raise RuntimeError("Actual quantized tensor storage differs from the planned representation")
    return {"precision": precision, "verified_quantized_module_count": len(found),
            "quantized_weight_bytes": weight_bytes, "quantization_state_bytes": state_bytes,
            "verification": "Exact planned modules, dimensions, packed weight dtypes and quantizer state "
                            "checked before LoRA attachment"}
