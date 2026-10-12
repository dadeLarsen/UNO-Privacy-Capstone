"""Quantization estimates on meta models and post-load checks on small real bitsandbytes layers."""

from copy import deepcopy
from unittest.mock import patch

from bitsandbytes.nn import Linear4bit, Linear8bitLt
import pytest
import torch

from privacy_bench.models.jev import quantization


def small_quantized_model(precision):
    """Real bitsandbytes CPU packing, without a CUDA allocation."""
    model = torch.nn.Module()
    if precision == "int8":
        model.linear = Linear8bitLt(64, 8, bias=False, has_fp16_weights=False)
    else:
        model.linear = Linear4bit(64, 8, bias=False, compute_dtype=torch.bfloat16, compress_statistics=False,
                                  quant_type="nf4", quant_storage=torch.uint8)
    model.linear.weight.data = torch.arange(512, dtype=torch.float32).reshape(8, 64).to(torch.bfloat16) / 512
    model.to("cpu")
    plan = {"precision": precision, "compute_dtype": "torch.bfloat16",
            "quantized_module_names": ["linear"], "quantized_module_shapes": {"linear": [8, 64]},
            "quantized_weight_bytes": 512 if precision == "int8" else 256,
            "quantization_state_bytes": 8 * 4 if precision == "int8" else 8 * 4 + 16 * 4}
    return model, plan


def test_int8_counts_linear_weights_scales_and_unquantized_embeddings_separately(text_config):
    config, plan = quantization.plan_quantization(text_config(), "int8", torch.bfloat16)
    assert (config.load_in_8bit, config.load_in_4bit, config.llm_int8_threshold) == (True, False, 6.0)
    assert not config.llm_int8_has_fp16_weight and not config.llm_int8_enable_fp32_cpu_offload
    assert plan["quantized_parameter_count"] == plan["quantized_weight_bytes"] == 20736
    assert plan["quantization_state_bytes"] == 584 * 4
    assert plan["unquantized_parameter_bytes"] == 8640 * 2
    assert plan["model_buffer_bytes"] == 2 * 4 * 4
    assert plan["estimated_base_weight_bytes"] == 40384
    assert plan["largest_unquantized_linear_bytes"] == 2048 * 2
    assert plan["loading_staging_bytes"] == 2048 * 4
    assert len(plan["quantized_module_names"]) == 15
    assert "lm_head" not in plan["quantized_module_names"] and "model.embed_tokens" not in plan["quantized_module_names"]
    assert "model.layers.0.linear_attn.conv1d" not in plan["quantized_module_names"]
    assert "model.layers.0.linear_attn.in_proj_qkv" in plan["quantized_module_names"]
    assert "lm_head" in config.llm_int8_skip_modules
    assert config.llm_int8_skip_modules == plan["excluded_module_names"]


def test_nf4_counts_block_scales_and_codebooks_without_double_quantization(text_config):
    config, plan = quantization.plan_quantization(text_config(), "nf4", torch.bfloat16)
    assert (config.load_in_4bit, config.load_in_8bit, config.bnb_4bit_quant_type) == (True, False, "nf4")
    assert config.bnb_4bit_compute_dtype == torch.bfloat16 and config.bnb_4bit_quant_storage == torch.uint8
    assert not config.bnb_4bit_use_double_quant
    assert plan["quantized_weight_bytes"] == 20736 // 2
    assert plan["quantization_state_bytes"] == 20736 // 64 * 4 + 15 * 16 * 4
    assert plan["unquantized_parameter_bytes"] == 8640 * 2
    assert plan["estimated_base_weight_bytes"] == 29936
    assert plan["loading_staging_bytes"] == plan["largest_unquantized_linear_bytes"]


def test_a_shared_embedding_and_lm_head_are_counted_once(text_config):
    _, untied = quantization.plan_quantization(text_config(), "int8", torch.bfloat16)
    _, tied = quantization.plan_quantization(text_config(tied=True), "int8", torch.bfloat16)
    assert tied["quantized_weight_bytes"] == untied["quantized_weight_bytes"]
    assert untied["unquantized_parameter_bytes"] - tied["unquantized_parameter_bytes"] == 128 * 32 * 2
    assert untied["estimated_base_weight_bytes"] - tied["estimated_base_weight_bytes"] == 128 * 32 * 2
    assert {"model.embed_tokens", "lm_head"} <= set(tied["excluded_module_names"])


def test_loading_dtype_changes_only_unquantized_storage_and_staging(text_config):
    _, half = quantization.plan_quantization(text_config(), "nf4", torch.bfloat16)
    _, full = quantization.plan_quantization(text_config(), "nf4", torch.float32)
    assert full["quantized_weight_bytes"] == half["quantized_weight_bytes"]
    assert full["quantization_state_bytes"] == half["quantization_state_bytes"]
    assert full["unquantized_parameter_bytes"] == half["unquantized_parameter_bytes"] * 2
    assert full["loading_staging_bytes"] == half["loading_staging_bytes"] * 2


def test_planning_materializes_no_tensors_and_never_loads_weights(text_config):
    original, observed = quantization.AutoModelForCausalLM.from_config, []

    def inspect_meta(*args, **kwargs):
        model = original(*args, **kwargs)
        observed.extend(tensor.device.type for tensor in (*model.parameters(), *model.buffers()))
        return model

    with patch.object(quantization.AutoModelForCausalLM, "from_config", side_effect=inspect_meta), \
            patch.object(quantization.AutoModelForCausalLM, "from_pretrained") as loader:
        quantization.plan_quantization(text_config(), "nf4", torch.bfloat16)
    assert observed and set(observed) == {"meta"}
    loader.assert_not_called()


@pytest.mark.parametrize("precision, dtype", [("bf16", torch.bfloat16), ("fp4", torch.bfloat16),
                                              ("int8", torch.int8), ("nf4", torch.float64)])
def test_unsupported_precision_or_dtype_fails_before_building_a_model(text_config, precision, dtype):
    with patch.object(quantization.AutoModelForCausalLM, "from_config") as factory, pytest.raises(ValueError):
        quantization.plan_quantization(text_config(), precision, dtype)
    factory.assert_not_called()


@pytest.mark.parametrize("precision", ["int8", "nf4"])
def test_real_small_quantized_weights_pass(precision):
    model, plan = small_quantized_model(precision)
    metadata = quantization.validate_quantization(model, precision, plan)
    assert metadata["verified_quantized_module_count"] == 1
    assert metadata["quantized_weight_bytes"] == plan["quantized_weight_bytes"]
    assert metadata["quantization_state_bytes"] == plan["quantization_state_bytes"]


@pytest.mark.parametrize("precision", ["int8", "nf4"])
def test_missing_and_extra_conversions_are_rejected(precision):
    model, plan = small_quantized_model(precision)
    model.linear = torch.nn.Linear(64, 8, bias=False)
    with pytest.raises(RuntimeError, match="missing"):
        quantization.validate_quantization(model, precision, plan)
    model, plan = small_quantized_model(precision)
    model.extra = Linear4bit(64, 8, bias=False)
    with pytest.raises(RuntimeError, match="unexpected"):
        quantization.validate_quantization(model, precision, plan)


def test_a_wrong_format_conversion_is_rejected():
    model, plan = small_quantized_model("nf4")
    plan["precision"] = "int8"
    with pytest.raises(RuntimeError, match="INT8"):
        quantization.validate_quantization(model, "int8", plan)


def test_unmaterialized_quantized_modules_fail():
    _, plan = small_quantized_model("nf4")
    with torch.device("meta"):
        model = torch.nn.Module()
        model.linear = Linear4bit(64, 8, bias=False)
    with pytest.raises(RuntimeError, match="not materialized"):
        quantization.validate_quantization(model, "nf4", plan)


def test_missing_int8_state_is_rejected():
    model, plan = small_quantized_model("int8")
    model.linear.weight.SCB = None
    with pytest.raises(RuntimeError, match="row-scale"):
        quantization.validate_quantization(model, "int8", plan)


@pytest.mark.parametrize("field, value", [("quant_type", "fp4"), ("blocksize", 128), ("nested", True),
                                          ("shape", (16, 32)), ("dtype", torch.float32)])
def test_a_wrong_nf4_state_is_rejected(field, value):
    model, plan = small_quantized_model("nf4")
    setattr(model.linear.weight.quant_state, field, value)
    with pytest.raises(RuntimeError, match="state differs"):
        quantization.validate_quantization(model, "nf4", plan)


@pytest.mark.parametrize("precision", ["int8", "nf4"])
def test_nonfinite_scales_are_rejected(precision):
    model, plan = small_quantized_model(precision)
    scales = model.linear.weight.SCB if precision == "int8" else model.linear.weight.quant_state.absmax
    scales[0] = float("nan")
    with pytest.raises(RuntimeError, match="scale"):
        quantization.validate_quantization(model, precision, plan)


@pytest.mark.parametrize("change", [
    {"precision": "int8"}, {"quantized_module_names": []}, {"quantized_module_names": ["linear", "linear"]},
    {"quantized_module_shapes": {}}, {"quantized_module_shapes": {"linear": [16, 32]}},
    {"quantized_weight_bytes": 255}, {"quantization_state_bytes": 95},
])
def test_incomplete_changed_or_corrupt_plans_are_rejected(change):
    model, original = small_quantized_model("nf4")
    plan = deepcopy(original)
    plan.update(change)
    with pytest.raises((ValueError, RuntimeError)):
        quantization.validate_quantization(model, "nf4", plan)
