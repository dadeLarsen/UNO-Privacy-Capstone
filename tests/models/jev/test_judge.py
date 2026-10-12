"""JEV decision math, encoding checks, and a tiny local Qwen/PEFT roundtrip on CPU."""

import json
import math
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from privacy_bench.core.files import digest_json
from privacy_bench.models import measure
from privacy_bench.models.jev import judge as backend
from privacy_bench.models.jev.judge import JEVJudge, choice_distribution, choice_prompt, validate_decision_artifacts


def stub(hidden, limit=1024):
    judge = JEVJudge(".", precision="fp32", device="cpu", max_input_tokens=limit, min_free_vram_gib=0.0)
    judge._temperature = 1.0
    judge._weight = torch.zeros((24, 2), dtype=torch.float32)
    judge._bias = torch.zeros(24, dtype=torch.float32)
    judge._backbone = Mock(return_value=SimpleNamespace(last_hidden_state=hidden))
    return judge


def test_head_bias_and_choice_temperature_match_analytic_probabilities():
    hidden = torch.tensor([2.0, -3.0], dtype=torch.bfloat16)
    weight, bias = torch.zeros((24, 2)), torch.zeros(24)
    weight[8], weight[9] = torch.tensor([2.0, 1.0]), torch.tensor([-1.0, 0.0])
    bias[8], bias[9] = 3, -2
    logits, probabilities = choice_distribution(hidden, weight, bias, 2.0, 10)
    assert logits.dtype == probabilities.dtype == torch.float32
    assert logits.tolist() == [2.0, -2.0] + [0.0] * 8
    denominator = math.exp(2) + math.exp(-2) + 8
    expected = [math.exp(2) / denominator, math.exp(-2) / denominator] + [1 / denominator] * 8
    assert probabilities.tolist() == pytest.approx(expected, abs=torch.finfo(torch.float32).eps)
    _, uncalibrated = choice_distribution(hidden, weight, bias, 1.0, 10)
    assert uncalibrated[0] > probabilities[0]


def test_inactive_binary_score_and_unused_choice_slots_never_enter_the_softmax():
    weight, bias = torch.full((24, 2), 1e30), torch.full((24,), 1e30)
    weight[8:18], bias[8:18] = 0, 0
    logits, probabilities = choice_distribution(torch.tensor([1.0, 2.0]), weight, bias, 1.0, 10)
    assert logits.tolist() == [0.0] * 10
    torch.testing.assert_close(probabilities, torch.full((10,), 0.1))


def test_option_count_selects_how_many_choice_slots_are_read():
    weight, bias = torch.zeros((24, 2)), torch.zeros(24)
    logits, probabilities = choice_distribution(torch.zeros(2), weight, bias, 1.0, 3)
    assert len(logits) == 3 and probabilities.tolist() == pytest.approx([1 / 3] * 3)
    for count in (1, 17):
        with pytest.raises(ValueError, match="2 to 16 options"):
            choice_distribution(torch.zeros(2), weight, bias, 1.0, count)


@pytest.mark.parametrize("values", [
    (torch.zeros((1, 2)), torch.zeros((24, 2)), torch.zeros(24), 1.0),
    (torch.zeros(2), torch.zeros((23, 2)), torch.zeros(24), 1.0),
    (torch.zeros(2), torch.zeros((24, 3)), torch.zeros(24), 1.0),
    (torch.zeros(2), torch.zeros((24, 2)), torch.zeros((24, 1)), 1.0),
    (torch.tensor([float("nan"), 0]), torch.zeros((24, 2)), torch.zeros(24), 1.0),
    (torch.tensor([float("inf"), 0]), torch.zeros((24, 2)), torch.zeros(24), 1.0),
    *[(torch.zeros(2), torch.zeros((24, 2)), torch.zeros(24), t) for t in (0, -1, float("nan"), float("inf"), True)],
])
def test_invalid_head_dimensions_hidden_values_and_temperatures_are_rejected(values):
    with pytest.raises(ValueError):
        choice_distribution(*values, 10)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_decision_logits_are_rejected(value):
    bias = torch.zeros(24)
    bias[8] = value
    with pytest.raises(ValueError, match="nonfinite decision logits"):
        choice_distribution(torch.zeros(2), torch.zeros((24, 2)), bias, 1.0, 10)


def test_choice_prompt_keeps_the_text_verbatim_and_letters_each_option():
    text = 'Sentence with "quotes", unicode café and\na newline.'
    prompt = choice_prompt(text, "Pick one.", ["First: a", "Second: b"])
    assert prompt == f"[kind] choice\n[state] {text}\n[question] Pick one.\n[options]\nA) First: a\nB) Second: b\n[decision]:"
    for bad in (None, "", "   "):
        with pytest.raises(ValueError, match="nonempty state"):
            choice_prompt(bad, "Pick one.", ["a", "b"])


def test_choose_reports_logits_tokens_and_no_generation_fields():
    judge = stub(torch.tensor([[[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]]))
    judge._weight[8 + 4, 0] = 3
    result = judge.choose({"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]}, 10)
    assert max(range(10), key=result["logits"].__getitem__) == 4
    assert result["probabilities"][4] == pytest.approx(math.exp(3) / (math.exp(3) + 9), abs=1e-6)
    assert (result["input_tokens"], result["input_token_ids"]) == (3, [4, 5, 6])
    assert result["input_token_sha256"] == digest_json([4, 5, 6])
    assert "output_tokens" not in result and "peak_allocated_bytes" not in result


@pytest.mark.parametrize("encoded", [
    None, {}, {"input_ids": [1]}, {"attention_mask": [1]},
    {"input_ids": [1], "attention_mask": [1], "token_type_ids": [0]}, {"input_ids": [], "attention_mask": []},
    {"input_ids": [[1]], "attention_mask": [1]}, {"input_ids": [True], "attention_mask": [1]},
    {"input_ids": [-1], "attention_mask": [1]}, {"input_ids": [1.0], "attention_mask": [1]},
    {"input_ids": [1], "attention_mask": []}, {"input_ids": [1], "attention_mask": [0]},
    {"input_ids": [1], "attention_mask": [2]}, {"input_ids": [1], "attention_mask": [True]},
])
def test_only_exact_unpadded_encodings_are_accepted(encoded):
    judge = stub(torch.zeros((1, 1, 2)))
    with pytest.raises(ValueError):
        judge.choose(encoded, 10)
    judge._backbone.assert_not_called()


def test_complete_prompt_length_is_checked_without_truncation():
    judge = stub(torch.zeros((1, 4, 2)), limit=4)
    encoded = {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}
    judge._tokenizer = Mock(return_value=encoded)
    assert judge.encode("prompt") == encoded
    judge._tokenizer.assert_called_once_with("prompt", add_special_tokens=False, truncation=False, padding=False,
                                             return_attention_mask=True)
    assert judge.choose(encoded, 10)["input_tokens"] == 4
    judge._backbone.reset_mock()
    overflow = {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1] * 5}
    judge._tokenizer.return_value = overflow
    with pytest.raises(ValueError, match="Input was not truncated"):
        judge.encode("prompt")
    with pytest.raises(ValueError, match="Input was not truncated"):
        judge.choose(overflow, 10)
    judge._backbone.assert_not_called()


def test_backbone_runs_one_sequence_in_inference_mode_without_cache():
    judge = stub(torch.zeros((1, 3, 2)))

    def forward(**inputs):
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        assert inputs["input_ids"].tolist() == [[4, 5, 6]] and inputs["attention_mask"].tolist() == [[1, 1, 1]]
        assert inputs["use_cache"] is False and inputs["return_dict"] is True
        return SimpleNamespace(last_hidden_state=torch.zeros((1, 3, 2)))

    judge._backbone.side_effect = forward
    judge.choose({"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]}, 10)
    judge._backbone.assert_called_once()


@pytest.mark.parametrize("shape", [(3, 2), (2, 3, 2), (1, 2, 2), (1, 3, 3)])
def test_unexpected_hidden_shapes_are_rejected(shape):
    with pytest.raises(ValueError, match="hidden-state shape"):
        stub(torch.zeros(shape)).choose({"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]}, 10)


def test_cuda_measurement_checks_headroom_then_synchronizes_around_the_forward_pass():
    judge = stub(torch.zeros((1, 3, 2)))
    judge._cuda, judge._reserve_bytes, events = True, 100, []

    def forward(**inputs):
        events.append("forward")
        return SimpleNamespace(last_hidden_state=torch.zeros((1, 3, 2)))

    judge._backbone.side_effect = forward
    encoded = {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]}
    with patch.object(torch.cuda, "mem_get_info", side_effect=lambda device: events.append("headroom") or (200, 300)), \
            patch.object(torch.cuda, "synchronize", side_effect=lambda device: events.append("synchronize")), \
            patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=lambda device: events.append("reset")), \
            patch.object(torch.cuda, "max_memory_allocated", return_value=150), \
            patch.object(torch.cuda, "max_memory_reserved", return_value=175), \
            patch.object(measure.time, "perf_counter", side_effect=[10.0, 10.25]):
        result = judge.choose(encoded, 10)
    assert events == ["headroom", "synchronize", "reset", "forward", "synchronize"]
    assert (result["seconds"], result["peak_allocated_bytes"], result["peak_reserved_bytes"]) == (0.25, 150, 175)
    judge._backbone.reset_mock()
    with patch.object(torch.cuda, "mem_get_info", return_value=(99, 300)), \
            pytest.raises(RuntimeError, match="below the configured"):
        judge.choose(encoded, 10)
    judge._backbone.assert_not_called()


@pytest.mark.parametrize("precision", ["int8", "nf4", "bf16"])
def test_quantization_and_bf16_are_cuda_only(precision):
    with pytest.raises(ValueError, match="CUDA"):
        JEVJudge(".", precision=precision, device="cpu", max_input_tokens=1024, min_free_vram_gib=0.0)


def test_published_null_task_type_and_qwen_mapping_accept_the_choice_temperature(decision_artifacts):
    judge, calibration, adapter, head = decision_artifacts()
    assert validate_decision_artifacts(judge, calibration, adapter, head, 2) == 2.0


@pytest.mark.parametrize("path, value", [
    (("hidden_size",), 3), (("weights_mode",), "merged"), (("adapter_subfolder",), "adapter_vllm"),
    (("softcap",), 30.0), (("slots", "num_slots"), 23), (("slots", "template_version"), "chat-v1"),
    (("slots", "ranges", "choice"), [7, 23]), (("slots", "verbalizers"), list(reversed(backend.VERBALIZERS))),
    (("verbalizer_ids",), list(range(23))), (("verbalizer_ids",), [0] * 24),
    (("verbalizer_ids",), [-1, *range(23)]), (("verbalizer_ids",), [True, *range(2, 25)]),
])
def test_incompatible_judge_layouts_and_verbalizers_are_rejected(decision_artifacts, path, value):
    judge, calibration, adapter, head = decision_artifacts()
    target = judge
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_decision_artifacts(judge, calibration, adapter, head, 2)


@pytest.mark.parametrize("head", [
    {"proj.weight": torch.zeros((24, 2))},
    {"proj.weight": torch.zeros((24, 2)), "proj.bias": torch.zeros(24), "extra": torch.zeros(1)},
    {"proj.weight": torch.zeros((23, 2)), "proj.bias": torch.zeros(24)},
    {"proj.weight": torch.zeros((24, 2), dtype=torch.bfloat16), "proj.bias": torch.zeros(24)},
    {"proj.weight": torch.zeros((24, 2)), "proj.bias": torch.zeros(24, dtype=torch.float64)},
    {"proj.weight": torch.zeros((24, 2)), "proj.bias": torch.zeros((24, 1))},
    {"proj.weight": torch.full((24, 2), float("nan")), "proj.bias": torch.zeros(24)},
    {"proj.weight": torch.zeros((24, 2)), "proj.bias": torch.full((24,), float("inf"))},
])
def test_head_requires_exact_finite_fp32_weights_and_bias(decision_artifacts, head):
    judge, calibration, adapter, _ = decision_artifacts()
    with pytest.raises(ValueError):
        validate_decision_artifacts(judge, calibration, adapter, head, 2)


def invalid_calibrations():
    cases = [{}, {"per_kind": None}, {"per_kind": {"choice": 1}},
             {"per_kind": {"noul": 1, "score": 1, "choice": 1, "extra": 1}}]
    for kind in backend.SLOT_RANGES:
        for value in (0, -1, True, "1.0", float("nan"), float("inf")):
            calibration = {"per_kind": {"noul": 1, "score": 1, "choice": 1}}
            calibration["per_kind"][kind] = value
            cases.append(calibration)
    return cases


@pytest.mark.parametrize("calibration", invalid_calibrations())
def test_calibration_requires_each_kind_with_a_positive_finite_number(decision_artifacts, calibration):
    judge, _, adapter, head = decision_artifacts()
    with pytest.raises(ValueError):
        validate_decision_artifacts(judge, calibration, adapter, head, 2)


@pytest.mark.parametrize("key, value", [
    ("peft_type", "PREFIX_TUNING"), ("task_type", "SEQ_CLS"), ("task_type", "CAUSAL_LM"), ("auto_mapping", {}),
    ("auto_mapping", None), ("auto_mapping", []), ("auto_mapping", {"base_model_class": "Qwen3_5ForConditionalGeneration"}),
    ("r", 16), ("lora_alpha", 32), ("target_modules", []), ("target_modules", ["o_proj"]), ("target_modules", "q_proj"),
])
def test_adapter_configuration_must_match_the_trained_judge(decision_artifacts, key, value):
    judge, calibration, adapter, head = decision_artifacts()
    adapter[key] = value
    with pytest.raises(ValueError, match="LoRA configuration"):
        validate_decision_artifacts(judge, calibration, adapter, head, 2)


def load_tiny(tiny_bundle, **overrides):
    settings = {"precision": "fp32", "device": "cpu", "max_input_tokens": 1024, "min_free_vram_gib": 0.0, **overrides}
    judge = JEVJudge(tiny_bundle["path"], **settings)
    return judge, judge.load()


def test_tiny_local_qwen_peft_decision_matches_a_direct_reference(tiny_bundle):
    with patch.object(backend.AutoModelForCausalLM, "from_pretrained",
                      wraps=backend.AutoModelForCausalLM.from_pretrained) as loader, \
            patch.object(backend.PeftModel, "from_pretrained", wraps=backend.PeftModel.from_pretrained) as adapter_loader, \
            patch("peft.utils.save_and_load.check_file_exists_on_hf_hub",
                  side_effect=AssertionError("Local JEV inference attempted a Hub lookup")) as hub_check:
        judge, metadata = load_tiny(tiny_bundle)
    assert loader.call_args.kwargs["local_files_only"] and not loader.call_args.kwargs["trust_remote_code"]
    assert loader.call_args.kwargs["dtype"] == torch.float32
    assert adapter_loader.call_args.kwargs["local_files_only"]
    assert adapter_loader.call_args.kwargs["torch_device"] == "cpu"
    hub_check.assert_not_called()
    assert metadata["adapter_merged"] is False and metadata["model_type"] == "qwen3_5_text"
    assert not judge._model.training
    json.dumps(metadata)
    prompt = choice_prompt("Sentence café retention .", "Pick one.", [str(i) for i in range(10)])
    encoded = judge.encode(prompt)
    assert encoded == dict(tiny_bundle["tokenizer"](prompt, add_special_tokens=False, return_attention_mask=True))
    tensors = {key: torch.tensor([value]) for key, value in encoded.items()}
    with torch.inference_mode():
        active = judge._backbone(**tensors, use_cache=False, return_dict=True).last_hidden_state[0, -1]
        with judge._model.disable_adapter():
            disabled = judge._backbone(**tensors, use_cache=False, return_dict=True).last_hidden_state[0, -1]
    assert torch.max(torch.abs(active - disabled)).item() > 1e-4
    head = tiny_bundle["head"]
    expected = (head["proj.weight"] @ active + head["proj.bias"])[8:18] / 2.0
    result = judge.choose(encoded, 10)
    torch.testing.assert_close(torch.tensor(result["logits"]), expected)
    torch.testing.assert_close(torch.tensor(result["probabilities"]), torch.softmax(expected, dim=0))
    without_adapter = (head["proj.weight"] @ disabled + head["proj.bias"])[8:18] / 2.0
    assert not torch.allclose(torch.tensor(result["logits"]), without_adapter)
    assert judge.choose(encoded, 10)["logits"] == result["logits"]


def test_requested_limit_above_the_checkpoint_fails_before_weights_load(tiny_bundle):
    with patch.object(backend.AutoModelForCausalLM, "from_pretrained") as loader, \
            pytest.raises(ValueError, match="exceeds model limit"):
        load_tiny(tiny_bundle, max_input_tokens=2049)
    loader.assert_not_called()


def test_tokenizer_identity_mismatch_fails_before_weights_load(tiny_bundle):
    tokenizer = Mock(model_max_length=2048)
    tokenizer.encode.return_value = [999]
    with patch.object(backend.AutoTokenizer, "from_pretrained", return_value=tokenizer), \
            patch.object(backend.AutoModelForCausalLM, "from_pretrained") as loader, \
            pytest.raises(ValueError, match="Tokenizer identity differs"):
        load_tiny(tiny_bundle)
    loader.assert_not_called()


def test_incomplete_base_loading_is_rejected_before_the_adapter_attaches(tiny_bundle):
    loading = {"missing_keys": ["model.layers.0.mlp.down_proj.weight"], "unexpected_keys": [],
               "mismatched_keys": [], "error_msgs": []}
    with patch.object(backend.AutoModelForCausalLM, "from_pretrained", return_value=(Mock(), loading)), \
            patch.object(backend.PeftModel, "from_pretrained") as adapter_loader, \
            pytest.raises(RuntimeError, match="model.layers.0.mlp.down_proj.weight"):
        load_tiny(tiny_bundle)
    adapter_loader.assert_not_called()


def cuda_patches(free_bytes):
    return [patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "is_bf16_supported", return_value=True),
            patch.object(torch.cuda, "set_device"),
            patch.object(torch.cuda, "mem_get_info", return_value=(free_bytes, 4 * free_bytes)),
            patch.object(torch.cuda, "synchronize")]


@pytest.mark.parametrize("precision", ["int8", "nf4"])
def test_cuda_quantization_passes_the_planned_config_without_cpu_offload(tiny_bundle, precision):
    quantization = object()
    plan = {"estimated_base_weight_bytes": 1024, "loading_staging_bytes": 512}
    patches = cuda_patches(2 * measure.GIB)
    with patch.object(backend, "plan_quantization", return_value=(quantization, plan)) as planner, \
            patch.object(backend.AutoModelForCausalLM, "from_pretrained",
                         side_effect=RuntimeError("Reached quantized loader")) as loader, \
            patches[0], patches[1], patches[2], patches[3], patches[4], \
            pytest.raises(RuntimeError, match="Reached quantized loader"):
        load_tiny(tiny_bundle, device="cuda", precision=precision)
    assert planner.call_args.args[1:] == (precision, torch.bfloat16)
    kwargs = loader.call_args.kwargs
    assert kwargs["device_map"] == {"": "cuda:0"} and kwargs["dtype"] == torch.bfloat16
    assert kwargs["quantization_config"] is quantization
    assert kwargs["local_files_only"] and not kwargs["trust_remote_code"]


@pytest.mark.parametrize("precision", ["int8", "nf4"])
def test_cuda_precheck_reserves_weights_adapter_head_and_loading_staging(tiny_bundle, precision):
    head_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tiny_bundle["head"].values())
    adapter_bytes = (tiny_bundle["path"] / "adapter/adapter_model.safetensors").stat().st_size
    plan = {"estimated_base_weight_bytes": 1024, "loading_staging_bytes": 512 if precision == "int8" else 256}
    required = 1024 + adapter_bytes + head_bytes + plan["loading_staging_bytes"]
    patches = cuda_patches(required - 1)
    with patch.object(backend, "plan_quantization", return_value=(object(), plan)), \
            patch.object(backend.AutoModelForCausalLM, "from_pretrained") as loader, \
            patches[0], patches[1], patches[2], patches[3], patches[4], \
            pytest.raises(RuntimeError, match="Insufficient free VRAM before loading"):
        load_tiny(tiny_bundle, device="cuda", precision=precision)
    loader.assert_not_called()
