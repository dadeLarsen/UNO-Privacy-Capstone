"""Chat generation logic with stand-in tokenizer and model; no checkpoint or GPU needed."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from privacy_bench.models.chat_model import ChatModel, greedy_config, model_class, visible_token_ids


class FakeGeneration(SimpleNamespace):
    def to_dict(self):
        return vars(self)

    def validate(self):
        pass


def test_greedy_resets_checkpoint_sampling_policy_without_mutating_the_original():
    original = FakeGeneration(do_sample=True, num_beams=4, num_beam_groups=2, num_return_sequences=3,
                              temperature=0.6, top_p=0.9, repetition_penalty=1.1, eos_token_id=[99, 98],
                              pad_token_id=None)
    defaults = FakeGeneration(do_sample=False, num_beams=1, num_beam_groups=1, num_return_sequences=1,
                              temperature=1.0, top_p=1.0, repetition_penalty=1.0, eos_token_id=None,
                              pad_token_id=None)
    result = greedy_config(original, defaults, 32)
    assert (result.temperature, result.top_p, result.repetition_penalty) == (1.0, 1.0, 1.0)
    assert (result.do_sample, result.num_beams, result.num_return_sequences) == (False, 1, 1)
    assert (result.eos_token_id, result.pad_token_id, result.max_new_tokens) == ([99, 98], 99, 32)
    assert original.do_sample


def test_only_terminal_eos_and_padding_are_removed():
    generation = SimpleNamespace(eos_token_id=[99, 98], pad_token_id=0)
    assert visible_token_ids([17, 99, 22, 98, 0], generation) == [17, 99, 22]
    assert visible_token_ids([17, 22], generation) == [17, 22]


@pytest.mark.parametrize("settings, message", [
    ({"precision": "fp16"}, "precision"),
    ({"max_new_tokens": 0}, "max_new_tokens"),
    ({"seed": -1}, "seed"),
])
def test_invalid_settings_fail_before_loading(tmp_path, settings, message):
    with pytest.raises(ValueError, match=message):
        ChatModel(tmp_path, **{"precision": "int8", "max_new_tokens": 32, "seed": 42, **settings})


class FakeTokenizer:
    def __init__(self, ids):
        self.ids, self.calls = ids, []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered prompt"

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"input_ids": list(self.ids), "attention_mask": [1] * len(self.ids)}

    @staticmethod
    def decode(ids, **kwargs):
        vocab = {4: "</think>", 5: '{"label_id":1}', 99: "<eos>"}
        return "".join(vocab[token] for token in ids)


def cpu_model(tokenizer, *, limit=100, max_new_tokens=32):
    model = ChatModel(".", precision="int8", max_new_tokens=max_new_tokens, seed=42)
    model._cuda, model._device = False, "cpu"
    model._tokenizer, model._context_limit = tokenizer, limit
    model._generation = SimpleNamespace(max_new_tokens=max_new_tokens, eos_token_id=99, pad_token_id=99)
    model._model = Mock()
    return model


def test_long_input_is_rejected_at_encoding_without_truncation():
    model = cpu_model(FakeTokenizer(range(10)), limit=12, max_new_tokens=4)
    with pytest.raises(ValueError, match="Input was not truncated"):
        model.encode("example")
    assert model._tokenizer.calls[-1][1]["truncation"] is False
    model._model.generate.assert_not_called()


def test_encode_renders_one_user_turn_with_thinking_disabled():
    tokenizer = FakeTokenizer([1, 2, 3])
    encoded = cpu_model(tokenizer).encode("Classify this")
    messages, kwargs = tokenizer.calls[0]
    assert messages == [{"role": "user", "content": "Classify this"}]
    assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    assert encoded == {"rendered": "rendered prompt", "encoded": {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}}


def test_encode_puts_an_optional_system_turn_first():
    tokenizer = FakeTokenizer([1, 2, 3])
    cpu_model(tokenizer).encode("Summarize this", system="You summarize policies.")
    assert tokenizer.calls[0][0] == [{"role": "system", "content": "You summarize policies."},
                                     {"role": "user", "content": "Summarize this"}]
    with pytest.raises(ValueError, match="system"):
        cpu_model(tokenizer).encode("Summarize this", system="  ")


def test_multimodal_chat_models_load_through_the_image_text_class():
    assert model_class("qwen3") is AutoModelForCausalLM
    assert model_class("mistral3") is AutoModelForImageTextToText


def test_generate_keeps_reasoning_text_and_reports_token_accounting():
    model = cpu_model(FakeTokenizer([1, 2, 3]))
    model._model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 99]])
    result = model.generate(model.encode("example"))
    assert result["raw_output"] == '</think>{"label_id":1}'
    assert result["raw_output_with_special_tokens"] == '</think>{"label_id":1}<eos>'
    assert (result["input_tokens"], result["output_tokens"], result["output_token_ids"]) == (3, 3, [4, 5, 99])
    assert len(result["input_token_sha256"]) == 64 and len(result["rendered_prompt_sha256"]) == 64
    assert result["reached_output_token_limit"] is False and result["seconds"] >= 0
    kwargs = model._model.generate.call_args.kwargs
    assert kwargs["generation_config"] is model._generation
    assert kwargs["input_ids"].tolist() == [[1, 2, 3]] and kwargs["attention_mask"].tolist() == [[1, 1, 1]]
