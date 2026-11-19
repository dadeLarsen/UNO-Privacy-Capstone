"""GGUF chat generation through llama.cpp, on tiny model files that these tests write."""

import json

import llama_cpp
import numpy as np
import pytest

from privacy_bench.models.gguf_chat_model import GgufChatModel

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
# Longer than the 32-byte buffer llama-cpp-python decodes a token into. No merge builds it from input text.
RULE = "-" * 40
# llama.cpp's byte-level BPE spells a newline as Ċ and a space as Ġ.
TOKENS = [*SPECIAL, "Ċ", "Ġ", *(chr(code) for code in range(33, 127)), RULE]
TEMPLATE = ("{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
            "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
            "{% if enable_thinking is false %}plain\n{% endif %}")
WIDTH = 8
SETTINGS = {"precision": "q4_k_m", "device": "cpu", "max_new_tokens": 3, "seed": 42}


def tiny_gguf(write_gguf, path, *, winner: str, file_type: int = 15):
    """A one-layer model that always predicts `winner`.

    Every token has the same embedding, the layer adds nothing to it, and only the winner's output row
    points along it.
    """
    embedding, output = np.zeros((len(TOKENS), WIDTH)), np.zeros((len(TOKENS), WIDTH))
    embedding[:, 0] = output[TOKENS.index(winner), 0] = 1
    ones, square = np.ones(WIDTH), np.zeros((WIDTH, WIDTH))
    layer = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_down", "ffn_up")
    return write_gguf(path, {
        "general.architecture": "llama", "general.file_type": file_type,
        "llama.context_length": 64, "llama.embedding_length": WIDTH, "llama.block_count": 1,
        "llama.feed_forward_length": WIDTH, "llama.attention.head_count": 2,
        "llama.attention.layer_norm_rms_epsilon": 1e-5,
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "default", "tokenizer.ggml.tokens": TOKENS,
        "tokenizer.ggml.token_type": [3] * len(SPECIAL) + [1] * (len(TOKENS) - len(SPECIAL)),
        "tokenizer.ggml.merges": [], "tokenizer.ggml.eos_token_id": TOKENS.index("<|im_end|>"),
        "tokenizer.ggml.add_bos_token": False, "tokenizer.chat_template": TEMPLATE,
    }, {"token_embd.weight": embedding, "output_norm.weight": ones, "output.weight": output,
        "blk.0.attn_norm.weight": ones, "blk.0.ffn_norm.weight": ones,
        **{f"blk.0.{name}.weight": square for name in layer}})


@pytest.fixture(scope="module")
def repeater(tmp_path_factory, write_gguf):
    return tiny_gguf(write_gguf, tmp_path_factory.mktemp("gguf") / "repeater.gguf", winner="a")


@pytest.mark.parametrize("settings, message", [
    ({"precision": "nf4"}, "precision 'nf4'"),
    ({"device": "cuda"}, "device 'cuda'"),
    ({"max_new_tokens": 0}, "max_new_tokens"),
    ({"seed": -1}, "seed"),
])
def test_invalid_settings_fail_before_loading(tmp_path, settings, message):
    with pytest.raises(ValueError, match=message):
        GgufChatModel(tmp_path / "model.gguf", **{**SETTINGS, **settings})


def test_a_missing_file_and_a_file_of_another_precision_are_refused(tmp_path, write_gguf):
    with pytest.raises(ValueError, match="pbench download"):
        GgufChatModel(tmp_path / "absent.gguf", **SETTINGS).load()
    with pytest.raises(ValueError, match="declares file type 7, not the 15 of precision q4_k_m"):
        GgufChatModel(tiny_gguf(write_gguf, tmp_path / "q8.gguf", winner="a", file_type=7), **SETTINGS).load()


def test_a_gpu_condition_is_refused_by_a_build_without_a_gpu_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(llama_cpp, "llama_supports_gpu_offload", lambda: False)
    with pytest.raises(RuntimeError, match="no GPU backend"):
        GgufChatModel(tmp_path / "model.gguf", **{**SETTINGS, "device": "gpu"}).load()


def test_tiny_file_loads_on_the_cpu_and_records_what_ran(repeater):
    model = GgufChatModel(repeater, **SETTINGS)
    metadata = model.load()
    # No GPU device, so a build with a GPU backend cannot take a cpu run's prompt batches.
    placement = model._llm.model_params
    assert (placement.n_gpu_layers, placement.split_mode, placement.main_gpu) == (0, llama_cpp.LLAMA_SPLIT_MODE_NONE,
                                                                                  -1)
    assert (metadata["backend"], metadata["device"], metadata["precision"]) == ("llama-cpp-chat-generation", "cpu",
                                                                                "q4_k_m")
    assert (metadata["file"], metadata["file_bytes"]) == ("repeater.gguf", repeater.stat().st_size)
    assert metadata["gguf_metadata"]["general.architecture"] == "llama"
    assert "tokenizer.chat_template" not in metadata["gguf_metadata"] and len(metadata["chat_template_sha256"]) == 64
    assert metadata["llama_cpp_python_version"] == llama_cpp.__version__
    # The file's own context is shorter than the one this backend allocates.
    assert metadata["context_tokens"] == 64 and metadata["threads"] >= 1 and metadata["load_seconds"] >= 0
    json.dumps(metadata)


def test_encode_renders_the_file_template_with_thinking_disabled(repeater):
    model = GgufChatModel(repeater, **SETTINGS)
    model.load()
    chat = model.encode("we keep data .", system="be brief")
    assert chat["rendered"] == ("<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nwe keep data .<|im_end|>\n"
                                "<|im_start|>assistant\nplain\n")
    ids = chat["encoded"]["input_ids"]
    # Each special token is one ID, and every other character is its own.
    assert len(ids) == len(chat["rendered"]) - 3 * (len("<|im_start|>") - 1) - 2 * (len("<|im_end|>") - 1)
    assert ids[0] == TOKENS.index("<|im_start|>")
    with pytest.raises(ValueError, match="system"):
        model.encode("we keep data .", system="  ")


def test_long_input_is_rejected_at_encoding_without_truncation(repeater):
    model = GgufChatModel(repeater, **{**SETTINGS, "max_new_tokens": 40})
    model.load()
    with pytest.raises(ValueError, match="exceeds the 64-token context. Input was not truncated"):
        model.encode("we keep data .")


def test_generation_stops_at_the_token_limit_and_every_call_starts_empty(repeater, monkeypatch):
    model = GgufChatModel(repeater, **SETTINGS)
    model.load()
    chat = model.encode("we keep data .")
    evaluated, evaluate = [], model._llm.eval

    def record(tokens):
        evaluated.append(len(tokens))
        evaluate(tokens)

    monkeypatch.setattr(model._llm, "eval", record)
    first, second = model.generate(chat), model.generate(chat)
    # Each call evaluates its whole prompt and then the two tokens it generated before the last.
    assert evaluated == [len(chat["encoded"]["input_ids"]), 1, 1] * 2
    assert (first["raw_output"], first["raw_output_with_special_tokens"]) == ("aaa", "aaa")
    assert (first["output_tokens"], first["output_token_ids"]) == (3, [TOKENS.index("a")] * 3)
    assert first["input_tokens"] == len(chat["encoded"]["input_ids"]) and first["reached_output_token_limit"] is True
    assert len(first["input_token_sha256"]) == 64 and len(first["rendered_prompt_sha256"]) == 64
    assert first["seconds"] >= 0 and "peak_allocated_bytes" not in first
    assert {**second, "seconds": None} == {**first, "seconds": None}


def test_generation_stops_at_the_end_of_turn_token(tmp_path, write_gguf):
    model = GgufChatModel(tiny_gguf(write_gguf, tmp_path / "silent.gguf", winner="<|im_end|>"), **SETTINGS)
    model.load()
    result = model.generate(model.encode("we keep data ."))
    assert (result["raw_output"], result["raw_output_with_special_tokens"]) == ("", "<|im_end|>")
    assert (result["output_tokens"], result["output_token_ids"]) == (1, [TOKENS.index("<|im_end|>")])
    assert result["reached_output_token_limit"] is False


def test_a_token_longer_than_the_library_buffer_is_decoded_whole(tmp_path, write_gguf):
    model = GgufChatModel(tiny_gguf(write_gguf, tmp_path / "ruler.gguf", winner=RULE), **SETTINGS)
    model.load()
    assert model.generate(model.encode("we keep data ."))["raw_output"] == RULE * 3
