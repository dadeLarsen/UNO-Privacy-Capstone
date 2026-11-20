"""GGUF sentence embeddings through llama.cpp, on tiny model files that these tests write."""

import json

import llama_cpp
import numpy as np
import pytest

from privacy_bench.models.gguf_sentence_embedder import GgufSentenceEmbedder

SPECIAL = ["<bos>", "<eos>"]
# llama.cpp's byte-level BPE spells a newline as Ċ and a space as Ġ.
TOKENS = [*SPECIAL, "Ċ", "Ġ", *(chr(code) for code in range(33, 127))]
WIDTH = 8
PROMPT = "q: "
SETTINGS = {"precision": "bf16", "device": "cpu", "max_input_tokens": 32}


def tiny_gguf(write_gguf, path, *, pooling: int | None = llama_cpp.LLAMA_POOLING_TYPE_MEAN, file_type: int = 32):
    """A one-layer model whose layer adds nothing, so every token's output is its own embedding.

    Token i's embedding points along dimension i % WIDTH, so a mean over tokens counts them by dimension.
    """
    embedding = np.zeros((len(TOKENS), WIDTH))
    embedding[np.arange(len(TOKENS)), np.arange(len(TOKENS)) % WIDTH] = 1
    ones, square = np.ones(WIDTH), np.zeros((WIDTH, WIDTH))
    layer = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_down", "ffn_up")
    return write_gguf(path, {
        "general.architecture": "llama", "general.file_type": file_type,
        "llama.context_length": 64, "llama.embedding_length": WIDTH, "llama.block_count": 1,
        "llama.feed_forward_length": WIDTH, "llama.attention.head_count": 2,
        "llama.attention.layer_norm_rms_epsilon": 1e-5,
        **({} if pooling is None else {"llama.pooling_type": pooling}),
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "default", "tokenizer.ggml.tokens": TOKENS,
        "tokenizer.ggml.token_type": [3] * len(SPECIAL) + [1] * (len(TOKENS) - len(SPECIAL)),
        "tokenizer.ggml.merges": [], "tokenizer.ggml.bos_token_id": 0, "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.add_bos_token": True, "tokenizer.ggml.add_eos_token": True,
        "tokenizer.chat_template": "{{ messages }}",
    }, {"token_embd.weight": embedding, "output_norm.weight": ones, "output.weight": np.zeros((len(TOKENS), WIDTH)),
        "blk.0.attn_norm.weight": ones, "blk.0.ffn_norm.weight": ones,
        **{f"blk.0.{name}.weight": square for name in layer}})


@pytest.fixture(scope="module")
def counter(tmp_path_factory, write_gguf):
    return tiny_gguf(write_gguf, tmp_path_factory.mktemp("gguf") / "counter.gguf")


def loaded(path, **settings):
    embedder = GgufSentenceEmbedder(path, **{**SETTINGS, **settings})
    return embedder, embedder.load()


@pytest.mark.parametrize("settings, message", [
    ({"precision": "fp32"}, "precision 'fp32'"),
    ({"device": "cuda"}, "device 'cuda'"),
    ({"max_input_tokens": 0}, "positive integer"),
])
def test_invalid_settings_fail_before_loading(tmp_path, settings, message):
    with pytest.raises(ValueError, match=message):
        GgufSentenceEmbedder(tmp_path / "model.gguf", **{**SETTINGS, **settings})


def test_a_missing_file_and_a_file_of_another_precision_are_refused(tmp_path, write_gguf):
    with pytest.raises(ValueError, match="pbench download"):
        loaded(tmp_path / "absent.gguf")
    with pytest.raises(ValueError, match="declares file type 7, not the 32 of precision bf16"):
        loaded(tiny_gguf(write_gguf, tmp_path / "q8.gguf", file_type=7))


def test_a_gpu_condition_is_refused_by_a_build_without_a_gpu_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(llama_cpp, "llama_supports_gpu_offload", lambda: False)
    with pytest.raises(RuntimeError, match="no GPU backend"):
        loaded(tmp_path / "model.gguf", device="gpu")


@pytest.mark.parametrize("pooling", [llama_cpp.LLAMA_POOLING_TYPE_CLS, None])
def test_a_file_that_does_not_pool_by_the_mean_is_refused(tmp_path, write_gguf, pooling):
    with pytest.raises(ValueError, match="not the mean"):
        loaded(tiny_gguf(write_gguf, tmp_path / "other.gguf", pooling=pooling))


def test_requested_limit_above_the_model_limit_is_rejected(counter):
    with pytest.raises(ValueError, match="exceeds the model limit 64"):
        loaded(counter, max_input_tokens=65)


def test_tiny_file_loads_on_the_cpu_and_records_what_ran(counter):
    embedder, metadata = loaded(counter)
    # No GPU device, so a build with a GPU backend cannot take a cpu run's batches.
    placement = embedder._llm.model_params
    assert (placement.n_gpu_layers, placement.split_mode, placement.main_gpu) == (0, llama_cpp.LLAMA_SPLIT_MODE_NONE,
                                                                                  -1)
    assert (metadata["backend"], metadata["device"], metadata["precision"]) == ("llama-cpp-embedding", "cpu", "bf16")
    assert (metadata["file"], metadata["file_bytes"]) == ("counter.gguf", counter.stat().st_size)
    assert (metadata["architecture"], metadata["pooling"], metadata["embedding_dimension"]) == ("llama", "mean", WIDTH)
    assert metadata["gguf_metadata"]["llama.pooling_type"] == "1"
    assert "tokenizer.chat_template" not in metadata["gguf_metadata"]
    assert (metadata["context_limit"], metadata["model_context_limit"]) == (32, 64)
    assert metadata["llama_cpp_python_version"] == llama_cpp.__version__
    assert metadata["threads"] >= 1 and metadata["load_seconds"] >= 0
    json.dumps(metadata)
    # The score array that llama-cpp-python allocates for sampling is gone; embedding never reads it.
    assert embedder._llm.scores.shape == (0, len(TOKENS))


def test_the_embedding_is_the_normalized_mean_over_every_token(counter):
    embedder, _ = loaded(counter)
    encoded = embedder.encode("ab ba", PROMPT)
    ids = encoded["input_ids"]
    # The file's start and end tokens surround the prompt and the text, one token per character.
    assert [TOKENS[token] for token in ids] == ["<bos>", "q", ":", "Ġ", "a", "b", "Ġ", "b", "a", "<eos>"]
    result = embedder.embed(encoded)
    counts = np.bincount(np.array(ids) % WIDTH, minlength=WIDTH)
    assert result["embedding"].dtype == np.float32 and result["embedding"].shape == (WIDTH,)
    assert result["embedding"] == pytest.approx(counts / np.linalg.norm(counts), abs=1e-6)
    assert (result["input_tokens"], result["input_token_ids"]) == (len(ids), ids)
    assert len(result["input_token_sha256"]) == 64 and result["seconds"] >= 0
    assert "peak_allocated_bytes" not in result


def test_every_call_starts_from_an_empty_context(counter):
    embedder, _ = loaded(counter)
    first = embedder.embed(embedder.encode("ab ba", PROMPT))["embedding"]
    other = embedder.embed(embedder.encode("zzzz", PROMPT))["embedding"]
    again = embedder.embed(embedder.encode("ab ba", PROMPT))["embedding"]
    assert np.array_equal(first, again) and not np.array_equal(first, other)


def test_inputs_over_the_limit_are_rejected_instead_of_truncated(counter):
    embedder, _ = loaded(counter, max_input_tokens=10)
    assert len(embedder.encode("ab ba", PROMPT)["input_ids"]) == 10
    with pytest.raises(ValueError, match="exceeds the 10-token limit. Input was not truncated"):
        embedder.encode("ab bab", PROMPT)
    for text in ("", "   ", None):
        with pytest.raises(ValueError, match="nonempty string"):
            embedder.encode(text, PROMPT)
