"""Sentence embeddings with a tiny local sentence-transformers checkpoint on CPU."""

import json
from unittest.mock import patch

import pytest
from sentence_transformers import SentenceTransformer
import torch

from privacy_bench.models import sentence_embedder as backend
from privacy_bench.models.sentence_embedder import SentenceEmbedder

PROMPT = "task: classification | query: "


def loaded(snapshot, limit=32):
    embedder = SentenceEmbedder(snapshot, precision="fp32", device="cpu", max_input_tokens=limit, text_only=False)
    return embedder, embedder.load()


@pytest.mark.parametrize("settings, message", [
    ({"precision": "fp16", "device": "cuda"}, "fp16"),
    ({"precision": "bf16", "device": "cpu"}, "fp16"),
    ({"max_input_tokens": 0}, "positive integer"),
    ({"text_only": "yes"}, "true or false"),
])
def test_invalid_settings_fail_before_loading(tmp_path, settings, message):
    values = {"precision": "fp32", "device": "cpu", "max_input_tokens": 32, "text_only": False, **settings}
    with pytest.raises(ValueError, match=message):
        SentenceEmbedder(tmp_path, **values)


def test_tiny_checkpoint_loads_locally_and_reports_its_pipeline(tiny_embedder):
    with patch.object(backend, "SentenceTransformer", wraps=SentenceTransformer) as loader:
        _, metadata = loaded(tiny_embedder)
    kwargs = loader.call_args.kwargs
    assert kwargs["local_files_only"] and not kwargs["trust_remote_code"]
    assert kwargs["model_kwargs"] == {"dtype": torch.float32} and kwargs["config_kwargs"] == {}
    assert metadata["pipeline"] == ["Transformer", "Pooling", "Normalize"]
    assert metadata["embedding_dimension"] == 16 and metadata["pooling"]["pooling_mode"] == "mean"
    assert (metadata["context_limit"], metadata["model_context_limit"]) == (32, 64)
    assert metadata["cuda_matmul_allow_tf32"] is False
    json.dumps(metadata)


def test_forward_embedding_equals_the_library_encode_with_the_same_prompt(tiny_embedder):
    embedder, _ = loaded(tiny_embedder)
    encoded = embedder.encode("we keep data for years .", PROMPT)
    assert embedder._model.tokenizer.decode(encoded["input_ids"]) == \
        "[CLS] task : classification | query : we keep data for years . [SEP]"
    result = embedder.embed(encoded)
    reference = embedder._model.encode("we keep data for years .", prompt=PROMPT, convert_to_tensor=True)
    assert torch.equal(result["embedding"], reference.float().cpu())
    assert result["embedding"].norm().item() == pytest.approx(1.0, abs=1e-6)
    assert (result["input_tokens"], len(result["input_token_sha256"])) == (len(encoded["input_ids"]), 64)
    assert result["seconds"] >= 0 and "peak_allocated_bytes" not in result


def test_embedding_leaves_the_encoded_input_as_it_was(tiny_embedder):
    # The harness keeps every encoded input for a whole run, so outputs added to one stay allocated.
    embedder, _ = loaded(tiny_embedder)
    encoded = embedder.encode("we keep data for years .", PROMPT)
    before = dict(encoded["features"])
    first = embedder.embed(encoded)
    assert encoded["features"].keys() == before.keys()
    assert all(encoded["features"][key] is value for key, value in before.items())
    assert torch.equal(embedder.embed(encoded)["embedding"], first["embedding"])


def test_inputs_over_the_limit_are_rejected_instead_of_truncated(tiny_embedder):
    embedder, _ = loaded(tiny_embedder, limit=12)
    assert len(embedder.encode("we keep data", PROMPT)["input_ids"]) == 11
    with pytest.raises(ValueError, match="Input was not truncated"):
        embedder.encode("we keep data for years", PROMPT)
    for text in ("", "   ", None):
        with pytest.raises(ValueError, match="nonempty string"):
            embedder.encode(text, PROMPT)


def test_requested_limit_above_the_model_limit_is_rejected(tiny_embedder):
    with pytest.raises(ValueError, match="exceeds the model limit 64"):
        loaded(tiny_embedder, limit=65)


def test_other_pipelines_are_rejected_before_loading(tiny_embedder, tmp_path):
    copy = tmp_path / "copy"
    copy.mkdir()
    for path in tiny_embedder.iterdir():
        if path.is_file():
            (copy / path.name).write_bytes(path.read_bytes())
    modules = json.loads((copy / "modules.json").read_text(encoding="utf-8"))
    (copy / "modules.json").write_text(json.dumps(modules[:2]), encoding="utf-8")
    with patch.object(backend, "SentenceTransformer") as loader, pytest.raises(ValueError, match="pipeline"):
        loaded(copy)
    loader.assert_not_called()


def test_text_only_drops_vision_and_audio_encoders_and_requires_them_to_exist(tmp_path):
    embedder = SentenceEmbedder(tmp_path, precision="fp32", device="cpu", max_input_tokens=8, text_only=True)
    assert embedder._config_overrides({"vision_config": {"x": 1}, "audio_config": {"y": 2}, "text_config": {}}) == \
        {"vision_config": None, "audio_config": None}
    assert embedder._config_overrides({"vision_config": {"x": 1}}) == {"vision_config": None}
    with pytest.raises(ValueError, match="multimodal checkpoint"):
        embedder._config_overrides({"text_config": {}})
    plain = SentenceEmbedder(tmp_path, precision="fp32", device="cpu", max_input_tokens=8, text_only=False)
    assert plain._config_overrides({"vision_config": {"x": 1}}) == {}
