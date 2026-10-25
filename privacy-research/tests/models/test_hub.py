import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from privacy_bench.models import hub

COMMIT = "a" * 40


@pytest.mark.parametrize("repo_id, revision, message", [
    ("owner/model", "main", "full 40-character commit"),
    ("owner/model", None, "full 40-character commit"),
    ("owner/model", "A" * 40, "full 40-character commit"),
    ("./local", COMMIT, "repository ID"),
    ("C:\\models\\x", COMMIT, "repository ID"),
    ("", COMMIT, "repository ID"),
])
def test_models_must_be_hub_ids_pinned_to_a_commit(repo_id, revision, message):
    with pytest.raises(ValueError, match=message):
        hub.require_commit(repo_id, revision)


def test_single_safetensors_checkpoint_does_not_require_a_shard_index(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").touch()
    # A stale .bin index must not make a valid safetensors checkpoint fail.
    (tmp_path / "pytorch_model.bin.index.json").write_text(json.dumps({
        "weight_map": {"x": "pytorch_model-00001-of-00001.bin"}}))
    hub.validate_weights(tmp_path)


def test_partial_shards_and_missing_weights_are_rejected(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="No standard Transformers weights"):
        hub.validate_weights(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"x": "model-00001-of-00001.safetensors"}}))
    with pytest.raises(ValueError, match="missing weight shards"):
        hub.validate_weights(tmp_path)


def test_context_limit_ignores_the_tokenizer_infinity_sentinel():
    value, candidates = hub.context_limit(SimpleNamespace(max_position_embeddings=4096),
                                          SimpleNamespace(model_max_length=10**30))
    assert value == 4096 and candidates == {"model.max_position_embeddings": 4096}
    with pytest.raises(ValueError, match="Cannot establish"):
        hub.context_limit(SimpleNamespace(), SimpleNamespace(model_max_length=10**30))


def test_snapshot_identity_requires_the_pinned_commit_directory(tmp_path):
    snapshot = tmp_path / "models--o--m" / "snapshots" / COMMIT
    assert hub.snapshot_identity("o/m", COMMIT, snapshot)["resolved_revision"] == COMMIT
    with pytest.raises(ValueError, match="is not commit"):
        hub.snapshot_identity("o/m", "b" * 40, snapshot)
    with pytest.raises(ValueError, match="is not commit"):
        hub.snapshot_identity("o/m", COMMIT, tmp_path)


def test_resolution_is_offline_and_keeps_the_original_error(tmp_path):
    message = "not cached\n" + "full original detail " * 50
    with patch.object(hub, "hf_hub_download", side_effect=OSError(message)) as cached, \
            patch.object(hub, "snapshot_download") as download:
        with pytest.raises(RuntimeError) as raised:
            hub.resolve_snapshot("o/m", COMMIT)
    assert message in str(raised.value) and "pbench download" in str(raised.value)
    assert cached.call_args.kwargs["local_files_only"] is True
    download.assert_not_called()


def test_download_prefers_safetensors_and_refuses_a_moved_commit():
    info = SimpleNamespace(sha=COMMIT, siblings=[SimpleNamespace(rfilename=name) for name in (
        "config.json", "model.safetensors", "pytorch_model.bin")])
    with patch.object(hub, "HfApi") as api, patch.object(hub, "snapshot_download", return_value="snap") as download:
        api.return_value.model_info.return_value = info
        hub.download_snapshot("o/m", COMMIT)
    patterns = download.call_args.kwargs["allow_patterns"]
    assert "*.safetensors" in patterns and "pytorch_model*.bin" not in patterns
    assert download.call_args.kwargs["ignore_patterns"] == ["consolidated*.safetensors"]
    assert download.call_args.kwargs["revision"] == COMMIT
    with patch.object(hub, "HfApi") as api, patch.object(hub, "snapshot_download") as download:
        api.return_value.model_info.return_value = SimpleNamespace(sha="b" * 40, siblings=[])
        with pytest.raises(ValueError, match="different commit"):
            hub.download_snapshot("o/m", COMMIT)
    download.assert_not_called()


def test_prequantized_checkpoints_are_rejected(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"bits": 4}}))
    with pytest.raises(ValueError, match="already has quantized"):
        hub.reject_prequantized(tmp_path)
