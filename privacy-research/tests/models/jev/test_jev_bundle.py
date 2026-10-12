"""Checkpoint completeness, offline resolution, and filtered download contracts."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from privacy_bench.models.jev import bundle

REPO = "autotrust/JEV-9B"
COMMIT = "4ab5dfb9331c4eb3a212742e1a1aa5446c1fda35"
SHARDS = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "models--autotrust--JEV-9B" / "snapshots" / COMMIT
    for name in (*bundle.REQUIRED_FILES, *SHARDS):
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"{}" if name.endswith(".json") else b"test tensor data")
    write_index(path, {"model.layers.0.weight": SHARDS[0], "model.layers.1.weight": SHARDS[1]})
    return path


def write_index(path, weight_map):
    (path / bundle.INDEX_NAME).write_text(json.dumps({"weight_map": weight_map}, ensure_ascii=False), encoding="utf-8")


def model_info(names=None, sha=COMMIT):
    if names is None:
        names = [*bundle.REQUIRED_FILES, *SHARDS, "adapter_vllm/adapter_model.safetensors", "README.md"]
    return SimpleNamespace(sha=sha, siblings=[SimpleNamespace(rfilename=name) for name in names])


def test_complete_bundle_validates_without_the_hub(snapshot):
    with patch.object(bundle, "hf_hub_download") as cached, patch.object(bundle, "HfApi") as api:
        assert bundle.validate_bundle(snapshot) == snapshot.resolve()
    cached.assert_not_called()
    api.assert_not_called()


@pytest.mark.parametrize("name", [*bundle.REQUIRED_FILES, *SHARDS])
def test_each_required_file_and_each_indexed_shard_is_required(snapshot, name):
    path = snapshot / name
    path.rename(path.with_name(path.name + ".absent"))
    with pytest.raises(ValueError, match="missing"):
        bundle.validate_bundle(snapshot)


@pytest.mark.parametrize("name, content", [
    (SHARDS[0], b""), ("head.safetensors", b""), ("judge_config.json", b"not-json"),
    ("calibration.json", b"[]"), ("tokenizer.json", b'{"version":1,"version":2}'),
])
def test_empty_files_invalid_json_and_non_object_metadata_are_rejected(snapshot, name, content):
    (snapshot / name).write_bytes(content)
    with pytest.raises(ValueError):
        bundle.validate_bundle(snapshot)


@pytest.mark.parametrize("weight_map", [
    None, {}, [], {"": SHARDS[0]}, {"tensor": None}, {"tensor": 12}, {"tensor": "pytorch_model.bin"},
    {"tensor": "head.safetensors"}, {"tensor": "model-00001-of-00002.safetensors"},
    {"tensor": "model-00000-of-00001.safetensors"}, {"tensor": "model-00001-of-00000.safetensors"},
    {"first": SHARDS[0], "second": "model-00002-of-00003.safetensors"},
])
def test_every_weight_index_entry_must_name_a_real_numbered_safetensors_shard(snapshot, weight_map):
    write_index(snapshot, weight_map)
    with pytest.raises(ValueError):
        bundle.validate_bundle(snapshot)


def test_weight_index_rejects_duplicate_tensor_keys(snapshot):
    (snapshot / bundle.INDEX_NAME).write_text('{"weight_map":{"same":"model-00001-of-00001.safetensors",'
                                              '"same":"model-00001-of-00001.safetensors"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON"):
        bundle.validate_bundle(snapshot)


@pytest.mark.parametrize("name", [
    "../model-00001-of-00001.safetensors", "/model-00001-of-00001.safetensors",
    "weights/../../model-00001-of-00001.safetensors", "./model-00001-of-00001.safetensors",
    "C:/model-00001-of-00001.safetensors", "C:model-00001-of-00001.safetensors",
    "..\\model-00001-of-00001.safetensors", "weights\\model-00001-of-00001.safetensors",
    "weights//model-00001-of-00001.safetensors", "weights:stream/model-00001-of-00001.safetensors",
])
def test_shard_paths_must_stay_inside_the_checkpoint(snapshot, name):
    write_index(snapshot, {"tensor": name})
    with pytest.raises(ValueError, match="inside the checkpoint"):
        bundle.validate_bundle(snapshot)


def test_resolution_is_offline_and_pinned(snapshot):
    with patch.object(bundle, "hf_hub_download", return_value=str(snapshot / "config.json")) as cached, \
            patch.object(bundle, "snapshot_download") as download, patch.object(bundle, "HfApi") as api:
        assert bundle.resolve_bundle(REPO, COMMIT) == snapshot.resolve()
    cached.assert_called_once_with(repo_id=REPO, filename="config.json", revision=COMMIT, local_files_only=True)
    download.assert_not_called()
    api.assert_not_called()


def test_offline_failure_keeps_the_full_original_error():
    message = "unavailable cached config\n" + "all original error detail " * 80
    with patch.object(bundle, "hf_hub_download", side_effect=OSError(message)), \
            patch.object(bundle, "snapshot_download") as download, pytest.raises(RuntimeError) as raised:
        bundle.resolve_bundle(REPO, COMMIT)
    assert message in str(raised.value) and "pbench download" in str(raised.value)
    download.assert_not_called()


@pytest.mark.parametrize("repo_id, revision", [(REPO, None), (REPO, "main"), ("./local", COMMIT), ("", COMMIT)])
def test_unpinned_or_local_references_are_rejected_before_any_lookup(repo_id, revision):
    with patch.object(bundle, "hf_hub_download") as cached, pytest.raises(ValueError):
        bundle.resolve_bundle(repo_id, revision)
    cached.assert_not_called()


def test_download_fetches_exactly_the_required_files_and_indexed_shards_at_the_commit(snapshot):
    with patch.object(bundle, "HfApi") as api, \
            patch.object(bundle, "hf_hub_download", return_value=str(snapshot / bundle.INDEX_NAME)) as index, \
            patch.object(bundle, "snapshot_download", return_value=str(snapshot)) as download:
        api.return_value.model_info.return_value = model_info()
        assert bundle.download_bundle(REPO, COMMIT) == snapshot.resolve()
    api.return_value.model_info.assert_called_once_with(REPO, revision=COMMIT)
    index.assert_called_once_with(repo_id=REPO, filename=bundle.INDEX_NAME, revision=COMMIT)
    download.assert_called_once_with(repo_id=REPO, revision=COMMIT, allow_patterns=[*bundle.REQUIRED_FILES, *SHARDS])


@pytest.mark.parametrize("missing", ["head.safetensors", "adapter/adapter_model.safetensors", SHARDS[1]])
def test_download_rejects_missing_metadata_or_shards_before_the_large_download(snapshot, missing):
    names = [name for name in (*bundle.REQUIRED_FILES, *SHARDS) if name != missing]
    with patch.object(bundle, "HfApi") as api, \
            patch.object(bundle, "hf_hub_download", return_value=str(snapshot / bundle.INDEX_NAME)), \
            patch.object(bundle, "snapshot_download") as download, pytest.raises(ValueError) as raised:
        api.return_value.model_info.return_value = model_info(names=names)
        bundle.download_bundle(REPO, COMMIT)
    assert missing in str(raised.value)
    download.assert_not_called()


def test_download_rejects_a_moved_commit_and_validates_the_completed_cache(snapshot, tmp_path):
    with patch.object(bundle, "HfApi") as api, patch.object(bundle, "snapshot_download") as download, \
            pytest.raises(ValueError, match="different commit"):
        api.return_value.model_info.return_value = model_info(sha="a" * 40)
        bundle.download_bundle(REPO, COMMIT)
    download.assert_not_called()
    with patch.object(bundle, "HfApi") as api, \
            patch.object(bundle, "hf_hub_download", return_value=str(snapshot / bundle.INDEX_NAME)), \
            patch.object(bundle, "snapshot_download", return_value=str(tmp_path / "incomplete")), \
            pytest.raises(ValueError, match="does not exist"):
        api.return_value.model_info.return_value = model_info()
        bundle.download_bundle(REPO, COMMIT)
