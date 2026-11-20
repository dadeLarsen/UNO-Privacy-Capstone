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
            hub.resolve_snapshot("o/m", COMMIT, "model-Q4_K_M.gguf")
    assert message in str(raised.value) and "pbench download" in str(raised.value)
    assert cached.call_args.kwargs["local_files_only"] is True
    download.assert_not_called()


def test_a_snapshot_is_located_by_the_file_named(tmp_path):
    found = tmp_path / "snapshots" / COMMIT / "model-Q4_K_M.gguf"
    with patch.object(hub, "hf_hub_download", return_value=str(found)) as cached:
        assert hub.resolve_snapshot("o/m", COMMIT, "model-Q4_K_M.gguf") == found.parent.resolve()
    assert cached.call_args.args == ("o/m", "model-Q4_K_M.gguf") and cached.call_args.kwargs["local_files_only"] is True


def test_download_fetches_only_the_file_named_and_refuses_a_moved_commit():
    info = SimpleNamespace(sha=COMMIT, siblings=[SimpleNamespace(rfilename=name) for name in (
        "model-Q4_K_M.gguf", "model-Q8_0.gguf")])
    with patch.object(hub, "HfApi") as api, patch.object(hub, "snapshot_download", return_value="snap") as download:
        api.return_value.model_info.return_value = info
        hub.download_snapshot("o/m", COMMIT, only="model-Q4_K_M.gguf")
        assert download.call_args.kwargs == {"revision": COMMIT, "allow_patterns": ["model-Q4_K_M.gguf"]}
        with pytest.raises(ValueError, match="has no file 'model-Q2_K.gguf'"):
            hub.download_snapshot("o/m", COMMIT, only="model-Q2_K.gguf")
    assert download.call_count == 1
    with patch.object(hub, "HfApi") as api, patch.object(hub, "snapshot_download") as download:
        api.return_value.model_info.return_value = SimpleNamespace(sha="b" * 40, siblings=[])
        with pytest.raises(ValueError, match="different commit"):
            hub.download_snapshot("o/m", COMMIT, only="model-Q4_K_M.gguf")
    download.assert_not_called()
