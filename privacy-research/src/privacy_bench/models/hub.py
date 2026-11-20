"""Pinned Hugging Face model files: explicit download, then offline resolution only."""

from pathlib import Path
import re

from huggingface_hub import HfApi, constants, hf_hub_download, snapshot_download

# The library reads HF_HUB_DISABLE_TELEMETRY once at import, so set the constant it checks per request.
constants.HF_HUB_DISABLE_TELEMETRY = True

COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def require_commit(repo_id, revision) -> str:
    if not isinstance(repo_id, str) or not repo_id.strip() or "\\" in repo_id or repo_id.startswith((".", "/", "~")):
        raise ValueError(f"Model id must be a Hugging Face repository ID: {repo_id!r}")
    if not isinstance(revision, str) or not COMMIT_PATTERN.fullmatch(revision):
        raise ValueError(f"Pin {repo_id} to a full 40-character commit hash; got revision {revision!r}")
    return revision


def resolve_snapshot(repo_id: str, revision: str, filename: str) -> Path:
    """Find a cached snapshot by one file at its top level, without touching the network."""
    require_commit(repo_id, revision)
    try:
        # One cached file locates the snapshot without requiring every repository file.
        found = hf_hub_download(repo_id, filename, revision=revision, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f"{repo_id} at {revision} is not in the local Hugging Face cache. "
                           f"Run `pbench download` while online first. Original error: {exc}") from exc
    return Path(found).parent.resolve()


def download_snapshot(repo_id: str, revision: str, only: str) -> Path:
    """Download the one file named from a pinned commit. A GGUF repository holds one file per precision."""
    require_commit(repo_id, revision)
    info = HfApi().model_info(repo_id, revision=revision)
    if info.sha != revision:
        raise ValueError(f"Hugging Face resolved {repo_id} {revision} to a different commit: {info.sha}")
    if only not in {item.rfilename for item in info.siblings}:
        raise ValueError(f"{repo_id} at {revision} has no file {only!r}")
    return Path(snapshot_download(repo_id, revision=revision, allow_patterns=[only])).resolve()


def snapshot_identity(repo_id: str, revision: str, snapshot: Path) -> dict:
    resolved = snapshot.name if snapshot.parent.name == "snapshots" and COMMIT_PATTERN.fullmatch(snapshot.name) else None
    if resolved != revision:
        raise ValueError(f"Snapshot {snapshot} is not commit {revision} of {repo_id}")
    return {"model": repo_id, "revision": revision, "resolved_revision": resolved, "local_snapshot": str(snapshot)}
