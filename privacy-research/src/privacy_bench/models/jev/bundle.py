"""Validate and explicitly cache a complete JEV checkpoint bundle."""

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from ...core.files import read_json
from ..hub import require_commit

INDEX_NAME = "model.safetensors.index.json"
REQUIRED_FILES = (
    "config.json", INDEX_NAME, "tokenizer.json", "tokenizer_config.json",
    "judge_config.json", "calibration.json", "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors", "head.safetensors",
)
SHARD_PATTERN = re.compile(r"model-(\d{5})-of-(\d{5})\.safetensors")


def read_object(path: Path) -> dict:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"JEV metadata must be a JSON object: {path}")
    return value


def _relative_shard(name):
    if not isinstance(name, str) or not name:
        raise ValueError(f"JEV weight index has an invalid shard filename: {name!r}")
    posix, windows = PurePosixPath(name), PureWindowsPath(name)
    if (posix.is_absolute() or windows.drive or windows.root or "\\" in name or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError(f"JEV weight index shard must stay inside the checkpoint: {name!r}")
    match = SHARD_PATTERN.fullmatch(posix.name)
    if match is None:
        raise ValueError(f"JEV base weights require numbered model safetensors shards: {name!r}")
    return posix, int(match[1]), int(match[2])


def index_shards(index_path: Path) -> list:
    index = read_object(index_path)
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"JEV checkpoint requires a nonempty weight_map: {index_path}")
    for key, value in weights.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"JEV weight index has an invalid tensor name: {key!r}")
        _relative_shard(value)
    shards = sorted(set(weights.values()))
    numbered = [_relative_shard(name) for name in shards]
    totals = {total for _, _, total in numbered}
    if (len(totals) != 1 or len(shards) != next(iter(totals))
            or {number for _, number, _ in numbered} != set(range(1, len(shards) + 1))):
        raise ValueError(f"JEV weight index must name every numbered base-model shard exactly once: {index_path}")
    return shards


def validate_bundle(snapshot) -> Path:
    """Check every required file and shard without loading any tensors."""
    snapshot = Path(snapshot).expanduser().resolve()
    if not snapshot.is_dir():
        raise ValueError(f"JEV checkpoint directory does not exist: {snapshot}")
    missing = [name for name in REQUIRED_FILES if not (snapshot / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete JEV checkpoint {snapshot}; missing files: {', '.join(missing)}")
    shards = index_shards(snapshot / INDEX_NAME)
    for name in (*REQUIRED_FILES, *shards):
        # Check the lexical path before following cache symlinks, whose targets
        # legitimately live in the cache's blob store.
        child = Path(os.path.abspath(snapshot / name))
        if not child.is_relative_to(snapshot):
            raise ValueError(f"JEV checkpoint file escapes its snapshot: {name!r}")
        if not child.is_file():
            raise ValueError(f"JEV checkpoint is missing weight shard: {child}")
        if child.stat().st_size == 0:
            raise ValueError(f"JEV checkpoint has an empty file: {child}")
        if name.endswith(".json") and name != INDEX_NAME:
            read_object(child)
    return snapshot


def resolve_bundle(repo_id: str, revision: str) -> Path:
    """Find a complete cached bundle; never downloads."""
    require_commit(repo_id, revision)
    try:
        config = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f"JEV checkpoint {repo_id} at {revision} is not cached. Run `pbench download` while "
                           f"online first. Original error: {exc}") from exc
    return validate_bundle(Path(config).parent)


def download_bundle(repo_id: str, revision: str) -> Path:
    """Download only the pinned base, primary adapter, head, tokenizer, and metadata."""
    require_commit(repo_id, revision)
    info = HfApi().model_info(repo_id, revision=revision)
    if info.sha != revision:
        raise ValueError(f"Hugging Face resolved {repo_id} {revision} to a different commit: {info.sha}")
    if info.siblings is None:
        raise ValueError("Hugging Face did not return the JEV checkpoint file list")
    names = {item.rfilename for item in info.siblings}
    missing = sorted(set(REQUIRED_FILES) - names)
    if missing:
        raise ValueError(f"JEV repository is missing required files: {', '.join(missing)}")
    cached_index = hf_hub_download(repo_id=repo_id, filename=INDEX_NAME, revision=revision)
    shards = index_shards(Path(cached_index))
    missing = sorted(set(shards) - names)
    if missing:
        raise ValueError(f"JEV repository is missing indexed weight shards: {', '.join(missing)}")
    snapshot = snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=[*REQUIRED_FILES, *shards])
    return validate_bundle(snapshot)
