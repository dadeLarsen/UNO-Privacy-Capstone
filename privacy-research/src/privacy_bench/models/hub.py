"""Pinned Hugging Face checkpoints: explicit download, then offline resolution only."""

import json
from pathlib import Path
import re

from huggingface_hub import HfApi, constants, hf_hub_download, snapshot_download

# The library reads HF_HUB_DISABLE_TELEMETRY once at import, so set the constant it checks per request.
constants.HF_HUB_DISABLE_TELEMETRY = True

COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
WEIGHT_FILES = (("model.safetensors.index.json", "model.safetensors"),
                ("pytorch_model.bin.index.json", "pytorch_model.bin"))
STANDARD_PATTERNS = ("*.json", "*.model", "*.txt", "*.jinja", "*.tiktoken", "tokenizer*", "vocab*")


def require_commit(repo_id, revision) -> str:
    if not isinstance(repo_id, str) or not repo_id.strip() or "\\" in repo_id or repo_id.startswith((".", "/", "~")):
        raise ValueError(f"Model id must be a Hugging Face repository ID: {repo_id!r}")
    if not isinstance(revision, str) or not COMMIT_PATTERN.fullmatch(revision):
        raise ValueError(f"Pin {repo_id} to a full 40-character commit hash; got revision {revision!r}")
    return revision


def validate_weights(snapshot: Path):
    """Accept standard single-file and sharded checkpoints, without loading weights."""
    if not (snapshot / "config.json").is_file():
        raise ValueError(f"Checkpoint has no config.json: {snapshot}")
    for index_name, single_name in WEIGHT_FILES:
        index_path = snapshot / index_name
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shards = set(index.get("weight_map", {}).values())
            if not shards:
                raise ValueError(f"Checkpoint has an empty weight index: {index_path}")
            missing = [name for name in sorted(shards) if not (snapshot / name).is_file()]
            if missing:
                raise ValueError(f"Checkpoint is missing weight shards: {', '.join(missing)}. "
                                 "Run `pbench download` to finish caching it.")
            return
        if (snapshot / single_name).is_file():
            return
    raise ValueError(f"No standard Transformers weights found in {snapshot}")


def resolve_snapshot(repo_id: str, revision: str) -> Path:
    """Find a cached snapshot without touching the network."""
    require_commit(repo_id, revision)
    try:
        # One cached file locates the snapshot without requiring every repository file.
        config = hf_hub_download(repo_id, "config.json", revision=revision, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f"{repo_id} at {revision} is not in the local Hugging Face cache. "
                           f"Run `pbench download` while online first. Original error: {exc}") from exc
    return Path(config).parent.resolve()


def download_snapshot(repo_id: str, revision: str, patterns=None) -> Path:
    """Download one pinned commit; by default standard config, tokenizer, and weight files."""
    require_commit(repo_id, revision)
    info = HfApi().model_info(repo_id, revision=revision)
    if info.sha != revision:
        raise ValueError(f"Hugging Face resolved {repo_id} {revision} to a different commit: {info.sha}")
    if patterns is None:
        names = {item.rfilename for item in info.siblings}
        safetensors = "model.safetensors" in names or "model.safetensors.index.json" in names
        # Prefer safetensors so repositories that also publish .bin files are not cached twice.
        patterns = [*STANDARD_PATTERNS, "*.safetensors" if safetensors else "pytorch_model*.bin"]
    return Path(snapshot_download(repo_id, revision=revision, allow_patterns=list(patterns))).resolve()


def snapshot_identity(repo_id: str, revision: str, snapshot: Path) -> dict:
    resolved = snapshot.name if snapshot.parent.name == "snapshots" and COMMIT_PATTERN.fullmatch(snapshot.name) else None
    if resolved != revision:
        raise ValueError(f"Snapshot {snapshot} is not commit {revision} of {repo_id}")
    return {"model": repo_id, "revision": revision, "resolved_revision": resolved, "local_snapshot": str(snapshot)}


def context_limit(model_config, tokenizer):
    """The smallest positive context length declared by the config or tokenizer."""
    config = model_config.get_text_config() if hasattr(model_config, "get_text_config") else model_config
    candidates = {}
    for key in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(config, key, None)
        if isinstance(value, int) and 0 < value < 10**12:
            candidates[f"model.{key}"] = value
    value = getattr(tokenizer, "model_max_length", None)
    if isinstance(value, int) and 0 < value < 10**12:
        candidates["tokenizer.model_max_length"] = value
    if not candidates:
        raise ValueError("Cannot establish this model's context limit from its config or tokenizer")
    return min(candidates.values()), candidates


def reject_prequantized(snapshot: Path):
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if any(config.get(key) for key in ("quantization_config", "compression_config", "load_in_4bit", "load_in_8bit")):
        raise ValueError("This checkpoint already has quantized or compressed weights. Use the original "
                         "unquantized checkpoint; precision is applied at load time.")
    return config
