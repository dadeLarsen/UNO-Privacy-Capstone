"""Hashing and JSON file helpers shared by every task."""

import hashlib
import json
from pathlib import Path


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_text(text: str) -> str:
    return digest_bytes(text.encode("utf-8"))


def digest_json(value) -> str:
    """Hash JSON content independently of key order and whitespace."""
    return digest_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False).encode("utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_json(path):
    """Read UTF-8 JSON, rejecting duplicate keys instead of keeping the last one."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique_keys)


def read_jsonl(path) -> list:
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line, object_pairs_hook=unique_keys))
        except ValueError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return records
