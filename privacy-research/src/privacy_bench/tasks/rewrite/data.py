"""Frozen policy passages, and the editable prompt files that say how to rewrite them."""

from dataclasses import dataclass
import tomllib

from ...core.files import digest_bytes, read_json
from ...core.paths import data_file, prompt_file

PLACEHOLDER = "{passage}"
PASSAGE_KEYS = {"id", "policy_first_id", "heading", "sentence_ids", "words", "text"}


@dataclass(frozen=True)
class Passage:
    id: str
    heading: str
    text: str


def load_passages(name: str) -> tuple:
    """Read data/rewrite/passages/<name>.json; returns (passages, identity)."""
    path = data_file("rewrite", "passages", name, suffix=".json")
    values = read_json(path)
    if not isinstance(values, dict) or values.get("name") != name or not isinstance(values.get("passages"), list):
        raise ValueError(f"{path} needs its own name and a list of passages")
    passages = []
    for entry in values["passages"]:
        if not isinstance(entry, dict) or set(entry) != PASSAGE_KEYS:
            raise ValueError(f"{path}: every passage needs exactly {', '.join(sorted(PASSAGE_KEYS))}")
        if not all(isinstance(entry[key], str) and entry[key].strip() for key in ("id", "heading", "text")):
            raise ValueError(f"{path}: passage id, heading, and text must be nonempty strings")
        passages.append(Passage(id=entry["id"], heading=entry["heading"], text=entry["text"]))
    if not passages or len({passage.id for passage in passages}) != len(passages):
        raise ValueError(f"{path} needs at least one passage and unique passage IDs")
    identity = {"passages": {"name": name, "sha256": digest_bytes(path.read_bytes()), "n": len(passages)}}
    return passages, identity


def parse_prompt(name: str, text: str) -> dict:
    """A prompt file has a required user template holding {passage} once, and an optional system message."""
    try:
        values = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Prompt {name!r} is not valid TOML: {exc}") from exc
    unknown = set(values) - {"system", "user"}
    if unknown:
        raise ValueError(f"Prompt {name!r} has unknown keys: {', '.join(sorted(unknown))}")
    user = values.get("user")
    if not isinstance(user, str) or user.count(PLACEHOLDER) != 1:
        raise ValueError(f"Prompt {name!r} needs a user string that contains {PLACEHOLDER} exactly once")
    system = values.get("system")
    if system is not None and (not isinstance(system, str) or not system.strip()):
        raise ValueError(f"Prompt {name!r}: system must be a nonempty string when present")
    return {"name": name, "system": system.strip() if system else None, "user": user.strip()}


def load_prompt(name: str) -> dict:
    """Read prompts/rewrite/<name>.toml."""
    return parse_prompt(name, prompt_file("rewrite", name, suffix=".toml").read_text(encoding="utf-8"))


def render(spec: dict, passage: Passage) -> dict:
    # replace, not str.format, so braces elsewhere in a prompt need no escaping
    return {"system": spec["system"], "user": spec["user"].replace(PLACEHOLDER, passage.text)}
