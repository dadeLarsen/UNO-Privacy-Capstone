"""Run conditions: one TOML file names the task, strategy, items, model, and options.

Example::

    task = "classification"
    strategy = "prompted"
    warmup = 1

    [items]
    dataset = "opp115-sentences"
    sample = "pilot-200-10cat"

    [model]
    id = "Qwen/Qwen3-14B"
    revision = "40c069824f4251a91eefaf281ebe4c544efd3e18"
    precision = "int8"

    [options]
    prompt = "definitions"

The core only reads ``task``, ``strategy``, and ``warmup``. The task validates ``[items]``,
and the strategy validates ``[model]`` and ``[options]``; both reject unknown keys.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import tomllib

from .files import digest_json

TOP_LEVEL_KEYS = {"task", "strategy", "warmup", "items", "model", "options"}
NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9.-]*")


@dataclass(frozen=True)
class Condition:
    name: str
    task: str
    strategy: str
    warmup: int
    items: dict
    model: dict
    options: dict
    source_text: str

    def as_dict(self) -> dict:
        return {"name": self.name, "task": self.task, "strategy": self.strategy, "warmup": self.warmup,
                "items": self.items, "model": self.model, "options": self.options}

    @property
    def sha256(self) -> str:
        return digest_json(self.as_dict())


def parse_condition(name: str, text: str) -> Condition:
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Condition name {name!r} must use lowercase letters, digits, '.' and '-'")
    try:
        values = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Condition {name!r} is not valid TOML: {exc}") from exc
    unknown = set(values) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"Condition {name!r} has unknown keys: {', '.join(sorted(unknown))}")
    for key in ("task", "strategy"):
        if not isinstance(values.get(key), str) or not NAME_PATTERN.fullmatch(values[key]):
            raise ValueError(f"Condition {name!r} needs a lowercase {key} name")
    warmup = values.get("warmup", 1)
    if type(warmup) is not int or warmup < 0:
        raise ValueError(f"Condition {name!r}: warmup must be a nonnegative integer")
    tables = {}
    for key in ("items", "model", "options"):
        table = values.get(key, {})
        if not isinstance(table, dict):
            raise ValueError(f"Condition {name!r}: [{key}] must be a table")
        tables[key] = table
    return Condition(name=name, task=values["task"], strategy=values["strategy"], warmup=warmup,
                     source_text=text, **tables)


def load_condition(path) -> Condition:
    path = Path(path)
    if path.suffix != ".toml":
        raise ValueError(f"Condition files use the .toml extension: {path}")
    return parse_condition(path.stem, path.read_text(encoding="utf-8"))


def take_settings(table: dict, context: str, required=(), optional=None) -> dict:
    """Return a copy of a condition table after checking its keys; optional maps keys to defaults."""
    optional = optional or {}
    unknown = set(table) - set(required) - set(optional)
    if unknown:
        raise ValueError(f"{context} has unknown keys: {', '.join(sorted(unknown))}")
    missing = [key for key in required if key not in table]
    if missing:
        raise ValueError(f"{context} is missing: {', '.join(missing)}")
    return {**optional, **table}
