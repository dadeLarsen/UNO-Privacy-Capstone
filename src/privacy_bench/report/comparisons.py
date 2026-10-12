"""Comparison files: which saved runs to put side by side, and how to score and label them.

Example::

    task = "classification"
    title = "Qwen3, skythrone/privacy-model, and JEV-9B"
    runs = [
        "classification/qwen3-14b-int8-definitions/20260918-220631Z",
        "classification/privacy-model-fp32/20261002-213049Z",
    ]

    [scoring]
    gold = "pilot-200-10cat-reviewed"

    [labels]
    "classification/privacy-model-fp32/20261002-213049Z" = "privacy-model"

Run paths are relative to results/runs/. ``[scoring]`` belongs to the task; ``[labels]``
optionally overrides legend entries.
"""

from dataclasses import dataclass
from pathlib import Path
import tomllib

from ..core.conditions import NAME_PATTERN

TOP_LEVEL_KEYS = {"task", "title", "runs", "scoring", "labels"}


@dataclass(frozen=True)
class Comparison:
    name: str
    task: str
    title: str | None
    runs: tuple
    scoring: dict
    labels: dict
    source_text: str


def parse_comparison(name: str, text: str) -> Comparison:
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Comparison name {name!r} must use lowercase letters, digits, '.' and '-'")
    try:
        values = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Comparison {name!r} is not valid TOML: {exc}") from exc
    unknown = set(values) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"Comparison {name!r} has unknown keys: {', '.join(sorted(unknown))}")
    if not isinstance(values.get("task"), str) or not NAME_PATTERN.fullmatch(values["task"]):
        raise ValueError(f"Comparison {name!r} needs a lowercase task name")
    title = values.get("title")
    if title is not None and (not isinstance(title, str) or not title.strip()):
        raise ValueError(f"Comparison {name!r}: title must be a nonempty string")
    runs = values.get("runs")
    if (not isinstance(runs, list) or not runs or any(not isinstance(run, str) or not run.strip() for run in runs)
            or len(set(runs)) != len(runs)):
        raise ValueError(f"Comparison {name!r} needs a nonempty list of distinct run paths")
    for run in runs:
        parts = run.split("/")
        if "\\" in run or any(part in ("", ".", "..") for part in parts) or Path(run).is_absolute():
            raise ValueError(f"Comparison {name!r}: run paths are relative to results/runs/ with '/' separators: {run!r}")
    scoring = values.get("scoring", {})
    labels = values.get("labels", {})
    if not isinstance(scoring, dict) or not isinstance(labels, dict):
        raise ValueError(f"Comparison {name!r}: [scoring] and [labels] must be tables")
    for run, label in labels.items():
        if run not in runs or not isinstance(label, str) or not label.strip():
            raise ValueError(f"Comparison {name!r}: label for {run!r} must name a listed run and be nonempty")
    return Comparison(name=name, task=values["task"], title=title, runs=tuple(runs), scoring=scoring,
                      labels=labels, source_text=text)


def load_comparison(path) -> Comparison:
    path = Path(path)
    if path.suffix != ".toml":
        raise ValueError(f"Comparison files use the .toml extension: {path}")
    return parse_comparison(path.stem, path.read_text(encoding="utf-8"))
