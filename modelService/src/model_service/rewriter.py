"""Adapter between the HTTP service and the benchmark's GGUF rewrite strategy.

The classifier and the rewriter are separate models, so this adapter has its own
condition and loads lazily on the first rewrite request. The prompt is the pinned
``summary-v3`` prompt from the benchmark; the service does not invent style-specific
prompts that were never evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from threading import RLock
from types import SimpleNamespace
from typing import Protocol

from privacy_bench.core.conditions import load_condition
from privacy_bench.tasks import get_task

from .taxonomy import TAXONOMY

NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")
WORD = re.compile(r"[a-z]+")
# Words whose loss could change a promise. This is a flag for review, not a verdict.
SENSITIVE = (
    "not", "no", "never", "only", "unless", "except", "may", "might", "could",
    "likely", "reasonable", "knowingly", "some", "all",
)
REASONING_TAG = re.compile(r"</?think|</?reasoning", re.IGNORECASE)


@dataclass(frozen=True)
class RewrittenPassage:
    text: str
    warnings: tuple[str, ...]


class Rewriter(Protocol):
    def load(self) -> dict:
        """Load the model unless it is loaded and return its safe provenance metadata."""

    def metadata(self) -> dict | None:
        """Return the model's metadata, or None before its first load."""

    def rewrite(self, text: str, style: str) -> RewrittenPassage:
        """Rewrite one passage and flag details a human should check."""


class PrivacyBenchRewriter:
    """Run one pinned rewrite condition on arbitrary passages."""

    def __init__(self, condition_path: Path):
        condition = load_condition(condition_path)
        if condition.task != "rewrite":
            raise ValueError(
                f"modelService needs a rewrite condition, not {condition.task!r}: {condition_path}"
            )
        self._condition = condition
        self._strategy = get_task(condition.task).build_strategy(condition)
        self._lock = RLock()
        self._loaded = False
        self._metadata: dict | None = None

    def load(self) -> dict:
        with self._lock:
            if self._loaded:
                return dict(self._metadata)
            loaded = self._strategy.load()
            prompt = getattr(self._strategy, "prompt", None)
            self._metadata = {
                "id": loaded.get("model"),
                "revision": loaded.get("revision"),
                "resolved_revision": loaded.get("resolved_revision"),
                "precision": loaded.get("precision"),
                "device": loaded.get("device"),
                "strategy": self._strategy.name,
                "prompt": prompt.get("name") if isinstance(prompt, dict) else None,
                "taxonomy_id": TAXONOMY["taxonomy_id"],
                "taxonomy_version": TAXONOMY["version"],
            }
            self._loaded = True
            return dict(self._metadata)

    def metadata(self) -> dict | None:
        with self._lock:
            return dict(self._metadata) if self._metadata is not None else None

    def rewrite(self, text: str, style: str) -> RewrittenPassage:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")
        with self._lock:
            self.load()
            prepared = self._strategy.prepare(SimpleNamespace(text=text))
            response = self._strategy.predict(prepared)
        rewritten = str(response.get("raw_output") or "").strip()
        if not rewritten:
            raise RuntimeError("The rewriter returned an empty response.")
        return RewrittenPassage(
            text=rewritten,
            warnings=_warnings(text, rewritten, response, style),
        )


def _warnings(source: str, rewritten: str, response: dict, style: str) -> tuple[str, ...]:
    warnings = [
        f"The evaluated summary-v3 prompt is used for all styles, including {style}; "
        "style-specific prompts have not been evaluated."
    ]
    source_numbers = set(NUMBER.findall(source))
    missing_numbers = sorted(source_numbers - set(NUMBER.findall(rewritten)))
    if missing_numbers:
        warnings.append("Check numbers from the source that are missing here: " + ", ".join(missing_numbers))

    source_words = set(WORD.findall(source.lower()))
    rewritten_words = set(WORD.findall(rewritten.lower()))
    missing_terms = [word for word in SENSITIVE if word in source_words and word not in rewritten_words]
    if missing_terms:
        warnings.append("Check conditions or hedges from the source that are missing here: " + ", ".join(missing_terms))

    if REASONING_TAG.search(rewritten):
        warnings.append("The response contains reasoning tags; review it before using it.")
    if response.get("reached_output_token_limit"):
        warnings.append("The response hit the token limit and may be incomplete.")
    return tuple(warnings)
