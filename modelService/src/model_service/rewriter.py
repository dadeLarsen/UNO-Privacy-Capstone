"""Adapter between the HTTP service and the benchmark's rewrite strategy.

Like the classifier, the rewriter is a pinned ``privacy_bench`` condition: it names the
GGUF file, the prompt file, the seed, and the output limit, so the service rewrites a
passage exactly as a benchmark run of that condition does.
"""

from dataclasses import dataclass
import gc
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Protocol

from privacy_bench.core.conditions import load_condition
from privacy_bench.tasks import get_task
from privacy_bench.tasks.rewrite.scoring import output_text, status


@dataclass(frozen=True)
class Rewrite:
    text: str
    # The benchmark's verdict on the output: valid, truncated (the output limit cut it), empty, or reasoning.
    status: str
    output_tokens: int


class Rewriter(Protocol):
    def load(self) -> dict:
        """Load the model unless it is loaded and return its safe provenance metadata."""

    def metadata(self) -> dict | None:
        """Return the model's metadata, or None before its first load. A release keeps it, and it never waits."""

    def rewrite(self, text: str) -> Rewrite:
        """Rewrite one passage. Raises ValueError for a passage that does not fit the model's context."""

    def release(self) -> None:
        """Free the model's memory. The next load brings the model back."""


class PrivacyBenchRewriter:
    """Run one pinned rewrite condition on arbitrary passages."""

    def __init__(self, condition_path: Path):
        condition = load_condition(condition_path)
        if condition.task != "rewrite":
            raise ValueError(f"The rewrite route needs a rewrite condition, not {condition.task!r}: {condition_path}")
        self._condition = condition
        self._strategy = get_task(condition.task).build_strategy(condition)
        self._lock = RLock()
        self._loaded = False
        self._metadata: dict | None = None

    @property
    def condition(self):
        return self._condition

    def load(self) -> dict:
        with self._lock:
            if self._loaded:
                return dict(self._metadata)
            loaded = self._strategy.load()
            self._metadata = {
                "id": loaded.get("model"),
                "revision": loaded.get("revision"),
                "resolved_revision": loaded.get("resolved_revision"),
                "precision": loaded.get("precision"),
                "device": loaded.get("device"),
                "strategy": self._strategy.name,
                "prompt": self._strategy.prompt["name"],
                "max_new_tokens": loaded.get("max_new_tokens"),
            }
            self._loaded = True
            return dict(self._metadata)

    def metadata(self) -> dict | None:
        # No lock, so /health never waits for a rewrite: load() sets the dict in one step and release() keeps it.
        metadata = self._metadata
        return dict(metadata) if metadata is not None else None

    def release(self) -> None:
        with self._lock:
            # Building a strategy loads no model, so a fresh one replaces the one that holds the model.
            self._strategy = get_task(self._condition.task).build_strategy(self._condition)
            self._loaded = False
            # The collection makes sure the dropped model is freed now, even if something holds it in a
            # reference cycle. llama.cpp then returns its memory, whichever thread releases.
            gc.collect()

    def rewrite(self, text: str) -> Rewrite:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")
        with self._lock:
            self.load()
            # prepare() refuses a passage whose prompt and output do not fit the context; nothing is truncated.
            response = self._strategy.predict(self._strategy.prepare(SimpleNamespace(text=text)))
            return Rewrite(text=output_text(response), status=status(response),
                           output_tokens=response["output_tokens"])
