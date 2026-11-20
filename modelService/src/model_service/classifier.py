"""Adapter between the HTTP service and the benchmark's classification strategies.

The service deliberately reuses ``privacy_bench`` instead of loading a model itself:
the condition pins the GGUF file, the fitted head, the instruction prefix, and the
label order, and the strategy checks the fitted weight hashes before inference.
"""

from collections import OrderedDict
from dataclasses import dataclass
import gc
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Protocol

from privacy_bench.core.conditions import load_condition
from privacy_bench.tasks import get_task
from privacy_bench.tasks.classification.scoring import parse_response

from .taxonomy import LABELS, TAXONOMY


@dataclass(frozen=True)
class Classification:
    label_id: int
    label: str
    scores: tuple[float, ...] | None = None


class Classifier(Protocol):
    def load(self) -> dict:
        """Load the model unless it is loaded and return its safe provenance metadata."""

    def metadata(self) -> dict | None:
        """Return the model's metadata, or None before its first load. A release keeps it."""

    def classify(self, sentence: str) -> Classification:
        """Classify one sentence."""

    def release(self) -> None:
        """Free the model's memory. The next load brings the model back."""


class PrivacyBenchClassifier:
    """Run one pinned classification condition on arbitrary sentences."""

    def __init__(self, condition_path: Path, *, cache_size: int = 4096):
        condition = load_condition(condition_path)
        if condition.task != "classification":
            raise ValueError(
                f"modelService needs a classification condition, not {condition.task!r}: {condition_path}"
            )
        self._condition = condition
        self._strategy = get_task(condition.task).build_strategy(condition)
        self._cache_size = cache_size
        self._cache: OrderedDict[str, Classification] = OrderedDict()
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
                "head": loaded.get("head"),
                "fit": loaded.get("fit"),
                "taxonomy_id": TAXONOMY["taxonomy_id"],
                "taxonomy_version": TAXONOMY["version"],
            }
            self._loaded = True
            return dict(self._metadata)

    def metadata(self) -> dict | None:
        with self._lock:
            return dict(self._metadata) if self._metadata is not None else None

    def release(self) -> None:
        with self._lock:
            # Building a strategy loads no model, so a fresh one replaces the one that holds the model.
            self._strategy = get_task(self._condition.task).build_strategy(self._condition)
            self._loaded = False
            # The collection makes sure the dropped model is freed now, even if something holds it in a
            # reference cycle. llama.cpp then returns its memory, whichever thread releases.
            gc.collect()

    def classify(self, sentence: str) -> Classification:
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError("sentence must be a nonempty string")
        with self._lock:
            self.load()
            cached = self._cache.get(sentence)
            if cached is not None:
                self._cache.move_to_end(sentence)
                return cached
            prepared = self._strategy.prepare(SimpleNamespace(text=sentence))
            response = self._strategy.predict(prepared)
            # The benchmark scores the answer in raw_output with this parser, so the service reads it the same way.
            label_id = parse_response(response.get("raw_output"))
            # The fitted head scores the ten categories in the benchmark's order.
            result = Classification(label_id=label_id, label=LABELS[label_id], scores=tuple(response["scores"]))
            if self._cache_size:
                self._cache[sentence] = result
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            return result
