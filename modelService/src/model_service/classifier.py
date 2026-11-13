"""Adapter between the HTTP service and the benchmark's classification strategies.

The service deliberately reuses ``privacy_bench`` instead of loading a model itself:
the condition pins the checkpoint, the fitted head, the instruction prefix, and the
label order, and the strategy checks the fitted weight hashes before inference.
"""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Protocol

from .taxonomy import LABELS, TAXONOMY


@dataclass(frozen=True)
class Classification:
    label_id: int
    label: str
    scores: tuple[float, ...] | None = None


class Classifier(Protocol):
    def load(self) -> dict:
        """Load the model once and return its safe provenance metadata."""

    def metadata(self) -> dict | None:
        """Return the loaded model's metadata, or None before it is loaded."""

    def classify(self, sentence: str) -> Classification:
        """Classify one sentence."""


class PrivacyBenchClassifier:
    """Run one pinned classification condition on arbitrary sentences."""

    def __init__(self, condition_path: Path, *, cache_size: int = 4096):
        from privacy_bench.core.conditions import load_condition
        from privacy_bench.tasks import get_task

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
        self._metadata: dict | None = None
        self._score_order: list[int] | None = None

    @property
    def condition(self):
        return self._condition

    def load(self) -> dict:
        with self._lock:
            if self._metadata is not None:
                return dict(self._metadata)
            from privacy_bench.tasks.classification.taxonomy import LABELS as MODEL_LABELS

            if tuple(MODEL_LABELS) != LABELS:
                raise ValueError("modelService/taxonomy.json does not match privacy_bench's category order")
            loaded = self._strategy.load()
            # A classifier-head checkpoint has its own label order, and its logits follow that order.
            mapping = loaded.get("label_mapping")
            if mapping is not None:
                by_label = {item["label_id"]: item["model_label_id"] for item in mapping}
                self._score_order = [by_label[label_id] for label_id in range(len(LABELS))]
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
            return dict(self._metadata)

    def metadata(self) -> dict | None:
        with self._lock:
            return dict(self._metadata) if self._metadata is not None else None

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
            from privacy_bench.tasks.classification.scoring import parse_response

            # Every strategy answers in raw_output, and the benchmark scores that answer with this parser.
            # model_label_id is the checkpoint's own index, which is not always the benchmark's label_id.
            label_id = parse_response(response.get("raw_output"))
            scores = response.get("scores")
            if scores is None:
                for key in ("similarities", "logits"):
                    if key in response:
                        scores = response[key]
                        break
            if scores is not None and self._score_order is not None:
                scores = [scores[index] for index in self._score_order]
            result = Classification(
                label_id=label_id,
                label=LABELS[label_id],
                scores=tuple(float(score) for score in scores) if scores is not None else None,
            )
            if self._cache_size:
                self._cache[sentence] = result
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            return result
