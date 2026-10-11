"""The contracts between the harness, tasks, and strategies.

A task owns its items, reference answers, and evaluation. A strategy is one way to perform
a task with one model backend. The harness runs any strategy over any task's items.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from .conditions import Condition


class Item(Protocol):
    id: str


@dataclass(frozen=True)
class Selection:
    """The frozen items for one run, plus a JSON-safe record of how they were chosen."""

    items: Sequence[Item]
    identity: dict
    references: Any = field(default=None, compare=False)


class Strategy(Protocol):
    name: str

    @property
    def input_spec(self) -> dict:
        """Everything that shapes what the model receives, independent of the item; hashed."""

    @property
    def timing_scope(self) -> str:
        """What a response's ``seconds`` measures."""

    def download(self) -> dict:
        """Cache the pinned model files while online; inference never downloads."""

    def load(self) -> dict:
        """Load the model and return its provenance metadata."""

    def prepare(self, item) -> Any:
        """Build the model input for one item; raise ValueError when it cannot be used as is."""

    def describe(self, prepared) -> dict:
        """Summarise a prepared input for the input audit, for example its token count."""

    def predict(self, prepared) -> dict:
        """Run the model on one prepared input and return the response fields."""


class Task(Protocol):
    name: str

    def select_items(self, condition: Condition) -> Selection:
        """Load and validate the items and references named by the condition."""

    def build_strategy(self, condition: Condition) -> Strategy:
        """Validate the strategy settings without loading any model."""

    def evaluate(self, selection: Selection, responses: list, output: Path) -> dict:
        """Score the responses, write derived files under output, and return the metrics."""

    def summary_line(self, metrics: dict) -> str:
        """One line of headline scores for the terminal."""
