"""The contracts between the harness, tasks, and strategies.

A task owns its items, reference answers, and evaluation. A strategy is one way to perform
a task with one model backend. The harness runs any strategy over any task's items.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Protocol, Sequence

from .conditions import Condition


class Item(Protocol):
    id: str


class Panel(NamedTuple):
    """One bar chart in a comparison figure."""

    field: str
    title: str
    axis: str
    scale: float
    style: str  # percent, score, seconds, memory, or tokens
    better: str  # higher, lower, or neutral


@dataclass(frozen=True)
class Selection:
    """The frozen items for one run, plus a JSON-safe record of how they were chosen."""

    items: Sequence[Item]
    identity: dict
    references: Any = field(default=None, compare=False)


class Strategy(Protocol):
    name: str
    timing_description: str  # what "seconds" covers, in words for chart notes
    generates_text: bool  # whether the answer is text the model wrote, rather than a chosen score

    @staticmethod
    def condition_label(spec: dict) -> str:
        """The chart group for runs with this input specification."""

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

    def score_for_comparison(self, run, scoring: dict) -> dict:
        """Rescore a saved run.

        Returns {"metrics": flat values, "cohort": what all compared runs must share, "item_ids": the
        scored items, whose responses alone the comparison's costs average}.
        """

    def quality_panels(self, scoring: dict) -> list:
        """The task's quality charts, as Panel values over the metric names it scores."""

    def condition_label(self, run) -> str:
        """The x-axis group for a run, for example its prompt or strategy."""

    def describe_comparison(self, rows: list, scoring: dict) -> dict:
        """{"context": one line under the title, "footnotes": notes on timing, validity, and scoring}."""

    def write_comparison_files(self, rows: list, scoring: dict, output: Path) -> list:
        """Write any task-specific comparison files, such as rewrites side by side; returns their paths."""

    def fit(self, spec) -> dict:
        """Fit what a fit file describes: {"report": JSON-safe details, "tensors": name -> float32 array}.

        A task with nothing to fit raises ValueError.
        """
