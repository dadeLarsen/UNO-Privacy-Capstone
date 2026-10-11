"""The classification task: frozen sentence samples, reviewed gold, and strict scoring."""

from ...core.conditions import Condition, take_settings
from ...core.paths import data_file
from ...core.files import read_json
from ...core.protocols import Selection
from .data import accepted_gold_digest, load_accepted_gold, read_dataset, select_rows
from .scoring import write_evaluation
from .strategies import STRATEGIES
from .taxonomy import LABELS, taxonomy_metadata


class ClassificationTask:
    name = "classification"

    def select_items(self, condition: Condition) -> Selection:
        """Load the sample and, when named, the reviewed gold before any model is loaded."""
        settings = take_settings(condition.items, f"Condition {condition.name!r} [items]",
                                 required=("dataset", "sample"), optional={"gold": None})
        dataset = read_dataset(data_file("datasets", settings["dataset"], suffix=".csv"))
        manifest = read_json(data_file("classification", "samples", settings["sample"], suffix=".json"))
        rows = select_rows(dataset, manifest)
        accepted, gold = None, None
        if settings["gold"] is not None:
            accepted = load_accepted_gold(data_file("classification", "gold", settings["gold"], suffix=".csv"), rows)
            gold = {"name": settings["gold"], "accepted_gold_sha256": accepted_gold_digest(rows, accepted)}
        identity = {
            "dataset": {"name": settings["dataset"], "sha256": dataset.sha256},
            "sample": {"name": settings["sample"], "manifest_sha256": manifest["manifest_sha256"], "n": len(rows)},
            "gold": gold, "labels": list(LABELS), **taxonomy_metadata(),
        }
        return Selection(items=rows, identity=identity, references=accepted)

    def build_strategy(self, condition: Condition):
        try:
            factory = STRATEGIES[condition.strategy]
        except KeyError:
            raise ValueError(f"Unknown classification strategy {condition.strategy!r}; "
                             f"available: {', '.join(sorted(STRATEGIES)) or 'none'}") from None
        return factory(condition)

    def evaluate(self, selection: Selection, responses: list, output) -> dict:
        return write_evaluation(output, selection.items, responses, selection.references)

    @staticmethod
    def summary_line(metrics: dict) -> str:
        line = (f"Rows: {metrics['n']}; accuracy: {metrics['accuracy']:.3%}; "
                f"macro F1: {metrics['macro_f1']:.4f}; valid: {metrics['valid_output_rate']:.3%}")
        if metrics["scoring_mode"] == "any_accepted_gold_label":
            line += (f"; accepted accuracy: {metrics['accepted_accuracy']:.3%}; "
                     f"macro accuracy: {metrics['macro_accuracy']:.3%}")
        return line
