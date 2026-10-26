"""The classification task: frozen sentence samples, reviewed gold, and strict scoring."""

from ...core.conditions import Condition, take_settings
from ...core.paths import data_file
from ...core.files import read_json
from ...core.protocols import Selection
from . import report
from .data import References, accepted_gold_digest, load_accepted_gold, read_dataset, sample_strata, select_rows
from .fitting import FITTERS
from .scoring import write_evaluation
from .strategies import STRATEGIES
from .taxonomy import LABELS, taxonomy_metadata


class ClassificationTask:
    name = "classification"

    def select_items(self, condition: Condition) -> Selection:
        """Load the sample and, when named, the reviewed gold before any model is loaded."""
        settings = take_settings(condition.items, f"Condition {condition.name!r} [items]",
                                 required=("dataset", "sample"), optional={"gold": None})
        # A dataset is either OPP-115's twelve-label export or a CSV already in the ten target categories.
        dataset = read_dataset(data_file("datasets", settings["dataset"], suffix=".csv"), allow_target=True)
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
        return Selection(items=rows, identity=identity,
                         references=References(accepted=accepted, strata=sample_strata(manifest, rows)))

    def build_strategy(self, condition: Condition):
        try:
            factory = STRATEGIES[condition.strategy]
        except KeyError:
            raise ValueError(f"Unknown classification strategy {condition.strategy!r}; "
                             f"available: {', '.join(sorted(STRATEGIES)) or 'none'}") from None
        return factory(condition)

    def evaluate(self, selection: Selection, responses: list, output) -> dict:
        references = selection.references
        return write_evaluation(output, selection.items, responses, references.accepted, references.strata)

    @staticmethod
    def summary_line(metrics: dict) -> str:
        line = (f"Rows: {metrics['n']}; accuracy: {metrics['accuracy']:.3%}; "
                f"macro F1: {metrics['macro_f1']:.4f}; valid: {metrics['valid_output_rate']:.3%}")
        if metrics["scoring_mode"] == "any_accepted_gold_label":
            line += (f"; accepted accuracy: {metrics['accepted_accuracy']:.3%}; "
                     f"macro accuracy: {metrics['macro_accuracy']:.3%}")
        return line

    def score_for_comparison(self, run, scoring: dict) -> dict:
        return report.score_for_comparison(self, run, scoring)

    @staticmethod
    def quality_panels(scoring: dict) -> list:
        return report.quality_panels(scoring)

    @staticmethod
    def condition_label(run) -> str:
        return report.condition_label(run)

    @staticmethod
    def describe_comparison(rows: list, scoring: dict) -> dict:
        return report.describe_comparison(rows, scoring)

    @staticmethod
    def write_comparison_files(rows: list, scoring: dict, output) -> list:
        return []

    def fit(self, spec) -> dict:
        try:
            fitter = FITTERS[spec.kind]
        except KeyError:
            raise ValueError(f"Unknown classification fit kind {spec.kind!r}; "
                             f"available: {', '.join(sorted(FITTERS)) or 'none'}") from None
        return fitter(spec).run()
