"""The rewrite task: frozen policy passages, a prompt file per style, and checks for a human reviewer."""

from pathlib import Path

from ...core.conditions import Condition, take_settings
from ...core.files import read_json
from ...core.paths import RUNS_DIR
from ...core.protocols import Panel, Selection
from ...core.runs import load_run, run_condition
from .data import load_passages
from .scoring import evaluate, review_markdown, write_evaluation
from .strategy import PromptedRewriteStrategy

STRATEGIES = {strategy.name: strategy for strategy in (PromptedRewriteStrategy,)}


class RewriteTask:
    name = "rewrite"

    def select_items(self, condition: Condition) -> Selection:
        settings = take_settings(condition.items, f"Condition {condition.name!r} [items]", required=("passages",))
        passages, identity = load_passages(settings["passages"])
        return Selection(items=passages, identity=identity)

    def build_strategy(self, condition: Condition):
        try:
            factory = STRATEGIES[condition.strategy]
        except KeyError:
            raise ValueError(f"Unknown rewrite strategy {condition.strategy!r}; "
                             f"available: {', '.join(sorted(STRATEGIES))}") from None
        return factory(condition)

    def evaluate(self, selection: Selection, responses: list, output) -> dict:
        return write_evaluation(output, selection.items, responses)

    @staticmethod
    def summary_line(metrics: dict) -> str:
        def percent(value):
            return "N/A" if value is None else f"{value:.1%}"
        words, grade = metrics["mean_output_words"], metrics["reading_grade"]
        return (f"Passages: {metrics['n']}; valid: {percent(metrics['valid_output_rate'])}; "
                f"numbers kept: {percent(metrics['numbers_kept_rate'])}; "
                f"qualifiers kept: {percent(metrics['qualifiers_kept_rate'])}; "
                f"mean words: {'N/A' if words is None else f'{words:.0f}'}; "
                f"reading grade: {'N/A' if grade is None else f'{grade:.1f}'}")

    def _selection(self, run) -> Selection:
        selection = self.select_items(run_condition(run))
        if selection.identity != run.record["items"]:
            raise ValueError(f"The passages named by {run.path} no longer match its record")
        return selection

    def score_for_comparison(self, run, scoring: dict) -> dict:
        take_settings(scoring, "Comparison [scoring]")
        selection = self._selection(run)
        metrics, _ = evaluate(selection.items, run.responses)
        flat = {key: value for key, value in metrics.items()
                if key not in ("status_counts", "qualifiers", "checks_scope", "style_scope")}
        return {"item_ids": [passage.id for passage in selection.items], "metrics": flat,
                "cohort": selection.identity}

    @staticmethod
    def quality_panels(scoring: dict) -> list:
        return [
            Panel("valid_output_rate", "Valid rewrites", "Finished, nonempty, no reasoning tags (%)", 100,
                  "percent", "higher"),
            Panel("numbers_kept_rate", "Numbers kept", "Source numbers still in the rewrite (%)", 100,
                  "percent", "higher"),
            Panel("qualifiers_kept_rate", "Qualifiers kept", "Source hedges and negations still present (%)",
                  100, "percent", "higher"),
            Panel("qualifier_added_rate", "Qualifiers added", "Rewrites adding an exception, only, or a negation (%)",
                  100, "percent", "lower"),
            Panel("mean_output_words", "Rewrite length", "Mean words / valid rewrite", 1, "tokens", "neutral"),
            Panel("reading_grade", "Reading grade", "Flesch-Kincaid grade, pooled", 1, "tokens", "lower"),
            Panel("mean_copy_density", "Copying", "Mean copied run (words)", 1, "tokens", "neutral"),
            Panel("format_break_rate", "Format breaks", "Introduction or Markdown (%)", 100, "percent", "lower"),
        ]

    @staticmethod
    def condition_label(run) -> str:
        return STRATEGIES[run.record["strategy"]].condition_label(read_json(run.path / "input.json"))

    @staticmethod
    def describe_comparison(rows: list, scoring: dict) -> dict:
        prompts = ", ".join(dict.fromkeys(row["condition_label"] for row in rows))
        notes = [
            "Inference time: prompt processing + generation; excludes model loading and input preparation.",
            "Nothing here scores quality; read side-by-side.md. Valid means finished before the token limit, "
            "nonempty, and free of reasoning tags.",
            "Numbers and qualifiers kept pool every passage's distinct numbers (section labels such as 8.2 "
            "excluded) and words such as may, unless, only, not, and knowingly; a drop flags a rewrite to "
            "check, and keeping them proves nothing. Qualifiers added: an exception, only, or a negation the "
            "passage lacks, which can reverse a promise; rewording such as irreversible to cannot be undone "
            "also counts.",
            f"Reading grade: Flesch-Kincaid, pooled over valid rewrites; the passages themselves score "
            f"{rows[0]['source_reading_grade']:.1f}. Copying: the copied-run length an average rewrite word "
            "sits in (Grusky et al., 2018); 1-2 means the model's own words. Format breaks: an opening such "
            "as \"Here's the summary:\", or any Markdown.",
        ]
        return {"context": f"{rows[0]['n']} passages  |  prompts: {prompts}", "footnotes": notes}

    def write_comparison_files(self, rows: list, scoring: dict, output: Path, runs_root=RUNS_DIR) -> list:
        """side-by-side.md: each passage once, followed by every compared run's rewrite of it."""
        runs = [load_run(Path(runs_root) / row["run"]) for row in rows]
        passages = self._selection(runs[0]).items
        columns = [(row["legend_label"] or f"{row['model']} {row['precision']} · {row['condition_label']}",
                    {response["id"]: response for response in run.responses}) for row, run in zip(rows, runs)]
        path = Path(output) / "side-by-side.md"
        path.write_text(review_markdown(passages, columns), encoding="utf-8")
        return [path]

    def fit(self, spec) -> dict:
        raise ValueError("The rewrite task has nothing to fit")
