"""The rewrite task: frozen policy passages, a prompt file per style, checks for a human reviewer, and blind
key-point judging."""

from pathlib import Path

from ...core.conditions import NAME_PATTERN, Condition, take_settings
from ...core.files import read_json
from ...core.paths import JUDGMENTS_DIR, RUNS_DIR
from ...core.protocols import Panel, Selection
from ...core.runs import load_run, run_condition
from . import judging
from .chart import draw_keypoints
from .data import load_passages
from .scoring import evaluate, review_markdown, write_evaluation
from .strategy import PromptedRewriteStrategy

STRATEGIES = {strategy.name: strategy for strategy in (PromptedRewriteStrategy,)}
JUDGE_PANELS = (
    Panel("keypoints_kept_share", "Key points kept", "Judged key points kept in full (%)", 100, "percent", "higher"),
    Panel("critical_kept_share", "Critical points kept", "Critical key points kept in full (%)", 100, "percent",
          "higher"),
    Panel("keypoints_contradicted_share", "Key points contradicted", "Judged key points contradicted (%)", 100,
          "percent", "lower"),
    Panel("misleading_rate", "Misleading rewrites", "Contradicted critical point or major invention (%)", 100,
          "percent", "lower"),
)


def judgment_directory(name) -> Path:
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        raise ValueError(f"[scoring] judgments must name a judgment under results/judgments/rewrite/, not {name!r}")
    return JUDGMENTS_DIR / "rewrite" / name


def judge_note(name) -> str:
    record = read_json(judgment_directory(name) / "judgment.json")
    checklist = record["checklist"]
    reviewed = f"reviewed by {checklist['reviewed_by']}" if checklist["reviewed_by"] else "not yet reviewed"
    return (f"Judge: {record['judge']}, blind to which model wrote each rewrite, against the key points in "
            f"{checklist['name']} ({reviewed}). Invalid rewrites are not judged. "
            f"Judgment: results/judgments/rewrite/{name}/")


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
        settings = take_settings(scoring, "Comparison [scoring]", optional={"judgments": None})
        selection = self._selection(run)
        metrics, _ = evaluate(selection.items, run.responses)
        flat = {key: value for key, value in metrics.items()
                if key not in ("status_counts", "qualifiers", "checks_scope", "style_scope")}
        cohort = selection.identity
        if settings["judgments"] is not None:
            judgment = judging.load(judgment_directory(settings["judgments"]), selection)
            flat.update(judging.run_scores(judgment, run))
            cohort = {**cohort, "judgments": {"name": settings["judgments"], "sha256": judgment["sha256"]}}
        return {"item_ids": [passage.id for passage in selection.items], "metrics": flat, "cohort": cohort}

    def _judged_runs(self, paths, runs_root) -> tuple:
        runs = {path: load_run(Path(runs_root) / path) for path in paths}
        selections = [self._selection(run) for run in runs.values()]
        if any(selection.identity != selections[0].identity for selection in selections):
            raise ValueError("Every judged run must rewrite the same passages")
        return runs, selections[0]

    def prepare_judgment(self, comparison, instructions: str, judge: str, output=None, runs_root=RUNS_DIR) -> Path:
        """Write blind prompts for a judge to grade every run in a comparison against the key points."""
        runs, selection = self._judged_runs(comparison.runs, runs_root)
        return judging.prepare(comparison.name, runs, selection, instructions, judge,
                               output or judgment_directory(comparison.name))

    def record_judgment(self, directory, runs_root=RUNS_DIR) -> dict:
        """Check a judge's replies and freeze the judgment."""
        paths = [entry["path"] for entry in read_json(Path(directory) / "judgment.json")["runs"]]
        runs, selection = self._judged_runs(paths, runs_root)
        return judging.record(directory, runs, selection)

    @staticmethod
    def quality_panels(scoring: dict) -> list:
        return [*RewriteTask.style_panels(), *(JUDGE_PANELS if scoring.get("judgments") else ())]

    @staticmethod
    def style_panels() -> list:
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
        if scoring.get("judgments"):
            notes[1] = "Valid means finished before the token limit, nonempty, and free of reasoning tags."
            notes.append(judge_note(scoring["judgments"]))
        return {"context": f"{rows[0]['n']} passages  |  prompts: {prompts}", "footnotes": notes}

    def write_comparison_files(self, rows: list, scoring: dict, output: Path, runs_root=RUNS_DIR) -> list:
        """side-by-side.md: each passage once, followed by every compared run's rewrite of it; with
        judgments, also keypoints.png and .svg."""
        runs = [load_run(Path(runs_root) / row["run"]) for row in rows]
        selection = self._selection(runs[0])
        passages = selection.items
        columns = [(row["legend_label"] or f"{row['model']} {row['precision']} · {row['condition_label']}",
                    {response["id"]: response for response in run.responses}) for row, run in zip(rows, runs)]
        path = Path(output) / "side-by-side.md"
        path.write_text(review_markdown(passages, columns), encoding="utf-8")
        if not scoring.get("judgments"):
            return [path]
        prompts = ", ".join(dict.fromkeys(row["condition_label"] for row in rows))
        notes = [
            "Each bar splits the key points of every rewrite the judge read: kept in full; partly kept, when a "
            "condition, exception, or limiting word such as may or only is lost or blurred; left out; or "
            "contradicted, when the rewrite says something the passage rules out.",
            "Critical key points are the ones a reader would be misled by losing. A rewrite could mislead when it "
            "contradicts a critical point or adds a major claim the passage does not make.",
            judge_note(scoring["judgments"]),
            f"Source: results/comparisons/{Path(output).name}/summary.csv",
        ]
        return [path, *draw_keypoints(rows, output, title="What the judge found in each model's rewrites",
                                      context=f"{rows[0]['n']} passages ({selection.identity['passages']['name']})  "
                                              f"|  prompts: {prompts}", footnotes=notes)]

    def fit(self, spec) -> dict:
        raise ValueError("The rewrite task has nothing to fit")
