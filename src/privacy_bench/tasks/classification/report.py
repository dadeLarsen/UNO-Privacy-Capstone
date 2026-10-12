"""How classification runs appear in comparisons: rescoring, panels, condition labels, and notes."""

from dataclasses import replace

from ...core.conditions import take_settings
from ...core.files import read_json
from ...core.protocols import Panel
from ...core.runs import Run, run_condition
from .scoring import evaluate
from .strategies import STRATEGIES

IDENTITY_WITHOUT_GOLD = ("dataset", "sample", "labels", "taxonomy_id", "taxonomy_version", "source_labels",
                         "source_to_target")


def scoring_settings(scoring: dict) -> dict:
    return take_settings(scoring, "Comparison [scoring]", optional={"gold": None})


def score_for_comparison(task, run: Run, scoring: dict) -> dict:
    """Rescore a run's saved responses against the comparison's gold, on the run's own sample."""
    gold = scoring_settings(scoring)["gold"]
    condition = run_condition(run)
    selection = task.select_items(replace(condition, items={**condition.items, "gold": gold}))
    recorded = run.record["items"]
    changed = [key for key in IDENTITY_WITHOUT_GOLD if selection.identity[key] != recorded[key]]
    if changed:
        raise ValueError(f"The data named by {run.path} no longer matches its record: {', '.join(changed)}")
    references = selection.references
    metrics, _, _ = evaluate(selection.items, run.responses, references.accepted, references.strata)
    reviewed = references.accepted is not None
    return {
        "metrics": {
            "sample": selection.identity["sample"]["name"], "gold": gold or "",
            "n": metrics["n"], "scoring_mode": metrics["scoring_mode"],
            "accuracy": metrics["accepted_accuracy"] if reviewed else metrics["accuracy"],
            "macro_accuracy": metrics.get("macro_accuracy"),
            "macro_accuracy_groups": metrics.get("macro_accuracy_groups"), "macro_f1": metrics["macro_f1"],
            "original_accuracy": metrics["accuracy"], "valid_output_rate": metrics["valid_output_rate"],
        },
        "cohort": selection.identity,
    }


def quality_panels(scoring: dict) -> list:
    reviewed = scoring_settings(scoring)["gold"] is not None
    second = (Panel("macro_accuracy", "Macro accuracy", "Equal weight per category (%)", 100, "percent",
                    "higher") if reviewed else
              Panel("macro_f1", "Macro-F1", "Equal weight per category", 1, "score", "higher"))
    return [Panel("accuracy", "Accuracy (accepted labels)" if reviewed else "Accuracy", "Correct sentences (%)",
                  100, "percent", "higher"),
            second,
            Panel("valid_output_rate", "Valid predictions", "Predictions with a valid category (%)", 100, "percent",
                  "higher")]


def condition_label(run: Run) -> str:
    return STRATEGIES[run.record["strategy"]].condition_label(read_json(run.path / "input.json"))


def describe_comparison(rows: list, scoring: dict) -> dict:
    gold = scoring_settings(scoring)["gold"]
    first = rows[0]
    strategies = list(dict.fromkeys(row["strategy"] for row in rows))
    timings = [STRATEGIES[name].timing_description for name in strategies]
    notes = ["Inference time: " + ", ".join(timings) + "; excludes model loading and input preparation."]
    if any(not STRATEGIES[name].generates_text for name in strategies):
        notes.append("Non-generative strategies write their top-scoring category as the answer, so they are always "
                     "valid; generative validity checks the JSON the model wrote.")
    if gold:
        grouping = ("the category the sample drew them for" if first["macro_accuracy_groups"] == "sample strata"
                    else "their original labels")
        notes.append(f"Accuracy accepts any category marked in {gold}; invalid outputs stay incorrect. "
                     f"Macro accuracy weights each category equally, grouping sentences by {grouping}.")
    else:
        notes.append("Accuracy accepts any original label of a sentence; macro-F1 uses the sentences with one "
                     "original label. Invalid outputs stay incorrect.")
    context = (f"{first['n']:,} sentences  |  sample {first['sample']}  |  "
               + (f"accepted labels {gold}" if gold else "original labels") + "  |  10 categories")
    return {"context": context, "footnotes": notes}
