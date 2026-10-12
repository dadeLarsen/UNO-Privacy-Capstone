"""Comparison files, summaries, and the matrix plot with a stand-in task."""

from dataclasses import dataclass
import json

from matplotlib.image import imread
import pytest

from privacy_bench.core.conditions import parse_condition
from privacy_bench.core.harness import execute
from privacy_bench.core.protocols import Panel, Selection
from privacy_bench.report.comparisons import parse_comparison
from privacy_bench.report.plot import draw_matrix, group_label
from privacy_bench.report.summary import COST_PANELS, summarize, write_summary


@dataclass(frozen=True)
class Item:
    id: str
    answer: str


class EchoStrategy:
    name = "echo"
    timing_scope = "echo"

    def __init__(self, condition):
        self.condition = condition

    @property
    def input_spec(self):
        return {"name": "echo", "flavour": self.condition.options["flavour"]}

    def load(self):
        return {"load_seconds": 1.5}

    def prepare(self, item):
        return item

    def describe(self, item):
        return {"input_tokens": 4}

    def predict(self, item):
        answer = item.answer if self.condition.model["precision"] != "int8" or item.id != "b" else "wrong"
        return {"answer": answer, "seconds": 0.5, "input_tokens": 4, "peak_allocated_bytes": 2**30}


class ToyTask:
    name = "toy"

    def __init__(self, items=("a", "b")):
        self.items = [Item(name, name.upper()) for name in items]

    def select_items(self, condition):
        return Selection(items=self.items, identity={"items": [item.id for item in self.items]})

    def build_strategy(self, condition):
        return EchoStrategy(condition)

    def evaluate(self, selection, responses, output):
        return {}

    def summary_line(self, metrics):
        return ""

    def score_for_comparison(self, run, scoring):
        correct = sum(response["answer"] == response["id"].upper() for response in run.responses)
        return {"metrics": {"n": len(run.responses), "accuracy": correct / len(run.responses)},
                "cohort": {"items": run.record["item_ids"], **scoring}}

    def quality_panels(self, scoring):
        return [Panel("accuracy", "Accuracy", "Correct (%)", 100, "percent", "higher")]

    def condition_label(self, run):
        return run.record["condition"]["options"]["flavour"].title()

    def describe_comparison(self, rows, scoring):
        return {"context": f"{rows[0]['n']} toy items", "footnotes": ["Toy footnote."]}


def make_run(root, name, precision, flavour, task=None):
    condition = parse_condition(name, f'task = "toy"\nstrategy = "echo"\n[model]\nid = "o/{name}"\n'
                                      f'revision = "{"a" * 40}"\nprecision = "{precision}"\n'
                                      f'[options]\nflavour = "{flavour}"\n')
    execute(task or ToyTask(), condition, root / "toy" / name / "run")
    return f"toy/{name}/run"


def comparison(runs):
    return parse_comparison("toy-comparison", f'task = "toy"\ntitle = "Toy comparison"\nruns = {json.dumps(runs)}\n')


def test_summary_rescores_each_run_and_adds_task_agnostic_costs(tmp_path):
    runs = [make_run(tmp_path, "m1", "bf16", "plain"), make_run(tmp_path, "m2", "int8", "plain")]
    rows = summarize(ToyTask(), comparison(runs), runs_root=tmp_path)
    assert [row["accuracy"] for row in rows] == [1.0, 0.5]
    assert [row["condition_label"] for row in rows] == ["Plain", "Plain"]
    assert rows[0]["mean_seconds"] == 0.5 and rows[0]["max_peak_allocated_bytes"] == 2**30
    assert rows[0]["load_seconds"] == 1.5 and rows[1]["precision"] == "int8"
    write_summary(rows, tmp_path / "out" / "summary.csv")
    assert (tmp_path / "out" / "summary.csv").read_text(encoding="utf-8").startswith("run,condition,strategy")


def test_runs_scored_on_different_items_are_refused(tmp_path):
    first = make_run(tmp_path, "m1", "bf16", "plain")
    second = make_run(tmp_path, "m2", "bf16", "spicy", task=ToyTask(items=("a", "c")))
    with pytest.raises(ValueError, match="same items"):
        summarize(ToyTask(), comparison([first, second]), runs_root=tmp_path)


def test_repeated_conditions_are_refused_instead_of_averaged(tmp_path):
    first = make_run(tmp_path, "m1", "bf16", "plain")
    task = ToyTask()
    condition = parse_condition("m1", (tmp_path / "toy" / "m1" / "run" / "condition.toml").read_text(encoding="utf-8"))
    execute(task, condition, tmp_path / "toy" / "m1" / "again")
    with pytest.raises(ValueError, match="one repetition at a time"):
        summarize(task, comparison([first, "toy/m1/again"]), runs_root=tmp_path)


def test_comparison_task_must_match(tmp_path):
    run = make_run(tmp_path, "m1", "bf16", "plain")
    other = parse_comparison("x", f'task = "classification"\nruns = ["{run}"]\n')
    with pytest.raises(ValueError, match="is for task 'classification'"):
        summarize(ToyTask(), other, runs_root=tmp_path)


@pytest.mark.parametrize("text, message", [
    ('runs = ["a/b"]\n', "task name"),
    ('task = "toy"\n', "nonempty list"),
    ('task = "toy"\nruns = []\n', "nonempty list"),
    ('task = "toy"\nruns = ["a/b", "a/b"]\n', "distinct"),
    ('task = "toy"\nruns = ["../a"]\n', "relative to results/runs"),
    ('task = "toy"\nruns = ["a\\\\b"]\n', "relative to results/runs"),
    ('task = "toy"\nruns = ["a/b"]\nextra = 1\n', "unknown keys: extra"),
    ('task = "toy"\nruns = ["a/b"]\n[labels]\n"c/d" = "x"\n', "must name a listed run"),
    ('task = "toy"\nruns = ["a/b"]\ntitle = ""\n', "title"),
])
def test_invalid_comparison_files_are_rejected(text, message):
    with pytest.raises(ValueError, match=message):
        parse_comparison("bad", text)


def test_matrix_plot_writes_png_and_svg_with_legend_overrides(tmp_path):
    runs = [make_run(tmp_path, "m1", "bf16", "plain"), make_run(tmp_path, "m2", "int8", "spicy")]
    parsed = parse_comparison("toy", 'task = "toy"\n' f"runs = {json.dumps(runs)}\n"
                                     f'[labels]\n"{runs[1]}" = "Custom label"\n')
    rows = summarize(ToyTask(), parsed, runs_root=tmp_path)
    assert rows[1]["legend_label"] == "Custom label"
    paths = draw_matrix(rows, [*ToyTask().quality_panels({}), *COST_PANELS], tmp_path / "plots",
                        title="Toy comparison", context="2 toy items", footnotes=["Toy footnote."])
    assert [path.name for path in paths] == ["matrix-overview.png", "matrix-overview.svg"]
    svg = paths[1].read_text(encoding="utf-8")
    assert "Custom label" in svg and "Toy footnote." in svg and "Accuracy · higher is better" in svg


def test_group_labels_wrap_each_line_splitting_words_only_at_hyphens():
    assert group_label("Embedding\nsimilarity\ncategories: search result") == (
        "Embedding\nsimilarity\ncategories:\nsearch result")
    assert group_label("Definitions + guidance") == "Definitions +\nguidance"
    assert group_label("an_unbreakable_identifier\n\nx") == "an_unbreakable_identifier\n\nx"
    assert group_label("linear-frequencies") == "linear-\nfrequencies"


def figure_pixels(tmp_path, flavours, folder):
    runs = [make_run(tmp_path / folder, f"m{i}", "bf16", flavour) for i, flavour in enumerate(flavours)]
    rows = summarize(ToyTask(), comparison(runs), runs_root=tmp_path / folder)
    paths = draw_matrix(rows, [*ToyTask().quality_panels({}), *COST_PANELS], tmp_path / folder / "plots",
                        title="Toy comparison", context="toy runs", footnotes=["Toy footnote."])
    return imread(paths[0]).shape[:2]


def test_matrix_plot_grows_to_fit_long_group_labels(tmp_path):
    short = figure_pixels(tmp_path, "abcde", "short")
    long = figure_pixels(tmp_path, [f"a very long flavour name with many words {i}" for i in range(5)], "long")
    assert short[1] == 16 * 180
    assert long[0] > short[0] and long[1] > short[1]


def test_many_groups_move_the_panels_into_fewer_columns_instead_of_widening_past_the_cap(tmp_path):
    three_columns = figure_pixels(tmp_path, "abcde", "few")
    fewer_columns = figure_pixels(tmp_path, [f"long flavour name {i}" for i in range(12)], "many")
    # Twelve groups would need a figure far wider than 24 inches in one row of three panels.
    assert fewer_columns[1] <= 24 * 180
    assert fewer_columns[0] > three_columns[0] * 1.4
