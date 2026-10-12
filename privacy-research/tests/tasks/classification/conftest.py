import csv
import json

import pytest

from privacy_bench.core.conditions import parse_condition
from privacy_bench.core.files import read_json
from privacy_bench.core.paths import data_file
from privacy_bench.tasks.classification.data import Row, read_dataset, select_rows
from privacy_bench.tasks.classification.taxonomy import LABELS, SOURCE_HEADER, SOURCE_LABELS, SOURCE_TO_TARGET


@pytest.fixture
def make_condition():
    """Return a function that builds a pilot classification condition from model and option tables."""
    def build(strategy, model, options):
        def table(values):
            return "".join(f"{key} = {json.dumps(value)}\n" for key, value in values.items())
        return parse_condition("test", f'task = "classification"\nstrategy = "{strategy}"\n'
                                       '[items]\ndataset = "opp115-sentences"\nsample = "pilot-200-10cat"\n'
                                       f"[model]\n{table(model)}[options]\n{table(options)}")
    return build


@pytest.fixture(scope="session")
def pilot_rows():
    dataset = read_dataset(data_file("datasets", "opp115-sentences", suffix=".csv"))
    rows = select_rows(dataset, read_json(data_file("classification", "samples", "pilot-200-10cat", suffix=".json")))
    return {row.id: row for row in rows}


@pytest.fixture
def write_dataset(tmp_path):
    """Return a function that writes synthetic rows as a twelve-label OPP-115 CSV."""
    def write(rows, filename="input.csv", header=SOURCE_HEADER):
        path = tmp_path / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for row in rows:
                original = row.source_labels if row.source_labels is not None else tuple(
                    SOURCE_TO_TARGET.index(label) for label in row.labels)
                writer.writerow([row.id, row.text, *[int(i in original) for i in range(len(SOURCE_LABELS))]])
        return path
    return write


@pytest.fixture
def balanced_dataset(write_dataset):
    rows = [Row(f"{label:02d}-{example}", f"Synthetic {label}/{example}", (label,))
            for label in range(len(LABELS)) for example in range(3)]
    rows.extend([Row("multi", "Two labels", (0, 1)),
                 Row("empty-label", "No labels", ()),
                 Row("collapsed-multi", "Two original Other labels", (9,), (5, 8))])
    return read_dataset(write_dataset(rows))
