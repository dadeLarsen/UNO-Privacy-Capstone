import csv

import pytest

from privacy_bench.tasks.classification.data import Row, read_dataset
from privacy_bench.tasks.classification.taxonomy import LABELS, SOURCE_HEADER, SOURCE_LABELS, SOURCE_TO_TARGET


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
