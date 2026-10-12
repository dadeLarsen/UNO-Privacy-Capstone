"""Datasets, frozen samples, and reviewed gold, including the real pilot files in data/."""

import copy
import csv
import json

import pytest

from privacy_bench.core.files import digest_json, read_json
from privacy_bench.core.paths import data_file
from privacy_bench.tasks.classification.data import (
    MANIFEST_SCHEMA_VERSION, SAMPLE_TASK, Row, accepted_gold_digest, load_accepted_gold,
    make_manifest, read_dataset, select_rows,
)
from privacy_bench.tasks.classification.taxonomy import (
    HEADER, LABELS, SOURCE_HEADER, SOURCE_LABELS, taxonomy_metadata,
)


def resigned(manifest):
    manifest = copy.deepcopy(manifest)
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = digest_json(manifest)
    return manifest


def write_target_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for row_id, text, labels in rows:
            writer.writerow([row_id, text, *[int(i in labels) for i in range(len(LABELS))]])
    return path


def test_csv_preserves_ids_text_quotes_and_newlines(write_dataset):
    rows = [Row("0007", 'A quoted "sentence", with a comma.\nAnd another line.', (0,)),
            Row('row,"x"', "Unicode sentence: café — résumé", (8,))]
    dataset = read_dataset(write_dataset(rows))
    assert [(r.id, r.text, r.labels) for r in dataset.rows] == [(r.id, r.text, r.labels) for r in rows]


def test_target_header_is_only_accepted_when_allowed(tmp_path):
    path = write_target_csv(tmp_path / "gold.csv", [("1", "Sentence", (0, 9))])
    assert read_dataset(path, allow_target=True).rows == [Row("1", "Sentence", (0, 9))]
    with pytest.raises(ValueError, match="original CSV header"):
        read_dataset(path)


def test_non_single_label_rows_are_excluded_without_mutating_source(balanced_dataset, tmp_path):
    path = tmp_path / "input.csv"
    original_bytes = path.read_bytes()
    assert balanced_dataset.audit["excluded_non_single_label_ids"] == ["multi", "empty-label", "collapsed-multi"]
    assert balanced_dataset.audit["excluded_multi_label_rows_collapsing_to_one_target"] == 1
    assert "before category merging" in balanced_dataset.audit["eligibility"]
    assert path.read_bytes() == original_bytes


def test_source_categories_map_to_ten_targets_with_all_three_extras_in_other(write_dataset):
    rows = [Row(str(i), f"Original category {name}", (), (i,)) for i, name in enumerate(SOURCE_LABELS)]
    dataset = read_dataset(write_dataset(rows))
    other_sources = {"Practice not covered", "Privacy contact information", "Introductory/Generic"}
    assert len(LABELS) == 10 and LABELS[9] == "Other"
    assert (len(HEADER), len(SOURCE_HEADER)) == (12, 14)
    for row, original in zip(dataset.rows, SOURCE_LABELS, strict=True):
        assert LABELS[row.label_id] == ("Other" if original in other_sources else original)
        assert row.source_labels == (SOURCE_LABELS.index(original),)
        assert row.eligible
    assert dataset.audit["single_label_support"]["Other"] == 3
    assert sum(dataset.audit["single_label_support"].values()) == 12


def test_multiple_original_other_annotations_remain_ineligible_after_mapping(write_dataset):
    rows = [Row("collapsed", "Original multi-label row", (), (5, 8, 10)),
            Row("eligible", "Single original label", (), (10,)),
            Row("zero", "No original labels", (), ())]
    dataset = read_dataset(write_dataset(rows))
    assert dataset.rows[0].labels == (9,)
    assert not dataset.rows[0].eligible
    with pytest.raises(ValueError, match="not single-label"):
        _ = dataset.rows[0].label_id
    assert dataset.audit["single_label_rows"] == 1
    assert dataset.audit["active_label_counts"] == {0: 1, 1: 1, 3: 1}
    assert dataset.audit["mapped_active_label_counts"] == {0: 1, 1: 2}


def test_balanced_manifest_is_reproducible_and_complete_per_class(balanced_dataset):
    manifest = make_manifest(balanced_dataset, per_class=2, seed=23)
    assert manifest == make_manifest(balanced_dataset, per_class=2, seed=23)
    rows = select_rows(balanced_dataset, manifest)
    assert len(rows) == 20
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["task"] == SAMPLE_TASK
    for key, value in taxonomy_metadata().items():
        assert manifest[key] == value
    assert [row.id for row in rows] == manifest["ids"]
    assert [sum(row.label_id == label for row in rows) for label in range(len(LABELS))] == [2] * len(LABELS)
    with pytest.raises(ValueError, match="only 3 single-label rows"):
        make_manifest(balanced_dataset, per_class=4)


def test_manifest_rejects_content_edit_without_new_hash(balanced_dataset):
    manifest = make_manifest(balanced_dataset, per_class=1)
    manifest["seed"] += 1
    with pytest.raises(ValueError, match="Manifest hash"):
        select_rows(balanced_dataset, manifest)


def test_manifest_rejects_changed_dataset_bytes(balanced_dataset, tmp_path):
    manifest = make_manifest(balanced_dataset, per_class=1)
    path = tmp_path / "input.csv"
    with path.open("ab") as handle:
        handle.write(b"\n")
    changed = read_dataset(path)
    assert changed.rows == balanced_dataset.rows
    with pytest.raises(ValueError, match="Dataset hash"):
        select_rows(changed, manifest)


@pytest.mark.parametrize("key, value, message", [
    ("labels", list(reversed(LABELS)), "category order"),
    ("ids", "duplicate", "duplicate IDs"),
    ("ids", ["does-not-exist"], "unknown IDs"),
    ("ids", ["multi"], "single-label"),
    ("ids", ["empty-label"], "single-label"),
    ("ids", ["collapsed-multi"], "originally single-label"),
    ("ids", [0], "string IDs"),
    ("ids", [], "string IDs"),
    ("schema_version", 999, "schema"),
    ("schema_version", 1, "Legacy 12-category"),
    ("task", "single_label_12_categories", "task"),
    ("taxonomy_id", "wrong", "taxonomy"),
    ("taxonomy_version", 999, "taxonomy"),
    ("source_labels", list(reversed(SOURCE_LABELS)), "taxonomy"),
    ("source_to_target", list(range(12)), "taxonomy"),
])
def test_resigned_manifest_still_rejects_wrong_mapping_and_invalid_ids(balanced_dataset, key, value, message):
    changed = make_manifest(balanced_dataset, per_class=1)
    changed[key] = [changed["ids"][0]] * 2 if value == "duplicate" else value
    with pytest.raises(ValueError, match=message):
        select_rows(balanced_dataset, resigned(changed))


def test_dataset_rejects_duplicate_ids_wrong_header_and_non_binary_values(write_dataset):
    rows = [Row("0", "Synthetic A", (0,)), Row("0", "Synthetic B", (1,))]
    with pytest.raises(ValueError, match="duplicate id"):
        read_dataset(write_dataset(rows))
    with pytest.raises(ValueError, match="header"):
        read_dataset(write_dataset(rows[:1], header=("id", "text", *reversed(SOURCE_LABELS))))
    path = write_dataset(rows[:1])
    with path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.reader(handle))
    records[1][2] = "true"
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(records)
    with pytest.raises(ValueError, match="Non-binary"):
        read_dataset(path)


def test_audit_identifies_conflicting_duplicate_text(write_dataset):
    rows = [Row("0", "Same synthetic sentence", (0,)), Row("1", "Same synthetic sentence", (0,)),
            Row("2", "Same synthetic sentence", (1,)), Row("3", "Different sentence", (2,))]
    audit = read_dataset(write_dataset(rows)).audit
    assert audit["single_label_duplicate_text_extra_rows"] == 2
    assert audit["single_label_conflicting_text_groups"] == 1
    assert audit["single_label_rows_in_conflicting_text_groups"] == 3


def test_reviewed_gold_must_cover_the_same_ids_and_text(tmp_path):
    rows = [Row("1", "First", (0,)), Row("2", "Second", (1,))]
    good = write_target_csv(tmp_path / "good.csv", [("2", "Second", (1, 9)), ("1", "First", (0,))])
    assert load_accepted_gold(good, rows) == {"1": frozenset({0}), "2": frozenset({1, 9})}
    cases = [
        ([("1", "First", (0,))], "exactly match"),
        ([("1", "First", (0,)), ("2", "Second", (1,)), ("3", "Third", (2,))], "exactly match"),
        ([("1", "First edited", (0,)), ("2", "Second", (1,))], "text differs"),
        ([("1", "First", (2,)), ("2", "Second", (1,))], "Original gold label was removed"),
        ([("1", "First", ()), ("2", "Second", (1,))], "nonempty valid category IDs"),
    ]
    for index, (content, message) in enumerate(cases):
        with pytest.raises(ValueError, match=message):
            load_accepted_gold(write_target_csv(tmp_path / f"bad-{index}.csv", content), rows)


def test_accepted_gold_digest_tracks_content_not_order():
    rows = [Row("002", "Second", (1,)), Row("001", "First", (0,))]
    digest = accepted_gold_digest(rows, {"002": frozenset({1, 9}), "001": frozenset({0, 2, 9})})
    assert accepted_gold_digest(rows[::-1], {"001": frozenset({9, 2, 0}), "002": frozenset({9, 1})}) == digest
    assert accepted_gold_digest(rows, {"002": frozenset({1, 9}), "001": frozenset({0, 2})}) != digest


class TestRealPilot:
    """The committed inputs must still agree with each other byte for byte."""

    def test_pilot_manifest_selects_200_balanced_rows_from_the_committed_csv(self):
        dataset = read_dataset(data_file("datasets", "opp115-sentences", suffix=".csv"))
        manifest = read_json(data_file("classification", "samples", "pilot-200-10cat", suffix=".json"))
        assert dataset.sha256 == "34dc5a59736e66caa570572f9cde340a1e3b1dcf02bedec726db9487b5542c8e"
        assert dataset.audit["total_rows"] == 10717 and dataset.audit["single_label_rows"] == 10690
        rows = select_rows(dataset, manifest)
        assert len(rows) == 200
        assert [sum(row.label_id == label for row in rows) for label in range(len(LABELS))] == [20] * 10
        assert manifest["manifest_sha256"] == "46702b141fee4aa75dba9dc8157a81661394790eeac386b49abfb06c9c5f1658"
        # The manifest records how it was drawn; redrawing must give the identical file
        # once integer audit keys become JSON strings, as they did when it was saved.
        assert json.loads(json.dumps(make_manifest(dataset, per_class=20, seed=42))) == manifest

    def test_reviewed_gold_matches_the_pilot_and_its_recorded_hash(self):
        dataset = read_dataset(data_file("datasets", "opp115-sentences", suffix=".csv"))
        rows = select_rows(dataset, read_json(data_file("classification", "samples", "pilot-200-10cat",
                                                        suffix=".json")))
        accepted = load_accepted_gold(data_file("classification", "gold", "pilot-200-10cat-reviewed",
                                                suffix=".csv"), rows)
        assert sum(len(labels) - 1 for labels in accepted.values()) == 54
        assert sum(len(labels) > 1 for labels in accepted.values()) == 53
        assert accepted_gold_digest(rows, accepted) == \
            "2b5d2860f9af75f144daf1517c4c18747a123d01695dfc22abf287b75fc6a03d"
