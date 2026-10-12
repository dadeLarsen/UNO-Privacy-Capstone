"""Datasets, frozen samples, and reviewed gold, including the real pilot files in data/."""

import copy
import csv
import json

import pytest

from privacy_bench.core.files import digest_json, read_json
from privacy_bench.core.paths import data_file
from privacy_bench.tasks.classification.data import (
    EXAMPLES_TASK, MANIFEST_SCHEMA_VERSION, MARKED_TASK, SAMPLE_TASK, TEST_PURPOSE, TUNING_PURPOSE, Row,
    accepted_gold_digest, increasing_id_runs, load_accepted_gold, make_example_manifest, make_full_manifest,
    make_half_manifests, make_manifest, make_random_manifest, make_stratified_manifest, normalized_text, read_dataset,
    sample_strata, select_examples, select_rows,
)
from privacy_bench.tasks.classification.taxonomy import (
    HEADER, LABELS, SOURCE_HEADER, SOURCE_LABELS, taxonomy_metadata,
)

APPSI_ACCEPTED_SHA256 = "bb7ff5aa677410e4ea51accee380281c90366087203b93e9170a99ae7792cd5c"


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


def test_reviewed_gold_must_cover_every_sentence_with_the_same_text(tmp_path):
    rows = [Row("1", "First", (0,)), Row("2", "Second", (1,))]
    good = write_target_csv(tmp_path / "good.csv", [("2", "Second", (1, 9)), ("1", "First", (0,))])
    assert load_accepted_gold(good, rows) == {"1": frozenset({0}), "2": frozenset({1, 9})}
    wider = write_target_csv(tmp_path / "wider.csv",
                             [("1", "First", (0,)), ("2", "Second", (1,)), ("3", "Third", (2,))])
    assert load_accepted_gold(wider, rows) == {"1": frozenset({0}), "2": frozenset({1})}
    cases = [
        ([("1", "First", (0,))], "missing for 1 of 2 sentences: 2"),
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


@pytest.fixture
def marked_dataset(tmp_path):
    """Ten-category sentences: three per category, except Do Not Track, which one two-label sentence carries."""
    rows = [(f"{label}-{n}", f"Sentence {label}/{n}", (label,))
            for label in range(len(LABELS)) if label != 7 for n in range(3)]
    rows += [("dnt", "Do Not Track and a policy change", (0, 7)), ("pair", "Choice and access", (2, 4)),
             ("none", "No label", ())]
    return read_dataset(write_target_csv(tmp_path / "marked.csv", rows), allow_target=True)


def test_full_and_random_samples_cover_labeled_sentences_without_strata(marked_dataset):
    labeled = [row.id for row in marked_dataset.rows if row.labels]
    full = make_full_manifest(marked_dataset)
    assert full["task"] == MARKED_TASK and full["selection"] == "all"
    rows = select_rows(marked_dataset, full)
    assert [row.id for row in rows] == labeled and sample_strata(full, rows) is None
    drawn = make_random_manifest(marked_dataset, 5, seed=3)
    assert drawn == make_random_manifest(marked_dataset, 5, seed=3)
    assert drawn["ids"] != make_random_manifest(marked_dataset, 5, seed=4)["ids"]
    rows = select_rows(marked_dataset, drawn)
    assert len(rows) == 5 and set(drawn["ids"]) <= set(labeled) and sample_strata(drawn, rows) is None
    for size in (0, len(labeled) + 1, True):
        with pytest.raises(ValueError, match="size must be an integer"):
            make_random_manifest(marked_dataset, size)


def test_balanced_sample_draws_the_scarcest_category_first_and_records_strata(marked_dataset):
    labels = {row.id: frozenset(row.labels) for row in marked_dataset.rows}
    manifest = make_stratified_manifest(marked_dataset, labels, labels_name="toy", per_class=2, seed=5)
    assert manifest == make_stratified_manifest(marked_dataset, labels, labels_name="toy", per_class=2, seed=5)
    rows = select_rows(marked_dataset, manifest)
    strata = sample_strata(manifest, rows)
    # Only one sentence carries Do Not Track, so it must be drawn for that category, not for Policy Change.
    assert strata["dnt"] == 7
    assert manifest["stratum_counts"] == {label: 1 if label == "Do Not Track" else 2 for label in LABELS}
    assert all(strata[row.id] in labels[row.id] for row in rows) and "none" not in strata
    assert manifest["balanced_on"] == {"name": "toy",
                                       "labels_sha256": accepted_gold_digest(marked_dataset.rows, labels)}
    with pytest.raises(ValueError, match="positive integer"):
        make_stratified_manifest(marked_dataset, labels, labels_name="toy", per_class=0)
    with pytest.raises(ValueError, match="cover every dataset sentence"):
        make_stratified_manifest(marked_dataset, {key: value for key, value in labels.items() if key != "none"},
                                 labels_name="toy", per_class=2)


@pytest.mark.parametrize("kind, key, change, message", [
    ("all", "ids", lambda ids: ids[::-1], "file order"),
    ("random", "ids", lambda ids: [*ids, "none"], "at least one label"),
    ("random", "selection", lambda value: "weird", "selection must be one of"),
    ("balanced", "strata", lambda strata: dict(list(strata.items())[1:]), "valid category stratum"),
    ("balanced", "strata", lambda strata: {key: 10 for key in strata}, "valid category stratum"),
])
def test_marked_samples_reject_bad_selections_unlabeled_sentences_and_missing_strata(marked_dataset, kind, key,
                                                                                     change, message):
    labels = {row.id: frozenset(row.labels) for row in marked_dataset.rows}
    manifest = {"all": make_full_manifest(marked_dataset),
                "random": make_random_manifest(marked_dataset, 5),
                "balanced": make_stratified_manifest(marked_dataset, labels, labels_name="toy", per_class=2)}[kind]
    manifest[key] = change(manifest[key])
    with pytest.raises(ValueError, match=message):
        select_rows(marked_dataset, resigned(manifest))


def test_halves_alternate_whole_runs_of_rising_ids(tmp_path):
    # Four runs of rising IDs: 10-12, 5-6, 3-4 (one unlabeled), and 1.
    rows = [("10", "a", (8,)), ("11", "b", (8,)), ("12", "c", (1,)), ("5", "d", (9,)), ("6", "e", (8, 9)),
            ("3", "f", (0,)), ("4", "g", ()), ("1", "h", (3,))]
    dataset = read_dataset(write_target_csv(tmp_path / "runs.csv", rows), allow_target=True)
    assert [[row.id for row in run] for run in increasing_id_runs(dataset.rows)] == \
        [["10", "11", "12"], ["5", "6"], ["3", "4"], ["1"]]
    halves = make_half_manifests(dataset)
    assert halves == make_half_manifests(dataset)
    tuning, test = halves["tuning"], halves["test"]
    assert tuning["ids"] == ["10", "11", "12", "3"] and test["ids"] == ["5", "6", "1"]
    assert (tuning["purpose"], test["purpose"]) == (TUNING_PURPOSE, TEST_PURPOSE)
    assert (tuning["half"], tuning["id_runs"]) == ("tuning", 4)
    assert (tuning["id_runs_in_half"], test["id_runs_in_half"]) == (2, 2)
    for manifest in (tuning, test):
        selected = select_rows(dataset, manifest)
        assert manifest["task"] == MARKED_TASK and sample_strata(manifest, selected) is None


@pytest.mark.parametrize("change, message", [
    (lambda manifest: {**manifest, "ids": manifest["ids"][::-1]}, "alternate ID runs in file order"),
    (lambda manifest: {**manifest, "ids": manifest["ids"][:-1]}, "alternate ID runs in file order"),
    (lambda manifest: {**manifest, "purpose": TEST_PURPOSE}, "name its half"),
    (lambda manifest: {**manifest, "half": "both"}, "name its half"),
])
def test_half_samples_must_match_their_split(tmp_path, change, message):
    rows = [("1", "a", (8,)), ("2", "b", (1,)), ("1000", "c", (9,)), ("3", "d", (0,)), ("4", "e", (8,))]
    dataset = read_dataset(write_target_csv(tmp_path / "runs.csv", rows), allow_target=True)
    with pytest.raises(ValueError, match=message):
        select_rows(dataset, resigned(change(make_half_manifests(dataset)["tuning"])))


def test_halves_need_integer_ids_and_two_runs(tmp_path):
    named = read_dataset(write_target_csv(tmp_path / "named.csv", [("1", "a", (8,)), ("x", "b", (8,))]),
                         allow_target=True)
    with pytest.raises(ValueError, match="decimal integer IDs"):
        make_half_manifests(named)
    rising = read_dataset(write_target_csv(tmp_path / "rising.csv", [("1", "a", (8,)), ("2", "b", (8,))]),
                          allow_target=True)
    with pytest.raises(ValueError, match="at least two ID runs"):
        make_half_manifests(rising)


def test_example_lists_leave_out_held_out_sentences_and_their_repeats(write_dataset, tmp_path):
    rows = [Row("held", "Held-out sentence", (0,)), Row("repeat", "  shared   TEXT ", (1,)),
            Row("keep-a", "Kept sentence", (2,)), Row("keep-b", "Another kept sentence", (2,)),
            Row("multi", "Two labels", (3, 4))]
    dataset = read_dataset(write_dataset(rows))
    other = read_dataset(write_target_csv(tmp_path / "other.csv", [("x", "Shared text", (5,))]), allow_target=True)
    held_out = [{"name": "own", "manifest": {"dataset_sha256": dataset.sha256, "manifest_sha256": "a" * 64},
                 "rows": [dataset.rows[0]]},
                {"name": "other", "manifest": {"dataset_sha256": other.sha256, "manifest_sha256": "b" * 64},
                 "rows": other.rows}]
    manifest = make_example_manifest(dataset, held_out)
    assert manifest == make_example_manifest(dataset, held_out) and manifest["task"] == EXAMPLES_TASK
    assert [row.id for row in select_examples(dataset, manifest)] == ["keep-a", "keep-b"]
    assert manifest["excluded_held_out_rows"] == 1 and manifest["excluded_repeated_text_ids"] == ["repeat"]
    assert manifest["label_counts"][LABELS[2]] == 2 and manifest["held_out"][1]["n"] == 1
    assert normalized_text("  shared   TEXT ") == normalized_text("Shared text")
    with pytest.raises(ValueError, match="single-label"):
        select_examples(dataset, resigned({**manifest, "ids": ["keep-a", "multi"]}))
    with pytest.raises(ValueError, match="task must be"):
        select_examples(dataset, resigned({**manifest, "task": SAMPLE_TASK}))
    with pytest.raises(ValueError, match="task must be"):
        select_rows(dataset, manifest)


class TestRealExamples:
    def test_the_example_list_redraws_exactly_and_shares_no_sentence_with_a_test_set(self):
        dataset = read_dataset(data_file("datasets", "opp115-sentences", suffix=".csv"))
        held_out, texts = [], set()
        for dataset_name, sample in (("opp115-sentences", "pilot-200-10cat"), ("appsi139-sentences", "appsi139-all")):
            source = read_dataset(data_file("datasets", dataset_name, suffix=".csv"), allow_target=True)
            manifest = read_json(data_file("classification", "samples", sample, suffix=".json"))
            held_out.append({"name": sample, "manifest": manifest, "rows": select_rows(source, manifest)})
            texts |= {normalized_text(row.text) for row in held_out[-1]["rows"]}
        saved = read_json(data_file("classification", "examples", "opp115-examples", suffix=".json"))
        assert json.loads(json.dumps(make_example_manifest(dataset, held_out))) == saved
        rows = select_examples(dataset, saved)
        assert len(rows) == 10417 and len(saved["excluded_repeated_text_ids"]) == 73
        pilot_ids = {row.id for row in held_out[0]["rows"]}
        assert not any(row.id in pilot_ids or normalized_text(row.text) in texts for row in rows)


class TestRealAPPSI:
    """APPSI-139, its accepted labels, its three samples, and its two halves must agree and redraw exactly."""

    def test_accepted_labels_keep_every_original_label_and_add_astra_labels(self):
        dataset = read_dataset(data_file("datasets", "appsi139-sentences", suffix=".csv"), allow_target=True)
        assert dataset.sha256 == "f1e74c1b1d21037b11b83e624789b027fb75c37fb577f55b9a93a66501862c84"
        assert dataset.audit["total_rows"] == 13291 and dataset.audit["single_label_rows"] == 12189
        accepted = load_accepted_gold(data_file("classification", "gold", "appsi139-astra", suffix=".csv"),
                                      dataset.rows)
        assert sum(accepted[row.id] != set(row.labels) for row in dataset.rows) == 4465
        assert accepted_gold_digest(dataset.rows, accepted) == APPSI_ACCEPTED_SHA256

    def test_samples_redraw_exactly_from_the_committed_files(self):
        dataset = read_dataset(data_file("datasets", "appsi139-sentences", suffix=".csv"), allow_target=True)
        accepted = load_accepted_gold(data_file("classification", "gold", "appsi139-astra", suffix=".csv"),
                                      dataset.rows)
        built = {"appsi139-all": make_full_manifest(dataset),
                 "appsi139-random-1000": make_random_manifest(dataset, 1000, seed=42),
                 "appsi139-balanced-50": make_stratified_manifest(dataset, accepted, labels_name="appsi139-astra",
                                                                  per_class=50, seed=42)}
        sizes = {}
        for name, manifest in built.items():
            saved = read_json(data_file("classification", "samples", name, suffix=".json"))
            assert json.loads(json.dumps(manifest)) == saved, name
            sizes[name] = len(select_rows(dataset, saved))
        assert sizes == {"appsi139-all": 13291, "appsi139-random-1000": 1000, "appsi139-balanced-50": 460}
        counts = built["appsi139-balanced-50"]["stratum_counts"]
        assert counts == {label: 10 if label == "Do Not Track" else 50 for label in LABELS}

    def test_halves_redraw_exactly_and_split_every_sentence_once(self):
        dataset = read_dataset(data_file("datasets", "appsi139-sentences", suffix=".csv"), allow_target=True)
        ids = {}
        for half, manifest in make_half_manifests(dataset).items():
            saved = read_json(data_file("classification", "samples", f"appsi139-{half}-half", suffix=".json"))
            assert json.loads(json.dumps(manifest)) == saved, half
            ids[half] = [row.id for row in select_rows(dataset, saved)]
        assert (len(ids["tuning"]), len(ids["test"])) == (7233, 6058)
        assert sorted(ids["tuning"] + ids["test"]) == sorted(row.id for row in dataset.rows)


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
