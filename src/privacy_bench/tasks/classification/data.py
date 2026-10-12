"""Sentence datasets, frozen samples, and reviewed gold labels for classification."""

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import io
from pathlib import Path
import random

from ...core.files import digest_bytes, digest_json
from .taxonomy import HEADER, LABELS, SOURCE_HEADER, SOURCE_LABELS, SOURCE_TO_TARGET, taxonomy_metadata

MANIFEST_SCHEMA_VERSION = 2
SAMPLE_TASK = "single_label_10_categories"
# Sentences in these samples may carry several labels; a prediction matching any of them is correct.
MARKED_TASK = "marked_labels_10_categories"
MARKED_SELECTIONS = ("all", "random", "balanced_per_marked_category")
# Example lists hold sentences a fit learns from; they are never scored.
EXAMPLES_TASK = "single_label_examples_10_categories"


@dataclass(frozen=True)
class References:
    """What a sample's responses are scored against, beyond each sentence's own labels."""

    accepted: dict | None  # sentence ID -> accepted label IDs, from a gold file
    strata: dict | None  # sentence ID -> the category a stratified sample drew it for


@dataclass(frozen=True)
class Row:
    id: str
    text: str
    labels: tuple
    source_labels: tuple | None = None

    @property
    def eligible(self) -> bool:
        """Keep the original single-label eligibility from before categories were merged."""
        original = self.labels if self.source_labels is None else self.source_labels
        return len(original) == 1 and len(self.labels) == 1

    @property
    def label_id(self) -> int:
        if not self.eligible:
            raise ValueError(f"Row {self.id} is not single-label")
        return self.labels[0]


@dataclass
class Dataset:
    rows: list
    sha256: str
    audit: dict


def read_dataset(path, *, allow_target=False) -> Dataset:
    """Read twelve-label OPP-115 annotations, or ten-category CSVs when allow_target is set."""
    path = Path(path)
    rows, seen = [], set()
    raw = path.read_bytes()
    # The csv module handles quoted commas and embedded newlines. IDs stay strings.
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    source_input = reader.fieldnames == list(SOURCE_HEADER)
    target_input = allow_target and reader.fieldnames == list(HEADER)
    if not source_input and not target_input:
        raise ValueError(f"Expected exact original CSV header/order: {','.join(SOURCE_HEADER)}")
    input_labels = SOURCE_LABELS if source_input else LABELS
    for line, item in enumerate(reader, 2):
        if None in item or any(value is None for value in item.values()):
            raise ValueError(f"Malformed CSV record {line}: wrong number of fields")
        row_id = item["id"]
        if not row_id or row_id in seen:
            raise ValueError(f"Missing or duplicate id at record {line}: {row_id!r}")
        if not item["text"].strip():
            raise ValueError(f"Empty text at id {row_id}")
        if any(item[label] not in ("0", "1") for label in input_labels):
            raise ValueError(f"Non-binary label at id {row_id}")
        seen.add(row_id)
        original = tuple(i for i, name in enumerate(input_labels) if item[name] == "1")
        mapped = tuple(sorted({SOURCE_TO_TARGET[i] for i in original})) if source_input else original
        rows.append(Row(row_id, item["text"], mapped, original if source_input else None))
    if not rows:
        raise ValueError("Dataset is empty")
    single = [row for row in rows if row.eligible]
    by_text = defaultdict(list)
    for row in single:
        by_text[row.text].append(row.label_id)
    conflicts = [values for values in by_text.values() if len(set(values)) > 1]
    support = Counter(row.label_id for row in single)
    source_support = Counter((row.source_labels if row.source_labels is not None
                              else row.labels)[0] for row in single)
    audit = {
        "total_rows": len(rows), "single_label_rows": len(single),
        "eligibility": "Exactly one original annotation label before category merging"
                       if source_input else "Exactly one target label (mapped CSV roundtrip)",
        "input_labels": list(input_labels),
        "active_label_counts": dict(sorted(Counter(
            len(row.source_labels if row.source_labels is not None else row.labels)
            for row in rows).items())),
        "mapped_active_label_counts": dict(sorted(Counter(len(row.labels) for row in rows).items())),
        "excluded_non_single_label_ids": [row.id for row in rows if not row.eligible],
        "excluded_multi_label_rows_collapsing_to_one_target": sum(
            not row.eligible and len(row.labels) == 1 for row in rows),
        "single_label_support": {label: support[i] for i, label in enumerate(LABELS)},
        "original_single_label_support": {label: source_support[i]
                                          for i, label in enumerate(input_labels)},
        "single_label_duplicate_text_extra_rows": len(single) - len(by_text),
        "single_label_conflicting_text_groups": len(conflicts),
        "single_label_rows_in_conflicting_text_groups": sum(map(len, conflicts)),
        "policy_ids_available": False,
    }
    return Dataset(rows, digest_bytes(raw), audit)


def make_manifest(dataset, per_class=20, seed=42) -> dict:
    """Freeze a balanced sample of originally single-label rows."""
    if per_class < 1:
        raise ValueError("per_class must be positive")
    single = [row for row in dataset.rows if row.eligible]
    if not single:
        raise ValueError("Dataset has no single-label rows")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in single:
        groups[row.label_id].append(row)
    selected = []
    for i in range(len(LABELS)):
        if len(groups[i]) < per_class:
            raise ValueError(f"{LABELS[i]} has only {len(groups[i])} single-label rows; reduce per_class")
        selected.extend(rng.sample(groups[i], per_class))
    rng.shuffle(selected)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        **taxonomy_metadata(),
        "dataset_sha256": dataset.sha256,
        "labels": list(LABELS),
        "task": SAMPLE_TASK,
        "purpose": "development_pilot_not_policy_held_out",
        "selection": "balanced_per_category",
        "per_class": per_class,
        "seed": seed, "ids": [row.id for row in selected],
        "audit": dataset.audit,
    }
    manifest["manifest_sha256"] = digest_json(manifest)
    return manifest


def _marked_manifest(dataset, selection: str, ids: list, **fields) -> dict:
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        **taxonomy_metadata(),
        "dataset_sha256": dataset.sha256,
        "labels": list(LABELS),
        "task": MARKED_TASK,
        "purpose": "test_on_a_different_corpus",
        "selection": selection,
        **fields,
        "ids": ids,
        "audit": dataset.audit,
    }
    manifest["manifest_sha256"] = digest_json(manifest)
    return manifest


def _labeled_rows(dataset) -> list:
    rows = [row for row in dataset.rows if row.labels]
    if not rows:
        raise ValueError("Dataset has no labeled rows")
    return rows


def make_full_manifest(dataset) -> dict:
    """Freeze every labeled sentence, in file order, for scoring against all of its labels."""
    return _marked_manifest(dataset, "all", [row.id for row in _labeled_rows(dataset)])


def make_random_manifest(dataset, size: int, seed=42) -> dict:
    """Freeze a simple random sample of labeled sentences, keeping the corpus's own category mix."""
    rows = _labeled_rows(dataset)
    if type(size) is not int or not 0 < size <= len(rows):
        raise ValueError(f"size must be an integer from 1 through {len(rows)}")
    chosen = random.Random(seed).sample(rows, size)
    return _marked_manifest(dataset, "random", [row.id for row in chosen], size=size, seed=seed)


def make_stratified_manifest(dataset, labels: dict, *, labels_name: str, per_class: int, seed=42) -> dict:
    """Freeze up to per_class sentences per category, drawn among sentences whose given labels include it.

    ``labels`` maps every dataset sentence to the label set to balance on, such as a gold file's accepted
    labels. The scarcest category draws first and a sentence is drawn for one category only; the manifest
    records that category as the sentence's stratum.
    """
    if type(per_class) is not int or per_class < 1:
        raise ValueError("per_class must be a positive integer")
    rows = _labeled_rows(dataset)
    if set(labels) != {row.id for row in dataset.rows}:
        raise ValueError("Balancing labels must cover every dataset sentence")
    available = {i: [row for row in rows if i in labels[row.id]] for i in range(len(LABELS))}
    rng = random.Random(seed)
    strata = {}
    for i in sorted(range(len(LABELS)), key=lambda category: (len(available[category]), category)):
        candidates = [row for row in available[i] if row.id not in strata]
        if not candidates:
            raise ValueError(f"No undrawn sentence carries {LABELS[i]!r}")
        for row in rng.sample(candidates, min(per_class, len(candidates))):
            strata[row.id] = i
    ids = list(strata)
    rng.shuffle(ids)
    counts = Counter(strata.values())
    return _marked_manifest(
        dataset, "balanced_per_marked_category", ids, per_class=per_class, seed=seed,
        balanced_on={"name": labels_name, "labels_sha256": accepted_gold_digest(dataset.rows, labels)},
        stratum_counts={label: counts[i] for i, label in enumerate(LABELS)},
        strata={row_id: strata[row_id] for row_id in ids})


def _check_marked_sample(dataset, manifest, rows):
    selection = manifest.get("selection")
    if selection not in MARKED_SELECTIONS:
        raise ValueError(f"Manifest selection must be one of {', '.join(MARKED_SELECTIONS)}")
    if any(not row.labels for row in rows):
        raise ValueError("Every sampled sentence needs at least one label")
    if selection == "all" and manifest["ids"] != [row.id for row in dataset.rows if row.labels]:
        raise ValueError("An 'all' sample must list every labeled sentence in file order")
    if selection == "balanced_per_marked_category":
        strata = manifest.get("strata")
        if (not isinstance(strata, dict) or set(strata) != set(manifest["ids"])
                or any(type(value) is not int or not 0 <= value < len(LABELS) for value in strata.values())):
            raise ValueError("A balanced sample must give one valid category stratum for every sentence")


def sample_strata(manifest, rows) -> dict | None:
    """The category each sentence was drawn for when the sample was stratified, otherwise None."""
    if manifest["task"] == SAMPLE_TASK:
        return {row.id: row.label_id for row in rows}
    if manifest["selection"] == "balanced_per_marked_category":
        return {row.id: manifest["strata"][row.id] for row in rows}
    return None


def _manifest_rows(dataset, manifest, tasks) -> list:
    """Check a frozen ID list's hash, dataset, and taxonomy, and return its rows in order."""
    content = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != digest_json(content):
        raise ValueError("Manifest hash does not match its contents; prepare a new manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported manifest schema; prepare a new 10-category manifest. "
                         "Legacy 12-category manifests cannot be reused.")
    if manifest.get("dataset_sha256") != dataset.sha256:
        raise ValueError("Dataset hash differs from the frozen manifest")
    if manifest.get("labels") != list(LABELS):
        raise ValueError("Manifest category order differs from the CSV contract")
    for key, value in taxonomy_metadata().items():
        if manifest.get(key) != value:
            raise ValueError(f"Manifest taxonomy field {key!r} differs; prepare a new 10-category manifest")
    if manifest.get("task") not in tasks:
        raise ValueError(f"Manifest task must be {' or '.join(map(repr, tasks))}; prepare a new manifest")
    ids = manifest.get("ids")
    if not isinstance(ids, list) or not ids or any(type(value) is not str for value in ids):
        raise ValueError("Manifest needs nonempty string IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("Manifest contains duplicate IDs")
    lookup = {row.id: row for row in dataset.rows}
    if any(row_id not in lookup for row_id in ids):
        raise ValueError("Manifest contains unknown IDs")
    return [lookup[row_id] for row_id in ids]


def normalized_text(text: str) -> str:
    """Sentence text compared without case or spacing, to find a sentence repeated elsewhere."""
    return " ".join(text.split()).lower()


def make_example_manifest(dataset, held_out: list) -> dict:
    """Freeze the single-label sentences a fit may learn from, keeping every test sentence out.

    ``held_out`` lists each test sample as {"name", "manifest", "rows"}. A held-out sample of this
    dataset removes its own sentences, and a sentence whose normalized text repeats any held-out
    sentence, in any dataset, is removed too.
    """
    held_ids = {row.id for sample in held_out if sample["manifest"]["dataset_sha256"] == dataset.sha256
                for row in sample["rows"]}
    held_texts = {normalized_text(row.text) for sample in held_out for row in sample["rows"]}
    candidates = [row for row in dataset.rows if row.eligible and row.id not in held_ids]
    repeated = [row.id for row in candidates if normalized_text(row.text) in held_texts]
    examples = [row for row in candidates if normalized_text(row.text) not in held_texts]
    if not examples:
        raise ValueError("No single-label sentence is left to learn from")
    counts = Counter(row.label_id for row in examples)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        **taxonomy_metadata(),
        "dataset_sha256": dataset.sha256,
        "labels": list(LABELS),
        "task": EXAMPLES_TASK,
        "purpose": "fit_examples",
        "selection": "single_label_outside_held_out_samples",
        "held_out": [{"name": sample["name"], "dataset_sha256": sample["manifest"]["dataset_sha256"],
                      "manifest_sha256": sample["manifest"]["manifest_sha256"], "n": len(sample["rows"])}
                     for sample in held_out],
        "excluded_held_out_rows": sum(row.eligible and row.id in held_ids for row in dataset.rows),
        "excluded_repeated_text_ids": repeated,
        "label_counts": {label: counts[i] for i, label in enumerate(LABELS)},
        "ids": [row.id for row in examples],
        "audit": dataset.audit,
    }
    manifest["manifest_sha256"] = digest_json(manifest)
    return manifest


def select_examples(dataset, manifest) -> list:
    """The single-label sentences of a frozen example list, in order."""
    rows = _manifest_rows(dataset, manifest, (EXAMPLES_TASK,))
    if any(not row.eligible for row in rows):
        raise ValueError("Examples may contain only originally single-label rows")
    return rows


def select_rows(dataset, manifest) -> list:
    rows = _manifest_rows(dataset, manifest, (SAMPLE_TASK, MARKED_TASK))
    if manifest["task"] == MARKED_TASK:
        _check_marked_sample(dataset, manifest, rows)
    elif any(not row.eligible for row in rows):
        raise ValueError("The sample may contain only originally single-label gold rows")
    return rows


def normalize_accepted_labels(rows, accepted_labels) -> dict:
    """Validate alternative gold answers without removing any original label of a row."""
    if not isinstance(accepted_labels, dict) or set(accepted_labels) != {row.id for row in rows}:
        raise ValueError("Accepted gold IDs must exactly match the selected rows")
    normalized = {}
    for row in rows:
        labels = accepted_labels[row.id]
        if (not isinstance(labels, (list, tuple, set, frozenset)) or not labels
                or any(type(label) is not int or not 0 <= label < len(LABELS) for label in labels)):
            raise ValueError(f"Accepted gold labels must be nonempty valid category IDs: {row.id}")
        normalized[row.id] = frozenset(labels)
        if not set(row.labels) <= normalized[row.id]:
            raise ValueError(f"Original gold label was removed: {row.id}")
    return normalized


def load_accepted_gold(path, rows) -> dict:
    """Read accepted answers for the given sentences, checking their text; the file may cover more sentences."""
    gold = read_dataset(path, allow_target=True)
    if gold.audit["input_labels"] != list(LABELS):
        raise ValueError("Reviewed gold requires the ten-category CSV header")
    indexed = {row.id: row for row in gold.rows}
    missing = [row.id for row in rows if row.id not in indexed]
    if missing:
        raise ValueError(f"Gold labels are missing for {len(missing)} of {len(rows)} sentences: {', '.join(missing)}")
    for row in rows:
        if indexed[row.id].text != row.text:
            raise ValueError(f"Reviewed gold text differs for ID {row.id!r}")
    return normalize_accepted_labels(rows, {row.id: indexed[row.id].labels for row in rows})


def accepted_gold_digest(rows, accepted) -> str:
    """Hash reviewed answers independently of row order and label order."""
    return digest_json({
        "labels": list(LABELS),
        "rows": [[row.id, row.text, sorted(accepted[row.id])] for row in sorted(rows, key=lambda item: item.id)],
    })
