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


def select_rows(dataset, manifest) -> list:
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
    if manifest.get("task") != SAMPLE_TASK:
        raise ValueError(f"Manifest task must be {SAMPLE_TASK!r}; prepare a new manifest")
    ids = manifest.get("ids")
    if not isinstance(ids, list) or not ids or any(type(value) is not str for value in ids):
        raise ValueError("Manifest needs nonempty string IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("Manifest contains duplicate IDs")
    lookup = {row.id: row for row in dataset.rows}
    if any(row_id not in lookup for row_id in ids):
        raise ValueError("Manifest contains unknown IDs")
    rows = [lookup[row_id] for row_id in ids]
    if any(not row.eligible for row in rows):
        raise ValueError("The sample may contain only originally single-label gold rows")
    return rows


def normalize_accepted_labels(rows, accepted_labels) -> dict:
    """Validate alternative gold answers without changing the original label of any row."""
    if not isinstance(accepted_labels, dict) or set(accepted_labels) != {row.id for row in rows}:
        raise ValueError("Accepted gold IDs must exactly match the selected rows")
    normalized = {}
    for row in rows:
        labels = accepted_labels[row.id]
        if (not isinstance(labels, (list, tuple, set, frozenset)) or not labels
                or any(type(label) is not int or not 0 <= label < len(LABELS) for label in labels)):
            raise ValueError(f"Accepted gold labels must be nonempty valid category IDs: {row.id}")
        normalized[row.id] = frozenset(labels)
        if row.label_id not in normalized[row.id]:
            raise ValueError(f"Original gold label was removed: {row.id}")
    return normalized


def load_accepted_gold(path, rows) -> dict:
    """Read reviewed answers for exactly the sampled sentences, checking IDs and text."""
    gold = read_dataset(path, allow_target=True)
    if gold.audit["input_labels"] != list(LABELS):
        raise ValueError("Reviewed gold requires the ten-category CSV header")
    indexed = {row.id: row for row in gold.rows}
    if set(indexed) != {row.id for row in rows}:
        raise ValueError("Reviewed gold IDs must exactly match the frozen sample sentences")
    for row in rows:
        if indexed[row.id].text != row.text:
            raise ValueError(f"Reviewed gold text differs for ID {row.id!r}")
    return normalize_accepted_labels(rows, {row_id: row.labels for row_id, row in indexed.items()})


def accepted_gold_digest(rows, accepted) -> str:
    """Hash reviewed answers independently of row order and label order."""
    return digest_json({
        "labels": list(LABELS),
        "rows": [[row.id, row.text, sorted(accepted[row.id])] for row in sorted(rows, key=lambda item: item.id)],
    })
