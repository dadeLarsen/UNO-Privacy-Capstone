"""Copy APPSI-139 from the AirLLM workspace into data/ and freeze its accepted labels and samples.

Usage::

    uv run python tools/prepare_appsi139.py ARCHIVE_DATASETS_DIR

ARCHIVE_DATASETS_DIR is privacy-classification/datasets in the AirLLM workspace. It holds:

- appsi139-privacy-model.csv: the archive's conversion of APPSI-139 to the ten categories, with one
  515-token sentence left out so that every model here can take every sentence.
- appsi139-reclassified.csv: the same sentences relabeled by ChatGPT Astra, which was asked to keep the
  original labels and add others where a sentence was ambiguous or mislabeled.

Both files must match their recorded SHA-256. The tool writes, and refuses to overwrite:

- data/datasets/appsi139-sentences.csv: the converted CSV, byte for byte.
- data/classification/gold/appsi139-astra.csv: every original label plus every label Astra marked.
  Astra dropped original labels on some sentences; the union keeps them, as the instruction intended.
- data/classification/samples/appsi139-all.json: every sentence.
- data/classification/samples/appsi139-random-1000.json: 1,000 sentences in the corpus's own mix.
- data/classification/samples/appsi139-balanced-50.json: up to 50 sentences per category, balanced on
  the accepted labels.
"""

import argparse
import csv
from pathlib import Path

from privacy_bench.core.files import digest_bytes, read_json, write_json
from privacy_bench.core.paths import DATA_DIR
from privacy_bench.tasks.classification.data import (
    load_accepted_gold, make_full_manifest, make_random_manifest, make_stratified_manifest, read_dataset,
    select_rows,
)
from privacy_bench.tasks.classification.taxonomy import HEADER, LABELS

SOURCES = {
    "appsi139-privacy-model.csv": "f1e74c1b1d21037b11b83e624789b027fb75c37fb577f55b9a93a66501862c84",
    "appsi139-reclassified.csv": "3b262b945551584350fbae27a351e64bc0246bd40ee7716ff159a17d6ce05397",
}
DATASET = DATA_DIR / "datasets" / "appsi139-sentences.csv"
GOLD = DATA_DIR / "classification" / "gold" / "appsi139-astra.csv"
SAMPLES = DATA_DIR / "classification" / "samples"


def read_source(folder: Path, name: str) -> bytes:
    data = (folder / name).read_bytes()
    if digest_bytes(data) != SOURCES[name]:
        raise ValueError(f"{folder / name} does not match its recorded SHA-256 {SOURCES[name]}")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path, help="privacy-classification/datasets in the AirLLM workspace")
    args = parser.parse_args(argv)
    manifests = {name: SAMPLES / f"{name}.json"
                 for name in ("appsi139-all", "appsi139-random-1000", "appsi139-balanced-50")}
    existing = [str(path) for path in (DATASET, GOLD, *manifests.values()) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite: {', '.join(existing)}")

    converted = read_source(args.archive, "appsi139-privacy-model.csv")
    read_source(args.archive, "appsi139-reclassified.csv")
    DATASET.write_bytes(converted)
    dataset = read_dataset(DATASET, allow_target=True)
    revised = read_dataset(args.archive / "appsi139-reclassified.csv", allow_target=True)
    if [(row.id, row.text) for row in dataset.rows] != [(row.id, row.text) for row in revised.rows]:
        raise ValueError("The relabeled CSV differs from the converted CSV in IDs, text, or order")

    with GOLD.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HEADER)
        for original, astra in zip(dataset.rows, revised.rows, strict=True):
            union = set(original.labels) | set(astra.labels)
            writer.writerow([original.id, original.text, *[int(i in union) for i in range(len(LABELS))]])
    accepted = load_accepted_gold(GOLD, dataset.rows)
    added = sum(accepted[row.id] != set(row.labels) for row in dataset.rows)
    dropped = sum(not set(row.labels) <= set(astra.labels) for row, astra in zip(dataset.rows, revised.rows))
    print(f"{len(dataset.rows)} sentences; Astra added labels to {added} and dropped an original label on "
          f"{dropped}, which the accepted labels keep")

    built = {
        "appsi139-all": make_full_manifest(dataset),
        "appsi139-random-1000": make_random_manifest(dataset, 1000, seed=42),
        "appsi139-balanced-50": make_stratified_manifest(dataset, accepted, labels_name="appsi139-astra",
                                                         per_class=50, seed=42),
    }
    for name, manifest in built.items():
        write_json(manifests[name], manifest)
        # A saved manifest must load back and select the same sentences.
        rows = select_rows(dataset, read_json(manifests[name]))
        print(f"{name}: {len(rows)} sentences")
    print(f"Wrote {DATASET}, {GOLD}, and {len(built)} sample manifests")


if __name__ == "__main__":
    main()
