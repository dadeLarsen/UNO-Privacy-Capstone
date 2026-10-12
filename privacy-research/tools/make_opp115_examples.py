"""Freeze the OPP-115 sentences that classification heads learn from.

Usage::

    uv run python tools/make_opp115_examples.py

Writes data/classification/examples/opp115-examples.json: every single-label OPP-115 sentence outside the
pilot, without any sentence whose text repeats a pilot or APPSI-139 sentence, so neither test set leaks
into a fit. Refuses to overwrite.
"""

from privacy_bench.core.files import read_json, write_json
from privacy_bench.core.paths import DATA_DIR, data_file
from privacy_bench.tasks.classification.data import make_example_manifest, read_dataset, select_examples, select_rows

HELD_OUT = (("opp115-sentences", "pilot-200-10cat"), ("appsi139-sentences", "appsi139-all"))
OUTPUT = DATA_DIR / "classification" / "examples" / "opp115-examples.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    dataset = read_dataset(data_file("datasets", "opp115-sentences", suffix=".csv"))
    held_out = []
    for dataset_name, sample in HELD_OUT:
        source = read_dataset(data_file("datasets", dataset_name, suffix=".csv"), allow_target=True)
        manifest = read_json(data_file("classification", "samples", sample, suffix=".json"))
        held_out.append({"name": sample, "manifest": manifest, "rows": select_rows(source, manifest)})
    OUTPUT.parent.mkdir(exist_ok=True)
    write_json(OUTPUT, make_example_manifest(dataset, held_out))
    saved = read_json(OUTPUT)
    rows = select_examples(dataset, saved)
    print(f"{len(rows)} examples; left out {saved['excluded_held_out_rows']} pilot sentences and "
          f"{len(saved['excluded_repeated_text_ids'])} sentences repeating a held-out sentence")
    print(", ".join(f"{label} {count}" for label, count in saved["label_counts"].items()))


if __name__ == "__main__":
    main()
