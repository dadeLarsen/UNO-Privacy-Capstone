"""Freeze a tuning half and a test half of APPSI-139 that share no policy.

Usage::

    uv run python tools/split_appsi139.py

APPSI-139 has no policy IDs, but each policy's sentences appear in rising ID order, so a run of rising IDs
holds whole policies. Runs alternate between the halves in file order, starting with the tuning half.
Writes, and refuses to overwrite:

- data/classification/samples/appsi139-tuning-half.json: sentences a fit may tune on, never reported on.
- data/classification/samples/appsi139-test-half.json: the other half, for reporting.
"""

from collections import Counter

from privacy_bench.core.files import read_json, write_json
from privacy_bench.core.paths import DATA_DIR, data_file
from privacy_bench.tasks.classification.data import load_accepted_gold, make_half_manifests, read_dataset, select_rows
from privacy_bench.tasks.classification.taxonomy import LABELS

SAMPLES = DATA_DIR / "classification" / "samples"


def main():
    outputs = {half: SAMPLES / f"appsi139-{half}-half.json" for half in ("tuning", "test")}
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite: {', '.join(existing)}")
    dataset = read_dataset(data_file("datasets", "appsi139-sentences", suffix=".csv"), allow_target=True)
    accepted = load_accepted_gold(data_file("classification", "gold", "appsi139-astra", suffix=".csv"), dataset.rows)
    for half, manifest in make_half_manifests(dataset).items():
        write_json(outputs[half], manifest)
        # A saved manifest must load back and select the same sentences.
        rows = select_rows(dataset, read_json(outputs[half]))
        shares = Counter()
        for row in rows:
            for label in accepted[row.id]:
                shares[label] += 1 / len(accepted[row.id])
        print(f"{half}: {len(rows)} sentences in {manifest['id_runs_in_half']} of {manifest['id_runs']} ID runs")
        print("  " + ", ".join(f"{LABELS[i]} {shares[i] / len(rows):.3f}" for i in range(len(LABELS))))


if __name__ == "__main__":
    main()
