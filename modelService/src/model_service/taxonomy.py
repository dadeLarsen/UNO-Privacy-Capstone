"""The ten categories the classifier returns.

Their names, their order, and the taxonomy's ID and version come from the benchmark, so a
``label_id`` always means what the model was fitted on. ``taxonomy.json`` adds what only the
app needs: a display name, a definition, and a color for each category, keyed by its name.
"""

import json
from pathlib import Path

from privacy_bench.tasks.classification.taxonomy import LABELS, TAXONOMY_ID, TAXONOMY_VERSION

TAXONOMY_PATH = Path(__file__).with_name("taxonomy.json")


def load_taxonomy(path: Path = TAXONOMY_PATH) -> dict:
    presentation = json.loads(path.read_text(encoding="utf-8"))
    if set(presentation) != set(LABELS):
        raise ValueError(
            f"{path} must name exactly the benchmark's categories: "
            f"missing {sorted(set(LABELS) - set(presentation))}, extra {sorted(set(presentation) - set(LABELS))}"
        )
    return {
        "taxonomy_id": TAXONOMY_ID,
        "version": TAXONOMY_VERSION,
        "categories": [{"id": label_id, "name": name, **presentation[name]} for label_id, name in enumerate(LABELS)],
    }


TAXONOMY = load_taxonomy()
