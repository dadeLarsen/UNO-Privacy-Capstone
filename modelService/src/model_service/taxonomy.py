"""The ten categories the classifier returns, loaded from the versioned JSON file."""

import json
from pathlib import Path

TAXONOMY_PATH = Path(__file__).with_name("taxonomy.json")
REQUIRED_FIELDS = ("id", "name", "display_name", "definition", "color")


def load_taxonomy(path: Path = TAXONOMY_PATH) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Taxonomy must be a JSON object: {path}")
    for key in ("taxonomy_id", "version"):
        if key not in data:
            raise ValueError(f"Taxonomy is missing {key!r}: {path}")
    categories = data.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"Taxonomy needs a nonempty categories list: {path}")
    if [category.get("id") for category in categories] != list(range(len(categories))):
        raise ValueError("Taxonomy category IDs must be 0..n-1 in order")
    names = [category.get("name") for category in categories]
    if len(set(names)) != len(names):
        raise ValueError("Taxonomy category names must be unique")
    for category in categories:
        for field in REQUIRED_FIELDS:
            value = category.get(field)
            if field == "id":
                if not isinstance(value, int):
                    raise ValueError("Taxonomy category IDs must be integers")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"Taxonomy category {category.get('id')!r} needs a nonempty {field}")
    return data


TAXONOMY = load_taxonomy()
LABELS = tuple(category["name"] for category in TAXONOMY["categories"])
