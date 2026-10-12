"""The ten target categories and their mapping from the twelve OPP-115 source labels."""

SOURCE_LABELS = (
    "Policy Change", "Third Party Sharing/Collection", "User Choice/Control",
    "Data Retention", "User Access Edit and Deletion", "Practice not covered",
    "International and Specific Audiences", "Data Security",
    "Privacy contact information", "Do Not Track", "Introductory/Generic",
    "First Party Collection/Use",
)
SOURCE_HEADER = ("id", "text", *SOURCE_LABELS)
LABELS = (
    "Policy Change", "Third Party Sharing/Collection", "User Choice/Control",
    "Data Retention", "User Access Edit and Deletion",
    "International and Specific Audiences", "Data Security", "Do Not Track",
    "First Party Collection/Use", "Other",
)
# Entries are target IDs in SOURCE_LABELS order. Source annotations stay intact.
SOURCE_TO_TARGET = (0, 1, 2, 3, 4, 9, 5, 6, 9, 7, 9, 8)
HEADER = ("id", "text", *LABELS)
TAXONOMY_ID = "opp115-10-other"
TAXONOMY_VERSION = 1


def taxonomy_metadata() -> dict:
    """Return a fresh, JSON-safe record of the exact source-to-target contract."""
    return {
        "taxonomy_id": TAXONOMY_ID,
        "taxonomy_version": TAXONOMY_VERSION,
        "source_labels": list(SOURCE_LABELS),
        "source_to_target": list(SOURCE_TO_TARGET),
    }
