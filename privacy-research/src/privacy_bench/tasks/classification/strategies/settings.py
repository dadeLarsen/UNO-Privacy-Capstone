"""Condition parsing and small helpers shared by the classification strategies."""

from ....core.conditions import Condition, take_settings
from ....core.files import digest_bytes
from ....models.hub import require_commit


def model_settings(condition: Condition, *, required=(), optional=None) -> dict:
    settings = take_settings(condition.model, f"Condition {condition.name!r} [model]",
                             required=("id", "revision", "precision", *required), optional=optional)
    require_commit(settings["id"], settings["revision"])
    return settings


def option_settings(condition: Condition, *, required=(), optional=None) -> dict:
    return take_settings(condition.options, f"Condition {condition.name!r} [options]",
                         required=required, optional=optional)


def text_sha256(text: str) -> str:
    return digest_bytes(text.encode("utf-8"))


def first_argmax(values) -> int:
    """Index of the largest value; the first index wins exact ties."""
    return max(range(len(values)), key=values.__getitem__)
