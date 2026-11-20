"""Condition parsing and small helpers shared by the classification strategies."""

from ....core.conditions import Condition, take_settings
from ....models.hub import require_commit


def model_settings(condition: Condition, *, required=(), optional=None) -> dict:
    settings = take_settings(condition.model, f"Condition {condition.name!r} [model]",
                             required=("id", "revision", "precision", *required), optional=optional)
    require_commit(settings["id"], settings["revision"])
    return settings


def option_settings(condition: Condition, *, required=(), optional=None) -> dict:
    return take_settings(condition.options, f"Condition {condition.name!r} [options]",
                         required=required, optional=optional)


def retired(condition: Condition) -> ValueError:
    """Why a strategy that only describes its recorded runs refuses a condition."""
    return ValueError(f"Condition {condition.name!r} names the retired strategy {condition.strategy!r}. Its "
                      "recorded runs still compare, but the code that ran it is gone; History in README.md "
                      "names the last commit that has it")


def first_argmax(values) -> int:
    """Index of the largest value; the first index wins exact ties."""
    return max(range(len(values)), key=values.__getitem__)
