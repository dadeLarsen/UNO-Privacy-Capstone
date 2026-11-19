"""The retired jev-choice strategy: a JEV judge chose one of the ten categories with its decision head.

Its backend is gone, so no condition runs it. What stays is what a comparison reads to label and annotate
its recorded runs.
"""

from ....core.conditions import Condition
from ..prompts import display_name
from .settings import retired


class JEVChoiceStrategy:
    name = "jev-choice"
    timing_description = "JEV forward pass + decision head"
    generates_text = False

    @staticmethod
    def condition_label(spec: dict) -> str:
        return "JEV choice\n" + display_name(spec["prompt_spec"]["name"]).lower()

    def __init__(self, condition: Condition):
        raise retired(condition)
