"""The retired prompted strategy: a chat model wrote one JSON category object, scored strictly.

Its transformers chat backend is gone, so no condition runs it. What stays is what a comparison reads to
label and annotate its recorded runs.
"""

from ....core.conditions import Condition
from ..prompts import display_name
from .settings import retired


class PromptedStrategy:
    name = "prompted"
    timing_description = "prompt processing + generation"
    generates_text = True

    @staticmethod
    def condition_label(spec: dict) -> str:
        return display_name(spec["prompt_spec"]["name"])

    def __init__(self, condition: Condition):
        raise retired(condition)
