"""The retired classifier-head strategy: a trained ten-category sequence classifier read the raw sentence.

Its transformers backend is gone, so no condition runs it. What stays is what a comparison reads to label and
annotate its recorded runs.
"""

from ....core.conditions import Condition
from .settings import retired


class ClassifierHeadStrategy:
    name = "classifier-head"
    timing_description = "classifier forward pass"
    generates_text = False

    @staticmethod
    def condition_label(spec: dict) -> str:
        return "Raw sentence\n(classifier)"

    def __init__(self, condition: Condition):
        raise retired(condition)
