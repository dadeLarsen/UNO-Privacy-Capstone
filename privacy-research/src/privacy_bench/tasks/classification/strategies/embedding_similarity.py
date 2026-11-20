"""The retired embedding-similarity strategy: the category whose description embeds nearest the sentence won.

Its sentence-transformers backend is gone, so no condition runs it. What stays is what a comparison reads to
label and annotate its recorded runs.
"""

import re

from ....core.conditions import Condition
from ..prompts import display_name
from .settings import retired


def instruction_name(prefix: str) -> str:
    """A chart name for an instruction prefix: the task of 'task: <task> | query: ', or 'document'."""
    query = re.fullmatch(r"task: (.+) \| query: ", prefix)
    if query:
        return query.group(1)
    return "document" if re.fullmatch(r"title: .+ \| text: ", prefix) else prefix.strip()


class EmbeddingSimilarityStrategy:
    name = "embedding-similarity"
    timing_description = "embedding forward pass"
    generates_text = False

    @staticmethod
    def condition_label(spec: dict) -> str:
        sentence, category = spec["sentence_prefix"], spec["category_prefix"]
        instructions = (instruction_name(sentence) if sentence == category else
                        f"sentence: {instruction_name(sentence)}\ncategories: {instruction_name(category)}")
        return "Embedding\nsimilarity\n" + display_name(spec["prompt_spec"]["name"]).lower() + "\n" + instructions

    def __init__(self, condition: Condition):
        raise retired(condition)
