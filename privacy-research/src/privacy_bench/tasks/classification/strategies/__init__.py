"""Ways to classify a sentence, by condition ``strategy`` name.

``prompted`` and ``jev-choice`` are retired: they stay registered so that comparisons can describe their
recorded runs, and each refuses a condition.
"""

from .classifier_head import ClassifierHeadStrategy
from .embedding_head import EmbeddingHeadStrategy
from .embedding_similarity import EmbeddingSimilarityStrategy
from .jev_choice import JEVChoiceStrategy
from .prompted import PromptedStrategy

STRATEGIES = {strategy.name: strategy for strategy in (
    PromptedStrategy, ClassifierHeadStrategy, JEVChoiceStrategy, EmbeddingSimilarityStrategy, EmbeddingHeadStrategy)}
