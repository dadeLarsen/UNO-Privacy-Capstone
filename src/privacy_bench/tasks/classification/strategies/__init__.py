"""Ways to classify a sentence, by condition ``strategy`` name."""

from .classifier_head import ClassifierHeadStrategy
from .prompted import PromptedStrategy

STRATEGIES = {strategy.name: strategy for strategy in (PromptedStrategy, ClassifierHeadStrategy)}
