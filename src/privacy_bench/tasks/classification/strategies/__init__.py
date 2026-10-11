"""Ways to classify a sentence, by condition ``strategy`` name."""

from .classifier_head import ClassifierHeadStrategy
from .jev_choice import JEVChoiceStrategy
from .prompted import PromptedStrategy

STRATEGIES = {strategy.name: strategy for strategy in (PromptedStrategy, ClassifierHeadStrategy, JEVChoiceStrategy)}
