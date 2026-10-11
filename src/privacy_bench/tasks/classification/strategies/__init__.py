"""Ways to classify a sentence, by condition ``strategy`` name."""

from .prompted import PromptedStrategy

STRATEGIES = {strategy.name: strategy for strategy in (PromptedStrategy,)}
