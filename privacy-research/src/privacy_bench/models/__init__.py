"""Model backends: load a pinned local checkpoint, run one input, and measure it.

Backends know nothing about tasks. They take text or token IDs and return model outputs
(generated text, logits, decision probabilities, embeddings); strategies inside a task
turn those outputs into task answers.
"""
