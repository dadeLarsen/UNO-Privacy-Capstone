"""Model backends: load a pinned local checkpoint, run one input, and measure it.

Backends know nothing about tasks. They take text or token IDs and return model outputs
(generated text, logits, decision probabilities, embeddings); strategies inside a task
turn those outputs into task answers.
"""

# On Windows the CUDA build of llama.cpp needs cuBLAS, and only torch's CUDA build ships it: llama_cpp loads
# once torch is imported and fails before. Importing torch here keeps that order for every backend.
import torch
