import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import DistilBertConfig, DistilBertForSequenceClassification, PreTrainedTokenizerFast

# A permuted, punctuated label order, as published by skythrone/privacy-model.
CLASSIFIER_LABELS = (
    "Data Retention", "Data Security", "Do Not Track", "First Party Collection/Use",
    "International and Specific Audiences", "Other", "Policy Change",
    "Third Party Sharing/Collection", "User Access, Edit and Deletion", "User Choice/Control",
)


@pytest.fixture(scope="session")
def tiny_classifier(tmp_path_factory):
    """A randomly initialized ten-label DistilBERT checkpoint saved like a Hub snapshot."""
    snapshot = tmp_path_factory.mktemp("classifier")
    vocabulary = {token: index for index, token in enumerate(
        ("[UNK]", "[PAD]", "[CLS]", "[SEP]", "We", "retain", "café", "data", "."))}
    core = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    core.pre_tokenizer = Whitespace()
    core.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                            sep_token="[SEP]", model_max_length=64,
                            model_input_names=["input_ids", "attention_mask"]).save_pretrained(snapshot)
    config = DistilBertConfig(
        vocab_size=len(vocabulary), max_position_embeddings=64, n_layers=1, n_heads=2, dim=16, hidden_dim=32,
        dropout=0.0, attention_dropout=0.0, seq_classif_dropout=0.0, problem_type="single_label_classification",
        id2label=dict(enumerate(CLASSIFIER_LABELS)), label2id={label: i for i, label in enumerate(CLASSIFIER_LABELS)})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        DistilBertForSequenceClassification(config).save_pretrained(snapshot)
    return snapshot
