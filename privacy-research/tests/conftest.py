import pytest
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Normalize, Pooling, Transformer
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (
    BertConfig, BertModel, DistilBertConfig, DistilBertForSequenceClassification, PreTrainedTokenizerFast,
)

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


@pytest.fixture(scope="session")
def tiny_embedder(tmp_path_factory):
    """A randomly initialized BERT with mean pooling and normalization, saved as a sentence-transformers model."""
    root = tmp_path_factory.mktemp("embedder")
    vocabulary = {token: index for index, token in enumerate([
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "task", ":", "classification", "|", "query",
        "we", "keep", "data", "for", "years", "."])}
    core = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    core.pre_tokenizer = Whitespace()
    core.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                        sep_token="[SEP]", model_max_length=64,
                                        model_input_names=["input_ids", "attention_mask"])
    config = BertConfig(vocab_size=len(vocabulary), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        bert = BertModel(config)
    bert_dir = root / "bert"
    bert.save_pretrained(bert_dir)
    tokenizer.save_pretrained(bert_dir)
    model = SentenceTransformer(modules=[Transformer(str(bert_dir)), Pooling(16, pooling_mode="mean"), Normalize()])
    snapshot = root / "snapshot"
    model.save(str(snapshot))
    return snapshot
