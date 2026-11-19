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
    Qwen3_5ForCausalLM, Qwen3_5TextConfig,
)

# A permuted, punctuated label order, as published by skythrone/privacy-model.
CLASSIFIER_LABELS = (
    "Data Retention", "Data Security", "Do Not Track", "First Party Collection/Use",
    "International and Specific Audiences", "Other", "Policy Change",
    "Third Party Sharing/Collection", "User Access, Edit and Deletion", "User Choice/Control",
)
CHAT_TEMPLATE = ("{% for message in messages %}{{ message['role'] }} : {{ message['content'] }} [EOS] {% endfor %}"
                 "{% if add_generation_prompt %}assistant :{% endif %}")


def tiny_text_config(vocab_size=128, hidden_size=32, tied=False, **tokens):
    head_dim = hidden_size // 4
    return Qwen3_5TextConfig(
        vocab_size=vocab_size, hidden_size=hidden_size, intermediate_size=hidden_size * 2,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=head_dim,
        layer_types=["linear_attention", "full_attention"], max_position_embeddings=2048,
        linear_conv_kernel_dim=4, linear_key_head_dim=head_dim, linear_value_head_dim=head_dim,
        linear_num_key_heads=2, linear_num_value_heads=4, tie_word_embeddings=tied,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0,
                         "mrope_section": [head_dim // 8, head_dim // 8, head_dim // 4]},
        **tokens,
    )


@pytest.fixture(scope="session")
def text_config():
    return tiny_text_config


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


@pytest.fixture(scope="session")
def tiny_chat_model(tmp_path_factory):
    """A randomly initialized hybrid Qwen checkpoint with a chat template, saved like a Hub snapshot."""
    snapshot = tmp_path_factory.mktemp("chat")
    vocabulary = {token: index for index, token in enumerate(
        ("[UNK]", "[PAD]", "[EOS]", "user", "assistant", ":", "we", "keep", "data", "for", "years", "."))}
    core = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    core.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
                            model_max_length=64, chat_template=CHAT_TEMPLATE,
                            model_input_names=["input_ids", "attention_mask"]).save_pretrained(snapshot)
    config = tiny_text_config(vocab_size=len(vocabulary), pad_token_id=vocabulary["[PAD]"],
                              eos_token_id=vocabulary["[EOS]"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        Qwen3_5ForCausalLM(config).save_pretrained(snapshot)
    return snapshot
