from copy import deepcopy

from peft import LoraConfig, get_peft_model
import pytest
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import torch
from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig

from privacy_bench.core.files import write_json
from privacy_bench.models.jev.judge import SLOT_RANGES, VERBALIZERS


def make_decision_artifacts(hidden_size=2):
    judge = {
        "hidden_size": hidden_size, "weights_mode": "unmerged", "adapter_subfolder": "adapter", "softcap": None,
        "slots": {"num_slots": 24, "ranges": deepcopy(SLOT_RANGES), "verbalizers": list(VERBALIZERS),
                  "template_version": "bare-v1"},
        "verbalizer_ids": list(range(4, 28)),
        "lora": {"r": 2, "alpha": 4, "target_modules": ["q_proj", "v_proj"]},
    }
    calibration = {"per_kind": {"noul": 1.0, "score": 1.1, "choice": 2.0}}
    adapter = {"peft_type": "LORA", "task_type": None,
               "auto_mapping": {"base_model_class": "Qwen3_5ForCausalLM",
                                "parent_library": "transformers.models.qwen3_5.modeling_qwen3_5"},
               "r": 2, "lora_alpha": 4, "target_modules": ["q_proj", "v_proj"]}
    head = {"proj.weight": torch.zeros((24, hidden_size), dtype=torch.float32),
            "proj.bias": torch.zeros(24, dtype=torch.float32)}
    return judge, calibration, adapter, head


@pytest.fixture
def decision_artifacts():
    return make_decision_artifacts


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


@pytest.fixture
def text_config():
    return tiny_text_config


@pytest.fixture(scope="session")
def tiny_bundle(tmp_path_factory):
    """Randomly initialized hybrid Qwen weights with a nonzero LoRA and decision head, saved locally."""
    directory = tmp_path_factory.mktemp("jev")
    vocabulary = {token: index for index, token in enumerate([
        "[UNK]", "[PAD]", "[EOS]", "[BOS]", *VERBALIZERS,
        "kind", "choice", "state", "question", "options", "decision", ":", "[", "]", ")",
        "Sentence", "café", "retention", ".",
    ])}
    core = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    core.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
                                        bos_token="[BOS]", model_max_length=2048,
                                        model_input_names=["input_ids", "attention_mask"])
    tokenizer.save_pretrained(directory)
    config = tiny_text_config(vocab_size=len(vocabulary), pad_token_id=vocabulary["[PAD]"],
                              bos_token_id=vocabulary["[BOS]"], eos_token_id=vocabulary["[EOS]"])
    with torch.random.fork_rng():
        torch.manual_seed(42)
        base = Qwen3_5ForCausalLM(config)
        base.save_pretrained(directory, max_shard_size="12KB")
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["q_proj", "v_proj"]))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    parameter.normal_(mean=0.0, std=0.2)
        model.peft_config["default"].base_model_name_or_path = "Qwen/Qwen3.5-9B"
        model.save_pretrained(directory / "adapter", save_embedding_layers=False)
        judge, calibration, _, head = make_decision_artifacts(config.hidden_size)
        judge["verbalizer_ids"] = [vocabulary[token] for token in VERBALIZERS]
        head["proj.weight"].normal_(mean=0.0, std=0.2)
        head["proj.bias"].copy_(torch.linspace(-0.2, 0.2, 24))
    write_json(directory / "judge_config.json", judge)
    write_json(directory / "calibration.json", calibration)
    save_file(head, str(directory / "head.safetensors"))
    return {"path": directory, "tokenizer": tokenizer, "head": head, "calibration": calibration}
