"""Sequence classification: stand-in models for edge cases, plus a real tiny CPU checkpoint."""

import json
import math
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from privacy_bench.models import measure, sequence_classifier as backend
from privacy_bench.models.sequence_classifier import SequenceClassifier


def stub(logits, limit=512):
    model = SequenceClassifier(".", device="cpu", max_input_tokens=limit)
    model._config = SimpleNamespace(num_labels=10)
    model._model = Mock(return_value=SimpleNamespace(logits=logits))
    return model


def test_logits_and_probabilities_stay_in_checkpoint_label_order():
    logits = torch.zeros((1, 10))
    logits[0, 6] = 3
    result = stub(logits).classify({"input_ids": [101, 1000, 102], "attention_mask": [1, 1, 1]})
    assert result["logits"] == logits[0].tolist()
    assert result["probabilities"][6] == pytest.approx(math.exp(3) / (math.exp(3) + 9), abs=1e-6)
    assert (result["input_tokens"], result["input_token_ids"]) == (3, [101, 1000, 102])
    assert len(result["input_token_sha256"]) == 64
    assert "peak_allocated_bytes" not in result and "output_tokens" not in result


def test_forward_receives_one_sentence_in_inference_mode():
    model = stub(torch.zeros((1, 10)))

    def forward(**inputs):
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        assert inputs["input_ids"].tolist() == [[101, 14, 102]] and inputs["input_ids"].dtype == torch.long
        assert inputs["attention_mask"].tolist() == [[1, 1, 1]]
        return SimpleNamespace(logits=torch.zeros((1, 10)))

    model._model.side_effect = forward
    model.classify({"input_ids": [101, 14, 102], "attention_mask": [1, 1, 1]})
    model._model.assert_called_once()


def test_encode_uses_the_exact_sentence_with_special_tokens_and_no_truncation():
    model = stub(torch.zeros((1, 10)), limit=4)
    model._tokenizer = Mock(return_value={"input_ids": [101, 12, 13, 102]})
    text = 'Sentence with "quotes", café and\na newline.'
    assert len(model.encode(text)["input_ids"]) == 4
    model._tokenizer.assert_called_once_with(text, add_special_tokens=True, truncation=False, padding=False,
                                             return_attention_mask=True)
    model._tokenizer.return_value = {"input_ids": [101, 12, 13, 14, 102]}
    with pytest.raises(ValueError, match="Input was not truncated"):
        model.encode("Too long")
    with pytest.raises(ValueError, match="Input was not truncated"):
        model.classify(model._tokenizer.return_value)
    model._model.assert_not_called()


@pytest.mark.parametrize("text", [None, "", "  ", {"text": "sentence"}, ["sentence"]])
def test_raw_input_must_be_a_nonempty_string(text):
    model = stub(torch.zeros((1, 10)))
    model._tokenizer = Mock()
    with pytest.raises(ValueError, match="raw sentence"):
        model.encode(text)
    model._tokenizer.assert_not_called()


@pytest.mark.parametrize("encoded", [None, {}, {"input_ids": []}, {"input_ids": [[1, 2]]}, {"input_ids": [True]},
                                     {"input_ids": [-1]}, {"input_ids": [1, 2], "attention_mask": [1]}])
def test_malformed_encodings_fail_before_forward(encoded):
    model = stub(torch.zeros((1, 10)))
    with pytest.raises(ValueError):
        model.classify(encoded)
    model._model.assert_not_called()


@pytest.mark.parametrize("logits", [torch.zeros(10), torch.zeros((2, 10)), torch.zeros((1, 9)), [[0.0] * 10],
                                    torch.full((1, 10), float("nan")), torch.full((1, 10), float("inf")),
                                    torch.zeros((1, 10), dtype=torch.long)])
def test_wrong_logit_shapes_and_nonfinite_values_are_rejected(logits):
    with pytest.raises(ValueError):
        stub(logits).classify({"input_ids": [101, 102]})


def test_missing_logits_are_an_explicit_error():
    model = stub(torch.zeros((1, 10)))
    model._model.return_value = SimpleNamespace()
    with pytest.raises(ValueError, match="no logits"):
        model.classify({"input_ids": [101, 102]})


def test_cuda_timing_synchronizes_around_the_forward_pass_and_records_peaks():
    model = stub(torch.zeros((1, 10)))
    model._cuda, events = True, []

    def forward(**inputs):
        events.append("forward")
        return SimpleNamespace(logits=torch.zeros((1, 10)))

    model._model.side_effect = forward
    with patch.object(torch.cuda, "synchronize", side_effect=lambda device: events.append("synchronize")), \
            patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=lambda device: events.append("reset")), \
            patch.object(torch.cuda, "max_memory_allocated", return_value=200), \
            patch.object(torch.cuda, "max_memory_reserved", return_value=300), \
            patch.object(measure.time, "perf_counter", side_effect=[10.0, 10.25]):
        result = model.classify({"input_ids": [101, 102]})
    assert events == ["synchronize", "reset", "forward", "synchronize"]
    assert (result["seconds"], result["peak_allocated_bytes"], result["peak_reserved_bytes"]) == (0.25, 200, 300)


def write_loading(model, loading):
    return patch.object(backend.AutoModelForSequenceClassification, "from_pretrained", return_value=(model, loading))


@pytest.mark.parametrize("field", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_every_nonempty_loading_diagnostic_is_rejected_with_full_details(tiny_classifier, field):
    details = ["classifier.café." + "weight_" * 60, "classifier.bias"]
    loading = {name: set() for name in ("missing_keys", "unexpected_keys", "mismatched_keys")}
    loading["error_msgs"] = []
    loading[field] = list(details) if field == "error_msgs" else set(details)
    model = Mock()
    with write_loading(model, loading), pytest.raises(RuntimeError) as raised:
        SequenceClassifier(tiny_classifier, device="cpu", max_input_tokens=64).load()
    message = str(raised.value)
    assert "Checkpoint did not load exactly" in message
    reported = json.loads(message.split("unused weights: ", 1)[1])
    assert reported[field] == (details if field == "error_msgs" else sorted(details))
    model.to.assert_not_called()
    model.eval.assert_not_called()


def test_requested_limit_above_the_model_limit_fails_before_weights_load(tiny_classifier):
    with patch.object(backend.AutoModelForSequenceClassification, "from_pretrained") as loader, \
            pytest.raises(ValueError, match="exceeds the model limit 64"):
        SequenceClassifier(tiny_classifier, device="cpu", max_input_tokens=65).load()
    loader.assert_not_called()


def test_real_tiny_checkpoint_loads_locally_in_fp32_and_classifies(tiny_classifier):
    model = SequenceClassifier(tiny_classifier, device="cpu", max_input_tokens=64)
    with patch.object(backend.AutoModelForSequenceClassification, "from_pretrained",
                      wraps=backend.AutoModelForSequenceClassification.from_pretrained) as loader:
        metadata = model.load()
    kwargs = loader.call_args.kwargs
    assert kwargs["local_files_only"] and not kwargs["trust_remote_code"] and kwargs["dtype"] == torch.float32
    assert metadata["model_class"] == "DistilBertForSequenceClassification"
    assert (metadata["device"], metadata["context_limit"], metadata["model_context_limit"]) == ("cpu", 64, 64)
    assert metadata["cuda_matmul_allow_tf32"] is False and metadata["float32_matmul_precision"] == "highest"
    assert all(metadata["loading_info"][field] == [] for field in ("missing_keys", "unexpected_keys",
                                                                    "mismatched_keys", "error_msgs"))
    assert metadata["id2label"]["8"] == "User Access, Edit and Deletion"
    json.dumps(metadata)
    encoded = model.encode("We retain café data.")
    assert encoded["input_ids"] == [2, 4, 5, 6, 7, 8, 3]
    result = model.classify(encoded)
    assert len(result["logits"]) == 10 and sum(result["probabilities"]) == pytest.approx(1, abs=1e-6)
    assert result == {**model.classify(encoded), "seconds": result["seconds"]}
