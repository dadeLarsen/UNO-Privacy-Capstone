import pytest

from model_service.classifier import PrivacyBenchClassifier
from model_service.settings import DEFAULT_CONDITION

pytest.importorskip("privacy_bench")


def test_default_condition_builds_without_loading_the_model():
    classifier = PrivacyBenchClassifier(DEFAULT_CONDITION)
    assert classifier.condition.strategy == "embedding-head"
    assert classifier.metadata() is None


def test_cpu_override_rejects_bf16_condition():
    with pytest.raises(ValueError, match="run only on CUDA"):
        PrivacyBenchClassifier(DEFAULT_CONDITION, device="cpu")
