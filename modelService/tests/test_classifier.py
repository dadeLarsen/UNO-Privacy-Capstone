import pytest

from model_service.classifier import PrivacyBenchClassifier
from model_service.settings import DEFAULT_CONDITIONS

pytest.importorskip("privacy_bench")


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_default_conditions_build_without_loading_the_model(device):
    classifier = PrivacyBenchClassifier(DEFAULT_CONDITIONS[device])
    assert classifier.condition.strategy == "embedding-head"
    assert classifier.condition.model["device"] == device
    assert classifier.metadata() is None


def test_both_default_conditions_use_the_same_fitted_head():
    cuda, cpu = (PrivacyBenchClassifier(DEFAULT_CONDITIONS[device]).condition for device in ("cuda", "cpu"))
    assert cuda.options == cpu.options
