import pytest

from model_service.classifier import PrivacyBenchClassifier
from model_service.settings import DEFAULT_CONDITIONS

pytest.importorskip("privacy_bench")

# privacy-model's checkpoint lists its labels alphabetically: the benchmark label_id at each checkpoint index.
CHECKPOINT_ORDER = [3, 6, 7, 8, 5, 9, 0, 1, 4, 2]
LOGITS = [0.0, 1.0, 2.0, 30.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


class CheckpointOrderStrategy:
    """Answers as the classifier-head strategy does: checkpoint index 3 wins, which is benchmark label 8."""

    name = "classifier-head"

    def load(self):
        return {"label_mapping": [{"model_label_id": index, "label_id": label_id}
                                  for index, label_id in enumerate(CHECKPOINT_ORDER)]}

    def prepare(self, row):
        return row.text

    def predict(self, prepared):
        return {"raw_output": '{"label_id": 8}', "model_label_id": 3, "logits": LOGITS}


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_default_conditions_build_without_loading_the_model(device):
    classifier = PrivacyBenchClassifier(DEFAULT_CONDITIONS[device])
    assert classifier.condition.strategy == "embedding-head"
    assert classifier.condition.model["device"] == device
    assert classifier.metadata() is None


def test_both_default_conditions_use_the_same_fitted_head():
    cuda, cpu = (PrivacyBenchClassifier(DEFAULT_CONDITIONS[device]).condition for device in ("cuda", "cpu"))
    assert cuda.options == cpu.options


def test_a_checkpoint_with_its_own_label_order_gets_benchmark_labels_and_scores(monkeypatch):
    classifier = PrivacyBenchClassifier(DEFAULT_CONDITIONS["cpu"])
    monkeypatch.setattr(classifier, "_strategy", CheckpointOrderStrategy())
    result = classifier.classify("We collect your email address.")
    assert (result.label_id, result.label) == (8, "First Party Collection/Use")
    assert result.scores == tuple(LOGITS[CHECKPOINT_ORDER.index(label_id)] for label_id in range(10))
    assert result.scores.index(max(result.scores)) == 8
