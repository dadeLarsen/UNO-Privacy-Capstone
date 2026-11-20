from threading import Event, Thread
import time

import pytest

from model_service.classifier import PrivacyBenchClassifier
from model_service.settings import DEFAULT_CONDITIONS

SCORES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 30.0, 9.0]


class FakeStrategy:
    """Answers as the embedding-head strategy does: one score per category, and category 8 wins."""

    name = "embedding-head"
    loads = 0

    def load(self):
        self.loads += 1
        return {"model": "o/m-GGUF", "head": "linear"}

    def prepare(self, row):
        return row.text

    def predict(self, prepared):
        return {"raw_output": '{"label_id": 8}', "model_label_id": 8, "scores": SCORES}


class SlowStrategy(FakeStrategy):
    """Embeds until the test lets it finish."""

    def __init__(self):
        self.embedding, self.finish = Event(), Event()

    def predict(self, prepared):
        self.embedding.set()
        self.finish.wait(5)
        return super().predict(prepared)


@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_default_conditions_build_without_loading_the_model(device):
    classifier = PrivacyBenchClassifier(DEFAULT_CONDITIONS[device])
    assert classifier.condition.strategy == "embedding-head"
    assert classifier.condition.model["device"] == device
    assert classifier.metadata() is None


def test_both_default_conditions_run_the_same_file_and_fitted_head_at_the_service_limit():
    gpu, cpu = (PrivacyBenchClassifier(DEFAULT_CONDITIONS[device]).condition for device in ("gpu", "cpu"))
    assert gpu.options == cpu.options
    assert gpu.options["max_input_tokens"] == 2048
    assert {**gpu.model, "device": "cpu"} == cpu.model
    assert cpu.model["file"] == "embeddinggemma-2-Q8_0.gguf"


def test_a_condition_that_names_no_gguf_file_is_refused(tmp_path):
    text = DEFAULT_CONDITIONS["cpu"].read_text(encoding="utf-8")
    condition = tmp_path / "no-file.toml"
    condition.write_text(text.replace('file = "embeddinggemma-2-Q8_0.gguf"\n', ""), encoding="utf-8")
    with pytest.raises(ValueError, match="names no GGUF file"):
        PrivacyBenchClassifier(condition)


def test_the_label_and_the_scores_are_the_strategy_s_answer(monkeypatch):
    classifier = PrivacyBenchClassifier(DEFAULT_CONDITIONS["cpu"])
    monkeypatch.setattr(classifier, "_strategy", FakeStrategy())
    result = classifier.classify("We collect your email address.")
    assert (result.label_id, result.label) == (8, "First Party Collection/Use")
    assert result.scores == tuple(SCORES)


def test_a_release_drops_the_loaded_strategy_keeps_the_metadata_and_the_next_use_loads_again(monkeypatch):
    classifier = PrivacyBenchClassifier(DEFAULT_CONDITIONS["cpu"], cache_size=0)
    loaded = FakeStrategy()
    monkeypatch.setattr(classifier, "_strategy", loaded)
    metadata = classifier.load()

    classifier.release()
    # The strategy held the model. Its replacement is built from the condition and has loaded nothing.
    assert classifier._strategy is not loaded
    assert classifier._strategy.name == "embedding-head"
    # /health reads this, and a released model must not look like one that is still loading.
    assert classifier.metadata() == metadata

    reloaded = FakeStrategy()
    monkeypatch.setattr(classifier, "_strategy", reloaded)
    assert classifier.classify("We collect your email address.").label_id == 8
    assert (loaded.loads, reloaded.loads) == (1, 1)


def test_the_metadata_does_not_wait_for_a_sentence_under_way(monkeypatch):
    classifier = PrivacyBenchClassifier(DEFAULT_CONDITIONS["cpu"], cache_size=0)
    strategy = SlowStrategy()
    monkeypatch.setattr(classifier, "_strategy", strategy)
    metadata = classifier.load()
    classifying = Thread(target=classifier.classify, args=("We collect your email address.",))
    classifying.start()
    assert strategy.embedding.wait(5)
    # /health reads the metadata, so it must answer while a request holds the model.
    started = time.monotonic()
    assert classifier.metadata() == metadata
    waited = time.monotonic() - started
    strategy.finish.set()
    classifying.join(5)
    assert waited < 1
