import pytest

from model_service.rewriter import PrivacyBenchRewriter
from model_service.settings import DEFAULT_CONDITIONS, DEFAULT_REWRITE_CONDITIONS


class FakeStrategy:
    """Answers as the benchmark's prompted rewrite strategy does."""

    name = "prompted"
    prompt = {"name": "fake-prompt"}
    loads = 0

    def load(self):
        self.loads += 1
        return {"model": "fake/rewriter", "revision": "0" * 40, "resolved_revision": "0" * 40,
                "precision": "q4_k_m", "device": "cpu", "max_new_tokens": 1024}

    def prepare(self, passage):
        if "endless" in passage.text:
            raise ValueError("Input has 5000 tokens; exceeds the 4096-token context.")
        return passage.text

    def predict(self, prepared):
        return {"raw_output": f"  Short: {prepared}\n", "output_tokens": 9,
                "reached_output_token_limit": "cut" in prepared}


@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_default_conditions_build_without_loading_the_model(device):
    rewriter = PrivacyBenchRewriter(DEFAULT_REWRITE_CONDITIONS[device])
    assert (rewriter.condition.task, rewriter.condition.strategy) == ("rewrite", "prompted")
    assert rewriter.condition.model["device"] == device
    assert rewriter.metadata() is None


def test_both_default_conditions_run_the_same_file_and_prompt():
    gpu, cpu = (PrivacyBenchRewriter(DEFAULT_REWRITE_CONDITIONS[device]).condition for device in ("gpu", "cpu"))
    assert gpu.options == cpu.options
    assert {**gpu.model, "device": "cpu"} == cpu.model


def test_a_classification_condition_is_refused():
    with pytest.raises(ValueError, match="needs a rewrite condition, not 'classification'"):
        PrivacyBenchRewriter(DEFAULT_CONDITIONS["cpu"])


def test_a_rewrite_is_the_benchmark_s_text_and_verdict_for_the_passage(monkeypatch):
    rewriter = PrivacyBenchRewriter(DEFAULT_REWRITE_CONDITIONS["cpu"])
    monkeypatch.setattr(rewriter, "_strategy", FakeStrategy())
    result = rewriter.rewrite("We keep data.")
    assert (result.text, result.status, result.output_tokens) == ("Short: We keep data.", "valid", 9)
    assert rewriter.rewrite("We cut data.").status == "truncated"
    assert rewriter.metadata() == {
        "id": "fake/rewriter", "revision": "0" * 40, "resolved_revision": "0" * 40, "precision": "q4_k_m",
        "device": "cpu", "strategy": "prompted", "prompt": "fake-prompt", "max_new_tokens": 1024,
    }
    # A passage that does not fit is refused, never truncated, and an empty one is not sent to the model.
    with pytest.raises(ValueError, match="exceeds the 4096-token context"):
        rewriter.rewrite("This endless passage.")
    with pytest.raises(ValueError, match="nonempty string"):
        rewriter.rewrite("   ")


def test_a_release_drops_the_loaded_strategy_keeps_the_metadata_and_the_next_use_loads_again(monkeypatch):
    rewriter = PrivacyBenchRewriter(DEFAULT_REWRITE_CONDITIONS["cpu"])
    loaded = FakeStrategy()
    monkeypatch.setattr(rewriter, "_strategy", loaded)
    metadata = rewriter.load()

    rewriter.release()
    # The strategy held the model. Its replacement is built from the condition and has loaded nothing.
    assert rewriter._strategy is not loaded
    assert rewriter._strategy.name == "prompted"
    assert rewriter.metadata() == metadata

    reloaded = FakeStrategy()
    monkeypatch.setattr(rewriter, "_strategy", reloaded)
    assert rewriter.rewrite("We keep data.").text == "Short: We keep data."
    assert (loaded.loads, reloaded.loads) == (1, 1)
