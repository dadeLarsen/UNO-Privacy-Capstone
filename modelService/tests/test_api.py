import time

import pytest
from fastapi.testclient import TestClient

from model_service.app import create_app
from model_service.classifier import Classification
from model_service.rewriter import Rewrite
from model_service.settings import Settings
from model_service.taxonomy import LABELS, TAXONOMY

DOES_NOT_FIT = ("Input has 5000 tokens; plus 1024 requested output tokens exceeds the 4096-token context. "
                "Input was not truncated.")


class FakeClassifier:
    def __init__(self, device="cpu"):
        self._device = device
        self._metadata = None
        self.loaded = False
        self.calls = []

    def load(self):
        if self._metadata is None:
            self._metadata = {
                "id": "fake/model",
                "revision": "0" * 40,
                "resolved_revision": "0" * 40,
                "precision": "fp32",
                "device": self._device,
                "strategy": "fake",
                "head": None,
                "fit": None,
                "taxonomy_id": TAXONOMY["taxonomy_id"],
                "taxonomy_version": TAXONOMY["version"],
            }
        self.loaded = True
        return dict(self._metadata)

    def metadata(self):
        return dict(self._metadata) if self._metadata is not None else None

    def release(self):
        self.loaded = False

    def classify(self, sentence):
        self.calls.append(sentence)
        label_id = 8 if "collect" in sentence.lower() else 1
        return Classification(label_id, LABELS[label_id], tuple([0.0] * len(LABELS)))


class FakeRewriter:
    def __init__(self):
        self._metadata = None
        self.loaded = False

    def load(self):
        self._metadata = {"id": "fake/rewriter", "device": "cpu", "prompt": "fake-prompt"}
        self.loaded = True
        return dict(self._metadata)

    def metadata(self):
        return dict(self._metadata) if self._metadata is not None else None

    def release(self):
        self.loaded = False

    def rewrite(self, text):
        if "endless" in text:
            raise ValueError(DOES_NOT_FIT)
        return Rewrite(text="In short: " + text.split(".")[0] + ".", status="valid", output_tokens=7)


@pytest.fixture
def classifier():
    return FakeClassifier()


@pytest.fixture
def rewriter():
    return FakeRewriter()


@pytest.fixture
def client(classifier, rewriter):
    with TestClient(create_app(Settings(), classifier, rewriter)) as test_client:
        yield test_client


def test_health_reports_the_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["status"] == "ok"
    assert body["model"]["id"] == "fake/model"


def test_categories_match_the_model_label_order(client):
    response = client.get("/categories")
    assert response.status_code == 200
    body = response.json()
    assert [category["name"] for category in body["categories"]] == list(LABELS)
    assert body["categories"][4]["display_name"] == "User Access, Edit and Deletion"


def test_analyze_returns_sentence_offsets_and_labels(client):
    text = "We collect your email. We share it with partners."
    response = client.post("/analyze", json={"text": text})
    assert response.status_code == 200
    body = response.json()
    assert body["characters"] == len(text)
    assert len(body["sentences"]) == 2
    first, second = body["sentences"]
    assert text[first["start"]:first["end"]] == first["text"]
    assert first["label_id"] == 8
    assert first["label"] == "First Party Collection/Use"
    assert second["label_id"] == 1
    assert second["label"] == "Third Party Sharing/Collection"


def test_scores_are_optional(client):
    response = client.post("/analyze", json={"text": "We collect data."})
    assert "scores" not in response.json()["sentences"][0]

    response = client.post("/analyze", json={"text": "We collect data.", "include_scores": True})
    assert len(response.json()["sentences"][0]["scores"]) == len(LABELS)


class RefusesOneSentence(FakeClassifier):
    def classify(self, sentence):
        if "endless" in sentence:
            raise ValueError("Input has 9000 tokens; exceeds the 8192-token limit.")
        return super().classify(sentence)


def test_a_sentence_the_model_refuses_does_not_fail_the_policy():
    text = "We collect your email. This endless sentence is refused. We share it with partners."
    with TestClient(create_app(Settings(), RefusesOneSentence(), FakeRewriter())) as test_client:
        response = test_client.post("/analyze", json={"text": text, "include_scores": True})
    assert response.status_code == 200
    first, refused, last = response.json()["sentences"]
    assert refused["text"] == "This endless sentence is refused."
    assert (refused["label_id"], refused["label"], refused["scores"]) == (None, None, None)
    assert refused["error"] == "Input has 9000 tokens; exceeds the 8192-token limit."
    for sentence, label_id in ((first, 8), (last, 1)):
        assert sentence["label_id"] == label_id
        assert len(sentence["scores"]) == len(LABELS)
        assert "error" not in sentence


def test_empty_policy_is_rejected(client):
    assert client.post("/analyze", json={"text": ""}).status_code == 422
    assert client.post("/analyze", json={"text": "   \n\n "}).status_code == 422


def test_too_long_policy_is_rejected(classifier, rewriter):
    settings = Settings(max_characters=10)
    with TestClient(create_app(settings, classifier, rewriter)) as test_client:
        response = test_client.post("/analyze", json={"text": "This policy is definitely too long."})
        assert response.status_code == 413
        assert test_client.post("/rewrite", json={"text": "This passage is too long as well."}).status_code == 413


def test_upload_txt_file(client):
    response = client.post(
        "/analyze/upload",
        files={"file": ("policy.txt", b"We collect data. We share it.", "text/plain")},
    )
    assert response.status_code == 200
    assert len(response.json()["sentences"]) == 2


def test_upload_rejects_non_text_files(client):
    response = client.post(
        "/analyze/upload",
        files={"file": ("policy.pdf", b"%PDF-1.7", "application/pdf")},
    )
    assert response.status_code == 415


def test_upload_rejects_invalid_utf8(client):
    response = client.post(
        "/analyze/upload",
        files={"file": ("policy.txt", b"\xff\xfe\xfa", "text/plain")},
    )
    assert response.status_code == 400


def test_lazy_mode_reports_not_ready_until_the_first_request(classifier, rewriter):
    settings = Settings(preload=False)
    with TestClient(create_app(settings, classifier, rewriter)) as test_client:
        assert test_client.get("/health").json()["ready"] is False
        assert test_client.post("/analyze", json={"text": "We collect data."}).status_code == 200
        assert test_client.get("/health").json()["ready"] is True


def test_rewrite_returns_the_text_and_the_model_that_wrote_it(client):
    text = "We collect your email. We share it with partners."
    response = client.post("/rewrite", json={"text": text})
    assert response.status_code == 200
    body = response.json()
    assert body["rewrite"] == "In short: We collect your email."
    assert (body["status"], body["output_tokens"], body["characters"]) == ("valid", 7, len(text))
    assert body["model"] == {"id": "fake/rewriter", "device": "cpu", "prompt": "fake-prompt"}


def test_a_passage_that_does_not_fit_the_context_is_refused_with_the_reason(client):
    response = client.post("/rewrite", json={"text": "This endless passage does not fit."})
    assert response.status_code == 413
    assert response.json()["detail"] == DOES_NOT_FIT


def test_an_empty_passage_is_rejected(client):
    assert client.post("/rewrite", json={"text": ""}).status_code == 422
    assert client.post("/rewrite", json={"text": "   \n\n "}).status_code == 422


def test_only_the_model_of_the_last_request_stays_loaded(client, classifier, rewriter):
    # The classifier loads at startup, and the rewriter on its first request.
    assert (classifier.loaded, rewriter.loaded) == (True, False)
    assert client.get("/health").json()["rewriter"] is None

    assert client.post("/rewrite", json={"text": "We collect your email."}).status_code == 200
    assert (classifier.loaded, rewriter.loaded) == (False, True)
    health = client.get("/health").json()
    # A released classifier still answers, so /health must not tell a caller to wait.
    assert (health["ready"], health["rewriter"]["id"]) == (True, "fake/rewriter")

    assert client.post("/analyze", json={"text": "We collect data."}).status_code == 200
    assert (classifier.loaded, rewriter.loaded) == (True, False)


def test_an_idle_model_is_released_and_the_next_request_loads_it_again(rewriter):
    classifier = FakeClassifier()
    with TestClient(create_app(Settings(idle_release_seconds=1), classifier, rewriter)) as test_client:
        assert classifier.loaded
        deadline = time.monotonic() + 10
        while classifier.loaded and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not classifier.loaded
        # The service still answers, so /health must not tell a caller to wait.
        health = test_client.get("/health").json()
        assert (health["status"], health["ready"]) == ("ok", True)
        assert test_client.post("/analyze", json={"text": "We collect data."}).status_code == 200
        assert classifier.loaded
