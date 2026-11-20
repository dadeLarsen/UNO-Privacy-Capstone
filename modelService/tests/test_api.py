import time

import pytest
from fastapi.testclient import TestClient

from model_service.app import create_app
from model_service.classifier import Classification
from model_service.rewriter import RewriterUnavailable, RewrittenPassage
from model_service.settings import Settings
from model_service.taxonomy import LABELS, TAXONOMY


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


@pytest.fixture
def classifier():
    return FakeClassifier()


@pytest.fixture
def client(classifier):
    with TestClient(create_app(Settings(), classifier)) as test_client:
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
    with TestClient(create_app(Settings(), RefusesOneSentence())) as test_client:
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


def test_too_long_policy_is_rejected(classifier):
    settings = Settings(max_characters=10)
    with TestClient(create_app(settings, classifier)) as test_client:
        response = test_client.post("/analyze", json={"text": "This policy is definitely too long."})
    assert response.status_code == 413


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


def test_lazy_mode_reports_not_ready_until_the_first_request(classifier):
    settings = Settings(preload=False)
    with TestClient(create_app(settings, classifier)) as test_client:
        assert test_client.get("/health").json()["ready"] is False
        assert test_client.post("/analyze", json={"text": "We collect data."}).status_code == 200
        assert test_client.get("/health").json()["ready"] is True


def test_an_idle_gpu_model_is_released_and_the_next_request_loads_it_again():
    classifier = FakeClassifier(device="cuda:0")
    with TestClient(create_app(Settings(idle_release_seconds=1), classifier)) as test_client:
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


class FakeRewriter:
    def load(self):
        return {
            "id": "fake/rewriter",
            "revision": "1" * 40,
            "precision": "q4_k_m",
            "device": "cpu",
            "strategy": "prompted",
            "prompt": "summary-v3",
            "taxonomy_id": TAXONOMY["taxonomy_id"],
            "taxonomy_version": TAXONOMY["version"],
        }

    def metadata(self):
        return self.load()

    def status(self):
        return {"status": "ready", "available": True, "ready": True, "model": self.load(), "setup": None}

    def rewrite(self, text, style):
        return RewrittenPassage("We use your email for promotions.", ("Check numbers from the source.",))


def test_rewrite_returns_source_linked_text_and_warnings(classifier):
    with TestClient(create_app(Settings(), classifier, FakeRewriter())) as test_client:
        response = test_client.post(
            "/rewrite",
            json={
                "text": "We may share your email address with partners.",
                "style": "simplify_for_adults",
                "source_sentence_ids": [3],
            },
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rewrites"][0]["source_sentence_ids"] == [3]
    assert body["rewrites"][0]["warnings"]
    assert body["model"]["prompt"] == "summary-v3"


class MissingRewriter(FakeRewriter):
    def load(self):
        raise RewriterUnavailable("Run `uv run pbench download ...` once.")


def test_missing_rewrite_model_returns_actionable_setup_error(classifier):
    with TestClient(create_app(Settings(), classifier, MissingRewriter())) as test_client:
        response = test_client.post(
            "/rewrite",
            json={"text": "We share your email.", "source_sentence_ids": [1]},
        )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "rewriter_model_not_downloaded"
    assert detail["retryable"] is False
    assert "pbench download" in detail["message"]
