import pytest
from fastapi.testclient import TestClient

from model_service.app import create_app
from model_service.classifier import Classification
from model_service.settings import Settings
from model_service.taxonomy import LABELS, TAXONOMY


class FakeClassifier:
    def __init__(self):
        self._metadata = None
        self.calls = []

    def load(self):
        if self._metadata is None:
            self._metadata = {
                "id": "fake/model",
                "revision": "0" * 40,
                "resolved_revision": "0" * 40,
                "precision": "fp32",
                "device": "cpu",
                "strategy": "fake",
                "head": None,
                "fit": None,
                "taxonomy_id": TAXONOMY["taxonomy_id"],
                "taxonomy_version": TAXONOMY["version"],
            }
        return dict(self._metadata)

    def metadata(self):
        return dict(self._metadata) if self._metadata is not None else None

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
