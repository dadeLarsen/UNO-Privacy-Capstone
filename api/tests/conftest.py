import json
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from api_service.app import create_app
from api_service.client import ModelServiceClient
from api_service.settings import Settings

# The fakes return exactly the keys of modelService's classifier and rewriter metadata.
MODEL = {
    "id": "ggml-org/embeddinggemma-2-GGUF",
    "revision": "6ea3e3a309d12a4779cfda98b8c19800d2d5da71",
    "resolved_revision": "6ea3e3a309d12a4779cfda98b8c19800d2d5da71",
    "precision": "q8_0",
    "device": "cpu",
    "strategy": "embedding-head",
    "head": "linear-balanced-tuned",
    "fit": "classification/embeddinggemma2-appsi139-prior-shift/20261012-043915Z",
    "taxonomy_id": "opp115-10-other",
    "taxonomy_version": 1,
}

REWRITE_MODEL = {
    "id": "unsloth/Qwen3.5-4B-GGUF",
    "revision": "e87f176479d0855a907a41277aca2f8ee7a09523",
    "resolved_revision": "e87f176479d0855a907a41277aca2f8ee7a09523",
    "precision": "q4_k_m",
    "device": "cpu",
    "strategy": "prompted",
    "prompt": "summary-v3",
    "max_new_tokens": 1024,
}

NAMES = {
    0: "Policy Change",
    1: "Third Party Sharing/Collection",
    2: "User Choice/Control",
    3: "Do Not Track",
    4: "International and Specific Audiences",
    5: "Data Security",
    6: "Data Retention",
    7: "User Access, Edit and Deletion",
    8: "First Party Collection/Use",
    9: "Other",
}

CATEGORIES = {
    "taxonomy_id": "opp115-10-other",
    "version": 1,
    "categories": [
        {
            "id": label_id,
            "name": name,
            "display_name": name,
            "definition": f"Definition of {name}.",
            "color": "#123456",
        }
        for label_id, name in NAMES.items()
    ],
}


def model_service_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/health":
        return httpx.Response(200, json={"status": "ok", "ready": True, "model": MODEL, "rewriter": None})
    if path == "/categories":
        return httpx.Response(200, json=CATEGORIES)
    if path == "/analyze":
        payload = json.loads(request.content)
        text = payload["text"]
        sentences = []
        for match in re.finditer(r"[^.!?]+[.!?]*", text):
            span = match.group().strip()
            if not span:
                continue
            start = match.start() + match.group().index(span)
            label_id = 8 if "collect" in span.lower() else 1
            sentence = {
                "id": len(sentences),
                "text": span,
                "start": start,
                "end": start + len(span),
                "label_id": label_id,
                "label": NAMES[label_id],
            }
            if payload.get("include_scores"):
                sentence["scores"] = [0.0] * 10
            sentences.append(sentence)
        return httpx.Response(
            200,
            json={
                "text_sha256": "0" * 64,
                "characters": len(text),
                "sentences": sentences,
                "model": MODEL,
            },
        )
    if path == "/rewrite":
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "text_sha256": "1" * 64,
                "characters": len(payload["text"]),
                "rewrite": "We use your email for promotions.",
                "status": "valid",
                "output_tokens": 8,
                "model": REWRITE_MODEL,
            },
        )
    return httpx.Response(404, json={"detail": "not found"})


@pytest.fixture
def default_model_service_handler():
    return model_service_handler


@pytest.fixture
def model_client_factory():
    def create(handler=model_service_handler) -> ModelServiceClient:
        http = httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:8000",
            trust_env=False,
        )
        return ModelServiceClient(http)

    return create


@pytest.fixture
def client(model_client_factory):
    app = create_app(Settings(), client=model_client_factory())
    with TestClient(app) as test_client:
        yield test_client
