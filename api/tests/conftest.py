import json
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from api_service.app import create_app
from api_service.client import ModelServiceClient
from api_service.settings import Settings

MODEL = {
    "id": "google/embeddinggemma-2",
    "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
    "precision": "fp32",
    "device": "cpu",
    "strategy": "embedding-head",
    "head": "linear-balanced-tuned",
    "taxonomy_id": "opp115-10-other",
    "taxonomy_version": 1,
}

REWRITE_MODEL = {
    **MODEL,
    "id": "unsloth/Qwen3.5-4B-GGUF",
    "revision": "e87f176479d0855a907a41277aca2f8ee7a09523",
    "precision": "q4_k_m",
    "strategy": "prompted",
    "prompt": "summary-v3",
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
        return httpx.Response(200, json={"status": "ok", "ready": True, "model": MODEL})
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
                "rewrites": [
                    {
                        "text": "We use your email for promotions.",
                        "source_sentence_ids": payload["source_sentence_ids"],
                        "warnings": ["Check conditions from the source that are missing here."],
                    }
                ],
                "model": REWRITE_MODEL,
            },
        )
    return httpx.Response(404, json={"detail": "not found"})


@pytest.fixture
def model_http() -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(model_service_handler),
        base_url="http://127.0.0.1:8002",
        trust_env=False,
    )


@pytest.fixture
def client(model_http: httpx.Client):
    app = create_app(Settings(max_characters=1000), client=ModelServiceClient(model_http))
    with TestClient(app) as test_client:
        yield test_client
