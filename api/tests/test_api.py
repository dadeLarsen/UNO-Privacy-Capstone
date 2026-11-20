import json
import time
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient

from api_service.app import create_app
from api_service.client import ModelServiceClient
from api_service.generation import JobCancelled
from api_service.settings import Settings

POLICY = "We collect your email address. We share it with marketing partners."


def new_policy(client: TestClient, text: str = POLICY) -> str:
    response = client.post("/api/v1/policies", json={"text": text})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def wait_for_job(client: TestClient, job_id: str) -> dict:
    for _ in range(200):
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] in ("succeeded", "failed", "cancelled"):
            return body
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_health_and_categories(client: TestClient) -> None:
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["model_service_ready"] is True

    categories = client.get("/api/v1/categories")
    assert categories.status_code == 200
    assert [item["id"] for item in categories.json()["categories"]] == list(range(10))


def test_upload_only_accepts_utf8_txt(client: TestClient) -> None:
    bad_type = client.post(
        "/api/v1/policies/upload",
        files={"file": ("policy.pdf", b"%PDF-1.7", "application/pdf")},
    )
    assert bad_type.status_code == 415
    assert bad_type.json()["detail"]["code"] == "unsupported_file_type"

    bad_encoding = client.post(
        "/api/v1/policies/upload",
        files={"file": ("policy.txt", b"\xff\xfe", "text/plain")},
    )
    assert bad_encoding.status_code == 400
    assert bad_encoding.json()["detail"]["code"] == "invalid_encoding"

    good = client.post(
        "/api/v1/policies/upload",
        files={"file": ("policy.txt", POLICY.encode("utf-8"), "text/plain")},
    )
    assert good.status_code == 201
    assert good.json()["source"] == "upload"
    assert good.json()["filename"] == "policy.txt"


def test_classification_is_returned_before_any_summary(client: TestClient) -> None:
    policy_id = new_policy(client)
    classified = client.post(f"/api/v1/policies/{policy_id}/classification", json={})
    assert classified.status_code == 200, classified.text
    body = classified.json()
    assert [item["label_id"] for item in body["sentences"]] == [8, 1]
    assert [count["count"] for count in body["counts"]].count(1) == 2

    snapshot = client.get(f"/api/v1/policies/{policy_id}/results")
    assert snapshot.status_code == 200
    assert snapshot.json()["phase"] == "classified"
    assert snapshot.json()["summaries"] == []
    assert snapshot.json()["rewrites"] == []


def test_summary_job_attaches_extractive_summaries(client: TestClient) -> None:
    policy_id = new_policy(client)
    assert client.post(f"/api/v1/policies/{policy_id}/classification").status_code == 200

    started = client.post(f"/api/v1/policies/{policy_id}/summaries", json={})
    assert started.status_code == 202, started.text
    job = wait_for_job(client, started.json()["id"])
    assert job["status"] == "succeeded", job
    assert job["result"]["kind"] == "summary"

    summaries = job["result"]["summaries"]
    assert len(summaries) == 2
    assert {item["label_id"] for item in summaries} == {8, 1}
    assert all(item["source_sentence_ids"] for item in summaries)
    assert "We collect your email address." in summaries[1]["text"]

    snapshot = client.get(f"/api/v1/policies/{policy_id}/results").json()
    assert snapshot["phase"] == "results_ready"
    assert len(snapshot["summaries"]) == 2


def test_summaries_require_classification_first(client: TestClient) -> None:
    policy_id = new_policy(client)
    response = client.post(f"/api/v1/policies/{policy_id}/summaries", json={})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "classification_required"


def rewrite_job(client: TestClient, body: dict) -> tuple[str, dict]:
    policy_id = new_policy(client)
    assert client.post(f"/api/v1/policies/{policy_id}/classification").status_code == 200
    started = client.post(f"/api/v1/policies/{policy_id}/rewrites", json=body)
    assert started.status_code == 202, started.text
    return policy_id, wait_for_job(client, started.json()["id"])


def with_rewrite(default_handler, **changes):
    """The fake modelService, with some fields of its /rewrite answer replaced."""

    def handler(request: httpx.Request) -> httpx.Response:
        response = default_handler(request)
        if request.url.path != "/rewrite":
            return response
        return httpx.Response(200, json={**response.json(), **changes})

    return handler


def test_rewrite_job_returns_source_linked_text_and_warnings(
    default_model_service_handler,
    model_client_factory,
) -> None:
    sent = []

    def recording(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rewrite":
            sent.append(json.loads(request.content))
        return default_model_service_handler(request)

    app = create_app(Settings(), client=model_client_factory(recording))
    with TestClient(app) as client:
        policy_id, job = rewrite_job(client, {"style": "simplify_for_kids", "sentence_ids": [1]})
        snapshot = client.get(f"/api/v1/policies/{policy_id}/results").json()

    assert sent == [{"text": "We share it with marketing partners."}]
    assert job["status"] == "succeeded", job
    result = job["result"]
    assert result["kind"] == "rewrite"
    assert result["style"] == "simplify_for_kids"
    assert result["model"]["prompt"] == "summary-v3"
    assert result["model"]["taxonomy_id"] is None
    assert result["rewrites"][0]["source_sentence_ids"] == [1]
    [warning] = result["rewrites"][0]["warnings"]
    assert "summary-v3" in warning and "simplify_for_kids" in warning
    assert snapshot["rewrites"][0]["text"] == "We use your email for promotions."


@pytest.mark.parametrize("status", ["truncated", "reasoning"])
def test_rewrite_status_adds_a_warning(
    status: str,
    default_model_service_handler,
    model_client_factory,
) -> None:
    handler = with_rewrite(default_model_service_handler, status=status)
    app = create_app(Settings(), client=model_client_factory(handler))
    with TestClient(app) as client:
        _, job = rewrite_job(client, {"style": "paraphrase"})
    assert job["status"] == "succeeded", job
    assert len(job["result"]["rewrites"][0]["warnings"]) == 2


@pytest.mark.parametrize("changes", [{"status": "empty", "rewrite": ""}, {"status": "missing"}])
def test_empty_or_unknown_rewrite_fails_the_job(
    changes: dict,
    default_model_service_handler,
    model_client_factory,
) -> None:
    handler = with_rewrite(default_model_service_handler, **changes)
    app = create_app(Settings(), client=model_client_factory(handler))
    with TestClient(app) as client:
        _, job = rewrite_job(client, {"style": "paraphrase"})
    assert job["status"] == "failed"
    assert job["error"]["code"] == "generation_failed"


def test_rewrite_requires_classification_first(client: TestClient) -> None:
    policy_id = new_policy(client)
    response = client.post(
        f"/api/v1/policies/{policy_id}/rewrites",
        json={"style": "paraphrase"},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "classification_required"


def test_unknown_sentence_ids_are_rejected(client: TestClient) -> None:
    policy_id = new_policy(client)
    assert client.post(f"/api/v1/policies/{policy_id}/classification").status_code == 200
    response = client.post(
        f"/api/v1/policies/{policy_id}/summaries",
        json={"sentence_ids": [99]},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unknown_sentence"


def test_job_can_be_cancelled(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def slow_summaries(*args: object, cancel: object, **kwargs: object):
        cancel.wait(2)  # type: ignore[attr-defined]
        raise JobCancelled

    monkeypatch.setattr("api_service.app.extractive_summaries", slow_summaries)
    policy_id = new_policy(client)
    assert client.post(f"/api/v1/policies/{policy_id}/classification").status_code == 200
    started = client.post(f"/api/v1/policies/{policy_id}/summaries", json={})
    job_id = started.json()["id"]

    running = client.get(f"/api/v1/jobs/{job_id}").json()
    assert running["status"] in ("queued", "running")
    assert client.delete(f"/api/v1/jobs/{job_id}").status_code == 204
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "cancelled"


def test_model_service_outage_returns_structured_error() -> None:
    def outage(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    http = httpx.Client(
        transport=httpx.MockTransport(outage),
        base_url="http://127.0.0.1:8002",
        trust_env=False,
    )
    app = create_app(Settings(), client=ModelServiceClient(http))
    with TestClient(app) as client:
        policy_id = new_policy(client)
        response = client.post(f"/api/v1/policies/{policy_id}/classification")
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["code"] == "model_service_unavailable"
        assert detail["retryable"] is True


def test_rewrite_job_reports_a_missing_rewriter_file(
    default_model_service_handler,
    model_client_factory,
) -> None:
    def missing_model(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rewrite":
            # modelService's answer when the rewriter's GGUF file is not in the local cache.
            return httpx.Response(
                503,
                json={
                    "detail": "Rewriter model unavailable: unsloth/Qwen3.5-4B-GGUF at "
                    "e87f176479d0855a907a41277aca2f8ee7a09523 is not in the local Hugging Face cache. "
                    "Run `pbench download` while online first."
                },
            )
        return default_model_service_handler(request)

    app = create_app(Settings(), client=model_client_factory(missing_model))
    with TestClient(app) as client:
        _, job = rewrite_job(client, {"style": "paraphrase"})
    assert job["status"] == "failed"
    assert job["error"]["code"] == "model_service_error"
    assert "pbench download" in job["error"]["message"]


def test_policy_ids_are_uuids(client: TestClient) -> None:
    policy_id = new_policy(client)
    assert UUID(policy_id)
