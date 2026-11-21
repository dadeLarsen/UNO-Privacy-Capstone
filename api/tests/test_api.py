import json
import threading
import time
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient

from api_service.app import create_app
from api_service.generation import JobCancelled
from api_service.schemas import MAX_POLICY_CHARACTERS
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


def upload(client: TestClient, data: bytes) -> httpx.Response:
    return client.post("/api/v1/policies/upload", files={"file": ("policy.txt", data, "text/plain")})


def test_a_policy_over_the_character_limit_is_refused_with_the_reason(client: TestClient) -> None:
    at_limit = "a" * MAX_POLICY_CHARACTERS
    assert client.post("/api/v1/policies", json={"text": at_limit}).status_code == 201
    assert upload(client, at_limit.encode()).status_code == 201
    # The most bytes that 200,000 characters can take: four each, after a byte order mark.
    assert upload(client, b"\xef\xbb\xbf" + "\U0001f512".encode() * MAX_POLICY_CHARACTERS).status_code == 201

    over_limit = at_limit + "a"
    for response in (
        client.post("/api/v1/policies", json={"text": over_limit}),
        # Few enough bytes to pass the upload route's read limit, so only the character count refuses it.
        upload(client, over_limit.encode()),
        upload(client, b"a" * (4 * MAX_POLICY_CHARACTERS + 4)),
    ):
        assert response.status_code == 413, response.text
        assert response.json()["detail"]["code"] == "policy_too_long"
        assert "200,000 characters" in response.json()["detail"]["message"]


def test_a_policy_expires_an_hour_after_its_last_use_not_its_creation(client: TestClient) -> None:
    policy_id = new_policy(client)
    assert client.post(f"/api/v1/policies/{policy_id}/classification").status_code == 200
    job_id = client.post(f"/api/v1/policies/{policy_id}/summaries", json={}).json()["id"]
    record = client.app.state.store.get_policy(UUID(policy_id))

    # Each request comes 59 minutes after the one before it; polling a job counts as use of its policy.
    for path in (f"/api/v1/policies/{policy_id}", f"/api/v1/jobs/{job_id}", f"/api/v1/policies/{policy_id}/results"):
        record.last_used -= 59 * 60
        assert client.get(path).status_code == 200, path

    record.last_used -= 61 * 60
    expired = client.get(f"/api/v1/policies/{policy_id}")
    assert expired.status_code == 404
    assert expired.json()["detail"]["code"] == "not_found"
    assert client.get(f"/api/v1/jobs/{job_id}").status_code == 404


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
    assert job["sections"] is None

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


def rewrite_job(client: TestClient, body: dict, text: str = POLICY) -> tuple[str, dict]:
    policy_id = new_policy(client, text)
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


def recording_rewrites(default_handler, sent: list[str]):
    """The fake modelService, which also notes the passage of each /rewrite request."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rewrite":
            sent.append(json.loads(request.content)["text"])
        return default_handler(request)

    return handler


# With a limit of 30 characters: two short paragraphs that fit one section together, a paragraph over the
# limit, and a last paragraph that fits behind the long one's last sentence.
SECTIONED_POLICY = "Aa one. Aa two.\nBb one.\n\nCc one is long. Cc two is long. Cc three is long.\nDd one."


def test_a_long_rewrite_goes_to_the_rewriter_one_section_at_a_time(
    default_model_service_handler,
    model_client_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("api_service.generation.SECTION_CHARACTERS", 30)
    sent: list[str] = []
    app = create_app(Settings(), client=model_client_factory(recording_rewrites(default_model_service_handler, sent)))
    with TestClient(app) as client:
        policy_id, job = rewrite_job(client, {"style": "paraphrase"}, SECTIONED_POLICY)
        snapshot = client.get(f"/api/v1/policies/{policy_id}/results").json()

    # Paragraphs stay whole where they fit; the long one is cut between its sentences.
    assert sent == ["Aa one. Aa two. Bb one.", "Cc one is long.", "Cc two is long.", "Cc three is long. Dd one."]
    assert job["status"] == "succeeded", job
    assert job["progress"] == {"stage": "done", "completed": 4, "total": 4, "message": None}
    assert [item["source_sentence_ids"] for item in job["result"]["rewrites"]] == [[0, 1, 2], [3], [4], [5, 6]]
    assert len(snapshot["rewrites"]) == 4


def test_a_heading_stays_with_the_paragraph_after_it(
    default_model_service_handler,
    model_client_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("api_service.generation.SECTION_CHARACTERS", 30)
    sent: list[str] = []
    app = create_app(Settings(), client=model_client_factory(recording_rewrites(default_model_service_handler, sent)))
    with TestClient(app) as client:
        # The heading fits behind the first paragraph, but then it would end that section.
        rewrite_job(client, {"style": "paraphrase"}, "Aa one is here.\n\nBb heading\n\nBb one is here.")

    assert sent == ["Aa one is here.", "Bb heading Bb one is here."]


def test_a_selection_is_rewritten_in_policy_order_and_breaks_where_it_skips_a_sentence(
    default_model_service_handler,
    model_client_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("api_service.generation.SECTION_CHARACTERS", 30)
    sent: list[str] = []
    app = create_app(Settings(), client=model_client_factory(recording_rewrites(default_model_service_handler, sent)))
    with TestClient(app) as client:
        # The selection skips sentence 1, which makes a paragraph break there. Sentences 2 and 3 stay
        # together, where a cut by length alone would put 2 behind 0.
        body = {"style": "paraphrase", "sentence_ids": [3, 0], "context_sentence_ids": [2]}
        _, job = rewrite_job(client, body, "Aa one is here. Aa two is here. Aa three. Aa four.")

    assert sent == ["Aa one is here.", "Aa three. Aa four."]
    assert [item["source_sentence_ids"] for item in job["result"]["rewrites"]] == [[0], [2, 3]]
    assert job["sections"] == [[0], [2, 3]]


def test_a_rewrite_of_a_policy_without_sentences_is_refused(client: TestClient) -> None:
    # modelService refuses such a policy itself. The fake returns no sentence for it.
    policy_id = new_policy(client, " ")
    assert client.post(f"/api/v1/policies/{policy_id}/classification").json()["sentences"] == []
    response = client.post(f"/api/v1/policies/{policy_id}/rewrites", json={"style": "paraphrase"})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "nothing_to_rewrite"


def test_a_cancelled_rewrite_keeps_its_finished_sections_and_starts_no_new_one(
    default_model_service_handler,
    model_client_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("api_service.generation.SECTION_CHARACTERS", 30)
    sent: list[str] = []
    third_section_started = threading.Event()
    release = threading.Event()
    workers: list[threading.Thread] = []

    def start_worker(target, *args: object) -> None:
        workers.append(threading.Thread(target=target, args=args, daemon=True))
        workers[-1].start()

    monkeypatch.setattr("api_service.app._start_thread", start_worker)

    def stuck_in_the_third_section(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rewrite" and len(sent) == 2:
            sent.append(json.loads(request.content)["text"])
            third_section_started.set()
            assert release.wait(5)
            return default_model_service_handler(request)
        return recording_rewrites(default_model_service_handler, sent)(request)

    app = create_app(Settings(), client=model_client_factory(stuck_in_the_third_section))
    with TestClient(app) as client:
        policy_id = new_policy(client, SECTIONED_POLICY)
        assert client.post(f"/api/v1/policies/{policy_id}/classification").status_code == 200
        started = client.post(f"/api/v1/policies/{policy_id}/rewrites", json={"style": "paraphrase"})
        job_id = started.json()["id"]
        # The job names every section's sentences in its first answer, also the two it never rewrites.
        assert started.json()["sections"] == [[0, 1, 2], [3], [4], [5, 6]]
        assert third_section_started.wait(5)

        running = client.get(f"/api/v1/jobs/{job_id}").json()
        assert running["status"] == "running"
        assert running["progress"] == {
            "stage": "rewriting",
            "completed": 2,
            "total": 4,
            "message": "Rewriting section 3 of 4",
        }
        assert len(running["result"]["rewrites"]) == 2

        assert client.delete(f"/api/v1/jobs/{job_id}").status_code == 204
        cancelled = client.get(f"/api/v1/jobs/{job_id}").json()
        assert cancelled["status"] == "cancelled"
        assert cancelled["progress"] == {"stage": "cancelled", "completed": 2, "total": 4, "message": None}
        assert [item["source_sentence_ids"] for item in cancelled["result"]["rewrites"]] == [[0, 1, 2], [3]]
        assert cancelled["sections"] == [[0, 1, 2], [3], [4], [5, 6]]
        snapshot = client.get(f"/api/v1/policies/{policy_id}/results").json()
        assert snapshot["phase"] == "results_ready"
        assert [item["id"] for item in snapshot["rewrites"]] == [item["id"] for item in cancelled["result"]["rewrites"]]

        # modelService finishes the section it has. The job drops that rewrite and sends no fourth section.
        release.set()
        workers[0].join(5)
        assert not workers[0].is_alive()
        assert len(sent) == 3
        assert client.get(f"/api/v1/jobs/{job_id}").json() == cancelled


def test_a_rewrite_that_fails_at_a_section_keeps_the_sections_before_it(
    default_model_service_handler,
    model_client_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("api_service.generation.SECTION_CHARACTERS", 30)
    sent: list[str] = []

    def refuses_the_third_section(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rewrite" and len(sent) == 2:
            return httpx.Response(413, json={"detail": "Input has 5000 tokens; it exceeds the 4096-token context."})
        return recording_rewrites(default_model_service_handler, sent)(request)

    app = create_app(Settings(), client=model_client_factory(refuses_the_third_section))
    with TestClient(app) as client:
        policy_id, job = rewrite_job(client, {"style": "paraphrase"}, SECTIONED_POLICY)
        snapshot = client.get(f"/api/v1/policies/{policy_id}/results").json()

    assert job["status"] == "failed"
    assert job["error"]["code"] == "model_service_rejected"
    assert "5000 tokens" in job["error"]["message"]
    assert job["progress"] == {"stage": "model_service_error", "completed": 2, "total": 4, "message": None}
    assert [item["source_sentence_ids"] for item in job["result"]["rewrites"]] == [[0, 1, 2], [3]]
    assert len(snapshot["rewrites"]) == 2


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


def test_model_service_outage_returns_structured_error(model_client_factory) -> None:
    def outage(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    app = create_app(Settings(), client=model_client_factory(outage))
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
