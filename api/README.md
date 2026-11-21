# `api/` — frontend-facing API

A localhost FastAPI service that sits between the browser and `modelService/`. The
browser calls this API; this API calls the classifier and sends classifications and
summaries back in one stable contract.

```
frontend (browser)  --HTTP-->  api/  --HTTP-->  modelService/  --local files--> models
```

Policy text is held in memory only and expires after an hour. Nothing is written to
disk, logged, or sent off the machine (FR-22 off by default, NFR-01, NFR-02).

## What works now

| Route | Response | Status |
| --- | --- | --- |
| `GET /health` | `Health` | Implemented; reports model-service readiness |
| `GET /categories` | `CategoryList` | Implemented; proxies modelService taxonomy with `id` added |
| `POST /policies` | `201 Policy` | Implemented; pasted text |
| `POST /policies/upload` | `201 Policy` | Implemented; UTF-8 `.txt` only |
| `GET /policies/{id}` | `Policy` | Implemented |
| `POST /policies/{id}/classification` | `Classification` | Implemented; calls modelService `/analyze` |
| `POST /policies/{id}/summaries` | `202 Job` | Implemented; **extractive** summaries from classified sentences |
| `POST /policies/{id}/rewrites` | `202 Job` | Implemented; calls modelService `/rewrite` and returns source-linked rewrites |
| `GET /jobs/{id}` | `Job` | Implemented; poll for progress and result |
| `DELETE /jobs/{id}` | `204` | Implemented; cancels a queued or running job |
| `GET /policies/{id}/results` | `AnalysisResults` | Implemented; non-blocking snapshot |

### Ordering guarantee

**Classification is returned first, on its own. The classification request never
starts a generation job and never waits for one.** After classification,
`GET /policies/{id}/results` returns `phase: "classified"` with empty `summaries` and
`rewrites`. After a summary job succeeds, the same route returns
`phase: "results_ready"` with summaries attached. `source_sentence_ids` on every
block points back to the original sentence spans.

### Summaries and rewrites

- **Summaries** are extractive: up to two already-classified sentences are quoted
  under each category, with their source IDs. This needs no model download and cannot
  change what the policy says. It is a stand-in until a generative summarizer is
  chosen.
- **Rewrites** call the pinned GGUF chat model already installed with `modelService/`;
  the API never downloads a model. A job selects the classified sentences named in
  `sentence_ids` (or the whole policy), sends only their exact text to modelService,
  and stores the returned rewrite with those sentences' IDs as `source_sentence_ids`.
  modelService runs one prompt, the evaluated `summary-v3`, for every style; the API
  says so in each rewrite's `warnings` rather than pretending the styles were evaluated
  separately, and adds a warning when modelService marks the rewrite `truncated` or
  `reasoning`. If the GGUF file has not been downloaded, the job fails with
  `model_service_error` and modelService's message, which says to run `pbench download`.

## Setup and run

Use two terminals from the repository root. The classifier must be running before
the API answers classification requests.

```powershell
# Terminal 1 — modelService, as documented in ../modelService/README.md
cd modelService
uv run model-service

# Terminal 2 — this API
cd api
uv sync
uv run python -m api_service
```

The API listens on `http://127.0.0.1:8001`. Start the Vite frontend with
`cd frontend; npm run dev` and it will call the API automatically. Set
`VITE_API_BASE_URL` in a frontend `.env` file to override `http://127.0.0.1:8001`.
From the repository root, `python3 run.py` starts the model service, this API, and
the frontend together with the URLs wired correctly.

`GET /api/v1/health` reports `status: "ok"` only when the model service is ready.
If the model service is down, classification returns HTTP 503 with
`detail.code: "model_service_unavailable"`. A policy over 200,000 characters, pasted or
uploaded, gets HTTP 413 with `detail.code: "policy_too_long"`.

### Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `API_MODEL_SERVICE_URL` | `http://127.0.0.1:8000` | Loopback URL of modelService |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8001` | Loopback bind address and port |
| `API_CORS_ORIGINS` | Vite dev and preview origins | Comma-separated loopback origins |
| `API_REQUEST_TIMEOUT_SECONDS` | `120` | Timeout for modelService calls |
| `API_POLICY_TTL_SECONDS` | `3600` | How long an in-memory policy and its jobs live |

`python -m api_service --host 127.0.0.1 --port 8001` overrides the host and port for
one run. Both must be loopback addresses.

## Tests

```powershell
cd api
uv run pytest
```

The tests use a fake modelService through `httpx.MockTransport`, so no model or GPU
is needed. They cover upload validation, the length limit, the
classification-first ordering, per-category summaries, job cancellation,
rewriter-unavailable errors, and modelService outages.

Frontend checks:

```powershell
cd frontend
npm run build
npm run lint
```

## Layout

```
api/
  README.md                 this document
  pyproject.toml             dependencies and test configuration
  src/api_service/
    app.py                   routes, error handling, and background job threads
    schemas.py               Pydantic request/response contract
    client.py                httpx adapter for modelService
    store.py                 in-memory policies and jobs with TTL
    generation.py            extractive summaries and rewriter-unavailable error
    errors.py                stable error codes for the frontend
    settings.py              loopback-only configuration
    __main__.py              `python -m api_service`
  tests/
    conftest.py              fake modelService transport and app fixture
    test_api.py              end-to-end contract tests
```

## Requirement mapping

- **FR-01, FR-04–FR-06:** sentence IDs, exact offsets, labels, counts, and taxonomy
  colors come from modelService; the UI highlights and filters without another call.
- **FR-02, FR-24:** UTF-8 `.txt` uploads only, with the same length limit as modelService.
- **FR-12, FR-16:** generated blocks carry `source_sentence_ids`.
- **FR-17, FR-18:** job progress, polling, and cancellation.
- **NFR-01, NFR-02:** loopback-only bind and CORS; no downloads, no text logging,
  no disk writes.
