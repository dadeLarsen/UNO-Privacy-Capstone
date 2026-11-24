# `api/` — frontend-facing API

A localhost FastAPI service that sits between the browser and `modelService/`. The
browser calls this API; this API calls the classifier and sends classifications and
summaries back in one stable contract.

```
frontend (browser)  --HTTP-->  api/  --HTTP-->  modelService/  --local files--> models
```

Policy text is held in memory only and expires an hour after its last use. Nothing is
written to disk, logged, or sent off the machine (FR-22 off by default, NFR-01, NFR-02).

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
| `POST /policies/{id}/rewrites` | `202 Job` | Implemented; calls modelService `/rewrite` once for each section and returns source-linked rewrites |
| `GET /jobs/{id}` | `Job` | Implemented; poll for progress and result, and for a rewrite's finished sections while it runs |
| `DELETE /jobs/{id}` | `204` | Implemented; cancels a queued or running job, which keeps its finished sections |
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
  `sentence_ids` and `context_sentence_ids` (or the whole policy), sends only their
  exact text to modelService, and stores each returned rewrite with its sentences' IDs
  as `source_sentence_ids`. modelService runs one prompt, the evaluated `summary-v3`,
  for every style; the API says so in each rewrite's `warnings` rather than pretending
  the styles were evaluated separately, and adds a warning when modelService marks the
  rewrite `truncated` or `reasoning`. If the GGUF file has not been downloaded, the job
  fails with `model_service_error` and modelService's message, which says to run
  `pbench download`.

### Sections, progress, and cancel

The rewriter takes 4,096 tokens with its prompt and its output, which leaves about
14,000 characters for the passage (three real policies had 4.9 to 5.3 characters for a
token), and the rewrites that were judged came from passages of at most 2,154
characters. So a rewrite job cuts its sentences, in policy order, into sections of
about 2,000 characters (`SECTION_CHARACTERS` in `generation.py`) and sends one
`/rewrite` request for each. The result holds one rewrite for each section, each with
the IDs of its own sentences (FR-13).

The job names the cut from its first answer on: `sections` lists every section as the
IDs of its sentences, such as `[[0, 1, 2], [3], [4], [5, 6]]`, and `result.rewrites[i]`
is the rewrite of `sections[i]`. The page uses it to put each rewrite beside its
sentences before the first one is finished, and to show which sections a cancel or a
failure left without a rewrite (FR-12). A summary job has no `sections`. A request that
leaves nothing to rewrite gets HTTP 422 with `detail.code: "nothing_to_rewrite"` and no
job.

- A section ends at a paragraph break where it can: a newline between two sentences,
  or a place where the selection skips a sentence. A paragraph over the limit is cut
  between sentences. A sentence over the limit is a section of its own, and the job
  fails at it if modelService refuses its length.
- A line that does not end a sentence, such as a heading or the line that opens a
  list, stays with the paragraph after it.
- The API does not choose context for a selection. It rewrites the sentences that the
  request names.

While a job runs, `progress` is `{"stage": "rewriting", "completed": 2, "total": 24,
"message": "Rewriting section 3 of 24"}`, and `result` holds the sections finished so
far (FR-17). The first section of a job takes longer when modelService must load the
rewriter first; the API cannot see that load, so it has no stage of its own.

`DELETE /jobs/{id}` marks the job `cancelled` at once. The job keeps the sections that
were finished at that moment, in `result` and in `GET /policies/{id}/results`, and its
`progress` keeps their count (FR-18). The job sends no further section. modelService
still finishes the section it has under way, and the job drops that rewrite: nothing
stops a generation in the middle, so a new rewrite waits for it. A job that fails at a
section keeps the sections before it in the same way, with the reason in `error`. A
job that ends before its first section leaves the policy's earlier rewrite in place.

Measured through the page on the desktop (RTX 4090, Ryzen 9 7950X3D), one run each,
with GitHub's privacy statement as its web page gives it to paste (40,628 characters,
309 sentences): the job made 24 sections. On the GPU it took 21 s, 0.4 to 1.9 s for a
section. On the CPU it took 4 min 48 s, 5.9 to 17.7 s for a section, so one section
stays far inside the 120 s that the API waits for a modelService call. A cancel left
the page idle 0.2 to 0.3 s after the click.

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
`detail.code: "model_service_unavailable"`. The model service opens its port only once
its classifier has loaded, so the page takes that code as "still starting": it says
that it waits for the model service and asks again every 2 s. A policy over 200,000
characters, pasted or uploaded, gets HTTP 413 with `detail.code: "policy_too_long"`. A
request for a policy that has expired gets HTTP 404 with `detail.code: "not_found"`;
the page then sends the text again on its own.

The page classifies a policy as soon as it holds one, so every edit of the text box
that settles for 0.6 s creates a policy here. Each lives until its hour without use
is over; no route deletes one. Measured through the page on the desktop, two runs on
each device with a fresh model service: GitHub's privacy statement had its labels 4.1
and 4.3 s after the paste on the GPU, and 7.0 s after it on the CPU both times. The
classification request took 3.3 and 6.0 s of that, and the wait for the text to
settle 0.6 s.

### Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `API_MODEL_SERVICE_URL` | `http://127.0.0.1:8000` | Loopback URL of modelService |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8001` | Loopback bind address and port |
| `API_CORS_ORIGINS` | Vite dev and preview origins | Comma-separated loopback origins |
| `API_REQUEST_TIMEOUT_SECONDS` | `120` | Timeout for modelService calls |
| `API_POLICY_TTL_SECONDS` | `3600` | How long an in-memory policy and its jobs live after the policy's last use; polling a job counts as use |

`python -m api_service --host 127.0.0.1 --port 8001` overrides the host and port for
one run. Both must be loopback addresses.

## Tests

```powershell
cd api
uv run pytest
```

The tests use a fake modelService through `httpx.MockTransport`, so no model or GPU
is needed. They cover upload validation, the length limit, expiry, the
classification-first ordering, per-category summaries, the cut into sections and the
job's list of them, progress, a cancel and a failure partway through a rewrite,
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
    generation.py            extractive summaries, the cut into sections, and one section's rewrite
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
- **FR-12, FR-16:** generated blocks carry `source_sentence_ids`, and a rewrite job
  lists every section's sentences from its start.
- **FR-13:** a rewrite covers the selection or the whole policy, one section at a time.
  Choosing context for a selection is not built.
- **FR-17, FR-18:** job progress by section, polling, and a cancel between sections
  that keeps the finished ones.
- **NFR-01, NFR-02:** loopback-only bind and CORS; no downloads, no text logging,
  no disk writes.
