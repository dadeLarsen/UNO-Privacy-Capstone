# modelService

A localhost-only Python service that runs the project's chosen classifier on a privacy
policy and returns one category per sentence. It is the first piece of the app's
backend; the browser UI is not wired to it yet, and rewriting is not implemented
because no rewriter has been chosen (AR-02).

The service reuses `privacy-research/` rather than loading a model itself. A
**condition** file pins the checkpoint, the fitted head, the instruction prefix, and
the category order, and the benchmark's strategy checks the fitted weight hashes
before inference. By default the service runs the prior-shifted EmbeddingGemma 2
linear head, the classifier that met NFR-05 on both APPSI-139 samples. It has one
condition for that head on an NVIDIA GPU (bf16) and one on the CPU (fp32), and picks
between them at startup, so every computer gets the same head.

## What it does

| Route | Input | Output |
| --- | --- | --- |
| `GET /health` | — | Whether the model is loaded, plus its pinned identity |
| `GET /categories` | — | The ten categories, definitions, display names, and colors |
| `POST /analyze` | JSON `{"text": "...", "include_scores": false}` | One label per sentence, with exact character offsets |
| `POST /analyze/upload` | multipart `.txt` file, optional `include_scores` query | Same response as `/analyze` |

Each sentence in the response has `id`, `text`, `start`, `end`, `label_id`, and
`label`. `start` and `end` are offsets into the policy exactly as submitted, so the
UI can highlight the original text without re-splitting it. `label_id` follows
`taxonomy.json` (the same order as `privacy_bench.tasks.classification.taxonomy`).

The service never logs policy text. It binds to a loopback address only, loads models
from the local Hugging Face cache, and never downloads at request time (NFR-01,
NFR-02).

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12. `modelService`
depends on `privacy-research` by path, so both folders must stay side by side.

First cache the pinned model files from the benchmark project while online. Both
default conditions pin the same checkpoint, so this one download serves the GPU and
the CPU:

```powershell
cd ../privacy-research
uv sync
uv run pbench download conditions/classification/appsi139/embeddinggemma2-bf16-head-linear-balanced-tuned-appsi139-test-half.toml
```

Some checkpoints may require accepting their license on Hugging Face first. The service
itself never downloads; it reads the snapshot from the local cache.

Then install the service:

```powershell
cd ../modelService
uv sync
uv run pytest
```

This install works with or without an NVIDIA GPU, because the CUDA build of torch
also runs on the CPU. That was checked with the GPU hidden
(`CUDA_VISIBLE_DEVICES=-1`), not yet on a computer without an NVIDIA driver.

**Optional, without an NVIDIA GPU:** the CPU build of torch is a much smaller
download.

```powershell
$env:UV_NO_SYNC = "1"    # keep `uv run` from installing the CUDA build
uv sync --no-group cu128 --group cpu
```

## Run

```powershell
uv run model-service
# or: uv run python -m model_service
```

The service listens on `http://127.0.0.1:8000`. At startup it runs the GPU condition
when torch finds an NVIDIA GPU with native bf16 (the RTX 30 series and newer), and
the CPU condition on every other computer. `/health` reports the device in use.

```powershell
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/analyze `
  -H "Content-Type: application/json" `
  -d '{"text":"We collect your email address. We share it with marketing partners."}'
curl -X POST "http://127.0.0.1:8000/analyze/upload?include_scores=true" `
  -F "file=@policy.txt"
```

Example response (abridged):

```json
{
  "text_sha256": "…",
  "characters": 72,
  "sentences": [
    {
      "id": 0,
      "text": "We collect your email address.",
      "start": 0,
      "end": 31,
      "label_id": 8,
      "label": "First Party Collection/Use"
    }
  ],
  "model": {
    "id": "google/embeddinggemma-2",
    "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
    "precision": "bf16",
    "device": "cuda:0",
    "strategy": "embedding-head",
    "head": "linear-balanced-tuned",
    "taxonomy_id": "opp115-10-other",
    "taxonomy_version": 1
  }
}
```

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_SERVICE_CONDITION` | unset | Condition TOML to run instead of a default condition. It names its own device, so it cannot be combined with a device |
| `MODEL_SERVICE_DEVICE` | the detected hardware | `cuda` or `cpu`: which of the two default conditions to run. Cannot be combined with a condition |
| `MODEL_SERVICE_HOST` | `127.0.0.1` | Must be a loopback address |
| `MODEL_SERVICE_PORT` | `8000` | Listening port |
| `MODEL_SERVICE_MAX_CHARACTERS` | `200000` | Maximum policy length |
| `MODEL_SERVICE_CACHE_SIZE` | `4096` | Sentences kept in the in-memory cache |
| `MODEL_SERVICE_PRELOAD` | `true` | Load the model at startup |
| `MODEL_SERVICE_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173` | Allowed browser origins |

Command-line flags `--condition`, `--device`, `--host`, `--port`, and `--no-preload`
override the environment for one run. The service refuses to start with both a
condition and a device, whether each comes from a flag or a variable.

## Layout

```
src/model_service/
  app.py         FastAPI routes, upload handling, and response shape
  classifier.py  privacy_bench condition/strategy adapter and sentence cache
  sentences.py   offset-preserving sentence splitter
  settings.py    environment and CLI configuration, and the GPU or CPU choice
  taxonomy.json  the ten categories, definitions, display names, and colors
tests/
  test_api.py        routes with a fake classifier (no model or GPU needed)
  test_classifier.py default condition wiring (skipped without privacy_bench)
  test_sentences.py  splitter behavior and offsets
  test_settings.py   the GPU or CPU choice, and the condition and device rule
  test_taxonomy.py   category order, checked against the benchmark
```

## Not implemented yet

- **Rewrites.** No rewriter has been evaluated (AR-02), so there is no `/rewrite`
  route. Adding one should follow the same pattern: a pinned condition, a strategy,
  and a response that links each block to its source sentence IDs.
- **Model choice by memory tier.** The service chooses between the GPU and the CPU
  at startup and runs the same head on both. Choosing a model by memory tier is
  FR-19. Only NVIDIA GPUs are used; a computer with an AMD, Intel, or Apple GPU runs
  on the CPU.
- **Idle GPU release.** The classifier stays loaded between requests. Releasing GPU
  memory when the service is idle (NFR-03) is not implemented.
- **History and caching across restarts.** The service keeps an in-memory cache only.
  Nothing is written to disk.
- **Browser wiring.** `frontend/` still uses example data and does not call this
  service.
