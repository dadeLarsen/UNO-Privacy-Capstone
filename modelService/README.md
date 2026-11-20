# modelService

A localhost-only Python service that runs the project's pinned models. It classifies
a privacy policy into one category per sentence and rewrites selected passages with
the pinned GGUF chat model. The browser reaches it through `api/`. All three UI styles
currently use the evaluated `summary-v3` prompt; style-specific prompts are not yet
evaluated.

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
| `GET /health` | — | Whether the service is ready to analyze, plus the model's pinned identity |
| `GET /categories` | — | The ten categories, definitions, display names, and colors |
| `POST /analyze` | JSON `{"text": "...", "include_scores": false}` | One label per sentence, with exact character offsets |
| `POST /analyze/upload` | multipart `.txt` file, optional `include_scores` query | Same response as `/analyze` |
| `POST /rewrite` | JSON `{"text": "...", "style": "...", "source_sentence_ids": [...]}` | A rewritten passage, review warnings, and the pinned rewriter's identity |

Each sentence in the response has `id`, `text`, `start`, `end`, `label_id`, and
`label`. `start` and `end` are offsets into the policy exactly as submitted, so the
UI can highlight the original text without re-splitting it. `label_id` follows
`privacy_bench.tasks.classification.taxonomy`, the category order the model was
fitted on.

A sentence the model cannot classify, such as one over the model's token limit, does
not fail the request. It comes back with `label_id` and `label` set to `null` and an
`error` that gives the reason, and every other sentence keeps its label. With
`include_scores`, its `scores` is `null` too.

The service never logs policy text. It binds to a loopback address only, loads models
from the local Hugging Face cache, and never downloads at request time (NFR-01,
NFR-02).

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12. `modelService`
depends on `privacy-research` by path, so both folders must stay side by side. Run
every command below from this folder. Installing the service also installs the
benchmark and its `pbench` command, so `privacy-research/` needs no setup of its own.

The default setup is CPU-only. It installs CPU builds of torch and `llama-cpp-python`
0.3.36, so it works on a computer with no NVIDIA driver and needs no C++ build tools
on Windows and Linux x64:

```powershell
uv sync
uv run pbench download ../privacy-research/conditions/classification/appsi139/embeddinggemma2-fp32-cpu-head-linear-balanced-tuned-appsi139-test-half.toml
uv run pytest
```

**With an NVIDIA GPU.** Opt into the CUDA 13.2 builds instead. This needs a newer
NVIDIA driver than the CUDA 12.8 build did: NVIDIA names 595.45.04 as the oldest for
Linux and gives no number for Windows, where it was checked on driver 610.88:

```powershell
$env:UV_NO_SYNC = "1"    # keep `uv run` from reinstalling the CPU default; set it in your profile
uv sync --no-group cpu --group cu132
uv run pbench download ../privacy-research/conditions/classification/appsi139/embeddinggemma2-bf16-head-linear-balanced-tuned-appsi139-test-half.toml
uv run pytest
```

Set `UV_NO_SYNC` before any other `uv` command. After pulling a change to `uv.lock`,
rerun the `uv sync` line.

Both setups install `llama-cpp-python` 0.3.36, which the rewriter uses through the
benchmark's GGUF chat backend. The service imports it only when a rewrite is actually
requested. The default `cpu` group uses a prebuilt CPU wheel on Windows and Linux x64;
it needs neither an NVIDIA driver nor CUDA libraries, even though the rewrite condition
places every layer on the CPU. The `cu132` group uses a prebuilt CUDA wheel, which does
need an NVIDIA driver, including for CPU-only placement. Other platforms still build
from source; on a Mac that needs the Xcode command line tools. The upstream 0.3.36 macOS
wheels currently fail zip validation, so their URLs are kept only as references in
`pyproject.toml`. The benchmark's README says more under Setup.

Some checkpoints may require accepting their license on Hugging Face first. The service
itself never downloads; it reads the snapshot from the local cache. Download the pinned
rewriter once while online:

```powershell
uv run pbench download ../privacy-research/conditions/rewrite/qwen3.5-4b-q4km-cpu-summary-v3.toml
```

GPU users can point that command at the `qwen3.5-4b-q4km-gpu-summary-v3.toml` condition
instead; both name the same GGUF file. A rewrite request without the cached file returns
HTTP 503 with this command in the error message.

## Run

```powershell
uv run model-service
# or: uv run python -m model_service
```

From the repository root, `python3 run.py` starts this service together with the API
and the frontend, then stops all three on Ctrl+C.

The service listens on `http://127.0.0.1:8002`. At startup it runs the GPU condition
when torch finds an NVIDIA GPU with native bf16 (the RTX 30 series and newer), and
the CPU condition on every other computer. `/health` reports the device in use.

On a GPU, the service frees the model's GPU memory after 300 seconds without an
analyze request (NFR-03); `MODEL_SERVICE_IDLE_RELEASE_SECONDS` changes the time. The
next request loads the model again before it answers, which took about 2 s on an
RTX 4090, and `/health` reports ready throughout. The process keeps its CUDA context
until it exits, so its GPU memory falls but not to zero. On the CPU the model stays
loaded.

These examples work in Windows PowerShell 5.1 and in PowerShell 7. They call
`curl.exe` because 5.1 makes `curl` an alias of `Invoke-WebRequest`, and they pipe the
JSON in because 5.1 strips the double quotes inside a program's arguments.

```powershell
curl.exe http://127.0.0.1:8002/health

'{"text":"We collect your email address. We share it with marketing partners."}' |
  curl.exe -X POST http://127.0.0.1:8002/analyze -H "Content-Type: application/json" -d "@-"

curl.exe -X POST "http://127.0.0.1:8002/analyze/upload?include_scores=true" -F "file=@policy.txt"
```

In cmd, the first and last commands work as written. Escape the JSON's quotes instead:

```bat
curl -X POST http://127.0.0.1:8002/analyze -H "Content-Type: application/json" -d "{\"text\":\"We collect your email address. We share it with marketing partners.\"}"
```

Example response (abridged):

```json
{
  "text_sha256": "…",
  "characters": 67,
  "sentences": [
    {
      "id": 0,
      "text": "We collect your email address.",
      "start": 0,
      "end": 30,
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
| `MODEL_SERVICE_CONDITION` | unset | Classification condition TOML to run instead of a default condition. It names its own device, so it cannot be combined with a device |
| `MODEL_SERVICE_REWRITE_CONDITION` | default for the selected device | GGUF rewrite condition TOML to use instead of the pinned Qwen3.5-4B `summary-v3` condition |
| `MODEL_SERVICE_DEVICE` | the detected hardware | `cuda` or `cpu`: which of the two default conditions to run. Cannot be combined with a condition |
| `MODEL_SERVICE_HOST` | `127.0.0.1` | Must be a loopback address |
| `MODEL_SERVICE_PORT` | `8002` | Listening port |
| `MODEL_SERVICE_MAX_CHARACTERS` | `200000` | Maximum policy length |
| `MODEL_SERVICE_CACHE_SIZE` | `4096` | Sentences kept in the in-memory cache |
| `MODEL_SERVICE_PRELOAD` | `true` | Load the model at startup |
| `MODEL_SERVICE_IDLE_RELEASE_SECONDS` | `300` | Seconds without an analyze request before the service frees the model's GPU memory. On the CPU the model stays loaded |
| `MODEL_SERVICE_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173` | Allowed browser origins |

Command-line flags `--condition`, `--rewrite-condition`, `--device`, `--host`, `--port`,
and `--no-preload` override the environment for one run. The service refuses to start with both a
condition and a device, whether each comes from a flag or a variable.

## Layout

```
src/model_service/
  app.py         FastAPI routes, upload handling, and response shape
  classifier.py  privacy_bench condition/strategy adapter and sentence cache
  idle.py        frees the model's GPU memory once the service has been idle
  sentences.py   offset-preserving sentence splitter
  settings.py    environment and CLI configuration, and the GPU or CPU choice
  taxonomy.py    the ten categories: names and order from privacy_bench, the rest from taxonomy.json
  taxonomy.json  each category's display name, definition, and color, keyed by its name
tests/
  test_api.py        routes with a fake classifier (no model or GPU needed)
  test_classifier.py default conditions, label order, and releasing the model
  test_idle.py       when an idle model is released, with a fake classifier and clock
  test_sentences.py  splitter behavior and offsets
  test_settings.py   the GPU or CPU choice, and the condition and device rule
  test_taxonomy.py   taxonomy.json names exactly the benchmark's ten categories
```

## Not implemented yet

- **Style-specific rewrite prompts.** All three UI styles use the evaluated
  `summary-v3` prompt. Each rewrite reports that limitation as a warning until separate
  prompts are evaluated.
- **Rewriter idle release.** The classifier is released after an idle period; the
  rewriter stays loaded until the process exits.
- **Model choice by memory tier.** The service chooses between the GPU and the CPU
  at startup and runs the same head on both. Choosing a model by memory tier is
  FR-19. Only NVIDIA GPUs are used; a computer with an AMD, Intel, or Apple GPU runs
  on the CPU.
- **History and caching across restarts.** The service keeps an in-memory cache only.
  Nothing is written to disk.
- **Browser wiring.** `frontend/` still uses example data and does not call this
  service.
