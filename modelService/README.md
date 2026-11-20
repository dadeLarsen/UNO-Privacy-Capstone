# modelService

A localhost-only Python service that runs the project's chosen classifier on a privacy
policy and returns one category per sentence, and that rewrites a passage in plain
language. It is the first piece of the app's backend; the browser UI is not wired to
it yet.

The service reuses `privacy-research/` rather than loading a model itself. A
**condition** file pins the model file, the fitted head, the instruction prefix, and
the category order, and the benchmark's strategy checks the fitted weight hashes
before inference. By default the service runs the prior-shifted EmbeddingGemma 2
linear head, the classifier that met NFR-05 on both APPSI-139 samples. The embeddings
come from a Q8_0 GGUF build of EmbeddingGemma 2 (296 MiB) through llama.cpp. The
service has one condition for the GPU and one for the CPU, with the same file and
head, and picks between them at startup, so every computer gets the same classifier.

The rewriter is a condition too. By default it is Qwen3.5-4B in a 4-bit GGUF file
(Q4_K_M, 2.55 GiB) with the benchmark's `summary-v3` prompt, the pair whose rewrites
`privacy-research/docs/rewriting-findings.md` judged. AR-02 has not settled the
rewriter, and the prompt is still being tuned; another prompt is another condition
file, with no change to the service. The classifier and the rewriter are never in
memory together.

## What it does

| Route | Input | Output |
| --- | --- | --- |
| `GET /health` | — | Whether the service is ready to analyze, plus each model's pinned identity |
| `GET /categories` | — | The ten categories, definitions, display names, and colors |
| `POST /analyze` | JSON `{"text": "...", "include_scores": false}` | One label per sentence, with exact character offsets |
| `POST /analyze/upload` | multipart `.txt` file, optional `include_scores` query | Same response as `/analyze` |
| `POST /rewrite` | JSON `{"text": "..."}` | One plain-language rewrite of the passage |

Each sentence in the response has `id`, `text`, `start`, `end`, `label_id`, and
`label`. `start` and `end` are offsets into the policy exactly as submitted, so the
UI can highlight the original text without re-splitting it. `label_id` follows
`privacy_bench.tasks.classification.taxonomy`, the category order the model was
fitted on.

A sentence the model cannot classify, such as one over the condition's token limit,
does not fail the request. It comes back with `label_id` and `label` set to `null` and
an `error` that gives the reason, and every other sentence keeps its label. With
`include_scores`, its `scores` is `null` too. The default conditions take 2,048 tokens
per sentence; the longest sentence in the benchmark's two datasets has 472.

`/rewrite` returns `rewrite`, the text, with `status`, `output_tokens`, and the `model`
that wrote it, which names the prompt. `status` is the benchmark's check of the output:
`valid`, `truncated` when the output limit of 1,024 tokens cut it, `empty`, or
`reasoning` when the model left reasoning tags in it. The passage, the prompt, and the
output must fit the model's context of 4,096 tokens together. A passage that does not
fit is refused with status 413 and its token count; nothing is truncated. The benchmark's
passages have 86 to 340 words and take 390 to 685 tokens with the prompt.

The service never logs policy text. It binds to a loopback address only, loads models
from the local Hugging Face cache, and never downloads at request time (NFR-01,
NFR-02).

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12. `modelService`
depends on `privacy-research` by path, so both folders must stay side by side. Run
every command below from this folder. Installing the service also installs the
benchmark and its `pbench` command, so `privacy-research/` needs no setup of its own.

Choose one of the two setups. Each installs the service, caches the two pinned model
files while online (296 MiB and 2.55 GiB), and runs the tests. A model's two default
conditions pin the same file, so either pair of downloads serves the GPU and the CPU.

**With an NVIDIA GPU.** This installs the CUDA 13.2 builds of torch and llama.cpp.
They need a newer NVIDIA driver than CUDA 12.8 did: NVIDIA names 595.45.04 as the
oldest for Linux and gives no number for Windows, where it was checked on driver
610.88:

```powershell
uv sync
uv run pbench download ../privacy-research/conditions/classification/appsi139/embeddinggemma2-gguf-q8-gpu-head-linear-balanced-tuned-limit2048-appsi139-test-half.toml
uv run pbench download ../privacy-research/conditions/rewrite/qwen3.5-4b-q4km-gpu-summary-v3.toml
uv run pytest
```

This setup also works without an NVIDIA GPU, because the CUDA build of llama.cpp also
runs on the CPU. The service then prints one line at startup, `ggml_cuda_init: failed
to initialize CUDA`, and runs the CPU condition. That was checked with the GPU hidden
(`CUDA_VISIBLE_DEVICES=-1`), not yet on a computer without an NVIDIA driver.

**Without an NVIDIA GPU.** This installs the much smaller CPU build of torch and
compiles llama.cpp:

```powershell
$env:UV_NO_SYNC = "1"    # keep `uv run` from installing the CUDA build; set it in your profile
uv sync --no-group cu132 --group cpu
uv run pbench download ../privacy-research/conditions/classification/appsi139/embeddinggemma2-gguf-q8-cpu-head-linear-balanced-tuned-limit2048-appsi139-test-half.toml
uv run pbench download ../privacy-research/conditions/rewrite/qwen3.5-4b-q4km-cpu-summary-v3.toml
uv run pytest
```

Set `UV_NO_SYNC` before any other `uv` command. After pulling a change to `uv.lock`,
rerun the `uv sync` line.

Both setups install `llama-cpp-python`, which runs both GGUF files through
llama.cpp. With an NVIDIA GPU, Windows and Linux on x64 get a built CUDA wheel. Without
one, and on every other platform, `uv sync` compiles the package: that needs CMake and a
C++ compiler, which are the Visual Studio build tools on Windows and the Xcode command
line tools on a Mac. A Mac's build includes llama.cpp's Metal GPU backend. The
benchmark's README says more under Setup. torch is still installed, because the
benchmark's other classifiers use it; the service itself no longer does.

Some checkpoints may require accepting their license on Hugging Face first. The service
itself never downloads; it reads the snapshot from the local cache.

## Run

```powershell
uv run model-service
# or: uv run python -m model_service
```

The service listens on `http://127.0.0.1:8000`. At startup it runs the GPU condition
when the installed llama.cpp build has a GPU backend that finds a GPU: CUDA with an
NVIDIA GPU on Windows and Linux, Metal on a Mac. Every other computer runs the CPU
condition. `/health` reports the device in use. The service was run on the desktop's
RTX 4090 and on its CPU; on a Mac only the benchmark has run these files.

Only one model is in memory at a time (NFR-03). The classifier loads at startup and the
rewriter on the first rewrite. Before a request uses one model, the service releases
the other, and a request waits while the other model still has a request under way, so
no model is released in the middle of one. On the desktop, a rewrite of an 86-word
passage that had to load the rewriter first took 2.0 s on the GPU and 8.0 s on the CPU,
against 0.5 s and 6.2 s with the rewriter loaded. Labeling 42 sentences right after a
rewrite took 1.4 s and 1.9 s, against 0.6 s and 1.0 s. Through label, rewrite, label,
rewrite, the process's working set peaked at 5.0 GiB on the CPU, and its GPU memory
at 3.7 GiB on the GPU.

The service also releases the loaded model after 300 seconds without an analyze or
rewrite request, on the GPU and on the CPU; `MODEL_SERVICE_IDLE_RELEASE_SECONDS`
changes the time. The next request loads the model again before it answers, which took
0.4 to 0.5 s for the classifier on the desktop, and `/health` reports ready throughout.
Measured on the desktop, a release of the classifier returned about 0.7 GiB of system
memory on the CPU. On the GPU it took the process from about 790 MiB of GPU memory to
390 MiB: the process keeps its CUDA context until it exits, so its GPU memory falls
but not to zero.

These examples work in Windows PowerShell 5.1 and in PowerShell 7. They call
`curl.exe` because 5.1 makes `curl` an alias of `Invoke-WebRequest`, and they pipe the
JSON in because 5.1 strips the double quotes inside a program's arguments.

```powershell
curl.exe http://127.0.0.1:8000/health

'{"text":"We collect your email address. We share it with marketing partners."}' |
  curl.exe -X POST http://127.0.0.1:8000/analyze -H "Content-Type: application/json" -d "@-"

curl.exe -X POST "http://127.0.0.1:8000/analyze/upload?include_scores=true" -F "file=@policy.txt"

'{"text":"We may retain account records for up to 30 days after you delete your account, except where a longer period is required by law."}' |
  curl.exe -X POST http://127.0.0.1:8000/rewrite -H "Content-Type: application/json" -d "@-"
```

In cmd, the first and third commands work as written. Escape the JSON's quotes instead:

```bat
curl -X POST http://127.0.0.1:8000/analyze -H "Content-Type: application/json" -d "{\"text\":\"We collect your email address. We share it with marketing partners.\"}"
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
    "id": "ggml-org/embeddinggemma-2-GGUF",
    "revision": "6ea3e3a309d12a4779cfda98b8c19800d2d5da71",
    "precision": "q8_0",
    "device": "gpu",
    "strategy": "embedding-head",
    "head": "linear-balanced-tuned",
    "taxonomy_id": "opp115-10-other",
    "taxonomy_version": 1
  }
}
```

Example `/rewrite` response (abridged):

```json
{
  "text_sha256": "…",
  "characters": 124,
  "rewrite": "…",
  "status": "valid",
  "output_tokens": 38,
  "model": {
    "id": "unsloth/Qwen3.5-4B-GGUF",
    "revision": "e87f176479d0855a907a41277aca2f8ee7a09523",
    "precision": "q4_k_m",
    "device": "gpu",
    "strategy": "prompted",
    "prompt": "summary-v3",
    "max_new_tokens": 1024
  }
}
```

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_SERVICE_CONDITION` | unset | Condition TOML to run instead of a default condition. It must name a GGUF file. It names its own device, so it cannot be combined with a device |
| `MODEL_SERVICE_REWRITE_CONDITION` | unset | Rewrite condition TOML to run instead of a default one. It names its own device, and a device may be set beside it |
| `MODEL_SERVICE_DEVICE` | the detected hardware | `gpu` or `cpu`: which default conditions to run, for the classifier and for the rewriter. Cannot be combined with `MODEL_SERVICE_CONDITION` |
| `MODEL_SERVICE_HOST` | `127.0.0.1` | Must be a loopback address |
| `MODEL_SERVICE_PORT` | `8000` | Listening port |
| `MODEL_SERVICE_MAX_CHARACTERS` | `200000` | Maximum length of a policy or of a passage |
| `MODEL_SERVICE_CACHE_SIZE` | `4096` | Sentences kept in the in-memory cache |
| `MODEL_SERVICE_PRELOAD` | `true` | Load the classifier at startup |
| `MODEL_SERVICE_IDLE_RELEASE_SECONDS` | `300` | Seconds without an analyze or rewrite request before the service releases the loaded model, on the GPU and on the CPU |
| `MODEL_SERVICE_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173` | Allowed browser origins |

Command-line flags `--condition`, `--rewrite-condition`, `--device`, `--host`,
`--port`, and `--no-preload` override the environment for one run. The service refuses
to start with both a classification condition and a device, whether each comes from a
flag or a variable.

## Layout

```
src/model_service/
  app.py         FastAPI routes, upload handling, and response shape
  classifier.py  privacy_bench condition/strategy adapter and sentence cache
  idle.py        keeps one model in memory, and none once the service has been idle
  rewriter.py    privacy_bench condition/strategy adapter for the rewrite route
  sentences.py   offset-preserving sentence splitter
  settings.py    environment and CLI configuration, and the GPU or CPU choice
  taxonomy.py    the ten categories: names and order from privacy_bench, the rest from taxonomy.json
  taxonomy.json  each category's display name, definition, and color, keyed by its name
tests/
  test_api.py        routes with a fake classifier and rewriter (no model or GPU needed)
  test_classifier.py default conditions, label order, and releasing the model
  test_idle.py       when a model is released, with fake models and a fake clock
  test_rewriter.py   default rewrite conditions, the rewrite's text and status, and releasing the model
  test_sentences.py  splitter behavior and offsets
  test_settings.py   the GPU or CPU choice, and the condition and device rule
  test_taxonomy.py   taxonomy.json names exactly the benchmark's ten categories
```

## Not implemented yet

- **The rest of rewriting.** `/rewrite` rewrites one passage with one summary prompt.
  It has none of the three styles (FR-09 to FR-11), does not link the rewrite to its
  source sentences (FR-12), does not rewrite a whole policy section by section
  (FR-13), and reports no progress and cannot be cancelled (FR-17, FR-18).
- **Model choice by memory tier.** The service chooses between the GPU and the CPU
  at startup and runs the same head on both. Choosing a model by memory tier is
  FR-19. An NVIDIA GPU is used through CUDA and an Apple GPU through Metal; a
  computer with an AMD or Intel GPU runs on the CPU.
- **History and caching across restarts.** The service keeps an in-memory cache only.
  Nothing is written to disk.
- **Browser wiring.** `frontend/` still uses example data and does not call this
  service.
