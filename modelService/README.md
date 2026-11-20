# modelService

A localhost-only Python service that runs the project's chosen classifier on a privacy
policy and returns one category per sentence. It is the first piece of the app's
backend; the browser UI is not wired to it yet, and rewriting is not implemented
because no rewriter has been chosen (AR-02).

The service reuses `privacy-research/` rather than loading a model itself. A
**condition** file pins the model file, the fitted head, the instruction prefix, and
the category order, and the benchmark's strategy checks the fitted weight hashes
before inference. By default the service runs the prior-shifted EmbeddingGemma 2
linear head, the classifier that met NFR-05 on both APPSI-139 samples. The embeddings
come from a Q8_0 GGUF build of EmbeddingGemma 2 (296 MiB) through llama.cpp. The
service has one condition for the GPU and one for the CPU, with the same file and
head, and picks between them at startup, so every computer gets the same classifier.

## What it does

| Route | Input | Output |
| --- | --- | --- |
| `GET /health` | — | Whether the service is ready to analyze, plus the model's pinned identity |
| `GET /categories` | — | The ten categories, definitions, display names, and colors |
| `POST /analyze` | JSON `{"text": "...", "include_scores": false}` | One label per sentence, with exact character offsets |
| `POST /analyze/upload` | multipart `.txt` file, optional `include_scores` query | Same response as `/analyze` |

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

The service never logs policy text. It binds to a loopback address only, loads models
from the local Hugging Face cache, and never downloads at request time (NFR-01,
NFR-02).

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12. `modelService`
depends on `privacy-research` by path, so both folders must stay side by side. Run
every command below from this folder. Installing the service also installs the
benchmark and its `pbench` command, so `privacy-research/` needs no setup of its own.

Choose one of the two setups. Each installs the service, caches the pinned model
file while online, and runs the tests. Both default conditions pin the same file, so
either download serves the GPU and the CPU.

**With an NVIDIA GPU.** This installs the CUDA 13.2 builds of torch and llama.cpp.
They need a newer NVIDIA driver than CUDA 12.8 did: NVIDIA names 595.45.04 as the
oldest for Linux and gives no number for Windows, where it was checked on driver
610.88:

```powershell
uv sync
uv run pbench download ../privacy-research/conditions/classification/appsi139/embeddinggemma2-gguf-q8-gpu-head-linear-balanced-tuned-limit2048-appsi139-test-half.toml
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
uv run pytest
```

Set `UV_NO_SYNC` before any other `uv` command. After pulling a change to `uv.lock`,
rerun the `uv sync` line.

Both setups install `llama-cpp-python`, which runs the classifier's GGUF file through
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
RTX 4090 and on its CPU; on a Mac only the benchmark has run this file.

The service releases the model after 300 seconds without an analyze request (NFR-03),
on the GPU and on the CPU; `MODEL_SERVICE_IDLE_RELEASE_SECONDS` changes the time. The
next request loads the model again before it answers, which took 0.4 to 0.5 s on the
desktop, and `/health` reports ready throughout. Measured on the desktop, a release
returned about 0.7 GiB of system memory on the CPU. On the GPU it took the process
from about 790 MiB of GPU memory to 390 MiB: the process keeps its CUDA context until
it exits, so its GPU memory falls but not to zero.

These examples work in Windows PowerShell 5.1 and in PowerShell 7. They call
`curl.exe` because 5.1 makes `curl` an alias of `Invoke-WebRequest`, and they pipe the
JSON in because 5.1 strips the double quotes inside a program's arguments.

```powershell
curl.exe http://127.0.0.1:8000/health

'{"text":"We collect your email address. We share it with marketing partners."}' |
  curl.exe -X POST http://127.0.0.1:8000/analyze -H "Content-Type: application/json" -d "@-"

curl.exe -X POST "http://127.0.0.1:8000/analyze/upload?include_scores=true" -F "file=@policy.txt"
```

In cmd, the first and last commands work as written. Escape the JSON's quotes instead:

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

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_SERVICE_CONDITION` | unset | Condition TOML to run instead of a default condition. It must name a GGUF file. It names its own device, so it cannot be combined with a device |
| `MODEL_SERVICE_DEVICE` | the detected hardware | `gpu` or `cpu`: which of the two default conditions to run. Cannot be combined with a condition |
| `MODEL_SERVICE_HOST` | `127.0.0.1` | Must be a loopback address |
| `MODEL_SERVICE_PORT` | `8000` | Listening port |
| `MODEL_SERVICE_MAX_CHARACTERS` | `200000` | Maximum policy length |
| `MODEL_SERVICE_CACHE_SIZE` | `4096` | Sentences kept in the in-memory cache |
| `MODEL_SERVICE_PRELOAD` | `true` | Load the model at startup |
| `MODEL_SERVICE_IDLE_RELEASE_SECONDS` | `300` | Seconds without an analyze request before the service releases the model, on the GPU and on the CPU |
| `MODEL_SERVICE_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173` | Allowed browser origins |

Command-line flags `--condition`, `--device`, `--host`, `--port`, and `--no-preload`
override the environment for one run. The service refuses to start with both a
condition and a device, whether each comes from a flag or a variable.

## Layout

```
src/model_service/
  app.py         FastAPI routes, upload handling, and response shape
  classifier.py  privacy_bench condition/strategy adapter and sentence cache
  idle.py        releases the model once the service has been idle
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

- **Rewrites.** No rewriter has been evaluated (AR-02), so there is no `/rewrite`
  route. Adding one should follow the same pattern: a pinned condition, a strategy,
  and a response that links each block to its source sentence IDs.
- **Model choice by memory tier.** The service chooses between the GPU and the CPU
  at startup and runs the same head on both. Choosing a model by memory tier is
  FR-19. An NVIDIA GPU is used through CUDA and an Apple GPU through Metal; a
  computer with an AMD or Intel GPU runs on the CPU.
- **History and caching across restarts.** The service keeps an in-memory cache only.
  Nothing is written to disk.
- **Browser wiring.** `frontend/` still uses example data and does not call this
  service.
