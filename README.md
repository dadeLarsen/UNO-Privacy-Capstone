# Privacy Policy Reader

*Working name.* A local web app that helps people understand privacy policies. It runs entirely on the user's computer: after a one-time setup, launching it and using every feature need no internet connection, and the policy text never leaves the machine.

The app rebuilds the parent thesis's classify-and-rewrite tool so that it runs locally. It is a capstone project for Dr. Alfredo J. Perez, built by Ethan McGee, Dade Larsen, Tim Pippin, and Nathan Johnson.

## What it does

1. **Paste a policy.** The text is kept exactly as entered and split into sentences, which every label and rewrite points back to.
2. **Analyze Categories.** Each sentence gets one of ten categories. A highlight toggle shades sentences by category, with a legend, per-category counts, and filters.
3. **Rewrite.** **Paraphrase**, **Simplify for Kids**, and **Simplify for Adults** rewrite a selection, or the whole policy when nothing is selected. Each rewrite appears beside the original, linked to the sentences it came from, and may not drop a condition, number, or hedge to read more simply.

The categories are the thesis's ten: First Party Collection/Use, Third Party Sharing/Collection, User Choice/Control, Do Not Track, International and Specific Audiences, Data Security, Policy Change, Data Retention, User Access, Edit and Deletion, and Other.

Not planned for the first version: chat about a policy, PDF or DOCX import, languages other than English, user-added models, and accounts, update checks, or telemetry.

## Targets

- **Hardware:** every core feature on a desktop or laptop with 8–12 GB of total memory and no dedicated GPU. Lighter and heavier tiers (2–4 GB, 16 GB or more) may get different models. All targets are provisional until tested.
- **Platform:** Windows 10/11 (x64), run from this repository: a JavaScript/TypeScript interface in the browser and a Python process for the models, both on localhost. Setup installs the dependencies and downloads the pinned models once; there is no installer. macOS (Apple Silicon) and Linux are optional.

## Status

The interface calls [`api/`](api/), which forwards policies to [`modelService/`](modelService/) and returns one category per sentence, source-linked category summaries, and source-linked rewrites. Classification, summaries, and rewriting all run locally. The summaries quote classified sentences, and all three rewrite styles use the one summary prompt; style-specific prompts are not built. The work so far also chooses the models:

- **Classifier:** a linear head on EmbeddingGemma 2 embeddings leads, with accuracy close to much larger models. The service runs it from a 296 MiB file through llama.cpp, at about 0.01 s per sentence on a GPU and 0.02 s on a CPU, timed on the desktop and on a MacBook. It has not yet been timed on a computer of the 8–12 GB target.
- **Rewriter:** not settled (AR-02). The service runs Qwen3.5-4B at 4-bit through llama.cpp with the `summary-v3` prompt, the pair that [the rewriting findings](privacy-research/docs/rewriting-findings.md) judged; the prompt is still being tuned.
- **Runtime:** AirLLM layer streaming, the original plan, was far too slow to use interactively and is now only a fallback for larger rewriters.

Results and their caveats are in the [model research README](privacy-research/README.md).

## Run locally

Setup needs [uv](https://docs.astral.sh/uv/), [Node.js](https://nodejs.org/) with npm, and the internet once. From this folder, run:

```bash
python3 run.py setup
```

It installs the model service, the API, and the frontend, and downloads the two pinned model files (296 MiB and 2.55 GiB). It stops at the first step that fails and names it. Where it finds `nvidia-smi`, it installs the CUDA 13.2 build of llama.cpp; elsewhere it compiles llama.cpp for the CPU, which needs CMake and a C++ compiler, as Setup in [`modelService/README.md`](modelService/README.md) explains. Run it again after pulling a change to a `uv.lock` or to `package-lock.json`.

Then start the whole app from this folder with:

```bash
python3 run.py
```

On Windows, use `python` in place of `python3`. The launcher starts the model service on `127.0.0.1:8000`, the API on `127.0.0.1:8001`, and Vite on `127.0.0.1:5173`, wires their URLs together, prefixes their logs, and stops all three when you press Ctrl+C. The first model start can take a while; classification works once the model log says it is ready.

Override any port with an environment variable:

```bash
MODEL_SERVICE_PORT=8100 API_PORT=8101 FRONTEND_PORT=5174 python3 run.py
```

## Repository

| Path | Contents |
| --- | --- |
| [docs/](docs/) | The [requirements](docs/Requirements-Document.md): goal model, functional and nonfunctional requirements, wireframes, object model, and the analyses that choose the models. The draft of Chapter 2 of the final report. |
| [api/](api/) | The localhost-only FastAPI layer the browser calls: uploads, classifications, summaries, and cancellation. Calls `modelService/`. Has its own setup. |
| [modelService/](modelService/) | The localhost-only Python service that classifies a policy and returns one category per sentence. Reuses the benchmark's pinned models. Has its own setup. |
| [privacy-research/](privacy-research/) | The model benchmark (`pbench`): run conditions, datasets, tracked run records, and results. Has its own setup. |

[INDEX.md](INDEX.md) maps every folder and file.
