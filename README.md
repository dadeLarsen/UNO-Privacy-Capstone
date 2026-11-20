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

The interface now calls [`api/`](api/), which forwards policies to [`modelService/`](modelService/) and returns one category per sentence, source-linked category summaries, and source-linked rewrites. Classification, summaries, and rewriting all run locally. The work so far also chooses the models:

- **Classifier:** a linear head on EmbeddingGemma 2 embeddings leads, at about 0.04 s per sentence and 0.6 GiB, with accuracy close to much larger models. It has not yet been timed on a computer without a GPU.
- **Rewriter:** Qwen3.5-4B Q4_K_M GGUF through llama.cpp, with the evaluated `summary-v3` prompt. It is about 10 s per passage on a desktop CPU and 2.6–2.7 s on tested GPUs. All three UI styles currently use `summary-v3`; style-specific prompts still need evaluation.
- **Runtime:** AirLLM layer streaming, the original plan, was far too slow to use interactively and is now only a fallback for larger rewriters.

Results and their caveats are in the [model research README](privacy-research/README.md).

## Run locally

After the one-time setup in [`modelService/README.md`](modelService/README.md) and [`api/README.md`](api/README.md), download the pinned rewriter once while online:

```bash
cd privacy-research
uv run pbench download conditions/rewrite/qwen3.5-4b-q4km-cpu-summary-v3.toml
cd ..
```

GPU users can download `conditions/rewrite/qwen3.5-4b-q4km-gpu-summary-v3.toml` instead; both name the same GGUF file. The service never downloads at request time. Then start the whole app from this folder with:

```bash
python3 run.py
```

On Windows, use `python run.py`. The launcher starts the model service on `127.0.0.1:8002`, the API on `127.0.0.1:8001`, and Vite on `127.0.0.1:5173`, wires their URLs together, prefixes their logs, and stops all three when you press Ctrl+C. The first model start can take a while; classification works once the model log says it is ready.

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
