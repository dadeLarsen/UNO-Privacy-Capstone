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

The app itself has not been started. The work so far chooses its models:

- **Classifier:** a linear head on EmbeddingGemma 2 embeddings leads, at about 0.04 s per sentence and 0.6 GiB, with accuracy close to much larger models. It has not yet been timed on a computer without a GPU.
- **Rewriter:** not chosen; no rewriting has been evaluated.
- **Runtime:** AirLLM layer streaming, the original plan, was far too slow to use interactively and is now only a fallback for larger rewriters.

Results and their caveats are in the [model research README](privacy-research/README.md).

## Repository

| Path | Contents |
| --- | --- |
| [docs/](docs/) | The [requirements](docs/Requirements-Document.md): goal model, functional and nonfunctional requirements, wireframes, object model, and the analyses that choose the models. The draft of Chapter 2 of the final report. |
| [privacy-research/](privacy-research/) | The model benchmark (`pbench`): run conditions, datasets, tracked run records, and results. Has its own setup. |

[INDEX.md](INDEX.md) maps every folder and file.
