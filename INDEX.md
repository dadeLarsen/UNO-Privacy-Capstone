# Index

A map of this repository. What the app is and where it stands: [README.md](README.md). Model results and how to run the benchmark: [privacy-research/README.md](privacy-research/README.md).

Data, configuration, and results entries have a status:

- **Current**: in use and reflects present thinking.
- **Frozen**: pinned by SHA-256 in samples, conditions, or run records. Never edit or resave it (opening a CSV in Excel and saving it breaks the hash); freeze a new file instead.
- **Historical**: kept for provenance. Don't mix its results with current runs.
- **Generated**: written by a command and not tracked unless its row says so; regenerate it rather than edit it.

## Start here

| Read | For |
| --- | --- |
| [README.md](README.md) | What the app does, its hardware and platform targets, and its status |
| [docs/Requirements-Document.md](docs/Requirements-Document.md) | What the app must do and how each requirement will be checked |
| [privacy-research/README.md](privacy-research/README.md) | Which models lead and why, plus setup and the `pbench` workflow |
| [privacy-research/docs/classification.md](privacy-research/docs/classification.md) | The classification task in detail: data, output contract, scoring, strategies, fitted heads |

## Top level

| Path | Contents |
| --- | --- |
| [docs/](docs/) | Requirements and design for the app |
| [privacy-research/](privacy-research/) | The model benchmark: a self-contained Python project with its own environment |
| [.gitignore](.gitignore) | Ignores for the whole repository; `privacy-research/` adds its own |

The app has no code yet. When it does, its folder gets a section here.

## App requirements: [docs/](docs/)

| File | Status | Contents |
| --- | --- | --- |
| [Requirements-Document.md](docs/Requirements-Document.md) | Current | Draft of Chapter 2 of the final report. Goal model, functional requirements FR-01 to FR-27 with MoSCoW priorities, wireframes, object model, nonfunctional requirements NFR-01 to NFR-09, analysis requirements AR-01, AR-02, and AR-04, external interfaces, and requirements not implemented. |
| [Requirements-Outline.md](docs/Requirements-Outline.md) | Current | The course's outline for that chapter. |
| [requirements-figures/](docs/requirements-figures/) | Current | The goal model ([goal-model.svg](docs/requirements-figures/goal-model.svg), and [goal-model-pistar.txt](docs/requirements-figures/goal-model-pistar.txt) to open in piStar with **File → Load Model**), the thesis interface screenshot (W1), and wireframes W2–W5 as SVG. |

## Model research: [privacy-research/](privacy-research/)

The `privacy_bench` package and its `pbench` command. Run every command from `privacy-research/`. Three kinds of file drive it:

- A **condition** (`conditions/`) names one task, strategy, sample, and pinned model. `pbench run` turns it into a run record under `results/runs/`.
- A **fit** (`fits/`) names labeled examples and the heads to learn from them. `pbench fit` writes the weights under `results/fits/`, and conditions refer to them.
- A **comparison** (`comparisons/`) lists saved runs and how to score them. `pbench compare` rescores them with the current code and writes a table and plots.

### Guides

| File | Status | Contents |
| --- | --- | --- |
| [README.md](privacy-research/README.md) | Current | Current direction, results with caveats, repository layout, setup, running and comparing, extending, and history. |
| [docs/classification.md](privacy-research/docs/classification.md) | Current | The ten categories and the OPP-115 mapping, every dataset and sample, the `{"label_id": N}` output contract, scoring rules, strategies, fitted heads, and prior shift. |

### Code: [src/privacy_bench/](privacy-research/src/privacy_bench/)

Layers depend only downward: `core`, `models`, and `report` never import a task, and no task imports another. [tests/test_architecture.py](privacy-research/tests/test_architecture.py) enforces it.

| File | Purpose |
| --- | --- |
| [cli.py](privacy-research/src/privacy_bench/cli.py) | `pbench run`, `fit`, `download`, `evaluate`, and `compare`. |
| **[core/](privacy-research/src/privacy_bench/core/)** | **Task-agnostic machinery** |
| [protocols.py](privacy-research/src/privacy_bench/core/protocols.py) | The `Task` and `Strategy` contracts between the harness, tasks, and strategies. Start here to understand the design. |
| [harness.py](privacy-research/src/privacy_bench/core/harness.py) | The one run loop: select items, preflight every input, warm up, predict, record, evaluate. |
| [conditions.py](privacy-research/src/privacy_bench/core/conditions.py) | Parses and validates condition files. |
| [runs.py](privacy-research/src/privacy_bench/core/runs.py) | The on-disk run record and what each of its files holds. |
| [fits.py](privacy-research/src/privacy_bench/core/fits.py) | Fit files, fit records, and the weight-hash check a run does before loading them. |
| [provenance.py](privacy-research/src/privacy_bench/core/provenance.py) | Records git commit and diff, interpreter, platform, and installed packages. |
| [paths.py](privacy-research/src/privacy_bench/core/paths.py), [files.py](privacy-research/src/privacy_bench/core/files.py) | Repository locations (including the frozen `data/` and the editable `prompts/`), hashing, and strict JSON reading (duplicate keys rejected). |
| **[models/](privacy-research/src/privacy_bench/models/)** | **Model backends: load a pinned checkpoint, run one input, measure it; know nothing about tasks** |
| [chat_model.py](privacy-research/src/privacy_bench/models/chat_model.py) | Greedy generation from a chat model, with an optional system message. Loads text-only and multimodal checkpoints (such as Ministral 3) and refuses one that does not load exactly. |
| [sequence_classifier.py](privacy-research/src/privacy_bench/models/sequence_classifier.py) | FP32 sequence classifiers (skythrone/privacy-model). |
| [sentence_embedder.py](privacy-research/src/privacy_bench/models/sentence_embedder.py) | sentence-transformers embeddings (EmbeddingGemma 2). |
| [jev/](privacy-research/src/privacy_bench/models/jev/) | JEV-9B: bundle validation and download (`bundle.py`), the LoRA adapter and decision head (`judge.py`), and quantization planning (`quantization.py`). |
| [hub.py](privacy-research/src/privacy_bench/models/hub.py) | Pinned Hugging Face snapshots: explicit download, then offline resolution only. Also the exact-load check every backend runs after loading weights. |
| [quantization.py](privacy-research/src/privacy_bench/models/quantization.py), [measure.py](privacy-research/src/privacy_bench/models/measure.py) | bitsandbytes INT8/NF4 settings; CUDA setup, timing, and peak memory. |
| **[report/](privacy-research/src/privacy_bench/report/)** | **Comparisons across saved runs** |
| [comparisons.py](privacy-research/src/privacy_bench/report/comparisons.py), [summary.py](privacy-research/src/privacy_bench/report/summary.py), [plot.py](privacy-research/src/privacy_bench/report/plot.py) | Parses comparison files, rescores each run through its task into `summary.csv`, and draws the bar matrix. |
| **[tasks/](privacy-research/src/privacy_bench/tasks/)** | **One package per task, registered in `TASKS` in [`__init__.py`](privacy-research/src/privacy_bench/tasks/__init__.py)** |
| [classification/task.py](privacy-research/src/privacy_bench/tasks/classification/task.py) | The classification task object: selects items and gold, builds strategies, evaluates. |
| [classification/taxonomy.py](privacy-research/src/privacy_bench/tasks/classification/taxonomy.py) | The ten categories and their mapping from OPP-115's twelve labels. |
| [classification/data.py](privacy-research/src/privacy_bench/tasks/classification/data.py) | Dataset loading, sample manifests (how every frozen sample was drawn), and accepted gold labels. |
| [classification/prompts.py](privacy-research/src/privacy_bench/tasks/classification/prompts.py) | Category descriptions and the chat prompt built from them. |
| [classification/scoring.py](privacy-research/src/privacy_bench/tasks/classification/scoring.py) | Strict response parsing and the metrics. |
| [classification/fitting.py](privacy-research/src/privacy_bench/tasks/classification/fitting.py) | The `embedding-heads` and `prior-shift` fitters. |
| [classification/report.py](privacy-research/src/privacy_bench/tasks/classification/report.py) | How classification runs are rescored, labeled, and annotated in comparisons. |
| [classification/strategies/](privacy-research/src/privacy_bench/tasks/classification/strategies/) | One file per strategy (`prompted`, `classifier_head`, `jev_choice`, `embedding_similarity`, `embedding_head`), registered in `STRATEGIES`; `settings.py` holds shared condition parsing. |
| [rewrite/task.py](privacy-research/src/privacy_bench/tasks/rewrite/task.py) | The rewrite task object: selects passages, builds the strategy, evaluates, and writes `side-by-side.md` for comparisons. |
| [rewrite/data.py](privacy-research/src/privacy_bench/tasks/rewrite/data.py) | Frozen passage files, and prompt files: an optional `system` message and a `user` template with one `{passage}`. |
| [rewrite/strategy.py](privacy-research/src/privacy_bench/tasks/rewrite/strategy.py) | The `prompted` strategy: renders the prompt file for each passage and generates with the chat backend. |
| [rewrite/scoring.py](privacy-research/src/privacy_bench/tasks/rewrite/scoring.py) | Validity, length, and the numbers and qualifiers a rewrite dropped (flags, not quality scores), and the review Markdown. |

### Tests: [tests/](privacy-research/tests/)

`uv run pytest` runs offline. The layout mirrors `src/privacy_bench/`, one test file per strategy; [conftest.py](privacy-research/tests/conftest.py) builds tiny classifier and embedder models on the CPU, and [models/jev/conftest.py](privacy-research/tests/models/jev/conftest.py) a tiny JEV bundle.

### Tools: [tools/](privacy-research/tools/)

One-time scripts that froze data files or imported runs. Each refuses to overwrite its output, so they document how a file was made rather than being run again.

| File | Status | Made |
| --- | --- | --- |
| [prepare_appsi139.py](privacy-research/tools/prepare_appsi139.py) | Historical | `appsi139-sentences.csv`, `appsi139-astra.csv`, and the APPSI-139 samples, copied from the AirLLM workspace. |
| [split_appsi139.py](privacy-research/tools/split_appsi139.py) | Historical | The APPSI-139 tuning and test halves. |
| [make_opp115_examples.py](privacy-research/tools/make_opp115_examples.py) | Historical | `opp115-examples.json`, the training sentences for fitted heads. |
| [import_legacy_runs.py](privacy-research/tools/import_legacy_runs.py) | Historical | The six runs imported from the AirLLM workspace. |
| [make_appsi139_sections.py](privacy-research/tools/make_appsi139_sections.py) | Historical | `appsi139-sections-20.json`, the rewrite passages, from the AirLLM workspace's `annotation.jsonl`. |

### Inputs: [data/](privacy-research/data/)

| Path | Status | Contents |
| --- | --- | --- |
| [datasets/opp115-sentences.csv](privacy-research/data/datasets/opp115-sentences.csv) | Frozen | 10,717 OPP-115 website-policy sentences with OPP-115's twelve label columns. |
| [datasets/appsi139-sentences.csv](privacy-research/data/datasets/appsi139-sentences.csv) | Frozen | 13,291 APPSI-139 app-policy sentences, already in the ten categories. |
| [classification/samples/](privacy-research/data/classification/samples/) | Frozen | Which sentences a run covers. `pilot-200-10cat` (OPP-115, 20 per category); APPSI-139's `all`, `random-1000`, and `balanced-50` (460 sentences); and its `tuning-half` (7,233, for fits only) and `test-half` (6,058, for reporting). |
| [classification/gold/](privacy-research/data/classification/gold/) | Frozen | Accepted labels. `pilot-200-10cat-reviewed.csv` adds 54 reviewed alternatives to the pilot's labels, with the reviewer's notes in `pilot-200-10cat-review.jsonl`; `appsi139-astra.csv` adds ChatGPT Astra's relabeling to APPSI-139's. |
| [classification/examples/opp115-examples.json](privacy-research/data/classification/examples/opp115-examples.json) | Frozen | The 10,417 OPP-115 sentences fitted heads learn from; none appears in a test sample. |
| [classification/prompts/definitions.json](privacy-research/data/classification/prompts/definitions.json) | Frozen | One definition per category, used by every strategy. |
| [rewrite/passages/appsi139-sections-20.json](privacy-research/data/rewrite/passages/appsi139-sections-20.json) | Frozen | 20 APPSI-139 policy sections (86–340 words, one per policy, across three length bands) for rewriting, with each section's heading and sentence IDs. |

### Configuration

| Path | Status | Contents |
| --- | --- | --- |
| [conditions/classification/](privacy-research/conditions/classification/) | Current | One run condition per file, on the OPP-115 pilot. Names read `<model>-<precision>-<method>`: `definitions` is the definitions prompt (or, for EmbeddingGemma 2, similarity to the definitions, with a suffix naming its instruction prefix), and `head-<name>` is a fitted head. `privacy-model-fp32` has no method: it classifies the raw sentence. |
| [conditions/classification/appsi139/](privacy-research/conditions/classification/appsi139/) | Current | The same conditions on APPSI-139 samples, named `…-appsi139-<sample>`. `-tuned` and `-full` heads are prior-shifted at the tuned strength and at strength 1. |
| [conditions/rewrite/](privacy-research/conditions/rewrite/) | Current | One rewrite condition per candidate model on the 20 passages, named `<model>-<precision>-<prompt>`. |
| [prompts/rewrite/](privacy-research/prompts/rewrite/) | Current | Rewrite prompt files, editable: each run copies the prompt it used into its `input.json`. `summary-baseline` is a generic summary prompt for a first comparison. |
| [fits/classification/embeddinggemma2-opp115-heads.toml](privacy-research/fits/classification/embeddinggemma2-opp115-heads.toml) | Current | The `centroid`, `linear-balanced`, and `linear-frequencies` heads on EmbeddingGemma 2, learned from the OPP-115 examples. |
| [fits/classification/embeddinggemma2-appsi139-prior-shift.toml](privacy-research/fits/classification/embeddinggemma2-appsi139-prior-shift.toml) | Current | Shifts both linear heads toward the APPSI-139 tuning half's category mix. |
| [comparisons/](privacy-research/comparisons/) | Current | Each file backs a table or finding in the research README: `pilot-embeddinggemma2-heads` and `pilot-models-embeddinggemma2` (OPP-115 pilot), `appsi139-random-1000`, `appsi139-balanced-50`, and `appsi139-all` (APPSI-139), `appsi139-test-half` and `appsi139-balanced-50-prior-shift` (prior shift), and `embeddinggemma2-instructions` (instruction prefixes). |
| [comparisons/qwen-privacy-jev.toml](privacy-research/comparisons/qwen-privacy-jev.toml) | Historical | The imported runs, made under transformers 5.16.1. |

### Results: [results/](privacy-research/results/)

| Path | Status | Contents |
| --- | --- | --- |
| `runs/<task>/<condition>/<UTC start>/`, for example [runs/classification/](privacy-research/results/runs/classification/) | Frozen | One record per run: `condition.toml`, `run.json` (identity, provenance, status), `input.json`, `input-audit.json`, `warmup.json`, and `responses.jsonl`. |
| Runs started `20260918-*` and `20261002-*` | Historical | The six runs imported from the AirLLM workspace, each with its original `legacy-run.json`. Reruns under transformers 5.19.0 sit beside them. |
| [fits/classification/](privacy-research/results/fits/classification/)`<fit>/<UTC start>/` | Frozen | `fit.toml`, `fit.json` (provenance, cross-validation results, and the weights' hash), and `weights.safetensors`. |
| `runs/**/evaluation/` | Generated | Scores from `pbench run` or `pbench evaluate`; rewrite runs add `review.md`, each passage above its rewrite. |
| `comparisons/<name>/` | Generated | `summary.csv` and `matrix-overview.png`/`.svg` from `pbench compare`, plus `side-by-side.md` for rewrite comparisons. Five are committed so readers can see results without running it: `appsi139-test-half`, `embeddinggemma2-instructions`, `pilot-embeddinggemma2-heads`, `pilot-models-embeddinggemma2`, and `qwen-privacy-jev`. |

### Project files

| File | Contents |
| --- | --- |
| [pyproject.toml](privacy-research/pyproject.toml), [uv.lock](privacy-research/uv.lock), [.python-version](privacy-research/.python-version) | Python 3.12 and every package pinned, transitive ones included; torch comes from the CUDA 12.8 index. |
| [.gitattributes](privacy-research/.gitattributes) | LF line endings, and no line-ending conversion under `data/` and `results/`, whose bytes are hashed. |
| [.gitignore](privacy-research/.gitignore) | The environment, caches, and the generated `results/` folders. |
