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
| [privacy-research/docs/rewriting-findings.md](privacy-research/docs/rewriting-findings.md) | Which rewriting model to use and why, in plain language, for the team and presentations |

## Top level

| Path | Contents |
| --- | --- |
| [docs/](docs/) | Requirements and design for the app |
| [run.py](run.py) | One-command localhost launcher for the model service, API, and Vite frontend |
| [api/](api/) | The localhost-only frontend-facing API: uploads, classifications, summaries, and jobs |
| [modelService/](modelService/) | The localhost-only model service: classifies a policy and returns one category per sentence |
| [privacy-research/](privacy-research/) | The model benchmark: a self-contained Python project with its own environment |
| [.gitignore](.gitignore) | Ignores for the whole repository; `privacy-research/` adds its own |

The browser interface now calls `api/`, which calls `modelService/`. Classification,
extractive category summaries, and GGUF rewrites work locally. The rewriter uses one
evaluated `summary-v3` prompt for all three UI styles; style-specific prompts are not
yet evaluated.

## Frontend-facing API: [api/](api/)

A small FastAPI service in its own folder. The browser talks only to this API; it
forwards text to `modelService/`, returns classifications first, and later returns
source-linked summaries and rewrites through pollable jobs. Policies and jobs live in
memory and expire after an hour. Summaries are extractive; rewrites call the pinned
GGUF rewriter in `modelService/`.

| File | Status | Contents |
| --- | --- | --- |
| [README.md](api/README.md) | Current | Setup, run, routes, configuration, tests, and what is not implemented |
| [pyproject.toml](api/pyproject.toml) | Current | Dependencies and test configuration |
| [src/api_service/](api/src/api_service/) | Current | `app.py` routes and jobs, `schemas.py` Pydantic contract, `client.py` modelService adapter, `store.py` in-memory state, `generation.py`, `errors.py`, `settings.py`, and `__main__.py` |
| [tests/](api/tests/) | Current | Contract tests with a fake modelService; no model or GPU needed |

## Model service: [modelService/](modelService/)

A small FastAPI service that reuses the benchmark's pinned classifier. It binds to
localhost only, never downloads at request time, and returns exact character offsets
for every sentence. It runs the same fitted head on an NVIDIA GPU or on the CPU, and
chooses between them at startup.

| File | Status | Contents |
| --- | --- | --- |
| [README.md](modelService/README.md) | Current | Setup, run, API examples, configuration, and what is not implemented |
| [pyproject.toml](modelService/pyproject.toml) | Current | Dependencies, the CUDA and CPU groups for torch and llama-cpp-python, and the `model-service` command |
| [src/model_service/](modelService/src/model_service/) | Current | `app.py` routes, `classifier.py` benchmark adapter, `idle.py` idle GPU release, `sentences.py` splitter, `settings.py` configuration and the GPU or CPU choice, and `taxonomy.py` with `taxonomy.json`, the benchmark's ten categories plus each one's display name, definition, and color |
| [tests/](modelService/tests/) | Current | API and idle-release tests with a fake classifier, and classifier, splitter, settings, and taxonomy tests; no model or GPU needed |

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
- A **judgment** grades a rewrite comparison's runs against each passage's key points. `pbench judge prepare` writes one blind prompt per passage under `results/judgments/`, a judge answers each, and `pbench judge record` checks the replies and freezes them; `[scoring] judgments` in a comparison then scores runs from it.

### Guides

| File | Status | Contents |
| --- | --- | --- |
| [README.md](privacy-research/README.md) | Current | Current direction, results with caveats, repository layout, setup, running and comparing, extending, and history. |
| [docs/classification.md](privacy-research/docs/classification.md) | Current | The ten categories and the OPP-115 mapping, every dataset and sample, the `{"label_id": N}` output contract, scoring rules, strategies, fitted heads, and prior shift. |
| [docs/rewriting-findings.md](privacy-research/docs/rewriting-findings.md) | Current | A plain-language summary of the judged rewriting results: nine models on 40 passages, the automatic word checks and why they matter, key facts kept or lost, examples of errors, and a recommendation per computer size, what compression changes, and its limits. Numbers and plots come from `rewrite-keypoints-v3`, `rewrite-keypoints-v3-holdout`, and `rewrite-precision-v3-holdout`. Since 2026-11-18 it also gives how fast Qwen3.5-4B runs through bitsandbytes and through llama.cpp on the desktop and on a MacBook, from `rewrite-llama-cpp-v3`, its holdout twin, `rewrite-llama-cpp-v3-torch215`, `rewrite-cpu-runtimes-v3`, and `rewrite-cpu-runtimes-v3-fp32`. Since 2026-11-19 it reports the key-point judging of that model's bitsandbytes and llama.cpp rewrites, six runs read side by side, from `rewrite-keypoints-llama-cpp-v3-desktop`, `rewrite-keypoints-llama-cpp-v3-macbook`, and their holdout twins; it marks the MacBook's bitsandbytes run and the memory figures as not saved in the repository. |

### Code: [src/privacy_bench/](privacy-research/src/privacy_bench/)

Layers depend only downward: `core`, `models`, and `report` never import a task, and no task imports another. [tests/test_architecture.py](privacy-research/tests/test_architecture.py) enforces it.

| File | Purpose |
| --- | --- |
| [cli.py](privacy-research/src/privacy_bench/cli.py) | `pbench run`, `fit`, `download`, `evaluate`, `compare`, and `judge`. |
| **[core/](privacy-research/src/privacy_bench/core/)** | **Task-agnostic machinery** |
| [protocols.py](privacy-research/src/privacy_bench/core/protocols.py) | The `Task` and `Strategy` contracts between the harness, tasks, and strategies. Start here to understand the design. |
| [harness.py](privacy-research/src/privacy_bench/core/harness.py) | The one run loop: select items, preflight every input, warm up, predict, record, evaluate. |
| [conditions.py](privacy-research/src/privacy_bench/core/conditions.py) | Parses and validates condition files. |
| [runs.py](privacy-research/src/privacy_bench/core/runs.py) | The on-disk run record and what each of its files holds. |
| [fits.py](privacy-research/src/privacy_bench/core/fits.py) | Fit files, fit records, and the weight-hash check a run does before loading them. |
| [provenance.py](privacy-research/src/privacy_bench/core/provenance.py) | Records git commit and diff, interpreter, platform, and installed packages. |
| [paths.py](privacy-research/src/privacy_bench/core/paths.py), [files.py](privacy-research/src/privacy_bench/core/files.py) | Repository locations (including the frozen `data/` and the editable `prompts/`), hashing, and strict JSON reading (duplicate keys rejected). |
| **[models/](privacy-research/src/privacy_bench/models/)** | **Model backends: load a pinned checkpoint, run one input, measure it; know nothing about tasks** |
| [gguf_chat_model.py](privacy-research/src/privacy_bench/models/gguf_chat_model.py) | The one chat backend: greedy generation from a GGUF chat model through llama.cpp (the `llama-cpp-python` package), with an optional system message. Runs a `q4_k_m` file on the CPU, or on the GPU backend of the installed llama.cpp build. |
| [sequence_classifier.py](privacy-research/src/privacy_bench/models/sequence_classifier.py) | FP32 sequence classifiers (skythrone/privacy-model). |
| [sentence_embedder.py](privacy-research/src/privacy_bench/models/sentence_embedder.py) | sentence-transformers embeddings (EmbeddingGemma 2). |
| [gguf_sentence_embedder.py](privacy-research/src/privacy_bench/models/gguf_sentence_embedder.py) | Sentence embeddings from a GGUF embedding model through llama.cpp, with the interface of `sentence_embedder.py`. Takes a `bf16` or `q8_0` file that pools by the mean, on the CPU or on the GPU backend of the installed llama.cpp build. |
| [hub.py](privacy-research/src/privacy_bench/models/hub.py) | Pinned Hugging Face snapshots: explicit download of a checkpoint or of one named file, then offline resolution only. Also the exact-load check every backend runs after loading weights. |
| [measure.py](privacy-research/src/privacy_bench/models/measure.py) | CUDA setup, timing, and peak memory. |
| **[report/](privacy-research/src/privacy_bench/report/)** | **Comparisons across saved runs** |
| [comparisons.py](privacy-research/src/privacy_bench/report/comparisons.py), [summary.py](privacy-research/src/privacy_bench/report/summary.py), [plot.py](privacy-research/src/privacy_bench/report/plot.py) | Parses comparison files, rescores each run through its task into `summary.csv`, and draws the bar matrix. |
| **[tasks/](privacy-research/src/privacy_bench/tasks/)** | **One package per task, registered in `TASKS` in [`__init__.py`](privacy-research/src/privacy_bench/tasks/__init__.py)** |
| [classification/task.py](privacy-research/src/privacy_bench/tasks/classification/task.py) | The classification task object: selects items and gold, builds strategies, evaluates. |
| [classification/taxonomy.py](privacy-research/src/privacy_bench/tasks/classification/taxonomy.py) | The ten categories and their mapping from OPP-115's twelve labels. |
| [classification/data.py](privacy-research/src/privacy_bench/tasks/classification/data.py) | Dataset loading, sample manifests (how every frozen sample was drawn), and accepted gold labels. |
| [classification/prompts.py](privacy-research/src/privacy_bench/tasks/classification/prompts.py) | Category descriptions, and the chat prompt that the retired `prompted` strategy built from them. |
| [classification/scoring.py](privacy-research/src/privacy_bench/tasks/classification/scoring.py) | Strict response parsing and the metrics. |
| [classification/fitting.py](privacy-research/src/privacy_bench/tasks/classification/fitting.py) | The `embedding-heads` and `prior-shift` fitters. |
| [classification/report.py](privacy-research/src/privacy_bench/tasks/classification/report.py) | How classification runs are rescored, labeled, and annotated in comparisons. |
| [classification/strategies/](privacy-research/src/privacy_bench/tasks/classification/strategies/) | One file per strategy (`prompted`, `classifier_head`, `jev_choice`, `embedding_similarity`, `embedding_head`), registered in `STRATEGIES`; `settings.py` holds shared condition parsing. `prompted` and `jev_choice` are retired: each keeps the name, label, and timing note that comparisons read for its recorded runs, and refuses a condition by name. `embedding_head` embeds through llama.cpp when the condition names a GGUF `file`, through sentence-transformers otherwise. |
| [rewrite/task.py](privacy-research/src/privacy_bench/tasks/rewrite/task.py) | The rewrite task object: selects passages, builds the strategy, evaluates, prepares and records judgments, and writes `side-by-side.md` and, with judgments, `keypoints.png` for comparisons. |
| [rewrite/judging.py](privacy-research/src/privacy_bench/tasks/rewrite/judging.py) | Blind key-point judging: the checklists, one prompt per passage with shuffled letters, strict checks of each reply (every point graded, every quote found in its rewrite), the frozen record, and each run's scores. |
| [rewrite/chart.py](privacy-research/src/privacy_bench/tasks/rewrite/chart.py) | `keypoints.png`: each run's key points kept, partly kept, left out, or contradicted, and its share of rewrites that could mislead. |
| [rewrite/data.py](privacy-research/src/privacy_bench/tasks/rewrite/data.py) | Frozen passage files, and prompt files: an optional `system` message and a `user` template with one `{passage}`. |
| [rewrite/strategy.py](privacy-research/src/privacy_bench/tasks/rewrite/strategy.py) | The `prompted` strategy: renders the prompt file for each passage and generates from the condition's GGUF `file` through llama.cpp. Its transformers backend is retired, so it refuses a condition that names no file; recorded transformers runs still compare. |
| [rewrite/scoring.py](privacy-research/src/privacy_bench/tasks/rewrite/scoring.py) | Validity, length, and the numbers and qualifiers a rewrite dropped or added (flags, not quality scores); copying, reading grade, and format breaks; and the review Markdown. |

### Tests: [tests/](privacy-research/tests/)

`uv run pytest` runs offline. The layout mirrors `src/privacy_bench/`, one test file per strategy; [conftest.py](privacy-research/tests/conftest.py) builds a tiny classifier and a tiny embedder on the CPU, and [models/conftest.py](privacy-research/tests/models/conftest.py) holds the GGUF writer with which [models/test_gguf_chat_model.py](privacy-research/tests/models/test_gguf_chat_model.py) and [models/test_gguf_sentence_embedder.py](privacy-research/tests/models/test_gguf_sentence_embedder.py) write their own tiny GGUF files.

### Tools: [tools/](privacy-research/tools/)

One-time scripts that froze data files or imported runs. Each refuses to overwrite its output, so they document how a file was made rather than being run again.

| File | Status | Made |
| --- | --- | --- |
| [prepare_appsi139.py](privacy-research/tools/prepare_appsi139.py) | Historical | `appsi139-sentences.csv`, `appsi139-astra.csv`, and the APPSI-139 samples, copied from the AirLLM workspace. |
| [split_appsi139.py](privacy-research/tools/split_appsi139.py) | Historical | The APPSI-139 tuning and test halves. |
| [make_opp115_examples.py](privacy-research/tools/make_opp115_examples.py) | Historical | `opp115-examples.json`, the training sentences for fitted heads. |
| [import_legacy_runs.py](privacy-research/tools/import_legacy_runs.py) | Historical | The six runs imported from the AirLLM workspace. Five used a strategy that is now retired, so it imports them only at the commit the research README names under History. |
| [make_appsi139_sections.py](privacy-research/tools/make_appsi139_sections.py) | Historical | `appsi139-sections-20.json` and `appsi139-sections-20-holdout.json`, the rewrite passages, from the AirLLM workspace's `annotation.jsonl`. Rerunning it checks that existing files still match their draw. |

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
| [rewrite/passages/appsi139-sections-20-holdout.json](privacy-research/data/rewrite/passages/appsi139-sections-20-holdout.json) | Frozen | 20 more sections (93–302 words), drawn the same way from the policies the first 20 did not use, for checking prompts tuned on the first 20. |
| [rewrite/keypoints/](privacy-research/data/rewrite/keypoints/) | Frozen | One key-point checklist per passage set, named after it: each passage's facts, which of them are critical, and notes on ambiguous wording. Drafted by Claude from the passages alone; `reviewed_by` records a human review. |

### Configuration

| Path | Status | Contents |
| --- | --- | --- |
| [conditions/classification/](privacy-research/conditions/classification/) | Current | One run condition per file, on the OPP-115 pilot. Names read `<model>-<precision>-<method>`: `definitions` is EmbeddingGemma 2's similarity to the definitions, with a suffix naming its instruction prefix, and `head-<name>` is a fitted head. `privacy-model-fp32` has no method: it classifies the raw sentence. The Qwen3-14B and JEV-9B conditions went with the retired `prompted` and `jev-choice` strategies; each of their runs keeps its own copy as `condition.toml`. |
| [conditions/classification/appsi139/](privacy-research/conditions/classification/appsi139/) | Current | The same conditions on APPSI-139 samples, named `…-appsi139-<sample>`. `-tuned` and `-full` heads are prior-shifted at the tuned strength and at strength 1. The two `fp32-cpu` conditions run the tuned balanced head on the CPU in fp32. The eight `gguf` conditions run that head on a GGUF build of EmbeddingGemma 2 through llama.cpp: the `bf16` and the `q8` file, each on the CPU and on the GPU, on the test half and on the balanced sample. Every other condition runs on the GPU. |
| [conditions/rewrite/](privacy-research/conditions/rewrite/) | Current | Rewrite conditions on the 20 passages, named `<model>-<precision>-<prompt>`. The two `q4km` conditions run a 4-bit GGUF build of Qwen3.5-4B through llama.cpp, on the CPU or, with `-gpu`, on its GPU backend. The 37 conditions of the transformers runs went with that backend; each of those runs keeps its own copy as `condition.toml`. |
| [conditions/rewrite/holdout/](privacy-research/conditions/rewrite/holdout/) | Current | The same two `q4km` conditions on the held-out passages, named `…-holdout`. The 28 transformers conditions went the same way. |
| [prompts/rewrite/](privacy-research/prompts/rewrite/) | Current | Rewrite prompt files, editable: each run copies the prompt it used into its `input.json`. `summary-baseline` is a generic summary prompt for a first comparison; `summary-v2` adds length, format, and faithfulness instructions; `summary-v3` also forbids making a promise stronger or weaker. |
| [prompts/rewrite/judge/](privacy-research/prompts/rewrite/judge/) | Current | Judge instructions, editable: each judgment copies the text it used. `keypoints` grades every rewrite against the key points and lists unsupported claims, with quotes. |
| [fits/classification/embeddinggemma2-opp115-heads.toml](privacy-research/fits/classification/embeddinggemma2-opp115-heads.toml) | Current | The `centroid`, `linear-balanced`, and `linear-frequencies` heads on EmbeddingGemma 2, learned from the OPP-115 examples. |
| [fits/classification/embeddinggemma2-appsi139-prior-shift.toml](privacy-research/fits/classification/embeddinggemma2-appsi139-prior-shift.toml) | Current | Shifts both linear heads toward the APPSI-139 tuning half's category mix. |
| [comparisons/](privacy-research/comparisons/) | Current | Each file backs a table or finding in the research README: `pilot-embeddinggemma2-heads` and `pilot-models-embeddinggemma2` (OPP-115 pilot), `appsi139-random-1000`, `appsi139-balanced-50`, and `appsi139-all` (APPSI-139), `appsi139-test-half` and `appsi139-balanced-50-prior-shift` (prior shift), `appsi139-test-half-cpu` and `appsi139-balanced-50-cpu` (the tuned balanced head on the CPU, beside its GPU run and privacy-model), `appsi139-test-half-torch215` and `appsi139-balanced-50-torch215` (the same GPU and CPU runs repeated on torch 2.15.0, beside privacy-model), `appsi139-test-half-llama-cpp` and `appsi139-balanced-50-llama-cpp` (that head on the BF16 and Q8_0 GGUF builds of EmbeddingGemma 2 through llama.cpp, on the GPU and on the CPU, beside those torch runs and privacy-model), `appsi139-balanced-50-llama-cpp-macbook` (the Q8_0 file on a MacBook's CPU and GPU, beside the desktop's torch runs and privacy-model), and `embeddinggemma2-instructions` (instruction prefixes). `rewrite-summary-prompts` (baseline and v2), `rewrite-summary-v2`, `rewrite-summary-v3` (v2 and v3), and `rewrite-summary-holdout` (v2 and v3 on the held-out passages) compare the candidate rewriters, `rewrite-precision-v3` and `rewrite-precision-v3-holdout` compare six of them at NF4 with their earlier precisions, and `rewrite-keypoints-v3` and `rewrite-keypoints-v3-holdout` score all nine, each at its tier's precision, from judgments of the same names. `rewrite-llama-cpp-v3` and `rewrite-llama-cpp-v3-holdout` put Qwen3.5-4B's Q4_K_M GGUF file, run through llama.cpp on the CPU, beside its BF16 and NF4 runs on the GPU from torch 2.8.0, and their `-torch215` twins beside the NF4 run on the GPU repeated on torch 2.15.0. `rewrite-cpu-runtimes-v3` and `rewrite-cpu-runtimes-v3-fp32` put it beside the NF4 run on the CPU through bitsandbytes, computing in bf16 and in float32. Each file holds one NF4 run, because one comparison cannot hold two runs of the same model, precision, device, and prompt. `rewrite-keypoints-llama-cpp-v3` and `rewrite-keypoints-llama-cpp-v3-holdout` are judging sets, not comparisons: each names the six runs of Qwen3.5-4B that one judgment of the same name grades side by side (NF4 through bitsandbytes from torch 2.8.0 and from 2.15.0, and the Q4_K_M file through llama.cpp on the desktop's and a MacBook's CPU and GPU), and `pbench compare` refuses them for that same rule. Their `-desktop` twins score the desktop's three runs on torch 2.15.0 from that judgment, and their `-macbook` twins score the MacBook's two runs beside the NF4 run judged in October. No README result uses the rewrite comparisons yet; `docs/rewriting-findings.md` reports the key-point, precision, llama.cpp, and CPU runtime ones. |
| [comparisons/qwen-privacy-jev.toml](privacy-research/comparisons/qwen-privacy-jev.toml) | Historical | The imported runs, made under transformers 5.16.1. |

### Results: [results/](privacy-research/results/)

| Path | Status | Contents |
| --- | --- | --- |
| `runs/<task>/<condition>/<UTC start>/`, for example [runs/classification/](privacy-research/results/runs/classification/) | Frozen | One record per run: `condition.toml`, `run.json` (identity, provenance, status), `input.json`, `input-audit.json`, `warmup.json`, and `responses.jsonl`. The runs of the retired `prompted` classification strategy, of `jev-choice`, and of the transformers rewrite backend stay and still compare, but this checkout cannot make them again. |
| Runs started `20260918-*` and `20261002-*` | Historical | The six runs imported from the AirLLM workspace, each with its original `legacy-run.json`. Reruns under transformers 5.19.0 sit beside them. |
| [fits/classification/](privacy-research/results/fits/classification/)`<fit>/<UTC start>/` | Frozen | `fit.toml`, `fit.json` (provenance, cross-validation results, and the weights' hash), and `weights.safetensors`. |
| [judgments/rewrite/](privacy-research/results/judgments/rewrite/)`<name>/` | Frozen | One judgment of a comparison's runs: `judgment.json` (runs, checklist and instructions, judge, which letter is which run, and the hash of every prompt and reply), `prompts/`, and the judge's `replies/`. Frozen once `pbench judge record` marks it recorded. |
| `runs/**/evaluation/` | Generated | Scores from `pbench run` or `pbench evaluate`; rewrite runs add `review.md`, each passage above its rewrite. |
| `comparisons/<name>/` | Generated | `summary.csv` and `matrix-overview.png`/`.svg` from `pbench compare`, plus `side-by-side.md` for rewrite comparisons and `keypoints.png`/`.svg` for those with judgments. Seventeen are committed so readers can see results without running it: `appsi139-test-half`, `embeddinggemma2-instructions`, `pilot-embeddinggemma2-heads`, `pilot-models-embeddinggemma2`, `qwen-privacy-jev`, `rewrite-summary-prompts`, `rewrite-summary-v2`, `rewrite-summary-v3`, `rewrite-summary-holdout`, `rewrite-precision-v3`, `rewrite-precision-v3-holdout`, `rewrite-keypoints-v3`, `rewrite-keypoints-v3-holdout`, `rewrite-keypoints-llama-cpp-v3-desktop`, `rewrite-keypoints-llama-cpp-v3-holdout-desktop`, `rewrite-keypoints-llama-cpp-v3-macbook`, and `rewrite-keypoints-llama-cpp-v3-holdout-macbook`. |

### Project files

| File | Contents |
| --- | --- |
| [pyproject.toml](privacy-research/pyproject.toml), [uv.lock](privacy-research/uv.lock), [.python-version](privacy-research/.python-version) | Python 3.12 and every package pinned, transitive ones included; the default `cpu` group uses CPU torch and a prebuilt llama-cpp-python 0.3.36 wheel on Windows and Linux x64, pinned by hash, while the opt-in `cu132` group uses CUDA 13.2 builds. Every other platform builds llama.cpp from source; the broken upstream macOS wheels are kept as commented references. |
| [.gitattributes](privacy-research/.gitattributes) | LF line endings, and no line-ending conversion under `data/` and `results/`, whose bytes are hashed. |
| [.gitignore](privacy-research/.gitignore) | The environment, caches, and the generated `results/` folders. |
