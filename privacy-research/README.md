# Privacy Policy Reader: model research

Experiments and measurements that decide which models the [Privacy Policy Reader](../README.md) should use, and the evidence behind each choice. The app needs two: a classifier that puts each sentence of a policy into one of ten categories, and a rewriter for its three actions (**Paraphrase**, **Simplify for Kids**, and **Simplify for Adults**). Both must run offline on a computer with **8–12 GB of total memory**. This work answers analysis requirements AR-01 and AR-02 in the [requirements](../docs/Requirements-Document.md#6-analysis-requirements).

All measurements so far used one Windows PC with an **RTX 4090 (24 GiB) and 64 GB RAM**.

## Current direction

| | **A. Classifier + separate rewriter** | **B. One base model with adapters** |
| --- | --- | --- |
| Shape | A small model labels sentences; a separate generative model rewrites. | One backbone; a task adapter classifies, and the same backbone rewrites. |
| Tested so far | `skythrone/privacy-model`, and EmbeddingGemma 2 with and without a fitted head. No rewriter tested. | `autotrust/JEV-9B`: a Qwen3.5-9B base with an unmerged LoRA judge adapter and a decision head. Classification only. |
| Open question | A fitted head on EmbeddingGemma 2 ties the large models when every category counts equally. Shifted toward APPSI-139's category mix, it ties privacy-model on that mix instead, but no head yet leads on both (below). | Whether the base model, with the adapter disabled, rewrites well enough. |

Other working decisions: hardware tiers (roughly 2–4, 8–12, and 16+ GB of RAM) may use different models; AirLLM layer streaming is tentative because it is very slow; and the app classifies one category per sentence for now, while the method should be able to return several later.

## Results so far

### OPP-115 pilot

All rows use the frozen **200-sentence pilot** (20 per category, from OPP-115) and the **reviewed gold labels**, where a prediction is correct if it matches any label a reviewer accepted. Every output was valid. Runs used transformers 5.19.0; time is mean seconds per sentence and memory is peak PyTorch-allocated CUDA memory. Source: [comparisons/pilot-embeddinggemma2-heads.toml](comparisons/pilot-embeddinggemma2-heads.toml); the last row comes from [comparisons/pilot-models-embeddinggemma2.toml](comparisons/pilot-models-embeddinggemma2.toml).

| Model | Strategy | Precision | Accuracy | Macro-F1¹ | Sec/sentence | Peak GPU |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| EmbeddingGemma 2 | fitted head: linear, classes weighted equally | BF16 | 0.910 | 0.822 | 0.041 | 0.54 GiB |
| Qwen3-14B | prompted, definitions | INT8 | 0.890 | 0.770 | 1.22 | 15.7 GiB |
| Qwen3-14B | prompted, definitions | NF4 | 0.885 | 0.759 | 0.49 | 10.2 GiB |
| EmbeddingGemma 2 | fitted head: linear, OPP-115 class frequencies | BF16 | 0.875 | 0.778 | 0.040 | 0.54 GiB |
| JEV-9B | choice head, definitions | BF16 | 0.860 | 0.758 | 0.11 | 16.9 GiB |
| JEV-9B | choice head, definitions | INT8 | 0.860 | 0.758 | 0.23 | 10.5 GiB |
| JEV-9B | choice head, definitions | NF4 | 0.855 | 0.748 | 0.12 | 7.7 GiB |
| EmbeddingGemma 2 | category averages of examples, nothing trained | BF16 | 0.855 | 0.757 | 0.042 | 0.54 GiB |
| skythrone/privacy-model | classifier head, raw sentence | FP32 | 0.770 | 0.640 | 0.003 | 0.26 GiB |
| EmbeddingGemma 2 | similarity to definitions, best instruction | BF16 | 0.655 | 0.575 | 0.042 | 0.54 GiB |
| EmbeddingGemma 2 | similarity to definitions | BF16 | 0.575 | 0.502 | 0.041 | 0.54 GiB |

¹ Macro-F1 is scored against the original single labels; accuracy uses the reviewed labels.

The three EmbeddingGemma 2 heads come from one fit on 10,417 OPP-115 sentences outside the pilot ([results/fits/classification/embeddinggemma2-opp115-heads/](results/fits/classification/embeddinggemma2-opp115-heads/)). Those sentences may come from the same policies as pilot sentences; in a scratch test, holding out whole stretches of the corpus around each pilot sentence cost the balanced head about 1.5 points.

**Caveats.** The pilot is balanced, so its accuracy does not estimate accuracy on real policies. Prompts were chosen on the same 200 sentences, and the dataset has no policy IDs, so nothing in the pilot is policy-held-out. APPSI-139, below, is the test on a different corpus.

### APPSI-139

APPSI-139 annotates sentences from app privacy policies, which no model here was tuned on, except possibly privacy-model, whose training data is not published, and the shifted heads [below](#shifting-the-heads-toward-appsi-139s-category-mix), which tuned on one half of it. A prediction is correct if it matches any **accepted label**: the original mapped labels plus every label ChatGPT Astra marked in a relabeling ([details](docs/classification.md#appsi-139)). The balanced sample takes up to 50 sentences per category (460 in all, as only 10 carry Do Not Track); its macro accuracy, in brackets, weighs each category equally. Sources: [comparisons/appsi139-random-1000.toml](comparisons/appsi139-random-1000.toml), [comparisons/appsi139-balanced-50.toml](comparisons/appsi139-balanced-50.toml), and [comparisons/appsi139-all.toml](comparisons/appsi139-all.toml).

| Model | Strategy | Precision | 1,000 random | 460 balanced (macro) | All 13,291 |
| --- | --- | --- | ---: | ---: | ---: |
| Qwen3-14B | prompted, definitions | INT8 | 0.863 | 0.885 (0.894) | |
| Qwen3-14B | prompted, definitions | NF4 | 0.869 | 0.883 (0.892) | |
| JEV-9B | choice head, definitions | BF16 | 0.844 | 0.885 (0.894) | |
| JEV-9B | choice head, definitions | INT8 | 0.849 | 0.891 (0.900) | |
| JEV-9B | choice head, definitions | NF4 | 0.841 | 0.887 (0.896) | |
| skythrone/privacy-model | classifier head, raw sentence | FP32 | 0.836 | 0.824 (0.790) | 0.860 |
| EmbeddingGemma 2 | fitted head: linear, OPP-115 class frequencies | BF16 | 0.814 | 0.807 (0.798) | 0.825 |
| EmbeddingGemma 2 | fitted head: linear, classes weighted equally | BF16 | 0.803 | 0.867 (0.878) | 0.812 |
| EmbeddingGemma 2 | category averages of examples, nothing trained | BF16 | 0.755 | 0.815 (0.830) | 0.774 |
| EmbeddingGemma 2 | similarity to definitions, best instruction | BF16 | 0.487 | 0.641 (0.670) | |

Against the original mapped labels alone, accuracy on the 1,000 random sentences is 0.65–0.71 (0.39 for the similarity run), and Qwen3-14B (0.694–0.699) falls just behind JEV-9B (0.713–0.714) and privacy-model (0.712). Speed and memory match the pilot; EmbeddingGemma 2 peaks at 0.57 GiB on the longest test-half sentence (460 tokens). Its runs on all 13,291 sentences report up to 1.33 GiB, but a leak caused that, not the sentences: each sentence's outputs stayed on the GPU until the run ended. The leak is fixed, and it never changed a prediction.

Findings:

- **A linear head on frozen EmbeddingGemma 2 embeddings ties the 14B chat model when every category counts equally,** at 0.04 s per sentence and 0.5–0.6 GiB. On the pilot it scored 0.910 against Qwen3-14B INT8's 0.890, and on the balanced APPSI-139 sample 0.867 against 0.883–0.891 for Qwen3-14B and JEV-9B; neither gap is beyond chance (exact McNemar p = 0.54, 0.42, 0.22). It beat privacy-model on both (p < 0.001 and p = 0.04).
- **On APPSI-139's natural mix, privacy-model and the large models lead.** Across all 13,291 sentences privacy-model scored 0.860 against 0.825 for the best head (p < 10⁻²⁰). Its lead comes from First Party Collection/Use, almost half of the corpus, where it scored 0.908 against 0.77–0.83, and from Other (0.943 against 0.79–0.81); the heads were ahead in every other category. On 1,000 random sentences Qwen3-14B NF4 scored 0.869, ahead of JEV-9B NF4's 0.841 (p = 0.009). Shifting a head toward APPSI-139's mix closes the First Party gap ([below](#shifting-the-heads-toward-appsi-139s-category-mix)).
- **Class weighting trades the balanced score against the natural one.** Weighting categories equally in training gave the best balanced scores; weighting them by OPP-115 frequency gave the best natural-mix scores. The category-averages head, which trains nothing, stays 5–6 points behind the better linear head everywhere.
- **EmbeddingGemma 2 without a head is far behind.** Matching sentences against one definition per category scored 0.575–0.655 on the pilot and 0.487 on random APPSI-139 sentences; its errors are near misses between categories that share a topic.
- **The instruction prefix helps EmbeddingGemma 2 only a little.** Against 0.575 with its classification instruction, sentence similarity scores 0.610 and search mode 0.605 with the sentence as the query or 0.655 with each category description as the query. Only the 0.655 gain is beyond chance on 200 sentences, and it was picked from three tries on the same pilot. Each instruction mostly changes which categories collect too many sentences; User Choice sentences go to Do Not Track under all four. Source: [comparisons/embeddinggemma2-instructions.toml](comparisons/embeddinggemma2-instructions.toml).
- **Library versions matter for JEV-9B.** Moving from transformers 5.16.1 to 5.19.0 made JEV 1.6–2.3× faster and shifted its logits slightly; two INT8 near-ties flipped in opposite directions. Qwen3-14B and privacy-model were unaffected. The original JEV runs are kept for comparison in [comparisons/qwen-privacy-jev.toml](comparisons/qwen-privacy-jev.toml).
- **Quantization barely changes accuracy.** Qwen3-14B: 0.890 INT8, 0.885 NF4 (0.900 BF16 with CPU offload, in the AirLLM archive). JEV-9B: 0.860, 0.860, 0.855. NF4 roughly halves memory.
- **Rewriting has not been evaluated at all.** Every measurement here is classification.

### Shifting the heads toward APPSI-139's category mix

A logistic head expects the category mix it trained on: an even mix for the balanced head, OPP-115's for the frequency head. A prior shift adds `strength × (log target share − log training share)` to each category's bias ([details](docs/classification.md#prior-shift)). APPSI-139 is split into a tuning half (7,233 sentences) and a test half (6,058). Each run of rising source IDs, which appears to hold one or more whole policies, stays in one half. The tuning half alone set the target mix (First Party Collection/Use 0.42, Other 0.11) and chose the strength from 0, 0.1, …, 1 by accuracy. Every number below comes from the test half or the balanced sample. Sources: [comparisons/appsi139-test-half.toml](comparisons/appsi139-test-half.toml) and [comparisons/appsi139-balanced-50-prior-shift.toml](comparisons/appsi139-balanced-50-prior-shift.toml).

| Model | Strategy | Shift strength | Test half (macro²) | 460 balanced (macro) |
| --- | --- | ---: | ---: | ---: |
| skythrone/privacy-model | classifier head, raw sentence | | 0.858 (0.835) | 0.824 (0.790) |
| EmbeddingGemma 2 | fitted head: linear, classes weighted equally | 0 | 0.816 (0.858) | 0.867 (0.878) |
| EmbeddingGemma 2 | fitted head: linear, classes weighted equally | 0.9, tuned | 0.858 (0.850) | 0.817 (0.792) |
| EmbeddingGemma 2 | fitted head: linear, classes weighted equally | 1 | 0.856 (0.845) | 0.811 (0.786) |
| EmbeddingGemma 2 | fitted head: linear, OPP-115 class frequencies | 0 | 0.832 (0.834) | 0.807 (0.798) |
| EmbeddingGemma 2 | fitted head: linear, OPP-115 class frequencies | 1, tuned | 0.850 (0.845) | 0.826 (0.808) |

² Over the seven categories that APPSI-139's original labels use.

- **The shift closes the natural-mix gap to privacy-model.** On the test half the tuned balanced head scored 0.8575 against privacy-model's 0.8584 (exact McNemar p = 0.90), up from 0.816 unshifted (p < 10⁻²¹). First Party Collection/Use rose from 0.771 to 0.895, against privacy-model's 0.909.
- **Other stays behind.** It rose only from 0.803 to 0.814, against privacy-model's 0.944. APPSI-139's Other holds contact information only, while OPP-115's also holds introductions and practices the categories do not cover, so this gap is about what Other means, not how common it is.
- **The shift gives up the balanced lead.** On the balanced sample the tuned balanced head fell from 0.867 to 0.817 (p = 0.003), level with privacy-model (p = 0.82) and behind Qwen3-14B and JEV-9B (p ≤ 0.001). Rarer categories lose: Policy Change fell from 0.92 to 0.76, Data Retention from 0.92 to 0.82, and Do Not Track from 10 of 10 to 5 of 10, while First Party Collection/Use rose from 0.78 to 0.90. No head here leads on both mixes. On the tuning half, macro accuracy peaked near strength 0.3–0.4, a possible compromise that the test half has not checked.
- **Strengths near 1 differ little.** Strength 0.9 and 1 disagree on 65 test-half sentences, split 36 to 29 (p = 0.46). The frequency head also tuned to full strength and finished at 0.850, behind the balanced head (p = 0.03).

Earlier findings kept in the AirLLM archive (`C:\Projects\AirLLM`), from runners not carried over:

- **Category definitions in the prompt help; extra guidance does not.** Qwen3-14B INT8 scored 0.815 with labels only, 0.890 with definitions, and 0.880 with definitions plus guidance.
- **Other models on the same pilot:** a frontier chat model (batch, external) 0.970; Qwen3.8-35B-A3B through llama.cpp, Q8_0, 0.890; phi-4 INT8 0.865; Saul-7B-Instruct INT8 0.820.
- **privacy-model beyond the pilot:** 0.731 accuracy on all 10,717 OPP-115 rows (it may have been trained on them), and 0.722 or 0.793 on APPSI-139 against the original or relabeled categories. The relabeled score used Astra's labels as they are; this repository's accepted labels also keep every original label, which gives 0.860.
- **Stock AirLLM layer streaming is far too slow for interactive use:** Qwen3-14B took about 394 s for 32 tokens, against about 1.7 s fully on the GPU in 4-bit.
- **Large idle GPU allocations lag the Windows desktop.** Holding 16 GiB on the display GPU caused severe lag; 8 and 12 GiB did not. The app's memory budget needs to account for this.
- **Reasoning output breaks the label format.** Qwen3.8 with thinking tags produced 0 valid answers out of 200.

## Repository layout

```
conditions/<task>/*.toml     one file per run condition: task, strategy, items, model, options
conditions/<task>/<corpus>/  conditions for another test corpus, such as appsi139/
fits/<task>/*.toml           one file per fit: what to learn from, the model, and the heads to fit
comparisons/*.toml           which saved runs to put side by side, and how to score them
prompts/<task>/*.toml        editable prompt files; each run copies the text it used into its input.json
data/                        source datasets, plus each task's samples, passages, gold labels, examples, and prompts
results/runs/<task>/<condition>/<UTC start>/   tracked run records
results/fits/<task>/<fit>/<UTC start>/         tracked fit records, with the fitted weights
results/comparisons/<name>/  summaries and plots from pbench compare; five are committed for reading
src/privacy_bench/
  core/       condition and fit files, run and fit records, the run loop, provenance
  models/     backends: chat generation, sequence classifiers, JEV judges, sentence embeddings
  tasks/      one package per task: classification and rewrite
  report/     task-agnostic summaries and plots
  cli.py      the pbench command
tools/        one-time code that imported runs and froze data files
docs/         each task in detail; classification.md today
```

A **model backend** loads a pinned checkpoint and runs one input without knowing what the output means. A **task** owns its items, reference answers, evaluation, and how its runs appear in comparisons. A **strategy** is one way to perform a task with one backend. `tests/test_architecture.py` enforces the layering: `core`, `models`, and `report` never import a task, and a task never imports another task.

## Setup

[uv](https://docs.astral.sh/uv/) manages the environment; every package, including transitive ones, is pinned in `pyproject.toml` and `uv.lock`. Run every command below from this folder. The tests run offline and build tiny models on the CPU.

Choose one of the two setups.

**With an NVIDIA GPU.** This installs the CUDA 12.8 build of torch, which every run in `results/` used:

```powershell
uv sync
uv run pytest
```

**Without an NVIDIA GPU.** This installs the much smaller CPU build of torch. The tests pass on it, but runs with `device = "cuda"` will not start:

```powershell
$env:UV_NO_SYNC = "1"    # keep `uv run` from installing the CUDA build; set it in your profile
uv sync --no-group cu128 --group cpu
uv run pytest
```

Set `UV_NO_SYNC` before any other `uv` command. After pulling a change to `uv.lock`, rerun the `uv sync` line.

Models resolve from the standard Hugging Face cache (`~/.cache/huggingface/hub`), and every condition pins a full commit. Runs never download, so fetch a condition's model once while online:

```powershell
uv run pbench download conditions/classification/embeddinggemma2-bf16-definitions.toml
```

## Running and comparing

Every task uses the same three commands. A condition file under `conditions/<task>/` describes one run, and a comparison file under `comparisons/` lists saved runs to put side by side:

```powershell
uv run pbench run conditions/<task>/<condition>.toml
uv run pbench compare comparisons/<name>.toml
uv run pbench evaluate results/runs/<task>/<condition>/<UTC start>
```

- `run` checks every input before the first model call and never truncates one. It refuses an existing output folder, keeps every completed response if the run fails or is interrupted, and records the git commit and any uncommitted diff, the full package list, the resolved model commit, and the exact input specification.
- `compare` rescores each listed run with the current code, refuses runs scored on different items or reference answers, and writes `summary.csv`, `matrix-overview.png`, and `matrix-overview.svg` under `results/comparisons/<name>/`.
- `evaluate` rewrites a run's `evaluation/` folder from its saved responses.

A comparison file names its `task` and its `runs`, as paths under `results/runs/`. An optional `[labels]` table replaces legend entries, and `[scoring]` holds settings that belong to the task. `group_by = "model"` groups the plot by model, with one bar per precision, to compare precisions under one condition.

### Classification

```powershell
uv run pbench run conditions/classification/embeddinggemma2-bf16-definitions.toml
uv run pbench compare comparisons/pilot-models-embeddinggemma2.toml
```

A classification comparison's `[scoring]` names the `gold` labels to score against and, optionally, a smaller `sample` that every run covers.

Some classification strategies apply weights fitted to labeled examples, such as a small classification head on frozen embeddings. Fit them once, before the runs that use them:

```powershell
uv run pbench fit fits/classification/embeddinggemma2-opp115-heads.toml
```

`fit` checks every example against the input limit before embedding any, refuses an existing output folder, and writes `fit.toml`, `fit.json`, and `weights.safetensors` under `results/fits/<task>/<fit>/<UTC start>/`. `fit.json` records the same provenance as a run, the cross-validation results, and the hash of the weights, which a run checks before it loads them.

[docs/classification.md](docs/classification.md) describes the task's data, scoring, and fits in detail.

### Rewrite

The rewrite task runs on 20 frozen APPSI-139 policy sections ([tools/make_appsi139_sections.py](tools/make_appsi139_sections.py) says how they were drawn). A second, held-out 20 from other policies checks prompts that were tuned by reading rewrites of the first 20; their conditions are in `conditions/rewrite/holdout/`. Each condition names a prompt file under `prompts/rewrite/`: an optional `system` message and a `user` template with one `{passage}` placeholder. Nothing is fitted.

```powershell
uv run pbench run conditions/rewrite/qwen3.5-4b-bf16-summary-baseline.toml
```

No reference rewrite exists, so the scores only flag outputs to check: whether a rewrite is valid (finished, nonempty, no reasoning tags), how many of the source's numbers and qualifiers (*may*, *unless*, *not*, …) it still contains, and whether it adds an exception, *only*, or a negation that the passage lacks. Three more describe how a rewrite is written: how much it copies from the passage (the copied-run length of Grusky et al., 2018), its Flesch–Kincaid reading grade beside the passages' own, and how often it opens with an introduction such as "Here's the summary:" or uses Markdown. Each run writes `evaluation/review.md`, which shows each passage above its rewrite. A rewrite comparison takes no `[scoring]`, and `compare` adds `side-by-side.md`, which shows every compared model's rewrite under each passage:

```toml
task = "rewrite"
runs = [
    "rewrite/qwen3.5-4b-bf16-summary-baseline/<UTC start>",
    "rewrite/gemma-4-e4b-int8-summary-baseline/<UTC start>",
]
```

## Extending

- **Another model for an existing strategy:** add a condition file with the model's pinned commit, then `pbench download` and `pbench run`.
- **Another rewrite prompt:** add a file under `prompts/rewrite/` and conditions that name it. Editing a prompt in place is fine while iterating; copy it under a new name once a variant is worth comparing, because comparisons label runs by prompt name.
- **A new strategy:** add a class under `tasks/<task>/strategies/` and register it in that package's `STRATEGIES`. Reuse a backend from `models/`, or add one there that knows nothing about the task.
- **A new kind of fit:** add a fitter to the task's `FITTERS` that turns a fit file into a report and named float32 tensors. The shared fit record stores and checks them.
- **A new task:** add `tasks/<name>/` with a task object that selects items, builds strategies, evaluates responses, and supplies its comparison panels, notes, and any extra comparison files, then register it in `tasks/__init__.py`. The run loop, run records, model backends, and cost panels apply unchanged.

## History

The benchmark began in the `C:\Projects\AirLLM` workspace, which stays as an archive of its original runs, logs, and AirLLM experiments. The six runs behind its last comparison plot were imported into `results/runs/` with [tools/import_legacy_runs.py](tools/import_legacy_runs.py): each import checked its condition, data, and per-sentence input hashes, and reproduced the original metrics exactly. Rerunning those conditions with this code reproduced every prediction: Qwen3-14B token for token although it ran under a newer transformers, and privacy-model and JEV-9B with bit-identical logits under the original libraries.
