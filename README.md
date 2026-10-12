# Privacy-policy reader: model research

Experiments and measurements for an offline privacy-policy reading app. The app itself will live in a separate repository; this one decides which models it should use and keeps the evidence.

## The app being researched

A desktop reader that helps people understand privacy policies, **without any internet connection** for installation, first launch, or use:

1. The user pastes a policy (later: imports a file).
2. Each sentence is classified into one of ten categories, and the app can highlight them.
3. Three rewrite actions explain a passage: **Paraphrase**, **Simplify for Kids**, and **Simplify for Adults**. The original text stays visible, and each rewrite links back to its source sentences.

The provisional hardware target is a desktop or laptop with **8–12 GB of total system memory**. All measurements so far used one Windows PC with an **RTX 4090 (24 GiB) and 64 GB RAM**. The design notes are in [references/](references/); they predate this repository, and their links to an ideas index, the thesis chapters, and `airllm-tests/` point outside it.

## Current direction

| | **A. Classifier + separate rewriter** | **B. One base model with adapters** |
| --- | --- | --- |
| Shape | A small model labels sentences; a separate generative model rewrites. | One backbone; a task adapter classifies, and the same backbone rewrites. |
| Tested so far | `skythrone/privacy-model`, and EmbeddingGemma 2 without training. No rewriter tested. | `autotrust/JEV-9B`: a Qwen3.5-9B base with an unmerged LoRA judge adapter and a decision head. Classification only. |
| Open question | The small classifiers tested so far are much less accurate (below). | Whether the base model, with the adapter disabled, rewrites well enough. |

Other working decisions: hardware tiers (roughly 2–4, 8–12, and 16+ GB of RAM) may use different models; AirLLM layer streaming is tentative because it is very slow; and the app classifies one category per sentence for now, while the method should be able to return several later.

## Results so far

All rows use the frozen **200-sentence pilot** (20 per category, from OPP-115) and the **reviewed gold labels**, where a prediction is correct if it matches any label a reviewer accepted. Every output was valid. Runs used transformers 5.19.0; time is mean seconds per sentence and memory is peak PyTorch-allocated CUDA memory. Source: [comparisons/pilot-models-embeddinggemma2.toml](comparisons/pilot-models-embeddinggemma2.toml).

| Model | Strategy | Precision | Accuracy | Macro-F1¹ | Sec/sentence | Peak GPU |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-14B | prompted, definitions | INT8 | 0.890 | 0.770 | 1.22 | 15.7 GiB |
| Qwen3-14B | prompted, definitions | NF4 | 0.885 | 0.759 | 0.49 | 10.2 GiB |
| JEV-9B | choice head, definitions | BF16 | 0.860 | 0.758 | 0.11 | 16.9 GiB |
| JEV-9B | choice head, definitions | INT8 | 0.860 | 0.758 | 0.23 | 10.5 GiB |
| JEV-9B | choice head, definitions | NF4 | 0.855 | 0.748 | 0.12 | 7.7 GiB |
| skythrone/privacy-model | classifier head, raw sentence | FP32 | 0.770 | 0.640 | 0.003 | 0.26 GiB |
| EmbeddingGemma 2 | similarity to definitions, no training | BF16 | 0.575 | 0.502 | 0.041 | 0.54 GiB |
| EmbeddingGemma 2 | similarity to definitions, no training | FP32 | 0.575 | 0.498 | 0.033 | 1.05 GiB |

¹ Macro-F1 is scored against the original single labels; accuracy uses the reviewed labels.

**Caveats.** The pilot is balanced, so its accuracy does not estimate accuracy on real policies. Prompts were chosen on the same 200 sentences, and the dataset has no policy IDs, so nothing is policy-held-out. A proper held-out evaluation is still to do.

Findings:

- **EmbeddingGemma 2 without training is far behind.** Its errors are near misses (First Party sentences go to Third Party, User Choice to Do Not Track, and Other scatters), so the natural next test for option A is a small trained classification head on its embeddings.
- **Library versions matter for JEV-9B.** Moving from transformers 5.16.1 to 5.19.0 made JEV 1.6–2.3× faster and shifted its logits slightly; two INT8 near-ties flipped in opposite directions. Qwen3-14B and privacy-model were unaffected. The original JEV runs are kept for comparison in [comparisons/qwen-privacy-jev.toml](comparisons/qwen-privacy-jev.toml).
- **Quantization barely changes accuracy.** Qwen3-14B: 0.890 INT8, 0.885 NF4 (0.900 BF16 with CPU offload, in the AirLLM archive). JEV-9B: 0.860, 0.860, 0.855. NF4 roughly halves memory.
- **Rewriting has not been evaluated at all.** Every measurement here is classification.

Earlier findings kept in the AirLLM archive (`C:\Projects\AirLLM`), from runners not carried over:

- **Category definitions in the prompt help; extra guidance does not.** Qwen3-14B INT8 scored 0.815 with labels only, 0.890 with definitions, and 0.880 with definitions plus guidance.
- **Other models on the same pilot:** a frontier chat model (batch, external) 0.970; Qwen3.8-35B-A3B through llama.cpp, Q8_0, 0.890; phi-4 INT8 0.865; Saul-7B-Instruct INT8 0.820.
- **privacy-model beyond the pilot:** 0.731 accuracy on all 10,717 OPP-115 rows (it may have been trained on them), and 0.722 or 0.793 on APPSI-139 against the original or relabeled categories.
- **Stock AirLLM layer streaming is far too slow for interactive use:** Qwen3-14B took about 394 s for 32 tokens, against about 1.7 s fully on the GPU in 4-bit.
- **Large idle GPU allocations lag the Windows desktop.** Holding 16 GiB on the display GPU caused severe lag; 8 and 12 GiB did not. The app's memory budget needs to account for this.
- **Reasoning output breaks the label format.** Qwen3.8 with thinking tags produced 0 valid answers out of 200.

## Repository layout

```
conditions/<task>/*.toml     one file per run condition: task, strategy, items, model, options
comparisons/*.toml           which saved runs to put side by side, and how to score them
data/                        source datasets, plus each task's samples, gold labels, and prompts
results/runs/<task>/<condition>/<UTC start>/   tracked run records
results/comparisons/<name>/  summaries and plots, regenerated and not tracked
src/privacy_bench/
  core/       condition files, run records, the run loop, provenance
  models/     backends: chat generation, sequence classifiers, JEV judges, sentence embeddings
  tasks/      one package per task; classification today
  report/     task-agnostic summaries and plots
  cli.py      the pbench command
tools/        one-time migration code
references/   design notes for the app
```

A **model backend** loads a pinned checkpoint and runs one input without knowing what the output means. A **task** owns its items, reference answers, evaluation, and how its runs appear in comparisons. A **strategy** is one way to perform a task with one backend. `tests/test_architecture.py` enforces the layering: `core`, `models`, and `report` never import a task, and a task never imports another task.

## Setup

[uv](https://docs.astral.sh/uv/) manages the environment; every package, including transitive ones, is pinned in `pyproject.toml` and `uv.lock`.

```powershell
uv sync
uv run pytest            # offline; some tests build tiny models on the CPU
```

Models resolve from the standard Hugging Face cache (`~/.cache/huggingface/hub`), and every condition pins a full commit. Runs never download, so fetch a condition's model once while online:

```powershell
uv run pbench download conditions/classification/embeddinggemma2-bf16-definitions.toml
```

## Running and comparing

```powershell
uv run pbench run conditions/classification/embeddinggemma2-bf16-definitions.toml
uv run pbench compare comparisons/pilot-models-embeddinggemma2.toml
uv run pbench evaluate results/runs/classification/<condition>/<UTC start>
```

- `run` checks every input before the first model call and never truncates one. It refuses an existing output folder, keeps every completed response if the run fails or is interrupted, and records the git commit and any uncommitted diff, the full package list, the resolved model commit, and the exact input specification.
- `compare` rescores each listed run with the current code, refuses runs scored on different items or reference answers, and writes `summary.csv`, `matrix-overview.png`, and `matrix-overview.svg` under `results/comparisons/<name>/`.
- `evaluate` rewrites a run's `evaluation/` folder from its saved responses.

## Extending

- **Another model for an existing strategy:** add a condition file with the model's pinned commit, then `pbench download` and `pbench run`.
- **A new strategy:** add a class under `tasks/<task>/strategies/` and register it in that package's `STRATEGIES`. Reuse a backend from `models/`, or add one there that knows nothing about the task.
- **A new task**, such as simplification: add `tasks/<name>/` with a task object that selects items, builds strategies, evaluates responses, and supplies its comparison panels and notes, then register it in `tasks/__init__.py`. The run loop, run records, model backends, and cost panels apply unchanged.

The classification task is described in [docs/classification.md](docs/classification.md).

## History

The benchmark began in the `C:\Projects\AirLLM` workspace, which stays as an archive of its original runs, logs, and AirLLM experiments. The six runs behind its last comparison plot were imported into `results/runs/` with [tools/import_legacy_runs.py](tools/import_legacy_runs.py): each import checked its condition, data, and per-sentence input hashes, and reproduced the original metrics exactly. Rerunning those conditions with this code reproduced every prediction: Qwen3-14B token for token although it ran under a newer transformers, and privacy-model and JEV-9B with bit-identical logits under the original libraries.
