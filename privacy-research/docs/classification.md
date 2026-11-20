# Classification benchmark

The classification task asks a model to put one privacy-policy sentence into one of ten categories. This page describes the data, the output contract, how runs are scored, and the strategies that exist. The [README](../README.md) covers running conditions and comparisons.

## Categories

| ID | Category |
| ---: | --- |
| 0 | Policy Change |
| 1 | Third Party Sharing/Collection |
| 2 | User Choice/Control |
| 3 | Data Retention |
| 4 | User Access Edit and Deletion |
| 5 | International and Specific Audiences |
| 6 | Data Security |
| 7 | Do Not Track |
| 8 | First Party Collection/Use |
| 9 | Other |

The source CSV, `data/datasets/opp115-sentences.csv`, keeps OPP-115's twelve annotation columns unchanged. The loader maps `Practice not covered`, `Privacy contact information`, and `Introductory/Generic` into **Other**; the nine specific categories keep their meaning. `Other` covers those three kinds of content, not uncertainty about which specific category applies.

## Data

- `data/datasets/opp115-sentences.csv`: 10,717 sentences. 10,690 have exactly one original label; 27 have two or three and are never sampled. 531 single-label rows repeat an earlier sentence's text, and 66 groups of identical text carry conflicting categories after mapping.
- `data/classification/samples/pilot-200-10cat.json`: the frozen pilot, 20 sentences per category drawn with seed 42. It pins the CSV by SHA-256, so never resave the CSV in Excel. Redrawing it with `make_manifest(dataset, per_class=20, seed=42)` reproduces the file exactly.
- `data/classification/gold/pilot-200-10cat-reviewed.csv`: the reviewed gold. It keeps every original label and adds 54 acceptable alternatives across 53 sentences. `pilot-200-10cat-review.jsonl` holds the reviewer's notes behind each change.
- `data/classification/prompts/definitions.json`: one short definition per category, used by every strategy.

The pilot is balanced, so its accuracy does not estimate accuracy on real policies. Prompts were chosen on the same 200 sentences, and the CSV has no policy IDs, so nothing here is policy-held-out.

### APPSI-139

APPSI-139 annotates sentences from app privacy policies, a different corpus from OPP-115's website policies, so it tests whether results carry over. `tools/prepare_appsi139.py` copied these files from the AirLLM workspace and checks the sources by SHA-256.

- `data/datasets/appsi139-sentences.csv`: 13,291 sentences, already in the ten categories. The AirLLM workspace's `prepare_appsi.py` mapped APPSI's topics onto them and left out sentences with no topic or with an Edit/Control, Permission Acquisition, or Cease Operation topic. It also left out one 515-token sentence, so every model here can take every sentence. 12,189 sentences have one label and 1,102 have two or three. This mapping never produces User Choice/Control, User Access Edit and Deletion, or Do Not Track, and its Other covers contact information only.
- `data/classification/gold/appsi139-astra.csv`: the accepted labels, which are every original label plus every label ChatGPT Astra marked when it relabeled the sentences. Astra was asked to keep the original labels and add others where a sentence was ambiguous or mislabeled. It added labels to 4,465 sentences, but it also dropped an original label on 3,595; these accepted labels keep every original label, as the instruction intended.
- `data/classification/samples/appsi139-all.json`: every sentence.
- `data/classification/samples/appsi139-random-1000.json`: 1,000 sentences drawn with seed 42, in the corpus's own category mix.
- `data/classification/samples/appsi139-balanced-50.json`: up to 50 sentences per category, drawn with seed 42 among sentences whose accepted labels include that category. The scarcest category draws first and each sentence is drawn for one category, its stratum. Only 10 sentences carry Do Not Track, so the sample has 460 sentences.
- `data/classification/samples/appsi139-tuning-half.json` and `appsi139-test-half.json`: APPSI-139 in two halves, for fits that tune on it. Its rows fall into 71 runs of rising source IDs. At every run boundary checked, a new policy starts, so each run appears to hold whole policies. The runs alternate between the halves in file order, starting with the tuning half: 7,233 tuning sentences in 36 runs and 6,058 test sentences in 35. A fit may tune only on a sample whose purpose is tuning, and its results are reported on the test half. `tools/split_appsi139.py` froze both, and a test redraws them exactly.

APPSI-139 has no policy IDs either, and 47 of its sentences repeat an OPP-115 sentence word for word.

## Output contract

Every strategy produces one JSON object per sentence:

```json
{"label_id": 9}
```

A chat model must write this itself. The parser accepts the bare object or the object inside one whole-response Markdown fence, and rejects anything else: prose, explanations, extra keys, duplicate keys, booleans, floats, strings, or IDs outside 0–9. A rejected response counts as incorrect and stays in the denominator; nothing is retried or repaired. Greedy decoding asks for this format; it is not grammar-constrained.

Strategies that do not generate text (a classifier head, a JEV decision head, embedding similarity) write their top-scoring category in the same format, so they are valid by construction.

## Scoring

- **Accuracy**: with a gold file, a prediction is correct when it matches any accepted label; without one, it must match one of the sentence's original labels.
- **Macro accuracy** (with a gold file): accepted-label accuracy per category, averaged with each category weighted equally. A stratified sample groups each sentence by the category it was drawn for; for the pilot that is its original label. Other samples group each sentence under each of its original labels.
- **Macro-F1**: computed against the original labels of the sentences that have exactly one, over all ten categories, with zero for undefined per-category values. That is every pilot sentence and 12,189 of APPSI-139's. The AirLLM workspace instead counted every unpredicted label of a multi-label sentence as a false negative, so its APPSI macro-F1 values are lower.
- **Valid outputs**: the share of responses that parsed. Invalid, missing, and failed responses count as incorrect and as false negatives for their gold category; they never create a false positive.

`pbench evaluate` and `pbench compare` always rescore from `responses.jsonl` with the current code, so a comparison never mixes scoring versions.

A comparison's `[scoring]` table takes `gold` and, optionally, `sample`. With a `sample`, every run is scored on that sample's sentences alone, and its costs average over those sentences too; each run must have a response for all of them. That puts runs made on all of APPSI-139 beside runs made on its test half.

## Strategies

| Strategy | Backend | What the model receives | Timing covers |
| --- | --- | --- | --- |
| `prompted` (retired) | chat model | the definitions prompt and the sentence as a JSON string, in one user turn | prompt processing and generation |
| `classifier-head` (retired) | sequence classifier | the raw sentence | the classifier forward pass |
| `jev-choice` (retired) | JEV judge | JEV's bare choice template with the ten categories and definitions as options A–J | the backbone forward pass and the decision head |
| `embedding-similarity` (retired) | sentence embedder | the sentence after the condition's `sentence_prefix`; each category's name and definition are embedded once, after its `category_prefix` | the embedding forward pass |
| `embedding-head` | sentence embedder, a GGUF file through llama.cpp | the sentence after the `prefix` its fit used; a fitted head from `results/fits/`, possibly prior-shifted, scores the categories (see [Fitted heads](#fitted-heads)) | the embedding forward pass; the head's 10-row product is not timed |

No strategy truncates an input: every input is checked against the condition's limit before the first model call, and one input over the limit stops the run.

Only `embedding-head` still runs. The other four strategies are retired, and `pbench run` refuses a condition that names one. The backends of `prompted` and `jev-choice`, the transformers chat backend and the JEV-9B judge with its LoRA adapter, left the code with bitsandbytes and peft. The backends of `classifier-head` and `embedding-similarity`, the sequence classifier that ran privacy-model and the sentence-transformers embedder, left with torch. Every recorded run stays and still compares: each strategy keeps the label and timing note that a comparison reads, and `build_prompt` in `tasks/classification/prompts.py` still defines the prompt that the `prompted` runs received. The [README](../README.md#history) names the last commit that runs each.

## Fitted heads

The heads were fitted once, with `pbench fit`. That command is retired with the torch embedder it used ([README](../README.md#history)), so this section says how the records under `results/fits/` were made; they still load.

A fit learned from labeled example sentences that no test sample contains. `data/classification/examples/opp115-examples.json` holds the 10,417 single-label OPP-115 sentences outside the pilot, minus the 73 whose text repeats a pilot or APPSI-139 sentence once case and spacing are ignored. `tools/make_opp115_examples.py` froze it, and a test redraws it exactly.

The `embedding-heads` fit embedded every example once with the torch build of EmbeddingGemma 2 in bf16, after the fit file's `prefix`, and fitted each head named in it. Every head is linear: a category's score is `weight @ embedding + bias` on the normalized embedding, and the highest score wins.

- `centroid`: each category's normalized mean example embedding, with zero bias, so the scores are cosine similarities. Nothing is trained.
- `logistic`: a multinomial logistic regression with `class_weight` `balanced` (every category weighs the same) or `none` (categories weigh as often as they occur among the examples). Stratified cross-validation on the examples, scored by balanced accuracy, chooses the inverse regularization strength C from `c_values`.

`fit.json` reports each head's cross-validated balanced accuracy, and for logistic heads the score at every C. These scores come from OPP-115 examples only, so they say nothing about the test samples.

A condition that uses a head names a GGUF `file` in its `[model]` table, and the strategy embeds with that file through llama.cpp. `id` and `revision` pin the repository that holds the file, so the condition also names the checkpoint the file was converted from, as `converted_from` and `converted_from_revision`. That must be the checkpoint the fit embedded with. The strategy accepts that checkpoint's bf16 fit for a `bf16` or `q8_0` file and refuses every other pair. The ten conditions under `conditions/classification/appsi139/` are of this kind, and the two named `limit2048` are the model service's defaults. With either file, on the GPU and on the CPU, their runs are not significantly less accurate than the torch runs or than privacy-model, so the head was not refit; the research README gives the numbers from `appsi139-test-half-llama-cpp` and `appsi139-balanced-50-llama-cpp`.

The recorded runs of the torch checkpoint named that checkpoint itself, in bf16 on the GPU or in fp32 on the CPU. That backend is retired, and the strategy refuses a condition that names no `file`.

A condition takes `device = "cpu"` or `"gpu"`. The backend sizes the llama.cpp context for `max_input_tokens`, so that no input is ever truncated. A load takes memory for that limit whether or not an input reaches it: on the desktop's CPU it added 2.8 GiB at 8,192 tokens, 0.7 GiB at 2,048, and 0.5 GiB at 512, measured by hand. A condition for a small computer should name the longest input it must accept, not the model's limit; the model service's `limit2048` conditions do. The backend drops the score array that llama-cpp-python allocates for sampling, because embedding never reads it and Windows counts all of it as committed memory, 1 MiB per token of the limit. With that array a load at 8,192 tokens committed 11.5 GiB.

### Prior shift

A logistic head expects categories in the mix it was trained on: an even mix with balanced weights, otherwise the examples' own mix. A `prior-shift` fit moves logistic heads of an `embedding-heads` fit, its base, toward the mix of a tuning sample. A logistic head's scores are log-odds, so a new category mix adds `log target share - log training share` to each category's bias. A head's `strength` scales that shift: 0 leaves the base head as it was, and 1 assumes the tuning mix in full.

- The target shares come from the tuning sample's accepted labels. Each sentence counts once, split evenly among its labels, and every category needs at least one sentence.
- A shifted head keeps its base head's weights. A fixed `strength` is used as given. A `tuned` one is the value in the fit file's `strengths` with the highest accepted-label accuracy on the tuning sentences; the smallest value wins a tie.
- The fit embedded the tuning sentences once, with the base fit's model and prefix, and `fit.json` reports accuracy and macro accuracy at every strength.
- A condition names a `prior-shift` fit like any other. The strategy then loads the base fit for the model and prefix, and refuses it unless its hashes match the ones the prior-shift fit recorded.
