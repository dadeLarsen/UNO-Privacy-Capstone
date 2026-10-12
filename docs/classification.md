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

## Strategies

| Strategy | Backend | What the model receives | Timing covers |
| --- | --- | --- | --- |
| `prompted` | chat model | the definitions prompt and the sentence as a JSON string, in one user turn | prompt processing and generation |
| `classifier-head` | sequence classifier | the raw sentence | the classifier forward pass |
| `jev-choice` | JEV judge | JEV's bare choice template with the ten categories and definitions as options A–J | the backbone forward pass and the decision head |
| `embedding-similarity` | sentence embedder | the sentence after the condition's `sentence_prefix`; each category's name and definition are embedded once, after its `category_prefix` | the embedding forward pass |
| `embedding-head` | sentence embedder | the sentence after the `prefix` its fit used; a fitted head from `results/fits/` scores the categories (see [Fitted heads](#fitted-heads)) | the embedding forward pass; the head's 10-row product is not timed |

No strategy truncates an input: every input is checked against the condition's limit before the first model call, and one input over the limit stops the run.

## Fitted heads

A fit learns from labeled example sentences that no test sample contains. `data/classification/examples/opp115-examples.json` holds the 10,417 single-label OPP-115 sentences outside the pilot, minus the 73 whose text repeats a pilot or APPSI-139 sentence once case and spacing are ignored. `tools/make_opp115_examples.py` froze it, and a test redraws it exactly.

The `embedding-heads` fit embeds every example once with a sentence embedder, after the fit file's `prefix`, and fits each head named in it. Every head is linear: a category's score is `weight @ embedding + bias` on the normalized embedding, and the highest score wins.

- `centroid`: each category's normalized mean example embedding, with zero bias, so the scores are cosine similarities. Nothing is trained.
- `logistic`: a multinomial logistic regression with `class_weight` `balanced` (every category weighs the same) or `none` (categories weigh as often as they occur among the examples). Stratified cross-validation on the examples, scored by balanced accuracy, chooses the inverse regularization strength C from `c_values`.

`fit.json` reports each head's cross-validated balanced accuracy, and for logistic heads the score at every C. These scores come from OPP-115 examples only, so they say nothing about the test samples.
