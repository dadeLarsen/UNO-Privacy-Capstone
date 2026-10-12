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

## Output contract

Every strategy produces one JSON object per sentence:

```json
{"label_id": 9}
```

A chat model must write this itself. The parser accepts the bare object or the object inside one whole-response Markdown fence, and rejects anything else: prose, explanations, extra keys, duplicate keys, booleans, floats, strings, or IDs outside 0–9. A rejected response counts as incorrect and stays in the denominator; nothing is retried or repaired. Greedy decoding asks for this format; it is not grammar-constrained.

Strategies that do not generate text (a classifier head, a JEV decision head, embedding similarity) write their top-scoring category in the same format, so they are valid by construction.

## Scoring

- **Accuracy**: with the reviewed gold, a prediction is correct when it matches any accepted label; without it, only the original label counts.
- **Macro accuracy** (reviewed gold): accepted-label accuracy averaged over the original categories, each weighted equally.
- **Macro-F1**: computed against the original single labels over all ten categories, with zero for undefined per-category values.
- **Valid outputs**: the share of responses that parsed. Invalid, missing, and failed responses count as incorrect and as false negatives for their gold category; they never create a false positive.

`pbench evaluate` and `pbench compare` always rescore from `responses.jsonl` with the current code, so a comparison never mixes scoring versions.

## Strategies

| Strategy | Backend | What the model receives | Timing covers |
| --- | --- | --- | --- |
| `prompted` | chat model | the definitions prompt and the sentence as a JSON string, in one user turn | prompt processing and generation |
| `classifier-head` | sequence classifier | the raw sentence | the classifier forward pass |
| `jev-choice` | JEV judge | JEV's bare choice template with the ten categories and definitions as options A–J | the backbone forward pass and the decision head |
| `embedding-similarity` | sentence embedder | the sentence after the condition's `sentence_prefix`; each category's name and definition are embedded once, after its `category_prefix` | the embedding forward pass |

No strategy truncates an input: every input is checked against the condition's limit before the first model call, and one input over the limit stops the run.
