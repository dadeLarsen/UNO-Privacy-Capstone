# Initial ideas: reading, rewriting, and category highlights

[Ideas index](README.md) · [Local architecture](local-architecture.md)

These are design possibilities for an app that requires **no internet connection for installation, first launch, or any action**; the feature details are proposals, not claims about an implemented system. The starting point is the [thesis interface](../docs/thesis-chapters-4-5/chapter-5/interface-and-actions.md): entered policy summaries, sentence classifications, a category legend, and four actions. The proposed app could retain those recognizable actions while making the original policy easier to read and compare with generated explanations.

## A document workspace with an explanation pane

The strongest starting direction is a reading workspace. A chat interface remains an option for later follow-up questions, but inspecting a policy benefits from a stable document, visible categories, and explanations beside their sources.

| Area | Initial idea |
| --- | --- |
| Top bar | Document title, **Running locally** status, and a compact model/settings control. |
| Left rail | Category names, matching-sentence counts, and filters. Collapse it on narrow windows. |
| Main pane | Original policy with paragraphs and headings preserved; optional sentence highlights. |
| Action bar | **Paraphrase**, **Simplify for Kids**, **Simplify for Adults**, and a **Highlight Categories** toggle. Show whether actions apply to a selection or the whole document. |
| Explanation pane | Generated rewrite, audience preset, links back to source passages, and copy/regenerate controls. On smaller screens, use Original/Explanation tabs. |
| Progress area | Concrete stages such as loading model, analyzing 12 of 40 sentences, or rewriting section 2 of 5, with Cancel. |

Start with pasted text and a clearly labeled example. Full-policy processing would be an intentional extension: the thesis describes summary entry and lists document uploads as future work. Text-file import could follow; PDF/DOCX parsing needs extraction previews because lost headings, columns, and footnotes can change meaning.

After pasting, an **Analyze Categories** control starts classification. Rewriting can be requested independently. Default rewrite scope to the selected passage when there is one, otherwise the entered text, and state that scope beside the buttons. Disable the highlighting toggle until the first validated labels arrive; during partial analysis, distinguish pending sentences from completed ones without labels.

Keep the original visible and unchanged. Switching a rewrite preset should create or select a separate result, and selecting an explanation should reveal its supporting source. A sentence inspector can show all assigned categories and allow a local correction. Keep corrections distinguishable from model output; they should not silently retrain anything.

Use ordinary reading preferences—font size, spacing, contrast, and explanation detail—across every preset. Age labels should not determine visual accessibility. Local history can be optional, with a visible delete action. Supply the complete app, dependencies, and model assets in an offline bundle or on local media. First launch can validate and prepare those local files, but must require no in-app downloads, accounts, or remote services.

## Give the three rewriting actions distinct jobs

The thesis uses paraphrasing and simplification somewhat interchangeably and does not publish their prompts or faithfulness checks. The following distinctions are **proposed product defaults**:

| Action | Proposed behavior | Details worth retaining |
| --- | --- | --- |
| **Paraphrase** | Restate the passage in clearer, neutral language with roughly equivalent detail. | Defined terms, actors, permissions, exceptions, quantities, and links. Avoid silently turning it into a summary. |
| **Simplify for Kids** | Use short sentences and concrete words; explain necessary terms briefly. | Who collects what, why, who receives it, and what the reader can do. Preserve important conditions even if this makes the explanation longer. |
| **Simplify for Adults** | Use plain language, manageable paragraphs, and direct explanations of choices. | Deadlines, settings, contact routes, limitations, and conditional promises. Use a respectful tone without assuming technical knowledge. |

**Audience ambiguity matters:** the thesis's Adults button is described in prose as serving older or senior users. Neither it nor Kids has a defined age band or reading level. Treat general-adult plain language as a new app proposal. If fidelity to that older-user emphasis is preferred, use helper text explaining it and validate the design with those users. Likewise, establish a Kids reading target through testing rather than claiming the thesis specified one. A later “More detail / Easier wording” control may be more useful than collecting ages.

Implementation can begin with one local instruction model and three versioned prompt templates. Each template receives the original passage, relevant surrounding context, and its writing objective. Tone and vocabulary instructions change; the requirement to preserve meaning does not. A model selector can stay in advanced settings until differences are useful to ordinary readers.

**Never rewrite the category list or the classifier's summary as if it were the policy.** Classifications can help navigation and passage selection, but the original text must remain the evidence for every rewrite. Otherwise, one mistaken classification can become a fabricated explanation. A rewrite can therefore be available before classification finishes.

For short text, one generation may suffice. For long text, split at section or paragraph boundaries, preserving nearby definitions and references. Rewrite each section against its source, then assemble in source order. Do not recursively summarize summaries. If a selection depends on “these partners” or “the exception above,” include the needed context or expose the unresolved reference. Extra context helps interpretation but should not silently expand the requested rewrite scope.

### A practical faithfulness example

Suppose the source says: “We may retain account records for up to 30 days after deletion, except where a longer period is required by law.”

| Preset | Illustrative candidate wording |
| --- | --- |
| Paraphrase | “After deletion, we may keep account records for up to 30 days. We may keep them longer if the law requires it.” |
| Kids | “After deletion, we may keep records about the account for up to 30 days. We may need to keep them longer if the law says we must.” |
| Adults | “We may keep account records for up to 30 days after deletion. A legal requirement may mean we keep them longer.” |

“Your data is deleted within 30 days” would lose **may**, change the scope of **account records**, and erase the exception. A shorter sentence is not automatically a better simplification. These examples illustrate a design criterion; they are not validated model outputs.

Store a mapping from each rewritten block to its source sentence IDs. Apply simple checks for changed numbers, dates, named parties, and missing negations; these identify review candidates, not semantic correctness. An optional second model pass can look for omitted conditions or unsupported claims, but agreement between models is not proof. When checks raise concerns, flag the affected block beside its source and support regeneration. Human review should remain part of evaluating the feature before claiming faithful simplification.

## Make Highlight Categories a reliable viewing feature

Once classifications exist, highlighting should be immediate and deterministic: **the model assigns labels; application code paints the source text**. Clicking the toggle should not trigger another inference request.

Retain the [ten thesis category names](../docs/thesis-chapters-4-5/chapter-4/dataset-and-taxonomy.md#ten-category-names) initially:

1. First Party Collection/Use
2. Third Party Sharing/Collection
3. User Choice/Control
4. Do Not Track
5. International and Specific Audiences
6. Data Security
7. Policy Change
8. Data Retention
9. User Access, Edit, and Deletion
10. Other

Keep this vocabulary in a versioned taxonomy file. The thesis notes list names without a full annotation guide; a tested classification design will also need clear definitions and examples. The taxonomy is a navigation scheme, not a complete assessment of a policy's implications.

A simple pipeline could be:

1. Preserve a canonical source-text version; create paragraphs and sentence spans with stable IDs and start/end offsets into that exact text.
2. Send small batches of sentence IDs and text, with useful paragraph context, to the local classifier. Request a list of allowed category IDs for each sentence.
3. Validate the response: reject unknown IDs and labels, deduplicate labels, and mark missing results for a bounded retry or review. A reasonable initial limit is one repair attempt. “Other” must remain a real category, separate from pending, failed, or unresolved classification.
4. Attach accepted label sets to the stored source spans. The renderer uses those spans directly, avoiding searches for text echoed by the LLM. Duplicate sentences and punctuation changes then cannot select the wrong occurrence.
5. Invalidate dependent classifications and rewrites when the source changes. Fix the offset convention across languages—for example, Python character indices and JavaScript string indices differ for some Unicode characters—and test it with emoji and non-English text.

Sentence classification is multi-label. In an all-categories view, a subtle sentence background plus a row of category chips is clearer than overlapping solid colors. Clicking a sentence reveals every label. In a single-category view, apply that category's highlight color to matching spans; when multiple filters are selected, default to “matches any selected category” and label this behavior. An optional “matches all” filter can follow.

Always pair colors with text labels and accessible focus states. Preserve readable contrast and source order. Category counts refer to matching sentences, so their sum can exceed the document's sentence count. A zero count means no matching labels were found, not that the corresponding practice is absent from the policy. Avoid displaying a model-written percentage as calibrated confidence.

## A small first version and useful experiments

A credible first version could support pasted English text, one local model, the ten categories, the four actions, cancellable jobs, and source-linked explanations. Render completed classification batches as they arrive. Serialize costly model jobs initially, and let a requested rewrite take priority over background work when practical. Cache results by source version, model, prompt version, taxonomy, and preset so switching views costs no inference time.

Before expanding to uploads, chat, or adjustable reading levels, try three focused evaluations:

- **Classification:** a reviewed set containing multi-label sentences, repeated text, and cross-sentence references; measure per-category precision/recall and inspect omissions. Hold out complete policies when comparing models to reduce source-policy leakage.
- **Rewriting:** examples with exceptions, negation, numbers, permissions, and definitions; review omitted facts, unsupported additions, and reader comprehension separately. Reading-level scores alone cannot establish comprehension or faithfulness.
- **Interaction:** test whether readers can locate sharing practices, explain a deletion exception, and return from a rewrite to its source. Measure cold-start wait, time to first useful result, full completion time, and cancellation on target machines.

The [thesis evaluation](../docs/thesis-chapters-4-5/chapter-5/evaluation-and-future-work.md) reports generation throughput but no formal usability, audience-comprehension, or paraphrase-faithfulness study. Its timings should not become local performance promises. The [classification limitations](../docs/thesis-chapters-4-5/chapter-4/results-and-limitations.md) also support treating sentence context and evaluation design as early experiments rather than settled assumptions.
