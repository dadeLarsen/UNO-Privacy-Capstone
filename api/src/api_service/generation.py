"""Generation behind the API jobs.

Summaries are extractive: they quote the already-classified sentences, so they need
no download and cannot change what the policy says. Rewrites call the local model
service's pinned GGUF rewriter; every result points back to the source sentence IDs
the model actually received.
"""

from __future__ import annotations

from threading import Event
from uuid import UUID, uuid4

from .client import ModelServiceClient
from .schemas import (
    Classification,
    ModelProfile,
    Rewrite,
    RewriteRequest,
    RewriteResult,
    Summary,
    SummaryResult,
)

MAX_SENTENCES_PER_SUMMARY = 2
# modelService's `status` is the benchmark's verdict on a rewrite; `empty` fails the job instead.
STATUS_WARNINGS = {
    "valid": None,
    "truncated": "The rewrite reached the output token limit and may be incomplete.",
    "reasoning": "The rewrite contains reasoning tags; review it before using it.",
}


class JobCancelled(Exception):
    """Raised when a job's cancel event was set before it finished."""


def extractive_summaries(
    policy_id: UUID,
    classification: Classification,
    *,
    category_ids: list[int] | None,
    sentence_ids: list[int] | None,
    cancel: Event,
) -> SummaryResult:
    allowed = set(sentence_ids) if sentence_ids else None
    sentences = [
        sentence
        for sentence in classification.sentences
        if (allowed is None or sentence.id in allowed) and sentence.label_id is not None
    ]
    wanted = set(category_ids) if category_ids else None
    summaries: list[Summary] = []
    total = len(classification.counts) or 1

    for index, count in enumerate(classification.counts):
        if cancel.is_set():
            raise JobCancelled
        if wanted is not None and count.label_id not in wanted:
            continue
        matches = [sentence for sentence in sentences if sentence.label_id == count.label_id]
        if not matches:
            continue
        chosen = matches[:MAX_SENTENCES_PER_SUMMARY]
        summaries.append(
            Summary(
                label_id=count.label_id,
                category=count.name,
                text=" ".join(sentence.text for sentence in chosen),
                source_sentence_ids=[sentence.id for sentence in chosen],
            )
        )

    if cancel.is_set():
        raise JobCancelled

    return SummaryResult(
        policy_id=policy_id,
        model=ModelProfile(
            id="local/extractive-summary",
            revision="extractive-v1",
            precision="n/a",
            device="local",
            strategy="extractive",
            taxonomy_id=classification.taxonomy_id,
            taxonomy_version=classification.taxonomy_version,
        ),
        taxonomy_id=classification.taxonomy_id,
        taxonomy_version=classification.taxonomy_version,
        summaries=summaries,
    )


def rewritten_passage(
    policy_id: UUID,
    classification: Classification,
    request: RewriteRequest,
    *,
    model_client: ModelServiceClient,
    cancel: Event,
) -> RewriteResult:
    if cancel.is_set():
        raise JobCancelled

    by_id = {sentence.id: sentence for sentence in classification.sentences}
    source_ids = list(request.sentence_ids or [sentence.id for sentence in classification.sentences])
    for sentence_id in request.context_sentence_ids:
        if sentence_id not in source_ids:
            source_ids.append(sentence_id)
    selected = [by_id[sentence_id] for sentence_id in source_ids if sentence_id in by_id]
    if not selected:
        raise ValueError("No classified sentences were selected for rewriting.")

    passage = " ".join(sentence.text for sentence in selected)
    body = model_client.rewrite(passage)
    if cancel.is_set():
        raise JobCancelled

    model = ModelProfile.model_validate(body["model"])
    status = body["status"]
    if status == "empty":
        raise ValueError("The model service returned an empty rewrite.")
    if status not in STATUS_WARNINGS:
        raise ValueError(f"The model service returned an unknown rewrite status: {status!r}.")
    # modelService runs one prompt, so the style changes nothing yet; say so rather than imply otherwise.
    warnings = [
        f"Every style uses the evaluated {model.prompt} prompt, including {request.style.value}; "
        "style-specific prompts have not been evaluated."
    ]
    if STATUS_WARNINGS[status]:
        warnings.append(STATUS_WARNINGS[status])

    return RewriteResult(
        policy_id=policy_id,
        model=model,
        style=request.style,
        rewrites=[
            Rewrite(
                id=uuid4(),
                text=body["rewrite"],
                source_sentence_ids=[sentence.id for sentence in selected],
                warnings=warnings,
            )
        ],
    )
