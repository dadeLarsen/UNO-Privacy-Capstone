"""Generation behind the API jobs.

Summaries are extractive: they quote the already-classified sentences, so they need
no download and cannot change what the policy says. Rewrites call the local model
service's pinned GGUF rewriter, one section of the policy at a time; every rewrite
points back to the source sentence IDs the model actually received.
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
    RewriteStyle,
    SentenceClassification,
    Summary,
    SummaryResult,
)

MAX_SENTENCES_PER_SUMMARY = 2
# About the longest passage whose rewrites were judged (2,154 characters), so a section stays within
# what privacy-research/docs/rewriting-findings.md measured.
SECTION_CHARACTERS = 2000
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

    for count in classification.counts:
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


def rewrite_sections(
    text: str,
    classification: Classification,
    request: RewriteRequest,
) -> list[list[SentenceClassification]]:
    """Cut the sentences to rewrite, in policy order, into sections of about SECTION_CHARACTERS.

    A section ends at a paragraph break where it can. A paragraph over the limit is cut between
    sentences, and a sentence over the limit is a section of its own.
    """
    wanted = None if request.sentence_ids is None else {*request.sentence_ids, *request.context_sentence_ids}
    selected = [sentence for sentence in classification.sentences if wanted is None or sentence.id in wanted]
    if not selected:
        raise ValueError("No classified sentences were selected for rewriting.")

    # A paragraph ends at a newline in the policy, or where the selection skips a sentence. A line that
    # does not end a sentence is a heading or opens a list, so it stays with the paragraph after it.
    paragraphs: list[list[SentenceClassification]] = []
    for sentence in selected:
        previous = paragraphs[-1][-1] if paragraphs else None
        if (
            previous
            and sentence.id == previous.id + 1
            and ("\n" not in text[previous.end : sentence.start] or not _ends_a_sentence(previous.text))
        ):
            paragraphs[-1].append(sentence)
        else:
            paragraphs.append([sentence])

    sections: list[list[SentenceClassification]] = []
    size = 0
    for paragraph in paragraphs:
        if sections and size + _characters(paragraph) <= SECTION_CHARACTERS:
            sections[-1].extend(paragraph)
            size += _characters(paragraph)
            continue
        sections.append([])
        size = 0
        for sentence in paragraph:
            if sections[-1] and size + _characters([sentence]) > SECTION_CHARACTERS:
                sections.append([])
                size = 0
            sections[-1].append(sentence)
            size += _characters([sentence])
    return sections


def _ends_a_sentence(text: str) -> bool:
    return text.rstrip("\"')]}”’").endswith((".", "!", "?"))


def _characters(sentences: list[SentenceClassification]) -> int:
    """The length these sentences add to a passage, with the space that joins each to the next."""
    return sum(len(sentence.text) + 1 for sentence in sentences)


def rewritten_section(
    section: list[SentenceClassification],
    style: RewriteStyle,
    model_client: ModelServiceClient,
) -> tuple[ModelProfile, Rewrite]:
    body = model_client.rewrite(" ".join(sentence.text for sentence in section))
    model = ModelProfile.model_validate(body["model"])
    status = body["status"]
    if status == "empty":
        raise ValueError("The model service returned an empty rewrite.")
    if status not in STATUS_WARNINGS:
        raise ValueError(f"The model service returned an unknown rewrite status: {status!r}.")
    # modelService runs one prompt, so the style changes nothing yet; say so rather than imply otherwise.
    warnings = [
        f"Every style uses the evaluated {model.prompt} prompt, including {style.value}; "
        "style-specific prompts have not been evaluated."
    ]
    if STATUS_WARNINGS[status]:
        warnings.append(STATUS_WARNINGS[status])

    return model, Rewrite(
        id=uuid4(),
        text=body["rewrite"],
        source_sentence_ids=[sentence.id for sentence in section],
        warnings=warnings,
    )
