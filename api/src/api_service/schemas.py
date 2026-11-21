"""Pydantic contract for the browser-facing API.

This describes what the frontend sends and receives and mirrors the shapes that
``modelService`` returns. It holds no transport or storage logic; the handlers in
``app.py`` use these models and call local services.

Field IDs follow the object model in ``docs/Requirements-Document.md``:
sentence IDs are the offsets returned by ``modelService``, ``label_id`` follows the
fitted classifier's taxonomy order, and every generated block names its sources.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Keep in sync with MODEL_SERVICE_MAX_CHARACTERS in modelService/src/model_service/settings.py.
MAX_POLICY_CHARACTERS = 200_000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PolicySource(str, Enum):
    PASTE = "paste"
    UPLOAD = "upload"
    EXAMPLE = "example"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GenerationKind(str, Enum):
    SUMMARY = "summary"
    REWRITE = "rewrite"


class RewriteStyle(str, Enum):
    PARAPHRASE = "paraphrase"
    SIMPLIFY_FOR_KIDS = "simplify_for_kids"
    SIMPLIFY_FOR_ADULTS = "simplify_for_adults"


class AnalysisPhase(str, Enum):
    """Where a policy is in the staged flow.

    Classification always comes first and is independent of generation:
    ``classified`` means sentences are ready, ``generating`` means at least one
    summary/rewrite job is running, and ``results_ready`` means generated blocks
    have also landed.
    """

    AWAITING_CLASSIFICATION = "awaiting_classification"
    CLASSIFIED = "classified"
    GENERATING = "generating"
    RESULTS_READY = "results_ready"


class Category(BaseModel):
    """One of the ten taxonomy entries; ``id`` is the classifier's ``label_id``."""

    model_config = ConfigDict(frozen=True)

    id: int = Field(ge=0, le=9)
    name: str = Field(description="Canonical name, e.g. 'First Party Collection/Use'.")
    display_name: str
    definition: str
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    tint: str | None = Field(
        default=None,
        description="Optional UI highlight tint; the frontend may derive it from color.",
    )


class CategoryList(BaseModel):
    taxonomy_id: str
    version: int
    categories: list[Category]


class ModelProfile(BaseModel):
    """Identity of the model that produced a classification or generation.

    Mirrors ``modelService``'s ``model`` object. ``prompt`` and ``prompt_version``
    are only set for summary/rewrite results and are not part of classifier metadata.
    The rewriter has no taxonomy, so ``taxonomy_id`` and ``taxonomy_version`` are
    unset for rewrite results.
    """

    id: str
    revision: str
    precision: str
    device: str
    strategy: str
    head: str | None = None
    fit: str | None = None
    taxonomy_id: str | None = None
    taxonomy_version: int | None = None
    prompt: str | None = None
    prompt_version: int | None = None


class PolicyCreate(BaseModel):
    """Pasted text (FR-01) or the built-in example (FR-03)."""

    # No max_length: the store refuses a long policy with 413 and a message that says why.
    text: str = Field(min_length=1)
    source: PolicySource = PolicySource.PASTE
    filename: str | None = Field(
        default=None,
        description="Set by the API for uploads; ignored for pasted text.",
    )


class Policy(BaseModel):
    """In-memory handle for an uploaded policy. The text itself is not echoed.

    The caller already holds the text, and FR-22 history is off by default, so the
    API stores only what it needs to serve results for this session.
    """

    id: UUID
    source: PolicySource
    filename: str | None = None
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    characters: int = Field(ge=1, le=MAX_POLICY_CHARACTERS)
    created_at: datetime = Field(default_factory=_utcnow)


class ClassificationRequest(BaseModel):
    include_scores: bool = Field(
        default=False,
        description="Include the ten per-category scores for every sentence.",
    )


class SentenceClassification(BaseModel):
    """A sentence span plus the model's label.

    ``start`` and ``end`` are character offsets into the policy exactly as
    submitted; the service validates ``text[ start : end ]`` against stored text.
    A sentence the model refuses keeps its offsets and carries ``error`` instead
    of a label, as modelService already does.
    """

    id: int = Field(ge=0)
    text: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    label_id: int | None = Field(default=None, ge=0, le=9)
    label: str | None = None
    scores: list[float] | None = None
    error: str | None = None

    @model_validator(mode="after")
    def offsets_and_label_are_consistent(self) -> SentenceClassification:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if self.label_id is None and not self.error:
            raise ValueError("a sentence without a label must explain why in error")
        return self


class CategoryCount(BaseModel):
    label_id: int = Field(ge=0, le=9)
    name: str
    count: int = Field(ge=0)


class Classification(BaseModel):
    policy_id: UUID
    model: ModelProfile
    taxonomy_id: str
    taxonomy_version: int
    sentences: list[SentenceClassification]
    counts: list[CategoryCount] = Field(
        default_factory=list,
        description="Counts in taxonomy order, for the legend in FR-06.",
    )


class SummaryRequest(BaseModel):
    """Scope of a per-category summary run (FR-16)."""

    category_ids: list[int] | None = Field(
        default=None,
        min_length=1,
        description="Categories to summarize; all ten when omitted.",
    )
    sentence_ids: list[int] | None = Field(
        default=None,
        min_length=1,
        description="Restrict the source sentences; all classified sentences when omitted.",
    )


class RewriteRequest(BaseModel):
    """Scope of a rewrite run (FR-09 to FR-13)."""

    style: RewriteStyle
    sentence_ids: list[int] | None = Field(
        default=None,
        min_length=1,
        description="The user's selection; the whole policy when omitted.",
    )
    context_sentence_ids: list[int] = Field(
        default_factory=list,
        description="Extra surrounding sentences needed for references outside the selection.",
    )


class Summary(BaseModel):
    label_id: int = Field(ge=0, le=9)
    category: str
    text: str = Field(min_length=1)
    source_sentence_ids: list[int] = Field(min_length=1)


class Rewrite(BaseModel):
    id: UUID
    text: str = Field(min_length=1)
    source_sentence_ids: list[int] = Field(min_length=1)
    warnings: list[str] = Field(
        default_factory=list,
        description="FR-15 flags: changed numbers, dates, parties, or negations.",
    )


class SummaryResult(BaseModel):
    kind: Literal["summary"] = "summary"
    policy_id: UUID
    model: ModelProfile
    taxonomy_id: str
    taxonomy_version: int
    summaries: list[Summary]


class RewriteResult(BaseModel):
    kind: Literal["rewrite"] = "rewrite"
    policy_id: UUID
    model: ModelProfile
    style: RewriteStyle
    rewrites: list[Rewrite]


GenerationResult = Annotated[SummaryResult | RewriteResult, Field(discriminator="kind")]


class JobProgress(BaseModel):
    stage: str = Field(
        description="e.g. 'queued', 'loading_model', 'summarizing', 'rewriting', 'done'."
    )
    completed: int = Field(ge=0)
    total: int = Field(ge=0)
    message: str | None = None


class ApiErrorInfo(BaseModel):
    code: str = Field(description="Stable code such as 'model_unavailable' or 'bad_upload'.")
    message: str
    retryable: bool = False


class Job(BaseModel):
    id: UUID
    policy_id: UUID
    kind: GenerationKind
    status: JobStatus
    progress: JobProgress | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    error: ApiErrorInfo | None = None
    result: GenerationResult | None = None

    @model_validator(mode="after")
    def result_matches_kind(self) -> Job:
        if self.status is JobStatus.SUCCEEDED and self.result is None:
            raise ValueError("a succeeded job must carry a result")
        if self.status is JobStatus.FAILED and self.error is None:
            raise ValueError("a failed job must carry an error")
        if self.result is not None and self.result.kind != self.kind.value:
            raise ValueError("result kind does not match the job kind")
        return self


class AnalysisResults(BaseModel):
    """Snapshot of whatever is ready *at this moment*.

    This is never a barrier: the frontend can fetch it right after classification,
    while ``summaries`` and ``rewrites`` are still empty, and fetch it again after a
    generation job succeeds. Use ``phase`` to tell those two cases apart.
    """

    policy: Policy
    phase: AnalysisPhase = AnalysisPhase.AWAITING_CLASSIFICATION
    classification: Classification | None = None
    summaries: list[Summary] = Field(default_factory=list)
    rewrites: list[Rewrite] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def phase_matches_content(self) -> AnalysisResults:
        if self.classification is not None and self.phase is AnalysisPhase.AWAITING_CLASSIFICATION:
            raise ValueError("a classification is present, so phase cannot be awaiting_classification")
        if self.phase is AnalysisPhase.RESULTS_READY:
            if self.classification is None:
                raise ValueError("results_ready requires a classification")
            if not (self.summaries or self.rewrites):
                raise ValueError("results_ready requires at least one summary or rewrite")
        return self


class Health(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    api_version: str
    model_service_ready: bool
    model: ModelProfile | None = None
    rewriter_ready: bool = False
    rewriter_status: str = "available"
    rewriter_setup: str | None = None


class ErrorEnvelope(BaseModel):
    """Standard error body, matching FastAPI's ``{"detail": ...}`` convention."""

    detail: ApiErrorInfo
