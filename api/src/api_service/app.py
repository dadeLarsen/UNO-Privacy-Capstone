"""FastAPI app that connects the browser to the localhost model service."""

from __future__ import annotations

import codecs
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .client import ModelServiceClient
from .errors import ApiProblem, ModelServiceError, PolicyTooLong
from .generation import JobCancelled, extractive_summaries, rewrite_sections, rewritten_section
from .schemas import (
    MAX_POLICY_CHARACTERS,
    AnalysisResults,
    ApiErrorInfo,
    CategoryCount,
    CategoryList,
    Classification,
    ClassificationRequest,
    ErrorEnvelope,
    GenerationKind,
    Health,
    Job,
    JobProgress,
    JobStatus,
    ModelProfile,
    Policy,
    PolicyCreate,
    PolicySource,
    Rewrite,
    RewriteRequest,
    RewriteResult,
    SummaryRequest,
)
from .settings import Settings
from .store import JobRecord, PolicyRecord, Store

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    client: ModelServiceClient | None = None,
    store: Store | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    owns_client = client is None
    model_client = client or ModelServiceClient.from_settings(settings)
    policies = store or Store(settings.policy_ttl_seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if owns_client:
            model_client.close()

    app = FastAPI(
        title="Privacy Policy Reader API",
        version="0.1.0",
        lifespan=lifespan,
        # Without this, the generated docs show FastAPI's own 422 body, which the handlers below never send.
        responses={"default": {"model": ErrorEnvelope, "description": "Error with a stable code"}},
    )
    app.state.settings = settings
    app.state.client = model_client
    app.state.store = policies
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    @app.exception_handler(ApiProblem)
    async def api_problem_handler(request: Request, exc: ApiProblem) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": {
                    "code": exc.code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "invalid_request",
                    "message": "The request could not be validated.",
                    "retryable": False,
                }
            },
        )

    @app.get("/api/v1/health", response_model=Health)
    def health() -> Health:
        try:
            body = model_client.health()
        except ApiProblem:
            return Health(status="unavailable", api_version=app.version, model_service_ready=False)
        model = ModelProfile.model_validate(body["model"]) if body.get("model") else None
        ready = bool(body.get("ready"))
        return Health(
            status="ok" if ready else "degraded",
            api_version=app.version,
            model_service_ready=ready,
            model=model,
        )

    @app.get("/api/v1/categories", response_model=CategoryList)
    def categories() -> CategoryList:
        return CategoryList.model_validate(model_client.categories())

    @app.post("/api/v1/policies", response_model=Policy, status_code=201)
    def create_policy(body: PolicyCreate) -> Policy:
        record = policies.create_policy(body.text, body.source, body.filename)
        return record.policy

    @app.post("/api/v1/policies/upload", response_model=Policy, status_code=201)
    def upload_policy(file: UploadFile = File(...)) -> Policy:
        filename = (file.filename or "").strip()
        if filename and Path(filename).suffix.lower() != ".txt":
            raise ApiProblem(
                status_code=415,
                code="unsupported_file_type",
                message="Upload a UTF-8 .txt file. PDF and DOCX import is not supported.",
            )
        # A character takes at most four bytes in UTF-8, and decoding drops a byte order mark.
        # This only bounds the read; the store counts the characters.
        max_bytes = MAX_POLICY_CHARACTERS * 4 + len(codecs.BOM_UTF8)
        data = file.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise PolicyTooLong
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ApiProblem(
                status_code=400,
                code="invalid_encoding",
                message="The file is not valid UTF-8 text.",
            ) from exc
        if not text.strip():
            raise ApiProblem(status_code=422, code="empty_policy", message="The file has no policy text.")
        record = policies.create_policy(text, PolicySource.UPLOAD, filename or None)
        return record.policy

    @app.get("/api/v1/policies/{policy_id}", response_model=Policy)
    def get_policy(policy_id: UUID) -> Policy:
        return policies.get_policy(policy_id).policy

    @app.post(
        "/api/v1/policies/{policy_id}/classification",
        response_model=Classification,
    )
    def classify(policy_id: UUID, body: ClassificationRequest | None = None) -> Classification:
        record = policies.get_policy(policy_id)
        result = model_client.analyze(record.text, bool(body and body.include_scores))
        category_list = CategoryList.model_validate(model_client.categories())
        counts = [
            CategoryCount(
                label_id=category.id,
                name=category.display_name,
                count=sum(
                    sentence.get("label_id") == category.id
                    for sentence in result.get("sentences", [])
                    if isinstance(sentence, dict)
                ),
            )
            for category in category_list.categories
        ]
        result["policy_id"] = str(policy_id)
        result["taxonomy_id"] = category_list.taxonomy_id
        result["taxonomy_version"] = category_list.version
        result["counts"] = [count.model_dump() for count in counts]
        classification = Classification.model_validate(result)
        policies.set_classification(policy_id, classification)
        return classification

    @app.post(
        "/api/v1/policies/{policy_id}/summaries",
        response_model=Job,
        status_code=202,
    )
    def create_summary_job(policy_id: UUID, body: SummaryRequest | None = None) -> Job:
        record = policies.get_policy(policy_id)
        if record.classification is None:
            raise ApiProblem(
                status_code=409,
                code="classification_required",
                message="Analyze categories before asking for summaries.",
            )
        request = body or SummaryRequest()
        _check_sentence_ids(record, request.sentence_ids)
        known_labels = {count.label_id for count in record.classification.counts}
        if request.category_ids and not set(request.category_ids) <= known_labels:
            raise ApiProblem(
                status_code=422,
                code="unknown_category",
                message="category_ids must be IDs from /categories.",
            )
        job = policies.create_job(policy_id, GenerationKind.SUMMARY.value)
        _start_thread(
            _run_summary,
            policies,
            job,
            record.classification.model_copy(deep=True),
            request,
        )
        return job.job.model_copy(deep=True)

    @app.post(
        "/api/v1/policies/{policy_id}/rewrites",
        response_model=Job,
        status_code=202,
    )
    def create_rewrite_job(policy_id: UUID, body: RewriteRequest) -> Job:
        record = policies.get_policy(policy_id)
        if record.classification is None:
            raise ApiProblem(
                status_code=409,
                code="classification_required",
                message="Analyze categories before asking for a rewrite.",
            )
        _check_sentence_ids(record, body.sentence_ids)
        _check_sentence_ids(record, body.context_sentence_ids or None)
        job = policies.create_job(policy_id, GenerationKind.REWRITE.value)
        _start_thread(
            _run_rewrite,
            policies,
            job,
            record.text,
            record.classification.model_copy(deep=True),
            body,
            model_client,
        )
        return job.job.model_copy(deep=True)

    @app.get("/api/v1/jobs/{job_id}", response_model=Job)
    def get_job(job_id: UUID) -> Job:
        return policies.get_job(job_id).job.model_copy(deep=True)

    @app.delete("/api/v1/jobs/{job_id}", status_code=204)
    def cancel_job(job_id: UUID) -> None:
        record = policies.get_job(job_id)
        if not policies.cancel_job(record.job.id):
            raise ApiProblem(
                status_code=409,
                code="job_not_cancellable",
                message="The job has already finished.",
            )

    @app.get("/api/v1/policies/{policy_id}/results", response_model=AnalysisResults)
    def get_results(policy_id: UUID) -> AnalysisResults:
        return policies.snapshot(policy_id)

    return app


def _check_sentence_ids(record: PolicyRecord, sentence_ids: list[int] | None) -> None:
    if not sentence_ids or record.classification is None:
        return
    known = {sentence.id for sentence in record.classification.sentences}
    if not set(sentence_ids) <= known:
        raise ApiProblem(
            status_code=422,
            code="unknown_sentence",
            message="sentence_ids must come from the classification result.",
        )


def _start_thread(target: object, *args: object) -> None:
    threading.Thread(target=target, args=args, daemon=True).start()


def _run_summary(
    store: Store,
    job: JobRecord,
    classification: Classification,
    request: SummaryRequest,
) -> None:
    job_id = job.job.id
    try:
        store.set_progress(job_id, "building_summaries", 0, 1)
        result = extractive_summaries(
            job.policy_id,
            classification,
            category_ids=request.category_ids,
            sentence_ids=request.sentence_ids,
            cancel=job.cancel,
        )
        if job.cancel.is_set():
            raise JobCancelled
        store.finish_job(
            job_id,
            status=JobStatus.SUCCEEDED,
            result=result,
            progress=JobProgress(stage="done", completed=1, total=1),
        )
    except JobCancelled:
        store.cancel_job(job_id)
    except Exception as exc:
        logger.exception("Summary job %s failed", job_id)
        store.finish_job(
            job_id,
            status=JobStatus.FAILED,
            error=ApiErrorInfo(code="generation_failed", message=str(exc), retryable=False),
            progress=JobProgress(stage="failed", completed=0, total=1),
        )


def _run_rewrite(
    store: Store,
    job: JobRecord,
    text: str,
    classification: Classification,
    request: RewriteRequest,
    model_client: ModelServiceClient,
) -> None:
    job_id = job.job.id
    rewrites: list[Rewrite] = []
    total = 1
    try:
        sections = rewrite_sections(text, classification, request)
        total = len(sections)
        for section in sections:
            # A cancel stops the job between two sections. modelService still finishes the one it has.
            if job.cancel.is_set():
                raise JobCancelled
            store.set_progress(
                job_id,
                "rewriting",
                len(rewrites),
                total,
                f"Rewriting section {len(rewrites) + 1} of {total}",
            )
            model, rewrite = rewritten_section(section, request.style, model_client)
            rewrites.append(rewrite)
        store.finish_job(
            job_id,
            status=JobStatus.SUCCEEDED,
            result=RewriteResult(policy_id=job.policy_id, model=model, style=request.style, rewrites=rewrites),
            progress=JobProgress(stage="done", completed=total, total=total),
        )
    except JobCancelled:
        store.cancel_job(job_id)
    except ModelServiceError as exc:
        store.finish_job(
            job_id,
            status=JobStatus.FAILED,
            error=ApiErrorInfo(code=exc.code, message=exc.message, retryable=exc.retryable),
            progress=JobProgress(stage="model_service_error", completed=len(rewrites), total=total),
        )
    except Exception as exc:
        logger.exception("Rewrite job %s failed", job_id)
        store.finish_job(
            job_id,
            status=JobStatus.FAILED,
            error=ApiErrorInfo(code="generation_failed", message=str(exc), retryable=False),
            progress=JobProgress(stage="failed", completed=len(rewrites), total=total),
        )
