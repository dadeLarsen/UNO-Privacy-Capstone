"""FastAPI app for classifying and rewriting privacy policies on the local machine."""

import asyncio
from contextlib import asynccontextmanager
import hashlib
import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .classifier import Classifier, PrivacyBenchClassifier
from .idle import IdleRelease
from .rewriter import PrivacyBenchRewriter, Rewriter
from .sentences import split_sentences
from .settings import Settings
from .taxonomy import TAXONOMY

logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The complete policy text, unchanged.")
    include_scores: bool = Field(False, description="Include the ten per-category scores for each sentence.")


class RewriteRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The passage to rewrite, unchanged.")


def create_app(settings: Settings | None = None, classifier: Classifier | None = None,
               rewriter: Rewriter | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    service = classifier or PrivacyBenchClassifier(settings.condition_path, cache_size=settings.cache_size)
    rewrite_service = rewriter or PrivacyBenchRewriter(settings.rewrite_condition_path)
    idle = IdleRelease((service, rewrite_service), settings.idle_release_seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.preload:
            # Only the classifier loads at startup: one model is in memory at a time, and analysis comes first.
            logger.info("Loading the classifier from %s", settings.condition_path.name)
            with idle.in_use(service):
                service.load()
            logger.info("Classifier ready")
        watcher = asyncio.create_task(_release_when_idle(idle))
        yield
        watcher.cancel()

    app = FastAPI(title="Privacy Policy Reader model service", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.classifier = service
    app.state.rewriter = rewrite_service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        metadata = service.metadata()
        # ready and model describe the classifier. rewriter is null until the first rewrite has loaded it.
        return {"status": "ok" if metadata is not None else "loading", "ready": metadata is not None,
                "model": metadata, "rewriter": rewrite_service.metadata()}

    @app.get("/categories")
    def categories():
        return {
            "taxonomy_id": TAXONOMY["taxonomy_id"],
            "version": TAXONOMY["version"],
            "categories": TAXONOMY["categories"],
        }

    def analyze_text(text: str, include_scores: bool) -> dict:
        # A release waits for every request under way, so it never unloads the model in the middle of one.
        with idle.in_use(service):
            return _analyze(service, settings, text, include_scores)

    @app.post("/analyze")
    def analyze(request: AnalyzeRequest):
        return analyze_text(request.text, request.include_scores)

    @app.post("/analyze/upload")
    def analyze_upload(file: UploadFile = File(...), include_scores: bool = False):
        filename = (file.filename or "").strip()
        if filename and Path(filename).suffix.lower() != ".txt":
            raise HTTPException(
                status_code=415,
                detail="Upload a UTF-8 .txt file. PDF and DOCX import is not supported.",
            )
        max_bytes = settings.max_characters * 4 + 1
        data = file.file.read(max_bytes)
        if len(data) >= max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Policy is longer than {settings.max_characters} characters.",
            )
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="The file is not valid UTF-8 text.") from None
        return analyze_text(text, include_scores)

    @app.post("/rewrite")
    def rewrite(request: RewriteRequest):
        # This waits for an analysis under way, then releases the classifier before the rewriter loads.
        with idle.in_use(rewrite_service):
            return _rewrite(rewrite_service, settings, request.text)

    return app


async def _release_when_idle(idle: IdleRelease) -> None:
    while True:
        try:
            # A check can wait for a lock and collect garbage, so it runs off the event loop.
            wait = await asyncio.to_thread(idle.check)
        except Exception:
            # The model stays loaded and still answers. Say why it was not released; the next idle period
            # tries again.
            logger.exception("Could not release the idle model")
            wait = idle.idle_seconds
        await asyncio.sleep(wait)


def _analyze(service: Classifier, settings: Settings, text: str, include_scores: bool) -> dict:
    if len(text) > settings.max_characters:
        raise HTTPException(
            status_code=413,
            detail=f"Policy is longer than {settings.max_characters} characters.",
        )
    sentences = split_sentences(text)
    if not sentences:
        raise HTTPException(status_code=422, detail="The policy has no sentences to classify.")
    try:
        metadata = service.load()
    except Exception as exc:
        # A traceback names code locations, not policy text.
        logger.exception("Classifier failed to load")
        raise HTTPException(status_code=503, detail=f"Classifier model unavailable: {exc}") from exc

    results = []
    for sentence in sentences:
        item = {"id": sentence.index, "text": sentence.text, "start": sentence.start, "end": sentence.end}
        try:
            classification = service.classify(sentence.text)
        except ValueError as exc:
            # The model cannot take this sentence, for example because it is over the token limit.
            # It is reported in place, and the rest of the policy still gets its labels.
            item.update(label_id=None, label=None, error=str(exc))
            if include_scores:
                item["scores"] = None
        except Exception as exc:
            logger.exception("Classification failed at sentence %d", sentence.index)
            raise HTTPException(status_code=500, detail="Classification failed.") from exc
        else:
            item.update(label_id=classification.label_id, label=classification.label)
            if include_scores:
                item["scores"] = list(classification.scores or ())
        results.append(item)

    return {
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text),
        "sentences": results,
        "model": metadata,
    }


def _rewrite(service: Rewriter, settings: Settings, text: str) -> dict:
    if len(text) > settings.max_characters:
        raise HTTPException(
            status_code=413,
            detail=f"Passage is longer than {settings.max_characters} characters.",
        )
    if not text.strip():
        raise HTTPException(status_code=422, detail="The passage has no text to rewrite.")
    try:
        metadata = service.load()
    except Exception as exc:
        # A traceback names code locations, not policy text.
        logger.exception("Rewriter failed to load")
        raise HTTPException(status_code=503, detail=f"Rewriter model unavailable: {exc}") from exc
    try:
        result = service.rewrite(text)
    except ValueError as exc:
        # The passage and the output do not fit the model's context. Nothing is truncated; the reason says by how much.
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Rewriting failed")
        raise HTTPException(status_code=500, detail="Rewriting failed.") from exc
    return {
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text),
        "rewrite": result.text,
        "status": result.status,
        "output_tokens": result.output_tokens,
        "model": metadata,
    }
