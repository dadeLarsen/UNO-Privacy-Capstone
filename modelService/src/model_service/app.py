"""FastAPI app for classifying uploaded privacy policies on the local machine."""

from contextlib import asynccontextmanager
import hashlib
import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .classifier import Classifier, PrivacyBenchClassifier
from .sentences import split_sentences
from .settings import Settings
from .taxonomy import TAXONOMY

logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The complete policy text, unchanged.")
    include_scores: bool = Field(False, description="Include the ten per-category scores for each sentence.")


def create_app(settings: Settings | None = None, classifier: Classifier | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    service = classifier or PrivacyBenchClassifier(settings.condition_path, cache_size=settings.cache_size)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.preload:
            logger.info("Loading the classifier from %s", settings.condition_path.name)
            service.load()
            logger.info("Classifier ready")
        yield

    app = FastAPI(title="Privacy Policy Reader model service", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.classifier = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        metadata = service.metadata()
        return {"status": "ok" if metadata is not None else "loading", "ready": metadata is not None, "model": metadata}

    @app.get("/categories")
    def categories():
        return {
            "taxonomy_id": TAXONOMY["taxonomy_id"],
            "version": TAXONOMY["version"],
            "categories": TAXONOMY["categories"],
        }

    @app.post("/analyze")
    def analyze(request: AnalyzeRequest):
        return _analyze(service, settings, request.text, request.include_scores)

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
        return _analyze(service, settings, text, include_scores)

    return app


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
        logger.error("Classifier failed to load (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail=f"Classifier model unavailable: {exc}") from exc

    results = []
    for sentence in sentences:
        try:
            classification = service.classify(sentence.text)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Sentence {sentence.index} cannot be classified: {exc}",
            ) from exc
        except Exception as exc:
            logger.error("Classification failed at sentence %d (%s)", sentence.index, type(exc).__name__)
            raise HTTPException(status_code=500, detail="Classification failed.") from exc
        item = {
            "id": sentence.index,
            "text": sentence.text,
            "start": sentence.start,
            "end": sentence.end,
            "label_id": classification.label_id,
            "label": classification.label,
        }
        if include_scores:
            item["scores"] = list(classification.scores or ())
        results.append(item)

    return {
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text),
        "sentences": results,
        "model": metadata,
    }
