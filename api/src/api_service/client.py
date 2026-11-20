"""Thin httpx adapter for the localhost model service."""

from __future__ import annotations

import httpx

from .errors import ModelServiceError
from .settings import Settings


class ModelServiceClient:
    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> ModelServiceClient:
        return cls(
            httpx.Client(
                base_url=settings.model_service_url,
                timeout=settings.request_timeout_seconds,
                trust_env=False,
            )
        )

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict:
        return self._request("GET", "/health")

    def categories(self) -> dict:
        return self._request("GET", "/categories")

    def analyze(self, text: str, include_scores: bool = False) -> dict:
        return self._request("POST", "/analyze", json={"text": text, "include_scores": include_scores})

    def rewrite(self, text: str) -> dict:
        return self._request("POST", "/rewrite", json={"text": text})

    def _request(self, method: str, path: str, **kwargs: object) -> dict:
        try:
            response = self._client.request(method, path, **kwargs)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ModelServiceError(
                status_code=503,
                code="model_service_timeout",
                message="The model service did not answer in time.",
                retryable=True,
            ) from exc
        except httpx.RequestError as exc:
            raise ModelServiceError(
                status_code=503,
                code="model_service_unavailable",
                message="The model service is not running. Start it with `uv run model-service`.",
                retryable=True,
            ) from exc
        except httpx.HTTPStatusError as exc:
            code, message, retryable = _problem_details(exc.response)
            raise ModelServiceError(
                status_code=502 if exc.response.status_code >= 500 else exc.response.status_code,
                code=code,
                message=message,
                retryable=retryable,
            ) from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise ModelServiceError(
                status_code=502,
                code="model_service_bad_response",
                message="The model service returned a non-JSON response.",
                retryable=True,
            ) from exc
        if not isinstance(body, dict):
            raise ModelServiceError(
                status_code=502,
                code="model_service_bad_response",
                message="The model service returned a JSON value that was not an object.",
                retryable=True,
            )
        return body


def _problem_details(response: httpx.Response) -> tuple[str, str, bool]:
    """Preserve a structured model-service error code instead of flattening it into text."""
    default_code = "model_service_error" if response.status_code >= 500 else "model_service_rejected"
    default_retryable = response.status_code >= 500
    fallback = f"Model service returned HTTP {response.status_code}."
    try:
        body = response.json()
    except ValueError:
        return default_code, response.text.strip() or fallback, default_retryable
    if not isinstance(body, dict):
        return default_code, str(body), default_retryable
    detail = body.get("detail", fallback)
    if isinstance(detail, dict):
        return (
            str(detail.get("code") or default_code),
            str(detail.get("message") or fallback),
            bool(detail.get("retryable", default_retryable)),
        )
    return default_code, str(detail), default_retryable
