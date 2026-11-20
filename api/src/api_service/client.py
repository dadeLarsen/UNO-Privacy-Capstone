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

    def rewrite(self, text: str, style: str, source_sentence_ids: list[int]) -> dict:
        return self._request(
            "POST",
            "/rewrite",
            json={
                "text": text,
                "style": style,
                "source_sentence_ids": source_sentence_ids,
            },
        )

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
            raise ModelServiceError(
                status_code=502 if exc.response.status_code >= 500 else exc.response.status_code,
                code="model_service_error" if exc.response.status_code >= 500 else "model_service_rejected",
                message=_detail(exc.response),
                retryable=exc.response.status_code >= 500,
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


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text.strip() or f"Model service returned HTTP {response.status_code}."
    detail = body.get("detail", f"Model service returned HTTP {response.status_code}.")
    if isinstance(detail, dict):
        return str(detail.get("message") or detail)
    return str(detail)
