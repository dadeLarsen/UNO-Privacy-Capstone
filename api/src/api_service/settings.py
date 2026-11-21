"""Local-only configuration for the frontend-facing API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

DEFAULT_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _loopback(value: str, label: str) -> str:
    if value not in LOOPBACK_HOSTS:
        raise ValueError(f"{label} must be a loopback address, got {value!r}")
    return value


@dataclass(frozen=True)
class Settings:
    model_service_url: str = "http://127.0.0.1:8000"
    host: str = "127.0.0.1"
    port: int = 8001
    cors_origins: tuple[str, ...] = DEFAULT_ORIGINS
    request_timeout_seconds: float = 120.0
    max_characters: int = 200_000
    policy_ttl_seconds: int = 3600

    @classmethod
    def from_env(cls) -> Settings:
        url = os.getenv("API_MODEL_SERVICE_URL", cls.model_service_url).rstrip("/")
        parsed = urlparse(url)
        _loopback(parsed.hostname or "", "API_MODEL_SERVICE_URL host")
        host = _loopback(os.getenv("API_HOST", cls.host), "API_HOST")
        origins = tuple(
            origin.strip()
            for origin in os.getenv("API_CORS_ORIGINS", ",".join(cls.cors_origins)).split(",")
            if origin.strip()
        )
        for origin in origins:
            _loopback(urlparse(origin).hostname or "", "CORS origin")
        return cls(
            model_service_url=url,
            host=host,
            port=int(os.getenv("API_PORT", str(cls.port))),
            cors_origins=origins,
            request_timeout_seconds=float(
                os.getenv("API_REQUEST_TIMEOUT_SECONDS", str(cls.request_timeout_seconds))
            ),
            max_characters=int(os.getenv("API_MAX_CHARACTERS", str(cls.max_characters))),
            policy_ttl_seconds=int(os.getenv("API_POLICY_TTL_SECONDS", str(cls.policy_ttl_seconds))),
        )
