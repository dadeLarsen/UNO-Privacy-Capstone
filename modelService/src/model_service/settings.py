"""Runtime configuration for the local model service.

Every value can be set with an environment variable; the command line can override
the condition, device, host, port, and preload behavior. The service refuses to bind
to anything except a loopback address (NFR-02).
"""

from dataclasses import dataclass
import os
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SERVICE_ROOT.parent
DEFAULT_CONDITION = (
    REPO_ROOT
    / "privacy-research"
    / "conditions"
    / "classification"
    / "appsi139"
    / "embeddinggemma2-bf16-head-linear-balanced-tuned-appsi139-test-half.toml"
)
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, not {value!r}")


def _env_int(name: str, default: int, *, minimum: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer, not {value!r}") from None
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}, not {parsed}")
    return parsed


@dataclass(frozen=True)
class Settings:
    condition: Path = DEFAULT_CONDITION
    device: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    max_characters: int = 200_000
    cache_size: int = 4096
    preload: bool = True
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS

    def __post_init__(self):
        if not isinstance(self.condition, Path):
            object.__setattr__(self, "condition", Path(self.condition))
        if not self.condition.is_file():
            raise ValueError(f"Condition file does not exist: {self.condition}")
        if self.device not in (None, "cuda", "cpu"):
            raise ValueError(f"device must be cuda, cpu, or unset, not {self.device!r}")
        if self.host not in LOOPBACK_HOSTS:
            raise ValueError(f"The service only listens on localhost; refusing host {self.host!r}")
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must be between 1 and 65535, not {self.port}")
        if self.max_characters < 1:
            raise ValueError(f"max_characters must be positive, not {self.max_characters}")
        if self.cache_size < 0:
            raise ValueError(f"cache_size must be nonnegative, not {self.cache_size}")
        if not self.cors_origins:
            raise ValueError("cors_origins must not be empty")

    @classmethod
    def from_env(cls) -> "Settings":
        condition = os.environ.get("MODEL_SERVICE_CONDITION")
        device = os.environ.get("MODEL_SERVICE_DEVICE")
        origins = os.environ.get("MODEL_SERVICE_CORS_ORIGINS")
        return cls(
            condition=Path(condition) if condition else DEFAULT_CONDITION,
            device=(device or "").strip().lower() or None,
            host=os.environ.get("MODEL_SERVICE_HOST", "127.0.0.1").strip(),
            port=_env_int("MODEL_SERVICE_PORT", 8000, minimum=1),
            max_characters=_env_int("MODEL_SERVICE_MAX_CHARACTERS", 200_000, minimum=1),
            cache_size=_env_int("MODEL_SERVICE_CACHE_SIZE", 4096, minimum=0),
            preload=_env_bool("MODEL_SERVICE_PRELOAD", True),
            cors_origins=(
                tuple(origin.strip() for origin in origins.split(",") if origin.strip())
                if origins
                else DEFAULT_CORS_ORIGINS
            ),
        )
