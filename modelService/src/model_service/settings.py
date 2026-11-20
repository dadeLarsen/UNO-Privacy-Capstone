"""Runtime configuration for the local model service.

Every value can be set with an environment variable; the command line can override
the condition, device, host, port, and preload behavior. The service refuses to bind
to anything except a loopback address (NFR-02).

Without a condition, the service runs one of two default conditions, the same fitted
head and the same GGUF file through llama.cpp on the GPU or on the CPU. The device
picks between them and is detected when unset. A condition names its own device, so
it cannot be combined with one.
"""

from dataclasses import dataclass
import os
from pathlib import Path

from privacy_bench.models.gguf_chat_model import gpu_offload_supported

SERVICE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SERVICE_ROOT.parent
CONDITIONS = REPO_ROOT / "privacy-research" / "conditions" / "classification" / "appsi139"
# The benchmark's conditions of this file take 8,192 tokens, and llama.cpp sizes its buffers for the limit.
# These take 2,048, four times the longest sentence in both datasets.
DEFAULT_CONDITIONS = {
    "gpu": CONDITIONS / "embeddinggemma2-gguf-q8-gpu-head-linear-balanced-tuned-limit2048-appsi139-test-half.toml",
    "cpu": CONDITIONS / "embeddinggemma2-gguf-q8-cpu-head-linear-balanced-tuned-limit2048-appsi139-test-half.toml",
}
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


def detect_device() -> str:
    """The device whose default condition this computer can run."""
    # llama.cpp answers for its own build: CUDA on Windows and Linux, Metal on a Mac. A CUDA build that sees
    # no NVIDIA GPU answers no, so a computer without one takes the CPU condition.
    return "gpu" if gpu_offload_supported() else "cpu"


@dataclass(frozen=True)
class Settings:
    # Holds only a condition that was given, never a default: replace() runs __post_init__ again, and a
    # default stored here would then look like a condition given together with a device.
    condition: Path | None = None
    device: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    max_characters: int = 200_000
    cache_size: int = 4096
    preload: bool = True
    # Seconds without a request before the model is released (NFR-03).
    idle_release_seconds: int = 300
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS

    def __post_init__(self):
        if self.condition is not None and not isinstance(self.condition, Path):
            object.__setattr__(self, "condition", Path(self.condition))
        if self.device not in (None, "gpu", "cpu"):
            raise ValueError(f"device must be gpu, cpu, or unset, not {self.device!r}")
        if self.condition is not None and self.device is not None:
            raise ValueError(
                f"Set a condition or a device, not both: got condition {self.condition} and device "
                f"{self.device!r}. A condition names its own device; a device only picks between the two "
                "default conditions. Check --condition, --device, MODEL_SERVICE_CONDITION, and "
                "MODEL_SERVICE_DEVICE."
            )
        if not self.condition_path.is_file():
            raise ValueError(f"Condition file does not exist: {self.condition_path}")
        if self.host not in LOOPBACK_HOSTS:
            raise ValueError(f"The service only listens on localhost; refusing host {self.host!r}")
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must be between 1 and 65535, not {self.port}")
        if self.max_characters < 1:
            raise ValueError(f"max_characters must be positive, not {self.max_characters}")
        if self.cache_size < 0:
            raise ValueError(f"cache_size must be nonnegative, not {self.cache_size}")
        if self.idle_release_seconds < 1:
            raise ValueError(f"idle_release_seconds must be positive, not {self.idle_release_seconds}")
        if not self.cors_origins:
            raise ValueError("cors_origins must not be empty")

    @property
    def condition_path(self) -> Path:
        """The condition to run: the one given, else the default for the device, detected when unset."""
        return self.condition or DEFAULT_CONDITIONS[self.device or detect_device()]

    @classmethod
    def from_env(cls) -> "Settings":
        condition = os.environ.get("MODEL_SERVICE_CONDITION")
        device = os.environ.get("MODEL_SERVICE_DEVICE")
        origins = os.environ.get("MODEL_SERVICE_CORS_ORIGINS")
        return cls(
            condition=Path(condition) if condition else None,
            device=(device or "").strip().lower() or None,
            host=os.environ.get("MODEL_SERVICE_HOST", "127.0.0.1").strip(),
            port=_env_int("MODEL_SERVICE_PORT", 8000, minimum=1),
            max_characters=_env_int("MODEL_SERVICE_MAX_CHARACTERS", 200_000, minimum=1),
            cache_size=_env_int("MODEL_SERVICE_CACHE_SIZE", 4096, minimum=0),
            preload=_env_bool("MODEL_SERVICE_PRELOAD", True),
            idle_release_seconds=_env_int("MODEL_SERVICE_IDLE_RELEASE_SECONDS", 300, minimum=1),
            cors_origins=(
                tuple(origin.strip() for origin in origins.split(",") if origin.strip())
                if origins
                else DEFAULT_CORS_ORIGINS
            ),
        )
