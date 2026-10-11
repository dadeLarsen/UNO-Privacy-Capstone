"""What produced a run: code revision, interpreter, platform, and installed packages."""

from datetime import datetime, timezone
import importlib.metadata
import platform
import subprocess
import sys

from .files import digest_bytes
from .paths import REPO_ROOT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(*arguments) -> bytes:
    result = subprocess.run(["git", "-C", str(REPO_ROOT), *arguments], capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed with exit code {result.returncode}:\n"
                           + result.stderr.decode("utf-8", "replace"))
    return result.stdout


def code_revision() -> dict:
    """Record the commit and any uncommitted changes, so a dirty run stays traceable."""
    status = _git("status", "--porcelain", "--untracked-files=all", "--", "src", "pyproject.toml", "uv.lock")
    diff = _git("diff", "HEAD", "--binary", "--", "src", "pyproject.toml", "uv.lock")
    return {
        "commit": _git("rev-parse", "HEAD").decode("ascii").strip(),
        "dirty": bool(status.strip()),
        "uncommitted_paths": status.decode("utf-8", "replace").splitlines(),
        "uncommitted_diff_sha256": digest_bytes(diff) if status.strip() else None,
    }


def environment() -> dict:
    packages = {distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions()}
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        "code": code_revision(),
    }
