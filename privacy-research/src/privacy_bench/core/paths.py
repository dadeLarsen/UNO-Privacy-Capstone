"""Locations inside the repository checkout."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
FITS_DIR = RESULTS_DIR / "fits"
COMPARISONS_DIR = RESULTS_DIR / "comparisons"
JUDGMENTS_DIR = RESULTS_DIR / "judgments"
# Prompts stay editable, unlike data/: every run copies the full prompt text into its input.json.
PROMPTS_DIR = REPO_ROOT / "prompts"


def require_checkout():
    """Fail clearly when the package is used outside an editable checkout."""
    if not (REPO_ROOT / "pyproject.toml").is_file() or not DATA_DIR.is_dir():
        raise RuntimeError(f"privacy_bench must run from its repository checkout; no data/ under {REPO_ROOT}")
    return REPO_ROOT


def _named_file(root: Path, kind: str, parts: tuple, suffix: str) -> Path:
    *folders, name = parts
    if not name or any(not part or "/" in part or "\\" in part or part in (".", "..") for part in parts):
        raise ValueError(f"Invalid {kind} file name: {'/'.join(parts)!r}")
    require_checkout()
    path = root.joinpath(*folders, name + suffix)
    if not path.is_file():
        raise ValueError(f"{kind.capitalize()} file does not exist: {path}")
    return path


def data_file(*parts: str, suffix: str) -> Path:
    """Resolve a named file under data/, refusing names that would leave that folder."""
    return _named_file(DATA_DIR, "data", parts, suffix)


def prompt_file(*parts: str, suffix: str) -> Path:
    """Resolve a named file under prompts/, refusing names that would leave that folder."""
    return _named_file(PROMPTS_DIR, "prompt", parts, suffix)
