"""Fitted weights that a strategy loads, such as a classification head, and the fit records that hold them.

A fit record directory, ``results/fits/<task>/<fit name>/<UTC start>/``, holds::

    fit.toml              the fit file: its task, its kind of fit, and one table per setting
    fit.json              identity, provenance, status, and the task's report on the fit
    weights.safetensors   the fitted tensors; fit.json records their names, shapes, and file hash

Records are only read here. The code that fitted them is retired; History in README.md names the last commit
that has it.
"""

from dataclasses import dataclass
from pathlib import Path
import tomllib

from safetensors.numpy import load_file

from .conditions import NAME_PATTERN
from .files import digest_bytes, digest_json, read_json
from .paths import FITS_DIR

FIT_SCHEMA_VERSION = 1
WEIGHTS_FILE = "weights.safetensors"


@dataclass(frozen=True)
class FitSpec:
    name: str
    task: str
    kind: str
    tables: dict
    source_text: str

    def as_dict(self) -> dict:
        return {"name": self.name, "task": self.task, "kind": self.kind, **self.tables}

    @property
    def sha256(self) -> str:
        return digest_json(self.as_dict())


def parse_fit_spec(name: str, text: str) -> FitSpec:
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Fit name {name!r} must use lowercase letters, digits, '.' and '-'")
    try:
        values = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Fit file {name!r} is not valid TOML: {exc}") from exc
    for key in ("task", "kind"):
        if not isinstance(values.get(key), str) or not NAME_PATTERN.fullmatch(values[key]):
            raise ValueError(f"Fit file {name!r} needs a lowercase {key} name")
    tables = {key: value for key, value in values.items() if key not in ("task", "kind")}
    loose = sorted(key for key, value in tables.items() if not isinstance(value, dict))
    if loose:
        raise ValueError(f"Fit file {name!r} allows only task and kind outside tables: {', '.join(loose)}")
    return FitSpec(name=name, task=values["task"], kind=values["kind"], tables=tables, source_text=text)


def resolve_fit(reference: str, *, root=FITS_DIR) -> Path:
    """The record directory for a reference such as 'classification/<fit name>/<UTC start>'."""
    parts = reference.split("/") if isinstance(reference, str) else []
    if (len(parts) != 3 or "\\" in reference or any(part in ("", ".", "..") for part in parts)
            or Path(reference).is_absolute()):
        raise ValueError(f"A fit reference is '<task>/<fit name>/<UTC start>' under results/fits/: {reference!r}")
    return Path(root).joinpath(*parts)


@dataclass(frozen=True)
class Fit:
    path: Path
    record: dict
    tensors: dict

    @property
    def reference(self) -> str:
        return f"{self.record['task']}/{self.record['fit']['name']}/{self.path.name}"


def load_fit(path) -> Fit:
    """Load a complete fit record, checking its fit file and weights against the hashes it recorded."""
    path = Path(path)
    if not (path / "fit.json").is_file():
        raise ValueError(f"No fit record at {path}")
    record = read_json(path / "fit.json")
    if not isinstance(record, dict) or record.get("schema_version") != FIT_SCHEMA_VERSION:
        raise ValueError(f"Not a schema {FIT_SCHEMA_VERSION} fit record: {path}")
    if record.get("status") != "complete":
        raise ValueError(f"Fit is not complete (status {record.get('status')!r}): {path}")
    spec = parse_fit_spec(record["fit"]["name"], (path / "fit.toml").read_text(encoding="utf-8"))
    if spec.sha256 != record["fit_sha256"]:
        raise ValueError(f"fit.toml differs from the fit file recorded in fit.json: {path}")
    if digest_bytes((path / WEIGHTS_FILE).read_bytes()) != record["weights_sha256"]:
        raise ValueError(f"{WEIGHTS_FILE} differs from the hash recorded in fit.json: {path}")
    tensors = load_file(str(path / WEIGHTS_FILE))
    if {name: list(value.shape) for name, value in tensors.items()} != record["tensors"]:
        raise ValueError(f"{WEIGHTS_FILE} tensors differ from the names and shapes in fit.json: {path}")
    return Fit(path, record, tensors)
