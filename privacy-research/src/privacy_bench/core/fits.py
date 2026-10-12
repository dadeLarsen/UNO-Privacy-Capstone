"""Fitted weights that a strategy loads, such as a classification head, and the fit files behind them.

A fit file names the task and the kind of fit; the task validates every table in it. Example::

    task = "classification"
    kind = "embedding-heads"

    [examples]
    dataset = "opp115-sentences"
    examples = "opp115-examples"

A fit record directory, ``results/fits/<task>/<fit name>/<UTC start>/``, holds::

    fit.toml              verbatim copy of the fit file
    fit.json              identity, provenance, status, and the task's report on the fit
    weights.safetensors   the fitted tensors; fit.json records their names, shapes, and file hash
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import tomllib

import numpy as np
from safetensors.numpy import load_file, save_file

from .conditions import NAME_PATTERN
from .files import digest_bytes, digest_json, read_json, write_json
from .paths import FITS_DIR
from .protocols import Task
from .provenance import environment, utc_now

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


def load_fit_spec(path) -> FitSpec:
    path = Path(path)
    if path.suffix != ".toml":
        raise ValueError(f"Fit files use the .toml extension: {path}")
    return parse_fit_spec(path.stem, path.read_text(encoding="utf-8"))


def fit_directory(task: str, name: str, *, root=FITS_DIR, started=None) -> Path:
    started = started or datetime.now(timezone.utc)
    return Path(root) / task / name / started.strftime("%Y%m%d-%H%M%SZ")


def resolve_fit(reference: str, *, root=FITS_DIR) -> Path:
    """The record directory for a reference such as 'classification/<fit name>/<UTC start>'."""
    parts = reference.split("/") if isinstance(reference, str) else []
    if (len(parts) != 3 or "\\" in reference or any(part in ("", ".", "..") for part in parts)
            or Path(reference).is_absolute()):
        raise ValueError(f"A fit reference is '<task>/<fit name>/<UTC start>' under results/fits/: {reference!r}")
    return Path(root).joinpath(*parts)


def execute_fit(task: Task, spec: FitSpec, output: Path) -> dict:
    """Fit what a fit file describes into a new record directory and return its fit.json content."""
    if spec.task != task.name:
        raise ValueError(f"Fit file {spec.name!r} is for task {spec.task!r}, not {task.name!r}")
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output already exists; choose a new fit directory: {output}")
    record = {
        "schema_version": FIT_SCHEMA_VERSION, "task": task.name, "kind": spec.kind,
        "fit": spec.as_dict(), "fit_sha256": spec.sha256, "environment": environment(),
        "status": "running", "started_utc": utc_now(),
    }
    output.mkdir(parents=True)
    (output / "fit.toml").write_text(spec.source_text, encoding="utf-8")
    write_json(output / "fit.json", record)
    try:
        fitted = task.fit(spec)
        tensors = fitted["tensors"]
        if not tensors or any(not isinstance(value, np.ndarray) or value.dtype != np.float32
                              for value in tensors.values()):
            raise ValueError("A fit must return named float32 arrays")
        # safetensors writes an array's buffer as row-major, so a column-major array, such as
        # scikit-learn's coef_, would load scrambled. Store row-major copies and read them back.
        tensors = {name: np.ascontiguousarray(value) for name, value in tensors.items()}
        save_file(tensors, str(output / WEIGHTS_FILE))
        saved = load_file(str(output / WEIGHTS_FILE))
        if set(saved) != set(tensors) or any(not np.array_equal(saved[name], value) for name, value in tensors.items()):
            raise ValueError(f"{WEIGHTS_FILE} does not read back as the fitted tensors")
        record.update({
            "report": fitted["report"],
            "tensors": {name: list(value.shape) for name, value in sorted(tensors.items())},
            "weights_sha256": digest_bytes((output / WEIGHTS_FILE).read_bytes()),
            "status": "complete",
        })
    except BaseException as exc:
        record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["finished_utc"] = utc_now()
        write_json(output / "fit.json", record)
        print(f"Saved: {output.resolve()}", flush=True)
    return record


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
