"""The on-disk run record shared by every task.

A run directory holds::

    condition.toml      verbatim copy of the condition file
    run.json            identity, provenance, status, and the input audit summary
    input.json          the strategy's input specification
    input-audit.json    per-item preflight results
    warmup.json         unscored warmup responses
    responses.jsonl     one response per item, in item order
    evaluation/         derived scores; regenerate with ``pbench evaluate``
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .conditions import Condition, parse_condition
from .files import digest_json, read_json, read_jsonl
from .paths import RUNS_DIR

SCHEMA_VERSION = 3
COST_FIELDS = ("seconds", "input_tokens", "output_tokens", "peak_allocated_bytes", "peak_reserved_bytes")


def run_directory(task: str, condition: str, *, root=RUNS_DIR, started=None) -> Path:
    started = started or datetime.now(timezone.utc)
    return Path(root) / task / condition / started.strftime("%Y%m%d-%H%M%SZ")


@dataclass(frozen=True)
class Run:
    path: Path
    record: dict
    responses: list

    @property
    def name(self) -> str:
        return f"{self.record['condition']['name']}/{self.path.name}"


def run_condition(run: Run) -> Condition:
    """The condition a run was made with, checked against the hash in its record."""
    condition = parse_condition(run.record["condition"]["name"],
                                (run.path / "condition.toml").read_text(encoding="utf-8"))
    if condition.sha256 != run.record["condition_sha256"]:
        raise ValueError(f"condition.toml differs from the condition recorded in run.json: {run.path}")
    return condition


def load_run(path, *, require_complete=True) -> Run:
    path = Path(path)
    record = read_json(path / "run.json")
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Not a schema {SCHEMA_VERSION} run record: {path}")
    if require_complete and record.get("status") != "complete":
        raise ValueError(f"Run is not complete (status {record.get('status')!r}): {path}")
    spec = read_json(path / "input.json")
    if digest_json(spec) != record.get("input_spec_sha256"):
        raise ValueError(f"input.json differs from the hash in run.json: {path}")
    responses = read_jsonl(path / "responses.jsonl")
    ids = record.get("item_ids")
    if not isinstance(ids, list) or len(set(ids)) != len(ids):
        raise ValueError(f"run.json needs unique item_ids: {path}")
    response_ids = [response.get("id") if isinstance(response, dict) else None for response in responses]
    if len(set(response_ids)) != len(response_ids) or not set(response_ids) <= set(ids):
        raise ValueError(f"Responses have duplicate or unknown item IDs: {path}")
    if record.get("recorded_responses") != len(responses):
        raise ValueError(f"run.json response count differs from responses.jsonl: {path}")
    if require_complete:
        if response_ids != ids:
            raise ValueError(f"A complete run needs one response per item, in item order: {path}")
        failed = [response["id"] for response in responses if response.get("error")]
        if failed:
            raise ValueError(f"Complete run has failed responses ({', '.join(failed)}): {path}")
    return Run(path, record, responses)


def cost_summary(responses: list) -> dict:
    """Average the measured costs over the responses that recorded them; absent stays None."""
    summary = {}
    for key in COST_FIELDS:
        values = [response.get(key) for response in responses]
        values = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        summary[f"{key}_count"] = len(values)
        summary[f"mean_{key}"] = sum(values) / len(values) if values else None
        if key.startswith("peak_"):
            summary[f"max_{key}"] = max(values) if values else None
    return summary
