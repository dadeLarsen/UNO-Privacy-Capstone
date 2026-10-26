"""Blind key-point judging: prompts for a judge, checked replies, and scores for comparisons.

A judgment covers one comparison's runs on one passage set. ``prepare`` writes a pending record with
one prompt per passage: the judge instructions, the passage, its numbered key points from
data/rewrite/keypoints/<passages>.json, and every run's valid rewrite under a letter shuffled per
passage, so the judge cannot tell which model wrote which. Invalid rewrites are not judged. A judge
outside pbench answers each prompt in replies/<passage>.json; ``record`` checks every reply against its
prompt, quotes included, and freezes the record. ``load`` and ``run_scores`` then score a run from it.
"""

import json
from pathlib import Path
import random
import re
import string
import tomllib

from ...core.files import digest_bytes, digest_text, read_json, unique_keys, write_json
from ...core.paths import data_file, prompt_file
from ...core.provenance import code_revision, utc_now
from .scoring import output_text, status

SCHEMA_VERSION = 1
VERDICTS = ("kept", "partial", "missing", "contradicted")
SEVERITIES = ("major", "minor")
# These verdicts say the rewrite changed a point, so the judge must quote the words that show it.
QUOTED = ("partial", "contradicted")
LETTERS = string.ascii_uppercase
POINT_KEYS = {"id", "point", "critical", "note"}
QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
ELLIPSIS = re.compile(r"\.\.\.|…")


def load_checklist(passages: list, identity: dict) -> tuple:
    """data/rewrite/keypoints/<passages>.json; returns (points by passage ID, checklist identity)."""
    name = identity["passages"]["name"]
    path = data_file("rewrite", "keypoints", name, suffix=".json")
    values = read_json(path)
    if not isinstance(values, dict) or values.get("name") != name or values.get("passages") != identity["passages"]:
        raise ValueError(f"{path} must name passage set {name!r} and match its SHA-256")
    entries = values.get("checklists")
    if (not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries)
            or [entry.get("passage") for entry in entries] != [passage.id for passage in passages]):
        raise ValueError(f"{path} needs one checklist per passage, in passage order")
    points = {}
    for entry in entries:
        items = entry.get("points")
        if not isinstance(items, list) or not items or not all(
                isinstance(item, dict) and {"id", "point", "critical"} <= set(item) <= POINT_KEYS
                and isinstance(item["critical"], bool)
                and all(isinstance(item.get(key, ""), str) and item.get(key, "x").strip()
                        for key in ("id", "point", "note"))
                for item in items):
            raise ValueError(f"{path}, passage {entry['passage']}: every point needs an id, a point, critical "
                             "true or false, and optionally a note")
        if len({item["id"] for item in items}) != len(items):
            raise ValueError(f"{path}, passage {entry['passage']}: point IDs must be unique")
        points[entry["passage"]] = items
    return points, {"name": name, "sha256": digest_bytes(path.read_bytes()), "reviewed_by": values.get("reviewed_by")}


def load_instructions(name: str) -> dict:
    """prompts/rewrite/judge/<name>.toml, which holds one key, instructions."""
    path = prompt_file("rewrite", "judge", name, suffix=".toml")
    try:
        values = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path} is not valid TOML: {exc}") from exc
    if set(values) != {"instructions"} or not isinstance(values["instructions"], str) or not values["instructions"].strip():
        raise ValueError(f"{path} needs exactly one key, instructions, holding nonempty text")
    text = values["instructions"].strip()
    return {"name": name, "sha256": digest_text(text), "text": text}


def render_prompt(instructions: str, passage, points: list, rewrites: dict) -> str:
    """rewrites maps each letter to its rewrite, in the order the judge reads them."""
    lines = [instructions, "", "Passage:", passage.text, "", "Key points:"]
    for item in points:
        note = f" Note: {item['note']}" if "note" in item else ""
        lines.append(f"{item['id']}. {'(critical) ' if item['critical'] else ''}{item['point']}{note}")
    for label, text in rewrites.items():
        lines += ["", f"Rewrite {label}:", text]
    return "\n".join(lines) + "\n"


def normalized(text: str) -> str:
    return " ".join(text.lower().translate(QUOTES).split())


def quoted(quote, rewrite: str) -> bool:
    """Whether every part of a quote, split at ellipses, appears in the rewrite, ignoring case and spacing."""
    if not isinstance(quote, str):
        return False
    fragments = [normalized(fragment).strip(" \"'.,;:") for fragment in ELLIPSIS.split(quote)]
    target = normalized(rewrite)
    return any(fragments) and all(fragment in target for fragment in fragments if fragment)


def parse_reply(text: str, labels, point_ids: list) -> dict:
    """One passage's reply, checked for form; returns each letter's verdicts, evidence, and unsupported claims."""
    try:
        values = json.loads(text, object_pairs_hook=unique_keys)
    except ValueError as exc:
        raise ValueError(f"not valid JSON: {exc}") from exc
    if not isinstance(values, dict) or set(values) != {"rewrites"} or not isinstance(values["rewrites"], list):
        raise ValueError('the reply must be exactly {"rewrites": [...]}')
    grades = {}
    for entry in values["rewrites"]:
        if not isinstance(entry, dict) or not {"label", "verdicts"} <= set(entry) <= {"label", "verdicts", "evidence",
                                                                                        "unsupported"}:
            raise ValueError("each rewrite needs label and verdicts, and may have evidence and unsupported")
        label = entry["label"]
        if label not in labels or label in grades:
            raise ValueError(f"unexpected or repeated label {label!r}")
        verdicts, evidence, unsupported = entry["verdicts"], entry.get("evidence", {}), entry.get("unsupported", [])
        if (not isinstance(verdicts, dict) or set(verdicts) != set(point_ids)
                or any(verdict not in VERDICTS for verdict in verdicts.values())):
            raise ValueError(f"rewrite {label} needs one of {', '.join(VERDICTS)} for each point: "
                             f"{', '.join(point_ids)}")
        if not isinstance(evidence, dict) or not set(evidence) <= set(point_ids):
            raise ValueError(f"rewrite {label}: evidence must map point numbers to quotes")
        if not isinstance(unsupported, list) or not all(
                isinstance(claim, dict) and set(claim) == {"quote", "claim", "severity"}
                and claim["severity"] in SEVERITIES and isinstance(claim["claim"], str) and claim["claim"].strip()
                for claim in unsupported):
            raise ValueError(f"rewrite {label}: each unsupported claim needs a quote, the claim, and a severity "
                             f"of {' or '.join(SEVERITIES)}")
        grades[label] = {"verdicts": verdicts, "evidence": evidence, "unsupported": unsupported}
    missing = sorted(set(labels) - set(grades))
    if missing:
        raise ValueError(f"no grades for rewrites {', '.join(missing)}")
    return grades


def unquoted(grades: dict, rewrites: dict) -> list:
    """Problems with a reply's quotes: each partial or contradicted verdict, and each claim, quotes its rewrite."""
    problems = []
    for label, grade in grades.items():
        for point, verdict in grade["verdicts"].items():
            if verdict in QUOTED and not quoted(grade["evidence"].get(point), rewrites[label]):
                problems.append(f"rewrite {label}, point {point}: a {verdict} verdict needs a quote from the rewrite")
        for claim in grade["unsupported"]:
            if not quoted(claim["quote"], rewrites[label]):
                problems.append(f"rewrite {label}: the quote {claim['quote']!r} is not in the rewrite")
    return problems


def judged_texts(passage_id: str, runs: dict) -> tuple:
    """Each run's rewrite of a passage when valid, and the status of each it leaves out."""
    texts, skipped = {}, {}
    for path, run in runs.items():
        response = next((response for response in run.responses if response["id"] == passage_id), None)
        if status(response) == "valid":
            texts[path] = output_text(response)
        else:
            skipped[path] = status(response)
    return texts, skipped


def prepare(name: str, runs: dict, selection, instructions: str, judge: str, output: Path) -> Path:
    """Write a pending judgment of runs (path -> run, in comparison order) into a new directory."""
    output = Path(output)
    if output.exists():
        raise ValueError(f"{output} already exists; a judgment is never overwritten")
    if len(runs) > len(LETTERS):
        raise ValueError(f"A judgment holds at most {len(LETTERS)} runs, one letter each")
    if not judge.strip():
        raise ValueError("Name the judge, for the record")
    checklist, checklist_identity = load_checklist(selection.items, selection.identity)
    spec = load_instructions(instructions)
    record = {"schema_version": SCHEMA_VERSION, "task": "rewrite", "name": name, "status": "pending",
              "created_utc": utc_now(), "code": code_revision(), "judge": judge.strip(),
              "runs": [{"path": path, "condition_sha256": run.record["condition_sha256"]} for path, run in runs.items()],
              "items": selection.identity, "checklist": checklist_identity, "instructions": spec, "passages": {}}
    (output / "prompts").mkdir(parents=True)
    (output / "replies").mkdir()
    for passage in selection.items:
        texts, skipped = judged_texts(passage.id, runs)
        order = list(texts)
        random.Random(f"{name}:{passage.id}").shuffle(order)
        labels = dict(zip(LETTERS, order))
        prompt = render_prompt(spec["text"], passage, checklist[passage.id],
                               {label: texts[path] for label, path in labels.items()})
        (output / "prompts" / f"{passage.id}.txt").write_text(prompt, encoding="utf-8", newline="\n")
        record["passages"][passage.id] = {
            "prompt_sha256": digest_text(prompt),
            "labels": {label: {"run": path, "rewrite_sha256": digest_text(texts[path])} for label, path in labels.items()},
            "not_judged": skipped,
        }
    write_json(output / "judgment.json", record)
    return output


def read_record(directory: Path, selection, *, recorded: bool) -> tuple:
    """A judgment's record and checklist, with its prompts and, once recorded, its replies checked by hash."""
    record = read_json(directory / "judgment.json")
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Not a schema {SCHEMA_VERSION} judgment: {directory}")
    if record["status"] != ("recorded" if recorded else "pending"):
        raise ValueError(f"Judgment {directory} is {record['status']}; "
                         + ("record it first with pbench judge record" if recorded else "it is already recorded"))
    if record["items"] != selection.identity:
        raise ValueError(f"Judgment {directory} covers other passages than these runs")
    checklist, identity = load_checklist(selection.items, selection.identity)
    if identity != record["checklist"]:
        raise ValueError(f"The checklist {identity['name']} changed after judgment {directory} used it")
    for passage_id, entry in record["passages"].items():
        prompt = (directory / "prompts" / f"{passage_id}.txt").read_text(encoding="utf-8")
        if digest_text(prompt) != entry["prompt_sha256"]:
            raise ValueError(f"The prompt for {passage_id} differs from the hash in {directory}/judgment.json")
        if recorded and digest_text(reply_text(directory, passage_id)) != record["replies"][passage_id]:
            raise ValueError(f"The reply for {passage_id} differs from the hash in {directory}/judgment.json")
    return record, checklist


def reply_text(directory: Path, passage_id: str) -> str:
    return (directory / "replies" / f"{passage_id}.json").read_text(encoding="utf-8")


def record(directory: Path, runs: dict, selection) -> dict:
    """Check every reply against its prompt and the runs, then freeze the judgment; returns its record."""
    directory = Path(directory)
    judgment, checklist = read_record(directory, selection, recorded=False)
    problems = []
    for path, run in runs.items():
        expected = next(entry["condition_sha256"] for entry in judgment["runs"] if entry["path"] == path)
        if run.record["condition_sha256"] != expected:
            problems.append(f"run {path} no longer matches the judgment")
    for passage_id, entry in judgment["passages"].items():
        texts, _ = judged_texts(passage_id, runs)
        rewrites = {label: texts.get(item["run"], "") for label, item in entry["labels"].items()}
        if any(digest_text(rewrites[label]) != item["rewrite_sha256"] for label, item in entry["labels"].items()):
            problems.append(f"{passage_id}: a rewrite differs from the one the judge saw")
            continue
        try:
            grades = parse_reply(reply_text(directory, passage_id), rewrites,
                                 [item["id"] for item in checklist[passage_id]])
        except (ValueError, OSError) as exc:
            problems.append(f"{passage_id}: {exc}")
            continue
        problems += [f"{passage_id}: {problem}" for problem in unquoted(grades, rewrites)]
    if problems:
        raise ValueError("The judgment cannot be recorded:\n" + "\n".join(problems))
    judgment.update(status="recorded", recorded_utc=utc_now(),
                    replies={passage_id: digest_text(reply_text(directory, passage_id))
                             for passage_id in judgment["passages"]})
    write_json(directory / "judgment.json", judgment)
    return judgment


def load(directory: Path, selection) -> dict:
    """A recorded judgment: its record, checklist, each passage's grades by run path, and its own hash."""
    directory = Path(directory)
    judgment, checklist = read_record(directory, selection, recorded=True)
    grades = {}
    for passage_id, entry in judgment["passages"].items():
        by_label = parse_reply(reply_text(directory, passage_id), entry["labels"],
                               [item["id"] for item in checklist[passage_id]])
        grades[passage_id] = {entry["labels"][label]["run"]: grade for label, grade in by_label.items()}
    return {"record": judgment, "checklist": checklist, "grades": grades,
            "sha256": digest_text((directory / "judgment.json").read_text(encoding="utf-8"))}


def run_scores(judgment: dict, run) -> dict:
    """What the judge found in one run's rewrites; shares count key points, rates count rewrites."""
    record = judgment["record"]
    matches = [entry for entry in record["runs"] if run.path.as_posix().endswith("/" + entry["path"])]
    if len(matches) != 1 or matches[0]["condition_sha256"] != run.record["condition_sha256"]:
        raise ValueError(f"Run {run.path} is not one judgment {record['name']} judged")
    path = matches[0]["path"]
    counts = {scope: dict.fromkeys(VERDICTS, 0) for scope in ("keypoints", "critical")}
    judged = misleading = major = 0
    for passage_id, entry in record["passages"].items():
        texts, skipped = judged_texts(passage_id, {path: run})
        labelled = [item for item in entry["labels"].values() if item["run"] == path]
        if [item["rewrite_sha256"] for item in labelled] != [digest_text(text) for text in texts.values()] or (
                skipped and entry["not_judged"].get(path) != skipped[path]):
            raise ValueError(f"Run {path} no longer matches judgment {record['name']} on passage {passage_id}")
        grade = judgment["grades"][passage_id].get(path)
        if grade is None:
            continue
        judged += 1
        critical = {item["id"] for item in judgment["checklist"][passage_id] if item["critical"]}
        for point, verdict in grade["verdicts"].items():
            counts["keypoints"][verdict] += 1
            if point in critical:
                counts["critical"][verdict] += 1
        has_major = any(claim["severity"] == "major" for claim in grade["unsupported"])
        major += has_major
        misleading += has_major or any(grade["verdicts"][point] == "contradicted" for point in critical)
    scores = {"judged_rewrites": judged}
    for scope, found in counts.items():
        total = sum(found.values())
        scores[f"{scope}_judged"] = total
        scores.update({f"{scope}_{verdict}_share": found[verdict] / total if total else None for verdict in VERDICTS})
    scores["misleading_rate"] = misleading / judged if judged else None
    scores["major_unsupported_rate"] = major / judged if judged else None
    return scores
