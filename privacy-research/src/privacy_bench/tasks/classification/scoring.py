"""Strict response parsing and scoring; missing and invalid outputs stay in the denominator."""

from collections import Counter
import csv
import json
from pathlib import Path
import re

from ...core.files import unique_keys, write_json
from .data import accepted_gold_digest, normalize_accepted_labels
from .taxonomy import HEADER, LABELS

RESPONSE_PARSING = "json_or_single_markdown_fence_v1"
DETAIL_COST_FIELDS = ("input_tokens", "output_tokens", "seconds", "peak_allocated_bytes", "peak_reserved_bytes")


def parse_response(raw) -> int:
    """Validate one category object, optionally wrapped in one whole-response code fence."""
    if not isinstance(raw, str):
        raise ValueError("Response must be text")
    # Permit presentation-only fences, without extracting JSON from prose or
    # repairing its contents. The same schema checks apply to both formats.
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```",
                          raw.strip(), flags=re.IGNORECASE | re.DOTALL)
    payload = fenced.group(1) if fenced else raw
    try:
        value = json.loads(payload, object_pairs_hook=unique_keys)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"label_id"}:
        raise ValueError("Expected exactly one JSON key: label_id")
    label_id = value["label_id"]
    if type(label_id) is not int or not 0 <= label_id < len(LABELS):
        raise ValueError(f"label_id must be an integer from 0 through {len(LABELS) - 1}")
    return label_id


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def evaluate(rows, responses, accepted_labels=None, strata=None):
    """Join responses to rows by ID and score them; returns metrics, details, and confusion.

    A prediction is correct when it matches any original label of its sentence. Per-category precision,
    recall and F1, and the confusion matrix, use the sentences with exactly one original label. With
    accepted labels, macro accuracy averages accepted accuracy per category, grouping each sentence by the
    category a stratified sample drew it for, or otherwise under each of its original labels.
    """
    if not rows or len({row.id for row in rows}) != len(rows):
        raise ValueError("Evaluation requires nonempty rows with unique IDs")
    if any(not row.labels for row in rows):
        raise ValueError("Every scored sentence needs at least one original label")
    if accepted_labels is not None:
        accepted_labels = normalize_accepted_labels(rows, accepted_labels)
    if strata is not None and (set(strata) != {row.id for row in rows} or any(
            type(value) is not int or not 0 <= value < len(LABELS) for value in strata.values())):
        raise ValueError("Strata must give one valid category for every scored sentence")
    expected = {row.id for row in rows}
    indexed = {}
    for response in responses:
        if not isinstance(response, dict) or type(response.get("id")) is not str:
            raise ValueError("Each response must have a string id")
        row_id = response["id"]
        if row_id not in expected or row_id in indexed:
            raise ValueError(f"Unknown or duplicate response id: {row_id}")
        indexed[row_id] = response
    details, confusion, single_label_rows = [], [[0] * (len(LABELS) + 1) for _ in LABELS], 0
    for row in rows:
        response = indexed.get(row.id)
        predicted, error, status = None, "", "valid"
        if response is None:
            status, error = "missing", "No response recorded"
            response = {}
        elif response.get("error"):
            status, error = "error", str(response["error"])
        else:
            try:
                predicted = parse_response(response.get("raw_output"))
            except ValueError as exc:
                status, error = "invalid", str(exc)
        if len(row.labels) == 1:
            single_label_rows += 1
            confusion[row.labels[0]][predicted if predicted is not None else len(LABELS)] += 1
        details.append({
            "id": row.id, "text": row.text, "gold_label_ids": json.dumps(list(row.labels)),
            "gold_labels": "; ".join(LABELS[i] for i in row.labels), "predicted_label_id": predicted,
            "predicted_label": LABELS[predicted] if predicted is not None else "",
            "status": status, "correct": int(predicted in row.labels),
            "raw_output": response.get("raw_output", ""), "error": error,
            **{key: response.get(key) for key in DETAIL_COST_FIELDS},
        })
        if strata is not None:
            details[-1]["stratum"] = LABELS[strata[row.id]]
        if accepted_labels is not None:
            details[-1].update({
                "accepted_correct": int(predicted in accepted_labels[row.id]),
                "accepted_label_ids": json.dumps(sorted(accepted_labels[row.id])),
            })
    categories, total_tp, total_fp, total_fn = [], 0, 0, 0
    for i, label in enumerate(LABELS):
        tp = confusion[i][i]
        support = sum(confusion[i])
        fp = sum(row[i] for row in confusion) - tp
        fn = support - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn
        categories.append({"label_id": i, "label": label, "support": support,
                           "tp": tp, "fp": fp, "fn": fn,
                           "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
                           "f1": _ratio(2 * tp, 2 * tp + fp + fn)})
    statuses = Counter(detail["status"] for detail in details)
    fenced_count = sum(detail["status"] == "valid" and detail["raw_output"].lstrip().startswith("```")
                       for detail in details)
    correct = sum(detail["correct"] for detail in details)
    metrics = {
        "n": len(rows), "recorded_responses": len(indexed), "correct": correct,
        "accuracy": _ratio(correct, len(rows)),
        "single_label_rows": single_label_rows,
        "macro_f1": sum(category["f1"] for category in categories) / len(LABELS),
        "micro_precision": _ratio(total_tp, total_tp + total_fp),
        "micro_recall": _ratio(total_tp, total_tp + total_fn),
        "micro_f1": _ratio(2 * total_tp, 2 * total_tp + total_fp + total_fn),
        "valid_output_rate": _ratio(statuses["valid"], len(rows)),
        "strict_json_valid_output_rate": _ratio(statuses["valid"] - fenced_count, len(rows)),
        "fenced_json_responses": fenced_count,
        "response_parsing": RESPONSE_PARSING,
        "status_counts": {status: statuses[status] for status in ("valid", "invalid", "error", "missing")},
        "per_category": categories,
        "scoring": "Accept one JSON category object, bare or enclosed by a single JSON/untagged Markdown fence; "
                   "reject extra prose, extra fields and malformed category values. "
                   "All sample rows included. A prediction is correct when it matches any original label "
                   "of its sentence. Invalid/missing outputs count as incorrect "
                   "and false negatives; they do not create a false positive for a valid category. "
                   "Precision, recall, F1 and the confusion matrix use the sentences with one original label. "
                   f"Macro F1 averages all {len(LABELS)} classes; zero denominators produce 0.",
    }
    if accepted_labels is not None:
        groups = {i: [] for i in range(len(LABELS))}
        for row, detail in zip(rows, details, strict=True):
            for group in ((strata[row.id],) if strata is not None else row.labels):
                groups[group].append(detail["accepted_correct"])
        for category in categories:
            members = groups[category["label_id"]]
            category["accepted_rows"] = len(members)
            category["accepted_correct"] = sum(members)
            category["accepted_accuracy"] = sum(members) / len(members) if members else None
        supported = [category for category in categories if category["accepted_rows"]]
        accepted_correct = sum(detail["accepted_correct"] for detail in details)
        metrics.update({
            "scoring_mode": "any_accepted_gold_label",
            "accepted_correct": accepted_correct,
            "accepted_accuracy": accepted_correct / len(rows),
            "macro_accuracy": sum(category["accepted_accuracy"] for category in supported) / len(supported),
            "macro_accuracy_groups": "sample strata" if strata is not None else "original labels",
            "accepted_gold_sha256": accepted_gold_digest(rows, accepted_labels),
            "accepted_scoring": "A single predicted label is correct if it matches ANY marked gold label. "
                                "Invalid/missing/error outputs remain incorrect. Macro accuracy averages "
                                "accepted-label accuracy over categories, grouping each sentence by its "
                                "sample stratum, or else under each of its original labels. "
                                "correct, accuracy, F1, confusion and details.correct use the original labels.",
        })
    else:
        metrics["scoring_mode"] = "original_labels"
    return metrics, details, confusion


def write_evaluation(output, rows, responses, accepted_labels=None, strata=None) -> dict:
    """Write metrics.json, details.csv, confusion.csv, and one-hot predictions.csv."""
    output = Path(output)
    metrics, details, confusion = evaluate(rows, responses, accepted_labels, strata)
    write_json(output / "metrics.json", metrics)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for row, detail in zip(rows, details, strict=True):
            label_id = detail["predicted_label_id"]
            # Blank category cells distinguish a failure from a valid prediction.
            cells = ([int(i == label_id) for i in range(len(LABELS))] if label_id is not None
                     else [""] * len(LABELS))
            writer.writerow([row.id, row.text, *cells])
    with (output / "details.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    with (output / "confusion.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold / predicted", *LABELS, "INVALID_OR_MISSING"])
        writer.writerows([[label, *values] for label, values in zip(LABELS, confusion)])
    return metrics
