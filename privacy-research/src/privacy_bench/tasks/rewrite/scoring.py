"""Checks that flag rewrites worth a second look, and the files a reviewer reads.

There is no reference rewrite, so nothing here measures quality. A rewrite is valid when it is
nonempty, finished before the token limit, and shows no reasoning tags. The numbers and qualifiers
checks find what the source says that the rewrite no longer does (FR-15); passing them proves nothing.
"""

import csv
from pathlib import Path
import re

from ...core.files import write_json

NUMBER = re.compile(r"\d+(?:[.,:/]\d+)*")
# Section labels such as "8.2 We ..." open a line or sentence; a summary should drop them, not keep them.
SECTION_LABEL = re.compile(r"(?:^|(?<=[.!?_]\s))\d+(?:\.\d+)+(?=\s)", re.M)
# Words whose loss can change what a sentence promises: hedges, conditions, exceptions, and negation.
QUALIFIERS = ("may", "might", "could", "except", "unless", "only", "if", "not", "never", "without")
QUALIFIER = re.compile(r"\b(" + "|".join(QUALIFIERS) + r")\b")
# Qwen, Granite, and LFM think inside <think>; Gemma 4 thinks in a <|channel>thought ... <channel|> block.
REASONING_TAGS = ("<think>", "</think>", "<|channel>", "<channel|>")


def output_text(response: dict | None) -> str:
    return (response or {}).get("raw_output", "").strip()


def status(response: dict | None) -> str:
    if response is None:
        return "missing"
    if response.get("error"):
        return "error"
    text = output_text(response)
    if not text:
        return "empty"
    if any(tag in text for tag in REASONING_TAGS):
        return "reasoning"
    return "truncated" if response.get("reached_output_token_limit") else "valid"


def qualifiers(text: str) -> set:
    text = text.lower().replace("’", "'").replace("cannot", "can not").replace("n't", " not")
    return set(QUALIFIER.findall(text))


def check(source: str, output: str) -> dict:
    numbers, kept_numbers = set(NUMBER.findall(SECTION_LABEL.sub(" ", source))), set(NUMBER.findall(output))
    words, kept_words = qualifiers(source), qualifiers(output)
    return {
        "numbers_in_source": len(numbers), "numbers_kept": len(numbers & kept_numbers),
        "missing_numbers": " ".join(sorted(numbers - kept_numbers)),
        "qualifiers_in_source": len(words), "qualifiers_kept": len(words & kept_words),
        "missing_qualifiers": " ".join(sorted(words - kept_words)),
    }


def evaluate(passages: list, responses: list) -> tuple:
    by_id = {response["id"]: response for response in responses}
    details = []
    for passage in passages:
        response = by_id.get(passage.id)
        output = output_text(response)
        source_words, output_words = len(passage.text.split()), len(output.split())
        details.append({
            "id": passage.id, "heading": passage.heading, "status": status(response),
            "source_words": source_words, "output_words": output_words,
            "compression": output_words / source_words, **check(passage.text, output),
            "output_tokens": (response or {}).get("output_tokens"), "seconds": (response or {}).get("seconds"),
        })
    valid = [detail for detail in details if detail["status"] == "valid"]
    counts = {name: sum(detail["status"] == name for detail in details)
              for name in ("valid", "truncated", "reasoning", "empty", "error", "missing")}

    def pooled(kept, total):
        found = sum(detail[total] for detail in details)
        return sum(detail[kept] for detail in details) / found if found else None

    metrics = {
        "n": len(details), "status_counts": counts, "valid_output_rate": len(valid) / len(details),
        "mean_output_words": sum(detail["output_words"] for detail in valid) / len(valid) if valid else None,
        "mean_compression": sum(detail["compression"] for detail in valid) / len(valid) if valid else None,
        "numbers_kept_rate": pooled("numbers_kept", "numbers_in_source"),
        "qualifiers_kept_rate": pooled("qualifiers_kept", "qualifiers_in_source"),
        "checks_scope": "numbers other than section labels, and qualifiers, pooled over every passage; "
                        "an invalid rewrite keeps what it shows",
        "qualifiers": list(QUALIFIERS),
    }
    return metrics, details


def review_markdown(passages: list, columns: list) -> str:
    """Each passage, then every column's rewrite of it; columns are (label, responses by ID)."""
    lines = []
    for number, passage in enumerate(passages, 1):
        lines += [f"## {number}. {passage.heading}", "", f"Passage `{passage.id}`, {len(passage.text.split())} words",
                  "", *[f"> {line}" for line in passage.text.splitlines()], ""]
        for label, by_id in columns:
            response = by_id.get(passage.id)
            output, flags = output_text(response), check(passage.text, output_text(response))
            notes = [f"{len(output.split())} words"]
            if status(response) != "valid":
                notes.append(f"**{status(response)}**")
            if flags["missing_numbers"]:
                notes.append(f"numbers dropped: {flags['missing_numbers']}")
            if flags["missing_qualifiers"]:
                notes.append(f"qualifiers dropped: {flags['missing_qualifiers']}")
            lines += [f"### {label}", "", f"*{'; '.join(notes)}*", "", output or "*(no output)*", ""]
    return "\n".join(lines)


def write_evaluation(output, passages: list, responses: list) -> dict:
    """Write metrics.json, details.csv, and review.md, which shows each passage above its rewrite."""
    output = Path(output)
    metrics, details = evaluate(passages, responses)
    write_json(output / "metrics.json", metrics)
    with (output / "details.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    by_id = {response["id"]: response for response in responses}
    (output / "review.md").write_text(review_markdown(passages, [("Rewrite", by_id)]), encoding="utf-8")
    return metrics
