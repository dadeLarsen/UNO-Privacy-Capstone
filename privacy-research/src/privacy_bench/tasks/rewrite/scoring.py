"""Checks that flag rewrites worth a second look, and the files a reviewer reads.

There is no reference rewrite, so nothing here measures quality. A rewrite is valid when it is
nonempty, finished before the token limit, and shows no reasoning tags. The numbers and qualifiers
checks find what the source says that the rewrite no longer does (FR-15); passing them proves nothing.
Copying, reading level, and format describe how a rewrite is written, not whether it is right.
"""

import csv
from pathlib import Path
import re

import pyphen

from ...core.files import write_json

NUMBER = re.compile(r"\d+(?:[.,:/]\d+)*")
# Section labels such as "8.2 We ..." open a line or sentence; a summary should drop them, not keep them.
SECTION_LABEL = re.compile(r"(?:^|(?<=[.!?_]\s))\d+(?:\.\d+)+(?=\s)", re.M)
# Words whose loss can change what a sentence promises: hedges, conditions, exceptions, and negation.
QUALIFIERS = ("may", "might", "could", "except", "unless", "only", "if", "not", "never", "without")
QUALIFIER = re.compile(r"\b(" + "|".join(QUALIFIERS) + r")\b")
# Qwen, Granite, and LFM think inside <think>; Gemma 4 thinks in a <|channel>thought ... <channel|> block.
REASONING_TAGS = ("<think>", "</think>", "<|channel>", "<channel|>")
TOKEN = re.compile(r"\w+")
MARKDOWN = re.compile(r"\*\*|__|^#{1,6}\s|^\s*(?:[-*•]|\d+[.)])\s", re.M)
# shortcut: the stock openers these runs produced; extend it when a model opens some other way.
INTRODUCTION = re.compile(r"(?:\*\*)?(?:here(?:['’]s| is| are)\b|sure\b|certainly\b|of course\b|okay\b|"
                          r"below is\b|summary\s*:)", re.I)
WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
SENTENCE_END = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s|$)")
# Hyphenation points plus one estimate syllables; pyphen ships its dictionaries, so nothing downloads.
HYPHENATOR = pyphen.Pyphen(lang="en_US")


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


def copying(source: str, output: str) -> dict:
    """Grusky et al. (2018): the share of output words inside a run copied from the source (coverage),
    and the copied-run length an average output word sits in (density)."""
    source, output = TOKEN.findall(source.lower()), TOKEN.findall(output.lower())
    if not output:
        return {"copy_coverage": None, "copy_density": None}
    runs, i = [], 0
    while i < len(output):
        best, j = 0, 0
        while j < len(source):
            if output[i] == source[j]:
                a, b = i, j
                while a < len(output) and b < len(source) and output[a] == source[b]:
                    a, b = a + 1, b + 1
                best, j = max(best, a - i), b
            else:
                j += 1
        runs.append(best)
        i += max(best, 1)
    return {"copy_coverage": sum(runs) / len(output), "copy_density": sum(n * n for n in runs) / len(output)}


def readability_counts(text: str) -> tuple:
    """Words, sentences, syllables, and words of three or more syllables, with Markdown removed."""
    words = sentences = syllables = long_words = 0
    for line in MARKDOWN.sub("", text).splitlines():
        found, line = WORD.findall(line), line.rstrip()
        if not found:
            continue
        counts = [len(HYPHENATOR.positions(word.lower())) + 1 for word in found]
        words, syllables = words + len(found), syllables + sum(counts)
        long_words += sum(count >= 3 for count in counts)
        # A line without closing punctuation, such as a list item or a heading, still ends a sentence.
        ends = [match.end() for match in SENTENCE_END.finditer(line)]
        sentences += len(ends) + (not ends or ends[-1] != len(line))
    return words, sentences, syllables, long_words


def reading_level(texts) -> dict:
    """Flesch-Kincaid grade pooled over the texts, with the two things that drive it."""
    words, sentences, syllables, long_words = (sum(column) for column in
                                               zip((0, 0, 0, 0), *map(readability_counts, texts)))
    if not words:
        return {"reading_grade": None, "words_per_sentence": None, "long_word_rate": None}
    return {"reading_grade": 0.39 * words / sentences + 11.8 * syllables / words - 15.59,
            "words_per_sentence": words / sentences, "long_word_rate": long_words / words}


def format_flags(output: str) -> dict:
    return {"introduction": bool(INTRODUCTION.match(output)), "markdown": bool(MARKDOWN.search(output))}


def evaluate(passages: list, responses: list) -> tuple:
    by_id = {response["id"]: response for response in responses}
    details, outputs = [], {}
    for passage in passages:
        response = by_id.get(passage.id)
        output = outputs[passage.id] = output_text(response)
        source_words, output_words = len(passage.text.split()), len(output.split())
        details.append({
            "id": passage.id, "heading": passage.heading, "status": status(response),
            "source_words": source_words, "output_words": output_words,
            "compression": output_words / source_words, **check(passage.text, output),
            **copying(passage.text, output), "reading_grade": reading_level([output])["reading_grade"],
            **format_flags(output),
            "output_tokens": (response or {}).get("output_tokens"), "seconds": (response or {}).get("seconds"),
        })
    valid = [detail for detail in details if detail["status"] == "valid"]
    counts = {name: sum(detail["status"] == name for detail in details)
              for name in ("valid", "truncated", "reasoning", "empty", "error", "missing")}

    def pooled(kept, total):
        found = sum(detail[total] for detail in details)
        return sum(detail[kept] for detail in details) / found if found else None

    def mean(field):
        return sum(detail[field] for detail in valid) / len(valid) if valid else None

    metrics = {
        "n": len(details), "status_counts": counts, "valid_output_rate": len(valid) / len(details),
        "mean_output_words": mean("output_words"), "mean_compression": mean("compression"),
        "numbers_kept_rate": pooled("numbers_kept", "numbers_in_source"),
        "qualifiers_kept_rate": pooled("qualifiers_kept", "qualifiers_in_source"),
        "mean_copy_coverage": mean("copy_coverage"), "mean_copy_density": mean("copy_density"),
        **reading_level(outputs[detail["id"]] for detail in valid),
        **{f"source_{key}": value for key, value in reading_level(passage.text for passage in passages).items()},
        "introduction_rate": mean("introduction"), "markdown_rate": mean("markdown"),
        "format_break_rate": (sum(detail["introduction"] or detail["markdown"] for detail in valid) / len(valid)
                              if valid else None),
        "checks_scope": "numbers other than section labels, and qualifiers, pooled over every passage; "
                        "an invalid rewrite keeps what it shows",
        "style_scope": "copying, format, and reading level over valid rewrites; reading level pooled, with "
                       "pyphen en_US syllables, Markdown removed, and every line ending a sentence",
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
            style = format_flags(output)
            notes += [note for note, found in (("introduction", style["introduction"]),
                                               ("Markdown", style["markdown"])) if found]
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
