"""Split policy text into sentences while keeping exact offsets into the original text.

The splitter never rewrites or normalizes the policy. It only chooses spans, so every
label can point back to the exact characters the model saw (FR-01, FR-04). Punctuation
inside common abbreviations and decimals does not end a sentence, and a blank line is a
hard boundary so headings without punctuation stay separate.
"""

from dataclasses import dataclass
import re

TERMINATORS = ".!?"
CLOSERS = "\"')]}\u201d\u2019"
PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
# Lowercase, without surrounding punctuation. Keep this list small and conservative.
ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
        "inc", "ltd", "co", "corp", "llc", "dept", "est", "approx",
        "vs", "etc", "al", "fig", "no",
        "e.g", "i.e", "u.s", "u.s.a", "a.m", "p.m",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    }
)


@dataclass(frozen=True)
class Sentence:
    index: int
    text: str
    start: int
    end: int


def split_sentences(text: str) -> list[Sentence]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    sentences: list[Sentence] = []
    cursor = 0
    for match in PARAGRAPH_BREAK.finditer(text):
        _split_span(text, cursor, match.start(), sentences)
        cursor = match.end()
    _split_span(text, cursor, len(text), sentences)
    return sentences


def _split_span(text: str, start: int, end: int, sentences: list[Sentence]) -> None:
    cursor = start
    index = start
    while index < end:
        if text[index] not in TERMINATORS:
            index += 1
            continue
        punctuation_end = index
        while punctuation_end < end and text[punctuation_end] in TERMINATORS:
            punctuation_end += 1
        boundary = punctuation_end
        while boundary < end and text[boundary] in CLOSERS:
            boundary += 1
        followed_by_space = boundary == end or text[boundary].isspace()
        if followed_by_space and not _is_abbreviation(text, index):
            _append(text, cursor, boundary, sentences)
            cursor = boundary
        index = punctuation_end
    _append(text, cursor, end, sentences)


def _append(text: str, start: int, end: int, sentences: list[Sentence]) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        sentences.append(Sentence(index=len(sentences), text=text[start:end], start=start, end=end))


def _is_abbreviation(text: str, punctuation_start: int) -> bool:
    token_start = punctuation_start
    while token_start > 0 and (text[token_start - 1].isalnum() or text[token_start - 1] in ".&'"):
        token_start -= 1
    token = text[token_start:punctuation_start].lower()
    if token in ABBREVIATIONS:
        return True
    # A single initial, as in "J. Smith", is not a sentence boundary.
    return len(token) == 1 and token.isalpha()
