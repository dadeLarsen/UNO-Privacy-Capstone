import pytest

from privacy_bench.tasks.rewrite.data import Passage
from privacy_bench.tasks.rewrite.scoring import (check, copying, evaluate, format_flags, readability_counts,
                                                 reading_level, review_markdown, status)

SOURCE = "We may keep records for 30 days, except where the law requires 90 days. We don't sell data."


@pytest.mark.parametrize("response, expected", [
    (None, "missing"),
    ({"error": "RuntimeError: boom"}, "error"),
    ({"raw_output": "  "}, "empty"),
    ({"raw_output": "<think>hmm</think> Short."}, "reasoning"),
    ({"raw_output": "<|channel>thought\nhmm<channel|>Short."}, "reasoning"),
    ({"raw_output": "Short.", "reached_output_token_limit": True}, "truncated"),
    ({"raw_output": "Short.", "reached_output_token_limit": False}, "valid"),
])
def test_status(response, expected):
    assert status(response) == expected


def test_checks_name_dropped_numbers_and_qualifiers_and_read_contractions_as_not():
    flags = check(SOURCE, "They keep records for up to 30 days and do not sell data.")
    assert (flags["numbers_in_source"], flags["numbers_kept"], flags["missing_numbers"]) == (2, 1, "90")
    assert (flags["qualifiers_in_source"], flags["missing_qualifiers"]) == (3, "except may")
    assert check(SOURCE, SOURCE)["missing_qualifiers"] == ""


def test_cannot_and_curly_contractions_negate_and_section_labels_are_not_numbers():
    assert check("We cannot guarantee its security.", "They guarantee its security.")["missing_qualifiers"] == "not"
    assert check("We do not sell data.", "We don’t sell data.")["missing_qualifiers"] == ""
    flags = check("8.1 We keep logs for 2.5 years. 8.2 We then delete them.", "Logs are kept for 2.5 years.")
    assert (flags["numbers_in_source"], flags["missing_numbers"]) == (1, "")


def test_copying_measures_the_runs_a_rewrite_shares_with_its_source():
    assert copying("a b c d", "a b x c d") == {"copy_coverage": 0.8, "copy_density": 1.6}
    assert copying("a b c d", "A, b, c, d.") == {"copy_coverage": 1, "copy_density": 4}
    assert copying("a b", "") == {"copy_coverage": None, "copy_density": None}


def test_readability_counts_lines_without_closing_punctuation_as_sentences_and_ignores_markdown():
    assert readability_counts("We keep data. We share it\n- with **partners**") == (8, 3, 10, 0)
    assert readability_counts('They said "stop." Then left') == (5, 2, 5, 0)
    assert readability_counts("We share information.")[3] == 1
    assert reading_level([]) == {"reading_grade": None, "words_per_sentence": None, "long_word_rate": None}
    level = reading_level(["We keep data.", "We share it with partners."])
    assert level["words_per_sentence"] == 4 and level["long_word_rate"] == 0


@pytest.mark.parametrize("text, introduction, markdown", [
    ("Here’s the summary:\n\nWe keep data.", True, False),
    ("Summary: we keep data.", True, False),
    ("Surely we keep data.", False, False),
    ("We collect the following information:\n- names", False, True),
    ("We keep **some** data.", False, True),
])
def test_format_flags_catch_introductions_and_markdown(text, introduction, markdown):
    assert format_flags(text) == {"introduction": introduction, "markdown": markdown}


def test_metrics_pool_checks_over_every_passage_and_average_length_over_valid_ones():
    passages = [Passage("a", "Retention", SOURCE), Passage("b", "Sharing", "We share data with 3 partners.")]
    responses = [{"id": "a", "raw_output": SOURCE, "reached_output_token_limit": False},
                 {"id": "b", "raw_output": "We share data.", "reached_output_token_limit": True}]
    metrics, details = evaluate(passages, responses)
    assert metrics["status_counts"]["valid"] == 1 and metrics["status_counts"]["truncated"] == 1
    assert metrics["valid_output_rate"] == 0.5 and metrics["mean_output_words"] == len(SOURCE.split())
    assert metrics["numbers_kept_rate"] == 2 / 3
    assert [detail["status"] for detail in details] == ["valid", "truncated"]
    assert metrics["mean_copy_density"] == copying(SOURCE, SOURCE)["copy_density"]
    assert metrics["format_break_rate"] == 0
    assert metrics["reading_grade"] == reading_level([SOURCE])["reading_grade"]
    assert metrics["source_reading_grade"] == reading_level([SOURCE, "We share data with 3 partners."])["reading_grade"]


def test_review_shows_each_passage_once_with_every_rewrite_and_its_flags():
    passages = [Passage("a", "Retention", SOURCE)]
    text = review_markdown(passages, [("Model A", {"a": {"id": "a", "raw_output": "Here is it: kept for 30 days."}}),
                                      ("Model B", {})])
    assert text.count(SOURCE) == 1 and "### Model A" in text and "### Model B" in text
    assert "numbers dropped: 90" in text and "*(no output)*" in text and "**missing**" in text
    assert "; introduction;" in text
