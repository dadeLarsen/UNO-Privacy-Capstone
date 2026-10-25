import pytest

from privacy_bench.tasks.rewrite.data import Passage
from privacy_bench.tasks.rewrite.scoring import check, evaluate, review_markdown, status

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


def test_metrics_pool_checks_over_every_passage_and_average_length_over_valid_ones():
    passages = [Passage("a", "Retention", SOURCE), Passage("b", "Sharing", "We share data with 3 partners.")]
    responses = [{"id": "a", "raw_output": SOURCE, "reached_output_token_limit": False},
                 {"id": "b", "raw_output": "We share data.", "reached_output_token_limit": True}]
    metrics, details = evaluate(passages, responses)
    assert metrics["status_counts"]["valid"] == 1 and metrics["status_counts"]["truncated"] == 1
    assert metrics["valid_output_rate"] == 0.5 and metrics["mean_output_words"] == len(SOURCE.split())
    assert metrics["numbers_kept_rate"] == 2 / 3
    assert [detail["status"] for detail in details] == ["valid", "truncated"]


def test_review_shows_each_passage_once_with_every_rewrite_and_its_flags():
    passages = [Passage("a", "Retention", SOURCE)]
    text = review_markdown(passages, [("Model A", {"a": {"id": "a", "raw_output": "Kept for 30 days."}}),
                                      ("Model B", {})])
    assert text.count(SOURCE) == 1 and "### Model A" in text and "### Model B" in text
    assert "numbers dropped: 90" in text and "*(no output)*" in text and "**missing**" in text
