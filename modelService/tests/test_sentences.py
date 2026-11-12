from model_service.sentences import split_sentences


def texts(text: str) -> list[str]:
    return [sentence.text for sentence in split_sentences(text)]


def test_offsets_point_back_to_the_original_text():
    text = "We collect data. You can opt out!\n\nContact us?"
    sentences = split_sentences(text)
    assert [sentence.index for sentence in sentences] == [0, 1, 2]
    for sentence in sentences:
        assert text[sentence.start:sentence.end] == sentence.text
        assert sentence.text == sentence.text.strip()


def test_splits_terminators_and_keeps_closing_quotes():
    assert texts('We collect data. You can opt out! He said "Stop." Then left.') == [
        "We collect data.",
        "You can opt out!",
        'He said "Stop."',
        "Then left.",
    ]


def test_does_not_split_common_abbreviations():
    assert texts("Acme Inc. collects data. The U.S. law applies.") == [
        "Acme Inc. collects data.",
        "The U.S. law applies.",
    ]


def test_does_not_split_a_single_initial():
    assert texts("J. Smith signed it. We agree.") == ["J. Smith signed it.", "We agree."]


def test_does_not_split_decimals():
    assert texts("Version 3.5 is out. Next.") == ["Version 3.5 is out.", "Next."]


def test_blank_line_is_a_hard_boundary():
    assert texts("Heading\n\nFirst sentence. Second.") == ["Heading", "First sentence.", "Second."]


def test_no_punctuation_is_one_sentence():
    assert texts("Just a heading") == ["Just a heading"]


def test_empty_and_whitespace_only_text_has_no_sentences():
    assert texts("") == []
    assert texts("   \n\n  \t") == []


def test_ellipsis_ends_a_sentence():
    assert texts("Wait... What happened?") == ["Wait...", "What happened?"]
