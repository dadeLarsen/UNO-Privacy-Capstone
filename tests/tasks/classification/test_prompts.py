import pytest

from privacy_bench.core.files import digest_json
from privacy_bench.tasks.classification.prompts import build_prompt, load_prompt, validate_prompt_spec
from privacy_bench.tasks.classification.taxonomy import LABELS, SOURCE_LABELS


@pytest.mark.parametrize("definitions", [{}, {label: f"Definition for {label}" for label in LABELS}])
def test_prompt_lists_exactly_the_ten_category_ids(definitions):
    spec = validate_prompt_spec({"name": "synthetic", "definitions": definitions, "guidance": []})
    prompt = build_prompt("Synthetic sentence", spec)
    assert "9: Other" in prompt
    assert "10:" not in prompt and "11:" not in prompt


def test_legacy_twelve_category_definitions_are_rejected():
    legacy = {"name": "legacy", "definitions": {label: "Definition" for label in SOURCE_LABELS}, "guidance": []}
    with pytest.raises(ValueError, match="all ten exact"):
        validate_prompt_spec(legacy)


@pytest.mark.parametrize("spec, message", [
    ({"name": "x", "definitions": {}}, "exactly name, definitions, and guidance"),
    ({"name": " ", "definitions": {}, "guidance": []}, "nonempty string"),
    ({"name": "x", "definitions": {label: " " for label in LABELS}, "guidance": []}, "nonempty string"),
    ({"name": "x", "definitions": {}, "guidance": "text"}, "list of strings"),
])
def test_malformed_prompt_specs_are_rejected(spec, message):
    with pytest.raises(ValueError, match=message):
        validate_prompt_spec(spec)


def test_sentence_is_embedded_as_a_json_string_and_guidance_is_appended():
    spec = {"name": "g", "definitions": {}, "guidance": ["Prefer specific categories."]}
    prompt = build_prompt('Say "hi"\nnow', spec)
    assert prompt.endswith('Sentence (JSON string):\n"Say \\"hi\\"\\nnow"')
    assert "Additional guidance:\nPrefer specific categories." in prompt


def test_committed_definitions_prompt_matches_the_hash_used_by_earlier_runs():
    spec = load_prompt("definitions")
    assert digest_json(spec) == "a25626922f5d711ece2aed2c7ac2f2908cae0aa73323f43448b9ec488b85e947"
    assert set(spec["definitions"]) == set(LABELS) and spec["guidance"] == []


def test_unknown_prompt_names_and_path_tricks_are_rejected():
    with pytest.raises(ValueError, match="does not exist"):
        load_prompt("missing")
    with pytest.raises(ValueError, match="Invalid data file name"):
        load_prompt("../samples/pilot-200-10cat")
