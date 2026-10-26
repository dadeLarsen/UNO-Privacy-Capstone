import pytest

from privacy_bench.core.files import read_json
from privacy_bench.core.paths import PROMPTS_DIR, data_file
from privacy_bench.tasks.rewrite.data import Passage, load_passages, load_prompt, parse_prompt, render

PROMPT = 'system = """\nBe careful.\n"""\nuser = """\nSummarize {x}:\n{passage}\n"""\n'


def test_prompt_strips_outer_whitespace_and_renders_braces_literally():
    spec = parse_prompt("p", PROMPT)
    assert spec == {"name": "p", "system": "Be careful.", "user": "Summarize {x}:\n{passage}"}
    assert render(spec, Passage("1-2", "H", "We keep data.")) == {"system": "Be careful.",
                                                                 "user": "Summarize {x}:\nWe keep data."}


def test_system_message_is_optional():
    assert parse_prompt("p", 'user = "Summarize: {passage}"')["system"] is None


@pytest.mark.parametrize("text, message", [
    ('user = "Summarize this"', "exactly once"),
    ('user = "{passage} and {passage}"', "exactly once"),
    ('user = "{passage}"\nstyle = "kids"', "unknown keys: style"),
    ('user = "{passage}"\nsystem = "  "', "system must be a nonempty string"),
    ('user = "{passage}', "not valid TOML"),
])
def test_malformed_prompts_are_rejected(text, message):
    with pytest.raises(ValueError, match=message):
        parse_prompt("p", text)


def test_baseline_prompt_and_frozen_passages_load():
    assert "{passage}" in load_prompt("summary-baseline")["user"]
    passages, identity = load_passages("appsi139-sections-20")
    assert len(passages) == 20 and len({passage.id for passage in passages}) == 20
    assert identity["passages"]["name"] == "appsi139-sections-20" and identity["passages"]["n"] == 20
    assert len(identity["passages"]["sha256"]) == 64


def test_every_prompt_file_parses():
    names = [path.stem for path in (PROMPTS_DIR / "rewrite").glob("*.toml")]
    assert "summary-v3" in names and all("{passage}" in load_prompt(name)["user"] for name in names)


def test_the_holdout_passages_come_from_policies_the_first_draw_did_not_use():
    def policies(name):
        passages = read_json(data_file("rewrite", "passages", name, suffix=".json"))["passages"]
        return {passage["policy_first_id"] for passage in passages}
    assert len(load_passages("appsi139-sections-20-holdout")[0]) == 20
    assert len(policies("appsi139-sections-20-holdout")) == 20
    assert not policies("appsi139-sections-20") & policies("appsi139-sections-20-holdout")


def test_prompt_names_cannot_leave_the_prompts_folder():
    with pytest.raises(ValueError, match="Invalid prompt file name"):
        load_prompt("../summary-baseline")
