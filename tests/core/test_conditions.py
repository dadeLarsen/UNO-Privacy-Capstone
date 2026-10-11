import pytest

from privacy_bench.core.conditions import load_condition, parse_condition, take_settings

VALID = """
task = "classification"
strategy = "prompted"
warmup = 2

[items]
dataset = "opp115-sentences"
sample = "pilot-200-10cat"

[model]
id = "Qwen/Qwen3-14B"
precision = "int8"

[options]
prompt = "definitions"
"""


def test_valid_condition_keeps_every_table_and_hashes_its_content():
    condition = parse_condition("qwen3-14b-int8", VALID)
    assert condition.task == "classification"
    assert condition.strategy == "prompted"
    assert condition.warmup == 2
    assert condition.items == {"dataset": "opp115-sentences", "sample": "pilot-200-10cat"}
    assert condition.model == {"id": "Qwen/Qwen3-14B", "precision": "int8"}
    assert condition.options == {"prompt": "definitions"}
    assert condition.source_text == VALID
    reordered = """
warmup = 2
strategy = "prompted"
task = "classification"
[options]
prompt = "definitions"
[model]
precision = "int8"
id = "Qwen/Qwen3-14B"
[items]
sample = "pilot-200-10cat"
dataset = "opp115-sentences"
"""
    assert parse_condition("qwen3-14b-int8", reordered).sha256 == condition.sha256
    assert parse_condition("other-name", VALID).sha256 != condition.sha256


def test_warmup_defaults_to_one_and_tables_default_to_empty():
    condition = parse_condition("minimal", 'task = "classification"\nstrategy = "prompted"\n')
    assert (condition.warmup, condition.items, condition.model, condition.options) == (1, {}, {}, {})


@pytest.mark.parametrize("text, message", [
    ('task = "classification"\nstrategy = "prompted"\nextra = 1\n', "unknown keys: extra"),
    ('strategy = "prompted"\n', "task"),
    ('task = "Classification"\nstrategy = "prompted"\n', "task"),
    ('task = "classification"\n', "strategy"),
    ('task = "classification"\nstrategy = "prompted"\nwarmup = -1\n', "warmup"),
    ('task = "classification"\nstrategy = "prompted"\nwarmup = true\n', "warmup"),
    ('task = "classification"\nstrategy = "prompted"\nmodel = "x"\n', r"\[model\] must be a table"),
    ('task = = "classification"\n', "not valid TOML"),
])
def test_invalid_conditions_are_rejected(text, message):
    with pytest.raises(ValueError, match=message):
        parse_condition("bad", text)


@pytest.mark.parametrize("name", ["Upper", "has space", "-leading", "", "under_score"])
def test_condition_names_are_restricted(name):
    with pytest.raises(ValueError, match="Condition name"):
        parse_condition(name, 'task = "classification"\nstrategy = "prompted"\n')


def test_load_condition_uses_the_file_stem_and_requires_toml(tmp_path):
    path = tmp_path / "qwen3-14b-int8.toml"
    path.write_text(VALID, encoding="utf-8")
    assert load_condition(path).name == "qwen3-14b-int8"
    with pytest.raises(ValueError, match=".toml"):
        load_condition(tmp_path / "qwen3-14b-int8.json")


def test_take_settings_applies_defaults_and_rejects_unknown_or_missing_keys():
    assert take_settings({"a": 1}, "ctx", required=("a",), optional={"b": 2}) == {"a": 1, "b": 2}
    assert take_settings({"a": 1, "b": 3}, "ctx", required=("a",), optional={"b": 2}) == {"a": 1, "b": 3}
    with pytest.raises(ValueError, match="ctx has unknown keys: c"):
        take_settings({"a": 1, "c": 0}, "ctx", required=("a",))
    with pytest.raises(ValueError, match="ctx is missing: a"):
        take_settings({}, "ctx", required=("a",))
