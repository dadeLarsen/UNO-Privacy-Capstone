"""Fit files and fit records: parsing, references, writing, and hash-checked loading."""

import numpy as np
import pytest

from privacy_bench.core.fits import (
    FIT_SCHEMA_VERSION, WEIGHTS_FILE, execute_fit, fit_directory, load_fit, parse_fit_spec, resolve_fit,
)
from privacy_bench.core.files import read_json, write_json

SPEC = 'task = "toy"\nkind = "lines"\n[options]\nslope = 2\n[heads.a]\nmethod = "x"\n'


class LineTask:
    """Fits y = slope * x on a fixed grid, or fails on request."""

    name = "toy"

    def __init__(self, fail=False, dtype=np.float32):
        self.fail, self.dtype = fail, dtype

    def fit(self, spec):
        if self.fail:
            raise RuntimeError("fit exploded")
        slope = spec.tables["options"]["slope"]
        return {"report": {"slope": slope},
                "tensors": {"line.weight": np.full((2, 3), slope, dtype=self.dtype),
                            "line.bias": np.zeros(2, dtype=self.dtype)}}


def test_fit_files_keep_task_kind_and_tables():
    spec = parse_fit_spec("toy-fit", SPEC)
    assert (spec.task, spec.kind) == ("toy", "lines")
    assert spec.tables == {"options": {"slope": 2}, "heads": {"a": {"method": "x"}}}
    assert spec.as_dict()["name"] == "toy-fit" and spec.sha256 == parse_fit_spec("toy-fit", SPEC).sha256


@pytest.mark.parametrize("name, text, message", [
    ("Toy", SPEC, "lowercase letters"),
    ("toy", 'kind = "lines"\n', "lowercase task"),
    ("toy", 'task = "toy"\n', "lowercase kind"),
    ("toy", 'task = "toy"\nkind = "lines"\nslope = 2\n', "only task and kind outside tables: slope"),
    ("toy", 'task = "toy"\nkind = [', "not valid TOML"),
])
def test_invalid_fit_files_are_rejected(name, text, message):
    with pytest.raises(ValueError, match=message):
        parse_fit_spec(name, text)


def test_fit_references_resolve_under_the_fits_folder(tmp_path):
    assert resolve_fit("toy/toy-fit/20261012-000000Z", root=tmp_path) == tmp_path / "toy" / "toy-fit" / "20261012-000000Z"
    for reference in ("toy/toy-fit", "toy/../x/y", "toy\\toy-fit\\x", "/toy/toy-fit/x", "toy//x", 3):
        with pytest.raises(ValueError, match="fit reference"):
            resolve_fit(reference, root=tmp_path)


def test_a_fit_record_round_trips_with_its_hashes(tmp_path):
    spec = parse_fit_spec("toy-fit", SPEC)
    output = fit_directory("toy", "toy-fit", root=tmp_path)
    record = execute_fit(LineTask(), spec, output)
    assert record["status"] == "complete" and record["schema_version"] == FIT_SCHEMA_VERSION
    assert record["tensors"] == {"line.bias": [2], "line.weight": [2, 3]} and record["report"] == {"slope": 2}
    assert (output / "fit.toml").read_text(encoding="utf-8") == SPEC
    assert record["environment"]["code"]["commit"]
    fit = load_fit(output)
    assert fit.reference == f"toy/toy-fit/{output.name}"
    np.testing.assert_array_equal(fit.tensors["line.weight"], np.full((2, 3), 2, dtype=np.float32))
    with pytest.raises(ValueError, match="already exists"):
        execute_fit(LineTask(), spec, output)


@pytest.mark.parametrize("tamper, message", [
    (lambda path: (path / WEIGHTS_FILE).write_bytes((path / WEIGHTS_FILE).read_bytes() + b" "), "weights"),
    (lambda path: (path / "fit.toml").write_text(SPEC.replace("2", "3"), encoding="utf-8"), "fit.toml differs"),
    (lambda path: write_json(path / "fit.json", {**read_json(path / "fit.json"), "status": "failed"}), "not complete"),
    (lambda path: write_json(path / "fit.json", {**read_json(path / "fit.json"), "schema_version": 9}), "schema 1"),
    (lambda path: write_json(path / "fit.json", {**read_json(path / "fit.json"), "tensors": {"line.weight": [2, 3]}}),
     "names and shapes"),
])
def test_loading_refuses_a_changed_or_unfinished_record(tmp_path, tamper, message):
    output = tmp_path / "fit"
    execute_fit(LineTask(), parse_fit_spec("toy-fit", SPEC), output)
    tamper(output)
    with pytest.raises(ValueError, match=message):
        load_fit(output)


def test_column_major_tensors_load_back_unchanged(tmp_path):
    """scikit-learn returns coef_ column-major; a raw buffer write would transpose-scramble it."""
    weight = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4))
    assert not weight.flags["C_CONTIGUOUS"]

    class ColumnMajorTask(LineTask):
        def fit(self, spec):
            return {"report": {}, "tensors": {"head.weight": weight}}

    execute_fit(ColumnMajorTask(), parse_fit_spec("toy-fit", SPEC), tmp_path / "fit")
    np.testing.assert_array_equal(load_fit(tmp_path / "fit").tensors["head.weight"], weight)


def test_a_failed_fit_is_recorded_and_raised(tmp_path):
    output = tmp_path / "fit"
    with pytest.raises(RuntimeError, match="fit exploded"):
        execute_fit(LineTask(fail=True), parse_fit_spec("toy-fit", SPEC), output)
    record = read_json(output / "fit.json")
    assert record["status"] == "failed" and record["error"] == "RuntimeError: fit exploded"
    assert not (output / WEIGHTS_FILE).exists()


def test_fits_must_return_float32_tensors_for_their_own_task(tmp_path):
    with pytest.raises(ValueError, match="float32"):
        execute_fit(LineTask(dtype=np.float64), parse_fit_spec("toy-fit", SPEC), tmp_path / "fit")
    other = parse_fit_spec("toy-fit", SPEC.replace('task = "toy"', 'task = "other"'))
    with pytest.raises(ValueError, match="is for task 'other'"):
        execute_fit(LineTask(), other, tmp_path / "other")
