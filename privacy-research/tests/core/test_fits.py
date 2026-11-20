"""Fit files and fit records: parsing, references, and hash-checked loading."""

import numpy as np
import pytest

from privacy_bench.cli import main
from privacy_bench.core.fits import WEIGHTS_FILE, load_fit, parse_fit_spec, resolve_fit
from privacy_bench.core.files import read_json, write_json
from privacy_bench.core.paths import FITS_DIR

SPEC = 'task = "toy"\nkind = "lines"\n[options]\nslope = 2\n[heads.a]\nmethod = "x"\n'


@pytest.fixture
def toy_fit(tmp_path, write_fit):
    return write_fit(tmp_path / "toy" / "toy-fit" / "20261012-000000Z", "toy-fit", SPEC, {"slope": 2},
                     {"line.weight": np.full((2, 3), 2, dtype=np.float32), "line.bias": np.zeros(2, dtype=np.float32)})


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


def test_a_fit_record_loads_with_its_reference_and_tensors(toy_fit):
    fit = load_fit(toy_fit)
    assert fit.reference == "toy/toy-fit/20261012-000000Z" and fit.record["report"] == {"slope": 2}
    np.testing.assert_array_equal(fit.tensors["line.weight"], np.full((2, 3), 2, dtype=np.float32))


@pytest.mark.parametrize("tamper, message", [
    (lambda path: (path / WEIGHTS_FILE).write_bytes((path / WEIGHTS_FILE).read_bytes() + b" "), "weights"),
    (lambda path: (path / "fit.toml").write_text(SPEC.replace("2", "3"), encoding="utf-8"), "fit.toml differs"),
    (lambda path: write_json(path / "fit.json", {**read_json(path / "fit.json"), "status": "failed"}), "not complete"),
    (lambda path: write_json(path / "fit.json", {**read_json(path / "fit.json"), "schema_version": 9}), "schema 1"),
    (lambda path: write_json(path / "fit.json", {**read_json(path / "fit.json"), "tensors": {"line.weight": [2, 3]}}),
     "names and shapes"),
])
def test_loading_refuses_a_changed_or_unfinished_record(toy_fit, tamper, message):
    tamper(toy_fit)
    with pytest.raises(ValueError, match=message):
        load_fit(toy_fit)


def test_every_recorded_fit_still_loads():
    records = sorted(path.parent for path in FITS_DIR.glob("*/*/*/fit.json"))
    assert len(records) >= 2
    for record in records:
        assert load_fit(record).tensors


def test_the_fit_command_refuses_and_points_to_the_history():
    with pytest.raises(SystemExit, match="pbench fit is retired.*History in README.md"):
        main(["fit", "fits/classification/embeddinggemma2-opp115-heads.toml"])
