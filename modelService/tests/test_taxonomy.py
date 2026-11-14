import json

import pytest

from model_service.taxonomy import LABELS, TAXONOMY_PATH, load_taxonomy


def test_the_json_names_exactly_the_ten_categories():
    assert len(LABELS) == 10
    assert sorted(json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))) == sorted(LABELS)


@pytest.mark.parametrize("name, problem", [
    ("Other", r"missing \['Other'\], extra \[\]"),
    ("Cookies", r"missing \[\], extra \['Cookies'\]"),
])
def test_a_missing_or_an_extra_name_is_refused(tmp_path, name, problem):
    presentation = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    if name in presentation:
        del presentation[name]
    else:
        presentation[name] = presentation["Other"]
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(presentation), encoding="utf-8")
    with pytest.raises(ValueError, match=problem):
        load_taxonomy(path)
