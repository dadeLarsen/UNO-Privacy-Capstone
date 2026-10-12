"""Strict parsing and honest scoring: failures stay in the denominator."""

import copy
import csv
import json

import pytest

from privacy_bench.tasks.classification.data import Row
from privacy_bench.tasks.classification.scoring import evaluate, parse_response, write_evaluation
from privacy_bench.tasks.classification.taxonomy import HEADER, LABELS

INVALID_PAYLOADS = [
    '{"label_id": true}', '{"label_id": false}', '{"label_id": "1"}',
    '{"label_id": 1.0}', '{"label_id": 1e0}', '{"label_id": null}',
    '{"label_id": -1}', '{"label_id": 10}', '{"label_id": 11}',
    '{"label_id": 12}', '{"label_id": NaN}',
    '{"label_id": 1, "label_id": 2}',
    '{"label_id": 1, "explanation": "reason"}',
    '{"label_id": 1} {"label_id": 2}',
    'Result: {"label_id": 1}',
    '{"label_id":', '', '{}', '[]', '[{"label_id": 1}]', 'null', '1',
]


def test_parser_accepts_only_the_integer_category_contract():
    assert parse_response(' {"label_id": 0}\n') == 0
    assert parse_response('{"label_id":9}') == 9


@pytest.mark.parametrize("raw", INVALID_PAYLOADS + [None, {"label_id": 1}])
def test_parser_rejects_invalid_payloads_bare_and_fenced(raw):
    with pytest.raises(ValueError):
        parse_response(raw)
    if isinstance(raw, str):
        with pytest.raises(ValueError):
            parse_response("```json\n" + raw + "\n```")


@pytest.mark.parametrize("raw", ['```json\n{"label_id": 4}\n```', '```\n{"label_id": 4}\n```',
                                 ' \n```JSON \r\n{\r\n  "label_id": 4\r\n}\r\n```\r\n'])
def test_parser_accepts_whole_response_json_or_untagged_fences(raw):
    assert parse_response(raw) == 4


BLOCK = '```json\n{"label_id": 4}\n```'


@pytest.mark.parametrize("raw", [
    "Here is the answer:\n" + BLOCK, BLOCK + "\nThis is my answer.", BLOCK + "\n" + BLOCK,
    BLOCK + '\n{"label_id": 4}', '```json\n{"label_id": 4}', '{"label_id": 4}\n```',
    '```python\n{"label_id": 4}\n```', '```json {"label_id": 4}```', '````json\n{"label_id": 4}\n````',
    "```json\n```", '```json\nResult: {"label_id": 4}\n```', "<think>It is access.</think>\n" + BLOCK,
])
def test_parser_rejects_prose_multiple_blocks_and_incomplete_fences(raw):
    with pytest.raises(ValueError):
        parse_response(raw)


def test_fenced_scoring_preserves_raw_and_reports_strict_format_compliance(tmp_path):
    rows = [Row(key, key, (0,)) for key in ("original", "alternative", "bare", "invalid", "error", "missing")]
    responses = [
        {"id": "original", "raw_output": '```json\n{"label_id": 0}\n```'},
        {"id": "alternative", "raw_output": '```\n{"label_id": 1}\n```'},
        {"id": "bare", "raw_output": '{"label_id": 0}'},
        {"id": "invalid", "raw_output": '```json\n{"label_id": "0"}\n```'},
        {"id": "error", "raw_output": '```json\n{"label_id": 0}\n```', "error": "generation failed"},
    ]
    original = copy.deepcopy(responses)
    metrics = write_evaluation(tmp_path, rows, responses, {row.id: [0, 1] for row in rows})
    assert metrics["correct"] == 2
    assert metrics["accepted_correct"] == 3
    assert metrics["accepted_accuracy"] == 0.5
    assert metrics["valid_output_rate"] == 0.5
    assert metrics["strict_json_valid_output_rate"] == 1 / 6
    assert metrics["fenced_json_responses"] == 2
    assert metrics["response_parsing"] == "json_or_single_markdown_fence_v1"
    assert responses == original
    with (tmp_path / "details.csv").open(encoding="utf-8", newline="") as handle:
        details = list(csv.DictReader(handle))
    assert [item["raw_output"] for item in details[:-1]] == [item["raw_output"] for item in original]
    assert [item["status"] for item in details] == ["valid", "valid", "valid", "invalid", "error", "missing"]


def test_metrics_include_invalid_missing_and_all_ten_classes():
    rows = [Row("a", "A", (0,)), Row("b", "B", (0,)), Row("c", "C", (1,)), Row("d", "D", (1,)), Row("e", "E", (2,))]
    responses = [
        {"id": "d", "raw_output": "invalid"},
        {"id": "c", "raw_output": '{"label_id":1}'},
        {"id": "b", "raw_output": '{"label_id":1}'},
        {"id": "a", "raw_output": '{"label_id":0}'},
    ]
    metrics, details, confusion = evaluate(rows, responses)
    assert metrics["scoring_mode"] == "original_single_label"
    assert (metrics["n"], metrics["recorded_responses"], metrics["correct"]) == (5, 4, 2)
    assert metrics["accuracy"] == pytest.approx(2 / 5)
    assert metrics["micro_precision"] == pytest.approx(2 / 3)
    assert metrics["micro_recall"] == pytest.approx(2 / 5)
    assert metrics["micro_f1"] == pytest.approx(1 / 2)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 1 / 2) / 10)
    assert metrics["valid_output_rate"] == pytest.approx(3 / 5)
    assert metrics["status_counts"] == {"valid": 3, "invalid": 1, "error": 0, "missing": 1}
    assert [detail["id"] for detail in details] == [row.id for row in rows]
    assert [detail["predicted_label_id"] for detail in details] == [0, 1, 1, None, None]
    assert [category["support"] for category in metrics["per_category"]] == [2, 2, 1, *([0] * 7)]
    assert "Macro F1 averages all 10 classes" in metrics["scoring"]
    assert (len(confusion), len(confusion[0])) == (10, 11)
    assert confusion[0][:3] == [1, 1, 0] and confusion[1][1] == 1
    assert confusion[1][10] == 1 and confusion[2][10] == 1
    assert sum(map(sum, confusion)) == 5


def test_response_join_is_by_id_independent_of_output_order():
    rows = [Row("001", "A", (0,)), Row("other", "B", (9,))]
    responses = [{"id": "other", "raw_output": '{"label_id":9}'}, {"id": "001", "raw_output": '{"label_id":0}'}]
    forward = evaluate(rows, responses)
    assert forward == evaluate(rows, list(reversed(responses)))
    assert forward[0]["accuracy"] == 1


@pytest.mark.parametrize("responses", [
    [{"id": "001", "raw_output": '{"label_id":0}'}] * 2, [{"id": "unknown"}], [{"id": 1}], [{}], [None],
])
def test_response_join_rejects_duplicate_unknown_and_nonstring_ids(responses):
    with pytest.raises(ValueError):
        evaluate([Row("001", "A", (0,))], responses)


def test_recorded_runtime_error_counts_as_incorrect_even_with_valid_raw_text():
    metrics, details, confusion = evaluate([Row("a", "A", (0,))], [
        {"id": "a", "raw_output": '{"label_id":0}', "error": "generation failed"}])
    assert metrics["status_counts"]["error"] == 1 and metrics["accuracy"] == 0
    assert details[0]["status"] == "error" and details[0]["predicted_label_id"] is None
    assert confusion[0][10] == 1


def test_predictions_use_ten_category_header_and_blank_cells_for_failures(tmp_path):
    rows = [Row("good", 'A "quoted" sentence,\ncontinued.', (3,)), Row("invalid", "B", (5,)), Row("missing", "C", (8,))]
    responses = [{"id": "invalid", "raw_output": "nonsense"}, {"id": "good", "raw_output": '{"label_id":7}'}]
    write_evaluation(tmp_path, rows, responses)
    with (tmp_path / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(HEADER)
        predictions = list(reader)
    assert [item["id"] for item in predictions] == [row.id for row in rows]
    assert predictions[0]["text"] == rows[0].text
    assert [predictions[0][label] for label in LABELS] == [str(int(i == 7)) for i in range(len(LABELS))]
    for item in predictions[1:]:
        assert [item[label] for label in LABELS] == [""] * len(LABELS)
    with (tmp_path / "details.csv").open(encoding="utf-8", newline="") as handle:
        assert [row["status"] for row in csv.DictReader(handle)] == ["valid", "invalid", "missing"]
    with (tmp_path / "confusion.csv").open(encoding="utf-8", newline="") as handle:
        confusion = list(csv.reader(handle))
    assert confusion[0] == ["gold / predicted", *LABELS, "INVALID_OR_MISSING"]
    assert confusion[4][8] == "1" and confusion[6][11] == "1" and confusion[9][11] == "1"


def test_accepted_labels_score_alternatives_without_changing_original_metrics():
    ids = ("original", "alt", "last-alt", "wrong", "invalid", "error", "missing")
    rows = [Row(identifier, identifier, (0 if index < 3 else 1,)) for index, identifier in enumerate(ids)]
    accepted = {row.id: [row.label_id, 2, 9] for row in rows}
    responses = [{"id": identifier, "raw_output": json.dumps({"label_id": label})}
                 for identifier, label in (("original", 0), ("alt", 2), ("last-alt", 9), ("wrong", 0))]
    responses += [{"id": "invalid", "raw_output": "bad JSON"},
                  {"id": "error", "raw_output": '{"label_id":9}', "error": "failed"}]
    original, original_details, original_confusion = evaluate(rows, responses)
    metrics, details, confusion = evaluate(rows, responses, accepted)
    assert metrics["scoring_mode"] == "any_accepted_gold_label"
    assert metrics["accepted_correct"] == 3
    assert metrics["accepted_accuracy"] == pytest.approx(3 / 7)
    assert metrics["macro_accuracy"] == 0.5  # Original groups: 3/3 and 0/4.
    assert [item["accepted_correct"] for item in details] == [1, 1, 1, 0, 0, 0, 0]
    assert [item["accepted_accuracy"] for item in metrics["per_category"]] == [1, 0, *([None] * 8)]
    assert confusion == original_confusion
    for key, value in original.items():
        if key not in ("per_category", "scoring_mode"):
            assert metrics[key] == value, key
    for actual, expected in zip(metrics["per_category"], original["per_category"], strict=True):
        assert {key: actual[key] for key in expected} == expected
    for actual, expected in zip(details, original_details, strict=True):
        assert {key: actual[key] for key in expected} == expected


@pytest.mark.parametrize("count", range(1, len(LABELS) + 1))
def test_accepted_labels_allow_one_through_all_categories(count):
    row = Row("001", "Sentence", (0,))
    for prediction in range(len(LABELS)):
        metrics, details, _ = evaluate([row], [{"id": row.id, "raw_output": json.dumps({"label_id": prediction})}],
                                       {row.id: list(range(count))})
        expected = int(prediction < count)
        assert metrics["accepted_correct"] == expected
        assert metrics["macro_accuracy"] == expected
        assert details[0]["accepted_correct"] == expected


@pytest.mark.parametrize("accepted, message", [
    ({}, "IDs"), ({"other": [0]}, "IDs"), ({"001": [0], "extra": [0]}, "IDs"),
    ({"001": [1, 9]}, "Original gold label was removed"),
    *[({"001": labels}, "valid category IDs")
      for labels in ([], "0", [True], [0.0], ["0"], [-1, 0], [0, len(LABELS)], [None])],
])
def test_accepted_labels_reject_mismatched_ids_invalid_labels_and_removed_originals(accepted, message):
    with pytest.raises(ValueError, match=message):
        evaluate([Row("001", "Sentence", (0,))], [], accepted)
