"""The review sheet (Ruler phase 4): people's blind labels, end to end.

A finished run's sample is exported as sheets for labellers, labelled
by someone who reads only the input columns, read back, and certified. Every
run uses the conformance provider with scripted answers, so each row's
delivered value is known, and so is what the certificate must say. The test
"labeller" fills cells by reading the ``input: marker`` column, as a person
reads inputs, and edits the files with openpyxl and csv directly, not with
the code under test.

Design: ``.claude/designs/certified-columns.md`` section 13.10.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pytest

from ondine import PipelineBuilder
from ondine.harness.contract import OutputField
from ondine.ruler import CertificationError
from ondine.ruler.intervals import hypergeom_bounds
from ondine.ruler.sheet import ReviewSheetError
from tests.e2e.conformance_harness import conformance_frame
from tests.e2e.test_contract_delivery_e2e import ScriptedClient

LABELS = ["refund", "billing"]
THREE = ["refund", "billing", "complaint"]

#: Cells that must never reach a labeller.
HIDDEN = "LEAK-7"


def _truth(row: int) -> str:
    """The true label of a row: the delivered one, except every seventh row."""
    delivered = LABELS[row % 2]
    return LABELS[(row + 1) % 2] if row % 7 == 0 else delivered


def _row_of(marker: str) -> int:
    return int(marker.split("-")[1])


def _frame(rows: int):
    frame = conformance_frame(rows)
    frame["text"] = [f"café {row}" for row in range(rows)]
    frame["secret"] = HIDDEN
    frame["gold_note"] = [f"GOLD-{row:04d}" for row in range(rows)]
    return frame


def _builder(client, rows: int, *, columns=("intent",), text=None):
    frame = _frame(rows)
    if text is not None:
        frame["text"] = text
    return (
        PipelineBuilder.create()
        .from_dataframe(
            frame,
            input_columns=["marker", "text"],
            output_columns=list(columns),
        )
        .with_prompt("Classify {marker} {text}")
        .with_custom_llm_client(client)
        .with_batch_size(1)
        .with_concurrency(1)
        .with_max_retries(1)
    )


def _single(rows: int, *, labels=LABELS, text=None):
    """A pipeline whose every row delivers ``LABELS[row % 2]``, not yet run."""
    client = ScriptedClient({row: LABELS[row % 2] for row in range(rows)})
    return _builder(client, rows, text=text).with_output_field("intent", allowed=labels)


def _run(rows: int, **options):
    pipeline = _single(rows, **options).build()
    return pipeline, pipeline.execute()


def _multi(rows: int):
    """Three fields: labels, an integer and a boolean, plus a number."""
    answers = {
        row: json.dumps(
            {
                "intent": LABELS[row % 2],
                "urgency": 1 + row % 5,
                "ok": row % 2 == 0,
                "score": row / 2,
            }
        )
        for row in range(rows)
    }
    pipeline = (
        _builder(
            ScriptedClient(answers), rows, columns=("intent", "urgency", "ok", "score")
        )
        .with_output_field("intent", allowed=LABELS)
        .with_output_field("urgency", type="integer", minimum=1, maximum=5)
        .with_output_field("ok", type="boolean")
        .with_output_field("score", type="number")
        .build()
    )
    return pipeline, pipeline.execute()


# ---------------------------------------------------------------------------
# The test labeller: reads inputs, writes labels, in either file format
# ---------------------------------------------------------------------------


def _letter(path: Path) -> str:
    return re.search(r"sheet-([A-Z])", path.name).group(1)  # type: ignore[union-attr]


def _records(path: Path) -> list[dict[str, Any]]:
    """The rows of a returned sheet, as {column: cell}, header excluded."""
    if path.suffix == ".xlsx":
        wb = openpyxl.load_workbook(path)
        ws = wb["Labels"]
        header = [c.value for c in ws[1]]
        return [
            dict(zip(header, [c.value for c in row], strict=False))
            for row in ws.iter_rows(min_row=2)
        ]
    text = path.read_bytes().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    return list(reader)


def _edit(path: Path, change) -> None:
    """Apply ``change(record) -> {column: value}`` to every row of a sheet."""
    if path.suffix == ".xlsx":
        wb = openpyxl.load_workbook(path)
        ws = wb["Labels"]
        header = [c.value for c in ws[1]]
        for r in range(2, ws.max_row + 1):
            record = {h: ws.cell(r, i + 1).value for i, h in enumerate(header)}
            for column, value in change(record).items():
                ws.cell(r, header.index(column) + 1).value = value
        wb.save(path)
        return
    text = path.read_bytes().decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text, newline="")))
    header = rows[0]
    for row in rows[1:]:
        record = dict(zip(header, row, strict=False))
        for column, value in change(record).items():
            row[header.index(column)] = "" if value is None else str(value)
    _write_csv(path, rows)


def _write_csv(path: Path, rows, *, delimiter=",", encoding="utf-8-sig") -> None:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, delimiter=delimiter).writerows(rows)
    path.write_bytes(buffer.getvalue().encode(encoding))


def _fill(export, labeller, *, only=None) -> None:
    """Every sheet of ``export``, labelled by ``labeller(row, letter) -> {field: cell}``."""
    for letter, path in export.sheets.items():
        if only is not None and letter not in only:
            continue

        def change(record, letter=letter):
            labels = labeller(_row_of(record["input: marker"]), letter)
            return {f"label: {field}": cell for field, cell in labels.items()}

        _edit(path, change)


def _truthful(row: int, letter: str) -> dict[str, Any]:
    return {"intent": _truth(row)}


def _manifest(folder: Path) -> dict[str, Any]:
    return json.loads((Path(folder) / "keep_private" / "manifest.json").read_text())


def _doubled_rows(folder: Path) -> list[int]:
    return list(_manifest(folder)["doubled"])


def _comparable(cert) -> dict[str, Any]:
    """A certificate's data minus what only people's labels add or word differently."""
    data = cert.to_dict()
    for key in ("claim_text", "sheet", "reference"):
        data.pop(key, None)
    for field in data["fields"]:
        field.pop("labellers", None)
    return data


def _certify(pipeline, folder, **options):
    labels = pipeline.read_review(folder)
    return pipeline.certify(labels.run, gold=labels, **options)


def _export_and_label(tmp_path, rows=300, size=60, **options):
    pipeline, result = _run(rows)
    sample = pipeline.draw_sample(result, size=size)
    folder = tmp_path / "review"
    export = pipeline.export_review(result, folder, sample=sample, **options)
    return pipeline, result, sample, export


# ---------------------------------------------------------------------------
# 1. Blind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["xlsx", "csv"])
def test_nothing_a_labeller_must_not_see_is_in_the_files_they_get(tmp_path, fmt):
    pipeline, result = _run(40)
    export = pipeline.export_review(result, tmp_path / "r", format=fmt)
    frame = result.to_pandas()
    harness = ["status", "reason", "checks", "repair_method", "delivered_by"]
    assert all(column in frame.columns for column in harness)

    blobs = []
    for path in sorted((tmp_path / "r" / "for_labellers").iterdir()):
        data = path.read_bytes()
        blobs.append(data)
        if path.suffix == ".xlsx":
            with zipfile.ZipFile(path) as book:
                blobs += [book.read(name) for name in book.namelist()]
    for blob in blobs:
        for forbidden in [HIDDEN, "GOLD-", "gold_note", "secret", *harness]:
            assert forbidden.encode() not in blob, forbidden

    sheets = sorted(p.name for p in (tmp_path / "r" / "for_labellers").iterdir())
    expected = (
        ["sheet-A.xlsx", "sheet-B.xlsx"]
        if fmt == "xlsx"
        else ["sheet-A.csv", "sheet-B.csv", "task.txt"]
    )
    assert sheets == expected
    for path in export.sheets.values():
        records = _records(path)
        assert list(records[0]) == [
            "item",
            "input: marker",
            "input: text",
            "label: intent",
            "notes",
        ]
        assert all(not record["label: intent"] for record in records)


def test_the_manifest_holds_no_delivered_value_and_run_json_stays_private(tmp_path):
    pipeline, result = _run(40)
    pipeline.export_review(result, tmp_path / "r", format="csv")

    private = sorted(p.name for p in (tmp_path / "r" / "keep_private").iterdir())
    manifest = (tmp_path / "r" / "keep_private" / "manifest.json").read_text()

    assert private == ["manifest.json", "run.json"]
    assert not list((tmp_path / "r" / "for_labellers").rglob("run.json"))
    for forbidden in ["refund", "billing", "values", HIDDEN, "café"]:
        assert forbidden not in manifest
    run = json.loads((tmp_path / "r" / "keep_private" / "run.json").read_text())
    assert run["values"]["intent"][0] == "refund"


def test_the_export_says_who_gets_which_file_and_what_must_never_be_sent(tmp_path):
    pipeline, result = _run(40)

    export = pipeline.export_review(result, tmp_path / "r")
    text = export.render()

    assert export.sheet_id in text
    for path in export.sheets.values():
        assert str(path) in text
    assert "keep_private" in text
    assert "never" in text.lower()
    assert "run.json" in text


def test_item_ids_are_opaque_and_carry_the_sheet_id(tmp_path):
    pipeline, result = _run(40)
    export = pipeline.export_review(result, tmp_path / "r", format="csv")

    items = [r["item"] for p in export.sheets.values() for r in _records(p)]

    assert all(re.fullmatch(rf"{export.sheet_id}-[AB]\d{{3}}", i) for i in items)
    assert len(set(items)) == len(items)


# ---------------------------------------------------------------------------
# 2. Order
# ---------------------------------------------------------------------------


def _expected_deal(sample, labellers=2, double=0.2):
    """The spec's dealing, written out independently of the code under test."""
    seed = int(
        hashlib.sha256(
            f"{sample.seed}:review:{labellers}:{double}".encode()
        ).hexdigest()[:16],
        16,
    )
    rng = np.random.default_rng(seed)
    n = len(sample.rows)
    k = math.ceil(round(double * n, 9)) if labellers == 2 else 0
    doubled = sorted(int(i) for i in rng.choice(n, size=k, replace=False))
    rest = [i for i in range(n) if i not in doubled]
    dealt = [rest[0::2], rest[1::2]][:labellers]
    sheets = []
    for own in dealt:
        members = sorted(doubled + own)
        order = rng.permutation(len(members))
        sheets.append([sample.rows[members[i]] for i in order])
    return sheets, sorted(sample.rows[i] for i in doubled)


def test_each_sheet_is_the_seeded_shuffle_not_input_order_and_exports_repeat(tmp_path):
    pipeline, result = _run(300)
    sample = pipeline.draw_sample(result, size=50)

    first = pipeline.export_review(result, tmp_path / "one", sample=sample)
    second = pipeline.export_review(result, tmp_path / "two", sample=sample)

    expected, doubled = _expected_deal(sample)
    for letter, rows in zip("AB", expected, strict=True):
        got = [_row_of(r["input: marker"]) for r in _records(first.sheets[letter])]
        assert got == rows
        assert got != sorted(got)
    assert _doubled_rows(tmp_path / "one") == doubled
    assert _records(first.sheets["A"]) == _records(second.sheets["A"])
    assert _records(first.sheets["B"]) == _records(second.sheets["B"])
    assert _manifest(tmp_path / "one") == _manifest(tmp_path / "two")


def test_a_run_whose_input_order_leaks_labels_is_still_shuffled(tmp_path):
    # Grouped by label, as the intent dev split is: input order would give the label away.
    pipeline, result = _run(100)
    export = pipeline.export_review(result, tmp_path / "r", format="csv")

    rows = [_row_of(r["input: marker"]) for r in _records(export.sheets["A"])]

    assert rows != sorted(rows)


# ---------------------------------------------------------------------------
# 3-4. Round trip, new process
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["xlsx", "csv"])
def test_a_labelled_sheet_certifies_like_the_same_labels_given_as_gold(tmp_path, fmt):
    pipeline, result, sample, export = _export_and_label(tmp_path, format=fmt)
    _fill(export, _truthful)

    labels = pipeline.read_review(tmp_path / "review")
    cert = pipeline.certify(labels.run, gold=labels)

    by_gold = pipeline.certify(
        result, gold={row: _truth(row) for row in sample.rows}, sample=sample
    )
    assert _comparable(cert) == _comparable(by_gold)
    assert cert.claim == "certificate"
    assert cert.claim_text == (
        "accuracy against blind human labels, on a sample Ondine drew"
    )
    assert f"sheet {export.sheet_id}" in cert.render()
    assert cert.sheet == export.sheet_id
    assert by_gold.sheet is None


def test_certify_takes_the_labels_with_the_result_in_the_same_process(tmp_path):
    pipeline, result, sample, export = _export_and_label(tmp_path)
    _fill(export, _truthful)

    labels = pipeline.read_review(tmp_path / "review", result=result)

    assert (
        pipeline.certify(result, gold=labels).to_dict()
        == pipeline.certify(labels.run, gold=labels).to_dict()
    )


def test_a_fresh_pipeline_reads_the_folder_without_the_result_and_certifies_the_same(
    tmp_path,
):
    pipeline, result, sample, export = _export_and_label(tmp_path)
    _fill(export, _truthful)
    now = _certify(pipeline, tmp_path / "review")

    later_pipeline = _single(300).build()
    labels = later_pipeline.read_review(tmp_path / "review")
    later = later_pipeline.certify(labels.run, gold=labels)

    assert later.to_dict() == now.to_dict()
    assert later.render() == now.render()
    assert labels.sample.rows == sample.rows


def test_another_runs_result_is_refused_at_import(tmp_path):
    pipeline, result, _, export = _export_and_label(tmp_path)
    _fill(export, _truthful)

    with pytest.raises(CertificationError, match="another run"):
        pipeline.read_review(tmp_path / "review", result=pipeline.execute())


def test_a_pipeline_with_another_contract_is_refused_at_import(tmp_path):
    pipeline, result, _, export = _export_and_label(tmp_path)
    _fill(export, _truthful)
    other = _single(300, labels=[*LABELS, "complaint"]).build()

    with pytest.raises(CertificationError, match="another output contract"):
        other.read_review(tmp_path / "review")


def test_a_saved_view_changed_after_export_is_refused(tmp_path):
    pipeline, result, _, export = _export_and_label(tmp_path)
    _fill(export, _truthful)
    run = tmp_path / "review" / "keep_private" / "run.json"
    data = json.loads(run.read_text())
    data["values"]["intent"][0] = "billing"
    run.write_text(json.dumps(data))

    with pytest.raises(CertificationError, match="fingerprint"):
        pipeline.read_review(tmp_path / "review")


def test_a_manifest_that_does_not_match_its_run_file_is_refused(tmp_path):
    pipeline, result, _, export = _export_and_label(tmp_path)
    _fill(export, _truthful)
    path = tmp_path / "review" / "keep_private" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["fingerprint"] = "0" * 64
    path.write_text(json.dumps(manifest))

    with pytest.raises(ReviewSheetError, match="run.json"):
        pipeline.read_review(tmp_path / "review")


def test_a_sample_the_redraw_does_not_reproduce_is_refused_naming_numpy(tmp_path):
    pipeline, result, _, export = _export_and_label(tmp_path)
    _fill(export, _truthful)
    path = tmp_path / "review" / "keep_private" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["sample"]["rows"] = manifest["sample"]["rows"][::-1]
    manifest["numpy"] = "0.0.1"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ReviewSheetError) as raised:
        pipeline.read_review(tmp_path / "review")

    assert "0.0.1" in str(raised.value)
    assert np.__version__ in str(raised.value)


def test_sheets_kept_elsewhere_are_read_from_where_they_are(tmp_path):
    pipeline, result, sample, export = _export_and_label(tmp_path)
    _fill(export, _truthful)
    elsewhere = tmp_path / "returned"
    elsewhere.mkdir()
    moved = elsewhere / "ada-labels.xlsx"
    export.sheets["A"].rename(moved)

    labels = pipeline.read_review(tmp_path / "review", sheets={"A": moved})

    assert labels.sample.rows == sample.rows
    with pytest.raises(ReviewSheetError, match="sheet-A.xlsx"):
        pipeline.read_review(tmp_path / "review")


# ---------------------------------------------------------------------------
# 5. Typos: every problem at once, located
# ---------------------------------------------------------------------------


def _typo_run(tmp_path):
    pipeline, result = _multi(60)
    export = pipeline.export_review(result, tmp_path / "r", sample=None, format="csv")

    def truthful(row, letter):
        return {
            "intent": _truth(row),
            "urgency": 1 + row % 5,
            "ok": "true" if row % 2 == 0 else "false",
            "score": str(row / 2),
        }

    _fill(export, truthful)
    return pipeline, export


def _set(path: Path, item: str, column: str, value: str) -> None:
    def change(record):
        return {column: value} if record["item"] == item else {}

    _edit(path, change)


def test_every_typo_is_listed_at_once_with_file_row_item_and_column(tmp_path):
    pipeline, export = _typo_run(tmp_path)
    a_items = [r["item"] for r in _records(export.sheets["A"])]
    b_items = [r["item"] for r in _records(export.sheets["B"])]
    _set(export.sheets["A"], a_items[3], "label: intent", "refnd")
    _set(export.sheets["A"], a_items[9], "label: intent", "xyzzy")
    _set(export.sheets["B"], b_items[1], "label: score", "3,5")
    _set(export.sheets["B"], b_items[4], "label: ok", "maybe")

    with pytest.raises(ReviewSheetError) as raised:
        pipeline.read_review(tmp_path / "r")

    problems = raised.value.problems
    assert len(problems) == 4
    assert [p.file for p in problems] == sorted(p.file for p in problems)
    located = {(p.file, p.item, p.column) for p in problems}
    assert located == {
        ("sheet-A.csv", a_items[3], "label: intent"),
        ("sheet-A.csv", a_items[9], "label: intent"),
        ("sheet-B.csv", b_items[1], "label: score"),
        ("sheet-B.csv", b_items[4], "label: ok"),
    }
    by_item = {p.item: p for p in problems}
    assert by_item[a_items[3]].suggestion == "refund"
    assert by_item[a_items[9]].suggestion is None
    assert "use a dot" in by_item[b_items[1]].message
    # Sheet row 1 is the header: item k of a sheet is on row k + 2.
    assert by_item[a_items[3]].row == 5
    assert by_item[b_items[4]].row == 6
    text = str(raised.value)
    assert "4 problems" in text
    assert "refnd" in text
    assert "did you mean 'refund'" in text
    assert all(p.item in text for p in problems)


def test_a_typo_never_becomes_an_unresolved_row(tmp_path):
    pipeline, export = _typo_run(tmp_path)
    item = _records(export.sheets["A"])[0]["item"]
    _set(export.sheets["A"], item, "label: intent", "Refund")

    with pytest.raises(ReviewSheetError, match="did you mean 'refund'"):
        pipeline.read_review(tmp_path / "r")


def test_more_than_fifty_problems_print_the_first_fifty_and_count_the_rest(tmp_path):
    pipeline, result = _run(300)
    export = pipeline.export_review(
        result, tmp_path / "r", sample=pipeline.draw_sample(result, size=70)
    )
    _fill(export, lambda row, letter: {"intent": "nonsense"})

    with pytest.raises(ReviewSheetError) as raised:
        pipeline.read_review(tmp_path / "r")

    assert len(raised.value.problems) > 50
    assert f"… {len(raised.value.problems) - 50} more" in str(raised.value)


# ---------------------------------------------------------------------------
# 6. Several answers, "cannot tell", empty
# ---------------------------------------------------------------------------


def test_several_answers_cannot_tell_and_empty_give_right_unresolved_and_unresolved(
    tmp_path,
):
    pipeline, result, sample, export = _export_and_label(tmp_path, labellers=1)
    rows = list(sample.rows)
    both, unsure, blank = rows[0], rows[1], rows[2]

    def labeller(row, letter):
        if row == both:
            return {"intent": "refund | billing"}
        if row == unsure:
            return {"intent": "?"}
        if row == blank:
            return {"intent": ""}
        return {"intent": _truth(row)}

    _fill(export, labeller)
    labels = pipeline.read_review(tmp_path / "review")
    intent = pipeline.certify(labels.run, gold=labels)["intent"]

    assert intent.unresolved_causes == {"cannot tell": 1, "not labelled": 1}
    est = intent.estimate
    right = sum(
        _truth(row) == LABELS[row % 2] for row in rows if row not in (unsure, blank)
    )
    # The row with both answers is right whatever it delivered.
    right += (_truth(both) != LABELS[both % 2]) * 1
    assert (est.compared, est.unresolved, est.right) == (60, 2, right)
    low, _ = hypergeom_bounds([right], 60, 300)
    _, high = hypergeom_bounds([right + 2], 60, 300)
    assert (est.low, est.high) == (int(low[0]) / 300, int(high[0]) / 300)


def test_an_answer_containing_a_bar_is_read_whole_before_it_is_split(tmp_path):
    labels = ["a | b", "c"]
    answers = {row: labels[row % 2] for row in range(40)}
    pipeline = (
        _builder(ScriptedClient(answers), 40)
        .with_output_field("intent", allowed=labels)
        .build()
    )
    result = pipeline.execute()
    export = pipeline.export_review(result, tmp_path / "r", labellers=1, format="csv")
    _fill(export, lambda row, letter: {"intent": labels[row % 2]})

    cert = _certify(pipeline, tmp_path / "r")

    assert cert["intent"].estimate.exact
    assert cert["intent"].estimate.right == 40


# ---------------------------------------------------------------------------
# 7. Another run's sheet
# ---------------------------------------------------------------------------


def test_a_sheet_from_another_run_is_refused_naming_both_sheet_ids(tmp_path):
    pipeline, result = _run(300)
    one = pipeline.export_review(result, tmp_path / "one")
    other = pipeline.execute()
    two = pipeline.export_review(other, tmp_path / "two")
    assert one.sheet_id != two.sheet_id
    _fill(one, _truthful)
    _fill(two, _truthful)
    two.sheets["A"].replace(one.sheets["A"])

    with pytest.raises(ReviewSheetError) as raised:
        pipeline.read_review(tmp_path / "one")

    message = str(raised.value)
    assert one.sheet_id in message
    assert two.sheet_id in message
    assert "another sample or run" in message
    assert len(raised.value.problems) == 1


# ---------------------------------------------------------------------------
# 8. Damage
# ---------------------------------------------------------------------------


def _csv_rows(path: Path) -> list[list[str]]:
    return list(
        csv.reader(io.StringIO(path.read_bytes().decode("utf-8-sig"), newline=""))
    )


def test_an_edited_input_a_deleted_row_a_duplicate_and_a_moved_row_are_four_problems(
    tmp_path,
):
    pipeline, result, sample, export = _export_and_label(
        tmp_path, rows=300, size=50, format="csv"
    )
    _fill(export, _truthful)
    a, b = export.sheets["A"], export.sheets["B"]
    rows_a, rows_b = _csv_rows(a), _csv_rows(b)
    text_column = rows_a[0].index("input: text")
    edited = rows_a[2][0]
    rows_a[2][text_column] = "something else"
    deleted = rows_a.pop(5)[0]
    moved = rows_a.pop(7)
    rows_b.append(moved)
    rows_b.append(list(rows_b[3]))
    _write_csv(a, rows_a)
    _write_csv(b, rows_b)

    with pytest.raises(ReviewSheetError) as raised:
        pipeline.read_review(tmp_path / "review")

    problems = raised.value.problems
    assert len(problems) == 4
    by_item = {p.item: p for p in problems}
    assert by_item[edited].column == "input: text"
    assert "input" in by_item[edited].message
    assert "missing" in by_item[deleted].message
    assert "duplicate" in by_item[rows_b[3][0]].message
    assert "other labeller" in by_item[moved[0]].message
    assert by_item[moved[0]].file == "sheet-B.csv"


def test_rows_sorted_into_another_order_are_accepted(tmp_path):
    pipeline, result, sample, export = _export_and_label(tmp_path, format="csv")
    _fill(export, _truthful)
    before = pipeline.read_review(tmp_path / "review")
    for path in export.sheets.values():
        rows = _csv_rows(path)
        _write_csv(path, [rows[0], *sorted(rows[1:], key=lambda r: r[0])])

    after = pipeline.read_review(tmp_path / "review")

    assert after == before


def test_extra_columns_and_notes_are_accepted(tmp_path):
    pipeline, result, sample, export = _export_and_label(tmp_path, format="csv")
    _fill(export, _truthful)
    before = pipeline.read_review(tmp_path / "review")
    for path in export.sheets.values():
        rows = _csv_rows(path)
        rows[0].append("my comment")
        header = rows[0]
        for row in rows[1:]:
            row[header.index("notes")] = "unsure about this one"
            row.append("x")
        _write_csv(path, rows)

    assert pipeline.read_review(tmp_path / "review") == before


def test_a_missing_label_column_and_an_unknown_one_are_listed(tmp_path):
    pipeline, result, sample, export = _export_and_label(tmp_path, format="csv")
    _fill(export, _truthful)
    rows = _csv_rows(export.sheets["A"])
    rows[0][rows[0].index("label: intent")] = "label: intnt"
    _write_csv(export.sheets["A"], rows)

    with pytest.raises(ReviewSheetError) as raised:
        pipeline.read_review(tmp_path / "review")

    text = str(raised.value)
    assert "label: intent" in text
    assert "label: intnt" in text
    assert "missing" in text
    assert "unknown" in text


def test_a_sheet_file_that_is_not_there_or_not_a_spreadsheet_is_listed(tmp_path):
    pipeline, result, sample, export = _export_and_label(tmp_path)
    _fill(export, _truthful)
    export.sheets["A"].unlink()
    export.sheets["B"].write_bytes(b"not a spreadsheet")

    with pytest.raises(ReviewSheetError) as raised:
        pipeline.read_review(tmp_path / "review")

    assert {p.file for p in raised.value.problems} == {"sheet-A.xlsx", "sheet-B.xlsx"}


# ---------------------------------------------------------------------------
# 9. Double labelling and kappa
# ---------------------------------------------------------------------------


def test_fifty_rows_with_two_labellers_double_ten_and_split_thirty_each(tmp_path):
    pipeline, result, sample, export = _export_and_label(
        tmp_path, size=50, format="csv"
    )

    doubled = _doubled_rows(tmp_path / "review")
    on_a = {_row_of(r["input: marker"]) for r in _records(export.sheets["A"])}
    on_b = {_row_of(r["input: marker"]) for r in _records(export.sheets["B"])}

    assert len(doubled) == 10
    assert (len(on_a), len(on_b)) == (30, 30)
    assert on_a & on_b == set(doubled)
    assert on_a | on_b == set(sample.rows)


def test_two_scripted_disagreements_give_eight_of_ten_and_the_hand_counted_kappa(
    tmp_path,
):
    pipeline, result, sample, export = _export_and_label(tmp_path, size=50)
    doubled = _doubled_rows(tmp_path / "review")
    flipped = set(doubled[:2])

    def labeller(row, letter):
        label = _truth(row)
        if letter == "B" and row in flipped:
            label = LABELS[1 - LABELS.index(label)]
        return {"intent": label}

    _fill(export, labeller)
    labels = pipeline.read_review(tmp_path / "review")
    agreement = pipeline.certify(labels.run, gold=labels)["intent"].labellers

    assert (agreement.doubled, agreement.both_answered) == (10, 10)
    assert (agreement.agree, agreement.split) == (8, 2)
    # By hand: A's refunds a, B's refunds b, over 10 pairs, 8 of which agree.
    a = sum(_truth(r) == "refund" for r in doubled)
    b = a + sum(1 if _truth(r) == "billing" else -1 for r in doubled[:2])
    expected_by_chance = (a * b + (10 - a) * (10 - b)) / 100
    kappa = (0.8 - expected_by_chance) / (1 - expected_by_chance)
    assert agreement.kappa == pytest.approx(kappa)
    assert agreement.kappa_rows == 10


# ---------------------------------------------------------------------------
# 10. The disagreement rule
# ---------------------------------------------------------------------------


def _doubled_run(tmp_path, *, rows=10, labels=THREE):
    pipeline, result = _run(300, labels=labels)
    sample = pipeline.draw_sample(result, size=rows)
    export = pipeline.export_review(
        result, tmp_path / "r", sample=sample, double=1.0, format="csv"
    )
    return pipeline, result, sample, export


def _other(delivered: str) -> str:
    return "billing" if delivered == "refund" else "refund"


def test_the_two_labellers_rule_on_every_pair_of_answers(tmp_path):
    pipeline, result, sample, export = _doubled_run(tmp_path)
    delivered = {row: LABELS[row % 2] for row in sample.rows}
    r = list(sample.rows)
    # row: (labeller A, labeller B)
    script = {
        r[0]: (lambda d: (d, d)),  # both right -> right
        r[1]: (lambda d: (d, _other(d))),  # right / wrong -> disagree
        r[2]: (lambda d: (_other(d), d)),  # wrong / right -> disagree
        r[3]: (lambda d: ("complaint", _other(d))),  # both wrong -> wrong
        r[4]: (lambda d: ("?", d)),  # ? / answer -> the answer counts
        r[5]: (lambda d: (_other(d), "?")),  # answer / ? -> the answer counts
        r[6]: (lambda d: ("?", "?")),  # neither -> cannot tell
        r[7]: (lambda d: ("", "")),  # neither -> not labelled
        r[8]: (lambda d: ("?", "")),  # one ? -> cannot tell
        r[9]: (lambda d: (f"{d} | {_other(d)}", _other(d))),  # one right, one wrong
    }

    def labeller(row, letter):
        a, b = script[row](delivered[row])
        return {"intent": a if letter == "A" else b}

    _fill(export, labeller)
    labels = pipeline.read_review(tmp_path / "r")
    intent = pipeline.certify(labels.run, gold=labels)["intent"]

    assert (intent.estimate.right, intent.estimate.wrong) == (2, 2)
    assert intent.estimate.unresolved == 6
    assert intent.unresolved_causes == {
        "labellers disagree": 3,
        "cannot tell": 2,
        "not labelled": 1,
    }
    wrong = {w.row: w for w in intent.wrong_rows}
    assert set(wrong) == {r[3], r[5]}
    # The reference is A's answers, then B's others.
    assert wrong[r[3]].reference == ("complaint", _other(delivered[r[3]]))
    assert wrong[r[5]].reference == (_other(delivered[r[5]]),)
    assert intent.labellers.split == 3
    assert intent.labellers.both_answered == 5
    assert intent.labellers.doubled == 10


def test_the_range_stays_valid_when_disagreement_hides_the_truth(tmp_path):
    pipeline, result, sample, export = _doubled_run(tmp_path, rows=10)

    def labeller(row, letter):
        d = LABELS[row % 2]
        return {"intent": d if letter == "A" else _other(d)}

    _fill(export, labeller)
    labels = pipeline.read_review(tmp_path / "r")
    est = pipeline.certify(labels.run, gold=labels)["intent"].estimate

    assert est.unresolved == 10
    assert est.right == 0
    assert est.low == 0.0
    _, high = hypergeom_bounds([10], 10, 300)
    assert est.high == int(high[0]) / 300


def test_seventy_percent_agreement_renders_the_noisy_labels_warning(tmp_path):
    pipeline, result, sample, export = _doubled_run(tmp_path, rows=10)
    rows = list(sample.rows)
    differ = set(rows[:3])

    def labeller(row, letter):
        d = LABELS[row % 2]
        return {"intent": _other(d) if letter == "B" and row in differ else d}

    _fill(export, labeller)
    labels = pipeline.read_review(tmp_path / "r")
    cert = pipeline.certify(labels.run, gold=labels)

    agreement = cert["intent"].labellers
    assert (agreement.agree, agreement.both_answered) == (7, 10)
    text = cert.render()
    assert "Warning: labellers often disagree" in text
    assert "agree on 7" in text
    assert "70.0%" in text
    assert "few rows" in text  # fewer than 20 rows both answered


def test_full_agreement_renders_no_warning_and_the_line_matches_the_design(tmp_path):
    pipeline, result, sample, export = _export_and_label(tmp_path, size=100)
    _fill(export, _truthful)
    labels = pipeline.read_review(tmp_path / "review")

    text = pipeline.certify(labels.run, gold=labels).render()

    assert "Warning: labellers often disagree" not in text
    assert (
        "Labellers: 2 (one sheet each); 20 rows labelled twice: agree on 20 (100.0%)"
        in text
    )
    assert "kappa" in text


# ---------------------------------------------------------------------------
# 11. A census of 12 rows
# ---------------------------------------------------------------------------


def test_a_census_of_twelve_rows_doubles_three_and_is_exact_when_all_are_labelled(
    tmp_path,
):
    pipeline, result = _run(12)
    export = pipeline.export_review(result, tmp_path / "r", format="csv")
    _fill(export, _truthful)

    labels = pipeline.read_review(tmp_path / "r")
    cert = pipeline.certify(labels.run, gold=labels)

    assert len(_doubled_rows(tmp_path / "r")) == 3
    assert (
        cert.claim_text == "accuracy against blind human labels, on every delivered row"
    )
    assert cert["intent"].estimate.exact
    assert cert["intent"].labellers.doubled == 3
    assert "exact: every delivered row compared" in cert.render()


def test_one_labeller_gives_one_sheet_and_says_agreement_is_not_measured(tmp_path):
    pipeline, result = _run(12)
    export = pipeline.export_review(result, tmp_path / "r", labellers=1, format="csv")
    _fill(export, _truthful)

    labels = pipeline.read_review(tmp_path / "r")
    cert = pipeline.certify(labels.run, gold=labels)

    assert list(export.sheets) == ["A"]
    assert _doubled_rows(tmp_path / "r") == []
    assert cert["intent"].labellers.labellers == 1
    assert "agreement not measured (one labeller)" in cert.render()


# ---------------------------------------------------------------------------
# 12. Two fields, one optional
# ---------------------------------------------------------------------------


def _optional_run(rows=20):
    answers = {
        row: json.dumps(
            {"intent": LABELS[row % 2], "note": None if row % 4 == 3 else "x"}
        )
        for row in range(rows)
    }
    pipeline = (
        _builder(ScriptedClient(answers), rows, columns=("intent", "note"))
        .with_output_field("intent", allowed=LABELS)
        .with_output_field("note", allowed=["x", "y"], required=False)
        .with_contract_policy(partial_rows="keep_valid_fields")
        .build()
    )
    return pipeline, pipeline.execute()


def test_a_dash_means_no_value_and_is_refused_on_a_required_field(tmp_path):
    pipeline, result = _optional_run()
    export = pipeline.export_review(result, tmp_path / "r", labellers=1, format="csv")

    def labeller(row, letter):
        labels = {"intent": LABELS[row % 2], "note": "x"}
        if row == 0:
            labels["note"] = "-"  # delivered "x", labelled "no value": wrong
        if row == 3:
            labels["note"] = "-"  # delivered nothing, labelled nothing: not compared
        if row == 1:
            labels["intent"] = "-"  # required: an error
        if row == 2:
            labels["intent"] = "?"
            labels["note"] = ""
        return labels

    _fill(export, labeller)
    with pytest.raises(ReviewSheetError) as raised:
        pipeline.read_review(tmp_path / "r")
    assert len(raised.value.problems) == 1
    problem = raised.value.problems[0]
    assert problem.column == "label: intent"
    assert "required" in problem.message

    _fill(export, lambda row, letter: {"intent": "billing"} if row == 1 else {})
    labels = pipeline.read_review(tmp_path / "r")
    cert = pipeline.certify(labels.run, gold=labels)

    note = cert["note"]
    assert note.estimate.population == 15
    assert [w.row for w in note.wrong_rows] == [0]
    assert note.unresolved_causes == {"not labelled": 1}
    assert cert["intent"].unresolved_causes == {"cannot tell": 1}
    assert cert["intent"].estimate.right == 19
    assert cert["intent"].estimate.wrong == 0


# ---------------------------------------------------------------------------
# 13. Numbers and booleans
# ---------------------------------------------------------------------------


def test_numeric_and_boolean_cells_from_a_spreadsheet_read_like_typed_text(tmp_path):
    pipeline, result = _multi(60)
    export = pipeline.export_review(result, tmp_path / "r", labellers=1)

    def labeller(row, letter):
        return {
            "intent": _truth(row),
            "urgency": 1 + row % 5 + (1 if row % 5 < 4 else 0),  # an xlsx integer cell
            "ok": row % 2 == 0,  # an xlsx boolean cell
            "score": row / 2 if row % 2 else str(row / 2),  # a float cell, or text
        }

    _fill(export, labeller)
    labels = pipeline.read_review(tmp_path / "r")
    strict = pipeline.certify(labels.run, gold=labels)
    loose = pipeline.certify(labels.run, gold=labels, tolerance={"urgency": 1})

    assert strict["ok"].estimate.exact
    assert strict["ok"].estimate.right == 60
    assert strict["score"].estimate.right == 60
    assert strict["urgency"].estimate.right == 12  # only rows with row % 5 == 4
    assert loose["urgency"].estimate.right == 60


def test_a_whole_number_in_a_spreadsheet_float_cell_reads_as_the_integer(tmp_path):
    pipeline, result = _multi(20)
    export = pipeline.export_review(result, tmp_path / "r", labellers=1)

    _fill(
        export,
        lambda row, letter: {
            "intent": _truth(row),
            "urgency": float(1 + row % 5),
            "ok": "TRUE" if row % 2 == 0 else "False",
            "score": row / 2,
        },
    )
    labels = pipeline.read_review(tmp_path / "r")

    assert pipeline.certify(labels.run, gold=labels)["urgency"].estimate.right == 20


# ---------------------------------------------------------------------------
# 14. Parity of the formats
# ---------------------------------------------------------------------------


def test_xlsx_csv_and_semicolon_cp1252_csv_read_to_the_same_labels(tmp_path):
    pipeline, result = _run(300)
    sample = pipeline.draw_sample(result, size=60)
    folders = {}
    for name, fmt in (("x", "xlsx"), ("c", "csv"), ("s", "csv")):
        export = pipeline.export_review(
            result, tmp_path / name, sample=sample, format=fmt
        )
        _fill(export, _truthful)
        folders[name] = export
    for path in folders["s"].sheets.values():
        _write_csv(path, _csv_rows(path), delimiter=";", encoding="cp1252")
    assert b"caf\xe9" in next(iter(folders["s"].sheets.values())).read_bytes()

    results = {name: pipeline.read_review(tmp_path / name) for name in folders}
    certs = {
        name: pipeline.certify(labels.run, gold=labels)
        for name, labels in results.items()
    }

    assert results["x"] == results["c"] == results["s"]
    assert certs["x"].to_dict() == certs["c"].to_dict() == certs["s"].to_dict()


def test_labels_typed_in_a_semicolon_cp1252_file_keep_their_accents(tmp_path):
    labels = ["café", "crème"]
    answers = {row: labels[row % 2] for row in range(20)}
    pipeline = (
        _builder(ScriptedClient(answers), 20)
        .with_output_field("intent", allowed=labels)
        .build()
    )
    result = pipeline.execute()
    export = pipeline.export_review(result, tmp_path / "r", labellers=1, format="csv")
    _fill(export, lambda row, letter: {"intent": labels[row % 2]})
    path = export.sheets["A"]
    _write_csv(path, _csv_rows(path), delimiter=";", encoding="cp1252")

    cert = _certify(pipeline, tmp_path / "r")

    assert cert["intent"].estimate.right == 20


# ---------------------------------------------------------------------------
# 15. Guards
# ---------------------------------------------------------------------------


def test_exporting_reading_and_certifying_changes_neither_the_frame_nor_the_report(
    tmp_path,
):
    pipeline, result, sample, export = _export_and_label(tmp_path)
    frame = result.to_pandas().copy()
    report = result.report.render()
    _fill(export, _truthful)

    labels = pipeline.read_review(tmp_path / "review", result=result)
    pipeline.certify(result, gold=labels)

    import pandas as pd

    pd.testing.assert_frame_equal(result.to_pandas(), frame)
    assert result.report.render() == report


def test_a_folder_that_is_not_empty_is_refused_so_returned_sheets_survive(tmp_path):
    pipeline, result = _run(40)
    pipeline.export_review(result, tmp_path / "r")

    with pytest.raises(CertificationError, match="not empty"):
        pipeline.export_review(result, tmp_path / "r")
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("mine")
    with pytest.raises(CertificationError, match="not empty"):
        pipeline.export_review(result, other)
    assert (other / "notes.txt").read_text() == "mine"
    empty = tmp_path / "empty"
    empty.mkdir()
    pipeline.export_review(result, empty)


def test_an_export_that_fails_while_writing_leaves_nothing_behind(tmp_path):
    long_text = ["x" * 40_000, *[f"t{row}" for row in range(1, 40)]]
    pipeline, result = _run(40, text=long_text)

    with pytest.raises(CertificationError, match='format="csv"'):
        pipeline.export_review(result, tmp_path / "new")
    assert not (tmp_path / "new").exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(CertificationError, match='format="csv"'):
        pipeline.export_review(result, empty)
    assert list(empty.iterdir()) == []

    pipeline.export_review(result, empty, format="csv")


def test_without_openpyxl_the_error_names_the_excel_extra_and_csv_still_works(
    tmp_path, monkeypatch
):
    pipeline, result = _run(40)
    csv_export = pipeline.export_review(result, tmp_path / "c", format="csv")
    _fill(csv_export, _truthful)
    xlsx_export = pipeline.export_review(result, tmp_path / "x")
    _fill(xlsx_export, _truthful)

    monkeypatch.setitem(sys.modules, "openpyxl", None)

    with pytest.raises(ImportError, match=r"ondine\[excel\]") as raised:
        pipeline.export_review(result, tmp_path / "y")
    assert 'format="csv"' in str(raised.value)
    assert not (tmp_path / "y").exists() or not any((tmp_path / "y").iterdir())
    with pytest.raises(ImportError, match=r"ondine\[excel\]"):
        pipeline.read_review(tmp_path / "x")
    assert pipeline.read_review(tmp_path / "c").sample.rows


def test_an_input_that_looks_like_a_formula_stays_a_string_cell(tmp_path):
    text = ["=1+1", "+SUM(A1)", "-5", "@cmd", *[f"t{row}" for row in range(4, 40)]]
    pipeline, result = _run(40, text=text)
    export = pipeline.export_review(result, tmp_path / "r", labellers=1)

    ws = openpyxl.load_workbook(export.sheets["A"])["Labels"]
    cells = {
        row[1].value: row[2]
        for row in ws.iter_rows(min_row=2)
        if row[2].value in text[:4]
    }
    assert set(cells) == {"row-0000", "row-0001", "row-0002", "row-0003"}
    assert all(cell.data_type == "s" for cell in cells.values())
    assert {c.value for c in cells.values()} == set(text[:4])
    assert all(cell.number_format == "@" for cell in ws["D"][1:])
    _fill(export, _truthful)
    assert pipeline.read_review(tmp_path / "r").sample.rows


def test_the_task_and_allowed_values_are_on_a_task_sheet_with_a_dropdown(tmp_path):
    pipeline, result = _run(40)
    export = pipeline.export_review(result, tmp_path / "r", task="Label the intent.")

    wb = openpyxl.load_workbook(export.sheets["A"])
    assert wb.sheetnames == ["Labels", "Task", "Values"]
    task = "\n".join(str(row[0].value) for row in wb["Task"].iter_rows())
    assert "Label the intent." in task
    assert "refund" in task
    assert "billing" in task
    assert "Classify" not in task  # task= replaces the prompt text
    assert [c.value for c in wb["Values"]["A"]] == [
        "label: intent",
        "refund",
        "billing",
    ]
    validations = wb["Labels"].data_validations.dataValidation
    assert len(validations) == 1
    assert validations[0].errorStyle == "warning"
    assert "Values" in validations[0].formula1


def test_the_task_text_is_the_prompt_and_system_prompt_verbatim_by_default(tmp_path):
    pipeline = _single(40).with_system_prompt("You are a bank agent.").build()
    result = pipeline.execute()

    export = pipeline.export_review(result, tmp_path / "r", format="csv")
    task = (tmp_path / "r" / "for_labellers" / "task.txt").read_text(encoding="utf-8")

    assert "Classify {marker} {text}" in task
    assert "You are a bank agent." in task
    assert "refund" in task
    assert "billing" in task
    assert export.sheet_id in export.render()


def test_a_run_with_no_contract_needs_the_same_fields_at_import(tmp_path):
    client = ScriptedClient({row: LABELS[row % 2] for row in range(40)})
    pipeline = _builder(client, 40).build()
    result = pipeline.execute()
    declared = [OutputField("intent", allowed=LABELS)]

    with pytest.raises(CertificationError, match="no output contract"):
        pipeline.export_review(result, tmp_path / "r")
    export = pipeline.export_review(result, tmp_path / "r", fields=declared)
    _fill(export, _truthful)
    with pytest.raises(CertificationError, match="no output contract"):
        pipeline.read_review(tmp_path / "r")
    with pytest.raises(CertificationError, match="another output contract"):
        pipeline.read_review(
            tmp_path / "r", fields=[OutputField("intent", allowed=[*LABELS, "x"])]
        )

    labels = pipeline.read_review(tmp_path / "r", fields=declared)
    cert = pipeline.certify(labels.run, gold=labels, fields=declared)

    assert cert["intent"].estimate.compared == 40
    # Raw text is read by the declarations, as gold would be.
    by_gold = pipeline.certify(
        result,
        gold={row: _truth(row) for row in labels.sample.rows},
        sample=labels.sample,
        fields=declared,
    )
    assert _comparable(cert) == _comparable(by_gold)


# ---------------------------------------------------------------------------
# Arguments and progress
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"labellers": 3}, "labellers"),
        ({"labellers": 0}, "labellers"),
        ({"double": 0.1}, "at least 0.2"),
        ({"double": 1.5}, "at most 1"),
        ({"format": "ods"}, "format"),
    ],
)
def test_arguments_that_cannot_make_a_sheet_are_refused_before_anything_is_written(
    tmp_path, options, message
):
    pipeline, result = _run(40)

    with pytest.raises(CertificationError, match=message):
        pipeline.export_review(result, tmp_path / "r", **options)

    assert not (tmp_path / "r").exists()


def test_a_sample_of_another_run_is_refused_at_export(tmp_path):
    pipeline, result = _run(300)
    other = pipeline.draw_sample(pipeline.execute(), size=30)

    with pytest.raises(CertificationError, match="drawn from run"):
        pipeline.export_review(result, tmp_path / "r", sample=other)


def test_a_streams_chunk_and_a_saved_view_cannot_be_exported(tmp_path):
    pipeline, result = _run(40)

    with pytest.raises(CertificationError):
        pipeline.export_review(pipeline.save_view(result), tmp_path / "r")  # type: ignore[arg-type]


def test_labels_carry_their_sample_so_another_one_is_refused(tmp_path):
    pipeline, result, sample, export = _export_and_label(tmp_path)
    _fill(export, _truthful)
    labels = pipeline.read_review(tmp_path / "review")

    with pytest.raises(CertificationError, match="own sample"):
        pipeline.certify(labels.run, gold=labels, sample=sample)


def test_labels_of_another_run_are_refused_by_certify(tmp_path):
    pipeline, result, sample, export = _export_and_label(tmp_path)
    _fill(export, _truthful)
    labels = pipeline.read_review(tmp_path / "review")

    with pytest.raises(CertificationError, match="drawn from run"):
        pipeline.certify(pipeline.execute(), gold=labels)


def test_progress_counts_labelled_unsure_and_empty_cells_per_sheet(tmp_path):
    pipeline, result, sample, export = _export_and_label(
        tmp_path, size=50, format="csv"
    )
    unsure = set(sample.rows[:4])
    blank = set(sample.rows[4:9])

    def labeller(row, letter):
        if row in unsure:
            return {"intent": "?"}
        if row in blank:
            return {"intent": ""}
        return {"intent": _truth(row)}

    _fill(export, labeller)
    labels = pipeline.read_review(tmp_path / "review")

    text = labels.render()
    assert export.sheet_id in text
    for letter in "AB":
        on_sheet = [
            _row_of(r["input: marker"]) for r in _records(export.sheets[letter])
        ]
        counts = (
            sum(r not in unsure | blank for r in on_sheet),
            sum(r in unsure for r in on_sheet),
            sum(r in blank for r in on_sheet),
        )
        line = f'intent: {counts[0]} labelled, {counts[1]} "?", {counts[2]} empty'
        assert line in text
    assert "Sheet A: 30 rows" in text
    assert "Sheet B: 30 rows" in text


def test_a_people_certificate_data_holds_the_sheet_and_agreement_a_gold_one_does_not(
    tmp_path,
):
    pipeline, result, sample, export = _export_and_label(tmp_path)
    _fill(export, _truthful)
    labels = pipeline.read_review(tmp_path / "review")
    people = pipeline.certify(labels.run, gold=labels).to_dict()
    gold = pipeline.certify(
        result, gold={r: _truth(r) for r in sample.rows}, sample=sample
    ).to_dict()

    assert people["sheet"] == export.sheet_id
    assert "labellers" in people["fields"][0]
    assert "sheet" not in gold
    assert "labellers" not in gold["fields"][0]


# ---------------------------------------------------------------------------
# CSV injection: a customer's text can start with = + - @
# ---------------------------------------------------------------------------

INJECTIONS = [
    '=HYPERLINK("http://example.test","click")',
    "+33612345678",
    "-1",
    "@SUM(1+1)",
    "\t=1+1",
    "'plain",
    "'=x",
    "''q",
    "007",
]


def _injection_run(tmp_path):
    text = [*INJECTIONS, *[f"t{row}" for row in range(len(INJECTIONS), 40)]]
    pipeline, result = _run(40, text=text)
    return pipeline, result, text


def test_inputs_a_spreadsheet_would_evaluate_round_trip_through_csv_unchanged(tmp_path):
    pipeline, result, text = _injection_run(tmp_path)
    export = pipeline.export_review(result, tmp_path / "r", labellers=1, format="csv")

    raw = _csv_rows(export.sheets["A"])
    assert not [c for row in raw for c in row if c[:1] in tuple("=+-@\t\r")]
    by_marker = {row[1]: row[2] for row in raw[1:]}
    assert by_marker["row-0000"].startswith("'=HYPERLINK")
    assert by_marker["row-0007"] == "''=x" or by_marker["row-0006"] == "''=x"
    _fill(export, _truthful)
    labels = pipeline.read_review(tmp_path / "r")

    assert labels.sample.rows == tuple(range(40))
    assert pipeline.certify(labels.run, gold=labels)["intent"].estimate.exact


def test_the_hash_is_of_the_unescaped_text_so_csv_and_xlsx_read_the_same(tmp_path):
    pipeline, result, text = _injection_run(tmp_path)
    sample = pipeline.draw_sample(result)
    exports = {
        fmt: pipeline.export_review(
            result, tmp_path / fmt, sample=sample, labellers=1, format=fmt
        )
        for fmt in ("xlsx", "csv")
    }
    for export in exports.values():
        _fill(export, _truthful)

    def items(folder):
        return {k: v["items"] for k, v in _manifest(folder)["sheets"].items()}

    assert items(tmp_path / "xlsx") == items(tmp_path / "csv")
    assert pipeline.read_review(tmp_path / "xlsx") == pipeline.read_review(
        tmp_path / "csv"
    )


# ---------------------------------------------------------------------------
# A few foreign rows do not hide the other problems; an edited manifest is refused
# ---------------------------------------------------------------------------


def test_a_foreign_row_among_the_sheets_own_is_listed_with_the_other_problems(tmp_path):
    pipeline, result, sample, export = _export_and_label(tmp_path, format="csv")
    _fill(export, _truthful)
    path = export.sheets["A"]
    rows = _csv_rows(path)
    rows[2][0] = "abcdef-A001"
    typo = rows[4][0]
    rows[4][rows[0].index("label: intent")] = "refnd"
    _write_csv(path, rows)

    with pytest.raises(ReviewSheetError) as raised:
        pipeline.read_review(tmp_path / "review")

    problems = raised.value.problems
    assert len(problems) == 3  # the foreign row, the typo, and the item it replaced
    assert {p.item for p in problems} >= {"abcdef-A001", typo}
    assert any("not an item of this review" in p.message for p in problems)
    assert any(p.suggestion == "refund" for p in problems)


def _edit_manifest(folder, change):
    path = folder / "keep_private" / "manifest.json"
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.__setitem__("doubled", m["doubled"][1:]),
        lambda m: m["sheets"]["A"]["items"].reverse(),
        # A row only sheet B has: B's first item may be a doubled row that
        # A also starts with, which would make the edit a no-op.
        lambda m: m["sheets"]["A"]["items"][0].__setitem__(
            "row",
            next(
                b["row"]
                for b in m["sheets"]["B"]["items"]
                if b["row"] not in {a["row"] for a in m["sheets"]["A"]["items"]}
            ),
        ),
        lambda m: m["sheets"]["A"]["items"][0].__setitem__("item", "ffffff-A001"),
        lambda m: m.__setitem__("sheet_seed", m["sheet_seed"] + 1),
        lambda m: m.__setitem__("double", 0.3),
    ],
)
def test_a_manifest_that_is_not_the_dealing_its_seed_gives_is_refused(tmp_path, change):
    pipeline, result, sample, export = _export_and_label(tmp_path, size=50)
    _fill(export, _truthful)
    _edit_manifest(tmp_path / "review", change)

    with pytest.raises(ReviewSheetError, match="not the dealing its seed gives"):
        pipeline.read_review(tmp_path / "review")


# ---------------------------------------------------------------------------
# Per-labeller cell states
# ---------------------------------------------------------------------------


def test_the_labels_say_which_cells_were_unsure_and_which_were_empty(tmp_path):
    pipeline, result, sample, export = _export_and_label(
        tmp_path, size=50, format="csv"
    )
    unsure, blank = sample.rows[0], sample.rows[1]

    def labeller(row, letter):
        if row == unsure:
            return {"intent": "?"}
        if row == blank:
            return {"intent": ""}
        return {"intent": _truth(row)}

    _fill(export, labeller)
    states = pipeline.read_review(tmp_path / "review").states

    assert set(states) == {"A", "B"}
    for letter in "AB":
        by_row = states[letter]["intent"]
        assert by_row.get(unsure, "unsure") == "unsure"
        assert by_row.get(blank, "empty") == "empty"
        assert set(by_row.values()) <= {"answer", "unsure", "empty"}
    assert states["A"]["intent"] != {}
