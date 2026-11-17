"""The two codecs between one in-memory table and a sheet file.

CSV (utf-8-sig, a delimiter read from the header, cp1252 when the bytes are
not UTF-8) and XLSX (openpyxl, imported lazily) read to the same table.

Design: ``.claude/designs/certified-columns.md`` section 13.2.
"""

from __future__ import annotations

import csv
import io
import sys

import openpyxl
import pytest

from ondine.ruler import CertificationError
from ondine.ruler.sheet_files import Table, cell_text, read_table, write_table

HEADER = ("item", "input: text", "label: intent", "notes")
ROWS = (
    ("a-A001", "plain", "", ""),
    ("a-A002", 'has, comma and "quotes"', "", "n"),
    ("a-A003", "two\nlines é ü", "", ""),
    ("a-A004", "=1+1", "", ""),
)
TABLE = Table(HEADER, ROWS)


def _csv_bytes(rows, *, delimiter=",", encoding="utf-8-sig"):
    buffer = io.StringIO(newline="")
    csv.writer(buffer, delimiter=delimiter).writerows(rows)
    return buffer.getvalue().encode(encoding)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def test_csv_is_written_with_a_byte_order_mark_and_reads_back_exactly(tmp_path):
    path = tmp_path / "s.csv"
    write_table(path, TABLE)

    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert read_table(path) == TABLE


@pytest.mark.parametrize("delimiter", [",", ";", "\t"])
def test_the_delimiter_is_read_from_the_header(tmp_path, delimiter):
    path = tmp_path / "s.csv"
    path.write_bytes(_csv_bytes([HEADER, *ROWS], delimiter=delimiter))

    assert read_table(path) == TABLE


def test_a_comma_inside_a_semicolon_file_does_not_fool_the_detection(tmp_path):
    path = tmp_path / "s.csv"
    rows = [("item", "input: a,b,c", "label: intent"), ("x-A001", "1,2,3", "refund")]
    path.write_bytes(_csv_bytes(rows, delimiter=";"))

    table = read_table(path)

    assert table.header == ("item", "input: a,b,c", "label: intent")
    assert table.rows == (("x-A001", "1,2,3", "refund"),)


def test_a_semicolon_in_a_column_name_of_a_comma_file_is_kept(tmp_path):
    path = tmp_path / "s.csv"
    rows = [("item", "input: a;b", "label: intent"), ("x-A001", "v", "refund")]
    path.write_bytes(_csv_bytes(rows))

    assert read_table(path).header == ("item", "input: a;b", "label: intent")


def test_without_an_item_header_the_widest_delimiter_wins(tmp_path):
    path = tmp_path / "s.csv"
    path.write_bytes(_csv_bytes([("a", "b", "c"), ("1", "2", "3")], delimiter=";"))

    assert read_table(path).header == ("a", "b", "c")


def test_bytes_that_are_not_utf8_are_read_as_cp1252(tmp_path):
    path = tmp_path / "s.csv"
    path.write_bytes(_csv_bytes([HEADER, ROWS[2]], delimiter=";", encoding="cp1252"))

    assert read_table(path).rows == (ROWS[2],)


def test_a_utf8_file_without_a_mark_and_with_one_read_the_same(tmp_path):
    plain, marked = tmp_path / "p.csv", tmp_path / "m.csv"
    plain.write_bytes(_csv_bytes([HEADER, *ROWS], encoding="utf-8"))
    marked.write_bytes(_csv_bytes([HEADER, *ROWS], encoding="utf-8-sig"))

    assert read_table(plain) == read_table(marked) == TABLE


def test_a_byte_cp1252_cannot_read_is_replaced_not_fatal(tmp_path):
    path = tmp_path / "s.csv"
    path.write_bytes(b"item;label: x\nabc;caf\x81\n")

    assert read_table(path).rows == (("abc", "caf�"),)


def test_trailing_blank_rows_are_dropped_and_short_rows_are_padded(tmp_path):
    path = tmp_path / "s.csv"
    path.write_bytes(b"item,label: x,notes\r\na-A001,refund\r\n\r\n,,\r\n\r\n")

    assert read_table(path).rows == (("a-A001", "refund", ""),)


def test_a_blank_row_inside_the_sheet_is_kept_so_row_numbers_hold(tmp_path):
    path = tmp_path / "s.csv"
    path.write_bytes(b"item,label: x\na-A001,r\n,\na-A002,s\n")

    assert read_table(path).rows == (("a-A001", "r"), ("", ""), ("a-A002", "s"))


def test_an_empty_file_is_an_empty_table(tmp_path):
    path = tmp_path / "s.csv"
    path.write_bytes(b"")

    assert read_table(path) == Table((), ())


# ---------------------------------------------------------------------------
# XLSX
# ---------------------------------------------------------------------------


def test_xlsx_reads_back_what_was_written_and_keeps_formula_text_as_text(tmp_path):
    path = tmp_path / "s.xlsx"
    write_table(path, TABLE, text_columns=("label: intent",))

    assert read_table(path) == TABLE
    sheet = openpyxl.load_workbook(path)["Labels"]
    assert sheet["B5"].value == "=1+1"
    assert sheet["B5"].data_type == "s"


def test_xlsx_numbers_and_booleans_become_text_the_csv_parser_reads(tmp_path):
    path = tmp_path / "s.xlsx"
    write_table(path, Table(("item", "label: n"), (("i1", ""), ("i2", ""), ("i3", ""))))
    book = openpyxl.load_workbook(path)
    sheet = book["Labels"]
    sheet["B2"], sheet["B3"], sheet["B4"] = 3.0, True, 3.5
    book.save(path)

    assert [row[1] for row in read_table(path).rows] == ["3", "true", "3.5"]


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (None, ""),
        ("x ", "x "),
        (12, "12"),
        (3.0, "3"),
        (3.5, "3.5"),
        (-0.0, "0"),
        (1e22, "10000000000000000000000"),
        (True, "true"),
        (False, "false"),
        (0.1 + 0.2, "0.30000000000000004"),
    ],
)
def test_cell_text_makes_every_xlsx_value_one_text(value, text):
    assert cell_text(value) == text


def test_a_workbook_with_the_labels_sheet_second_is_still_read(tmp_path):
    path = tmp_path / "s.xlsx"
    book = openpyxl.Workbook()
    book.active.title = "Notes"
    book.active["A1"] = "unrelated"
    labels = book.create_sheet("Labels")
    labels.append(["item", "label: x"])
    labels.append(["i1", "refund"])
    book.save(path)

    assert read_table(path) == Table(("item", "label: x"), (("i1", "refund"),))


def test_a_workbook_with_no_sheet_named_labels_reads_the_first(tmp_path):
    path = tmp_path / "s.xlsx"
    book = openpyxl.Workbook()
    book.active.append(["item", "label: x"])
    book.active.append(["i1", "refund"])
    book.save(path)

    assert read_table(path).rows == (("i1", "refund"),)


def test_trailing_empty_xlsx_rows_are_dropped(tmp_path):
    path = tmp_path / "s.xlsx"
    write_table(path, TABLE)
    book = openpyxl.load_workbook(path)
    book["Labels"]["A40"].number_format = "@"
    book.save(path)

    assert read_table(path) == TABLE


def test_an_input_too_long_for_a_cell_is_refused_before_writing(tmp_path):
    path = tmp_path / "s.xlsx"

    with pytest.raises(CertificationError, match='format="csv"'):
        write_table(path, Table(("item", "input: t"), (("i1", "x" * 40_000),)))

    assert not path.exists()


def test_the_task_values_and_a_dropdown_are_written_for_xlsx_only(tmp_path):
    path = tmp_path / "s.xlsx"
    write_table(
        path,
        TABLE,
        task="Label it.\nSecond line.",
        choices={"label: intent": ["refund", "billing"]},
        text_columns=("label: intent",),
    )

    book = openpyxl.load_workbook(path)
    assert book.sheetnames == ["Labels", "Task", "Values"]
    assert [c.value for c in book["Task"]["A"]] == ["Label it.", "Second line."]
    assert [c.value for c in book["Values"]["A"]] == [
        "label: intent",
        "refund",
        "billing",
    ]
    (validation,) = book["Labels"].data_validations.dataValidation
    assert str(validation.sqref) == "C2:C5"
    assert validation.errorStyle == "warning"
    assert validation.allow_blank


def test_the_file_type_is_decided_by_the_suffix(tmp_path):
    with pytest.raises(CertificationError, match="csv or xlsx"):
        write_table(tmp_path / "s.ods", TABLE)
    with pytest.raises(CertificationError, match="csv or xlsx"):
        read_table(tmp_path / "s.ods")


def test_without_openpyxl_xlsx_names_the_extra_and_csv_still_works(
    tmp_path, monkeypatch
):
    good = tmp_path / "g.xlsx"
    write_table(good, TABLE)
    monkeypatch.setitem(sys.modules, "openpyxl", None)

    with pytest.raises(ImportError, match=r"ondine\[excel\]") as raised:
        write_table(tmp_path / "s.xlsx", TABLE)
    assert 'format="csv"' in str(raised.value)
    with pytest.raises(ImportError, match=r"ondine\[excel\]"):
        read_table(good)
    write_table(tmp_path / "s.csv", TABLE)
    assert read_table(tmp_path / "s.csv") == TABLE


# ---------------------------------------------------------------------------
# CSV injection: a spreadsheet app acts on a cell that starts with = + - @
# ---------------------------------------------------------------------------

DANGEROUS = [
    '=HYPERLINK("http://x","y")',
    "+33612345678",
    "-1",
    "@SUM(1)",
    "\t=1",
    "\r=1",
    "'plain",
    "'=x",
    "''q",
    "'",
    "=",
    "plain",
    "a=b",
    "a'b",
]


def _raw_cells(path):
    text = path.read_bytes().decode("utf-8-sig")
    return [cell for row in csv.reader(io.StringIO(text, newline="")) for cell in row]


@pytest.mark.parametrize("text", DANGEROUS)
def test_every_text_round_trips_through_csv_and_never_starts_a_cell_with_a_trigger(
    tmp_path, text
):
    path = tmp_path / "s.csv"
    table = Table(("item", "input: t", "label: x"), (("i1", text, ""),))

    write_table(path, table)

    assert read_table(path) == table
    assert not any(cell[:1] in tuple("=+-@\t\r") and cell for cell in _raw_cells(path))


def test_an_apostrophe_is_added_once_and_removed_once(tmp_path):
    path = tmp_path / "s.csv"
    write_table(path, Table(("item", "input: t"), (("i1", "=x"), ("i2", "'=x"))))

    cells = _raw_cells(path)
    assert "'=x" in cells
    assert "''=x" in cells
    assert "=x" not in cells


def test_a_cell_a_person_typed_is_read_as_typed_unless_it_has_our_apostrophe(tmp_path):
    path = tmp_path / "s.csv"
    # A label typed as "-" (no value), and an apostrophe a labeller's own text starts with.
    path.write_bytes(
        _csv_bytes([("item", "label: x"), ("i1", "-"), ("i2", "'hello"), ("i3", "'-")])
    )

    assert [row[1] for row in read_table(path).rows] == ["-", "'hello", "-"]


def test_xlsx_cells_are_not_escaped(tmp_path):
    path = tmp_path / "s.xlsx"
    table = Table(("item", "input: t"), (("i1", "=x"), ("i2", "'=x"), ("i3", "+1")))

    write_table(path, table)

    assert read_table(path) == table
