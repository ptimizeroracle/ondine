"""Sheet files: one table in memory, two thin codecs (CSV and XLSX).

The review sheet is built, read and checked as a :class:`Table` of text, so
the two formats cannot drift apart: each codec only moves a table to and from
a file.

* **CSV** is written as UTF-8 with a byte-order mark, so Excel reads accents.
  A cell that starts with ``= + - @``, a tab or a carriage return would be
  evaluated, or turned into a number, when a labeller opens the file (a
  customer's text can start with ``=HYPERLINK(...)``). Such a cell gets one
  leading ``'`` on export, and one is removed on import, so every text
  round-trips and the text the sheet's hashes see is unchanged.
  It is read as UTF-8 (with or without the mark), or as cp1252 when the bytes
  are not UTF-8. The delimiter (``,`` ``;`` or tab: French-locale Excel writes
  ``;``) is read from the header.
* **XLSX** needs openpyxl (the ``excel`` extra), imported only when an XLSX
  file is written or read. Text goes into string cells, so a value that starts
  with ``=`` is never a formula. Numbers and booleans a person typed come back
  as the text a CSV cell would hold (``3.0`` as ``3``), so one parser reads both.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ondine.ruler.delivered import CertificationError
from ondine.utils.optional_dependencies import raise_excel_extra_error

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

#: The worksheet people label in; the others hold the task and the dropdown values.
LABELS_SHEET = "Labels"
TASK_SHEET = "Task"
VALUES_SHEET = "Values"

#: The most text an Excel cell holds.
XLSX_CELL_LIMIT = 32_767

_DELIMITERS = (",", ";", "\t")

#: A CSV cell starting with one of these is a formula or a number to a spreadsheet app.
_TRIGGERS = "=+-@\t\r"


@dataclass(frozen=True)
class Table:
    """A sheet as text: the header, then each row padded to the header's width."""

    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


def cell_text(value: Any) -> str:
    """One XLSX cell value as the text a CSV cell would hold."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return repr(value)
    return str(value)


def write_table(
    path: Path,
    table: Table,
    *,
    task: str = "",
    choices: Mapping[str, Sequence[str]] | None = None,
    text_columns: Sequence[str] = (),
) -> None:
    """Write ``table`` to ``path``, as CSV or XLSX by its suffix.

    Args:
        path: ``.csv`` or ``.xlsx``.
        table: The sheet.
        task: XLSX only: the text of the Task worksheet.
        choices: XLSX only: column -> values, kept on a Values worksheet that
            backs a dropdown on the column. The dropdown only warns, so
            several answers can still be typed.
        text_columns: XLSX only: columns formatted as text, so Excel cannot
            turn ``1-2`` into a date.

    Raises:
        CertificationError: The suffix is neither, or a cell is too long for XLSX.
        ImportError: XLSX without openpyxl (it names ``ondine[excel]``).
    """
    kind = _kind(path)
    if kind == "csv":
        buffer = io.StringIO(newline="")
        escaped = [
            [_escape(cell) for cell in row] for row in (table.header, *table.rows)
        ]
        csv.writer(buffer).writerows(escaped)
        path.write_bytes(buffer.getvalue().encode("utf-8-sig"))
        return
    _write_xlsx(path, table, task, choices or {}, set(text_columns))


def read_table(path: Path) -> Table:
    """Read a CSV or XLSX file (by its suffix) as a table of text.

    Trailing blank rows are dropped and short rows padded. Blank rows between
    others are kept, so a row's position is its row in the spreadsheet.

    Raises:
        CertificationError: The suffix is neither.
        ImportError: XLSX without openpyxl (it names ``ondine[excel]``).
    """
    if _kind(path) == "csv":
        return _table(_read_csv(path))
    return _table(_read_xlsx(path))


def _kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in (".csv", ".xlsx"):
        raise CertificationError(
            f"{path.name}: a review sheet is a csv or xlsx file, not {suffix or 'a file with no suffix'}"
        )
    return suffix[1:]


def _table(records: list[list[str]]) -> Table:
    while records and not any(records[-1]):
        records.pop()
    if not records:
        return Table((), ())
    header = tuple(cell.strip() for cell in records[0])
    width = len(header)
    rows = tuple(tuple((row + [""] * width)[:width]) for row in records[1:])
    return Table(header, rows)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> list[list[str]]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    delimiter = _delimiter(text)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    return [[_unescape(cell) for cell in row] for row in reader]


def _escape(text: str) -> str:
    """``text`` as a CSV cell: one ``'`` in front if a spreadsheet would act on it."""
    if text and (text[0] in _TRIGGERS or text[0] == "'"):
        return "'" + text
    return text


def _unescape(cell: str) -> str:
    """The text of a CSV cell: one ``'`` removed if :func:`_escape` could have added it."""
    if cell[:1] == "'" and cell[1:2] and (cell[1] in _TRIGGERS or cell[1] == "'"):
        return cell[1:]
    return cell


def _delimiter(text: str) -> str:
    """The delimiter that reads the header as ours: ``item`` first, else the widest."""
    widest, best = 0, ","
    for delimiter in _DELIMITERS:
        header = next(
            csv.reader(io.StringIO(text, newline=""), delimiter=delimiter), []
        )
        if header and header[0].strip() == "item" and len(header) > 1:
            return delimiter
        if len(header) > widest:
            widest, best = len(header), delimiter
    return best


# ---------------------------------------------------------------------------
# XLSX
# ---------------------------------------------------------------------------


def _openpyxl() -> Any:
    try:
        import openpyxl
    except ImportError as error:
        try:
            raise_excel_extra_error("XLSX review sheets", error)
        except ImportError as friendly:
            raise ImportError(
                f'{friendly}; or use format="csv", which needs no extra'
            ) from error
    return openpyxl


def require_xlsx() -> None:
    """Raise the ``ondine[excel]`` error now if openpyxl is not installed."""
    _openpyxl()


def _write_xlsx(
    path: Path,
    table: Table,
    task: str,
    choices: Mapping[str, Sequence[str]],
    text_columns: set[str],
) -> None:
    openpyxl = _openpyxl()
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    for row in table.rows:
        for text in row:
            if len(text) > XLSX_CELL_LIMIT:
                raise CertificationError(
                    f"an input holds {len(text):,} characters; an Excel cell holds "
                    f'{XLSX_CELL_LIMIT:,}. Use format="csv" for this data'
                )
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = LABELS_SHEET
    bold = Font(bold=True)
    wrapped = Alignment(wrap_text=True, vertical="top")
    for number, name in enumerate(table.header, start=1):
        _put(sheet, 1, number, name).font = bold
        label = name in text_columns
        sheet.column_dimensions[get_column_letter(number)].width = (
            18 if name == "item" else 28 if label else 60
        )
    for r, row in enumerate(table.rows, start=2):
        for number, text in enumerate(row, start=1):
            cell = _put(sheet, r, number, text)
            name = table.header[number - 1]
            if name in text_columns:
                cell.number_format = "@"
            elif name != "item":
                cell.alignment = wrapped
    sheet.freeze_panes = "B2"
    if task:
        notes = book.create_sheet(TASK_SHEET)
        notes.column_dimensions["A"].width = 110
        for r, line in enumerate(task.split("\n"), start=1):
            _put(notes, r, 1, line).alignment = wrapped
    if choices:
        values = book.create_sheet(VALUES_SHEET)
        for number, (name, options) in enumerate(choices.items(), start=1):
            _put(values, 1, number, name).font = bold
            for r, option in enumerate(options, start=2):
                _put(values, r, number, option)
            column = get_column_letter(number)
            if name in table.header and table.rows:
                target = get_column_letter(table.header.index(name) + 1)
                dropdown = DataValidation(
                    type="list",
                    formula1=f"={VALUES_SHEET}!${column}$2:${column}${len(options) + 1}",
                    allow_blank=True,
                    showErrorMessage=True,
                    errorStyle="warning",
                    errorTitle="Not one of the allowed values",
                    error='Pick a value from the list, or type several separated by " | ".',
                )
                dropdown.add(f"{target}2:{target}{len(table.rows) + 1}")
                sheet.add_data_validation(dropdown)
    book.save(path)


def _put(sheet: Any, row: int, column: int, text: str) -> Any:
    """A string cell, never a formula, and empty when ``text`` is."""
    cell = sheet.cell(row=row, column=column)
    if text:
        cell.value = text
        cell.data_type = "s"
    return cell


def _read_xlsx(path: Path) -> list[list[str]]:
    openpyxl = _openpyxl()
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        name = LABELS_SHEET if LABELS_SHEET in book.sheetnames else book.sheetnames[0]
        return [
            [cell_text(value) for value in row]
            for row in book[name].iter_rows(values_only=True)
        ]
    finally:
        book.close()
