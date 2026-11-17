"""The review sheet: people's blind labels for a sample, as files.

A finished run's sample is exported as one sheet per labeller (CSV or XLSX),
people label them, and the sheets are read back as the Ruler's
:class:`~ondine.ruler.source.Reference`. This module owns the layout, the
dealing of rows to labellers, the cell syntax, the manifest that binds the
sheets to the run, and the validation that lists every problem at once. The
file formats are :mod:`ondine.ruler.sheet_files`; the comparison is
:mod:`ondine.ruler.ruler`. Nothing here knows the pipeline.

**What a labeller gets** (``for_labellers/``): the sample's input columns, the
task text and each field's allowed values or type. Never the delivered value,
the harness columns, other source columns, few-shot examples or the rendered
prompt. **What stays here** (``keep_private/``): ``manifest.json`` (no
delivered value) and ``run.json`` (the run's answers).

Rows are shuffled per sheet from a seed derived from the sample, items are
opaque ids, and exporting again gives the same sheets: nobody can reshuffle
until agreement looks good.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from ondine.ruler.cells import NO_VALUE, UNSURE, Cell, read_cell
from ondine.ruler.delivered import CertificationError, DeliveredValues, cell_value
from ondine.ruler.ruler import declaration_of
from ondine.ruler.sample import Sample
from ondine.ruler.sheet_files import Table, read_table, require_xlsx, write_table
from ondine.ruler.source import (
    CANNOT_TELL,
    NOT_LABELLED,
    Labellers,
    Reference,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ondine.harness.contract import OutputContract

#: A manifest of another version is refused.
FORMAT_VERSION = "review-v1"

FOR_LABELLERS = "for_labellers"
KEEP_PRIVATE = "keep_private"
MANIFEST_FILE = "manifest.json"
RUN_FILE = "run.json"
TASK_FILE = "task.txt"

ITEM = "item"
NOTES = "notes"
INPUT_PREFIX = "input: "
LABEL_PREFIX = "label: "
FORMATS = ("xlsx", "csv")
LETTERS = "AB"
#: Fewer than this share of rows labelled twice cannot measure agreement (D6).
MIN_DOUBLE = 0.2
#: Problems printed in an error; the rest are counted.
SHOWN = 50

_ITEM = re.compile(r"([0-9a-f]{6})-([A-Z])(\d+)")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewProblem:
    """One thing wrong with the returned sheets, located as exactly as it can be.

    Attributes:
        message: What is wrong.
        file: The sheet file, or None for a problem of the whole review.
        row: The sheet row, the header being row 1.
        item: The item id.
        column: The column header.
        col: The column number, from 1.
        suggestion: For a label that is nearly an allowed value, that value.
    """

    message: str
    file: str | None = None
    row: int | None = None
    item: str | None = None
    column: str | None = None
    col: int | None = None
    suggestion: str | None = None

    @property
    def cell(self) -> str | None:
        """The cell in spreadsheet notation, such as ``D5``."""
        if self.row is None or self.col is None:
            return None
        return f"{_column_letter(self.col)}{self.row}"

    def line(self) -> str:
        """The problem as one line of an error message."""
        where = [
            f"row {self.row}" if self.row is not None else None,
            f"item {self.item}" if self.item else None,
            f"{self.column} ({self.cell})"
            if self.column and self.cell
            else self.column,
        ]
        place = ", ".join(part for part in (self.file, *where) if part)
        hint = f" — did you mean {self.suggestion!r}?" if self.suggestion else ""
        return f"{place}: {self.message}{hint}" if place else f"{self.message}{hint}"


class ReviewSheetError(CertificationError):
    """The review cannot be read; it lists every problem it found.

    Attributes:
        problems: Every problem, as data, sorted by file, sheet row and column.
    """

    def __init__(self, problems: Sequence[ReviewProblem]) -> None:
        self.problems: tuple[ReviewProblem, ...] = tuple(
            sorted(
                problems,
                key=lambda p: (p.file or "", p.row or 0, p.col or 0, p.item or ""),
            )
        )
        super().__init__(_message(self.problems))

    @classmethod
    def refuse(cls, message: str, file: str | None = None) -> ReviewSheetError:
        """An error of one line, for a review refused outright."""
        return cls([ReviewProblem(message, file)])


def _message(problems: Sequence[ReviewProblem]) -> str:
    if len(problems) == 1 and problems[0].row is None and problems[0].item is None:
        return problems[0].line()
    shown = "\n  ".join(problem.line() for problem in problems[:SHOWN])
    more = f"\n  … {len(problems) - SHOWN} more" if len(problems) > SHOWN else ""
    noun = "problem" if len(problems) == 1 else "problems"
    return (
        f"{len(problems)} {noun} in the returned review sheets; fix them and read "
        f"again:\n  {shown}{more}"
    )


def _column_letter(number: int) -> str:
    letters = ""
    while number:
        number, rest = divmod(number - 1, 26)
        letters = chr(65 + rest) + letters
    return letters


# ---------------------------------------------------------------------------
# Dealing, seeds, ids, hashes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Deal:
    """Who labels which rows.

    Attributes:
        doubled: The rows both labellers are given, in input order.
        sheets: Each labeller's rows, in the order of their sheet.
    """

    doubled: tuple[int, ...]
    sheets: tuple[tuple[int, ...], ...]


def sheet_seed(sample_seed: int, labellers: int, double: float) -> int:
    """The seed of the dealing and the shuffles: derived, never chosen."""
    text = f"{sample_seed}:review:{labellers}:{float(double)}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def sheet_id_of(seed: int, rows: Sequence[int]) -> str:
    """A short id of one sheet set: its seed and the rows it holds."""
    text = f"{seed}:{','.join(map(str, rows))}"
    return hashlib.sha256(text.encode()).hexdigest()[:6]


def item_id(sheet_id: str, letter: str, position: int) -> str:
    """The opaque id of the item at ``position`` (from 0) of a labeller's sheet."""
    return f"{sheet_id}-{letter}{position + 1:03d}"


def deal(rows: Sequence[int], labellers: int, double: float, seed: int) -> Deal:
    """Pick the doubled rows, deal the rest alternately, then shuffle each sheet.

    One generator runs in this fixed order, so the same ``seed`` always gives
    the same sheets. Two labellers each get every doubled row; the others go
    alternately to A and B in input order. One labeller gets every row.
    """
    rng = np.random.default_rng(seed)
    n = len(rows)
    # round() first: 0.2 * 15 is 3.0000000000000004, and 4 rows is not a fifth of 15.
    twice = min(n, math.ceil(round(double * n, 9))) if labellers == 2 else 0
    picked = sorted(int(i) for i in rng.choice(n, size=twice, replace=False))
    chosen = set(picked)
    rest = [i for i in range(n) if i not in chosen]
    dealt = [rest[0::2], rest[1::2]] if labellers == 2 else [rest]
    sheets = []
    for own in dealt:
        members = sorted(picked + own)
        order = rng.permutation(len(members))
        sheets.append(tuple(rows[members[i]] for i in order))
    return Deal(tuple(rows[i] for i in picked), tuple(sheets))


def input_hash(text: str) -> str:
    """A hash of an input cell that survives a spreadsheet and a re-sort.

    The text is compared after Unicode NFC, with newlines unified and outer
    spaces stripped, so what Excel normalises does not look like an edit.
    """
    unified = (
        unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    )
    return hashlib.sha256(unified.strip().encode()).hexdigest()[:16]


def sheet_text(value: Any) -> str:
    """An input value as the text written to a sheet cell: no control characters."""
    cleaned = cell_value(value)
    return "" if cleaned is None else _CONTROL.sub("", str(cleaned))


# ---------------------------------------------------------------------------
# The manifest: binds the sheets to one run, holds no delivered value
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """One row of one labeller's sheet: its id, its run row, a hash of each input."""

    id: str
    row: int
    inputs: Mapping[str, str]


@dataclass(frozen=True)
class SheetPlan:
    """What one labeller was given: the file and the items, in sheet order."""

    file: str
    items: tuple[Item, ...]


@dataclass(frozen=True)
class Manifest:
    """Everything the import checks the returned sheets against.

    It holds no delivered value, so it is safe to commit.
    """

    sheet_id: str
    run_id: str
    fingerprint: str
    row_count: int
    contract_fingerprint: str
    declared: bool
    prompt_fingerprint: str
    task_fingerprint: str
    task_source: str
    sample: Mapping[str, Any]
    sheet_seed: int
    double: float
    labellers: int
    doubled: tuple[int, ...]
    format: str
    inputs: tuple[str, ...]
    labels: tuple[str, ...]
    sheets: Mapping[str, SheetPlan]
    numpy: str
    ondine: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": FORMAT_VERSION,
            "sheet_id": self.sheet_id,
            "run_id": self.run_id,
            "fingerprint": self.fingerprint,
            "row_count": self.row_count,
            "contract_fingerprint": self.contract_fingerprint,
            "declared": self.declared,
            "prompt_fingerprint": self.prompt_fingerprint,
            "task": {"fingerprint": self.task_fingerprint, "source": self.task_source},
            "sample": dict(self.sample),
            "sheet_seed": self.sheet_seed,
            "double": self.double,
            "labellers": self.labellers,
            "doubled": list(self.doubled),
            "format": self.format,
            "inputs": list(self.inputs),
            "labels": list(self.labels),
            "sheets": {
                letter: {
                    "file": plan.file,
                    "items": [
                        {"item": item.id, "row": item.row, "inputs": dict(item.inputs)}
                        for item in plan.items
                    ],
                }
                for letter, plan in self.sheets.items()
            },
            "numpy": self.numpy,
            "ondine": self.ondine,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        if data.get("version") != FORMAT_VERSION:
            raise ReviewSheetError.refuse(
                f"{MANIFEST_FILE} is version {data.get('version')!r}; "
                f"this Ondine reads {FORMAT_VERSION!r}"
            )
        try:
            return cls(
                sheet_id=str(data["sheet_id"]),
                run_id=str(data["run_id"]),
                fingerprint=str(data["fingerprint"]),
                row_count=int(data["row_count"]),
                contract_fingerprint=str(data["contract_fingerprint"]),
                declared=bool(data["declared"]),
                prompt_fingerprint=str(data["prompt_fingerprint"]),
                task_fingerprint=str(data["task"]["fingerprint"]),
                task_source=str(data["task"]["source"]),
                sample=dict(data["sample"]),
                sheet_seed=int(data["sheet_seed"]),
                double=float(data["double"]),
                labellers=int(data["labellers"]),
                doubled=tuple(int(row) for row in data["doubled"]),
                format=str(data["format"]),
                inputs=tuple(str(name) for name in data["inputs"]),
                labels=tuple(str(name) for name in data["labels"]),
                sheets={
                    str(letter): SheetPlan(
                        str(plan["file"]),
                        tuple(
                            Item(
                                str(entry["item"]),
                                int(entry["row"]),
                                {str(k): str(v) for k, v in entry["inputs"].items()},
                            )
                            for entry in plan["items"]
                        ),
                    )
                    for letter, plan in data["sheets"].items()
                },
                numpy=str(data["numpy"]),
                ondine=str(data["ondine"]),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ReviewSheetError.refuse(
                f"{MANIFEST_FILE} is malformed: {type(error).__name__}: {error}"
            ) from error


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewExport:
    """Where an export put its files, and what to do with them.

    Attributes:
        folder: The review folder.
        sheet_id: The id every item carries.
        format: "xlsx" or "csv".
        rows: Sampled rows.
        doubled: Rows both labellers are given.
        sheets: Labeller letter -> the sheet file to send them.
        task_file: The task text to send with each CSV sheet (CSV only).
        private: Files that must never be sent.
    """

    folder: Path
    sheet_id: str
    format: str
    rows: int
    doubled: int
    sheets: Mapping[str, Path]
    task_file: Path | None
    private: tuple[Path, ...]

    def render(self) -> str:
        """Who gets which file, and what must never be sent."""
        twice = f", {self.doubled} labelled twice" if self.doubled else ""
        lines = [
            f"Review sheet {self.sheet_id}: {self.rows} rows, "
            f"{len(self.sheets)} labeller{'s' if len(self.sheets) > 1 else ''}"
            f"{twice}, {self.format}",
            "  Send each labeller their own file, and nothing else:",
        ]
        lines += [f"    {letter}: {path}" for letter, path in self.sheets.items()]
        if self.task_file is not None:
            lines.append(f"    and the task text with each sheet: {self.task_file}")
        lines.append(
            f"  Never send {self.folder / KEEP_PRIVATE}: run.json holds the run's "
            "answers, and manifest.json maps each item to its row."
        )
        lines.append(
            f"  Read the returned sheets with pipeline.read_review({str(self.folder)!r})."
        )
        return "\n".join(lines)


def export(
    folder: str | Path,
    delivered: DeliveredValues,
    sample: Sample,
    contract: OutputContract,
    names: Sequence[str],
    *,
    labellers: int,
    double: float,
    fmt: str,
    task: str | None,
    system_prompt: str | None,
    template: str,
    prompt_fingerprint: str,
    declared: bool,
) -> ReviewExport:
    """Write the review folder for ``sample``: sheets to send, files to keep.

    Everything is checked before anything is written, and what was written is
    removed if writing fails, so a failed export leaves the folder as it was.

    Raises:
        CertificationError: The arguments cannot make a sheet, the folder is
            not empty, the sample is not this run's, or a delivered value
            cannot be saved.
        ImportError: ``fmt`` is "xlsx" and openpyxl is not installed.
    """
    if labellers not in (1, 2):
        raise CertificationError(
            f"labellers must be 1 or 2, not {labellers!r}: more are not supported yet"
        )
    double = float(double)
    if double < MIN_DOUBLE:
        raise CertificationError(
            f"double must be at least {MIN_DOUBLE}: agreement is measured on a "
            "fifth of the rows at least"
        )
    if double > 1:
        raise CertificationError(
            "double must be at most 1 (1 means every row is labelled twice)"
        )
    if fmt not in FORMATS:
        raise CertificationError(f"format must be 'xlsx' or 'csv', not {fmt!r}")
    root = Path(folder)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise CertificationError(
            f"{root} is not empty; export into a new folder, so sheets that came "
            "back are never overwritten"
        )
    sample.check(delivered, names)
    if sample.inputs.columns.empty:
        raise CertificationError(
            "the sample holds no input columns (it was restored from a review); "
            "draw one with draw_sample"
        )
    view = json.dumps(delivered.to_dict(), allow_nan=False)
    if fmt == "xlsx":
        require_xlsx()

    seed = sheet_seed(sample.seed, labellers, double)
    sheet_id = sheet_id_of(seed, sample.rows)
    dealt = deal(sample.rows, labellers, double, seed)
    columns = list(sample.inputs.columns)
    where = {row: position for position, row in enumerate(sample.rows)}
    texts = {c: [sheet_text(v) for v in sample.inputs[c].tolist()] for c in columns}
    body = _task_text(task, system_prompt, template, contract, names)

    header = (
        ITEM,
        *(INPUT_PREFIX + c for c in columns),
        *(LABEL_PREFIX + n for n in names),
        NOTES,
    )
    plans: dict[str, SheetPlan] = {}
    tables: dict[str, Table] = {}
    for letter, rows in zip(LETTERS, dealt.sheets, strict=False):
        items, records = [], []
        for position, row in enumerate(rows):
            at = where[row]
            ident = item_id(sheet_id, letter, position)
            cells = {c: texts[c][at] for c in columns}
            items.append(Item(ident, row, {c: input_hash(v) for c, v in cells.items()}))
            records.append((ident, *cells.values(), *([""] * len(names)), ""))
        plans[letter] = SheetPlan(f"sheet-{letter}.{fmt}", tuple(items))
        tables[letter] = Table(header, tuple(records))

    from ondine import __version__

    manifest = Manifest(
        sheet_id=sheet_id,
        run_id=delivered.run_id,
        fingerprint=delivered.fingerprint,
        row_count=delivered.row_count,
        contract_fingerprint=contract.fingerprint,
        declared=declared,
        prompt_fingerprint=prompt_fingerprint,
        task_fingerprint=hashlib.sha256(body.encode()).hexdigest()[:16],
        task_source="task" if task else "prompt",
        sample=sample.to_dict(),
        sheet_seed=seed,
        double=double,
        labellers=labellers,
        doubled=dealt.doubled,
        format=fmt,
        inputs=tuple(columns),
        labels=tuple(names),
        sheets=plans,
        numpy=np.__version__,
        ondine=__version__,
    )
    choices = _choices(contract, names)
    public, private = root / FOR_LABELLERS, root / KEEP_PRIVATE
    created = not root.exists()
    try:
        public.mkdir(parents=True)
        private.mkdir(parents=True)
        for letter, table in tables.items():
            write_table(
                public / plans[letter].file,
                table,
                task=body,
                choices=choices,
                text_columns=[LABEL_PREFIX + n for n in names],
            )
        task_file = None
        if fmt == "csv":
            task_file = public / TASK_FILE
            task_file.write_text(body + "\n", encoding="utf-8")
        (private / MANIFEST_FILE).write_text(
            json.dumps(manifest.to_dict(), indent=1) + "\n", encoding="utf-8"
        )
        (private / RUN_FILE).write_text(view, encoding="utf-8")
    except BaseException:
        shutil.rmtree(root if created else public, ignore_errors=True)
        if not created:
            shutil.rmtree(private, ignore_errors=True)
        raise
    return ReviewExport(
        folder=root,
        sheet_id=sheet_id,
        format=fmt,
        rows=len(sample.rows),
        doubled=len(dealt.doubled),
        sheets={letter: public / plan.file for letter, plan in plans.items()},
        task_file=task_file,
        private=(private / MANIFEST_FILE, private / RUN_FILE),
    )


def _choices(contract: OutputContract, names: Sequence[str]) -> dict[str, list[str]]:
    """Field -> the values its dropdown offers: allowed values, or true and false."""
    out: dict[str, list[str]] = {}
    for name in names:
        declaration = declaration_of(contract, name)
        if declaration.allowed:
            out[LABEL_PREFIX + name] = [str(value) for value in declaration.allowed]
        elif declaration.type == "boolean":
            out[LABEL_PREFIX + name] = ["true", "false"]
    return out


def _task_text(
    task: str | None,
    system_prompt: str | None,
    template: str,
    contract: OutputContract,
    names: Sequence[str],
) -> str:
    """What the Task worksheet and ``task.txt`` say: the task, the values, the syntax."""
    if task:
        lines = [task]
    else:
        lines = ["The model was asked:"]
        if system_prompt:
            lines += ["", "System prompt:", system_prompt]
        lines += ["", "Prompt:", template]
    lines += ["", "Fill each label column for every row, from the input columns only."]
    for name in names:
        lines.append(
            f"  {LABEL_PREFIX}{name}: {_describe(declaration_of(contract, name))}"
        )
    lines += [
        "",
        "How to fill a cell:",
        "  one answer: type it exactly, as listed",
        "  several answers are acceptable: separate them with |, such as a | b",
        f"  you cannot tell: type {UNSURE} alone in the cell",
        "  leave a cell empty only if you did not label the row",
        f"  the field has no value (optional fields only): type {NO_VALUE}",
        "  booleans: true or false, in any case; numbers: use a dot, such as 3.5",
        "Do not change the item or input columns, and do not delete rows. "
        "You may sort the rows. Extra columns and the notes column are ignored.",
    ]
    return "\n".join(lines)


def _describe(declaration: Any) -> str:
    if declaration.allowed:
        text = "one of " + ", ".join(str(value) for value in declaration.allowed)
    elif declaration.type == "boolean":
        text = "true or false"
    else:
        kind = "a whole number" if declaration.type == "integer" else "a number"
        low, high = declaration.minimum, declaration.maximum
        if low is not None and high is not None:
            text = f"{kind} from {low:g} to {high:g}"
        elif low is not None:
            text = f"{kind}, at least {low:g}"
        elif high is not None:
            text = f"{kind}, at most {high:g}"
        else:
            text = kind
    return (
        text if declaration.required else f"{text} (optional: {NO_VALUE} for no value)"
    )


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


class Progress(NamedTuple):
    """How many of a field's cells on one sheet are labelled, "?" or empty."""

    labelled: int
    unsure: int
    empty: int


@dataclass(frozen=True)
class SheetProgress:
    """How far one labeller got, per field."""

    sheet: str
    rows: int
    counts: Mapping[str, Progress]


@dataclass(frozen=True)
class ReviewLabels:
    """People's labels, read and checked: pass them to ``certify`` as the gold.

    Attributes:
        sheet_id: The sheets' id; the certificate names it.
        labellers: 1 or 2.
        fields: The certified fields.
        run: The exported run's view, restored: certify it.
        sample: The sample the sheets were made from, redrawn; it carries no
            input columns.
        reference: The labels as the Ruler's reference.
        progress: Per sheet, per field, the cells labelled, "?" and empty.
        states: Labeller letter -> field -> run row -> "answer", "unsure" (a
            ``?``) or "empty", for every row on that labeller's sheet.
    """

    sheet_id: str
    labellers: int
    fields: tuple[str, ...]
    run: DeliveredValues = field(repr=False)
    sample: Sample = field(repr=False, compare=False)
    reference: Reference = field(repr=False)
    progress: tuple[SheetProgress, ...]
    states: Mapping[str, Mapping[str, Mapping[int, str]]] = field(repr=False)

    def render(self) -> str:
        """Progress per sheet: cells labelled, marked "?" and left empty."""
        doubled = self.reference.labellers.doubled if self.reference.labellers else ()
        twice = f" ({len(doubled)} labelled twice)" if doubled else ""
        lines = [
            f"Review sheet {self.sheet_id}: {len(self.sample.rows)} sampled rows, "
            f"{self.labellers} labeller{'s' if self.labellers > 1 else ''}{twice}"
        ]
        for sheet in self.progress:
            lines.append(f"  Sheet {sheet.sheet}: {sheet.rows} rows")
            lines += [
                f'    {name}: {c.labelled} labelled, {c.unsure} "?", {c.empty} empty'
                for name, c in sheet.counts.items()
            ]
        lines.append(
            '  A cell marked "?" or left empty is unresolved: it widens the range, '
            "it never narrows it."
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class SavedReview:
    """A review folder's private files, read: the manifest and the run's view."""

    manifest: Manifest
    view: DeliveredValues


def load_review(folder: str | Path) -> SavedReview:
    """Read ``keep_private`` of a review folder and check the two files agree.

    Raises:
        ReviewSheetError: The folder holds no review, a file is malformed, or
            ``run.json`` is not the run the manifest names.
        CertificationError: ``run.json`` was changed after it was saved.
    """
    private = Path(folder) / KEEP_PRIVATE
    manifest = Manifest.from_dict(_load_json(private / MANIFEST_FILE, folder))
    view = DeliveredValues.from_dict(_load_json(private / RUN_FILE, folder))
    if (view.run_id, view.fingerprint) != (manifest.run_id, manifest.fingerprint):
        raise ReviewSheetError.refuse(
            f"{RUN_FILE} is not the run these sheets were exported with "
            f"(run {view.run_id[:8]}…, delivered values {view.fingerprint[:8]}…; "
            f"the manifest names run {manifest.run_id[:8]}…, "
            f"values {manifest.fingerprint[:8]}…)"
        )
    return SavedReview(manifest, view)


def _load_json(path: Path, folder: str | Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ReviewSheetError.refuse(
            f"{folder} is not a review folder: {KEEP_PRIVATE}/{path.name} not found"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ReviewSheetError.refuse(f"{path.name} cannot be read: {error}") from error
    if not isinstance(data, dict):
        raise ReviewSheetError.refuse(f"{path.name} is not a JSON object")
    return data


def read_labels(
    saved: SavedReview,
    contract: OutputContract,
    names: Sequence[str],
    folder: str | Path,
    *,
    sheets: Mapping[str, str | Path] | None = None,
    current: DeliveredValues | None = None,
) -> ReviewLabels:
    """Read and check the returned sheets; list every problem at once.

    Args:
        saved: The review's private files.
        contract: The rules the labels are read by: this pipeline's.
        names: The certifiable fields, as this pipeline has them.
        folder: The review folder.
        sheets: Labeller letter -> the returned file, where it is not the
            exported one.
        current: This session's delivered values, to check they are the
            exported run's.

    Raises:
        ReviewSheetError: The review is refused outright (another run, values,
            contract or sample; a sheet of another sample), or one or more
            sheets have problems.
    """
    manifest = saved.manifest
    if contract.fingerprint != manifest.contract_fingerprint:
        raise ReviewSheetError.refuse(
            "the sheets were exported under another output contract "
            f"({manifest.contract_fingerprint[:8]}…) than this pipeline's "
            f"({contract.fingerprint[:8]}…); build the pipeline with the same "
            "declarations and contract policy"
        )
    if tuple(names) != manifest.labels:
        raise ReviewSheetError.refuse(
            f"the sheets label {', '.join(manifest.labels)}, not "
            f"{', '.join(names)}, which this pipeline certifies"
        )
    if current is not None:
        if current.run_id != manifest.run_id:
            raise ReviewSheetError.refuse(
                f"these sheets were exported from run {manifest.run_id[:8]}…, "
                f"another run than this result's ({current.run_id[:8]}…)"
            )
        if current.fingerprint != manifest.fingerprint:
            raise ReviewSheetError.refuse(
                "the delivered values changed since the sheets were exported"
            )
    try:
        sample = Sample.restore(manifest.sample, saved.view, names)
    except CertificationError as error:
        raise ReviewSheetError.refuse(
            f"{error}; the sheets were exported with numpy {manifest.numpy}, "
            f"this is numpy {np.__version__}"
        ) from error
    _check_deal(manifest, sample)

    read = {
        letter: _read_sheet(
            letter,
            plan,
            Path(sheets[letter])
            if sheets and letter in sheets
            else Path(folder) / FOR_LABELLERS / Path(plan.file).name,
            manifest,
            contract,
            names,
        )
        for letter, plan in manifest.sheets.items()
    }
    problems = [p for sheet in read.values() for p in sheet.problems]
    moved = set().union(*(sheet.moved for sheet in read.values()))
    for letter, plan in manifest.sheets.items():
        if read[letter].usable:
            problems += [
                ReviewProblem(
                    "missing: this item is not in the returned sheet "
                    "(was a row deleted?)",
                    read[letter].file,
                    item=item.id,
                )
                for item in plan.items
                if item.id not in read[letter].seen and item.id not in moved
            ]
    if problems:
        raise ReviewSheetError(problems)
    return _assemble(manifest, read, names, saved.view, sample)


def _check_deal(manifest: Manifest, sample: Sample) -> None:
    """Refuse a manifest that is not the dealing its own seed gives.

    The dealing is derived, so it is recomputed from the redrawn sample: the
    sheet seed and id, the doubled rows, and who got which rows in which
    order. An edited manifest (rows moved, items renamed) differs.
    """
    seed = sheet_seed(sample.seed, manifest.labellers, manifest.double)
    dealt = deal(sample.rows, manifest.labellers, manifest.double, seed)
    letters = LETTERS[: manifest.labellers]
    plans = manifest.sheets
    consistent = (
        seed == manifest.sheet_seed
        and sheet_id_of(seed, sample.rows) == manifest.sheet_id
        and dealt.doubled == manifest.doubled
        and tuple(plans) == tuple(letters)
        and all(
            tuple(item.row for item in plans[letter].items) == rows
            and tuple(item.id for item in plans[letter].items)
            == tuple(item_id(manifest.sheet_id, letter, i) for i in range(len(rows)))
            for letter, rows in zip(letters, dealt.sheets, strict=True)
        )
    )
    if not consistent:
        raise ReviewSheetError.refuse(
            f"{MANIFEST_FILE} is not the dealing its seed gives (edited, or "
            f"another numpy: exported with numpy {manifest.numpy}, this is "
            f"numpy {np.__version__}); the sheets cannot be matched to rows"
        )


@dataclass
class _SheetRead:
    """What reading one returned sheet found."""

    file: str
    usable: bool = True
    problems: list[ReviewProblem] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    moved: set[str] = field(default_factory=set)
    cells: dict[int, dict[str, Cell]] = field(default_factory=dict)


def _read_sheet(
    letter: str,
    plan: SheetPlan,
    path: Path,
    manifest: Manifest,
    contract: OutputContract,
    names: Sequence[str],
) -> _SheetRead:
    read = _SheetRead(path.name)
    if not path.is_file():
        read.usable = False
        read.problems.append(ReviewProblem(f"the sheet is not at {path}", path.name))
        return read
    try:
        table = read_table(path)
    except ImportError:
        raise
    except Exception as error:
        read.usable = False
        message = f"cannot be read as a {path.suffix.lstrip('.')} sheet: {type(error).__name__}: {error}"
        read.problems.append(ReviewProblem(message, path.name))
        return read
    _refuse_foreign(table, manifest.sheet_id, path.name)

    index: dict[str, int] = {}
    for position, name in enumerate(table.header):
        if name in index:
            read.problems.append(
                ReviewProblem(
                    "duplicate column", path.name, 1, column=name, col=position + 1
                )
            )
        index.setdefault(name, position)
    wanted = [
        ITEM,
        *(INPUT_PREFIX + c for c in manifest.inputs),
        *(LABEL_PREFIX + n for n in names),
    ]
    for name in wanted:
        if name not in index:
            read.usable = False
            read.problems.append(
                ReviewProblem(f"missing column {name!r}", path.name, 1, column=name)
            )
    for position, name in enumerate(table.header):
        if name.startswith((INPUT_PREFIX, LABEL_PREFIX)) and name not in wanted:
            read.problems.append(
                ReviewProblem(
                    f"unknown column {name!r}",
                    path.name,
                    1,
                    column=name,
                    col=position + 1,
                )
            )
    if not read.usable:
        return read

    by_id = {item.id: item for item in plan.items}
    for offset, record in enumerate(table.rows):
        number = offset + 2
        if not any(cell.strip() for cell in record):
            continue
        item = record[index[ITEM]].strip()

        def problem(message, column=ITEM, *, number=number, item=item, **kw):
            return ReviewProblem(
                message,
                path.name,
                number,
                item or None,
                column,
                index[column] + 1,
                **kw,
            )

        found = _ITEM.fullmatch(item)
        if (
            found is None
            or found[1] != manifest.sheet_id
            or found[2] not in manifest.sheets
        ):
            read.problems.append(problem("not an item of this review"))
        elif found[2] != letter:
            read.moved.add(item)
            read.problems.append(
                problem(
                    f"this item belongs to the other labeller's file "
                    f"({manifest.sheets[found[2]].file}); each labeller keeps their own rows"
                )
            )
        elif item not in by_id:
            read.problems.append(problem("not an item of this sheet"))
        elif item in read.seen:
            read.problems.append(problem("duplicate item: this row appears twice"))
        else:
            read.seen.add(item)
            _read_row(
                read, by_id[item], record, index, problem, contract, names, manifest
            )
    return read


def _read_row(read, planned, record, index, problem, contract, names, manifest) -> None:
    """One sheet row's inputs checked against the manifest, its labels read."""
    for column in manifest.inputs:
        name = INPUT_PREFIX + column
        if input_hash(record[index[name]]) != planned.inputs[column]:
            read.problems.append(
                problem(
                    "the input differs from the exported one (edited, or a "
                    "column sorted alone)",
                    name,
                )
            )
    row = {}
    for name in names:
        column = LABEL_PREFIX + name
        cell = read_cell(contract, name, record[index[column]], planned.row)
        for found in cell.problems:
            why = "; ".join(found.reasons) or "needs a format fix; give the exact value"
            hint = f" ({cell.hint})" if cell.hint else ""
            read.problems.append(
                problem(
                    f"{found.raw!r} — {why}{hint}",
                    column,
                    suggestion=cell.suggestions.get(str(found.raw)),
                )
            )
        row[name] = cell
    read.cells[planned.row] = row


def _refuse_foreign(table: Table, sheet_id: str, file: str) -> None:
    """Refuse a sheet none of whose items carry this sheet's id.

    A sheet with some of its own items and a few foreign ones is read, and
    each foreign row is listed with the other problems.
    """
    if ITEM not in table.header:
        return
    at = table.header.index(ITEM)
    ids = {
        found[1]
        for record in table.rows
        if (found := _ITEM.fullmatch(record[at].strip()))
    }
    if ids and sheet_id not in ids:
        raise ReviewSheetError.refuse(
            f"{file} is from sheet {', '.join(sorted(ids))} (another sample or "
            f"run), but this folder holds sheet {sheet_id}",
            file,
        )


def _assemble(
    manifest: Manifest,
    read: Mapping[str, _SheetRead],
    names: Sequence[str],
    view: DeliveredValues,
    sample: Sample,
) -> ReviewLabels:
    """The checked cells as the Ruler's reference, and each sheet's progress."""
    owners: dict[int, list[str]] = {}
    for letter, plan in manifest.sheets.items():
        for item in plan.items:
            owners.setdefault(item.row, []).append(letter)
    answers: dict[str, dict[int, Any]] = {name: {} for name in names}
    second: dict[str, dict[int, Any]] = {name: {} for name in names}
    causes: dict[str, dict[int, str]] = {name: {} for name in names}
    for row, letters in owners.items():
        for name in names:
            cells = [read[letter].cells[row][name] for letter in sorted(letters)]
            if cells[0].state == "answer":
                answers[name][row] = cells[0].raw
            if len(cells) == 2 and cells[1].state == "answer":
                second[name][row] = cells[1].raw
            if not any(cell.state == "answer" for cell in cells):
                unsure = any(cell.state == "unsure" for cell in cells)
                causes[name][row] = CANNOT_TELL if unsure else NOT_LABELLED
    by_sheet = {
        letter: {
            name: {row: cells[name].state for row, cells in read[letter].cells.items()}
            for name in names
        }
        for letter in manifest.sheets
    }
    progress = []
    for letter in manifest.sheets:
        counts = {}
        for name in names:
            states = Counter(cell[name].state for cell in read[letter].cells.values())
            counts[name] = Progress(states["answer"], states["unsure"], states["empty"])
        progress.append(SheetProgress(letter, len(read[letter].cells), counts))
    reference = Reference(
        "people",
        answers,
        causes,
        sheet=manifest.sheet_id,
        labellers=Labellers(manifest.labellers, frozenset(manifest.doubled), second),
    )
    return ReviewLabels(
        sheet_id=manifest.sheet_id,
        labellers=manifest.labellers,
        fields=tuple(names),
        run=view,
        sample=sample,
        reference=reference,
        progress=tuple(progress),
        states=by_sheet,
    )
