"""How reference answers become a claim about each field.

Reads each reference value by the field's own contract rules, the way the
model's answer was read (:func:`read_reference`); compares it with the
delivered value (:func:`_compare`); and turns the comparisons into exact
ranges, breakdowns, confusions and wrong rows.

The reference is one :class:`~ondine.ruler.source.Reference`, whoever made
it (:mod:`ondine.ruler.source`). Its kind and the rows it covers decide the claim:

* labels, on a sample Ondine drew: a certificate with a range;
* labels, on every delivered row: a census certificate, exact;
* labels, on rows the caller picked: a check — agreement on those rows, no
  range;
* a reference model's answers: the same comparison makes an audit, agreement
  with that model, never accuracy;
* people's labels from a review sheet, on a sample Ondine drew: a
  certificate, "accuracy against blind human labels". Two labellers may have
  answered the same row; their answers are judged one by one against the
  delivered value (:func:`_compare`), and their agreement is a number of its own.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import pandas as pd

from ondine.harness.judge import read_field
from ondine.ruler.certificate import (
    AUDIT,
    CERTIFICATE,
    CHECK,
    FEW_ROWS,
    Certificate,
    Confusion,
    Estimate,
    FieldCertificate,
    Group,
    LabellerAgreement,
    WrongRow,
)
from ondine.ruler.delivered import CertificationError
from ondine.ruler.intervals import hypergeom_bounds
from ondine.ruler.source import (
    ABSENT,
    LABELLERS_DISAGREE,
    NO_ANSWER,
    ReferenceProblem,
)

if TYPE_CHECKING:
    from ondine.harness.contract import OutputContract, OutputField
    from ondine.ruler.delivered import DeliveredValues
    from ondine.ruler.sample import Sample
    from ondine.ruler.source import Reference

#: Confusions listed per field.
TOP_CONFUSIONS = 5


class Outcome(NamedTuple):
    """What comparing one row's delivered value with its reference found.

    Attributes:
        state: "right", "wrong" or "unresolved".
        value: The delivered value.
        accepted: The acceptable references; for a row two labellers
            answered, the first's answers, then the second's others.
        cause: Why the row is unresolved, when the comparison itself knows
            (the labellers disagree); None when the reference says, or says nothing.
        pair: For a row two labellers were given: each one's acceptable
            answers (empty when they gave none); None for any other row.
    """

    state: str
    value: Any
    accepted: tuple[Any, ...]
    cause: str | None = None
    pair: tuple[tuple[Any, ...], tuple[Any, ...]] | None = None


def certifiable(contract: OutputContract) -> tuple[list[str], list[tuple[str, str]]]:
    """The fields v1 can certify, and (field, reason) for the others."""
    fields: list[str] = []
    skipped: list[tuple[str, str]] = []
    for declaration in contract.fields:
        if _kind(declaration) is not None:
            fields.append(declaration.name)
        else:
            skipped.append(
                (
                    declaration.name,
                    f"{declaration.type} without allowed values: free text is not certified yet",
                )
            )
    return fields, skipped


def _kind(declaration: OutputField) -> str | None:
    if declaration.type in ("boolean", "integer", "number"):
        return declaration.type
    if declaration.type == "text" and declaration.allowed:
        return "labels"
    return None


def resolve_gold(
    gold: Any,
    fields: Sequence[str],
    row_count: int,
    frame: pd.DataFrame | None = None,
) -> dict[str, dict[int, Any]]:
    """Each field's reference value per row, from what the caller passed.

    ``gold`` is a column name of the result frame; values aligned with the
    input rows; a mapping (or a pandas Series) row → value; or, for several
    fields, a mapping field → any of those. Missing cells are left out
    (unresolved). Without ``frame`` (a saved view of a run has none) a column
    name is refused.
    """
    if isinstance(gold, Mapping) and gold and all(key in fields for key in gold):
        per_field = dict(gold)
    elif len(fields) == 1:
        per_field = {fields[0]: gold}
    else:
        raise CertificationError(
            f"the run has {len(fields)} certifiable fields ({', '.join(fields)}); "
            "pass gold as a mapping field -> labels"
        )
    out: dict[str, dict[int, Any]] = {}
    for name, source in per_field.items():
        if isinstance(source, str):
            if frame is None:
                raise CertificationError(
                    f"gold for {name!r} names the column {source!r}, but a saved "
                    "view has no result frame; pass the labels as values or a "
                    "mapping row -> label"
                )
            if source not in frame.columns:
                raise CertificationError(
                    f"no column {source!r} in the result for field {name!r}"
                )
            source = frame[source].tolist()
        if isinstance(source, Mapping | pd.Series):
            # A Series is read by its index, the row positions, so a slice
            # such as gold[sample.rows] works like a mapping.
            cells = {int(row): value for row, value in source.items()}
        else:
            values = list(source)
            if len(values) != row_count:
                raise CertificationError(
                    f"gold for {name!r} has {len(values)} values; the run has {row_count} rows"
                )
            cells = dict(enumerate(values))
        out[name] = {row: value for row, value in cells.items() if not _missing(value)}
    return out


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, list | tuple | set | frozenset):
        return all(_missing(v) for v in value)
    return False


def certify(
    delivered: DeliveredValues,
    contract: OutputContract,
    fields: Sequence[str],
    not_certified: Sequence[tuple[str, str]],
    reference: Reference,
    *,
    sample: Sample | None = None,
    tolerance: Mapping[str, float] | None = None,
    read_delivered: bool = False,
) -> Certificate:
    """Compare delivered values with a reference and state the claim.

    Args:
        delivered: The run's delivered values.
        contract: The rules values are read and compared by.
        fields: The fields to certify (all certifiable).
        not_certified: Output fields left out, with the reason.
        reference: What to compare with; its kind and coverage decide the
            claim (a certificate, an audit or a check).
        sample: The run's Ondine-drawn sample, if the reference covers it.
        tolerance: Field -> allowed absolute difference, for numbers.
        read_delivered: Read delivered values by the contract too (a run with
            no contract delivers raw text).

    Raises:
        CertificationError: The sample belongs to another run or other
            values, or reference values break the field's rules (every one
            listed).
    """
    tolerance = dict(tolerance or {})
    if sample is not None:
        sample.check(delivered, fields)
    declarations = {d.name: d for d in contract.fields}
    labellers = reference.labellers
    problems: list[ReferenceProblem] = []
    outcomes: dict[str, dict[int, Outcome]] = {}
    for name in fields:
        population = delivered.population(name)
        answers = reference.answers[name]
        second_answers = labellers.second.get(name, {}) if labellers else {}
        rows = _compared_rows(population, answers, sample)
        outcomes[name] = {}
        for row in rows:
            value = delivered.values[name][row]
            if read_delivered:
                value = _read_delivered(contract, name, value)
            accepted: tuple[Any, ...] = ()
            if row in answers:
                accepted, found = read_reference(contract, name, answers[row], row)
                problems += found
            second: tuple[Any, ...] | None = None
            if labellers is not None and row in labellers.doubled:
                second = ()
                if row in second_answers:
                    second, found = read_reference(
                        contract, name, second_answers[row], row
                    )
                    problems += found
            outcomes[name][row] = _compare(
                value, accepted, tolerance.get(name, 0.0), second
            )
    if problems:
        shown = "\n  ".join(problem.line() for problem in problems[:50])
        more = f"\n  … {len(problems) - 50} more" if len(problems) > 50 else ""
        raise CertificationError(
            f"{len(problems)} reference value(s) break the field's rules; fix them and certify again:\n  {shown}{more}"
        )

    coverage = _coverage(delivered, fields, reference, sample)
    results = tuple(
        _field_certificate(
            delivered,
            declarations[name],
            outcomes[name],
            coverage,
            tolerance.get(name, 0.0),
            # Labels give no cause for an unresolved row; other sources do.
            None if reference.kind == "labels" else reference.causes.get(name, {}),
            labeller_agreement(
                outcomes[name],
                labellers.count,
                _kind(declarations[name]),
                tolerance.get(name, 0.0),
            )
            if labellers is not None
            else None,
        )
        for name in fields
    )
    return Certificate(
        claim=_claim(reference, coverage),
        claim_text=_claim_text(reference, coverage, results),
        run_id=delivered.run_id,
        fingerprint=delivered.fingerprint,
        contract_fingerprint=contract.fingerprint,
        basis=_basis(coverage, sample, results),
        seed=sample.seed if sample is not None else None,
        fields=results,
        not_certified=tuple(not_certified),
        row_accuracy=_row_accuracy(delivered, fields, outcomes, coverage)
        if len(fields) > 1
        else None,
        sheet=reference.sheet,
    )


def _compared_rows(
    population: Sequence[int], gold: Mapping[int, Any], sample: Sample | None
) -> list[int]:
    if sample is not None:
        drawn = set(sample.rows)
        return [row for row in population if row in drawn]
    return [row for row in population if row in gold]


def _coverage(delivered, fields, reference, sample) -> str:
    """ "sample", "census" or CHECK: which rows the reference covers."""
    if sample is not None:
        return "census" if sample.census else "sample"
    full = all(
        all(row in reference.answers[name] for row in delivered.population(name))
        for name in fields
    )
    return "census" if full else CHECK


def _claim(reference, coverage) -> str:
    """CERTIFICATE, AUDIT or CHECK: what kind of statement this is."""
    if reference.kind == "model":
        return AUDIT
    return CHECK if coverage == CHECK else CERTIFICATE


def _read_delivered(contract: OutputContract, name: str, value: Any) -> Any:
    verdict = read_field(contract, name, value)
    return verdict.value if verdict.outcome == "value" else _Unreadable(value)


class _Unreadable:
    """A delivered value (no contract) that breaks the declaration: counted wrong."""

    def __init__(self, raw: Any) -> None:
        self.raw = raw

    def __repr__(self) -> str:
        return repr(self.raw)

    def __eq__(self, other: object) -> bool:
        return False

    __hash__ = object.__hash__


def read_reference(
    contract: OutputContract, name: str, raw: Any, row: int
) -> tuple[tuple[Any, ...], list[ReferenceProblem]]:
    """The values a reference cell accepts, read by the field's contract rules.

    ``raw`` is one answer or a list of acceptable answers; an empty answer is
    skipped. Each is read as a model's answer would be, no more loosely.
    A value that breaks the field's rules, or would need a format fix, is not
    accepted: it becomes a problem and the others are still read.
    :data:`~ondine.ruler.source.ABSENT` means "no value", which only an
    optional field can have.

    Returns:
        The accepted values (``None`` for an answer that means "no value"),
        empty when the cell gives no usable answer; and the problems found.
    """
    raws = list(raw) if isinstance(raw, list | tuple | set | frozenset) else [raw]
    accepted: list[Any] = []
    problems: list[ReferenceProblem] = []
    for item in raws:
        if item is ABSENT:
            if declaration_of(contract, name).required:
                reason = "'-' means no value, but this field is required"
                problems.append(ReferenceProblem(name, row, "-", (reason,)))
            else:
                accepted.append(None)
            continue
        if _missing(item):
            continue
        verdict = read_field(contract, name, item)
        if verdict.outcome == "failed" or verdict.format_fixed:
            reasons = tuple(r.text for r in verdict.reasons)
            problems.append(ReferenceProblem(name, row, item, reasons))
            continue
        accepted.append(verdict.value if verdict.outcome == "value" else None)
    return tuple(accepted), problems


def declaration_of(contract: OutputContract, name: str) -> OutputField:
    """The declaration of output field ``name``."""
    return next(d for d in contract.fields if d.name == name)


def _compare(
    value: Any,
    accepted: tuple[Any, ...],
    tol: float,
    second: tuple[Any, ...] | None = None,
) -> Outcome:
    """Judge a delivered value against its reference.

    ``second`` is the second labeller's acceptable answers on a row two
    labellers were given (empty if they gave none), None on any other row.
    Each labeller's answers are judged on their own: both right is right,
    both wrong is wrong, one of each is unresolved ("labellers disagree").
    If only one answered, that answer counts; if neither, the row is
    unresolved and its cause is the reference's.
    """
    if second is None:  # the common row, and the one that runs per delivered row
        if not accepted:
            return Outcome("unresolved", value, ())
        right = any(_equal(value, a, tol) for a in accepted)
        return Outcome("right" if right else "wrong", value, accepted)
    pair = (accepted, second)
    if accepted and second:
        first_right = any(_equal(value, a, tol) for a in accepted)
        second_right = any(_equal(value, a, tol) for a in second)
        merged = accepted + tuple(a for a in second if a not in accepted)
        if first_right != second_right:
            return Outcome("unresolved", value, merged, LABELLERS_DISAGREE, pair)
        return Outcome("right" if first_right else "wrong", value, merged, None, pair)
    return _compare(value, accepted or second, tol)._replace(pair=pair)


def _equal(value: Any, ref: Any, tol: float) -> bool:
    if isinstance(value, _Unreadable) or value is None or ref is None:
        return False
    if isinstance(value, bool) or isinstance(ref, bool):
        return bool(value == ref)
    if isinstance(value, int | float) and isinstance(ref, int | float):
        return abs(value - ref) <= tol
    return bool(value == ref)


def _estimate(population: int, outcomes: Sequence[str], check: bool) -> Estimate:
    n = len(outcomes)
    right = sum(o == "right" for o in outcomes)
    unresolved = sum(o == "unresolved" for o in outcomes)
    if check or n == 0:
        return Estimate(population, n, right, unresolved, None, None)
    if n == population:
        return Estimate(
            population,
            n,
            right,
            unresolved,
            right / population,
            (right + unresolved) / population,
        )
    lo, _ = hypergeom_bounds([right], n, population)
    _, hi = hypergeom_bounds([right + unresolved], n, population)
    return Estimate(
        population,
        n,
        right,
        unresolved,
        int(lo[0]) / population,
        int(hi[0]) / population,
    )


def _field_certificate(
    delivered, declaration, outcomes, coverage, tol, causes=None, agreement=None
) -> FieldCertificate:
    """``causes``: row -> the reference's cause for a row with no answer, or
    None to report none; a cause the comparison found (an outcome's own) wins."""
    name = declaration.name
    population = delivered.population(name)
    check = coverage == CHECK
    states = [outcomes[row].state for row in sorted(outcomes)]
    estimate = _estimate(len(population), states, check)

    def groups(key) -> tuple[Group, ...]:
        sizes = Counter(key(row) for row in population)
        compared: dict[Any, list[str]] = {}
        for row in sorted(outcomes):
            compared.setdefault(key(row), []).append(outcomes[row].state)
        lines = []
        for value, size in sizes.most_common():
            est = _estimate(size, compared.get(value, []), check)
            few = coverage == "sample" and est.resolved < FEW_ROWS
            lines.append(Group(str(value), est, few))
        return tuple(lines)

    wrong = [(row, o) for row, o in sorted(outcomes.items()) if o.state == "wrong"]
    confusions = Counter(
        (repr(o.value) if isinstance(o.value, _Unreadable) else o.value, o.accepted[0])
        for _, o in wrong
    )
    return FieldCertificate(
        field=name,
        kind=_kind(declaration) or "labels",
        rule=_rule(declaration, tol),
        estimate=estimate,
        by_status=groups(lambda row: delivered.statuses[row].value),
        by_model=groups(lambda row: delivered.models[row] or "unknown"),
        by_value=groups(lambda row: delivered.values[name][row]),
        confusions=tuple(
            Confusion(d, r, c) for (d, r), c in confusions.most_common(TOP_CONFUSIONS)
        ),
        wrong_rows=tuple(
            WrongRow(
                row,
                o.value if not isinstance(o.value, _Unreadable) else o.value.raw,
                o.accepted,
            )
            for row, o in wrong
        ),
        not_delivered=_not_delivered(delivered, name, population),
        unresolved_causes=dict(
            Counter(
                o.cause or causes.get(row, NO_ANSWER)
                for row, o in sorted(outcomes.items())
                if o.state == "unresolved"
            )
        )
        if causes is not None
        else {},
        labellers=agreement,
    )


def cohens_kappa(pairs: Sequence[tuple[Any, Any]]) -> float | None:
    """Cohen's kappa of two labellers' single answers, row by row.

    Observed agreement corrected for the agreement their label frequencies
    give by chance: 1 for perfect agreement, 0 for chance, negative below it.
    ``None`` has no value to say: no pair, or both only ever used one label.
    """
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(a == b for a, b in pairs) / n
    first = Counter(a for a, _ in pairs)
    second = Counter(b for _, b in pairs)
    chance = sum(first[label] * second[label] for label in first) / (n * n)
    if chance == 1:
        return None
    return float((observed - chance) / (1 - chance))


def labeller_agreement(
    outcomes: Mapping[int, Outcome], labellers: int, kind: str | None, tol: float
) -> LabellerAgreement:
    """How far the labellers agreed, on a field's compared rows they both had.

    With one labeller nothing is measured. Kappa is for labels and booleans,
    and uses the rows where each person gave exactly one answer.
    """
    if labellers < 2:
        return LabellerAgreement(labellers, 0, 0, 0, 0, None, 0)
    pairs = [o.pair for _, o in sorted(outcomes.items()) if o.pair is not None]
    both = [(a, b) for a, b in pairs if a and b]
    singles = [(a[0], b[0]) for a, b in both if len(a) == 1 and len(b) == 1]
    scored = kind in ("labels", "boolean")
    return LabellerAgreement(
        labellers=labellers,
        doubled=len(pairs),
        both_answered=len(both),
        agree=sum(_share(a, b, tol) for a, b in both),
        split=sum(o.cause == LABELLERS_DISAGREE for o in outcomes.values()),
        kappa=cohens_kappa(singles) if scored else None,
        kappa_rows=len(singles) if scored else 0,
    )


def _share(first: tuple[Any, ...], second: tuple[Any, ...], tol: float) -> bool:
    """Whether two answer sets have a value in common (no value agrees with no value)."""
    return any(
        (a is None and b is None) or _equal(a, b, tol) for a in first for b in second
    )


def _not_delivered(delivered, name, population) -> dict[str, int]:
    inside = set(population)
    causes: Counter[str] = Counter()
    for row in range(delivered.row_count):
        if row in inside:
            continue
        status = delivered.statuses[row].value
        causes[status if status in ("rejected", "failed") else "empty"] += 1
    return dict(causes)


def _rule(declaration, tol) -> str:
    kind = _kind(declaration)
    if kind == "labels":
        return (
            "compared as allowed values, after the contract's normalisation and aliases"
        )
    if kind == "boolean":
        return "compared as booleans"
    return (
        f"compared as numbers, within ±{tol:g}"
        if tol
        else "compared as numbers, exactly"
    )


def _row_accuracy(delivered, fields, outcomes, coverage) -> Estimate:
    population = sorted(
        set.intersection(*(set(delivered.population(n)) for n in fields))
    )
    compared = [row for row in population if all(row in outcomes[n] for n in fields)]
    states = []
    for row in compared:
        per = [outcomes[n][row].state for n in fields]
        states.append(
            "wrong"
            if "wrong" in per
            else "unresolved"
            if "unresolved" in per
            else "right"
        )
    return _estimate(len(population), states, coverage == CHECK)


def _claim_text(reference, coverage: str, results) -> str:
    if reference.kind == "model":
        rows = "every delivered row" if coverage == "census" else "a sample Ondine drew"
        return f"agreement with {reference.model}, on {rows} — an audit, not accuracy"
    if reference.kind == "people":
        rows = "every delivered row" if coverage == "census" else "a sample Ondine drew"
        return f"accuracy against blind human labels, on {rows}"
    if coverage == "sample":
        return "accuracy against your labels, on a sample Ondine drew"
    if coverage == "census":
        return "accuracy against your labels, on every delivered row"
    labelled = max((r.estimate.compared for r in results), default=0)
    return f"agreement with your labels on the {labelled} rows you labelled — not a claim about the column"


def _basis(coverage: str, sample, results) -> str:
    if sample is not None:
        return f"sample of {len(sample.rows)} of {sample.population_size} delivered rows ({sample.rule})"
    if coverage == "census":
        return "every delivered row has a label"
    return "rows you labelled, not drawn by Ondine"
