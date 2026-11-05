"""How reference answers become a claim about each field.

Reads each reference (gold) value by the field's own contract rules, the way
the model's answer was read; compares it with the delivered value; and turns
the comparisons into exact ranges, breakdowns, confusions and wrong rows.

Three claims, decided by which rows have a reference:

* a sample Ondine drew: a certificate with a range;
* every delivered row: a census certificate, exact;
* rows the caller picked: a check — agreement on those rows, no range.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from ondine.harness.judge import read_field
from ondine.ruler.certificate import (
    CERTIFICATE,
    CHECK,
    Certificate,
    Confusion,
    Estimate,
    FieldCertificate,
    Group,
    WrongRow,
)
from ondine.ruler.delivered import CertificationError
from ondine.ruler.intervals import hypergeom_bounds

if TYPE_CHECKING:
    from ondine.harness.contract import OutputContract, OutputField
    from ondine.ruler.delivered import DeliveredValues
    from ondine.ruler.sample import Sample

#: Below this many resolved sampled rows a breakdown line is marked "few rows".
FEW_ROWS = 20
#: Confusions listed per field.
TOP_CONFUSIONS = 5

_UNRESOLVED = object()


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
    gold: Any, fields: Sequence[str], frame: pd.DataFrame
) -> dict[str, dict[int, Any]]:
    """Each field's reference value per row, from what the caller passed.

    ``gold`` is a column name of the result frame; values aligned with the
    input rows; a mapping (or a pandas Series) row → value; or, for several
    fields, a mapping field → any of those. Missing cells are left out
    (unresolved).
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
            if len(values) != len(frame):
                raise CertificationError(
                    f"gold for {name!r} has {len(values)} values; the run has {len(frame)} rows"
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
    gold: Mapping[str, Mapping[int, Any]],
    *,
    sample: Sample | None = None,
    tolerance: Mapping[str, float] | None = None,
    read_delivered: bool = False,
) -> Certificate:
    """Compare delivered values with reference values and state the claim.

    Args:
        delivered: The run's delivered values.
        contract: The rules values are read and compared by.
        fields: The fields to certify (all certifiable).
        not_certified: Output fields left out, with the reason.
        gold: Field -> row -> reference value, missing rows left out.
        sample: The run's Ondine-drawn sample, if the references cover it.
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
        _check_sample(sample, delivered, fields)
    declarations = {d.name: d for d in contract.fields}
    problems: list[str] = []
    outcomes: dict[str, dict[int, tuple[str, Any, tuple[Any, ...]]]] = {}
    for name in fields:
        population = delivered.population(name)
        rows = _compared_rows(population, gold[name], sample)
        outcomes[name] = {}
        for row in rows:
            ref = gold[name].get(row, _UNRESOLVED)
            value = delivered.values[name][row]
            if read_delivered:
                value = _read_delivered(contract, name, value)
            outcomes[name][row] = _compare(
                contract, name, value, ref, tolerance.get(name, 0.0), row, problems
            )
    if problems:
        shown = "\n  ".join(problems[:50])
        more = f"\n  … {len(problems) - 50} more" if len(problems) > 50 else ""
        raise CertificationError(
            f"{len(problems)} reference value(s) break the field's rules; fix them and certify again:\n  {shown}{more}"
        )

    claim = _claim(delivered, fields, gold, sample)
    results = tuple(
        _field_certificate(
            delivered,
            declarations[name],
            outcomes[name],
            claim,
            tolerance.get(name, 0.0),
        )
        for name in fields
    )
    return Certificate(
        claim=CERTIFICATE if claim != CHECK else CHECK,
        claim_text=_claim_text(claim, results),
        run_id=delivered.run_id,
        fingerprint=delivered.fingerprint,
        contract_fingerprint=contract.fingerprint,
        basis=_basis(claim, sample, results),
        seed=sample.seed if sample is not None else None,
        fields=results,
        not_certified=tuple(not_certified),
        row_accuracy=_row_accuracy(delivered, fields, outcomes, claim)
        if len(fields) > 1
        else None,
    )


def _check_sample(
    sample: Sample, delivered: DeliveredValues, fields: Sequence[str]
) -> None:
    if sample.run_id != delivered.run_id:
        raise CertificationError(
            f"the sample was drawn from run {sample.run_id[:8]}…, not this run "
            f"({delivered.run_id[:8]}…); draw a sample from this result"
        )
    if sample.fingerprint != delivered.fingerprint:
        raise CertificationError(
            "the delivered values changed since the sample was drawn; "
            "the sample no longer represents the column"
        )
    outside = [name for name in fields if name not in sample.fields]
    if outside:
        raise CertificationError(
            f"the sample was not drawn for {', '.join(map(repr, outside))}"
        )


def _compared_rows(
    population: Sequence[int], gold: Mapping[int, Any], sample: Sample | None
) -> list[int]:
    if sample is not None:
        drawn = set(sample.rows)
        return [row for row in population if row in drawn]
    return [row for row in population if row in gold]


def _claim(delivered, fields, gold, sample) -> str:
    """ "sample", "census" or CHECK, for the run as a whole."""
    if sample is not None:
        return "census" if sample.census else "sample"
    full = all(
        all(row in gold[name] for row in delivered.population(name)) for name in fields
    )
    return "census" if full else CHECK


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


def _compare(
    contract, name, value, ref, tol, row, problems
) -> tuple[str, Any, tuple[Any, ...]]:
    """("right" | "wrong" | "unresolved", delivered, acceptable references)."""
    if ref is _UNRESOLVED:
        return ("unresolved", value, ())
    raws = list(ref) if isinstance(ref, list | tuple | set | frozenset) else [ref]
    accepted: list[Any] = []
    for raw in raws:
        if _missing(raw):
            continue
        verdict = read_field(contract, name, raw)
        if verdict.outcome == "failed" or verdict.format_fixed:
            why = (
                "; ".join(r.text for r in verdict.reasons)
                or "needs a format fix; give the exact value"
            )
            problems.append(f"row {row}, {name}: {raw!r} — {why}")
            continue
        accepted.append(verdict.value if verdict.outcome == "value" else None)
    if not accepted:
        return ("unresolved", value, ())
    right = any(_equal(value, a, tol) for a in accepted)
    return ("right" if right else "wrong", value, tuple(accepted))


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
    delivered, declaration, outcomes, claim, tol
) -> FieldCertificate:
    name = declaration.name
    population = delivered.population(name)
    check = claim == CHECK
    states = [outcomes[row][0] for row in sorted(outcomes)]
    estimate = _estimate(len(population), states, check)

    def groups(key) -> tuple[Group, ...]:
        sizes = Counter(key(row) for row in population)
        compared: dict[Any, list[str]] = {}
        for row in sorted(outcomes):
            compared.setdefault(key(row), []).append(outcomes[row][0])
        lines = []
        for value, size in sizes.most_common():
            est = _estimate(size, compared.get(value, []), check)
            few = claim == "sample" and est.resolved < FEW_ROWS
            lines.append(Group(str(value), est, few))
        return tuple(lines)

    wrong = [(row, o) for row, o in sorted(outcomes.items()) if o[0] == "wrong"]
    confusions = Counter(
        (repr(o[1]) if isinstance(o[1], _Unreadable) else o[1], o[2][0])
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
            WrongRow(row, o[1] if not isinstance(o[1], _Unreadable) else o[1].raw, o[2])
            for row, o in wrong
        ),
        not_delivered=_not_delivered(delivered, name, population),
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


def _row_accuracy(delivered, fields, outcomes, claim) -> Estimate:
    population = sorted(
        set.intersection(*(set(delivered.population(n)) for n in fields))
    )
    compared = [row for row in population if all(row in outcomes[n] for n in fields)]
    states = []
    for row in compared:
        per = [outcomes[n][row][0] for n in fields]
        states.append(
            "wrong"
            if "wrong" in per
            else "unresolved"
            if "unresolved" in per
            else "right"
        )
    return _estimate(len(population), states, claim == CHECK)


def _claim_text(claim: str, results) -> str:
    if claim == "sample":
        return "accuracy against your labels, on a sample Ondine drew"
    if claim == "census":
        return "accuracy against your labels, on every delivered row"
    rows = max((r.estimate.compared for r in results), default=0)
    return f"agreement with your labels on the {rows} rows you labelled — not a claim about the column"


def _basis(claim: str, sample, results) -> str:
    if sample is not None:
        return f"sample of {len(sample.rows)} of {sample.population_size} delivered rows ({sample.rule})"
    if claim == "census":
        return "every delivered row has a label"
    return "rows you labelled, not drawn by Ondine"
