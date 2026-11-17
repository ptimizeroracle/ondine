"""What a certification claims, on what basis: data first, text second.

A certificate describes one run. It never changes the run's delivered frame
and is not part of its run report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from dataclasses import field as _field
from typing import Any

#: The claim kinds. A certificate states accuracy; an audit agreement with a
#: reference model, never accuracy; a check only agreement on rows the caller
#: picked, with no claim on the column.
CERTIFICATE = "certificate"
AUDIT = "audit"
CHECK = "check"

#: Printed on every audit. Agreement with a stronger model overstated
#: accuracy by 4 to 12.5 points on our golden sets (design, section 1).
AUDIT_WARNING = (
    "an audit, not accuracy: models share mistakes, across vendors too, so "
    "agreement can overstate accuracy (by 4 to 12.5 points on our test sets)"
)

METHOD = "exact hypergeometric 95% range; unresolved rows counted wrong for the low end, right for the high end"

#: Below this many resolved sampled rows a breakdown line is marked "few rows";
#: so is an agreement measured on fewer rows answered by both labellers.
FEW_ROWS = 20

#: Labeller agreement below either of these prints the noisy-labels warning.
MIN_AGREEMENT = 0.8
MIN_KAPPA = 0.6


@dataclass(frozen=True)
class Estimate:
    """Right values among the rows compared, with an exact range when there is one.

    Attributes:
        population: Rows the estimate is about (N).
        compared: Rows compared, resolved or not (n).
        right: Compared rows whose value equals the reference.
        unresolved: Compared rows with no usable reference.
        low: Lower end of the 95% range on the population's accuracy, or None
            for a check.
        high: Upper end, or None for a check.
    """

    population: int
    compared: int
    right: int
    unresolved: int
    low: float | None
    high: float | None

    @property
    def resolved(self) -> int:
        return self.compared - self.unresolved

    @property
    def wrong(self) -> int:
        return self.resolved - self.right

    @property
    def accuracy(self) -> float | None:
        """Right over resolved compared rows; None when none resolved."""
        return self.right / self.resolved if self.resolved else None

    @property
    def exact(self) -> bool:
        """True for a census with every row resolved: the accuracy is not estimated."""
        return self.compared == self.population and self.unresolved == 0


@dataclass(frozen=True)
class Group:
    """One breakdown line: a status, a model or a delivered value.

    ``few_rows`` marks a group with fewer than 20 resolved sampled rows, whose
    range is too wide to read much into.
    """

    name: str
    estimate: Estimate
    few_rows: bool


@dataclass(frozen=True)
class Confusion:
    """A delivered value the reference names as another, and how often."""

    delivered: Any
    reference: Any
    count: int


@dataclass(frozen=True)
class WrongRow:
    """A compared row whose delivered value the reference contradicts."""

    row: int
    delivered: Any
    reference: tuple[Any, ...]


@dataclass(frozen=True)
class LabellerAgreement:
    """How far two people agreed on the rows both labelled, for one field.

    Counted on the field's compared rows that both labellers were given.

    Attributes:
        labellers: 1 or 2; with 1, nothing below is measured.
        doubled: Compared rows labelled twice.
        both_answered: Of those, rows both people gave an answer for (not
            "cannot tell", not empty).
        agree: Of those, rows whose answer sets share a value (numbers within
            the tolerance).
        split: Rows the two judged differently against the delivered value, so
            they are unresolved ("labellers disagree").
        kappa: Cohen's kappa on the rows where each person gave exactly one
            answer, for labels and booleans; None for numbers, or when both
            always used one label (kappa is undefined).
        kappa_rows: Rows kappa was computed on.
    """

    labellers: int
    doubled: int
    both_answered: int
    agree: int
    split: int
    kappa: float | None
    kappa_rows: int


@dataclass(frozen=True)
class FieldCertificate:
    """The claim for one output field, on that field's own population.

    Attributes:
        field: The output column.
        kind: "labels", "boolean", "integer" or "number".
        rule: How two values were compared, in words; part of the claim.
        estimate: The headline.
        by_status: One line per status present in the population.
        by_model: One line per delivering model.
        by_value: One line per delivered value, largest first.
        confusions: The most frequent confusions among compared rows.
        wrong_rows: Every compared row the reference contradicts.
        not_delivered: Rows outside the population, by cause (``rejected``,
            ``failed``, ``empty``), never certified.
        unresolved_causes: Why compared rows have no reference value, by
            cause, when the reference says (an audit, people's labels);
            empty otherwise.
        labellers: People's agreement on this field; None for any other
            reference.
    """

    field: str
    kind: str
    rule: str
    estimate: Estimate
    by_status: tuple[Group, ...]
    by_model: tuple[Group, ...]
    by_value: tuple[Group, ...]
    confusions: tuple[Confusion, ...]
    wrong_rows: tuple[WrongRow, ...]
    not_delivered: dict[str, int]
    unresolved_causes: dict[str, int] = _field(default_factory=dict)
    labellers: LabellerAgreement | None = None


@dataclass(frozen=True)
class Certificate:
    """A run's certificate or check, bound to the values it measured.

    Attributes:
        claim: :data:`CERTIFICATE`, :data:`AUDIT` or :data:`CHECK`.
        claim_text: What the numbers mean, in words.
        run_id: The run certified.
        fingerprint: Hash of the delivered values measured.
        contract_fingerprint: The rules values were read and compared by.
        basis: Which rows were compared and how they were chosen.
        seed: The sample's derived seed, or None without a sample.
        fields: One claim per certified field.
        not_certified: (field, reason) for each output field left out.
        row_accuracy: Rows with every certified field right, when there are
            several fields; None otherwise.
        method: How ranges are computed.
        reference: An audit's :class:`~ondine.ruler.reference.ReferenceAnswers`,
            attached by ``Pipeline.audit`` (pass the certificate back as
            ``previous`` to reuse them); None otherwise.
        sheet: The review sheet's id, when people's labels were the reference.
    """

    claim: str
    claim_text: str
    run_id: str
    fingerprint: str
    contract_fingerprint: str
    basis: str
    seed: int | None
    fields: tuple[FieldCertificate, ...]
    not_certified: tuple[tuple[str, str], ...]
    row_accuracy: Estimate | None
    method: str = METHOD
    reference: Any = None
    sheet: str | None = None

    def __getitem__(self, field: str) -> FieldCertificate:
        for item in self.fields:
            if item.field == field:
                return item
        raise KeyError(field)

    def to_dict(self) -> dict[str, Any]:
        """Plain data, for storing or comparing certificates.

        What only people's labels have (``sheet``, ``labellers``) is left out
        when a certificate has none, so other certificates' data is as before.
        """
        data = asdict(self)
        if self.sheet is None:
            del data["sheet"]
        for entry, item in zip(data["fields"], self.fields, strict=True):
            if item.labellers is None:
                del entry["labellers"]
        return data

    def render(self, *, values: int = 10, wrong: int = 10) -> str:
        """The certificate as text.

        Args:
            values: Delivered values listed per field, largest first.
            wrong: Wrong rows listed per field.
        """
        title = {CERTIFICATE: "Certificate", AUDIT: "Audit"}.get(self.claim, "Check")
        lines = [f"{title}: {self.claim_text}", f"  Basis:  {self.basis}"]
        if self.claim == AUDIT:
            lines.append(f"  Warning: {AUDIT_WARNING}")
            if self.reference is not None:
                lines.append(
                    f"  Reference: {self.reference.model}, cost ${self.reference.cost:.4f}"
                )
        for item in self.fields:
            lines += _render_field(item, self.claim, values, wrong)
        if self.row_accuracy is not None:
            label = (
                "Rows where every field agrees"
                if self.claim == AUDIT
                else "Rows with every field right"
            )
            lines.append(f"  {label}: {_headline(self.row_accuracy, self.claim)}")
        for name, reason in self.not_certified:
            lines.append(f"  Not certified: {name} ({reason})")
        lines.append(f"  Method: {self.method}")
        lines.append(
            f"  Bound to: run {self.run_id[:8]}…, delivered values {self.fingerprint[:8]}…, "
            f"contract {self.contract_fingerprint[:8]}…"
            + (f", seed {self.seed}" if self.seed is not None else "")
            + (f", sheet {self.sheet}" if self.sheet is not None else "")
        )
        return "\n".join(lines)


def _render_field(
    item: FieldCertificate, claim: str, values: int, wrong: int
) -> list[str]:
    est = item.estimate
    causes = ""
    if item.unresolved_causes:
        causes = (
            " ("
            + ", ".join(f"{n} {cause}" for cause, n in item.unresolved_causes.items())
            + ")"
        )
    lines = [
        "",
        f"  {item.field} ({item.kind})",
        f"    {'Accuracy' if claim == CERTIFICATE else 'Agreement'}: {_headline(est, claim)}",
        f"    Compared: {est.compared} of {est.population} delivered rows; "
        f"{est.unresolved} unresolved{causes}; {item.rule}",
    ]
    if item.labellers is not None:
        lines += _render_agreement(item.labellers)
    if item.not_delivered:
        parts = ", ".join(
            f"{count} {cause}" for cause, count in item.not_delivered.items()
        )
        lines.append(f"    Not delivered, not certified: {parts}")
    for label, groups in (("By status", item.by_status), ("By model", item.by_model)):
        if groups:
            lines.append(
                f"    {label}: " + " · ".join(_group(g, claim) for g in groups)
            )
    if item.by_value:
        shown = item.by_value[:values]
        lines.append("    By value:")
        lines += [f"      {_group(g, claim)}" for g in shown]
        if len(item.by_value) > values:
            lines.append(f"      … {len(item.by_value) - values} more values")
    if item.confusions:
        lines.append(
            "    Confusions: "
            + "; ".join(
                f"{c.delivered} → {c.reference} ({c.count})" for c in item.confusions
            )
        )
    if item.wrong_rows:
        rows = ", ".join(str(w.row) for w in item.wrong_rows[:wrong])
        more = (
            f" … {len(item.wrong_rows) - wrong} more"
            if len(item.wrong_rows) > wrong
            else ""
        )
        lines.append(f"    Wrong rows ({len(item.wrong_rows)}): {rows}{more}")
    return lines


def _render_agreement(agreement: LabellerAgreement) -> list[str]:
    """The labeller line, and a warning line when the labels look noisy."""
    if agreement.labellers < 2:
        return ["    Labellers: 1; agreement not measured (one labeller)"]
    twice, both = agreement.doubled, agreement.both_answered
    text = f"    Labellers: 2 (one sheet each); {twice} rows labelled twice"
    noisy = False
    if both == 0:
        text += ", no row answered by both"
    else:
        if both != twice:
            text += f", {both} answered by both"
        share = agreement.agree / both
        few = ", few rows" if both < FEW_ROWS else ""
        text += f": agree on {agreement.agree} ({share:.1%}{few})"
        if agreement.kappa is not None:
            rows = agreement.kappa_rows
            noun = "row" if rows == 1 else "rows"
            text += f", kappa {agreement.kappa:.2f} on {rows} single-answer {noun}"
        elif agreement.kappa_rows:
            text += ", kappa not defined (one label used)"
        noisy = share < MIN_AGREEMENT or (
            agreement.kappa is not None and agreement.kappa < MIN_KAPPA
        )
    text += f"; {agreement.split} split on the delivered value (unresolved)"
    lines = [text]
    if noisy:
        lines.append(
            "    Warning: labellers often disagree; the labels are noisy, so "
            "accuracy against them is loose (guide: Labels from people)"
        )
    return lines


def _headline(est: Estimate, claim: str) -> str:
    if est.accuracy is None:
        return "no resolved row"
    text = f"{est.accuracy:.1%}"
    if claim == CHECK or est.low is None or est.high is None:
        return f"{text} ({est.right} of {est.resolved})"
    if est.exact:
        return f"{text} (exact: every delivered row compared)"
    return f"{text} (95% range {est.low:.1%}–{est.high:.1%})"


def _group(group: Group, claim: str) -> str:
    note = ", few rows" if group.few_rows else ""
    est = group.estimate
    if est.compared == 0:
        return f"{group.name}: none compared ({est.population} delivered)"
    if est.accuracy is None:
        return f"{group.name}: {est.compared} compared, none resolved ({est.population} delivered)"
    if claim == CHECK or est.low is None or est.high is None or est.exact:
        return f"{group.name} {est.accuracy:.1%} ({est.right}/{est.resolved}{note})"
    return f"{group.name} {est.accuracy:.1%} [{est.low:.1%}–{est.high:.1%}] (n={est.compared}{note})"
