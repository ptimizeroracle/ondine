"""What a certification claims, on what basis: data first, text second.

A certificate describes one run. It never changes the run's delivered frame
and is not part of its run report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

#: The claim kinds. A certificate states accuracy; a check only agreement on
#: rows the caller picked, with no claim on the column.
CERTIFICATE = "certificate"
CHECK = "check"

METHOD = "exact hypergeometric 95% range; unresolved rows counted wrong for the low end, right for the high end"


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


@dataclass(frozen=True)
class Certificate:
    """A run's certificate or check, bound to the values it measured.

    Attributes:
        claim: :data:`CERTIFICATE` or :data:`CHECK`.
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

    def __getitem__(self, field: str) -> FieldCertificate:
        for item in self.fields:
            if item.field == field:
                return item
        raise KeyError(field)

    def to_dict(self) -> dict[str, Any]:
        """Plain data, for storing or comparing certificates."""
        return asdict(self)

    def render(self, *, values: int = 10, wrong: int = 10) -> str:
        """The certificate as text.

        Args:
            values: Delivered values listed per field, largest first.
            wrong: Wrong rows listed per field.
        """
        title = "Certificate" if self.claim == CERTIFICATE else "Check"
        lines = [f"{title}: {self.claim_text}", f"  Basis:  {self.basis}"]
        for item in self.fields:
            lines += _render_field(item, self.claim, values, wrong)
        if self.row_accuracy is not None:
            lines.append(
                f"  Rows with every field right: {_headline(self.row_accuracy, self.claim)}"
            )
        for name, reason in self.not_certified:
            lines.append(f"  Not certified: {name} ({reason})")
        lines.append(f"  Method: {self.method}")
        lines.append(
            f"  Bound to: run {self.run_id[:8]}…, delivered values {self.fingerprint[:8]}…, "
            f"contract {self.contract_fingerprint[:8]}…"
            + (f", seed {self.seed}" if self.seed is not None else "")
        )
        return "\n".join(lines)


def _render_field(
    item: FieldCertificate, claim: str, values: int, wrong: int
) -> list[str]:
    est = item.estimate
    lines = [
        "",
        f"  {item.field} ({item.kind})",
        f"    {'Accuracy' if claim == CERTIFICATE else 'Agreement'}: {_headline(est, claim)}",
        f"    Compared: {est.compared} of {est.population} delivered rows; "
        f"{est.unresolved} unresolved; {item.rule}",
    ]
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
