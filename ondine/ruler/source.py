"""What delivered values are compared with, whoever made it.

:class:`Reference` is the one input of the Ruler's comparison: labels you
wrote, a reference model's answers, later people's labels. It knows nothing
of where the answers came from beyond its kind, so the sources that make
one (:mod:`ondine.ruler.reference` for a model) depend on this module and
the Ruler (:mod:`ondine.ruler.ruler`) does not depend on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The cause of a row with no answer when the source gave none.
NO_ANSWER = "no answer"

#: What a reference is: "labels" the caller wrote, a reference "model"'s
#: answers, or "people"'s labels (no producer yet). It decides the claim.
ReferenceKind = Literal["labels", "model", "people"]


@dataclass(frozen=True)
class Reference:
    """What delivered values are compared with, per field and row.

    Attributes:
        kind: Where the answers come from; decides the claim, not the rows.
        answers: Field -> row -> the raw answer, or a list of acceptable
            answers. A row without one is unresolved. Read by the field's
            contract rules (:func:`ondine.ruler.ruler.read_reference`).
        causes: Field -> row -> why that row has no answer, when the source
            says. Empty for labels, whose unresolved rows give no cause.
        model: The reference model's name, for ``kind="model"``.
    """

    kind: ReferenceKind
    answers: Mapping[str, Mapping[int, Any]] = field(repr=False)
    causes: Mapping[str, Mapping[int, str]] = field(default_factory=dict, repr=False)
    model: str | None = None


@dataclass(frozen=True)
class ReferenceProblem:
    """One reference answer that cannot be used, as data.

    Attributes:
        field: The output field.
        row: The input row.
        raw: The answer as given.
        reasons: Why the field's rules reject it; empty when it was readable
            but only after a format fix, which a reference must not need.
    """

    field: str
    row: int
    raw: Any
    reasons: tuple[str, ...]

    def line(self) -> str:
        """The problem as one line of an error message."""
        why = "; ".join(self.reasons) or "needs a format fix; give the exact value"
        return f"row {self.row}, {self.field}: {self.raw!r} — {why}"
