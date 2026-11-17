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

#: Why a row people labelled has no answer, or no single one. The first two
#: are known from the sheets; the third only from comparing the two
#: labellers' answers with the delivered value, so the comparison sets it.
NOT_LABELLED = "not labelled"
CANNOT_TELL = "cannot tell"
LABELLERS_DISAGREE = "labellers disagree"

#: What a reference is: "labels" the caller wrote, a reference "model"'s
#: answers, or "people"'s labels (a review sheet). It decides the claim.
ReferenceKind = Literal["labels", "model", "people"]


class _Absent:
    """The answer "this field has no value", where the field has no token for it."""

    def __repr__(self) -> str:
        return "ABSENT"


#: An answer that means "no value" for an optional field, even when the field
#: declares no absent token. Read by :func:`ondine.ruler.ruler.read_reference`.
ABSENT: Any = _Absent()


@dataclass(frozen=True)
class Labellers:
    """Who labelled, when people did: a second answer set for doubled rows.

    ``Reference.answers`` holds the first labeller's answers (labeller A's on
    a doubled row, the one labeller's on any other); the second labeller's
    answers on doubled rows are here.

    Attributes:
        count: 1 or 2.
        doubled: The rows both labellers were given, whether or not they answered.
        second: Field -> row -> the second labeller's raw answer, or a list of
            acceptable answers, on a doubled row where they gave one.
    """

    count: int
    doubled: frozenset[int] = frozenset()
    second: Mapping[str, Mapping[int, Any]] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class Reference:
    """What delivered values are compared with, per field and row.

    Attributes:
        kind: Where the answers come from; decides the claim, not the rows.
        answers: Field -> row -> the raw answer, or a list of acceptable
            answers. A row without one is unresolved. Read by the field's
            contract rules (:func:`ondine.ruler.ruler.read_reference`).
        causes: Field -> row -> why that row has no answer, when the source
            says; the default cause of an unresolved row. A cause only the
            comparison can know (the labellers disagree) is the comparison's
            own. Empty for labels, whose unresolved rows give no cause.
        model: The reference model's name, for ``kind="model"``.
        sheet: The review sheet's id, for ``kind="people"``.
        labellers: Who labelled and the second answer set, for
            ``kind="people"``.
    """

    kind: ReferenceKind
    answers: Mapping[str, Mapping[int, Any]] = field(repr=False)
    causes: Mapping[str, Mapping[int, str]] = field(default_factory=dict, repr=False)
    model: str | None = None
    sheet: str | None = None
    labellers: Labellers | None = None


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
