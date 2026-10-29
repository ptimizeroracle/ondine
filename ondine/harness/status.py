"""The six row statuses a client can see, plus the one only the ledger sees.

Every input row ends a run with exactly one status. The vocabulary lives in its
own module because three parties name it and none of them should own it: the
row ledger decides a row's status, delivery writes it into the ``status``
column, and the run report counts it. Deciding *which* status a row gets is not
here — that rule lives in ``ondine.harness.ledger`` and nowhere else.

The values are the lowercase words the client reads in the dataframe, so the
enum's value is the column's content with no translation table in between.
"""

from __future__ import annotations

from enum import Enum


class RowStatus(str, Enum):
    """What happened to one input row, in the client's words.

    ``str`` mixin rather than ``StrEnum`` so the module stays valid on
    Python 3.10; ``RowStatus.ACCEPTED == "accepted"`` holds either way.
    """

    #: Passed every declared check on the first judged answer.
    ACCEPTED = "accepted"
    #: Passed every declared check, but only after an extra step (a format
    #: fix, or a later attempt after an earlier answer failed a check).
    REPAIRED = "repaired"
    #: An answer existed but failed a check, and nothing fixed it. Delivered
    #: empty.
    REJECTED = "rejected"
    #: Failed a check, delivered anyway because the client opted in to
    #: "deliver flagged" for the run.
    UNVERIFIED = "unverified"
    #: No usable answer: the provider or a stage failed, a required check never
    #: ran, or the run stopped before this row. Delivered empty.
    FAILED = "failed"
    #: No contract was declared; the row holds exactly what Ondine produces
    #: today.
    UNCHECKED = "unchecked"
    #: Internal only: no event has decided this row yet. Never delivered — a
    #: row still pending when the ledger is snapshotted becomes FAILED.
    PENDING = "pending"


#: The statuses a delivered row can carry, in the order a report lists them.
FINAL_STATUSES: tuple[RowStatus, ...] = (
    RowStatus.ACCEPTED,
    RowStatus.REPAIRED,
    RowStatus.REJECTED,
    RowStatus.UNVERIFIED,
    RowStatus.FAILED,
    RowStatus.UNCHECKED,
)
