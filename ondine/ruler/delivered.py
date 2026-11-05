"""What a finished run delivered, read once, for sampling and certifying.

The one place that knows how to read a result: the delivered frame for the
values, the row ledger for each row's status and model. The sample and the
Ruler read this view and never the result itself.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ondine.core.models import ExecutionResult
    from ondine.harness.status import RowStatus


class CertificationError(ValueError):
    """A sample or certificate cannot be made; the message lists every cause."""


@dataclass(frozen=True)
class DeliveredValues:
    """The delivered value of each field, per input row, with its provenance.

    Attributes:
        run_id: The run's execution id, as text.
        row_count: Input rows in the run.
        statuses: Each row's final status, by row.
        models: The model that produced each row's delivered value, or None.
        values: Field name -> each row's delivered value, None when empty.
        fingerprint: Hash of every field's delivered values; equal iff the
            delivered values are.
    """

    run_id: str
    row_count: int
    statuses: tuple[RowStatus, ...]
    models: tuple[str | None, ...]
    values: dict[str, tuple[Any, ...]]
    fingerprint: str

    def population(self, field: str) -> tuple[int, ...]:
        """Rows that delivered a value for ``field``, in input order.

        Decided by the value, not the status: under ``keep_valid_fields`` a
        rejected row still delivers the fields that passed, and those values
        reached the client.
        """
        cells = self.values[field]
        return tuple(row for row in range(self.row_count) if cells[row] is not None)

    @classmethod
    def read(cls, result: ExecutionResult, fields: Sequence[str]) -> DeliveredValues:
        """Read ``fields`` from a finished result.

        Raises:
            CertificationError: The result has no row ledger (built by hand),
                is one chunk of a stream, or lacks one of ``fields``.
        """
        if getattr(result, "awaiting_merge", False):
            raise CertificationError(
                "this is one chunk of a stream; certify the result of execute() "
                "(streams are not certified yet)"
            )
        snapshot = result.row_ledger
        if snapshot is None:
            raise CertificationError(
                "the result has no row ledger; certify a pipeline's result"
            )
        frame = result.to_pandas()
        missing = [name for name in fields if name not in frame.columns]
        if missing:
            raise CertificationError(
                f"the result has no column {', '.join(map(repr, missing))}"
            )
        outcomes = sorted(snapshot.rows, key=lambda outcome: outcome.row)
        if len(frame) != len(outcomes):
            raise CertificationError(
                f"the result frame has {len(frame)} rows but the run has {len(outcomes)}; "
                "certify the result as execute() returned it"
            )
        values = {
            name: tuple(_cell(v) for v in frame[name].tolist()) for name in fields
        }
        digest = hashlib.sha256(
            json.dumps(sorted(values.items()), default=str, sort_keys=True).encode()
        ).hexdigest()
        return cls(
            run_id=str(result.execution_id),
            row_count=len(outcomes),
            statuses=tuple(outcome.status for outcome in outcomes),
            models=tuple(outcome.delivered_by for outcome in outcomes),
            values=values,
            fingerprint=digest,
        )


def _cell(value: Any) -> Any:
    """A frame cell as a plain value: None for every kind of empty."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    # numpy scalars from the frame, as plain Python values.
    return value.item() if isinstance(value, np.generic) else value
