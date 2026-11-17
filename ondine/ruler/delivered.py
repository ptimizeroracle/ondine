"""What a finished run delivered, read once, for sampling and certifying.

The one place that knows how to read a result: the delivered frame for the
values, the row ledger for each row's status and model. The sample and the
Ruler read this view and never the result itself.

The view is also what outlives the process: :meth:`DeliveredValues.to_dict`
and :meth:`DeliveredValues.from_dict` save and restore it as plain JSON data,
so a run can be certified days later without its ``ExecutionResult`` (which
cannot be pickled and is not kept).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ondine.harness.status import RowStatus

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ondine.core.models import ExecutionResult

#: Saved views carry this version; a view of another version is refused.
VIEW_VERSION = "delivered-v1"


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
        checked: Whether the run's output contract was checked. If not, the
            values are the model's raw text and the Ruler reads them by the
            declarations the caller gives.
        contract_fingerprint: The checked contract the values were judged by,
            or None when ``checked`` is false. The contract itself is not
            saved: the pipeline's own code supplies it, and this binds the two.
    """

    run_id: str
    row_count: int
    statuses: tuple[RowStatus, ...]
    models: tuple[str | None, ...]
    values: dict[str, tuple[Any, ...]]
    fingerprint: str
    checked: bool
    contract_fingerprint: str | None

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
            name: tuple(cell_value(v) for v in frame[name].tolist()) for name in fields
        }
        declared = result.contract
        contract = declared if declared is not None and declared.checked else None
        return cls(
            run_id=str(result.execution_id),
            row_count=len(outcomes),
            statuses=tuple(outcome.status for outcome in outcomes),
            models=tuple(outcome.delivered_by for outcome in outcomes),
            values=values,
            fingerprint=_fingerprint(values),
            checked=contract is not None,
            contract_fingerprint=contract.fingerprint if contract else None,
        )

    def require_fields(self, fields: Sequence[str]) -> None:
        """Refuse a view whose fields are not exactly ``fields``.

        A view's fingerprint covers the fields it holds, and a sample binds
        that fingerprint: a view of other fields than the pipeline certifies
        would refuse the sample drawn on the result.
        """
        missing = [name for name in fields if name not in self.values]
        extra = [name for name in self.values if name not in fields]
        if missing or extra:
            parts = [
                f"{label} {', '.join(map(repr, names))}"
                for label, names in (("missing", missing), ("extra", extra))
                if names
            ]
            raise CertificationError(
                f"the saved view's fields are not the pipeline's certifiable "
                f"fields ({'; '.join(parts)}); save it with Pipeline.save_view"
            )

    def to_dict(self) -> dict[str, Any]:
        """The view as plain data that ``json.dumps`` writes and ``from_dict`` restores.

        Values must be None, booleans, numbers or text: those round-trip
        through JSON exactly. Anything else (a list, a date, a decimal) would
        come back as another value, so it is refused. The data carries a
        digest of itself, so ``from_dict`` can tell it was changed.

        Raises:
            CertificationError: A delivered value is not one of those types,
                or is a number JSON cannot hold (infinity).
        """
        for name, cells in self.values.items():
            for row, value in enumerate(cells):
                if not _plain(value):
                    raise CertificationError(
                        f"cannot save {name!r}: row {row} holds {value!r} "
                        f"({type(value).__name__}); only text, numbers, booleans "
                        "and empty values survive a save exactly"
                    )
        payload: dict[str, Any] = {
            "version": VIEW_VERSION,
            "run_id": self.run_id,
            "row_count": self.row_count,
            "statuses": [status.value for status in self.statuses],
            "models": list(self.models),
            "values": {name: list(cells) for name, cells in self.values.items()},
            "fingerprint": self.fingerprint,
            "checked": self.checked,
            "contract_fingerprint": self.contract_fingerprint,
        }
        return {**payload, "digest": _digest(payload)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DeliveredValues:
        """Restore a view saved by :meth:`to_dict`, checking it was not altered.

        Nothing is coerced: every entry must have exactly the type
        :meth:`to_dict` wrote. The fingerprint is recomputed from the values,
        exactly as :meth:`read` computes it, and the digest from the whole
        data; both must equal the saved ones. They catch a mistake (an edited
        or damaged file), not forgery.

        Raises:
            CertificationError: The data is not a view of this version, is
                malformed (a missing entry, or one of another type), or was
                changed after it was saved.
        """
        if data.get("version") != VIEW_VERSION:
            raise CertificationError(
                f"the saved view is version {data.get('version')!r}; "
                f"this Ondine reads {VIEW_VERSION!r}"
            )
        try:
            view = _parse(data)
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise CertificationError(
                f"the saved view is malformed: {type(error).__name__}: {error}"
            ) from error
        if _fingerprint(view.values) != view.fingerprint:
            raise CertificationError(
                "the saved view's values do not match its fingerprint; "
                "it was changed or damaged after it was saved"
            )
        payload = {key: value for key, value in data.items() if key != "digest"}
        try:
            intact = _digest(payload) == data["digest"]
        except (TypeError, ValueError):  # an entry JSON cannot write was added
            intact = False
        if not intact:
            raise CertificationError(
                "the saved view does not match its digest; "
                "it was changed or damaged after it was saved"
            )
        return view


def _parse(data: Mapping[str, Any]) -> DeliveredValues:
    """The view in ``data``, every entry checked for its exact type."""
    row_count = _exact(data["row_count"], int, "row_count")
    statuses = tuple(RowStatus(_exact(v, str, "status")) for v in data["statuses"])
    models = tuple(_exact_or_none(v, str, "model") for v in data["models"])
    values = {}
    for name, cells in _exact(data["values"], dict, "values").items():
        cells = _exact(cells, list, f"values of {name!r}")
        if not all(_plain(cell) for cell in cells):
            raise TypeError(f"values of {name!r} hold a value that is not plain")
        values[_exact(name, str, "field name")] = tuple(cells)
    _exact(data["digest"], str, "digest")
    sizes = {len(statuses), len(models), *(len(c) for c in values.values())}
    if sizes != {row_count}:
        raise ValueError(
            f"expected {row_count} entries for every field, status and model"
        )
    return DeliveredValues(
        run_id=_exact(data["run_id"], str, "run_id"),
        row_count=row_count,
        statuses=statuses,
        models=models,
        values=values,
        fingerprint=_exact(data["fingerprint"], str, "fingerprint"),
        checked=_exact(data["checked"], bool, "checked"),
        contract_fingerprint=_exact_or_none(
            data["contract_fingerprint"], str, "contract_fingerprint"
        ),
    )


def _exact(value: Any, expected: type, what: str) -> Any:
    """``value`` if it is exactly of type ``expected`` (no coercion, no subclass)."""
    if type(value) is not expected:
        raise TypeError(
            f"{what} must be {expected.__name__}, not {type(value).__name__}"
        )
    return value


def _exact_or_none(value: Any, expected: type, what: str) -> Any:
    return None if value is None else _exact(value, expected, what)


def _digest(payload: Mapping[str, Any]) -> str:
    """Hash of a whole saved view (without its own digest), as JSON writes it."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _fingerprint(values: Mapping[str, tuple[Any, ...]]) -> str:
    """Hash of every field's delivered values, by name; the one hashing."""
    return hashlib.sha256(
        json.dumps(sorted(values.items()), default=str, sort_keys=True).encode()
    ).hexdigest()


def _plain(value: Any) -> bool:
    """True for a value that survives a JSON round trip unchanged."""
    if value is None or type(value) in (bool, int, str):
        return True
    return type(value) is float and math.isfinite(value)


def cell_value(value: Any) -> Any:
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
