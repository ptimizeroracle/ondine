"""The rows drawn to certify a run.

One sample per run, drawn by Ondine from the rows that delivered a value for
at least one certifiable field. Each field's sample is the drawn rows inside
that field's population: a simple random sample of it, given its size.

The seed is derived from the run (its id and the fingerprint of what it
delivered), never chosen by the caller: drawing again on the same result
returns the same rows, so a sample cannot be redrawn until the number looks
good.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ondine.ruler.delivered import CertificationError
from ondine.ruler.intervals import hypergeom_bounds

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from ondine.ruler.delivered import DeliveredValues

DEFAULT_SIZE = 200


@dataclass(frozen=True, eq=False)
class Sample:
    """Rows Ondine drew from one run, bound to what that run delivered.

    Attributes:
        run_id: The run the rows were drawn from.
        fingerprint: The delivered values they were drawn from. Certifying a
            result whose values differ is refused.
        seed: Derived from the run; recorded, never chosen.
        fields: The certifiable fields the population was built from.
        population_size: Rows that delivered at least one of ``fields``.
        rows: The drawn rows, in input order.
        rule: How the size was set, in words.
        inputs: The drawn rows' input columns, indexed by row: what a person
            needs to label them. Never the model's answer.
    """

    run_id: str
    fingerprint: str
    seed: int
    fields: tuple[str, ...]
    population_size: int
    rows: tuple[int, ...]
    rule: str
    inputs: pd.DataFrame = field(repr=False)

    @property
    def census(self) -> bool:
        """True when every row of the population was drawn: no sampling range."""
        return len(self.rows) == self.population_size


def draw(
    delivered: DeliveredValues,
    fields: Sequence[str],
    inputs: pd.DataFrame,
    *,
    size: int | None = None,
    precision: float | None = None,
) -> Sample:
    """Draw the run's sample.

    Args:
        delivered: The run's delivered values.
        fields: The certifiable fields.
        inputs: The run's input columns, one row per input row, in order.
        size: Rows to draw (default 200). Exclusive with ``precision``.
        precision: Half-width the range should not exceed at worst (50%
            accuracy), as a share, e.g. 0.03; sets the size before drawing.

    Raises:
        CertificationError: Both or a non-positive ``size``/``precision``, or
            no delivered row to draw from.
    """
    if size is not None and precision is not None:
        raise CertificationError("give a size or a precision, not both")
    union = sorted({row for name in fields for row in delivered.population(name)})
    if not union:
        raise CertificationError("no row delivered a value to certify")
    if precision is not None:
        if not 0 < precision < 0.5:
            raise CertificationError("precision must be between 0 and 0.5, e.g. 0.03")
        size = size_for_precision(precision, len(union))
        rule = f"precision ±{precision:.1%} at worst ({size} rows)"
    else:
        size = DEFAULT_SIZE if size is None else size
        if size < 1:
            raise CertificationError("size must be at least 1")
        rule = f"{size} rows"
    seed = int(
        hashlib.sha256(
            f"{delivered.run_id}:{delivered.fingerprint}".encode()
        ).hexdigest()[:16],
        16,
    )
    if size >= len(union):
        rows = tuple(union)
        rule = f"census: all {len(union)} delivered rows"
    else:
        picked = np.random.default_rng(seed).choice(
            len(union), size=size, replace=False
        )
        rows = tuple(union[i] for i in sorted(picked))
    return Sample(
        run_id=delivered.run_id,
        fingerprint=delivered.fingerprint,
        seed=seed,
        fields=tuple(fields),
        population_size=len(union),
        rows=rows,
        rule=rule,
        inputs=inputs.iloc[list(rows)].copy(),
    )


def size_for_precision(precision: float, population: int) -> int:
    """Smallest sample whose exact 95% range is at most ±``precision`` at 50%."""
    low, high = 1, population
    while low < high:
        n = (low + high) // 2
        lo, hi = hypergeom_bounds([n // 2], n, population)
        if (hi[0] - lo[0]) / population / 2 <= precision:
            high = n
        else:
            low = n + 1
    return low
