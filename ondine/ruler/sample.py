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
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ondine.ruler.delivered import CertificationError
from ondine.ruler.intervals import hypergeom_bounds

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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

    def check(self, delivered: DeliveredValues, fields: Sequence[str]) -> None:
        """Refuse a run, values or fields this sample was not drawn for.

        Raises:
            CertificationError: The sample is of another run, the delivered
                values changed since it was drawn, or it was not drawn for
                one of ``fields``.
        """
        if self.run_id != delivered.run_id:
            raise CertificationError(
                f"the sample was drawn from run {self.run_id[:8]}…, not this run "
                f"({delivered.run_id[:8]}…); draw a sample from this result"
            )
        if self.fingerprint != delivered.fingerprint:
            raise CertificationError(
                "the delivered values changed since the sample was drawn; "
                "the sample no longer represents the column"
            )
        outside = [name for name in fields if name not in self.fields]
        if outside:
            raise CertificationError(
                f"the sample was not drawn for {', '.join(map(repr, outside))}"
            )

    def to_dict(self) -> dict[str, Any]:
        """What :meth:`restore` needs to redraw this sample: plain data, no inputs."""
        return {
            "seed": self.seed,
            "fields": list(self.fields),
            "population_size": self.population_size,
            "rows": list(self.rows),
            "rule": self.rule,
        }

    @classmethod
    def restore(
        cls, saved: Mapping[str, Any], delivered: DeliveredValues, fields: Sequence[str]
    ) -> Sample:
        """The sample saved by :meth:`to_dict`, redrawn from ``delivered`` and compared.

        The draw is derived from the run, so redrawing with the saved size
        must give the saved rows. If it does not (another numpy, or other
        values), the saved rows are refused: they are not a random sample of
        what is certified. The restored sample holds no input columns.

        Raises:
            CertificationError: The redraw differs from the saved sample.
        """
        rows = tuple(saved["rows"])
        seed, redrawn, _, population = _pick(delivered, fields, len(rows), None)
        if (seed, redrawn, population) != (
            saved["seed"],
            rows,
            saved["population_size"],
        ) or list(fields) != list(saved["fields"]):
            raise CertificationError(
                "the saved sample cannot be redrawn from this run's values "
                f"(redraw gives {len(redrawn)} rows of {population}, seed {seed}; "
                f"saved: {len(rows)} rows of {saved['population_size']}, "
                f"seed {saved['seed']}); it is not a random sample of them"
            )
        return cls(
            run_id=delivered.run_id,
            fingerprint=delivered.fingerprint,
            seed=seed,
            fields=tuple(fields),
            population_size=population,
            rows=rows,
            rule=str(saved["rule"]),
            inputs=pd.DataFrame(index=list(rows)),
        )


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
    seed, rows, rule, population = _pick(delivered, fields, size, precision)
    return Sample(
        run_id=delivered.run_id,
        fingerprint=delivered.fingerprint,
        seed=seed,
        fields=tuple(fields),
        population_size=population,
        rows=rows,
        rule=rule,
        inputs=inputs.iloc[list(rows)].copy(),
    )


def _pick(
    delivered: DeliveredValues,
    fields: Sequence[str],
    size: int | None,
    precision: float | None,
) -> tuple[int, tuple[int, ...], str, int]:
    """The derived seed, the drawn rows, the rule and the population size.

    The one draw: :func:`draw` makes a sample of it, :meth:`Sample.restore`
    redraws it to compare.
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
    return seed, rows, rule, len(union)


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
