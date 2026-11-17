"""What asking a row again would cost, priced before anything is sent.

One rule, shared by every step that asks a row it already has answers for
(re-ask, escalation, the Ruler's model audit): the row's tokens per attempt in
the run, priced at the asking model's rates, every token at both its input
and its output price, so the estimate errs high; then :data:`COST_MARGIN`.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ondine.harness.ledger import RowOutcome

#: Margin on an estimated call: a later prompt (with a correction) is longer
#: than the row's first one.
COST_MARGIN = Decimal("1.25")


def call_cost(outcome: RowOutcome, client: Any | None = None) -> Decimal:
    """One more call for ``outcome``'s row, before the margin.

    Args:
        outcome: The row, with the answers the run received for it.
        client: The client to be asked; None for the run's own model, whose
            price per attempt the row already shows.
    """
    attempts = len(outcome.attempts) or 1
    if client is None:
        return outcome.cost / attempts
    tokens = sum(record.tokens for record in outcome.attempts) // attempts
    return Decimal(str(client.calculate_cost(tokens, tokens)))
