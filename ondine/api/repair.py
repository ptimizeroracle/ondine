"""Repair passes: ask again, after the run's own pass, rows whose answer failed.

A run's repairs are a plan: an ordered tuple of :class:`RepairStep`. Each step
runs up to ``rounds`` rounds, and each round asks only the rows still failing
a check, each with its correction (``ondine.harness.reask``) appended to the
prompt it was first asked. A round is a sub-pipeline whose ledger merges into
the run's as later attempts stamped with the step's name, so the ledger's
usual rule decides each row (a passing answer makes it ``repaired``) and
delivery names the step that made it.

The plan is re-ask on the run's own model, then escalation: the same pass,
asking a stronger model the rows re-ask could not fix.

Designs: ``.claude/designs/re-ask.md``, ``.claude/designs/escalation.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd

from ondine.adapters.containers.result_container import ResultContainerImpl
from ondine.harness.reask import (
    REASK_COLUMN,
    correction,
    failing_rows,
    instruction,
    last_failure,
)
from ondine.utils import get_logger

if TYPE_CHECKING:
    from ondine.core.models import ExecutionResult
    from ondine.core.specifications import LLMSpec
    from ondine.harness.contract import OutputContract
    from ondine.harness.ledger import RowLedger, RowOutcome

logger = get_logger(__name__)

#: Margin on a repair's estimated cost: the correction makes its prompt longer
#: than the row's first one.
COST_MARGIN = Decimal("1.25")


@dataclass(frozen=True)
class RepairStep:
    """One kind of repair, run for up to ``rounds`` rounds.

    Attributes:
        name: Stamped on every answer the step adds (``AttemptRecord.step``)
            and shown as the row's ``repair_method``.
        rounds: Most rounds; each asks only the rows still failing.
        client: The client this step asks; None for the run's own.
        llm: The spec that client was built from, when there is one; the
            step's passes run under it, so its provider options apply.
    """

    name: str
    rounds: int
    client: Any | None = None
    llm: LLMSpec | None = None


#: Runs one round: the rows to ask (input columns plus the correction) and the
#: step asking. Returns the round's result, its ledger not yet merged.
RunPass = Callable[[pd.DataFrame, RepairStep], "ExecutionResult"]


def repair_plan(
    contract: OutputContract, escalation: RepairStep | None = None
) -> tuple[RepairStep, ...]:
    """The repairs a run asks for, in the order they run.

    Re-ask, when ``contract``'s policy sets it, then ``escalation``. Nothing
    without a checked contract: no check, nothing to repair.
    """
    if not contract.checked:
        return ()
    steps = []
    if contract.policy.re_ask:
        steps.append(RepairStep("re-ask", contract.policy.re_ask))
    if escalation is not None:
        steps.append(escalation)
    return tuple(steps)


def run_repairs(
    plan: Sequence[RepairStep],
    *,
    result: ExecutionResult,
    inputs: Sequence[dict[str, Any]],
    ledger: RowLedger,
    contract: OutputContract,
    output_columns: Sequence[str],
    max_budget: Decimal | float | None,
    run_pass: RunPass,
) -> None:
    """Run ``plan`` over the rows of ``ledger`` whose answer failed a check.

    Values each round delivers are written into ``result``'s rows, and its
    cost and tokens added to ``result``'s, so delivery and the budget see
    them.

    Budget: a round asks only the leading rows (in input order) the run's
    remaining budget can pay for, each estimated with :data:`COST_MARGIN`:
    on the run's model, from the row's own earlier answers; on another
    model, the row's tokens priced at that model's rates, every token at
    both its input and output price, so the estimate errs high. The round
    itself runs uncapped, so it never stops halfway with answers paid for
    and lost.

    Args:
        inputs: The run's input rows, by input position.
        max_budget: The run's cap, or None.
        run_pass: Runs one round (see :data:`RunPass`).
    """
    for step in plan:
        for round_number in range(1, step.rounds + 1):
            failing = failing_rows(ledger.snapshot())
            if max_budget is not None:
                remaining = Decimal(str(max_budget)) - Decimal(
                    str(result.costs.total_cost)
                )
                failing = _affordable(failing, remaining, step)
            if not failing:
                break
            logger.info(
                f"{step.name} round {round_number}/{step.rounds}: "
                f"{len(failing)} row(s) failed a check"
            )
            _run_round(
                step,
                failing,
                result,
                inputs,
                ledger,
                contract,
                output_columns,
                run_pass,
            )


def _run_round(
    step: RepairStep,
    failing: Sequence[RowOutcome],
    result: ExecutionResult,
    inputs: Sequence[dict[str, Any]],
    ledger: RowLedger,
    contract: OutputContract,
    output_columns: Sequence[str],
    run_pass: RunPass,
) -> None:
    rows = [outcome.row for outcome in failing]
    asked = []
    for outcome in failing:
        row = dict(inputs[outcome.row])
        answer, reasons, failed = last_failure(outcome)
        row[REASK_COLUMN] = correction(
            answer, reasons, instruction=instruction(contract, failed)
        )
        asked.append(row)

    pass_result = run_pass(pd.DataFrame(asked), step)
    pass_result.merge_ledger_into(ledger, rows, retry=True, step=step.name)

    result_rows = result.to_list()
    for new_idx, row in enumerate(pass_result.to_list()):
        for column in output_columns:
            result_rows[rows[new_idx]][column] = row.get(column)
    result.data = ResultContainerImpl(
        data=result_rows,
        columns=list(result_rows[0].keys()) if result_rows else [],
    )
    result.costs.total_cost += pass_result.costs.total_cost
    result.costs.total_tokens += pass_result.costs.total_tokens


def _affordable(
    failing: Sequence[RowOutcome], remaining: Decimal, step: RepairStep
) -> list[RowOutcome]:
    """The leading rows of ``failing`` whose estimated repairs fit ``remaining``."""
    chosen = []
    spent = Decimal("0")
    for outcome in failing:
        estimate = _estimate(outcome, step) * COST_MARGIN
        if spent + estimate > remaining:
            break
        spent += estimate
        chosen.append(outcome)
    return chosen


def _estimate(outcome: RowOutcome, step: RepairStep) -> Decimal:
    """One call's cost for ``outcome``'s row on ``step``'s model, before margin."""
    attempts = len(outcome.attempts) or 1
    if step.client is None:
        return outcome.cost / attempts
    tokens = sum(record.tokens for record in outcome.attempts) // attempts
    return Decimal(str(step.client.calculate_cost(tokens, tokens)))
