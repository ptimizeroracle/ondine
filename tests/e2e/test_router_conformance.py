"""End-to-end conformance for the LiteLLM Router path: failover keeps rows.

Ondine delegates load balancing and failover wholesale to LiteLLM's Router, so
the existing conformance suite — which swaps in a whole fake client — never
touches any of it. These tests keep the real `UnifiedLiteLLMClient` and its
Router and instead fake the *deployments underneath it* (see `router_ledger`),
which is the only way to assert what failover is supposed to guarantee:

- a row whose first deployment is down is still answered, by another one;
- the answer a row ends up with is still *its own* answer, not a neighbour's,
  even though completion order and deployment now both vary;
- a deployment that refuses one row loses that row only, not the rows around it;
- a whole-fleet outage fails loudly instead of returning empty success.

The Router's own retry/cooldown timing is nondeterministic, so every assertion
here is on the *outcome* — which rows each deployment answered — never on how
many times the Router happened to probe a dead one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ondine import PipelineBuilder
from tests.e2e.conformance_harness import (
    conformance_frame,
    expected_answers,
    token_for,
)
from tests.e2e.router_ledger import (
    AlwaysDownDeployment,
    LedgerDeployment,
    deployment_entry,
    registered_deployments,
)

if TYPE_CHECKING:
    import pandas as pd

PROMPT = "Answer for {marker}"


def build_router_pipeline(frame: pd.DataFrame, model_list: list[dict], **options):
    """A real pipeline whose Router routes over the given fake deployments."""
    builder = (
        PipelineBuilder.create()
        .from_dataframe(frame, input_columns=["marker"], output_columns=["answer"])
        .with_prompt(PROMPT)
        .with_router(
            model_list=model_list,
            routing_strategy="simple-shuffle",
            num_retries=options.get("num_retries", 2),
        )
    )
    if "batch_size" in options:
        builder = builder.with_batch_size(options["batch_size"])
    if "concurrency" in options:
        builder = builder.with_concurrency(options["concurrency"])
    if "error_policy" in options:
        builder = builder.with_error_policy(options["error_policy"])
    return builder.build()


def test_failover_serves_every_row_from_the_healthy_deployment():
    """A dead primary must cost throughput, not rows.

    One deployment is down for the whole run and one is healthy, sharing a
    group. Every row must come back with its own answer, served by the healthy
    deployment — and the down one must actually have been tried, or the test
    proves nothing about failover.

    Catches: a row routed to the dead deployment being dropped, filled with a
    placeholder, or given another row's answer instead of failing over.
    """
    rows = 10
    down = AlwaysDownDeployment("primary-down")
    healthy = LedgerDeployment("backup-ok")
    mapping = {"pdown": down, "pok": healthy}
    model_list = [
        deployment_entry("pdown", "primary-down"),
        deployment_entry("pok", "backup-ok"),
    ]

    with registered_deployments(mapping):
        result = build_router_pipeline(
            conformance_frame(rows), model_list, batch_size=1, concurrency=4
        )
        answers = result.execute().to_pandas()["answer"].tolist()

    assert answers == expected_answers(rows)
    assert healthy.answered_tokens == {token_for(i) for i in range(rows)}
    # Each row served exactly once — failover must not double-answer.
    assert all(count == 1 for count in healthy.answered.values())
    # Failover was genuinely exercised, not sidestepped by luck.
    assert down.attempts > 0


@pytest.mark.parametrize(
    ("rows", "batch_size", "concurrency"),
    [
        (12, 1, 1),  # one row per call, in order — the baseline
        (12, 4, 1),  # a failed batch must fail over as a whole, still aligned
        (12, 1, 4),  # concurrent rows finish out of order across deployments
        (12, 4, 4),  # batching and concurrency together, where drift hides
    ],
)
def test_failover_preserves_alignment_across_batching_and_concurrency(
    rows, batch_size, concurrency
):
    """Row N keeps row N's answer even when N's batch fails over mid-run.

    When a batched call lands on the dead deployment the whole batch fails and
    is retried elsewhere; the danger is the re-issued batch coming back matched
    to the wrong rows. Only per-row values can see that.
    """
    down = AlwaysDownDeployment("down")
    healthy = LedgerDeployment("ok")
    model_list = [
        deployment_entry("dn", "down"),
        deployment_entry("ok", "ok"),
    ]

    with registered_deployments({"dn": down, "ok": healthy}):
        result = build_router_pipeline(
            conformance_frame(rows),
            model_list,
            batch_size=batch_size,
            concurrency=concurrency,
        )
        answers = result.execute().to_pandas()["answer"].tolist()

    assert answers == expected_answers(rows)
    assert healthy.answered_tokens == {token_for(i) for i in range(rows)}


def test_load_balancing_across_healthy_deployments_neither_drops_nor_duplicates():
    """Two live deployments must partition the rows, not lose or share them.

    With both deployments healthy the Router spreads rows across them however
    it likes. What must hold regardless of the split: every row is answered
    exactly once, somewhere. A dropped row is a missing token; a row sent twice
    is a token answered by both. Which deployment served a given row is not
    asserted — that is load balancing's business, and it is nondeterministic.
    """
    rows = 20
    a = LedgerDeployment("deploy-a")
    b = LedgerDeployment("deploy-b")
    model_list = [
        deployment_entry("da", "deploy-a"),
        deployment_entry("db", "deploy-b"),
    ]

    with registered_deployments({"da": a, "db": b}):
        result = build_router_pipeline(
            conformance_frame(rows), model_list, batch_size=1, concurrency=8
        )
        answers = result.execute().to_pandas()["answer"].tolist()

    assert answers == expected_answers(rows)
    combined = a.answered + b.answered
    assert set(combined) == {token_for(i) for i in range(rows)}  # nothing lost
    assert all(count == 1 for count in combined.values())  # nothing duplicated


def test_total_outage_fails_loudly_instead_of_reporting_empty_success():
    """When the whole fleet is down, the run must raise, not return success.

    Every deployment is unreachable, so no row can be answered. A pipeline that
    swallowed this would hand back a frame full of placeholders with
    ``success=True``; the guard against a zero-output run must fire instead.
    """
    down_a = AlwaysDownDeployment("down-a")
    down_b = AlwaysDownDeployment("down-b")
    model_list = [
        deployment_entry("dna", "down-a"),
        deployment_entry("dnb", "down-b"),
    ]

    with registered_deployments({"dna": down_a, "dnb": down_b}):
        pipeline = build_router_pipeline(
            conformance_frame(8), model_list, batch_size=1, concurrency=4
        )
        with pytest.raises(Exception, match="0 valid outputs|no usable output"):
            pipeline.execute()


def test_a_failing_deployment_loses_only_its_own_rows():
    """A poison row is skipped alone; its neighbours are answered correctly.

    The single deployment refuses two specific rows and answers the rest. Under
    the default skip policy the refused rows must come back marked skipped and
    recorded in ``result.errors`` — and, crucially, *only* those rows. A
    failure that also corrupted or dropped an adjacent row would be far worse
    than the skip itself, and a row-count check would miss it entirely.
    """
    rows = 8
    poisoned = {token_for(3), token_for(5)}
    deployment = LedgerDeployment("only", fail_tokens=poisoned)
    model_list = [deployment_entry("only", "only")]

    with registered_deployments({"only": deployment}):
        result = build_router_pipeline(
            conformance_frame(rows),
            model_list,
            batch_size=1,
            concurrency=4,
            num_retries=0,  # a single deployment: retrying the poison cannot help
            error_policy="skip",
        ).execute()

    answers = result.to_pandas()["answer"].tolist()
    expected = expected_answers(rows)

    for index in range(rows):
        if token_for(index) in poisoned:
            assert answers[index] == "[SKIPPED]"
        else:
            assert answers[index] == expected[index]

    assert deployment.answered_tokens == {
        token_for(i) for i in range(rows) if token_for(i) not in poisoned
    }
    assert result.errors is not None
    assert len(result.errors) == len(poisoned)


@pytest.mark.xfail(
    reason="#254: a run that skipped rows still reports success=True. The loss "
    "is recorded in result.errors, but success does not reflect it.",
    strict=True,
)
def test_partial_loss_should_not_report_success():
    """A run that dropped rows should not call itself a success (#254).

    This is the same run as the test above: two rows are lost and recorded, the
    rest succeed. The rows land in ``result.errors``, but ``result.success``
    stays True — so a caller that trusts ``success`` ships a dataset with holes
    in it. Pinned as xfail until the success contract accounts for skipped rows.
    """
    poisoned = {token_for(3), token_for(5)}
    deployment = LedgerDeployment("only", fail_tokens=poisoned)
    model_list = [deployment_entry("only", "only")]

    with registered_deployments({"only": deployment}):
        result = build_router_pipeline(
            conformance_frame(8),
            model_list,
            batch_size=1,
            concurrency=4,
            num_retries=0,
            error_policy="skip",
        ).execute()

    assert result.success is False
