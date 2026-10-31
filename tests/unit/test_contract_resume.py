"""Resume refuses a run whose output contract changed (harness decision D2).

Resume re-judges the answers it restores from the response cache. Under other
rules those rows would silently get other statuses than the first attempt
gave them. So the pipeline records a checked contract in the run's
checkpoint, and a resume under a different one is refused before any model
call, with a message naming what changed.

Without a contract nothing changes: a run that declares none resumes as
before.

Every test drives a real pipeline: the first run stops on a scripted provider
failure and leaves a checkpoint; the second resumes from it with a fresh
client, whose call ledger proves whether a model was called.
"""

from __future__ import annotations

import pytest

from ondine import PipelineBuilder
from tests.e2e.conformance_harness import (
    LedgerClient,
    ProviderError,
    conformance_frame,
    token_for,
)

ROWS = 5
#: The row the first run fails on: rows before it are checkpointed.
FAILING_ROW = 3


def _pipeline(
    client: LedgerClient,
    checkpoint_dir,
    *,
    max_length: int | None = 200,
    on_failure: str | None = None,
):
    """A one-row-per-call pipeline that checkpoints after every row.

    ``max_length=None`` declares no contract at all. ``on_failure`` adds a
    run policy to the contract.
    """
    builder = (
        PipelineBuilder.create()
        .from_dataframe(
            conformance_frame(ROWS),
            input_columns=["marker"],
            output_columns=["answer"],
        )
        .with_prompt("Answer for {marker}")
        .with_custom_llm_client(client)
        .with_max_retries(1)
        .with_batch_size(1)
        .with_concurrency(1)
        .with_error_policy("fail")
        .with_checkpoint_dir(str(checkpoint_dir))
        .with_checkpoint_interval(1)
    )
    if max_length is not None:
        builder = builder.with_output_field("answer", max_length=max_length)
    if on_failure is not None:
        builder = builder.with_contract_policy(on_failure=on_failure)
    return builder.build()


def _interrupted(checkpoint_dir, **contract):
    """Run until the scripted failure; return the session to resume."""
    first = _pipeline(
        LedgerClient(fail_tokens={token_for(FAILING_ROW)}), checkpoint_dir, **contract
    )
    with pytest.raises(ProviderError):
        first.execute()
    return first.session_id


# ---------------------------------------------------------------------------
# A changed contract is refused before any model call, naming the change
# ---------------------------------------------------------------------------


def test_a_changed_field_is_refused_naming_it_before_any_model_call(tmp_path):
    session = _interrupted(tmp_path, max_length=200)
    client = LedgerClient()

    with pytest.raises(ValueError, match="the output contract changed") as refusal:
        _pipeline(client, tmp_path, max_length=100).execute(resume_from=session)

    message = str(refusal.value)
    assert str(session) in message
    assert "field 'answer' changed" in message
    assert "run policy" not in message
    assert client.calls == {}


def test_a_changed_policy_is_refused_naming_it_before_any_model_call(tmp_path):
    session = _interrupted(tmp_path)
    client = LedgerClient()

    with pytest.raises(ValueError, match="run policy changed") as refusal:
        _pipeline(client, tmp_path, on_failure="deliver_flagged").execute(
            resume_from=session
        )

    assert "field 'answer'" not in str(refusal.value)
    assert client.calls == {}


def test_a_contract_dropped_on_resume_is_refused(tmp_path):
    """Resuming without the contract would deliver what it had rejected."""
    session = _interrupted(tmp_path)
    client = LedgerClient()

    with pytest.raises(ValueError, match="this run declares none"):
        _pipeline(client, tmp_path, max_length=None).execute(resume_from=session)

    assert client.calls == {}


def test_a_contract_added_on_resume_is_refused(tmp_path):
    """Nothing proves the rows already answered meet a contract added later."""
    session = _interrupted(tmp_path, max_length=None)
    client = LedgerClient()

    with pytest.raises(ValueError, match="records no contract"):
        _pipeline(client, tmp_path).execute(resume_from=session)

    assert client.calls == {}


# ---------------------------------------------------------------------------
# The same contract, or none: resume as before
# ---------------------------------------------------------------------------


def test_the_same_contract_resumes(tmp_path):
    session = _interrupted(tmp_path)
    client = LedgerClient()

    result = _pipeline(client, tmp_path).execute(resume_from=session)

    assert result.success
    assert len(result.data) == ROWS
    assert token_for(FAILING_ROW) in client.calls
    assert token_for(0) not in client.calls  # restored, not asked again


def test_a_run_without_a_contract_resumes_as_before(tmp_path):
    session = _interrupted(tmp_path, max_length=None)
    client = LedgerClient()

    result = _pipeline(client, tmp_path, max_length=None).execute(resume_from=session)

    assert result.success
    assert len(result.data) == ROWS
    assert token_for(FAILING_ROW) in client.calls
    assert token_for(0) not in client.calls
