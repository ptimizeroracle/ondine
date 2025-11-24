"""
E2E test for checkpoint and resume functionality.

Validates that pipelines can crash mid-execution and resume from
the last checkpoint without data loss.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ondine import PipelineBuilder
from ondine.core.exceptions import BudgetExceededError


@pytest.mark.integration
@pytest.mark.parametrize(
    "provider,model,api_key_env",
    [
        ("openai", "gpt-4o-mini", "OPENAI_API_KEY"),
        ("groq", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    ],
)
def test_checkpoint_and_resume_after_budget_stop(provider, model, api_key_env):
    """
    Test checkpoint/resume by using budget limit to force a stop.

    Simulates a crash by hitting budget limit, then resumes to completion.
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        pytest.skip(f"{api_key_env} not set")

    # Create test data
    df = pd.DataFrame({"text": [f"Text {i}" for i in range(30)]})

    # Use temp directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)

        # PHASE 1: Run with low budget (will stop partway)
        pipeline_phase1 = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Echo: {{text}}")
            .with_llm(provider=provider, model=model, api_key=api_key, temperature=0.0)
            .with_batch_size(30)
            .with_processing_batch_size(5)  # 6 API calls total
            .with_max_budget(0.01)  # Very low budget (will stop after ~2-3 calls)
            .with_checkpoint_dir(str(checkpoint_dir))
            .build()
        )

        result_phase1 = pipeline_phase1.execute()

        # Verify phase 1 stopped due to budget
        assert not result_phase1.success, "Phase 1 should have stopped due to budget"
        rows_phase1 = len(result_phase1.data)
        assert rows_phase1 < 30, f"Should have stopped before all 30 rows. Got {rows_phase1}"

        # Get checkpoint ID
        checkpoint_files = list(checkpoint_dir.glob("*.json"))
        assert len(checkpoint_files) > 0, "No checkpoint file created"

        # Extract session ID from checkpoint filename
        checkpoint_file = checkpoint_files[0]
        session_id = checkpoint_file.stem

        print(f"\n{provider.upper()} Checkpoint Test - Phase 1:")
        print(f"  Stopped at: {rows_phase1}/30 rows")
        print(f"  Cost: ${result_phase1.costs.total_cost:.4f}")
        print(f"  Checkpoint: {session_id}")

        # PHASE 2: Resume from checkpoint with higher budget
        pipeline_phase2 = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Echo: {{text}}")
            .with_llm(provider=provider, model=model, api_key=api_key, temperature=0.0)
            .with_batch_size(30)
            .with_processing_batch_size(5)
            .with_max_budget(0.50)  # Higher budget to complete
            .with_checkpoint_dir(str(checkpoint_dir))
            .build()
        )

        # Resume from checkpoint
        from uuid import UUID

        result_phase2 = pipeline_phase2.execute(resume_from=UUID(session_id))

        # Verify phase 2 completed
        assert result_phase2.success, f"{provider} phase 2 should have completed"
        assert len(result_phase2.data) == 30, (
            f"Should have all 30 rows after resume. Got {len(result_phase2.data)}"
        )

        # Verify no duplicate processing (cost should be incremental)
        total_cost = result_phase2.costs.total_cost
        assert total_cost > result_phase1.costs.total_cost, (
            "Phase 2 should have additional cost"
        )

        print(f"\n{provider.upper()} Checkpoint Test - Phase 2:")
        print(f"  Completed: {len(result_phase2.data)}/30 rows")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  ✅ Checkpoint & resume working correctly")


@pytest.mark.integration
def test_checkpoint_prevents_duplicate_work():
    """
    Test that resuming from checkpoint doesn't re-process completed rows.

    Validates that checkpointing saves progress correctly.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    df = pd.DataFrame({"text": [f"Unique_{i}" for i in range(10)]})

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)

        # Phase 1: Process first 5 rows (force stop with budget)
        pipeline1 = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Return exactly: {{text}}")
            .with_llm(
                provider="groq",
                model="llama-3.3-70b-versatile",
                api_key=api_key,
                temperature=0.0,
            )
            .with_batch_size(10)
            .with_processing_batch_size(1)  # 10 API calls
            .with_max_budget(0.005)  # Stop after ~5 rows
            .with_checkpoint_dir(str(checkpoint_dir))
            .build()
        )

        result1 = pipeline1.execute()
        cost_phase1 = result1.costs.total_cost
        rows_phase1 = len(result1.data)

        # Phase 2: Resume (should only process remaining rows)
        checkpoint_files = list(checkpoint_dir.glob("*.json"))
        session_id = checkpoint_files[0].stem

        pipeline2 = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Return exactly: {{text}}")
            .with_llm(
                provider="groq",
                model="llama-3.3-70b-versatile",
                api_key=api_key,
                temperature=0.0,
            )
            .with_batch_size(10)
            .with_processing_batch_size(1)
            .with_max_budget(0.50)
            .with_checkpoint_dir(str(checkpoint_dir))
            .build()
        )

        from uuid import UUID

        result2 = pipeline2.execute(resume_from=UUID(session_id))

        # Verify incremental cost (not re-processing)
        cost_phase2 = result2.costs.total_cost
        incremental_cost = cost_phase2 - cost_phase1

        # Incremental cost should be roughly proportional to remaining rows
        remaining_rows = 10 - rows_phase1
        expected_ratio = remaining_rows / rows_phase1
        actual_ratio = incremental_cost / cost_phase1

        # Allow 50% tolerance for API variance
        assert 0.5 * expected_ratio <= actual_ratio <= 1.5 * expected_ratio, (
            f"Cost ratio suggests duplicate work. "
            f"Expected ~{expected_ratio:.2f}, got {actual_ratio:.2f}"
        )

        print(f"\nDuplicate Work Prevention Test:")
        print(f"  Phase 1: {rows_phase1} rows, ${cost_phase1:.4f}")
        print(f"  Phase 2: {10 - rows_phase1} rows, ${incremental_cost:.4f}")
        print(f"  ✅ No duplicate processing detected")

        assert result2.success
        assert len(result2.data) == 10

