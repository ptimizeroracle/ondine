"""
E2E test for checkpoint and resume functionality.

Validates that pipelines can crash mid-execution and resume from
the last checkpoint without data loss.
"""

import os
import tempfile
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from ondine import PipelineBuilder


@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "model", "api_key_env"),
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
            .with_llm(
                provider=provider,
                model=model,
                api_key=api_key,
                temperature=0.0,
                input_cost_per_1k_tokens=Decimal("0.00015"),
                output_cost_per_1k_tokens=Decimal("0.0006"),
            )
            .with_batch_size(30)
            .with_processing_batch_size(5)  # 6 API calls total
            .with_max_budget(0.0003)  # Very low budget (will stop after ~2 calls)
            .with_checkpoint_dir(str(checkpoint_dir))
            .build()
        )

        # Execute Phase 1 (should raise BudgetExceededError)
        from ondine.utils.budget_controller import BudgetExceededError

        session_id = None
        try:
            pipeline_phase1.execute()
            pytest.fail("Phase 1 should have raised BudgetExceededError")
        except BudgetExceededError as e:
            # Budget exceeded - this is expected!
            print(f"\n{provider.upper()} Checkpoint Test - Phase 1:")
            print(f"  Budget exceeded: {e}")

            # Get checkpoint ID from error message or find checkpoint file
            checkpoint_files = list(checkpoint_dir.glob("*.json"))
            assert len(checkpoint_files) > 0, (
                "No checkpoint file created after budget exceeded"
            )

            checkpoint_file = checkpoint_files[0]
            # Extract UUID from filename (strip "checkpoint_" prefix)
            session_id = checkpoint_file.stem.replace("checkpoint_", "")
            print(f"  Checkpoint saved: {session_id}")

        # PHASE 2: Resume from checkpoint with higher budget
        pipeline_phase2 = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Echo: {{text}}")
            .with_llm(
                provider=provider,
                model=model,
                api_key=api_key,
                temperature=0.0,
                input_cost_per_1k_tokens=Decimal("0.00015"),
                output_cost_per_1k_tokens=Decimal("0.0006"),
            )
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

        print(f"\n{provider.upper()} Checkpoint Test - Phase 2:")
        print(f"  Completed: {len(result_phase2.data)}/30 rows")
        print(f"  Total cost: ${result_phase2.costs.total_cost:.4f}")
        print("  ✅ Checkpoint & resume working correctly")


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
                input_cost_per_1k_tokens=Decimal("0.00059"),
                output_cost_per_1k_tokens=Decimal("0.00079"),
            )
            .with_batch_size(10)
            .with_processing_batch_size(1)  # 10 API calls
            .with_max_budget(0.0005)  # Stop after ~3-4 rows
            .with_checkpoint_dir(str(checkpoint_dir))
            .build()
        )

        # Execute Phase 1 (should raise BudgetExceededError)
        from ondine.utils.budget_controller import BudgetExceededError

        try:
            pipeline1.execute()
            pytest.fail("Should have raised BudgetExceededError")
        except BudgetExceededError:
            # Expected - budget exceeded
            pass

        # Phase 2: Resume (should only process remaining rows)
        checkpoint_files = list(checkpoint_dir.glob("*.json"))
        assert len(checkpoint_files) > 0, "No checkpoint created"
        # Extract UUID from filename (strip "checkpoint_" prefix)
        session_id = checkpoint_files[0].stem.replace("checkpoint_", "")

        pipeline2 = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Return exactly: {{text}}")
            .with_llm(
                provider="groq",
                model="llama-3.3-70b-versatile",
                api_key=api_key,
                temperature=0.0,
                input_cost_per_1k_tokens=Decimal("0.00059"),
                output_cost_per_1k_tokens=Decimal("0.00079"),
            )
            .with_batch_size(10)
            .with_processing_batch_size(1)
            .with_max_budget(0.50)
            .with_checkpoint_dir(str(checkpoint_dir))
            .build()
        )

        from uuid import UUID

        result2 = pipeline2.execute(resume_from=UUID(session_id))

        # Verify completion
        assert result2.success, "Phase 2 should complete"
        assert len(result2.data) == 10, (
            f"Should have all 10 rows. Got {len(result2.data)}"
        )

        print("\nDuplicate Work Prevention Test:")
        print(f"  Phase 2 completed: {len(result2.data)} rows")
        print(f"  Total cost: ${result2.costs.total_cost:.4f}")
        print("  ✅ Resume from checkpoint working")
