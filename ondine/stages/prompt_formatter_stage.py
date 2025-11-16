"""Prompt formatting stage for template-based prompt generation."""

from decimal import Decimal
from typing import Any

import pandas as pd
from jinja2 import Template as Jinja2Template

from ondine.core.models import (
    CostEstimate,
    PromptBatch,
    RowMetadata,
    ValidationResult,
)
from ondine.core.specifications import PromptSpec
from ondine.stages.pipeline_stage import PipelineStage


class PromptFormatterStage(
    PipelineStage[tuple[pd.DataFrame, PromptSpec], list[PromptBatch]]
):
    """
    Format prompts using template and row data.

    Responsibilities:
    - Extract input columns from rows
    - Format prompts using template
    - Batch prompts for efficient processing
    - Attach metadata for tracking
    """

    def __init__(self, batch_size: int = 100, use_jinja2: bool = False):
        """
        Initialize prompt formatter stage.

        Args:
            batch_size: Number of prompts per batch
            use_jinja2: Use Jinja2 for template rendering
        """
        super().__init__("PromptFormatter")
        self.batch_size = batch_size
        self.use_jinja2 = use_jinja2

    def process(
        self, input_data: tuple[pd.DataFrame, PromptSpec], context: Any
    ) -> list[PromptBatch]:
        """Format prompts from DataFrame rows."""
        df, prompt_spec = input_data

        prompts: list[str] = []
        metadata_list: list[RowMetadata] = []

        # Extract template variables and system message
        template_str = prompt_spec.template
        system_message = prompt_spec.system_message

        # Create template renderer
        if self.use_jinja2:
            # Note: autoescape=False is intentional for LLM prompts (not HTML)
            # We're generating text prompts, not web content, so HTML escaping
            # would corrupt the prompt data sent to the LLM
            template = Jinja2Template(template_str, autoescape=False)  # noqa: S701

        # Format prompt for each row
        # Performance optimization: Use itertuples() instead of iterrows() for 10× speedup
        # itertuples() returns namedtuples which are much faster than Series objects
        import time

        total_rows = len(df)
        start_time = time.time()
        last_log_time = start_time
        last_log_pct = 0

        self.logger.info(f"Formatting {total_rows:,} prompts...")

        for row_count, row in enumerate(df.itertuples(index=True), 1):
            # Hybrid progress: Log every 10% OR every 30 seconds (only for slow operations)
            current_time = time.time()
            current_pct = int((row_count / total_rows) * 100)
            elapsed = current_time - start_time

            # Only log progress if operation is taking >5 seconds
            should_log = elapsed > 5 and (
                (current_pct >= last_log_pct + 10 and current_pct <= 90)  # Every 10%
                or (current_time - last_log_time >= 30)  # OR every 30s
            )

            if should_log:
                elapsed = current_time - start_time
                throughput = row_count / elapsed if elapsed > 0 else 0
                eta = (total_rows - row_count) / throughput if throughput > 0 else 0

                self.logger.info(
                    f"Formatting: {current_pct}% ({row_count:,}/{total_rows:,}) | "
                    f"{throughput:,.0f} rows/s | ETA: {eta:.0f}s"
                )
                last_log_time = current_time
                last_log_pct = current_pct

            try:
                # Extract index (first element of namedtuple)
                idx = row[0]

                # Extract input columns from namedtuple
                # Build row_data dict from column names and namedtuple attributes
                row_data = {}
                for col in df.columns:
                    if col in template_str:
                        # Get attribute by column name (namedtuples have column names as attributes)
                        row_data[col] = getattr(row, col)

                # Format prompt (Jinja2 or f-string)
                if self.use_jinja2:
                    prompt = template.render(**row_data)
                else:
                    prompt = template_str.format(**row_data)

                # Add few-shot examples if specified (but NOT system message)
                if prompt_spec.few_shot_examples:
                    examples_text = self._format_few_shot_examples(
                        prompt_spec.few_shot_examples
                    )
                    prompt = f"{examples_text}\n\n{prompt}"

                # NOTE: Do NOT add system message to prompt here
                # It will be passed separately via metadata for caching optimization

                prompts.append(prompt)

                # Create metadata with system message for LLM stage
                # Get 'id' column if it exists
                row_id = getattr(row, "id", None) if hasattr(row, "id") else None
                metadata = RowMetadata(
                    row_index=idx,
                    row_id=row_id,
                    custom={"system_message": system_message}
                    if system_message
                    else None,
                )
                metadata_list.append(metadata)

            except (KeyError, AttributeError) as e:
                self.logger.warning(f"Missing template variable at row {idx}: {e}")
                continue
            except Exception as e:
                self.logger.error(f"Error formatting prompt at row {idx}: {e}")
                continue

        # Create batches
        batches: list[PromptBatch] = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]
            batch_metadata = metadata_list[i : i + self.batch_size]

            batch = PromptBatch(
                prompts=batch_prompts,
                metadata=batch_metadata,
                batch_id=i // self.batch_size,
            )
            batches.append(batch)

        # Final summary
        total_time = time.time() - start_time
        throughput = len(prompts) / total_time if total_time > 0 else 0
        self.logger.info(
            f"✓ Formatted {len(prompts):,} prompts in {total_time:.1f}s ({throughput:,.0f} rows/s)"
        )

        return batches

    def _format_few_shot_examples(self, examples: list[dict[str, str]]) -> str:
        """
        Format few-shot examples for prompt.

        Args:
            examples: List of example dicts with 'input' and 'output'

        Returns:
            Formatted examples text
        """
        formatted = ["Here are some examples:\n"]

        for i, example in enumerate(examples, 1):
            formatted.append(f"Example {i}:")
            formatted.append(f"Input: {example.get('input', '')}")
            formatted.append(f"Output: {example.get('output', '')}")
            formatted.append("")

        return "\n".join(formatted)

    def validate_input(
        self, input_data: tuple[pd.DataFrame, PromptSpec]
    ) -> ValidationResult:
        """Validate DataFrame and prompt specification."""
        result = ValidationResult(is_valid=True)

        df, prompt_spec = input_data

        # Check DataFrame not empty
        if df.empty:
            result.add_error("DataFrame is empty")

        # Check template variables exist in DataFrame
        template = prompt_spec.template
        import re

        variables = re.findall(r"\{(\w+)\}", template)
        missing_vars = set(variables) - set(df.columns)

        if missing_vars:
            result.add_error(f"Template variables not in DataFrame: {missing_vars}")

        return result

    def estimate_cost(
        self, input_data: tuple[pd.DataFrame, PromptSpec]
    ) -> CostEstimate:
        """Prompt formatting has no LLM cost."""
        return CostEstimate(
            total_cost=Decimal("0.0"),
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            rows=len(input_data[0]),
        )
