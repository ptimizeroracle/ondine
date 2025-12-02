"""Prompt formatting stage for template-based prompt generation."""

import time
from decimal import Decimal
from typing import Any

from jinja2 import Template as Jinja2Template

from ondine.core.data_container import DataContainer
from ondine.core.models import (
    CostEstimate,
    PromptBatch,
    RowMetadata,
    ValidationResult,
)
from ondine.core.specifications import PromptSpec
from ondine.stages.pipeline_stage import PipelineStage


class PromptFormatterStage(
    PipelineStage[tuple[DataContainer, PromptSpec], list[PromptBatch]]
):
    """
    Format prompts using template and row data.

    Now accepts DataContainer instead of pd.DataFrame for
    framework-agnostic processing.

    Responsibilities:
    - Extract input columns from rows
    - Format prompts using template
    - Batch prompts for efficient processing
    - Attach metadata for tracking
    """

    def __init__(self, batch_size: int = 100, use_jinja2: bool | None = None):
        """
        Initialize prompt formatter stage.

        Args:
            batch_size: Number of prompts per batch
            use_jinja2: Template rendering mode
                - None (default): Auto-detect based on {{ }} syntax
                - True: Force Jinja2 rendering
                - False: Force Python .format() rendering
        """
        super().__init__("PromptFormatter")
        self.batch_size = batch_size
        self.use_jinja2 = use_jinja2

    def process(
        self, input_data: tuple[DataContainer, PromptSpec], context: Any
    ) -> list[PromptBatch]:
        """Format prompts from DataContainer rows."""
        container_or_df, prompt_spec = input_data

        # Auto-wrap pandas DataFrame for backward compatibility
        try:
            import pandas as pd

            if isinstance(container_or_df, pd.DataFrame):
                from ondine.adapters.containers import PandasContainer

                container = PandasContainer(container_or_df)
            else:
                container = container_or_df
        except ImportError:
            container = container_or_df

        prompts: list[str] = []
        metadata_list: list[RowMetadata] = []

        # Extract template variables and system message
        template_str = prompt_spec.template
        system_message = prompt_spec.system_message

        # Determine template rendering mode
        if self.use_jinja2 is None:
            # Auto-detect: Use Jinja2 if template contains {{ }}
            use_jinja2 = "{{" in template_str
            if use_jinja2:
                self.logger.info(
                    "Auto-detected Jinja2 syntax ({{variable}}), enabling Jinja2 renderer"
                )
        else:
            # Respect explicit user choice (True or False)
            use_jinja2 = self.use_jinja2

        # Create template renderer
        jinja_template = None
        if use_jinja2:
            # Note: autoescape=False is intentional for LLM prompts (not HTML)
            # We're generating text prompts, not web content, so HTML escaping
            # would corrupt the prompt data sent to the LLM
            jinja_template = Jinja2Template(template_str, autoescape=False)  # noqa: S701

        # Format prompt for each row
        total_rows = len(container)
        start_time = time.time()
        last_log_time = start_time
        last_log_pct = 0

        self.logger.info(f"Formatting {total_rows:,} prompts...")

        for row_count, row in enumerate(container, 1):
            # row is now a dict (Row type)
            idx = row.get(
                "_index", row_count - 1
            )  # Use _index if available, else position

            # Hybrid progress: Log every 10% OR every 30 seconds (only for slow operations)
            current_time = time.time()
            current_pct = int((row_count / total_rows) * 100) if total_rows > 0 else 100
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
                # Extract only template variables from row
                # Note: For Jinja2, we pass all row data since variable extraction
                # from Jinja2 templates is complex (filters, expressions, etc.)
                if use_jinja2:
                    row_data = dict(row)
                else:
                    row_data = {k: v for k, v in row.items() if k in template_str}

                # Format prompt (Jinja2 or f-string)
                if use_jinja2 and jinja_template:
                    prompt = jinja_template.render(**row_data)
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
                row_id = row.get("id")
                metadata = RowMetadata(
                    row_index=idx,
                    row_id=row_id,
                    custom={"system_message": system_message}
                    if system_message
                    else None,
                )
                metadata_list.append(metadata)

            except KeyError as e:
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
        self, input_data: tuple[DataContainer, PromptSpec]
    ) -> ValidationResult:
        """Validate DataContainer and prompt specification."""
        result = ValidationResult(is_valid=True)

        container, prompt_spec = input_data

        # Check container not empty
        if len(container) == 0:
            result.add_error("DataContainer is empty")

        # Check template variables exist in container columns
        template = prompt_spec.template
        import re

        variables = re.findall(r"\{(\w+)\}", template)
        missing_vars = set(variables) - set(container.columns)

        if missing_vars:
            result.add_error(f"Template variables not in container: {missing_vars}")

        return result

    def estimate_cost(
        self, input_data: tuple[DataContainer, PromptSpec]
    ) -> CostEstimate:
        """Prompt formatting has no LLM cost."""
        return CostEstimate(
            total_cost=Decimal("0.0"),
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            rows=len(input_data[0]),
        )
