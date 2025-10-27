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

        # Extract template variables
        template_str = prompt_spec.template

        # Create template renderer
        if self.use_jinja2:
            # Note: autoescape=False is intentional for LLM prompts (not HTML)
            # We're generating text prompts, not web content, so HTML escaping
            # would corrupt the prompt data sent to the LLM
            template = Jinja2Template(template_str, autoescape=False)  # noqa: S701

        # Format prompt for each row
        for idx, row in df.iterrows():
            try:
                # Extract input columns
                row_data = {col: row[col] for col in df.columns if col in template_str}

                # Format prompt (Jinja2 or f-string)
                if self.use_jinja2:
                    prompt = template.render(**row_data)
                else:
                    prompt = template_str.format(**row_data)

                # Add few-shot examples if specified
                if prompt_spec.few_shot_examples:
                    examples_text = self._format_few_shot_examples(
                        prompt_spec.few_shot_examples
                    )
                    prompt = f"{examples_text}\n\n{prompt}"

                # Add system message if specified
                if prompt_spec.system_message:
                    prompt = f"{prompt_spec.system_message}\n\n{prompt}"

                prompts.append(prompt)

                # Create metadata
                metadata = RowMetadata(
                    row_index=idx,
                    row_id=row.get("id", None),
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

        self.logger.info(
            f"Formatted {len(prompts)} prompts into {len(batches)} batches"
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
