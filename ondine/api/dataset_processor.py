"""
DatasetProcessor - Simplified convenience wrapper.

For users who just want to process data with minimal configuration.
"""

import pandas as pd

from ondine.api.pipeline_builder import PipelineBuilder


class DatasetProcessor:
    """
    Simplified API for single-prompt, single-column use cases.

    This is a convenience wrapper around PipelineBuilder for users
    who don't need fine-grained control.

    Example:
        processor = DatasetProcessor(
            data="data.csv",
            input_column="description",
            output_column="cleaned",
            prompt="Clean this text: {description}",
            llm_config={"provider": "openai", "model": "gpt-4o-mini"}
        )
        result = processor.run()
    """

    def __init__(
        self,
        data: str | pd.DataFrame,
        input_column: str,
        output_column: str,
        prompt: str,
        llm_config: dict[str, any],
    ):
        """
        Initialize dataset processor.

        Args:
            data: CSV file path or DataFrame
            input_column: Input column name
            output_column: Output column name
            prompt: Prompt template
            llm_config: LLM configuration dict
        """
        self.data = data
        self.input_column = input_column
        self.output_column = output_column
        self.prompt = prompt
        self.llm_config = llm_config

        # Build pipeline internally
        builder = PipelineBuilder.create()

        # Configure data source
        if isinstance(data, str):
            builder.from_csv(
                data,
                input_columns=[input_column],
                output_columns=[output_column],
            )
        elif isinstance(data, pd.DataFrame):
            builder.from_dataframe(
                data,
                input_columns=[input_column],
                output_columns=[output_column],
            )
        else:
            raise ValueError("data must be file path or DataFrame")

        # Configure prompt
        builder.with_prompt(prompt)

        # Configure LLM
        provider = llm_config.get("provider", "openai")
        model = llm_config.get("model", "gpt-4o-mini")
        api_key = llm_config.get("api_key")
        temperature = llm_config.get("temperature", 0.0)
        max_tokens = llm_config.get("max_tokens")

        builder.with_llm(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Build pipeline
        self.pipeline = builder.build()

    def run(self) -> pd.DataFrame:
        """
        Execute processing and return results.

        Returns:
            DataFrame with results
        """
        result = self.pipeline.execute()
        return result.data

    def run_sample(self, n: int = 10) -> pd.DataFrame:
        """
        Test on first N rows.

        Args:
            n: Number of rows to process

        Returns:
            DataFrame with sample results
        """
        # Create sample pipeline
        if isinstance(self.data, str):
            df = pd.read_csv(self.data).head(n)
        else:
            df = self.data.head(n)

        builder = (
            PipelineBuilder.create()
            .from_dataframe(
                df,
                input_columns=[self.input_column],
                output_columns=[self.output_column],
            )
            .with_prompt(self.prompt)
            .with_llm(
                provider=self.llm_config.get("provider", "openai"),
                model=self.llm_config.get("model", "gpt-4o-mini"),
                api_key=self.llm_config.get("api_key"),
                temperature=self.llm_config.get("temperature", 0.0),
            )
        )

        sample_pipeline = builder.build()
        result = sample_pipeline.execute()
        return result.data

    def estimate_cost(self) -> float:
        """
        Estimate total processing cost.

        Returns:
            Estimated cost in USD
        """
        estimate = self.pipeline.estimate_cost()
        return float(estimate.total_cost)
