"""
Base pipeline stage abstraction.

Defines the contract for all processing stages using Template Method
pattern for execution flow.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from ondine.core.models import CostEstimate, ValidationResult
from ondine.utils import get_logger

# Type variables for input and output
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")

logger = get_logger(__name__)


class PipelineStage(ABC, Generic[TInput, TOutput]):
    """
    Abstract base class for all pipeline stages.

    Implements Template Method pattern with hooks for extensibility. All stages
    follow Single Responsibility Principle and are composable via the Chain of
    Responsibility pattern.

    Stages in the pipeline:
    1. DataLoaderStage - Load data from source
    2. PromptFormatterStage - Format prompts with data
    3. LLMInvocationStage - Call LLM API
    4. ResponseParserStage - Parse LLM responses
    5. ResultWriterStage - Write results to output

    Example:
        ```python
        class CustomStage(PipelineStage[pd.DataFrame, pd.DataFrame]):
            def process(self, input_data, context):
                # Custom processing logic
                return processed_data

            def validate_input(self, input_data):
                # Validation logic
                return ValidationResult(is_valid=True)
        ```
    """

    def __init__(self, name: str):
        """
        Initialize pipeline stage.

        Args:
            name: Human-readable stage name
        """
        self.name = name
        self.logger = get_logger(f"{__name__}.{name}")

    @abstractmethod
    def process(self, input_data: TInput, context: Any) -> TOutput:
        """
        Core processing logic (must be implemented by subclasses).

        Args:
            input_data: Input data for this stage
            context: Execution context with shared state

        Returns:
            Processed output data
        """
        pass

    @abstractmethod
    def validate_input(self, input_data: TInput) -> ValidationResult:
        """
        Validate input before processing.

        Args:
            input_data: Input to validate

        Returns:
            ValidationResult with errors/warnings
        """
        pass

    def execute(self, input_data: TInput, context: Any) -> TOutput:
        """
        Execute stage with pre/post hooks (Template Method).

        This method orchestrates the execution flow and should not
        be overridden.

        Args:
            input_data: Input data
            context: Execution context

        Returns:
            Processed output

        Raises:
            ValueError: If input validation fails
        """
        self.logger.info(f"Starting stage: {self.name}")

        # Pre-processing hook
        self.before_process(context)

        # Validate input
        validation = self.validate_input(input_data)
        if not validation.is_valid:
            error_msg = f"Input validation failed: {validation.errors}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if validation.warnings:
            for warning in validation.warnings:
                self.logger.warning(warning)

        # Core processing
        try:
            result = self.process(input_data, context)
            self.logger.info(f"Completed stage: {self.name}")

            # Post-processing hook
            self.after_process(result, context)

            return result
        except Exception as e:
            self.logger.error(f"Stage {self.name} failed: {e}")
            error_decision = self.on_error(e, context)
            raise error_decision

    def before_process(self, context: Any) -> None:
        """
        Hook called before processing (default: no-op).

        Args:
            context: Execution context
        """
        pass

    def after_process(self, result: TOutput, context: Any) -> None:
        """
        Hook called after successful processing (default: no-op).

        Args:
            result: Processing result
            context: Execution context
        """
        pass

    def on_error(self, error: Exception, context: Any) -> Exception:
        """
        Hook called on processing error (default: re-raise).

        Args:
            error: The exception that occurred
            context: Execution context

        Returns:
            Exception to raise (can transform error)
        """
        return error

    @abstractmethod
    def estimate_cost(self, input_data: TInput) -> CostEstimate:
        """
        Estimate processing cost for this stage.

        Args:
            input_data: Input data to estimate for

        Returns:
            Cost estimate
        """
        pass
