"""
Custom exceptions for Ondine pipeline execution.

This module defines exception types for error classification and handling.
Distinguishes between retryable (transient) and non-retryable (fatal) errors.
"""


class NonRetryableError(Exception):
    """
    Base class for errors that should not be retried.

    These are fatal errors that indicate configuration issues, invalid inputs,
    or permanent failures that cannot be resolved by retrying the operation.

    Examples of non-retryable errors:
    - Model doesn't exist or has been decommissioned
    - Invalid API key or authentication failure
    - File not found or invalid configuration
    - Quota/credits exhausted (not rate limit)

    Retryable errors (handled separately):
    - Rate limit errors (429)
    - Network timeouts or connection errors
    - Temporary API unavailability (503)

    Example:
        ```python
        try:
            result = pipeline.execute()
        except NonRetryableError as e:
            # Fatal error - don't retry, fix configuration
            logger.error(f"Pipeline failed with fatal error: {e}")
            sys.exit(1)
        ```
    """

    pass


class ModelNotFoundError(NonRetryableError):
    """
    Model doesn't exist or has been decommissioned.

    Raised when:
    - LLM provider returns "model not found" error
    - Model has been decommissioned/deprecated
    - Invalid model name specified

    Example:
        ```python
        # This will raise ModelNotFoundError if model is invalid
        pipeline = (
            PipelineBuilder.create()
            .with_llm(provider="groq", model="llama-3.1-70b-versatile")  # Decommissioned
            .build()
        )
        ```

    Resolution:
        Update model name to a valid, current model.
    """

    pass


class InvalidAPIKeyError(NonRetryableError):
    """
    API key is invalid, missing, or authentication failed.

    Raised when:
    - API key environment variable not set
    - API key is invalid or expired
    - Authentication fails (401, 403)
    - Insufficient permissions

    Example:
        ```python
        # This will raise InvalidAPIKeyError if OPENAI_API_KEY not set
        pipeline = (
            PipelineBuilder.create()
            .with_llm(provider="openai", model="gpt-4o-mini")
            .build()
        )
        result = pipeline.execute()  # Fails here if no API key
        ```

    Resolution:
        Set valid API key in environment variable or configuration.
    """

    pass


class ConfigurationError(NonRetryableError):
    """
    Invalid configuration or missing required resources.

    Raised when:
    - Input file doesn't exist
    - Invalid parameter values
    - Missing required configuration
    - Incompatible settings

    Example:
        ```python
        # This will raise ConfigurationError if file doesn't exist
        pipeline = (
            PipelineBuilder.create()
            .from_csv("nonexistent.csv", ...)
            .build()
        )
        result = pipeline.execute()  # Fails here
        ```

    Resolution:
        Fix configuration or ensure required files/resources exist.
    """

    pass


class QuotaExceededError(NonRetryableError):
    """
    API quota or credits exhausted.

    Raised when:
    - Account credits/quota exhausted
    - Billing issue or payment required
    - Account suspended

    Note:
        This is different from rate limiting (429), which is retryable.
        Quota errors indicate a permanent issue until credits are added.

    Example:
        ```python
        try:
            result = pipeline.execute()
        except QuotaExceededError:
            # Need to add credits or upgrade plan
            logger.error("Account quota exceeded - add credits")
        ```

    Resolution:
        Add credits, upgrade plan, or resolve billing issues.
    """

    pass
