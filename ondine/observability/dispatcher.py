"""
Event dispatcher for coordinating observer notifications.

Ensures that events are delivered to all observers even if some fail.
"""

import logging
import traceback
from typing import Any

from ondine.observability.base import PipelineObserver

logger = logging.getLogger(__name__)


class ObserverDispatcher:
    """
    Coordinates event dispatch to all registered observers.

    Ensures:
    - Events are sent to all observers
    - Observer failures don't crash the pipeline
    - Errors are logged for debugging
    - Clean shutdown with flush and close

    Design Note: This implementation uses synchronous dispatch for simplicity.
    The architecture is ready for async dispatch if needed in the future
    (replace `dispatch()` with `async def dispatch()` and use asyncio.gather()).

    Example:
        observers = [LangfuseObserver(config), OpenTelemetryObserver(config)]
        dispatcher = ObserverDispatcher(observers)

        # Dispatch events
        dispatcher.dispatch("llm_call", LLMCallEvent(...))

        # At pipeline end
        dispatcher.flush_all()
        dispatcher.close_all()
    """

    def __init__(self, observers: list[PipelineObserver]):
        """
        Initialize dispatcher with observers.

        Args:
            observers: List of observer instances to notify
        """
        self.observers = observers
        self.logger = logger

    def dispatch(self, event_type: str, event: Any) -> None:
        """
        Dispatch event to all observers.

        Each observer is called in isolation - if one fails,
        others still receive the event. This ensures pipeline
        reliability isn't compromised by observability issues.

        Args:
            event_type: Event type name (e.g., "llm_call", "pipeline_start")
            event: Event object to dispatch

        Note: Errors are logged but not raised to prevent observer
        failures from crashing the pipeline.
        """
        method_name = f"on_{event_type}"

        for observer in self.observers:
            try:
                # Get the method for this event type
                method = getattr(observer, method_name, None)

                if method is None:
                    # Observer doesn't implement this event handler (OK - it's optional)
                    continue

                # Call the observer method
                method(event)

            except Exception as e:
                # Log the error but continue to other observers
                # Observer failures should NEVER crash the pipeline
                self.logger.error(
                    f"Observer {observer.__class__.__name__} failed on {event_type}: {e}\n"
                    f"Traceback:\n{traceback.format_exc()}\n"
                    f"Pipeline will continue without this observer's data."
                )

    def flush_all(self) -> None:
        """
        Flush all observers.

        Calls flush() on each observer to ensure buffered events are sent.
        This should be called at the end of pipeline execution.
        """
        for observer in self.observers:
            try:
                observer.flush()
            except Exception as e:
                self.logger.error(
                    f"Observer {observer.__class__.__name__} flush failed: {e}\n"
                    f"Some events may not have been sent."
                )

    def close_all(self) -> None:
        """
        Close all observers.

        Calls close() on each observer to clean up resources.
        This should be called when the pipeline is fully complete.
        """
        for observer in self.observers:
            try:
                observer.close()
            except Exception as e:
                self.logger.error(
                    f"Observer {observer.__class__.__name__} close failed: {e}\n"
                    f"Resource cleanup may be incomplete."
                )

    def add_observer(self, observer: PipelineObserver) -> None:
        """
        Add an observer dynamically.

        Args:
            observer: Observer instance to add
        """
        self.observers.append(observer)

    def remove_observer(self, observer: PipelineObserver) -> None:
        """
        Remove an observer dynamically.

        Args:
            observer: Observer instance to remove
        """
        if observer in self.observers:
            self.observers.remove(observer)
