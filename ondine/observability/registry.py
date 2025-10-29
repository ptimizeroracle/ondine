"""
Observer registry for plugin-based observability.

Allows dynamic registration and discovery of observer implementations.
"""

from collections.abc import Callable

from ondine.observability.base import PipelineObserver


class ObserverRegistry:
    """
    Global registry of observer implementations.

    Allows dynamic observer registration and instantiation via decorator pattern.

    Example:
        @observer("my_observer")
        class MyObserver(PipelineObserver):
            def on_llm_call(self, event):
                print(f"LLM called: {event.model}")

        # Later, instantiate by name
        observer_class = ObserverRegistry.get("my_observer")
        observer = observer_class(config={...})
    """

    _observers: dict[str, type[PipelineObserver]] = {}

    @classmethod
    def register(cls, name: str, observer_class: type[PipelineObserver]) -> None:
        """
        Register an observer implementation.

        Args:
            name: Observer identifier (e.g., "langfuse", "opentelemetry")
            observer_class: Observer class implementing PipelineObserver

        Raises:
            ValueError: If observer already registered or invalid class
        """
        if name in cls._observers:
            raise ValueError(
                f"Observer '{name}' already registered. "
                f"Use a different name or unregister first."
            )

        if not issubclass(observer_class, PipelineObserver):
            raise ValueError(
                f"Observer class must inherit from PipelineObserver, "
                f"got {observer_class.__name__}"
            )

        cls._observers[name] = observer_class

    @classmethod
    def get(cls, name: str) -> type[PipelineObserver]:
        """
        Get observer class by name.

        Args:
            name: Observer identifier

        Returns:
            Observer class (not instantiated)

        Raises:
            ValueError: If observer not found
        """
        if name not in cls._observers:
            available = ", ".join(cls._observers.keys()) if cls._observers else "none"
            raise ValueError(
                f"Observer '{name}' not found. Available observers: {available}"
            )

        return cls._observers[name]

    @classmethod
    def list_observers(cls) -> list[str]:
        """
        List all registered observer names.

        Returns:
            List of observer identifiers
        """
        return list(cls._observers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if observer is registered.

        Args:
            name: Observer identifier

        Returns:
            True if registered, False otherwise
        """
        return name in cls._observers

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister an observer (mainly for testing).

        Args:
            name: Observer identifier
        """
        cls._observers.pop(name, None)


def observer(name: str) -> Callable:
    """
    Decorator to register an observer.

    Automatically registers the decorated class with the ObserverRegistry.

    Args:
        name: Observer identifier (e.g., "langfuse", "opentelemetry")

    Returns:
        Decorator function

    Example:
        @observer("custom_logger")
        class CustomLogger(PipelineObserver):
            def on_llm_call(self, event):
                logging.info(f"LLM: {event.model}")
    """

    def decorator(observer_class: type[PipelineObserver]) -> type[PipelineObserver]:
        ObserverRegistry.register(name, observer_class)
        return observer_class

    return decorator
