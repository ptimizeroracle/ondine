"""
Prometheus metrics export for monitoring.

Provides instrumentation for external monitoring systems.
"""

_PROMETHEUS_IMPORT_ERROR: ImportError | None = None

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    _PROMETHEUS_AVAILABLE = True
except ImportError as exc:
    Counter = Gauge = Histogram = start_http_server = None  # type: ignore[assignment,misc]
    _PROMETHEUS_AVAILABLE = False
    _PROMETHEUS_IMPORT_ERROR = exc

from ondine.utils import get_logger

logger = get_logger(__name__)


class PrometheusMetrics:
    """
    Prometheus metrics exporter.

    Follows Single Responsibility: only handles metrics export.
    """

    def __init__(self, port: int = 9090):
        """
        Initialize Prometheus metrics.

        Args:
            port: Port for metrics HTTP server
        """
        if not _PROMETHEUS_AVAILABLE:
            raise ImportError(
                "Prometheus metrics support is not installed. "
                "Install with: pip install 'ondine[observability]'"
            ) from _PROMETHEUS_IMPORT_ERROR

        self.port = port
        self._server_started = False

        # Define metrics
        self.requests_total = Counter(
            "llm_requests_total",
            "Total LLM requests",
            ["provider", "model", "stage"],
        )

        self.request_duration = Histogram(
            "llm_request_duration_seconds",
            "LLM request duration in seconds",
            ["provider", "stage"],
        )

        self.cost_total = Gauge(
            "llm_cost_total_usd",
            "Total cost in USD",
            ["provider"],
        )

        self.errors_total = Counter(
            "llm_errors_total",
            "Total errors",
            ["stage", "error_type"],
        )

        self.rows_processed = Gauge(
            "llm_rows_processed_total",
            "Total rows processed",
            ["stage"],
        )

        self.rows_per_second = Gauge(
            "llm_rows_per_second",
            "Processing throughput",
        )

    def start_server(self) -> None:
        """Start HTTP server for metrics endpoint."""
        if not self._server_started:
            try:
                start_http_server(self.port)
                self._server_started = True
                logger.info(f"Prometheus metrics server started on port {self.port}")
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}")

    def record_request(
        self, provider: str, model: str, stage: str, duration: float
    ) -> None:
        """
        Record LLM request metrics.

        Args:
            provider: Provider name
            model: Model name
            stage: Stage name
            duration: Request duration in seconds
        """
        self.requests_total.labels(provider=provider, model=model, stage=stage).inc()

        self.request_duration.labels(provider=provider, stage=stage).observe(duration)

    def record_cost(self, provider: str, cost: float) -> None:
        """
        Record cost metric.

        Args:
            provider: Provider name
            cost: Cost in USD
        """
        self.cost_total.labels(provider=provider).set(cost)

    def record_error(self, stage: str, error_type: str) -> None:
        """
        Record error metric.

        Args:
            stage: Stage name
            error_type: Error type
        """
        self.errors_total.labels(stage=stage, error_type=error_type).inc()

    def record_rows_processed(self, stage: str, count: int) -> None:
        """
        Record rows processed.

        Args:
            stage: Stage name
            count: Number of rows
        """
        self.rows_processed.labels(stage=stage).set(count)

    def record_throughput(self, rows_per_second: float) -> None:
        """
        Record processing throughput.

        Args:
            rows_per_second: Throughput metric
        """
        self.rows_per_second.set(rows_per_second)
