"""
Health check API for pipeline monitoring.

Provides status information for operational monitoring.
"""

from datetime import datetime
from typing import Any

from ondine.api.pipeline import Pipeline
from ondine.utils import get_logger

logger = get_logger(__name__)


class HealthCheck:
    """
    Health check API for monitoring pipeline status.

    Provides information about pipeline health and readiness.
    """

    def __init__(self, pipeline: Pipeline):
        """
        Initialize health check.

        Args:
            pipeline: Pipeline instance to monitor
        """
        self.pipeline = pipeline
        self.last_check: datetime | None = None
        self.last_status: dict[str, Any] = {}

    def check(self) -> dict[str, Any]:
        """
        Perform health check.

        Returns:
            Health status dictionary
        """
        self.last_check = datetime.now()

        status = {
            "status": "healthy",
            "timestamp": self.last_check.isoformat(),
            "pipeline_id": str(self.pipeline.id),
            "checks": {},
        }

        # Check LLM provider configuration
        try:
            llm_spec = self.pipeline.specifications.llm
            status["checks"]["llm_provider"] = {
                "status": "ok",
                "provider": llm_spec.provider.value,
                "model": llm_spec.model,
            }
        except Exception as e:
            status["checks"]["llm_provider"] = {
                "status": "error",
                "error": str(e),
            }
            status["status"] = "unhealthy"

        # Check data source configuration
        try:
            dataset_spec = self.pipeline.specifications.dataset
            source_exists = True

            if dataset_spec.source_path:
                source_exists = dataset_spec.source_path.exists()

            status["checks"]["data_source"] = {
                "status": "ok" if source_exists else "warning",
                "source_type": dataset_spec.source_type.value,
                "exists": source_exists,
            }
        except Exception as e:
            status["checks"]["data_source"] = {
                "status": "error",
                "error": str(e),
            }
            status["status"] = "unhealthy"

        # Check checkpoint storage
        try:
            checkpoint_dir = self.pipeline.specifications.processing.checkpoint_dir
            status["checks"]["checkpoint_storage"] = {
                "status": "ok",
                "directory": str(checkpoint_dir),
                "exists": checkpoint_dir.exists(),
            }
        except Exception as e:
            status["checks"]["checkpoint_storage"] = {
                "status": "warning",
                "error": str(e),
            }

        # Store last status
        self.last_status = status

        return status

    def is_healthy(self) -> bool:
        """
        Check if pipeline is healthy.

        Returns:
            True if healthy
        """
        status = self.check()
        return status["status"] == "healthy"

    def get_readiness(self) -> dict[str, Any]:
        """
        Get readiness status.

        Returns:
            Readiness information
        """
        validation = self.pipeline.validate()

        return {
            "ready": validation.is_valid,
            "errors": validation.errors,
            "warnings": validation.warnings,
            "timestamp": datetime.now().isoformat(),
        }
