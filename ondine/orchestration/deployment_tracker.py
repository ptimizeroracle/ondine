"""
Deployment tracker for LiteLLM Router distribution monitoring.

Maps internal LiteLLM deployment hashes to user-friendly identifiers
and tracks request distribution across deployments.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class DeploymentInfo:
    """Information about a single deployment."""

    id: str
    label: str
    model: str
    provider: str


class DeploymentTracker:
    """
    Tracks LiteLLM Router deployment distribution.

    LiteLLM Router uses internal hash IDs for deployments that don't match
    the user-configured model_id. This class provides:

    1. Dynamic mapping: First-Seen-First-Assigned mapping of hash IDs to
       friendly names from the original configuration.

    2. Distribution tracking: Counts requests per deployment for summary
       reporting.

    Example:
        tracker = DeploymentTracker(router.model_list)

        # During processing
        for response in responses:
            hash_id = extract_deployment_id(response)
            friendly_id = tracker.register_deployment(hash_id)
            tracker.record_request(hash_id)

        # After processing
        print(tracker.get_distribution_summary())
    """

    def __init__(self, model_list: list[dict[str, Any]] | None = None):
        """
        Initialize tracker with Router's model list.

        Args:
            model_list: LiteLLM Router's model_list configuration.
                        Each entry should have 'model_name', 'model_id',
                        and 'litellm_params' with 'model'.
        """
        self._hash_to_friendly: dict[str, str] = {}
        self._hash_to_label: dict[str, str] = {}
        self._distribution: dict[str, int] = {}
        self._available_names: list[DeploymentInfo] = []

        if model_list:
            self._initialize_from_model_list(model_list)

    def _initialize_from_model_list(self, model_list: list[dict[str, Any]]) -> None:
        """Parse model_list to prepare friendly name queue."""
        for i, dep in enumerate(model_list):
            # Get friendly name and model info
            friendly_id = dep.get("model_id", dep.get("model_name", "unknown"))

            # Ensure ID is unique for visualization even if model_name is shared
            # (Router requires shared model_name for load balancing)
            unique_id = f"{friendly_id}_{i}"

            litellm_params = dep.get("litellm_params", {})
            model = litellm_params.get("model", "unknown")

            # Parse provider from model string
            provider = model.split("/")[0] if "/" in model else ""
            model_short = model.split("/")[1] if "/" in model else model

            self._available_names.append(
                DeploymentInfo(
                    id=unique_id,
                    label=f"{provider}/{model_short} ({friendly_id})",
                    model=model,
                    provider=provider,
                )
            )

    def register_deployment(self, hash_id: str) -> str:
        """
        Register a deployment hash and return its friendly ID.

        Uses First-Seen-First-Assigned strategy: the first unknown hash
        gets the first available friendly name from the config.

        Args:
            hash_id: Internal LiteLLM deployment hash

        Returns:
            Friendly ID for display (or hash_id if no names available)
        """
        if hash_id in self._hash_to_friendly:
            return self._hash_to_friendly[hash_id]

        if self._available_names:
            info = self._available_names.pop(0)
            self._hash_to_friendly[hash_id] = info.id
            self._hash_to_label[hash_id] = info.label
            return info.id

        # No friendly names left, use hash as-is
        return hash_id

    def get_friendly_id(self, hash_id: str) -> str | None:
        """
        Get friendly ID for a hash without registering.

        Args:
            hash_id: Internal LiteLLM deployment hash

        Returns:
            Friendly ID if registered, None otherwise
        """
        return self._hash_to_friendly.get(hash_id)

    def get_label(self, hash_id: str) -> str:
        """
        Get display label for a deployment.

        Args:
            hash_id: Internal LiteLLM deployment hash

        Returns:
            Human-readable label (e.g., "groq/llama-3.3 (fast-model)")
        """
        return self._hash_to_label.get(hash_id, "")

    def record_request(self, hash_id: str) -> None:
        """
        Record a request to a deployment.

        Args:
            hash_id: Internal LiteLLM deployment hash
        """
        self._distribution[hash_id] = self._distribution.get(hash_id, 0) + 1

    def get_distribution(self) -> dict[str, int]:
        """
        Get raw distribution counts.

        Returns:
            Dict mapping deployment hash to request count
        """
        return self._distribution.copy()

    def get_distribution_summary(self) -> dict[str, dict[str, Any]]:
        """
        Get distribution summary with friendly names and percentages.

        Returns:
            Dict with deployment info including friendly_id, label,
            count, and percentage.
        """
        total = sum(self._distribution.values())
        summary = {}

        for hash_id, count in sorted(self._distribution.items()):
            friendly_id = self._hash_to_friendly.get(hash_id, hash_id)
            label = self._hash_to_label.get(hash_id, "")
            percentage = (count / total * 100) if total > 0 else 0

            summary[hash_id] = {
                "friendly_id": friendly_id,
                "label": label,
                "count": count,
                "percentage": percentage,
            }

        return summary

    def get_deployments_for_progress(self) -> list[dict[str, Any]]:
        """
        Get deployment info formatted for progress tracker initialization.

        Returns:
            List of dicts with model_id, label, and weight for each deployment.
        """
        return [
            {
                "model_id": info.id,
                "label": info.label,
                "weight": 1.0,  # Assume equal distribution for UI
            }
            for info in self._available_names
        ]

    @property
    def total_requests(self) -> int:
        """Total number of recorded requests."""
        return sum(self._distribution.values())

    def __repr__(self) -> str:
        return (
            f"DeploymentTracker(deployments={len(self._hash_to_friendly)}, "
            f"total_requests={self.total_requests})"
        )
