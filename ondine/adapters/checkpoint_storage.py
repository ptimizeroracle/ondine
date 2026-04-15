"""
Checkpoint storage for fault tolerance.

Provides persistent storage of execution state to enable resume after
failures.
"""

import gzip
import json
import pickle  # nosec B403 — checkpoint files are local-only, not user-supplied
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

from ondine.core.models import CheckpointInfo


class CheckpointStorage(ABC):
    """
    Abstract base for checkpoint storage implementations.

    Follows Strategy pattern for pluggable storage backends.
    """

    @abstractmethod
    def save(self, session_id: UUID, data: dict[str, Any]) -> bool:
        """
        Save checkpoint data.

        Args:
            session_id: Unique session identifier
            data: Checkpoint data to save

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def load(self, session_id: UUID) -> dict[str, Any] | None:
        """
        Load latest checkpoint data.

        Args:
            session_id: Session identifier

        Returns:
            Checkpoint data or None if not found
        """
        pass

    @abstractmethod
    def list_checkpoints(self) -> list[CheckpointInfo]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint information
        """
        pass

    @abstractmethod
    def delete(self, session_id: UUID) -> bool:
        """
        Delete checkpoint for session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted
        """
        pass

    @abstractmethod
    def exists(self, session_id: UUID) -> bool:
        """
        Check if checkpoint exists.

        Args:
            session_id: Session identifier

        Returns:
            True if exists
        """
        pass


class LocalFileCheckpointStorage(CheckpointStorage):
    """
    Local filesystem checkpoint storage implementation.

    Stores checkpoints as JSON files for human readability and debugging.
    """

    def __init__(
        self,
        checkpoint_dir: Path = Path(".checkpoints"),
        use_json: bool = True,
    ):
        """
        Initialize local file checkpoint storage.

        Args:
            checkpoint_dir: Directory for checkpoints
            use_json: Use JSON format (True) or pickle (False)
        """
        self.checkpoint_dir = checkpoint_dir
        self.use_json = use_json

        # Create directory if doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, session_id: UUID) -> Path:
        """Get checkpoint file path for session."""
        if not self.use_json:
            return self.checkpoint_dir / f"checkpoint_{session_id}.pkl"
        return self.checkpoint_dir / f"checkpoint_{session_id}.json.gz"

    def _get_legacy_checkpoint_path(self, session_id: UUID) -> Path:
        """Get legacy uncompressed path for migration."""
        return self.checkpoint_dir / f"checkpoint_{session_id}.json"

    def save(self, session_id: UUID, data: dict[str, Any]) -> bool:
        """Save checkpoint to local file (gzip-compressed JSON by default)."""
        checkpoint_path = self._get_checkpoint_path(session_id)

        checkpoint_data = {
            "version": "1.0",
            "session_id": str(session_id),
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }

        try:
            if self.use_json:
                payload = json.dumps(checkpoint_data, default=str).encode("utf-8")
                with gzip.open(checkpoint_path, "wb", compresslevel=1) as f:
                    f.write(payload)
            else:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(checkpoint_data, f)

            return True
        except Exception:
            return False

    def load(self, session_id: UUID) -> dict[str, Any] | None:
        """Load checkpoint from local file (supports both gzip and legacy JSON)."""
        checkpoint_path = self._get_checkpoint_path(session_id)

        if not checkpoint_path.exists():
            legacy = self._get_legacy_checkpoint_path(session_id)
            if legacy.exists():
                checkpoint_path = legacy
            else:
                return None

        try:
            if not self.use_json:
                with open(checkpoint_path, "rb") as f:
                    checkpoint_data = pickle.load(f)  # nosec B301
            elif checkpoint_path.suffix == ".gz":
                with gzip.open(checkpoint_path, "rb") as f:
                    checkpoint_data = json.loads(f.read().decode("utf-8"))
            else:
                with open(checkpoint_path) as f:
                    checkpoint_data = json.load(f)

            return checkpoint_data.get("data")  # type: ignore[no-any-return]
        except Exception:
            return None

    def list_checkpoints(self) -> list[CheckpointInfo]:
        """List all checkpoints in directory."""
        checkpoints = []

        if self.use_json:
            patterns = ["*.json.gz", "*.json"]
        else:
            patterns = ["*.pkl"]

        seen: set[UUID] = set()
        for pattern in patterns:
            for checkpoint_file in self.checkpoint_dir.glob(pattern):
                try:
                    stem = checkpoint_file.name
                    session_id_str = stem.split("checkpoint_", 1)[1].split(".")[0]
                    session_id = UUID(session_id_str)
                    if session_id in seen:
                        continue
                    seen.add(session_id)

                    stat = checkpoint_file.stat()

                    data = self.load(session_id)
                    rows_processed = data.get("last_processed_row", 0) if data else 0
                    stage = (
                        str(
                            data.get(
                                "current_stage_index",
                                data.get("current_stage", "unknown"),
                            )
                        )
                        if data
                        else "unknown"
                    )
                    total_rows = data.get("total_rows", 0) if data else 0
                    cost_so_far = (
                        Decimal(str(data.get("accumulated_cost", 0)))
                        if data
                        else Decimal("0")
                    )

                    checkpoints.append(
                        CheckpointInfo(
                            checkpoint_id=session_id,
                            session_id=session_id,
                            timestamp=datetime.fromtimestamp(stat.st_mtime),
                            rows_processed=rows_processed,
                            total_rows=total_rows,
                            cost_so_far=cost_so_far,
                            stage=stage,
                            path=str(checkpoint_file),
                        )
                    )
                except Exception:  # nosec B112
                    continue

        return sorted(checkpoints, key=lambda x: x.timestamp, reverse=True)

    def delete(self, session_id: UUID) -> bool:
        """Delete checkpoint file (both compressed and legacy)."""
        deleted = False
        for path in [
            self._get_checkpoint_path(session_id),
            self._get_legacy_checkpoint_path(session_id),
        ]:
            if path.exists():
                try:
                    path.unlink()
                    deleted = True
                except Exception:  # nosec B112
                    continue
        return deleted

    def exists(self, session_id: UUID) -> bool:
        """Check if checkpoint exists (compressed or legacy)."""
        return (
            self._get_checkpoint_path(session_id).exists()
            or self._get_legacy_checkpoint_path(session_id).exists()
        )

    def cleanup_old_checkpoints(self, days: int = 7) -> int:
        """
        Delete checkpoints older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of checkpoints deleted
        """
        deleted = 0
        cutoff = datetime.now().timestamp() - (days * 86400)

        patterns = ["*.json.gz", "*.json", "*.pkl"]
        for pattern in patterns:
            for checkpoint_file in self.checkpoint_dir.glob(pattern):
                if checkpoint_file.stat().st_mtime < cutoff:
                    try:
                        checkpoint_file.unlink()
                        deleted += 1
                    except Exception:  # nosec B112
                        continue

        return deleted
