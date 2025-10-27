"""
Checkpoint storage for fault tolerance.

Provides persistent storage of execution state to enable resume after
failures.
"""

import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
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
        ext = ".json" if self.use_json else ".pkl"
        return self.checkpoint_dir / f"checkpoint_{session_id}{ext}"

    def save(self, session_id: UUID, data: dict[str, Any]) -> bool:
        """Save checkpoint to local file."""
        checkpoint_path = self._get_checkpoint_path(session_id)

        # Add metadata
        checkpoint_data = {
            "version": "1.0",
            "session_id": str(session_id),
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }

        try:
            if self.use_json:
                with open(checkpoint_path, "w") as f:
                    json.dump(
                        checkpoint_data,
                        f,
                        indent=2,
                        default=str,  # Handle non-serializable types
                    )
            else:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(checkpoint_data, f)

            return True
        except Exception:
            return False

    def load(self, session_id: UUID) -> dict[str, Any] | None:
        """Load checkpoint from local file."""
        checkpoint_path = self._get_checkpoint_path(session_id)

        if not checkpoint_path.exists():
            return None

        try:
            if self.use_json:
                with open(checkpoint_path) as f:
                    checkpoint_data = json.load(f)
            else:
                with open(checkpoint_path, "rb") as f:
                    checkpoint_data = pickle.load(f)

            return checkpoint_data.get("data")
        except Exception:
            return None

    def list_checkpoints(self) -> list[CheckpointInfo]:
        """List all checkpoints in directory."""
        checkpoints = []

        pattern = "*.json" if self.use_json else "*.pkl"
        for checkpoint_file in self.checkpoint_dir.glob(pattern):
            try:
                # Extract session ID from filename
                session_id_str = checkpoint_file.stem.replace("checkpoint_", "")
                session_id = UUID(session_id_str)

                # Get file stats
                stat = checkpoint_file.stat()

                # Try to load checkpoint for additional info
                data = self.load(session_id)
                row_index = data.get("last_processed_row", 0) if data else 0
                stage_index = data.get("current_stage_index", 0) if data else 0

                checkpoints.append(
                    CheckpointInfo(
                        session_id=session_id,
                        checkpoint_path=str(checkpoint_file),
                        row_index=row_index,
                        stage_index=stage_index,
                        timestamp=datetime.fromtimestamp(stat.st_mtime),
                        size_bytes=stat.st_size,
                    )
                )
            except Exception:  # nosec B112
                # Skip invalid checkpoint files
                continue

        return sorted(checkpoints, key=lambda x: x.timestamp, reverse=True)

    def delete(self, session_id: UUID) -> bool:
        """Delete checkpoint file."""
        checkpoint_path = self._get_checkpoint_path(session_id)

        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                return True
            except Exception:
                return False
        return False

    def exists(self, session_id: UUID) -> bool:
        """Check if checkpoint exists."""
        return self._get_checkpoint_path(session_id).exists()

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

        pattern = "*.json" if self.use_json else "*.pkl"
        for checkpoint_file in self.checkpoint_dir.glob(pattern):
            if checkpoint_file.stat().st_mtime < cutoff:
                try:
                    checkpoint_file.unlink()
                    deleted += 1
                except Exception:  # nosec B112
                    continue

        return deleted
