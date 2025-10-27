"""State management for checkpointing and recovery."""

from uuid import UUID

from ondine.adapters.checkpoint_storage import CheckpointStorage
from ondine.core.models import CheckpointInfo
from ondine.orchestration.execution_context import ExecutionContext
from ondine.utils import get_logger

logger = get_logger(__name__)


class StateManager:
    """
    Manages execution state persistence and recovery.

    Follows Single Responsibility: only handles state management.
    Uses Strategy pattern for pluggable storage backends.
    """

    def __init__(self, storage: CheckpointStorage, checkpoint_interval: int = 500):
        """
        Initialize state manager.

        Args:
            storage: Checkpoint storage backend
            checkpoint_interval: Rows between checkpoints
        """
        self.storage = storage
        self.checkpoint_interval = checkpoint_interval
        self._last_checkpoint_row = 0

    def should_checkpoint(self, current_row: int) -> bool:
        """
        Check if checkpoint should be saved.

        Args:
            current_row: Current row index

        Returns:
            True if checkpoint due
        """
        return (current_row - self._last_checkpoint_row) >= self.checkpoint_interval

    def save_checkpoint(self, context: ExecutionContext) -> bool:
        """
        Save checkpoint for execution context.

        Args:
            context: Execution context to save

        Returns:
            True if successful
        """
        try:
            checkpoint_data = context.to_checkpoint()
            success = self.storage.save(context.session_id, checkpoint_data)

            if success:
                self._last_checkpoint_row = context.last_processed_row
                logger.info(f"Checkpoint saved at row {context.last_processed_row}")

            return success
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def load_checkpoint(self, session_id: UUID) -> ExecutionContext | None:
        """
        Load checkpoint for session.

        Args:
            session_id: Session identifier

        Returns:
            Restored execution context or None
        """
        try:
            checkpoint_data = self.storage.load(session_id)

            if checkpoint_data is None:
                return None

            context = ExecutionContext.from_checkpoint(checkpoint_data)
            logger.info(f"Checkpoint loaded from row {context.last_processed_row}")

            return context
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def can_resume(self, session_id: UUID) -> bool:
        """
        Check if session can be resumed.

        Args:
            session_id: Session identifier

        Returns:
            True if checkpoint exists
        """
        return self.storage.exists(session_id)

    def cleanup_checkpoints(self, session_id: UUID) -> bool:
        """
        Delete checkpoints for session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted
        """
        try:
            success = self.storage.delete(session_id)
            if success:
                logger.info(f"Checkpoints cleaned up for session {session_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints: {e}")
            return False

    def list_checkpoints(self) -> list[CheckpointInfo]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint information
        """
        return self.storage.list_checkpoints()
