"""
Streaming processor with bounded memory and backpressure.

Provides a producer/consumer pattern with bounded queues to prevent
memory overflow when processing large datasets. Natural backpressure
is achieved by blocking the producer when the queue is full.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import polars as pl

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ChunkResult(Generic[T]):
    """Result of processing a single chunk."""

    chunk_index: int
    data: T
    rows_processed: int
    error: Exception | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingStats:
    """Statistics for streaming processing."""

    total_chunks: int = 0
    total_rows: int = 0
    successful_chunks: int = 0
    failed_chunks: int = 0
    current_queue_size: int = 0
    peak_queue_size: int = 0


class StreamingProcessor:
    """
    Process data streams with bounded memory using producer/consumer pattern.

    Uses asyncio.Queue with maxsize to create natural backpressure:
    - When queue is full, producer waits (prevents memory overflow)
    - When queue is empty, consumer waits (prevents busy-waiting)

    Example:
        processor = StreamingProcessor(max_pending_chunks=3)

        async for result in processor.process_stream(
            data_source=loader.stream_chunks(),
            processor=process_chunk,
        ):
            save_result(result)

    Memory Guarantee:
        Peak memory = max_pending_chunks * chunk_size * avg_row_size
        Independent of total dataset size.
    """

    def __init__(
        self,
        max_pending_chunks: int = 3,
        error_policy: str = "skip",  # "skip", "fail", "retry"
    ):
        """
        Initialize streaming processor.

        Args:
            max_pending_chunks: Maximum chunks in queue (controls memory)
            error_policy: How to handle chunk processing errors
        """
        if max_pending_chunks < 1:
            raise ValueError("max_pending_chunks must be at least 1")

        self.max_pending_chunks = max_pending_chunks
        self.error_policy = error_policy
        self._stats = StreamingStats()

    @property
    def stats(self) -> StreamingStats:
        """Get current processing statistics."""
        return self._stats

    async def process_stream(
        self,
        data_source: AsyncIterator[pl.DataFrame],
        processor: Callable[[pl.DataFrame], Awaitable[pl.DataFrame]],
        on_chunk_complete: Callable[[ChunkResult], Awaitable[None]] | None = None,
    ) -> AsyncIterator[pl.DataFrame]:
        """
        Process data stream with bounded memory.

        Args:
            data_source: Async iterator yielding DataFrame chunks
            processor: Async function to process each chunk
            on_chunk_complete: Optional callback after each chunk

        Yields:
            Processed DataFrame chunks

        Note:
            Backpressure is automatic - if consumer is slow, producer waits.
        """
        # Reset stats
        self._stats = StreamingStats()

        # Create bounded queue for backpressure
        queue: asyncio.Queue[pl.DataFrame | None] = asyncio.Queue(
            maxsize=self.max_pending_chunks
        )

        # Track producer task
        producer_task: asyncio.Task | None = None
        producer_error: Exception | None = None

        async def producer():
            """Load chunks into queue (blocks when full = backpressure)."""
            nonlocal producer_error
            chunk_index = 0

            try:
                async for chunk in data_source:
                    # This blocks when queue is full (backpressure!)
                    await queue.put(chunk)

                    # Update stats
                    self._stats.current_queue_size = queue.qsize()
                    self._stats.peak_queue_size = max(
                        self._stats.peak_queue_size, queue.qsize()
                    )

                    logger.debug(
                        f"Producer: queued chunk {chunk_index}, "
                        f"queue size: {queue.qsize()}/{self.max_pending_chunks}"
                    )
                    chunk_index += 1

            except Exception as e:
                producer_error = e
                logger.error(f"Producer error: {e}")

            finally:
                # Signal end of stream
                await queue.put(None)
                logger.debug(f"Producer: finished, total chunks: {chunk_index}")

        # Start producer
        producer_task = asyncio.create_task(producer())

        try:
            chunk_index = 0

            while True:
                # Get next chunk (blocks when empty)
                chunk = await queue.get()

                if chunk is None:
                    # End of stream
                    break

                # Update queue stats
                self._stats.current_queue_size = queue.qsize()
                self._stats.total_chunks += 1

                try:
                    # Process chunk
                    logger.debug(
                        f"Consumer: processing chunk {chunk_index}, rows: {len(chunk)}"
                    )

                    result_df = await processor(chunk)
                    self._stats.total_rows += len(chunk)
                    self._stats.successful_chunks += 1

                    # Callback if provided
                    if on_chunk_complete:
                        chunk_result = ChunkResult(
                            chunk_index=chunk_index,
                            data=result_df,
                            rows_processed=len(chunk),
                        )
                        await on_chunk_complete(chunk_result)

                    yield result_df

                except Exception as e:
                    self._stats.failed_chunks += 1
                    logger.error(f"Error processing chunk {chunk_index}: {e}")

                    if self.error_policy == "fail":
                        raise
                    elif self.error_policy == "skip":
                        # Callback with error
                        if on_chunk_complete:
                            chunk_result = ChunkResult(
                                chunk_index=chunk_index,
                                data=chunk,  # Original chunk
                                rows_processed=0,
                                error=e,
                            )
                            await on_chunk_complete(chunk_result)
                        continue
                    # "retry" would need more complex logic

                finally:
                    chunk_index += 1

        finally:
            # Cancel producer if still running
            if producer_task and not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass

            # Check for producer errors
            if producer_error:
                logger.error(f"Producer failed: {producer_error}")

        logger.info(
            f"Streaming complete: {self._stats.successful_chunks} chunks, "
            f"{self._stats.total_rows} rows, "
            f"{self._stats.failed_chunks} failures"
        )

    async def process_with_concurrency(
        self,
        data_source: AsyncIterator[pl.DataFrame],
        processor: Callable[[pl.DataFrame], Awaitable[pl.DataFrame]],
        concurrency: int = 1,
    ) -> AsyncIterator[pl.DataFrame]:
        """
        Process chunks with concurrent processing within each chunk.

        This is useful when the processor itself can benefit from
        parallelism (e.g., multiple LLM calls per chunk).

        Args:
            data_source: Async iterator yielding DataFrame chunks
            processor: Async function to process each chunk
            concurrency: Number of concurrent processors

        Yields:
            Processed DataFrame chunks (may be out of order if concurrency > 1)
        """
        if concurrency == 1:
            # Simple case - just use regular processing
            async for result in self.process_stream(data_source, processor):
                yield result
            return

        # For concurrent processing, use a semaphore
        semaphore = asyncio.Semaphore(concurrency)
        pending: list[asyncio.Task] = []
        results: asyncio.Queue[pl.DataFrame | None] = asyncio.Queue()

        async def process_one(chunk: pl.DataFrame, index: int):
            async with semaphore:
                try:
                    result = await processor(chunk)
                    await results.put(result)
                except Exception as e:
                    logger.error(f"Error processing chunk {index}: {e}")
                    if self.error_policy == "fail":
                        await results.put(None)  # Signal error
                        raise

        async def producer():
            index = 0
            async for chunk in data_source:
                task = asyncio.create_task(process_one(chunk, index))
                pending.append(task)
                index += 1

            # Wait for all tasks
            await asyncio.gather(*pending, return_exceptions=True)
            await results.put(None)  # Signal end

        producer_task = asyncio.create_task(producer())

        try:
            while True:
                result = await results.get()
                if result is None:
                    break
                yield result
        finally:
            if not producer_task.done():
                producer_task.cancel()

    def __repr__(self) -> str:
        return (
            f"StreamingProcessor(max_pending_chunks={self.max_pending_chunks}, "
            f"error_policy={self.error_policy!r})"
        )
