"""Tests for BatchProcessor."""

from decimal import Decimal

from ondine.core.models import LLMResponse, PromptBatch, RowMetadata
from ondine.stages.batch_processor import BatchMap, BatchProcessor, PromptItem


class TestPromptItem:
    """Test PromptItem dataclass."""

    def test_create_prompt_item(self):
        """Test creating a PromptItem."""
        metadata = RowMetadata(row_index=0)
        item = PromptItem(prompt="Hello", metadata=metadata, batch_id="batch-1")

        assert item.prompt == "Hello"
        assert item.metadata == metadata
        assert item.batch_id == "batch-1"


class TestBatchMap:
    """Test BatchMap dataclass."""

    def test_add_and_get(self):
        """Test adding and getting mappings."""
        batch_map = BatchMap()
        batch_map.add(0, 0)
        batch_map.add(0, 1)
        batch_map.add(1, 0)

        assert batch_map.get(0) == (0, 0)
        assert batch_map.get(1) == (0, 1)
        assert batch_map.get(2) == (1, 0)

    def test_len(self):
        """Test length."""
        batch_map = BatchMap()
        assert len(batch_map) == 0

        batch_map.add(0, 0)
        batch_map.add(0, 1)
        assert len(batch_map) == 2

    def test_iter(self):
        """Test iteration."""
        batch_map = BatchMap()
        batch_map.add(0, 0)
        batch_map.add(1, 0)

        items = list(batch_map)
        assert items == [(0, 0), (1, 0)]


class TestBatchProcessor:
    """Test BatchProcessor functionality."""

    def test_flatten_single_batch(self):
        """Test flattening a single batch."""
        batch = PromptBatch(
            prompts=["Hello", "World"],
            metadata=[RowMetadata(row_index=0), RowMetadata(row_index=1)],
            batch_id="batch-1",
        )

        items, batch_map = BatchProcessor.flatten([batch])

        assert len(items) == 2
        assert items[0].prompt == "Hello"
        assert items[1].prompt == "World"
        assert len(batch_map) == 2
        assert batch_map.get(0) == (0, 0)
        assert batch_map.get(1) == (0, 1)

    def test_flatten_multiple_batches(self):
        """Test flattening multiple batches."""
        batch1 = PromptBatch(
            prompts=["A", "B"],
            metadata=[RowMetadata(row_index=0), RowMetadata(row_index=1)],
            batch_id="batch-1",
        )
        batch2 = PromptBatch(
            prompts=["C"],
            metadata=[RowMetadata(row_index=2)],
            batch_id="batch-2",
        )

        items, batch_map = BatchProcessor.flatten([batch1, batch2])

        assert len(items) == 3
        assert [i.prompt for i in items] == ["A", "B", "C"]
        assert batch_map.get(0) == (0, 0)  # A is batch 0, prompt 0
        assert batch_map.get(1) == (0, 1)  # B is batch 0, prompt 1
        assert batch_map.get(2) == (1, 0)  # C is batch 1, prompt 0

    def test_reconstruct_preserves_order(self):
        """Test reconstruction preserves original order."""
        batch1 = PromptBatch(
            prompts=["A", "B"],
            metadata=[RowMetadata(row_index=0), RowMetadata(row_index=1)],
            batch_id="batch-1",
        )
        batch2 = PromptBatch(
            prompts=["C"],
            metadata=[RowMetadata(row_index=2)],
            batch_id="batch-2",
        )

        items, batch_map = BatchProcessor.flatten([batch1, batch2])

        # Create responses in same order
        responses = [
            LLMResponse(
                text=f"Response {item.prompt}",
                tokens_in=10,
                tokens_out=5,
                model="test",
                cost=Decimal("0.01"),
                latency_ms=100.0,
            )
            for item in items
        ]

        response_batches = BatchProcessor.reconstruct(
            responses, [batch1, batch2], batch_map
        )

        assert len(response_batches) == 2
        # Now responses are LLMResponse objects, so we check their text
        assert [r.text for r in response_batches[0].responses] == [
            "Response A",
            "Response B",
        ]
        assert [r.text for r in response_batches[1].responses] == ["Response C"]
        assert response_batches[0].batch_id == "batch-1"
        assert response_batches[1].batch_id == "batch-2"

    def test_reconstruct_calculates_metrics(self):
        """Test reconstruction calculates batch metrics."""
        batch = PromptBatch(
            prompts=["A", "B"],
            metadata=[RowMetadata(row_index=0), RowMetadata(row_index=1)],
            batch_id="batch-1",
        )

        items, batch_map = BatchProcessor.flatten([batch])

        responses = [
            LLMResponse(
                text="R1",
                tokens_in=10,
                tokens_out=5,
                model="test",
                cost=Decimal("0.01"),
                latency_ms=100.0,
            ),
            LLMResponse(
                text="R2",
                tokens_in=20,
                tokens_out=10,
                model="test",
                cost=Decimal("0.02"),
                latency_ms=200.0,
            ),
        ]

        response_batches = BatchProcessor.reconstruct(responses, [batch], batch_map)

        assert response_batches[0].tokens_used == 45  # 10+5+20+10
        assert response_batches[0].cost == Decimal("0.03")
        assert response_batches[0].latencies_ms == [100.0, 200.0]

    def test_calculate_total_rows_simple(self):
        """Test calculating total rows for simple batches."""
        batch1 = PromptBatch(
            prompts=["A", "B"],
            metadata=[RowMetadata(row_index=0), RowMetadata(row_index=1)],
            batch_id="batch-1",
        )
        batch2 = PromptBatch(
            prompts=["C"],
            metadata=[RowMetadata(row_index=2)],
            batch_id="batch-2",
        )

        total = BatchProcessor.calculate_total_rows([batch1, batch2])
        assert total == 3

    def test_calculate_total_rows_aggregated(self):
        """Test calculating total rows for aggregated batches."""
        # Aggregated batch: 1 prompt represents 10 rows
        batch = PromptBatch(
            prompts=["Aggregated prompt"],
            metadata=[
                RowMetadata(row_index=0, custom={"is_batch": True, "batch_size": 10})
            ],
            batch_id="batch-1",
        )

        total = BatchProcessor.calculate_total_rows([batch])
        assert total == 10

    def test_get_batch_size_simple(self):
        """Test getting batch size for simple row."""
        metadata = RowMetadata(row_index=0)
        assert BatchProcessor.get_batch_size(metadata) == 1

    def test_get_batch_size_aggregated(self):
        """Test getting batch size for aggregated row."""
        metadata = RowMetadata(row_index=0, custom={"is_batch": True, "batch_size": 25})
        assert BatchProcessor.get_batch_size(metadata) == 25

    def test_get_batch_size_aggregated_no_size(self):
        """Test getting batch size for aggregated row without explicit size."""
        metadata = RowMetadata(row_index=0, custom={"is_batch": True})
        assert BatchProcessor.get_batch_size(metadata) == 1  # Default
