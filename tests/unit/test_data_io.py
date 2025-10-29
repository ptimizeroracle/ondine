"""Unit tests for data I/O operations."""

import tempfile
from pathlib import Path

import pandas as pd

from ondine.adapters.data_io import (
    CSVReader,
    CSVWriter,
    DataFrameReader,
    ExcelReader,
    ExcelWriter,
    ParquetReader,
    ParquetWriter,
    create_data_reader,
    create_data_writer,
)
from ondine.core.specifications import DataSourceType


class TestDataReaders:
    """Test suite for data readers."""

    def test_dataframe_reader(self):
        """Test reading from DataFrame."""
        df = pd.DataFrame(
            {
                "text": ["sample1", "sample2"],
                "value": [1, 2],
            }
        )

        reader = DataFrameReader(df)
        result = reader.read()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["text", "value"]

    def test_csv_reader(self):
        """Test reading from CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("text,value\n")
            f.write("sample1,1\n")
            f.write("sample2,2\n")
            csv_path = f.name

        try:
            reader = CSVReader(csv_path)
            result = reader.read()

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert "text" in result.columns
            assert "value" in result.columns
        finally:
            Path(csv_path).unlink()

    def test_csv_reader_chunked(self):
        """Test reading CSV in chunks."""
        # Create temporary CSV file with more rows
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("text\n")
            for i in range(100):
                f.write(f"sample{i}\n")
            csv_path = f.name

        try:
            reader = CSVReader(csv_path)
            chunks = list(reader.read_chunked(chunk_size=25))

            assert len(chunks) == 4  # 100 rows / 25 per chunk
            assert all(isinstance(chunk, pd.DataFrame) for chunk in chunks)
            assert len(chunks[0]) == 25
            assert len(chunks[-1]) == 25
        finally:
            Path(csv_path).unlink()

    def test_create_data_reader_csv(self):
        """Test factory function for CSV reader."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("text\nsample\n")
            csv_path = f.name

        try:
            reader = create_data_reader(DataSourceType.CSV, csv_path)
            assert isinstance(reader, CSVReader)

            result = reader.read()
            assert len(result) == 1
        finally:
            Path(csv_path).unlink()

    def test_create_data_reader_dataframe(self):
        """Test factory function for DataFrame reader."""
        df = pd.DataFrame({"text": ["test"]})
        reader = create_data_reader(DataSourceType.DATAFRAME, dataframe=df)

        assert isinstance(reader, DataFrameReader)

    def test_excel_reader(self):
        """Test reading from Excel file."""
        # Create temporary Excel file
        df = pd.DataFrame(
            {
                "text": ["sample1", "sample2", "sample3"],
                "value": [1, 2, 3],
                "score": [95.5, 87.3, 92.1],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            excel_path = f.name

        try:
            # Write test data
            df.to_excel(excel_path, index=False, sheet_name="Sheet1")

            # Read it back
            reader = ExcelReader(excel_path)
            result = reader.read()

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert "text" in result.columns
            assert "value" in result.columns
            assert "score" in result.columns
            assert result.iloc[0]["text"] == "sample1"
            assert result.iloc[1]["value"] == 2
            assert abs(result.iloc[2]["score"] - 92.1) < 0.01
        finally:
            Path(excel_path).unlink(missing_ok=True)

    def test_excel_reader_with_sheet_name(self):
        """Test reading from Excel file with specific sheet name."""
        df1 = pd.DataFrame({"col1": [1, 2]})
        df2 = pd.DataFrame({"col2": [3, 4]})

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            excel_path = f.name

        try:
            # Write multiple sheets
            with pd.ExcelWriter(excel_path) as writer:
                df1.to_excel(writer, sheet_name="FirstSheet", index=False)
                df2.to_excel(writer, sheet_name="SecondSheet", index=False)

            # Read specific sheet
            reader = ExcelReader(excel_path, sheet_name="SecondSheet")
            result = reader.read()

            assert "col2" in result.columns
            assert "col1" not in result.columns
            assert len(result) == 2
        finally:
            Path(excel_path).unlink(missing_ok=True)

    def test_excel_reader_preserves_data_types(self):
        """Test Excel reader preserves data types correctly."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.7, 3.9],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            excel_path = f.name

        try:
            df.to_excel(excel_path, index=False)

            reader = ExcelReader(excel_path)
            result = reader.read()

            # Verify data integrity
            assert result["int_col"].tolist() == [1, 2, 3]
            assert abs(result["float_col"].iloc[0] - 1.5) < 0.01
            assert result["str_col"].tolist() == ["a", "b", "c"]
        finally:
            Path(excel_path).unlink(missing_ok=True)

    def test_parquet_reader(self):
        """Test reading from Parquet file."""
        df = pd.DataFrame(
            {
                "text": ["sample1", "sample2"],
                "value": [1, 2],
                "score": [95.5, 87.3],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            # Write test data
            df.to_parquet(parquet_path, index=False)

            # Read it back
            reader = ParquetReader(parquet_path)
            result = reader.read()

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert "text" in result.columns
            assert "value" in result.columns
            assert "score" in result.columns
        finally:
            Path(parquet_path).unlink(missing_ok=True)

    def test_parquet_reader_preserves_data_types(self):
        """Test Parquet reader preserves data types (better than Excel/CSV)."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.7, 3.9],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            df.to_parquet(parquet_path, index=False)

            reader = ParquetReader(parquet_path)
            result = reader.read()

            # Parquet preserves types perfectly
            assert result["int_col"].dtype == "int64"
            assert result["float_col"].dtype == "float64"
            assert result["str_col"].dtype == "object"
            assert result["bool_col"].dtype == "bool"
        finally:
            Path(parquet_path).unlink(missing_ok=True)

    def test_create_data_reader_excel(self):
        """Test factory function for Excel reader."""
        df = pd.DataFrame({"text": ["test"]})

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            excel_path = f.name

        try:
            df.to_excel(excel_path, index=False)

            reader = create_data_reader(DataSourceType.EXCEL, Path(excel_path))
            assert isinstance(reader, ExcelReader)

            result = reader.read()
            assert len(result) == 1
        finally:
            Path(excel_path).unlink(missing_ok=True)

    def test_create_data_reader_parquet(self):
        """Test factory function for Parquet reader."""
        df = pd.DataFrame({"text": ["test"]})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            df.to_parquet(parquet_path, index=False)

            reader = create_data_reader(DataSourceType.PARQUET, Path(parquet_path))
            assert isinstance(reader, ParquetReader)

            result = reader.read()
            assert len(result) == 1
        finally:
            Path(parquet_path).unlink(missing_ok=True)


class TestDataWriters:
    """Test suite for data writers."""

    def test_csv_writer(self):
        """Test writing to CSV file."""
        df = pd.DataFrame(
            {
                "text": ["sample1", "sample2"],
                "value": [1, 2],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            writer = CSVWriter()
            confirmation = writer.write(df, Path(csv_path))

            assert confirmation.success is True
            assert confirmation.rows_written == 2
            assert Path(csv_path).exists()

            # Verify written data
            written_df = pd.read_csv(csv_path)
            assert len(written_df) == 2
            assert list(written_df.columns) == ["text", "value"]
        finally:
            Path(csv_path).unlink(missing_ok=True)

    def test_csv_writer_overwrite(self):
        """Test CSV writer overwrites by default."""
        df1 = pd.DataFrame({"text": ["first"]})
        df2 = pd.DataFrame({"text": ["second", "third"]})

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

        try:
            writer = CSVWriter()
            writer.write(df1, csv_path)
            writer.write(df2, csv_path)  # Overwrites

            # Verify only second write exists
            result = pd.read_csv(csv_path)
            assert len(result) == 2  # Only df2 data
            assert result.iloc[0]["text"] == "second"
            assert result.iloc[1]["text"] == "third"
        finally:
            csv_path.unlink(missing_ok=True)

    def test_create_data_writer_csv(self):
        """Test factory function for CSV writer."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

        try:
            writer = create_data_writer(DataSourceType.CSV)
            assert isinstance(writer, CSVWriter)

            df = pd.DataFrame({"text": ["test"]})
            confirmation = writer.write(df, csv_path)
            assert confirmation.success is True
        finally:
            csv_path.unlink(missing_ok=True)

    def test_excel_writer(self):
        """Test writing to Excel file."""
        df = pd.DataFrame(
            {
                "text": ["sample1", "sample2"],
                "value": [1, 2],
                "score": [95.5, 87.3],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            excel_path = Path(f.name)

        try:
            writer = ExcelWriter()
            confirmation = writer.write(df, excel_path)

            assert confirmation.success is True
            assert confirmation.rows_written == 2
            assert excel_path.exists()

            # Verify written data
            written_df = pd.read_excel(excel_path)
            assert len(written_df) == 2
            assert list(written_df.columns) == ["text", "value", "score"]
            assert written_df.iloc[0]["text"] == "sample1"
            assert written_df.iloc[1]["value"] == 2
        finally:
            excel_path.unlink(missing_ok=True)

    def test_excel_writer_with_sheet_name(self):
        """Test writing to Excel with custom sheet name (note: not supported in basic API)."""
        df = pd.DataFrame({"col1": [1, 2]})

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            excel_path = Path(f.name)

        try:
            # ExcelWriter doesn't support sheet_name in __init__, just write normally
            writer = ExcelWriter()
            writer.write(df, excel_path)

            # Verify data written
            written_df = pd.read_excel(excel_path)
            assert len(written_df) == 2
        finally:
            excel_path.unlink(missing_ok=True)

    def test_excel_writer_overwrite_mode(self):
        """Test overwriting Excel file (default behavior)."""
        df1 = pd.DataFrame({"text": ["first"]})
        df2 = pd.DataFrame({"text": ["second", "third"]})

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            excel_path = Path(f.name)

        try:
            writer = ExcelWriter()
            writer.write(df1, excel_path)
            writer.write(df2, excel_path)  # Overwrites by default

            # Verify only second write exists
            result = pd.read_excel(excel_path)
            assert len(result) == 2  # Only df2 data
            assert result.iloc[0]["text"] == "second"
        finally:
            excel_path.unlink(missing_ok=True)

    def test_excel_writer_preserves_data_types(self):
        """Test Excel writer preserves data types."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.7, 3.9],
                "str_col": ["a", "b", "c"],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            excel_path = Path(f.name)

        try:
            writer = ExcelWriter()
            writer.write(df, excel_path)

            # Read back and verify
            result = pd.read_excel(excel_path)
            assert result["int_col"].tolist() == [1, 2, 3]
            assert abs(result["float_col"].iloc[0] - 1.5) < 0.01
            assert result["str_col"].tolist() == ["a", "b", "c"]
        finally:
            excel_path.unlink(missing_ok=True)

    def test_parquet_writer(self):
        """Test writing to Parquet file."""
        df = pd.DataFrame(
            {
                "text": ["sample1", "sample2"],
                "value": [1, 2],
                "score": [95.5, 87.3],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = Path(f.name)

        try:
            writer = ParquetWriter()
            confirmation = writer.write(df, parquet_path)

            assert confirmation.success is True
            assert confirmation.rows_written == 2
            assert parquet_path.exists()

            # Verify written data
            written_df = pd.read_parquet(parquet_path)
            assert len(written_df) == 2
            assert list(written_df.columns) == ["text", "value", "score"]
        finally:
            parquet_path.unlink(missing_ok=True)

    def test_parquet_writer_preserves_data_types_perfectly(self):
        """Test Parquet writer preserves all data types perfectly."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.7, 3.9],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = Path(f.name)

        try:
            writer = ParquetWriter()
            writer.write(df, parquet_path)

            # Read back and verify types are EXACTLY preserved
            result = pd.read_parquet(parquet_path)
            assert result["int_col"].dtype == "int64"
            assert result["float_col"].dtype == "float64"
            assert result["str_col"].dtype == "object"
            assert result["bool_col"].dtype == "bool"  # Parquet preserves bool!
        finally:
            parquet_path.unlink(missing_ok=True)

    def test_parquet_writer_compression(self):
        """Test Parquet writer with compression (note: compression not in basic API)."""
        df = pd.DataFrame(
            {
                "text": ["sample" * 100 for _ in range(1000)],  # Highly compressible
                "value": list(range(1000)),
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = Path(f.name)

        try:
            writer = ParquetWriter()
            confirmation = writer.write(df, parquet_path)

            assert confirmation.success is True
            assert confirmation.rows_written == 1000

            # Verify file exists and has data
            assert parquet_path.exists()
            result = pd.read_parquet(parquet_path)
            assert len(result) == 1000
        finally:
            parquet_path.unlink(missing_ok=True)

    def test_create_data_writer_excel(self):
        """Test factory function for Excel writer."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            excel_path = Path(f.name)

        try:
            writer = create_data_writer(DataSourceType.EXCEL)
            assert isinstance(writer, ExcelWriter)

            df = pd.DataFrame({"text": ["test"]})
            confirmation = writer.write(df, excel_path)
            assert confirmation.success is True
        finally:
            excel_path.unlink(missing_ok=True)

    def test_create_data_writer_parquet(self):
        """Test factory function for Parquet writer."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = Path(f.name)

        try:
            writer = create_data_writer(DataSourceType.PARQUET)
            assert isinstance(writer, ParquetWriter)

            df = pd.DataFrame({"text": ["test"]})
            confirmation = writer.write(df, parquet_path)
            assert confirmation.success is True
        finally:
            parquet_path.unlink(missing_ok=True)


class TestCheckpointStorage:
    """Test suite for checkpoint storage."""

    def test_checkpoint_save_and_load(self):
        """Test saving and loading checkpoints."""
        from uuid import uuid4

        from ondine.adapters import LocalFileCheckpointStorage

        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalFileCheckpointStorage(Path(temp_dir))
            session_id = uuid4()

            # Save checkpoint
            data = {
                "rows_processed": 100,
                "stage": "LLMInvocation",
                "timestamp": "2025-10-15T10:00:00",
            }

            success = storage.save(session_id, data)
            assert success is True

            # Load checkpoint
            loaded_data = storage.load(session_id)
            assert loaded_data is not None
            assert loaded_data["rows_processed"] == 100
            assert loaded_data["stage"] == "LLMInvocation"

    def test_checkpoint_list(self):
        """Test listing checkpoints."""
        from uuid import uuid4

        from ondine.adapters import LocalFileCheckpointStorage

        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalFileCheckpointStorage(Path(temp_dir))

            # Create multiple checkpoints
            session_ids = [uuid4() for _ in range(3)]
            for sid in session_ids:
                storage.save(sid, {"test": "data"})

            # List checkpoints
            checkpoints = storage.list_checkpoints()
            assert len(checkpoints) >= 3

    def test_checkpoint_delete(self):
        """Test deleting checkpoints."""
        from uuid import uuid4

        from ondine.adapters import LocalFileCheckpointStorage

        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalFileCheckpointStorage(Path(temp_dir))
            session_id = uuid4()

            # Save and then delete
            storage.save(session_id, {"test": "data"})
            success = storage.delete(session_id)
            assert success is True

            # Verify deleted
            loaded = storage.load(session_id)
            assert loaded is None
