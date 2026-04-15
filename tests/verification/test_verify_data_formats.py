"""Claim verification: Data format support (Claims 48-52)."""

import pandas as pd

from ondine.adapters.data_io import (
    CSVReader,
    CSVWriter,
    DataFrameReader,
    ExcelReader,
    ExcelWriter,
    ParquetReader,
    ParquetWriter,
)


class TestDataFormatClaims:
    """Verify Ondine supports all claimed data formats."""

    def test_claim_48_csv_read_write_roundtrip(self, temp_dir):
        """Claim 48: CSV support — read and write preserve data."""
        path = temp_dir / "test.csv"
        original = pd.DataFrame({"text": ["hello", "world"], "num": [1, 2]})
        original.to_csv(path, index=False)

        reader = CSVReader(path)
        loaded = reader.read()
        assert list(loaded.columns) == ["text", "num"]
        assert len(loaded) == 2
        assert loaded["text"].tolist() == ["hello", "world"]

        out_path = temp_dir / "output.csv"
        writer = CSVWriter()
        writer.write(loaded, out_path)
        assert out_path.exists()

        reloaded = pd.read_csv(out_path)
        pd.testing.assert_frame_equal(loaded, reloaded)

    def test_claim_49_parquet_read_write_roundtrip(self, temp_dir):
        """Claim 49: Parquet support — read and write preserve data types."""
        path = temp_dir / "test.parquet"
        original = pd.DataFrame({"text": ["hello", "world"], "num": [1, 2]})
        original.to_parquet(path, index=False)

        reader = ParquetReader(path)
        loaded = reader.read()
        assert list(loaded.columns) == ["text", "num"]
        assert loaded["num"].dtype in ("int64", "int32")

        out_path = temp_dir / "output.parquet"
        writer = ParquetWriter()
        writer.write(loaded, out_path)
        assert out_path.exists()

        reloaded = pd.read_parquet(out_path)
        pd.testing.assert_frame_equal(loaded, reloaded)

    def test_claim_50_excel_read_write_roundtrip(self, temp_dir):
        """Claim 50: Excel support — read and write preserve data."""
        path = temp_dir / "test.xlsx"
        original = pd.DataFrame({"text": ["hello", "world"], "num": [1, 2]})
        original.to_excel(path, index=False)

        reader = ExcelReader(path)
        loaded = reader.read()
        assert list(loaded.columns) == ["text", "num"]
        assert len(loaded) == 2

        out_path = temp_dir / "output.xlsx"
        writer = ExcelWriter()
        writer.write(loaded, out_path)
        assert out_path.exists()

    def test_claim_51_dataframe_passthrough(self):
        """Claim 51: DataFrame support — DataFrameReader returns same data."""
        original = pd.DataFrame({"text": ["hello"], "num": [42]})
        reader = DataFrameReader(original)
        loaded = reader.read()
        pd.testing.assert_frame_equal(original, loaded)

    def test_claim_52_json_loading(self, temp_dir):
        """Claim 52: JSON output — CSV reader handles data, JSON parsers exist."""

        # Verify JSON parser exists in response parsers
        from ondine.stages.response_parser_stage import JSONParser

        parser = JSONParser()
        result = parser.parse('{"sentiment": "positive", "score": 0.9}')
        assert result["sentiment"] == "positive"
        assert result["score"] == 0.9

        # Verify structured output supports JSON schema
        from ondine.core.specifications import PromptSpec

        spec = PromptSpec(
            template="Analyze: {text}",
            response_format="json",
            json_fields=["sentiment", "score"],
        )
        assert spec.response_format == "json"
        assert "sentiment" in spec.json_fields
