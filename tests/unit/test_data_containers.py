"""Unit tests for DataContainer implementations."""

from pathlib import Path

import pytest

from ondine.adapters.containers import (
    DictListContainer,
    PandasContainer,
    PolarsContainer,
    ResultContainerImpl,
    StreamingCSVContainer,
)
from ondine.core.data_container import DataContainer, Row


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a sample CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    csv_path.write_text("id,name,value\n1,Alice,100\n2,Bob,200\n3,Charlie,300\n")
    return csv_path


@pytest.fixture
def sample_rows() -> list[Row]:
    """Sample row data for testing."""
    return [
        {"id": 1, "name": "Alice", "value": 100},
        {"id": 2, "name": "Bob", "value": 200},
        {"id": 3, "name": "Charlie", "value": 300},
    ]


class TestDictListContainer:
    """Tests for DictListContainer."""

    def test_init_empty(self):
        """Test empty initialization."""
        container = DictListContainer()
        assert len(container) == 0
        assert container.columns == []

    def test_init_with_data(self, sample_rows: list[Row]):
        """Test initialization with data."""
        container = DictListContainer(sample_rows)
        assert len(container) == 3
        assert container.columns == ["id", "name", "value"]

    def test_iteration(self, sample_rows: list[Row]):
        """Test iteration over rows."""
        container = DictListContainer(sample_rows)
        rows = list(container)
        assert len(rows) == 3
        assert rows[0]["name"] == "Alice"

    def test_getitem(self, sample_rows: list[Row]):
        """Test indexing."""
        container = DictListContainer(sample_rows)
        assert container[0]["name"] == "Alice"
        assert container[2]["name"] == "Charlie"

    def test_append(self):
        """Test appending rows."""
        container = DictListContainer()
        container.append({"id": 1, "name": "Test"})
        assert len(container) == 1
        assert container[0]["name"] == "Test"

    def test_extend(self, sample_rows: list[Row]):
        """Test extending with multiple rows."""
        container = DictListContainer()
        container.extend(sample_rows)
        assert len(container) == 3

    def test_select(self, sample_rows: list[Row]):
        """Test column selection."""
        container = DictListContainer(sample_rows)
        selected = container.select(["id", "name"])
        assert selected.columns == ["id", "name"]
        assert "value" not in list(selected)[0]

    def test_filter(self, sample_rows: list[Row]):
        """Test row filtering."""
        container = DictListContainer(sample_rows)
        filtered = container.filter(lambda r: r["value"] > 150)
        assert len(filtered) == 2

    def test_map(self, sample_rows: list[Row]):
        """Test row mapping."""
        container = DictListContainer(sample_rows)
        mapped = container.map(lambda r: {**r, "doubled": r["value"] * 2})
        assert list(mapped)[0]["doubled"] == 200

    def test_sort(self, sample_rows: list[Row]):
        """Test sorting."""
        container = DictListContainer(sample_rows)
        sorted_container = container.sort("value", reverse=True)
        assert list(sorted_container)[0]["name"] == "Charlie"

    def test_head(self, sample_rows: list[Row]):
        """Test head."""
        container = DictListContainer(sample_rows)
        head = container.head(2)
        assert len(head) == 2

    def test_tail(self, sample_rows: list[Row]):
        """Test tail."""
        container = DictListContainer(sample_rows)
        tail = container.tail(2)
        assert len(tail) == 2
        assert tail[1]["name"] == "Charlie"

    def test_to_list(self, sample_rows: list[Row]):
        """Test conversion to list."""
        container = DictListContainer(sample_rows)
        result = container.to_list()
        assert result == sample_rows

    def test_from_records(self):
        """Test creation from records."""
        records = [(1, "Alice"), (2, "Bob")]
        container = DictListContainer.from_records(records, ["id", "name"])
        assert len(container) == 2
        assert container[0]["name"] == "Alice"

    def test_from_dict(self):
        """Test creation from column dict."""
        data = {"id": [1, 2], "name": ["Alice", "Bob"]}
        container = DictListContainer.from_dict(data)
        assert len(container) == 2
        assert container[0]["id"] == 1

    def test_to_dict(self, sample_rows: list[Row]):
        """Test conversion to column dict."""
        container = DictListContainer(sample_rows)
        result = container.to_dict()
        assert result["id"] == [1, 2, 3]
        assert result["name"] == ["Alice", "Bob", "Charlie"]


class TestStreamingCSVContainer:
    """Tests for StreamingCSVContainer."""

    def test_init(self, sample_csv: Path):
        """Test initialization."""
        container = StreamingCSVContainer(sample_csv)
        assert container.path == sample_csv

    def test_columns(self, sample_csv: Path):
        """Test column names."""
        container = StreamingCSVContainer(sample_csv)
        assert container.columns == ["id", "name", "value"]

    def test_len(self, sample_csv: Path):
        """Test row count."""
        container = StreamingCSVContainer(sample_csv)
        assert len(container) == 3

    def test_iteration(self, sample_csv: Path):
        """Test iteration."""
        container = StreamingCSVContainer(sample_csv)
        rows = list(container)
        assert len(rows) == 3
        assert rows[0]["name"] == "Alice"

    def test_multiple_iterations(self, sample_csv: Path):
        """Test multiple iterations (file reopens)."""
        container = StreamingCSVContainer(sample_csv)

        first = list(container)
        second = list(container)

        assert first == second

    def test_select(self, sample_csv: Path):
        """Test column selection."""
        container = StreamingCSVContainer(sample_csv)
        selected = container.select(["id", "name"])
        rows = list(selected)
        assert "value" not in rows[0]

    def test_head(self, sample_csv: Path):
        """Test head."""
        container = StreamingCSVContainer(sample_csv)
        head = container.head(2)
        assert len(head) == 2

    def test_sample(self, sample_csv: Path):
        """Test random sampling."""
        container = StreamingCSVContainer(sample_csv)
        sample = container.sample(2, seed=42)
        assert len(sample) == 2

    def test_schema(self, sample_csv: Path):
        """Test schema inference."""
        container = StreamingCSVContainer(sample_csv)
        schema = container.schema
        assert schema["id"] is int
        assert schema["name"] is str


class TestPolarsContainer:
    """Tests for PolarsContainer."""

    def test_init(self, sample_rows: list[Row]):
        """Test initialization from Polars DataFrame."""
        import polars as pl

        df = pl.DataFrame(sample_rows)
        container = PolarsContainer(df)
        assert len(container) == 3

    def test_iteration(self, sample_rows: list[Row]):
        """Test iteration."""
        import polars as pl

        df = pl.DataFrame(sample_rows)
        container = PolarsContainer(df)
        rows = list(container)
        assert len(rows) == 3
        assert rows[0]["name"] == "Alice"

    def test_columns(self, sample_rows: list[Row]):
        """Test columns."""
        import polars as pl

        df = pl.DataFrame(sample_rows)
        container = PolarsContainer(df)
        assert container.columns == ["id", "name", "value"]

    def test_to_list(self, sample_rows: list[Row]):
        """Test conversion to list."""
        import polars as pl

        df = pl.DataFrame(sample_rows)
        container = PolarsContainer(df)
        result = container.to_list()
        assert len(result) == 3

    def test_from_csv(self, sample_csv: Path):
        """Test creation from CSV."""
        container = PolarsContainer.from_csv(sample_csv)
        assert len(container) == 3

    def test_to_polars(self, sample_rows: list[Row]):
        """Test to_polars returns underlying DataFrame."""
        import polars as pl

        df = pl.DataFrame(sample_rows)
        container = PolarsContainer(df)
        result = container.to_polars()
        assert isinstance(result, pl.DataFrame)

    def test_to_pandas(self, sample_rows: list[Row]):
        """Test conversion to Pandas."""
        import pandas as pd
        import polars as pl

        df = pl.DataFrame(sample_rows)
        container = PolarsContainer(df)
        result = container.to_pandas()
        assert isinstance(result, pd.DataFrame)


class TestPandasContainer:
    """Tests for PandasContainer."""

    def test_init(self, sample_rows: list[Row]):
        """Test initialization from Pandas DataFrame."""
        import pandas as pd

        df = pd.DataFrame(sample_rows)
        container = PandasContainer(df)
        assert len(container) == 3

    def test_iteration(self, sample_rows: list[Row]):
        """Test iteration."""
        import pandas as pd

        df = pd.DataFrame(sample_rows)
        container = PandasContainer(df)
        rows = list(container)
        assert len(rows) == 3
        assert rows[0]["name"] == "Alice"

    def test_columns(self, sample_rows: list[Row]):
        """Test columns."""
        import pandas as pd

        df = pd.DataFrame(sample_rows)
        container = PandasContainer(df)
        assert container.columns == ["id", "name", "value"]

    def test_to_pandas(self, sample_rows: list[Row]):
        """Test to_pandas returns underlying DataFrame."""
        import pandas as pd

        df = pd.DataFrame(sample_rows)
        container = PandasContainer(df)
        result = container.to_pandas()
        assert isinstance(result, pd.DataFrame)

    def test_from_csv(self, sample_csv: Path):
        """Test creation from CSV."""
        container = PandasContainer.from_csv(sample_csv)
        assert len(container) == 3


class TestResultContainerImpl:
    """Tests for ResultContainerImpl."""

    def test_init(self, sample_rows: list[Row]):
        """Test initialization."""
        container = ResultContainerImpl(sample_rows)
        assert len(container) == 3

    def test_iteration(self, sample_rows: list[Row]):
        """Test iteration."""
        container = ResultContainerImpl(sample_rows)
        rows = list(container)
        assert len(rows) == 3

    def test_to_csv(self, sample_rows: list[Row], tmp_path: Path):
        """Test CSV output."""
        container = ResultContainerImpl(sample_rows)
        output_path = tmp_path / "output.csv"
        container.to_csv(output_path)
        assert output_path.exists()

        # Verify content
        content = output_path.read_text()
        assert "Alice" in content

    def test_to_parquet(self, sample_rows: list[Row], tmp_path: Path):
        """Test Parquet output."""
        container = ResultContainerImpl(sample_rows)
        output_path = tmp_path / "output.parquet"
        container.to_parquet(output_path)
        assert output_path.exists()

    def test_to_json(self, sample_rows: list[Row], tmp_path: Path):
        """Test JSON output."""
        container = ResultContainerImpl(sample_rows)
        output_path = tmp_path / "output.jsonl"
        container.to_json(output_path)
        assert output_path.exists()

    def test_to_pandas(self, sample_rows: list[Row]):
        """Test conversion to Pandas."""
        import pandas as pd

        container = ResultContainerImpl(sample_rows)
        result = container.to_pandas()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_to_polars(self, sample_rows: list[Row]):
        """Test conversion to Polars."""
        import polars as pl

        container = ResultContainerImpl(sample_rows)
        result = container.to_polars()
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3

    def test_select(self, sample_rows: list[Row]):
        """Test column selection."""
        container = ResultContainerImpl(sample_rows)
        selected = container.select(["id", "name"])
        assert selected.columns == ["id", "name"]

    def test_filter(self, sample_rows: list[Row]):
        """Test row filtering."""
        container = ResultContainerImpl(sample_rows)
        filtered = container.filter(lambda r: r["value"] > 150)
        assert len(filtered) == 2

    def test_merge_with(self, sample_rows: list[Row]):
        """Test merging containers."""
        container1 = ResultContainerImpl(sample_rows[:2])
        container2 = ResultContainerImpl(sample_rows[2:])
        merged = container1.merge_with(container2)
        assert len(merged) == 3

    def test_from_pandas(self, sample_rows: list[Row]):
        """Test creation from Pandas."""
        import pandas as pd

        df = pd.DataFrame(sample_rows)
        container = ResultContainerImpl.from_pandas(df)
        assert len(container) == 3

    def test_from_polars(self, sample_rows: list[Row]):
        """Test creation from Polars."""
        import polars as pl

        df = pl.DataFrame(sample_rows)
        container = ResultContainerImpl.from_polars(df)
        assert len(container) == 3


class TestDataContainerProtocol:
    """Tests for DataContainer protocol compliance."""

    def test_dict_list_is_data_container(self, sample_rows: list[Row]):
        """Test DictListContainer implements DataContainer."""
        container = DictListContainer(sample_rows)
        assert isinstance(container, DataContainer)

    def test_streaming_csv_is_data_container(self, sample_csv: Path):
        """Test StreamingCSVContainer implements DataContainer."""
        container = StreamingCSVContainer(sample_csv)
        assert isinstance(container, DataContainer)

    def test_polars_is_data_container(self, sample_rows: list[Row]):
        """Test PolarsContainer implements DataContainer."""
        import polars as pl

        df = pl.DataFrame(sample_rows)
        container = PolarsContainer(df)
        assert isinstance(container, DataContainer)

    def test_pandas_is_data_container(self, sample_rows: list[Row]):
        """Test PandasContainer implements DataContainer."""
        import pandas as pd

        df = pd.DataFrame(sample_rows)
        container = PandasContainer(df)
        assert isinstance(container, DataContainer)


class TestResultContainerDataFrameIdioms:
    """``result.data`` must accept the DataFrame idioms the docs show.

    ``result.data`` was a pandas DataFrame until the container replaced it.
    The quickstart, the README notebook and a dozen examples still write
    ``print(result.data[["product", "category"]])`` — the first thing a new
    user runs after their first pipeline — and it raised
    ``TypeError: list indices must be integers or slices, not list``.
    """

    @staticmethod
    def _container():
        from ondine.adapters.containers.result_container import ResultContainerImpl

        return ResultContainerImpl(
            data=[
                {"product": "laptop", "category": "Electronics", "price": 999},
                {"product": "apple", "category": "Food", "price": 1},
            ],
            columns=["product", "category", "price"],
        )

    def test_a_list_of_columns_selects_those_columns_in_that_order(self):
        selected = self._container()[["category", "product"]]

        assert selected.columns == ["category", "product"]
        assert selected.to_list() == [
            {"category": "Electronics", "product": "laptop"},
            {"category": "Food", "product": "apple"},
        ]

    def test_selecting_an_unknown_column_names_it(self):
        with pytest.raises(KeyError, match="colour"):
            self._container()[["product", "colour"]]

    def test_printing_shows_the_rows_as_a_table(self):
        printed = str(self._container()[["product", "category"]])

        lines = printed.splitlines()
        assert lines[0].split() == ["product", "category"]
        assert lines[1].split() == ["0", "laptop", "Electronics"]
        assert lines[2].split() == ["1", "apple", "Food"]

    def test_to_string_passes_pandas_options_through(self):
        printed = self._container()[["product"]].to_string(index=False)

        assert [line.strip() for line in printed.splitlines()] == [
            "product",
            "laptop",
            "apple",
        ]

    def test_to_dict_matches_the_dataframe_shape(self):
        assert self._container()[["product", "price"]].to_dict(orient="records") == [
            {"product": "laptop", "price": 999},
            {"product": "apple", "price": 1},
        ]


class TestResultContainerMoreDataFrameIdioms:
    """Idioms the shipped examples use that the container still rejected."""

    @staticmethod
    def _container():
        from ondine.adapters.containers.result_container import ResultContainerImpl

        return ResultContainerImpl(
            data=[
                {"product": "laptop", "category": "Electronics"},
                {"product": "phone", "category": "Electronics"},
                {"product": "apple", "category": "Food"},
            ],
            columns=["product", "category"],
        )

    def test_to_csv_accepts_index_false_like_a_dataframe(self, tmp_path):
        """``result.data.to_csv(path, index=False)`` is in examples and docs."""
        path = tmp_path / "out.csv"

        self._container().to_csv(path, index=False)

        assert path.read_text().splitlines() == [
            "product,category",
            "laptop,Electronics",
            "phone,Electronics",
            "apple,Food",
        ]

    def test_a_column_counts_its_values_most_common_first(self):
        counts = self._container()["category"].value_counts()

        assert counts.to_dict() == {"Electronics": 2, "Food": 1}
        assert list(counts.index) == ["Electronics", "Food"]
