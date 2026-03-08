"""Helpers for optional dependency error messaging."""

from __future__ import annotations


def _matches_missing_dependency(
    exc: Exception, dependency_names: tuple[str, ...]
) -> bool:
    """Return True when the exception indicates a missing optional dependency."""
    module_name = getattr(exc, "name", None)
    if isinstance(module_name, str) and module_name in dependency_names:
        return True

    exc_text = str(exc).lower()
    return any(name in exc_text for name in dependency_names)


def raise_excel_extra_error(operation: str, exc: Exception) -> None:
    """Raise a friendly error for missing Excel support dependencies."""
    if _matches_missing_dependency(exc, ("openpyxl", "xlrd")):
        raise ImportError(
            f"{operation} requires the 'excel' extra. "
            "Install with: pip install 'ondine[excel]'"
        ) from exc
    raise exc


def raise_parquet_extra_error(operation: str, exc: Exception) -> None:
    """Raise a friendly error for missing Parquet support dependencies."""
    if _matches_missing_dependency(exc, ("pyarrow", "fastparquet")):
        raise ImportError(
            f"{operation} requires the 'parquet' extra. "
            "Install with: pip install 'ondine[parquet]'"
        ) from exc
    raise exc
