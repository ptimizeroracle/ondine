"""Helpers for optional dependency error messaging."""

from __future__ import annotations

import re
from typing import NoReturn


def _matches_missing_dependency(
    exc: Exception, dependency_names: tuple[str, ...]
) -> bool:
    """Return True when the exception indicates a missing optional dependency."""
    module_name = getattr(exc, "name", None)
    if isinstance(module_name, str) and module_name in dependency_names:
        return True

    exc_text = str(exc).lower()
    if "no module named" not in exc_text and "cannot import name" not in exc_text:
        return False

    for name in dependency_names:
        patterns = (
            rf"no module named ['\"]?{re.escape(name)}['\"]?",
            rf"cannot import name ['\"]?{re.escape(name)}['\"]?",
        )
        if any(re.search(pattern, exc_text) for pattern in patterns):
            return True
    return False


def raise_excel_extra_error(operation: str, exc: Exception) -> NoReturn:
    """Raise a friendly error for missing Excel support dependencies."""
    if _matches_missing_dependency(exc, ("openpyxl", "xlrd")):
        raise ImportError(
            f"{operation} requires the 'excel' extra. "
            "Install with: pip install 'ondine[excel]'"
        ) from exc
    raise exc


def raise_parquet_extra_error(operation: str, exc: Exception) -> NoReturn:
    """Raise a friendly error for missing Parquet support dependencies."""
    if _matches_missing_dependency(exc, ("pyarrow", "fastparquet")):
        raise ImportError(
            f"{operation} requires the 'parquet' extra. "
            "Install with: pip install 'ondine[parquet]'"
        ) from exc
    raise exc
