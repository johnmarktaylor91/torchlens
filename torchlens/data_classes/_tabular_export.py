"""Shared tabular export helpers for TorchLens record objects."""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import pandas as pd


class _PandasExportable(Protocol):
    """Protocol for records and accessors that expose ``to_pandas``."""

    def to_pandas(self) -> "pd.DataFrame":
        """Export this object as a pandas DataFrame."""


class TabularExportMixin:
    """Mixin implementing CSV, Parquet, and JSON exports via ``to_pandas``."""

    def to_csv(self: _PandasExportable, path: str | PathLike[str], **kwargs: Any) -> None:
        """Write this object's tabular export to CSV.

        Parameters
        ----------
        path:
            Output CSV path.
        **kwargs:
            Additional keyword arguments forwarded to ``DataFrame.to_csv``.
        """

        self.to_pandas().to_csv(path, index=False, **kwargs)

    def to_parquet(self: _PandasExportable, path: str | PathLike[str], **kwargs: Any) -> None:
        """Write this object's tabular export to Parquet.

        Parameters
        ----------
        path:
            Output Parquet path.
        **kwargs:
            Additional keyword arguments forwarded to ``DataFrame.to_parquet``.

        Raises
        ------
        ImportError
            If ``pyarrow`` is unavailable.
        """

        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "to_parquet requires pyarrow. Install with: pip install torchlens[io]"
            ) from exc
        from ..export import _parquet_safe_dataframe

        _parquet_safe_dataframe(self.to_pandas()).to_parquet(path, **kwargs)

    def to_json(self: _PandasExportable, path: str | PathLike[str], **kwargs: Any) -> None:
        """Write this object's tabular export to JSON.

        Parameters
        ----------
        path:
            Output JSON path.
        **kwargs:
            Additional keyword arguments forwarded to ``DataFrame.to_json``.
        """

        from ..export import _parquet_safe_dataframe

        _parquet_safe_dataframe(self.to_pandas()).to_json(path, **kwargs)
