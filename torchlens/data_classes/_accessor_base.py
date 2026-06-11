"""Shared base class for TorchLens dict-like accessors."""

from __future__ import annotations

import weakref
from collections.abc import Iterator, Mapping
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Accessor(Generic[T]):
    """Generic dict-like accessor for ordered TorchLens log objects."""

    def __init__(
        self,
        items: Mapping[Any, T],
        *,
        source_ref: weakref.ReferenceType[Any] | None = None,
        item_list: list[T] | None = None,
    ) -> None:
        """Initialize the accessor.

        Parameters
        ----------
        items:
            Mapping from lookup keys to log objects.
        source_ref:
            Optional weak reference to the owning Trace-like object.
        item_list:
            Optional explicit ordered values. When omitted, mapping insertion
            order is used.
        """
        self._dict = dict(items)
        self._list = list(item_list) if item_list is not None else list(self._dict.values())
        if source_ref is not None:
            self._source_ref = source_ref

    def __getitem__(self, key: int | str) -> T:
        """Return an item by ordinal index, exact key, or subclass-specific lookup."""
        if isinstance(key, int):
            return self._list[key]
        if key in self._dict:
            return self._dict[key]
        if ":" in key:
            resolved = self._resolve_pass_qualified(key)
            if resolved is not None:
                return resolved
        resolved = self._resolve_substring(key)
        if resolved is not None:
            return resolved
        suggestions = self._suggest(key)
        if suggestions:
            suggestion_str = ", ".join(repr(item) for item in suggestions)
            raise KeyError(f"{self._item_kind} '{key}' not found. Did you mean {suggestion_str}?")
        raise KeyError(f"{self._item_kind} '{key}' not found.")

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` resolves to an item."""
        if isinstance(key, int):
            return 0 <= key < len(self._list)
        if not isinstance(key, str):
            return False
        if key in self._dict:
            return True
        if ":" in key and self._resolve_pass_qualified(key) is not None:
            return True
        return self._resolve_substring(key) is not None

    def __iter__(self) -> Iterator[T]:
        """Iterate over log objects in accessor order."""
        return iter(self._list)

    def __len__(self) -> int:
        """Return the number of primary items."""
        return len(self._dict)

    def __dir__(self) -> list[str]:
        """Return Python attributes plus lookup keys for tab completion."""
        return sorted(set(super().__dir__()) | {str(key) for key in self._dict})

    def _ipython_key_completions_(self) -> list[str]:
        """Return lookup keys for IPython ``obj[...]`` completion."""
        return [str(key) for key in self._dict]

    def keys(self) -> list[Any]:
        """Return primary lookup keys."""
        return list(self._dict.keys())

    def values(self) -> list[T]:
        """Return primary values."""
        return list(self._dict.values())

    def items(self) -> list[tuple[Any, T]]:
        """Return ``(key, value)`` pairs."""
        return list(self._dict.items())

    def to_pandas(self) -> Any:
        """Return a dataframe containing one row per accessor item.

        Returns
        -------
        Any
            Pandas DataFrame built from each item's ``to_pandas`` export.
        """

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        frames = [item.to_pandas() for item in self._list if hasattr(item, "to_pandas")]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _resolve_pass_qualified(self, key: str) -> T | None:
        """Resolve a pass-qualified key, if supported by the subclass.

        Parameters
        ----------
        key:
            Lookup key containing pass-qualified notation.

        Returns
        -------
        T | None
            Matching item, or ``None`` when unsupported/not found.
        """
        return None

    def _resolve_substring(self, key: str) -> T | None:
        """Resolve a substring key, if supported by the subclass.

        Parameters
        ----------
        key:
            Lookup key that did not exactly match.

        Returns
        -------
        T | None
            Matching item, or ``None`` when unsupported/not found.
        """
        return None

    def _suggest(self, key: str) -> list[str]:
        """Return lookup suggestions for ``key``.

        Parameters
        ----------
        key:
            Lookup key that did not resolve.

        Returns
        -------
        list[str]
            Candidate lookup strings.
        """
        return []

    @property
    def _item_kind(self) -> str:
        """Return the display name used in generic ``KeyError`` messages."""
        return type(self).__name__.removesuffix("Accessor")

    def __repr__(self) -> str:
        """Return a compact accessor summary."""
        keys = list(self._dict.keys())[:5]
        suffix = "..." if len(self._dict) > 5 else ""
        return f"{type(self).__name__} with {len(self)} items: {keys}{suffix}"
