"""Structured intervention site collections for sweeps."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from .selectors import BaseSelector, contains, func, module


def _as_tuple(value: Any | Iterable[Any] | None) -> tuple[Any, ...]:
    """Normalize a scalar or iterable option to a tuple.

    Parameters
    ----------
    value:
        Scalar, iterable, or ``None``.

    Returns
    -------
    tuple[Any, ...]
        Normalized tuple.
    """

    if value is None:
        return (None,)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


@dataclass(frozen=True)
class SiteSpec:
    """One concrete site sweep entry.

    Parameters
    ----------
    layer_pattern:
        Layer selector, module substring, or label substring.
    op:
        Optional operation name to intersect with the layer pattern.
    mode:
        Optional sweep mode label.
    selector:
        Selector used for hook or resolver matching.
    """

    layer_pattern: Any
    op: Any
    mode: Any
    selector: Any


@dataclass(frozen=True)
class SiteCollection:
    """Structured collection of site sweep entries.

    Parameters
    ----------
    layer_pattern:
        Layer selector, module substring, or label substring.
    ops:
        Operation names to sweep.
    modes:
        Mode labels to sweep.
    entries:
        Concrete site entries.
    """

    layer_pattern: Any
    ops: tuple[Any, ...]
    modes: tuple[Any, ...]
    entries: tuple[SiteSpec, ...]

    def __iter__(self) -> Iterator[SiteSpec]:
        """Iterate over concrete site entries.

        Yields
        ------
        SiteSpec
            One sweep entry.
        """

        yield from self.entries

    def __len__(self) -> int:
        """Return the number of concrete site entries.

        Returns
        -------
        int
            Entry count.
        """

        return len(self.entries)

    def selectors(self) -> tuple[Any, ...]:
        """Return selectors for all concrete entries.

        Returns
        -------
        tuple[Any, ...]
            Selector tuple in sweep order.
        """

        return tuple(entry.selector for entry in self.entries)

    def to_hook_pairs(self, hook: Any) -> list[tuple[Any, Any]]:
        """Return ``(site, hook)`` pairs for hook-plan normalization.

        Parameters
        ----------
        hook:
            Hook callable or helper spec to attach at every site.

        Returns
        -------
        list[tuple[Any, Any]]
            Hook-plan pairs.
        """

        return [(entry.selector, hook) for entry in self.entries]


def _selector_for(layer_pattern: Any, op: Any) -> Any:
    """Build a selector for one layer/operation combination.

    Parameters
    ----------
    layer_pattern:
        Layer selector, module substring, or label substring.
    op:
        Optional operation name.

    Returns
    -------
    Any
        Selector-like object.
    """

    if isinstance(layer_pattern, BaseSelector):
        selector = layer_pattern
    elif isinstance(layer_pattern, str) and "." in layer_pattern:
        selector = module(layer_pattern)
    elif isinstance(layer_pattern, str):
        selector = contains(layer_pattern)
    else:
        selector = layer_pattern
    if op is None:
        return selector
    op_selector = func(str(op))
    if isinstance(selector, BaseSelector):
        return selector & op_selector
    return op_selector


def sites(
    layer_pattern: Any, ops: Any | Iterable[Any] | None = None, modes: Any = None
) -> SiteCollection:
    """Create a structured site collection for sweeps.

    Parameters
    ----------
    layer_pattern:
        Layer selector, module substring, or label substring.
    ops:
        Operation name or iterable of operation names.
    modes:
        Mode label or iterable of mode labels.

    Returns
    -------
    SiteCollection
        Sweep-ready site collection.
    """

    op_values = _as_tuple(ops)
    mode_values = _as_tuple(modes)
    entries = tuple(
        SiteSpec(
            layer_pattern=layer_pattern,
            op=op,
            mode=mode,
            selector=_selector_for(layer_pattern, op),
        )
        for mode in mode_values
        for op in op_values
    )
    return SiteCollection(
        layer_pattern=layer_pattern,
        ops=op_values,
        modes=mode_values,
        entries=entries,
    )


__all__ = ["SiteCollection", "SiteSpec", "sites"]
