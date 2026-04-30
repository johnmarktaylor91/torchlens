"""Selector constructors for the planned TorchLens intervention site grammar."""

from collections.abc import Callable
from typing import Any

from .errors import _not_implemented


def label(value: str) -> Any:
    """Create a future exact-label selector.

    Parameters
    ----------
    value:
        TorchLens final or raw layer label.
    """

    return _not_implemented("label", "Phase 2")


def func(name: str) -> Any:
    """Create a future function-name selector.

    Parameters
    ----------
    name:
        Function name to match.
    """

    return _not_implemented("func", "Phase 2")


def module(address: str) -> Any:
    """Create a future module-address selector.

    Parameters
    ----------
    address:
        Module address to match.
    """

    return _not_implemented("module", "Phase 2")


def contains(text: str) -> Any:
    """Create a future substring selector.

    Parameters
    ----------
    text:
        Substring to match against labels or module context.
    """

    return _not_implemented("contains", "Phase 2")


def where(predicate: Callable[[Any], bool], *, name_hint: str | None = None) -> Any:
    """Create a future predicate selector.

    Parameters
    ----------
    predicate:
        Callable that will evaluate future layer-log objects.
    name_hint:
        Optional human-readable name for diagnostics and saved specs.
    """

    return _not_implemented("where", "Phase 2")


def in_module(layer_log: Any, address: str) -> bool:
    """Check future module containment for a layer log.

    Parameters
    ----------
    layer_log:
        Future layer-log object to inspect.
    address:
        Module address to test.
    """

    return _not_implemented("in_module", "Phase 2")


__all__ = ["contains", "func", "in_module", "label", "module", "where"]
