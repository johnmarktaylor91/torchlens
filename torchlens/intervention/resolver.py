"""Selector resolution ownership for future TorchLens intervention sites."""

from typing import Any

from .errors import _not_implemented


def resolve_sites(log: Any, selector: Any) -> Any:
    """Resolve future intervention selectors against a model log.

    Parameters
    ----------
    log:
        ModelLog-like object to resolve against.
    selector:
        Future selector or accepted selector shorthand.

    Returns
    -------
    Any
        Reserved resolution result.

    Raises
    ------
    NotImplementedError
        Always raised until Phase 2 implements selector resolution.
    """

    return _not_implemented("resolve_sites", "Phase 2")


def find_sites(log: Any, selector: Any) -> Any:
    """Build a future site table for selector matches.

    Parameters
    ----------
    log:
        ModelLog-like object to search.
    selector:
        Future selector or accepted selector shorthand.

    Returns
    -------
    Any
        Reserved site table.

    Raises
    ------
    NotImplementedError
        Always raised until Phase 2 implements site lookup.
    """

    return _not_implemented("find_sites", "Phase 2")


__all__ = ["find_sites", "resolve_sites"]
