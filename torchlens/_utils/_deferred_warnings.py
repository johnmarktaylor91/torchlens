"""Park one warning category while a call runs, then drop it or re-emit it."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

#: One parked ``warnings.showwarning`` call: message, category, positional and keyword args.
DeferredWarnings = list[tuple[Any, Any, tuple[Any, ...], dict[str, Any]]]


@contextmanager
def defer_warning_category(category: type[Warning], deferred: DeferredWarnings) -> Iterator[None]:
    """Hold back warnings of ``category`` in ``deferred``; forward everything else.

    The hold sits behind the warnings filters (only warnings that would be
    shown reach it). The caller decides afterwards: drop the parked warnings
    when the failure they describe never reaches the user, or re-emit them
    with :func:`flush_deferred_warnings` on every path that re-raises.

    Parameters
    ----------
    category:
        Warning class (and subclasses) to park.
    deferred:
        List the parked ``showwarning`` calls are appended to.

    Yields
    ------
    None
        Control while the hold is installed.
    """

    forward = warnings.showwarning

    def hold(message: Any, shown_category: Any, *args: Any, **kwargs: Any) -> None:
        """Park ``category`` warnings in ``deferred``; forward the rest."""

        if isinstance(shown_category, type) and issubclass(shown_category, category):
            deferred.append((message, shown_category, args, kwargs))
            return
        forward(message, shown_category, *args, **kwargs)

    warnings.showwarning = hold
    try:
        yield
    finally:
        if warnings.showwarning is hold:
            warnings.showwarning = forward


def flush_deferred_warnings(deferred: DeferredWarnings) -> None:
    """Re-emit parked warnings through the current warning handler and clear them.

    Parameters
    ----------
    deferred:
        Warnings parked by :func:`defer_warning_category`.
    """

    for message, category, args, kwargs in deferred:
        warnings.showwarning(message, category, *args, **kwargs)
    deferred.clear()
