"""Content keys for container-cell pooling, read from write-time tokens.

``op.py``'s ``_pool_container_cells`` keys each non-empty candidate cell
through a ``content_key(value, id_counts)`` callable. :func:`make_content_key`
builds it: without a token registry it is the full content walk with the
alias guard; with one (``_trace_core.pools.ContainerTokens``) a registered
cell either returns the exact content key its writer stated (``TOKEN_EXACT``)
or reuses the content key its token's first cell computed, so the walk runs
at most once per distinct token instead of once per cell. Either way the returned key
is the cell's real content key, or ``None`` when the cell must not pool.
Standard library only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

#: When true (tests only), every token hit also walks the cell and raises if
#: the walked key differs from the token's cached key (proof step 4 of the
#: per-site interning design: a token is canonical and its content is fixed).
TOKEN_DEBUG = False

ContentKey = Callable[[Any, dict[int, int]], Any]


def make_content_key(container_pool_key: Callable[..., Any], tokens: Any = None) -> ContentKey:
    """Return the per-cell content-key callable for one pooling pass.

    Parameters
    ----------
    container_pool_key:
        ``op.py``'s ``_container_pool_key`` (passed in so this module never
        loads ``op``).
    tokens:
        The trace's ``ContainerTokens``, or ``None`` for the full walk only.
    """

    def walked_key(value: Any, id_counts: dict[int, int]) -> Any:
        """Walk ``value``; ``None`` when unpoolable or a nested container is aliased."""

        visited_ids: list[int] = []
        key = container_pool_key(value, 0, visited_ids)
        if key is None or any(id_counts[oid] > 1 for oid in visited_ids):
            return None
        return key

    if tokens is None or not tokens.by_id:
        return walked_key
    by_id_get = tokens.by_id.get
    keys = tokens.keys

    def token_key(value: Any, id_counts: dict[int, int]) -> Any:
        """Key a registered cell by its token, any other cell by the walk."""

        entry = by_id_get(id(value))
        if entry is None or entry[0] is not value:
            return walked_key(value, id_counts)
        # The caller already refused an aliased ``value``; the full walk would
        # also refuse it when a nested container is aliased.
        for oid in entry[2]:
            if oid in id_counts and id_counts[oid] > 1:
                return None
        token = entry[1]
        if entry[3] & 1:  # TOKEN_EXACT: the writer stated the content key itself
            if TOKEN_DEBUG:
                _check_token_key(value, token, walked_key(value, id_counts))
            return token
        if token in keys:
            key = keys[token]
            if TOKEN_DEBUG:
                _check_token_key(value, key, walked_key(value, id_counts))
            return key
        key = walked_key(value, id_counts)
        if key is not None:
            keys[token] = key
        return key

    return token_key


def _check_token_key(value: Any, cached: Any, walked: Any) -> None:
    """Raise when a token's cached key is not the cell's walked content key."""

    if walked != cached:
        raise RuntimeError(
            f"container token key mismatch for {value!r}: cached {cached!r}, walked {walked!r}"
        )


def members_inert(tokens: Any, value: Any) -> bool:
    """Return whether ``value`` is registered in ``tokens`` with no poolable member."""

    entry = tokens.by_id.get(id(value))
    return entry is not None and entry[0] is value and bool(entry[3] & 2)
