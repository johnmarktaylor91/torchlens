"""Writer side of the trace's write-time container tokens.

A writer that builds a fresh exact container whose content key it can state
cheaply registers it with the capturing trace, so the freeze's container
pooling (``_data_classes._container_key``) reads that key instead of walking
the container. The table is ``WrapperRuntimeWorkspace.container_tokens``:
``id(container) -> (container, token, nested_ids, flags)``, created by the
first registration, taken and dropped by that trace's compaction, which
leaves ``False`` behind so nothing registers under the trace afterwards.

``flags`` (``_trace_core.pools``): :data:`TOKEN_EXACT` means ``token`` IS the
container's content key (``op.py`` ``_container_pool_key``), otherwise it is
a content token whose first cell's walked key is reused; :data:`TOKEN_INERT`
means the container holds no member the immutable pool could replace, so the
member sweep may skip it.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

from .. import _state

#: ``token`` is the container's exact content key.
TOKEN_EXACT = 1
#: The container holds no member the immutable pool could replace.
TOKEN_INERT = 2


def token_table(trace: Any) -> dict[int, Any] | None:
    """Return ``trace``'s registration table, creating it; ``None`` after its freeze."""

    workspace = getattr(trace, "_wrapper_runtime_ws", None)
    if workspace is None:
        return None
    table = workspace.container_tokens
    if table is None:
        table = workspace.container_tokens = {}
    elif table is False:
        return None
    return table


def register_active(
    container: Any, token: Hashable, nested_ids: tuple[int, ...], flags: int
) -> None:
    """Register a fresh container with the capturing trace (no-op outside a capture).

    Called once per emit, so the table lookup is inlined (no ``token_table`` call).
    """

    trace = _state._active_trace
    if trace is None:
        return
    workspace = trace._wrapper_runtime_ws
    table = workspace.container_tokens
    if table is None:
        table = workspace.container_tokens = {}
    elif table is False:
        return
    table[id(container)] = (container, token, nested_ids, flags)
