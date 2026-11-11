"""Writer side of the trace's write-time container tokens.

A writer that builds a fresh exact container whose content key it can state
cheaply registers it with the capturing trace, so the freeze's container
pooling (``_data_classes._container_key``) reads that key instead of walking
the container. The table is ``WrapperRuntimeWorkspace.container_tokens``:
``id(container) -> (container, token, flags)``, created by the first
registration and handed to the trace core's pool when the capture workspaces
drop (``_data_classes._compaction.hand_off_container_tokens``), which leaves
``False`` behind so nothing registers under the trace afterwards.

``flags`` (``_trace_core.pools``): :data:`TOKEN_EXACT` means ``token`` IS the
container's content key (``op.py`` ``_container_pool_key``), otherwise it is
a content token whose first cell's walked key is reused; :data:`TOKEN_INERT`
means the container holds no member the immutable pool could replace, so the
member sweep may skip it.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

#: ``token`` is the container's exact content key.
TOKEN_EXACT = 1
#: The container holds no member the immutable pool could replace.
TOKEN_INERT = 2


def token_table(trace: Any) -> dict[int, Any] | None:
    """Return ``trace``'s registration table, creating it; ``None`` once handed off."""

    try:
        workspace = trace._wrapper_runtime_ws
    except AttributeError:  # dropped with the capture workspaces
        return None
    if workspace is None:
        return None
    table = workspace.container_tokens
    if table is None:
        table = workspace.container_tokens = {}
    elif table is False:
        return None
    return table


def register(trace: Any, container: Any, token: Hashable, flags: int) -> None:
    """Register a fresh container with ``trace`` (no-op when it has no live table).

    Called once per emit, so the table lookup is inlined (no ``token_table`` call).
    """

    try:
        workspace = trace._wrapper_runtime_ws
    except AttributeError:  # dropped with the capture workspaces
        return
    if workspace is None:
        return
    table = workspace.container_tokens
    if table is None:
        table = workspace.container_tokens = {}
    elif table is False:
        return
    table[id(container)] = (container, token, flags)
