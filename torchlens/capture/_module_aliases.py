"""Module alias addresses published to capture-time selectors.

A shared submodule is registered under every attribute name that holds it,
while ``model.modules()`` yields it once. Model preparation records every name
in the module's ``all_addresses``; these helpers hand that alias map to the
selectors that run during capture.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trace import Trace


def _capture_module_aliases(trace: Trace) -> dict[str, str] | None:
    """Return the prepared model's alias address -> canonical address map.

    Model preparation records every registered name of a shared module in its
    metadata's ``all_addresses`` (canonical first).

    Parameters
    ----------
    trace:
        Trace whose per-session model preparation has run.

    Returns
    -------
    dict[str, str] | None
        The alias map (empty when no module is shared), or ``None`` once
        finalization has handed the metadata to ``trace.modules`` (post-hoc
        evaluation reads the alias map there).
    """

    workspace = getattr(trace, "_module_capture_ws", None)
    metadata = getattr(workspace, "module_metadata", None)
    if not metadata:
        return None
    return {
        alias: primary
        for primary, meta in metadata.items()
        for alias in meta.get("all_addresses", ())
        if alias != primary
    }


def _module_alias_scope(trace: Trace) -> contextlib.AbstractContextManager[None]:
    """Publish the prepared model's alias addresses to capture-time selectors.

    Live intervention hooks and capture-time ``save=`` predicates see only the
    canonical address on module frames, so ``tl.module("alias")`` /
    ``tl.in_module("alias:2")`` resolve through this map, exactly as
    ``trace.modules["alias"]`` does post hoc.

    Parameters
    ----------
    trace:
        Trace whose per-session model preparation has run.

    Returns
    -------
    contextlib.AbstractContextManager[None]
        The selector alias scope, or a no-op once the metadata moved to
        ``trace.modules``.
    """

    from ..ir.selector_eval import module_alias_scope

    aliases = _capture_module_aliases(trace)
    if aliases is None:
        return contextlib.nullcontext()
    return module_alias_scope(aliases)
