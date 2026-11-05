"""Module-boundary host lookup for injected-op records."""

from __future__ import annotations

from typing import Any

from .._utils._record_group_table import record_member


def _module_boundary_host(ops: tuple[Any, ...], host_label: str) -> Any | None:
    """Resolve a module-boundary firing's host op from its ``address:pass`` label.

    A ``tl.module(...)`` rule fires at a real module hook, where the only
    identity that exists is the module call label (``fc1:1``); no op carries
    that raw label. The host is the retained op that PRODUCED that module
    call's output: the ``interventionreplacement`` op the capture mints at
    module exit when the hook replaced the value (preferred -- it carries
    ``is_edited=True``), else the module's own output op. Both list the module call
    in ``output_of_module_calls``. ``None`` when no retained op exits that module
    call (the record stays unanchored and the codec refuses to persist it typed).
    """

    fallback: Any | None = None
    for op in ops:
        exits = record_member(op, "output_of_module_calls", None) or ()
        if host_label not in exits:
            continue
        if getattr(op, "is_edited", False):
            return op
        if fallback is None:
            fallback = op
    return fallback
