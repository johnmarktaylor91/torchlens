"""Trace-state vocabulary and the capture-state predicates shared across layers.

``TraceState`` moved to :mod:`torchlens._vocab.trace_state` (ratified V2 relocation,
architecture memo 3.3, C01 item 6); every historical spelling and the pickle-visible
identity stay unchanged, and new lower-layer consumers import from the vocabulary home.

This basis module also holds the predicate that recognises a reduce-and-discard
capture (``tl.trace(model, x, save="stats")``), so the engine (replay and validation
entry) and ``torchlens.stats`` read the same answer without the engine importing the
readers' package.
"""

from __future__ import annotations

from typing import Any

from ._vocab.trace_state import TraceState

__tl_layer__ = "FACADE"
__all__ = ["STATS_REDUCER_REPR", "TraceState", "is_stats_capture"]

#: The ``save="stats"`` reducer's fixed repr; a Trace records it as its transform repr,
#: which survives save and load, so a reloaded stats trace is still recognised.
STATS_REDUCER_REPR = "tl.stats save='stats' spine reducer"

#: Where ``Trace`` is defined; a basis module identifies it by class, not by import
#: (the ``_record_type`` precedent; the canonical reader is
#: ``_data_classes._trace_missing_attrs.is_live_trace``).
_TRACE_MODULE = __name__.rsplit(".", 1)[0] + "._data_classes.trace"


def _is_live_trace(obj: Any) -> bool:
    """Return whether ``obj`` is a ``Trace`` that still holds its stores."""

    for cls in type(obj).__mro__:
        if cls.__name__ == "Trace" and cls.__module__ == _TRACE_MODULE:
            return not obj.__dict__.get("_tl_cleaned_up", False)
    return False


def is_stats_capture(trace: Any) -> bool:
    """Whether ``trace`` was captured with ``save="stats"`` (live or reloaded)."""

    if trace._options_save_raw_activations if _is_live_trace(trace) else True:
        return False
    transform = trace._options_activation_transform if _is_live_trace(trace) else None
    if getattr(transform, "_tl_stats_reducer", False):
        return True
    return trace._activation_transform_repr == STATS_REDUCER_REPR  # noqa: SLF001 -- declared Trace field
