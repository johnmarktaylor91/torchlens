"""``region()``: user regions on the one span stack (torchnative W1.2).

A region is a named user bracket riding the SAME span registry as every
other altitude -- never a parallel stack. It carries occurrence and parent
ids, escaped/bounded labels, scalar-only metadata, and is INERT without a
consumer: with no active profiler session and no active TorchLens capture,
entering a region costs a counter increment and records nothing.

Consumers:

- an active :func:`torchlens.observability.session` records a
  ``user_region`` span (and, in owned/borrowed profiler modes, brackets the
  body with ``torch.profiler.record_function`` so the region shows up in
  native traces);
- an active TorchLens capture records the region through the shipped
  ``torchlens.observers.span`` record-tier surface (label unification
  consumed: one span vocabulary, not a second one).

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any

from .. import _state, observers
from ._session import active_session
from ._spans import escape_label, validate_scalar_metadata

__tl_layer__ = "L5"


class _RegionState(threading.local):
    """Per-thread region nesting and occurrence counters."""

    def __init__(self) -> None:
        self.stack: list[str] = []
        self.occurrences: Counter[str] = Counter()


_REGIONS = _RegionState()


@dataclass(frozen=True)
class RegionRecord:
    """One region occurrence: identity, nesting, metadata, consumers hit."""

    name: str
    occurrence_index: int
    parent_name: str | None
    metadata: dict[str, Any]
    span_id: int | None
    recorded_on_capture: bool


@contextmanager
def region(name: str, **metadata: Any) -> Iterator[RegionRecord]:
    """Bracket a user region on the one span stack.

    Parameters
    ----------
    name:
        Region label; escaped and bounded before recording.
    **metadata:
        Scalar-only metadata (bool / int / float / str); anything else
        refuses typed (``region_metadata_invalid``).

    Yields
    ------
    RegionRecord
        The occurrence record (``span_id`` is None when no session consumer
        was active).
    """

    label = escape_label(name)
    validated = validate_scalar_metadata("region", metadata)
    state = _REGIONS
    state.occurrences[label] += 1
    occurrence = state.occurrences[label]
    parent = state.stack[-1] if state.stack else None
    session = active_session()
    span_id: int | None = None
    capture_active = _state._active_trace is not None
    record = RegionRecord(
        name=label,
        occurrence_index=occurrence,
        parent_name=parent,
        metadata=dict(validated),
        span_id=None,
        recorded_on_capture=capture_active,
    )
    state.stack.append(label)
    try:
        with ExitStack() as stack:
            if session is not None:
                span_id = session.registry.open(
                    label,
                    altitude="op",
                    owner="user_region",
                    metadata={**validated, "occurrence": occurrence},
                )
                record = RegionRecord(
                    name=label,
                    occurrence_index=occurrence,
                    parent_name=parent,
                    metadata=dict(validated),
                    span_id=span_id,
                    recorded_on_capture=capture_active,
                )
                if session.profiler is not None:
                    import torch.profiler as _torch_profiler

                    stack.enter_context(
                        _torch_profiler.record_function(f"torchlens::region::{label}")
                    )
            if capture_active:
                # Record-tier support rides the SHIPPED record-span surface
                # (one span vocabulary; never a parallel stack).
                span_record = stack.enter_context(observers.span(label, direction="both"))
                span_record["region"] = {
                    "occurrence": occurrence,
                    "parent": parent,
                    "metadata": dict(validated),
                }
            yield record
    finally:
        if span_id is not None and session is not None:
            session.registry.close(span_id)
        if state.stack and state.stack[-1] == label:
            state.stack.pop()


__all__ = ["RegionRecord", "region"]
