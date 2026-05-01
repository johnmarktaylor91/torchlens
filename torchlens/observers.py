"""User observer helpers for taps, scalar logs, and record spans."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch

from . import _state


@dataclass(frozen=True)
class TapRecord:
    """One observed activation value.

    Parameters
    ----------
    value:
        Detached activation snapshot.
    site_label:
        Capture-time site label.
    span_names:
        Active span names when the tap fired.
    timestamp:
        Monotonic timestamp.
    """

    value: torch.Tensor
    site_label: str | None
    span_names: tuple[str, ...]
    timestamp: float


@dataclass
class TapObserver:
    """Callable hook that records activations without modifying them.

    Parameters
    ----------
    site:
        Selector-like site where this tap should be registered.
    """

    site: Any
    records: list[TapRecord] = field(default_factory=list)

    def __call__(self, activation: torch.Tensor, *, hook: Any) -> torch.Tensor:
        """Record an activation and return it unchanged.

        Parameters
        ----------
        activation:
            Activation observed at the hook site.
        hook:
            Hook context supplied by TorchLens.

        Returns
        -------
        torch.Tensor
            The original activation.
        """

        with _state.pause_logging():
            value = activation.detach().clone()
        span_names = tuple(str(span["name"]) for span in _state._active_record_spans)
        self.records.append(
            TapRecord(
                value=value,
                site_label=getattr(hook.layer_log, "layer_label", None),
                span_names=span_names,
                timestamp=time.monotonic(),
            )
        )
        return activation

    def values(self) -> list[torch.Tensor]:
        """Return observed activation values.

        Returns
        -------
        list[torch.Tensor]
            Detached activation snapshots in observation order.
        """

        return [record.value for record in self.records]

    def clear(self) -> None:
        """Clear previously observed records.

        Returns
        -------
        None
            The observer is mutated in place.
        """

        self.records.clear()


def tap(site: Any) -> TapObserver:
    """Create a tap observer for a site.

    Parameters
    ----------
    site:
        Selector-like site to observe.

    Returns
    -------
    TapObserver
        Callable observer with ``records`` and ``values()`` accessors.
    """

    return TapObserver(site=site)


@contextmanager
def record_span(name: str) -> Iterator[dict[str, Any]]:
    """Record a named observer span around captures or hook execution.

    Parameters
    ----------
    name:
        Span name.

    Yields
    ------
    dict[str, Any]
        Mutable span metadata record.
    """

    span = {"name": str(name), "start": time.monotonic(), "end": None}
    _state._active_record_spans.append(span)
    model_log = _state._active_model_log
    if model_log is not None:
        model_log.observer_spans.append(span)
    try:
        yield span
    finally:
        span["end"] = time.monotonic()
        if _state._active_record_spans and _state._active_record_spans[-1] is span:
            _state._active_record_spans.pop()
        elif span in _state._active_record_spans:
            _state._active_record_spans.remove(span)


def active_span_records() -> list[dict[str, Any]]:
    """Return currently active span records.

    Returns
    -------
    list[dict[str, Any]]
        Active span records.
    """

    return list(_state._active_record_spans)


__all__ = ["TapObserver", "TapRecord", "active_span_records", "record_span", "tap"]
