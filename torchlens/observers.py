"""User observer helpers for taps, scalar logs, and record spans."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

import torch

from . import _state


@dataclass(frozen=True)
class TapRecord:
    """One observed tensor value.

    Parameters
    ----------
    value:
        Detached tensor snapshot.
    site_label:
        Capture-time site label.
    span_names:
        Active span names when the tap fired.
    timestamp:
        Monotonic timestamp.
    direction:
        Direction in which the tap fired.
    grad_kind:
        Gradient payload kind for backward records.
    backward_call_index:
        One-based backward call index for backward records.
    """

    value: torch.Tensor
    site_label: str | None
    span_names: tuple[str, ...]
    timestamp: float
    direction: Literal["forward", "backward"]
    grad_kind: Literal["grad_input", "grad_output"] | None = None
    backward_call_index: int | None = None


@dataclass
class TapObserver:
    """Callable hook that records outs without modifying them.

    Parameters
    ----------
    site:
        Selector-like site where this tap should be registered.
    direction:
        Direction in which this tap should fire.
    """

    site: Any
    direction: Literal["forward", "backward", "both"] = "forward"
    records: list[TapRecord] = field(default_factory=list)

    def __call__(self, out: torch.Tensor, *, hook: Any) -> torch.Tensor:
        """Record an out and return it unchanged.

        Parameters
        ----------
        out:
            Activation observed at the hook site.
        hook:
            Hook context supplied by TorchLens.

        Returns
        -------
        torch.Tensor
            The original out.
        """

        with _state.pause_logging():
            value = out.detach().clone()
        span_names = tuple(str(span["name"]) for span in _state._active_record_spans)
        self.records.append(
            TapRecord(
                value=value,
                site_label=getattr(hook.layer_log, "layer_label", None),
                span_names=span_names,
                timestamp=time.monotonic(),
                direction="forward",
            )
        )
        return out

    def record_backward(
        self,
        grad_input: tuple[torch.Tensor | None, ...],
        *,
        grad_output: tuple[torch.Tensor | None, ...] | None,
        grad_fn_log: Any,
        call_index: int,
        run_ctx: dict[str, Any],
    ) -> None:
        """Record a backward gradient and leave autograd gradients unchanged.

        Parameters
        ----------
        grad_input:
            Autograd grad_input tuple at the grad_fn hook.
        grad_output:
            Autograd grad_output tuple at the grad_fn hook, when available.
        grad_fn_log:
            GradFn site whose hook fired.
        call_index:
            One-based backward call index.
        run_ctx:
            Shared hook run context. Accepted for hook API compatibility.

        Returns
        -------
        None
            Backward taps observe only and do not mutate gradients.
        """

        del run_ctx
        grad_value, grad_kind = _first_tensor_grad(grad_output, "grad_output")
        if grad_value is None:
            grad_value, grad_kind = _first_tensor_grad(grad_input, "grad_input")
        if grad_value is None:
            return
        with _state.pause_logging():
            value = grad_value.detach().clone()
        span_names = tuple(str(span["name"]) for span in _state._active_record_spans)
        self.records.append(
            TapRecord(
                value=value,
                site_label=getattr(grad_fn_log, "label", None),
                span_names=span_names,
                timestamp=time.monotonic(),
                direction="backward",
                grad_kind=grad_kind,
                backward_call_index=call_index,
            )
        )

    def values(self) -> list[torch.Tensor]:
        """Return observed out values.

        Returns
        -------
        list[torch.Tensor]
            Detached out snapshots in observation order.
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


def _first_tensor_grad(
    grads: tuple[torch.Tensor | None, ...] | None,
    grad_kind: Literal["grad_input", "grad_output"],
) -> tuple[torch.Tensor | None, Literal["grad_input", "grad_output"] | None]:
    """Return the first tensor gradient in a hook payload.

    Parameters
    ----------
    grads:
        Autograd gradient tuple, or ``None``.
    grad_kind:
        Kind to annotate if a tensor is found.

    Returns
    -------
    tuple[torch.Tensor | None, Literal["grad_input", "grad_output"] | None]
        Tensor gradient and its kind, or ``(None, None)``.
    """

    if grads is None:
        return None, None
    for grad in grads:
        if isinstance(grad, torch.Tensor):
            return grad, grad_kind
    return None, None


def tap(
    site: Any,
    *,
    direction: Literal["forward", "backward", "both"] = "forward",
) -> TapObserver:
    """Create a tap observer for a site.

    Parameters
    ----------
    site:
        Selector-like site to observe.
    direction:
        Direction in which the tap should fire.

    Returns
    -------
    TapObserver
        Callable observer with ``records`` and ``values()`` accessors.
    """

    if direction not in {"forward", "backward", "both"}:
        raise ValueError("direction must be 'forward', 'backward', or 'both'.")
    return TapObserver(site=site, direction=direction)


@contextmanager
def record_span(
    name: str,
    *,
    direction: Literal["forward", "backward", "both"] = "both",
) -> Iterator[dict[str, Any]]:
    """Record a named observer span around captures or hook execution.

    Parameters
    ----------
    name:
        Span name.
    direction:
        Direction scope metadata for this span.

    Yields
    ------
    dict[str, Any]
        Mutable span metadata record.
    """

    if direction not in {"forward", "backward", "both"}:
        raise ValueError("direction must be 'forward', 'backward', or 'both'.")
    span = {"name": str(name), "direction": direction, "start": time.monotonic(), "end": None}
    _state._active_record_spans.append(span)
    trace = _state._active_trace
    if trace is not None:
        trace.observer_spans.append(span)
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
