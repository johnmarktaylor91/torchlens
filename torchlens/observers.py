"""User observer helpers for taps, scalar logs, and record spans."""

from __future__ import annotations

import time
import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from . import _state


@dataclass(frozen=True)
class TapRecord:
    """One observed tensor value.

    ``site_label`` is a resolved property, not a stored field: the tap fires
    during capture when only the internal raw label (``relu_1_3_raw``) exists,
    but users index the trace by the public label (``relu_1_2``). The record
    keeps the raw label plus a weak reference to the capturing trace and resolves
    the public label lazily through the trace's raw-to-final label map, so
    ``record.site_label`` returns a label that actually indexes the public trace.

    Parameters
    ----------
    value:
        Detached tensor snapshot.
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
    span_names: tuple[str, ...]
    timestamp: float
    direction: Literal["forward", "backward"]
    grad_kind: Literal["grad_input", "grad_output"] | None = None
    backward_call_index: int | None = None
    _raw_site_label: str | None = None
    #: L6 stage 3: per-site selection mask stored when the tap was
    #: created from a resolved selection (session-time disclosure).
    selection_mask: torch.Tensor | None = None
    _trace_ref: Callable[[], Any] | None = field(default=None, compare=False, repr=False)

    @property
    def site_label(self) -> str | None:
        """Return the public capture-site label resolved from the raw label.

        Falls back to the raw label when the capturing trace is unavailable
        (e.g. never bound, or already garbage collected) or exposes no raw-to-
        final label map (which is the case for backward grad_fn labels).
        """

        raw = self._raw_site_label
        if raw is None:
            return None
        resolver = self._trace_ref
        trace = resolver() if resolver is not None else None
        mapping = getattr(trace, "_raw_to_final_layer_labels", None)
        if isinstance(mapping, Mapping):
            return mapping.get(raw, raw)
        return raw


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
        span_names = _active_span_names("forward")
        self.records.append(
            TapRecord(
                value=value,
                span_names=span_names,
                timestamp=time.monotonic(),
                direction="forward",
                _raw_site_label=_hook_layer_label(hook.layer_log),
                _trace_ref=_active_trace_ref(),
                selection_mask=self._selection_mask_for(hook.layer_log, value),
            )
        )
        return out

    def _selection_mask_for(self, layer_log: Any, value: torch.Tensor) -> torch.Tensor | None:
        """Return the stored per-site mask when the tap site is a resolved selection.

        The Selection contributes the SITE SET; each firing record stores the
        matching site's mask (fresh materialization). Non-selection sites and
        unmatched/mismatched shapes store no mask.
        """

        site = self.site
        entries = getattr(site, "__selection__", None) and getattr(site, "_entries", None)
        if not entries:
            return None
        layer_label = None
        if layer_log is not None:
            layer_label = (
                layer_log.get("layer_label")
                if hasattr(layer_log, "get")
                else getattr(layer_log, "layer_label", None)
            )
        for entry in entries:
            if entry.kind == "ACT" and entry.site_key[0] == layer_label:
                mask = entry.mask
                if tuple(mask.shape) == tuple(value.shape):
                    return mask
        return None

    def record_backward(
        self,
        grad_input: tuple[torch.Tensor | None, ...],
        *,
        grad_output: tuple[torch.Tensor | None, ...] | None,
        grad_fn_handle: Any,
        call_index: int,
        run_ctx: dict[str, Any],
    ) -> None:
        """Record a backward gradient and leave autograd gradients unchanged.

        Parameters
        ----------
        grad_input:
            Autograd grad_input tuple at the grad_fn_handle hook.
        grad_output:
            Autograd grad_output tuple at the grad_fn_handle hook, when available.
        grad_fn_handle:
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
        span_names = _active_span_names("backward")
        self.records.append(
            TapRecord(
                value=value,
                span_names=span_names,
                timestamp=time.monotonic(),
                direction="backward",
                grad_kind=grad_kind,
                backward_call_index=call_index,
                _raw_site_label=getattr(grad_fn_handle, "label", None),
                _trace_ref=_active_trace_ref(),
            )
        )

    def values(self, masked: bool = False) -> list[torch.Tensor]:
        """Return observed out values.

        Parameters
        ----------
        masked:
            ``False`` (default): the FULL snapshots, exactly the shipped
            behavior. ``True``: each record's stored selection mask is applied
            and fresh masked copies (the selected elements, flat) are
            returned; records without a stored mask return the full snapshot
            clone. Callers may not override the stored mask in v1.

        Returns
        -------
        list[torch.Tensor]
            Detached out snapshots in observation order.
        """

        if not masked:
            return [record.value for record in self.records]
        results: list[torch.Tensor] = []
        for record in self.records:
            if record.selection_mask is None:
                results.append(record.value.clone())
            else:
                results.append(torch.masked_select(record.value, record.selection_mask))
        return results

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


def _active_trace_ref() -> Callable[[], Any] | None:
    """Return a weak reference to the currently capturing trace, if any.

    A weak reference avoids keeping the whole trace graph alive through observer
    records; the record's ``site_label`` property falls back to the raw label if
    the trace has since been collected.
    """

    trace = _state._active_trace
    if trace is None:
        return None
    try:
        return weakref.ref(trace)
    except TypeError:
        return None


def _active_span_names(direction: Literal["forward", "backward"]) -> tuple[str, ...]:
    """Return active span names whose declared direction includes ``direction``.

    A span opened with ``direction="forward"`` scopes only forward observation and
    a ``direction="backward"`` span only backward observation; a ``"both"`` span
    scopes both. Enforcing this here stops a forward-only span from tagging a
    backward tap record (and vice versa), which previously happened because every
    active span was attached regardless of its declared direction.
    """

    return tuple(
        str(span["name"])
        for span in _state._active_record_spans.get()
        if span.get("direction") in (direction, "both")
    )


def _hook_layer_label(layer_log: Any) -> str | None:
    """Return a hook layer label from mapping or attribute context.

    Parameters
    ----------
    layer_log:
        Hook layer context supplied by TorchLens.

    Returns
    -------
    str | None
        Site label when present.
    """

    if hasattr(layer_log, "get"):
        label = layer_log.get("layer_label")
        return str(label) if label is not None else None
    label = getattr(layer_log, "layer_label", None)
    return str(label) if label is not None else None


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
def span(
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
    span_record = {
        "name": str(name),
        "direction": direction,
        "start": time.monotonic(),
        "end": None,
    }
    _state._active_record_spans.set(_state._active_record_spans.get() + (span_record,))
    trace = _state._active_trace
    if trace is not None:
        trace.observer_spans.append(span_record)
    try:
        yield span_record
    finally:
        span_record["end"] = time.monotonic()
        active = _state._active_record_spans.get()
        if active and active[-1] is span_record:
            _state._active_record_spans.set(active[:-1])
        elif span_record in active:
            _state._active_record_spans.set(
                tuple(record for record in active if record is not span_record)
            )


def active_span_records() -> list[dict[str, Any]]:
    """Return currently active span records.

    Returns
    -------
    list[dict[str, Any]]
        Active span records.
    """

    return list(_state._active_record_spans.get())


def log_value(name: str, value: Any) -> None:
    """Record an arbitrary scalar-like value on the active ``Trace``.

    Canonical home of the capture-time value writer (moved from
    ``torchlens.report``, which keeps a compatibility alias): observers own
    capture-time annotation writers, reporting surfaces own the readers.
    Values land under ``trace.annotations["logged_values"]`` and read back
    through ``Trace.logged_values``; they render in ``tl.report.explain`` and
    ``Trace.to_agent_json``.

    Parameters
    ----------
    name:
        Value name.
    value:
        Scalar or JSON-like value to record.

    Raises
    ------
    RuntimeError
        If no TorchLens capture is active.
    """

    trace = _state._active_trace
    if trace is None:
        raise RuntimeError(
            "torchlens.observers.log_value() (compat alias "
            "torchlens.report.log_value) must be called during trace."
        )
    trace.annotations.setdefault("logged_values", {})[str(name)] = value


__all__ = ["TapObserver", "TapRecord", "active_span_records", "log_value", "span", "tap"]
