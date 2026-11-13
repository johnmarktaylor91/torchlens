"""TensorFlow ``trace(stop_after=...)``: halt the eager forward at a frontier op.

The op callback that records an op is the halt point. After a callback's outputs are
recorded, :class:`TFHaltRuntime` evaluates the stop predicate on each output's
``RecordContext``; on a match it latches the stop request on the trace (torch
``evaluate_halt`` parity, so a swallowed signal can never settle COMPLETE) and raises
``StopSignal`` out of the callback. TensorFlow propagates a callback exception through
the eager executor, so post-frontier user code never runs, and the session catches the
signal with the captured prefix intact. Finalization is the shared single-pass path,
so the halted prefix materializes and validates like a full capture.

When an op-level edit is pending on the frontier op, the halt waits until the writable
wrap layer has substituted the edited value (torch halts after the edit), then fires
with the edited tensor as the frontier output.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..._capture._stop_after_policy import settle_unfired_stop_after
from ..._capture.outcome import StopRequest
from ..._ir.predicate import RecordContext
from ...recording._exceptions import PredicateError
from ...recording._halt import StopSignal
from ..registry import BackendUnsupportedError

#: Recorded batch one op callback hands its observers.
RecordedBatch = list[tuple[str, RecordContext, Any]]


class CallbackAbort(BaseException):
    """Carry an observer's exception out of an op callback unchanged.

    Keras re-creates an ``Exception`` raised inside a layer call with an amended
    message, which drops typed fields such as ``code``. A ``BaseException`` passes
    through Keras and the eager executor untouched (as the halt signal does), and the
    capture entry re-raises the original error.
    """

    def __init__(self, error: Exception) -> None:
        """Store the original observer exception."""

        super().__init__(error)
        self.error = error


class TFHaltRuntime:
    """Session-scoped ``stop_after`` state for one TensorFlow eager capture."""

    def __init__(self, halt: Any, trace: Any, selector_shaped: bool) -> None:
        """Validate and store the stop predicate.

        Parameters
        ----------
        halt
            Public ``stop_after=`` value: a selector or a callable returning bool.
        trace
            The trace being captured (receives the stop-request latch).
        selector_shaped
            Whether ``halt`` is a selector (unfired refuses) rather than a callable (warns).
        """

        if not callable(halt):
            raise BackendUnsupportedError(
                "tf backend trace(stop_after=...) requires a selector or a callable "
                f"predicate returning bool; got {type(halt).__name__!s}.",
                code="trace_option_unsupported",
                option="stop_after",
            )
        self.halt = halt
        self.trace = trace
        self.selector_shaped = selector_shaped
        #: Set by the intervention layer: whether an op-level edit is pending on a label.
        self.edit_pending: Callable[[str], bool] = lambda _label: False
        #: Labels of the frontier op's outputs, which the halt cut off before Python.
        self.cut_labels: frozenset[str] = frozenset()
        self._deferred: RecordContext | None = None

    def observe(self, recorded: RecordedBatch) -> None:
        """Evaluate the stop predicate on one callback's recorded outputs.

        Parameters
        ----------
        recorded
            ``(label_raw, context, tensor)`` per recorded output, in output order.
        """

        if self._deferred is not None:
            return
        for label_raw, context, tensor in recorded:
            if not self._fires(context):
                continue
            self.cut_labels = frozenset(item[0] for item in recorded)
            if self.edit_pending(label_raw):
                self._deferred = context
                return
            self._raise(context, tensor)

    def release(self, label_raw: str, tensor: Any) -> None:
        """Fire a deferred halt once the wrap layer substituted the frontier op.

        Parameters
        ----------
        label_raw
            Raw label the wrap layer just presented.
        tensor
            The value that continues downstream (the edited tensor).
        """

        if self._deferred is not None and self._deferred.label_raw == label_raw:
            self._raise(self._deferred, tensor)

    def settle_unfired(self, trace: Any) -> None:
        """Refuse an unfired selector, warn on an unfired callable (torch policy)."""

        settle_unfired_stop_after(trace, self.halt, selector_shaped=self.selector_shaped)

    def _fires(self, context: RecordContext) -> bool:
        """Return whether the stop predicate fired for one output context."""

        result = self.halt(context)
        if self.selector_shaped:
            return bool(result)
        if not isinstance(result, bool):
            raise PredicateError(
                "halt predicate must return bool. "
                "Remedy: return True or False from the halt predicate.",
                ctx=context,
                result=result,
                code="predicate_return_invalid",
            )
        return result

    def _raise(self, context: RecordContext, frontier: Any) -> None:
        """Latch the stop request on the trace, then raise the halt signal."""

        label = context.label_raw
        self.trace.__dict__["_stop_requested"] = StopRequest(
            kind="halt",
            reason=label,
            boundary_kind=context.kind,
            boundary_label=label,
        )
        raise StopSignal(
            label, frontier_output=frontier, boundary_kind=context.kind, boundary_label=label
        )


def chain_output_hooks(
    *hooks: Callable[[RecordedBatch], None] | None,
) -> Callable[[RecordedBatch], None] | None:
    """Return one session ``output_hook`` running each present hook in order.

    Parameters
    ----------
    *hooks
        Observers, ``None`` entries skipped. Edit observers come before the halt
        observer so a pending edit is known when the halt predicate fires.

    Returns
    -------
    Callable | None
        The chained observer, or ``None`` when no hook is present.
    """

    present = [hook for hook in hooks if hook is not None]
    if not present:
        return None

    def chained(recorded: RecordedBatch) -> None:
        """Run every observer on one recorded batch; carry errors out as ``CallbackAbort``."""

        try:
            for hook in present:
                hook(recorded)
        except Exception as error:
            raise CallbackAbort(error) from error

    return chained


def apply_halt_outcome(trace: Any, signal: StopSignal | None) -> None:
    """Write the structural halt fields a halted capture carries (torch parity).

    Parameters
    ----------
    trace
        Trace being finalized.
    signal
        The halt that ended the forward, or ``None`` for a full forward.
    """

    if signal is None:
        return
    trace.is_stopped = True
    trace.stop_reason = trace._graph_stop_frontier = signal.reason
    trace.transformed_output = None


def refuse_static_halt(halt: Any, reason: str) -> None:
    """Refuse ``stop_after=`` on a graph-only (static FuncGraph) entry.

    Parameters
    ----------
    halt
        Requested stop site, or ``None``.
    reason
        Why the entry is graph-only.
    """

    if halt is not None:
        raise BackendUnsupportedError(
            "tf trace(stop_after=...) requires eager live capture; static FuncGraph "
            f"capture ({reason}) records a frozen graph without running it, so there "
            "is no forward to stop. Trace an eager-executable callable instead.",
            code="trace_option_unsupported",
            option="stop_after",
        )
