"""Deferred record contexts for a lean ``tl.module`` read.

A ``RecordContext`` carries the last ``history_size`` contexts of its pass, and
each of those carries its own window, so building any context needs every
context before it. A lean recording (``lean=True``, provisional spelling) whose
save selector is a plain ``tl.module`` union reads no context until a selected
module exits: an op context never matches the union at op time (it has no
module-output candidate), the halt slot is off and a lean journal skips the
shell events of unkept ops. Under those conditions the capture defers every
context as the plain inputs ``_build_record_context`` would receive, and builds
them in order only when something reads one: a selected module's exit, or any
site that iterates the history window. Each built context equals the one the
eager capture builds, because it receives the same inputs and the same window.

This module imports no capture internals: the context builder and the module
union matcher are handed in by the installer, so it stays a leaf of the import
graph.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch

from .._ir.predicate import ModuleStackFrame, RecordContext


@dataclass(frozen=True, slots=True)
class _BoundaryProbe:
    """The module-output candidate an op context gains at one module exit."""

    output_of_module_calls: tuple[str, ...]


def lazy_contexts_eligible(options: Any, module_spec: Any, no_tensor_capture: bool) -> bool:
    """Return whether a pass may defer its record contexts.

    Parameters
    ----------
    options:
        Resolved recording settings of the pass.
    module_spec:
        Capture policy every module boundary event gets (``default_module``).
    no_tensor_capture:
        Whether the pass is a dry run that reads every context afterwards.

    Returns
    -------
    bool
        True only for a lean pass with no halt, intervention, narration,
        lookback, source-event selection or gradient selection, a non-empty
        history window and unkept module boundary events. The caller also
        requires a plain ``tl.module`` union save selector.
    """

    return bool(
        options.lean is True
        and not no_tensor_capture
        and options.halt is None
        and options.intervene is None
        and options.echo is None
        and not options.lookback
        and options.history_size > 0
        and not options.include_source_events
        and options.save_grads is None
        and options.default_grad is False
        and not (module_spec.save_out or module_spec.save_metadata)
    )


def tensor_fields(tensor: Any) -> dict[str, Any]:
    """Return the tensor facts a record context reads, taken now.

    Parameters
    ----------
    tensor:
        The event's tensor.

    Returns
    -------
    dict[str, Any]
        ``shape``, ``dtype``, ``tensor_device`` and ``tensor_requires_grad`` for a
        tensor (the values ``_build_record_context`` reads from it), else the
        object under ``tensor`` so the builder treats it as the eager path does.
    """

    if isinstance(tensor, torch.Tensor):
        return {
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "tensor_device": tensor.device,
            "tensor_requires_grad": tensor.requires_grad,
        }
    return {"tensor": tensor}


class LazyContextHistory(deque[RecordContext]):
    """History window that defers record-context construction.

    Installed as ``RecordingState.history`` for one eligible pass. Eager sites
    keep working unchanged: iterating the window builds every deferred context
    first and yields the last ``window`` ones, ``append`` takes an eagerly built
    context and ``len`` never exceeds the window, so ``append_context`` never
    trims. Only iteration, ``append``, ``len`` and ``clear`` are supported.
    Deferred contexts reach ``all_contexts`` and ``op_context_by_raw_label``
    when they are built; contexts that nothing reads are never built.
    """

    def __init__(
        self,
        state: Any,
        *,
        build: Callable[..., RecordContext],
        window: int,
        module_addresses: frozenset[str],
        matches: Callable[[Any, frozenset[str]], bool],
    ) -> None:
        super().__init__()
        self.state = state
        self._build = build
        self._window = window
        self._module_addresses = module_addresses
        self._matches = matches
        self._entries: list[Any] = []
        self._built = 0

    def __iter__(self) -> Iterator[RecordContext]:
        self.materialize()
        return iter(self._entries[max(0, len(self._entries) - self._window) :])

    def __len__(self) -> int:
        return min(len(self._entries), self._window)

    def append(self, ctx: RecordContext) -> None:
        """Take an eagerly built context, which saw every earlier context built."""

        if self._built != len(self._entries):
            raise RuntimeError("an eager record context skipped deferred contexts")
        self._entries.append(ctx)
        self._built += 1

    def clear(self) -> None:
        """Drop the pass's contexts (the per-pass reset)."""

        self._entries.clear()
        self._built = 0

    def defer(
        self,
        kind: str,
        data: dict[str, Any],
        event_index: int,
        step_index: int | None,
        time_since_pass_start: float,
    ) -> None:
        """Defer one event's context with the inputs the eager build would use.

        Parameters
        ----------
        kind:
            Record-context kind.
        data:
            Event fields (``op_log_or_op_data``), tensor facts already taken.
        event_index:
            The event's index in the pass.
        step_index:
            The op step index, ``None`` for sources and module boundaries.
        time_since_pass_start:
            Seconds since the pass started, taken at the event.
        """

        state = self.state
        self._entries.append(
            {
                "kind": kind,
                "op_log_or_op_data": data,
                "module_stack": tuple(state.module_stack),
                "pass_index": state.pass_index,
                "event_index": event_index,
                "step_index": step_index,
                "time_since_pass_start": time_since_pass_start,
                "include_source_events": state.options.include_source_events,
                "sample_id": state.sample_id,
            }
        )

    def defer_module_boundary(
        self, kind: str, frame: ModuleStackFrame, label: str, time_since_pass_start: float = 0.0
    ) -> None:
        """Count and defer one module enter or exit event of ``frame``."""

        self.state.event_index += 1
        data = {
            "label": label,
            "address": frame.address,
            "module_type": frame.module_type,
            "module_pass_number": frame.pass_number,
        }
        self.defer(kind, data, self.state.event_index, None, time_since_pass_start)

    def materialize(self) -> None:
        """Build every deferred context in event order."""

        entries = self._entries
        state = self.state
        window = self._window
        for index in range(self._built, len(entries)):
            ctx = self._build(history=entries[max(0, index - window) : index], **entries[index])
            entries[index] = ctx
            state.all_contexts.append(ctx)
            if ctx.kind == "op" and ctx.raw_label is not None:
                state.op_context_by_raw_label[ctx.raw_label] = ctx
        self._built = len(entries)

    def latest(self) -> RecordContext:
        """Build the deferred contexts and return the newest one."""

        self.materialize()
        return self._entries[-1]

    def prepare_module_boundary(self, module_call_label: str) -> None:
        """Build the deferred contexts when this module call is selected.

        Parameters
        ----------
        module_call_label:
            ``"<address>:<call index>"`` of the exiting module call.

        Notes
        -----
        The boundary retention step reads the op contexts of a selected
        module's outputs; it finds none for an unselected call and records
        nothing there, exactly as the eager capture's evaluation answers.
        """

        if self._built == len(self._entries):
            return
        if self._matches(_BoundaryProbe((module_call_label,)), self._module_addresses):
            self.materialize()


def lazy_history_of(state: Any) -> LazyContextHistory | None:
    """Return the deferred history installed on ``state``, if any."""

    history = getattr(state, "history", None)
    return history if type(history) is LazyContextHistory else None
