"""The MLX ``tl.record`` producer: lean per-op events, values only when selected.

The producer satisfies ``torchlens.recording._backend_events.BackendRecordProducer``
by duck typing, so this module imports without the record interface. Each pass
installs the MLX wrappers ``tl.trace(..., backend="mlx")`` installs, bound to a
recording backend whose ``emit_mlx_operation`` emits one metadata-only event per
wrapped call instead of building an ``OpEvent``. Labels come from the same
reservation sequence in the same order (input leaves, then per call its closure
constants, then its outputs), so every raw label equals the label ``tl.trace``
gives that op.

MLX computes lazily. The producer holds the arrays the sink requested and
realizes them together in one ``mx.eval`` before delivering them, so an
unselected value is never realized for the recording and never copied. Held
values are bounded: once their bytes reach ``chunk_bytes`` the forward pauses
while the held chunk is realized and delivered (to RAM, or through the
``storage=`` disk writer), and the rest are delivered with the output when the
forward returns or a decision says stop. When a decision says stop
the wrapper raises out of the forward, so no op after the stop frontier runs.

The scratch ``Trace`` a pass opens only carries the capture-depth, module-stack
and label counters the wrappers read; it is never materialized, finalized or
validated, and it keeps no op payloads.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from ... import _state
from ..._data_classes._trace_state_io import fill_unset_declared_fields
from ..._data_classes.trace import Trace
from ..._ir.capture_events import CaptureEvents
from ..._ir.events import ModuleFrame
from ..._ir.predicate import ModuleStackFrame
from ..._ir.refs import DeviceRef, DtypeRef, ReservedLabel
from ..._ir.semantics import CapturePolicy
from ._retention import DEFAULT_CHUNK_BYTES
from .backend import MLXBackend
from .external_arrays import external_array_ids
from .model_prep import discover_mlx_module_tree


class _StopRecording(Exception):
    """Raised inside a wrapper to unwind the forward at a stop decision."""


class MLXPayloadCodec:
    """Payload conversions for selected MLX values."""

    def ram_copy(self, value: Any) -> Any:
        """Return a realized MLX array that later writes to ``value`` cannot change.

        Parameters
        ----------
        value
            One MLX array.

        Returns
        -------
        Any
            An independent, evaluated MLX array with the same dtype and values.
        """

        import mlx.core as mx

        with _state.pause_logging():
            copy = mx.array(value)
            mx.eval(copy)
        return copy

    def to_disk_tensor(self, value: Any) -> Any:
        """Return a CPU ``torch.Tensor`` holding the same numbers as ``value``.

        Parameters
        ----------
        value
            One MLX array.

        Returns
        -------
        torch.Tensor
            CPU tensor with the matching dtype; bfloat16 travels through float32,
            which holds every bfloat16 value exactly.
        """

        import mlx.core as mx
        import numpy as np
        import torch

        with _state.pause_logging():
            if value.dtype == mx.bfloat16:
                host = np.array(value.astype(mx.float32), copy=True)
                return torch.from_numpy(host).to(torch.bfloat16)
            return torch.from_numpy(np.array(value, copy=True))


@dataclass(frozen=True)
class _Prepared:
    """The model a recorder records."""

    model: Any


class MLXRecordProducer:
    """``tl.record`` producer for eager MLX callables and ``mlx.nn.Module`` roots."""

    backend = "mlx"
    supported_options: frozenset[str] = frozenset()

    def __init__(self, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> None:
        """Set up the codec, the held-value byte bound and the counters tests read.

        Parameters
        ----------
        chunk_bytes
            Bytes of requested values held before the forward pauses to deliver them.
        """

        self.codec = MLXPayloadCodec()
        self.chunk_bytes = chunk_bytes
        self.passes = 0
        self.ops_emitted = 0
        self.values_delivered = 0
        self.drains = 0

    def prepare(self, model: Any) -> _Prepared:
        """Return the prepared model (module discovery runs per pass, as in ``tl.trace``)."""

        return _Prepared(model=model)

    def pause_capture(self) -> Any:
        """Pause MLX op logging while the sink runs predicates and codecs."""

        return _state.pause_logging()

    def run_pass(
        self,
        prepared: _Prepared,
        input_args: Any,
        input_kwargs: Mapping[str, Any],
        sink: Any,
    ) -> Any:
        """Run one forward, emitting events; return the output, or ``None`` after a stop."""

        self.passes += 1
        backend = _RecordingMLXBackend(sink, self)
        model = prepared.model
        module_tree = discover_mlx_module_tree(model)
        session = _scratch_session(model, module_tree, backend)
        args = backend._normalize_input_args(input_args)
        kwargs = dict(input_kwargs)
        output: Any = None
        try:
            backend.wrap(model, module_tree)
            backend._label_source_arrays(session, args, kwargs)
            with _state.active_logging(session):
                output = cast(Any, model)(*args, **kwargs)
        except _StopRecording:
            output = None
        finally:
            backend.unwrap(model)
        backend.deliver_requested(output)
        backend.tensor_store.clear()
        return output


def _scratch_session(model: Any, module_tree: Any, backend: MLXBackend) -> Trace:
    """Open the unmaterialized session the MLX wrappers read during one pass."""

    session = Trace(model_class_name=type(model).__name__)
    fill_unset_declared_fields(session)
    session.capture_events = CaptureEvents()
    session._mlx_saved_payloads = []
    session._mlx_capture_depth = 0
    session._mlx_module_stack = []
    session._mlx_intervention_plan = None
    session._mlx_halt_selector = None
    session._mlx_param_ids = (
        frozenset(module_tree.param_address_by_id) if module_tree is not None else frozenset()
    )
    session._mlx_external_arrays = external_array_ids(model, backend.is_tensor)
    session._mlx_unwrapped_intermediates = 0
    session._mlx_op_captures = []
    return session


class _RecordingMLXBackend(MLXBackend):
    """An MLX backend whose wrapper events go to a record sink, not a trace."""

    def __init__(self, sink: Any, producer: MLXRecordProducer) -> None:
        """Bind the sink the wrappers' events go to."""

        super().__init__()
        self._sink = sink
        self._producer = producer
        self._requested: deque[tuple[Any, Any]] = deque()
        self._requested_bytes = 0

    def _capture_policy(self, session: Any) -> CapturePolicy:
        """Keep no payloads on the scratch session (closure constants included)."""

        del session
        return CapturePolicy(save_payload=False, save_grad=False)

    def _label_source_arrays(self, trace: Trace, args: list[Any], kwargs: dict[Any, Any]) -> None:
        """Label each input array and emit it as an ``input`` event."""

        sources = [(f"input.arg_{index}", arg) for index, arg in enumerate(args)]
        sources += [(f"input.{key}", value) for key, value in kwargs.items()]
        events = trace.capture_events
        for label, array in sources:
            if not self.is_tensor(array):
                continue
            self.tensor_store.set_label(array, label)
            events.raw_layer_counter += 1
            site = self._input_reservation(label, events.raw_layer_counter)
            self._emit(site, "input", array, (), ())

    def emit_mlx_operation(  # noqa: PLR0913 - overrides MLXBackend.emit_mlx_operation, same signature
        self,
        trace: Trace,
        op_name: str,
        func: object,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: object,
        *,
        module_stack: tuple[ModuleFrame, ...] | None = None,
    ) -> Any:
        """Emit one event per array output of a wrapped call; return ``output``."""

        del func
        outputs = tuple(self._iter_arrays(output))
        if not outputs:
            return output
        events = trace.capture_events
        first_new = len(events.op_events)
        self._record_closure_constants(trace, args, kwargs, events)
        for event in events.op_events[first_new:]:
            self._emit_constant(event, args, kwargs)
        reserved = events.reserve_label_block(op_name, len(outputs))
        events.func_call_id_counter += 1
        frames = _stack_frames(module_stack or tuple(trace._mlx_module_stack))
        for site, array in zip(reserved, outputs, strict=True):
            # Same order as the trace path: label the output, then read parents.
            self.tensor_store.set_label(array, site.label_raw)
            parents = tuple(edge.parent_label_raw for edge in self._parent_edges(args, kwargs)[0])
            self._emit(site, op_name, array, parents, frames)
        return output

    def _emit_constant(self, event: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Emit a closure-constant source op ``_record_closure_constants`` just labeled."""

        label = event.label_raw
        leaves = [leaf for value in (*args, *kwargs.values()) for leaf in self._iter_arrays(value)]
        array = next(leaf for leaf in leaves if self.tensor_store.get_label(leaf) == label)
        site = ReservedLabel(
            label, label, event.raw_index, event.type_index, event.layer_type, label
        )
        self._emit(site, "constant", array, (), ())

    def _emit(
        self,
        site: ReservedLabel,
        func_name: str,
        array: Any,
        parents: tuple[str, ...],
        frames: tuple[ModuleStackFrame, ...],
    ) -> None:
        """Emit one event, hold its array when requested, unwind when told to stop.

        ``func_name`` ``"input"`` marks an input leaf; every other name is an op event.
        """

        # Deferred: the record interface is optional at import time and imports torch.
        from ...recording._backend_events import BackendOpEvent

        kind = "input" if func_name == "input" else "op"
        event = BackendOpEvent(
            raw_label=site.label_raw,
            layer_type=site.layer_type,
            type_index=site.type_index,
            raw_index=site.raw_index,
            parent_labels=parents,
            func_name=func_name,
            shape=self._shape(array),
            dtype=DtypeRef.from_value(self._dtype(array)),
            device=DeviceRef.from_value(self._device(array)),
            module_stack=frames,
            func_call_id=site.raw_index,
            kind=kind,
        )
        if kind == "op":
            self._producer.ops_emitted += 1
        with _state.pause_logging():
            decision = self._sink.emit(event)
        if decision.request is not None:
            self._requested.append((decision.request, array))
            self._requested_bytes += int(getattr(array, "nbytes", 0))
            if self._requested_bytes >= self._producer.chunk_bytes:
                self._drain(())
        if decision.stop:
            raise _StopRecording

    def deliver_requested(self, output: Any) -> None:
        """Deliver every value still held, realized together with the output."""

        self._drain(tuple(self._iter_arrays(output)))

    def _drain(self, extra_values: tuple[Any, ...]) -> None:
        """Realize the held values (and ``extra_values``) once, deliver, and drop them."""

        held = self._requested
        self._requested, self._requested_bytes = deque(), 0
        values = [array for _request, array in held]
        with _state.pause_logging():
            if values or extra_values:
                cast(Any, self.mx).eval(*values, *extra_values)
            del values
            while held:
                request, array = held.popleft()
                self._sink.deliver(request, array)
                self._producer.values_delivered += 1
        self._producer.drains += 1


def _stack_frames(module_stack: tuple[ModuleFrame, ...]) -> tuple[ModuleStackFrame, ...]:
    """Return the predicate-context frames of an MLX module stack, outermost first."""

    return tuple(
        ModuleStackFrame(
            address=frame.address,
            module_type=frame.module_type,
            module_id=0,
            pass_number=int(frame.call_index),
        )
        for frame in module_stack
    )


def mlx_record_producer() -> MLXRecordProducer:
    """Return a fresh MLX record producer (the ``fastlog`` capability binding)."""

    return MLXRecordProducer()
