"""The Paddle ``tl.record`` producer: lean per-op events, values only when selected.

The producer satisfies ``torchlens.recording._backend_events.BackendRecordProducer``
by duck typing, so this module imports without the record interface. Each pass
installs the Paddle wrappers ``tl.trace(..., backend="paddle")`` installs, bound to
a recording backend whose ``emit_paddle_operation`` emits one metadata-only event
per wrapped call instead of building an ``OpEvent``. Labels come from the same
reservation sequence in the same order (input leaves, then per call its unlabeled
buffer or constant leaves, then its outputs), so every raw label equals the label
``tl.trace`` gives that op. A value reaches the sink only when the sink requests
it. When a decision says stop, the wrapper raises out of the forward, so no op
after the stop frontier runs.

The scratch ``Trace`` a pass opens only carries the capture-depth, module-stack
and label counters the wrappers read; it is never materialized, finalized or
validated, and it keeps no op payloads (source leaves excepted: buffers and
closure constants, which the model holds anyway).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from ... import _state
from ..._data_classes._trace_state_io import fill_unset_declared_fields
from ..._data_classes.trace import Trace
from ..._ir.capture_events import CaptureEvents
from ..._ir.events import ModuleFrame
from ..._ir.predicate import ModuleStackFrame
from ..._ir.refs import DtypeRef, ReservedLabel
from ._capture_helpers import paddle_layer_type
from .backend import PaddleBackend, PaddleOpCapture, _device_ref_from_paddle_place
from .model_prep import cleanup_model_session, discover_paddle_module_tree, prepare_model_session
from .source_leaves import PADDLE_BUFFER_OP, PaddleSourceLeaves, record_source_leaves
from .wrappers import is_alias_allowed_op, unwrap_paddle, wrap_paddle


class _StopRecording(Exception):
    """Raised inside a wrapper to unwind the forward at a stop decision."""


class PaddlePayloadCodec:
    """Payload conversions for selected Paddle values."""

    def ram_copy(self, value: Any) -> Any:
        """Return an independent detached copy of ``value`` on its own place.

        Parameters
        ----------
        value
            One Paddle tensor.

        Returns
        -------
        Any
            A Paddle tensor that later in-place writes to ``value`` cannot change.
        """

        with _state.pause_logging():
            return value.detach().clone()

    def to_disk_tensor(self, value: Any) -> Any:
        """Return a CPU ``torch.Tensor`` holding the same numbers as ``value``.

        Parameters
        ----------
        value
            One Paddle tensor.

        Returns
        -------
        torch.Tensor
            CPU tensor with the matching dtype (bfloat16 keeps its bits).
        """

        import numpy as np
        import torch

        with _state.pause_logging():
            host = np.array(value.numpy(), copy=True)
        if "bfloat16" in str(value.dtype):
            return torch.from_numpy(host.view(np.int16)).view(torch.bfloat16)
        return torch.from_numpy(host)


@dataclass(frozen=True)
class _Prepared:
    """A Paddle model discovered once per recorder."""

    model: Any
    module_tree: Any


class PaddleRecordProducer:
    """``tl.record`` producer for eager Paddle callables and ``paddle.nn.Layer`` roots."""

    backend = "paddle"
    supported_options: frozenset[str] = frozenset()

    def __init__(self) -> None:
        """Set up the codec and the per-recorder counters tests read."""

        self.codec = PaddlePayloadCodec()
        self.passes = 0
        self.ops_emitted = 0

    def prepare(self, model: Any) -> _Prepared:
        """Discover the module tree of ``model`` once (``None`` for a raw callable)."""

        return _Prepared(model=model, module_tree=discover_paddle_module_tree(model))

    def run_pass(
        self,
        prepared: _Prepared,
        input_args: Any,
        input_kwargs: Mapping[str, Any],
        sink: Any,
    ) -> Any:
        """Run one forward, emitting events; return the output, or ``None`` after a stop."""

        self.passes += 1
        backend = _RecordingPaddleBackend(sink, self)
        session = _scratch_session(prepared.model)
        tree = prepared.module_tree
        args = backend._normalize_input_args(input_args)
        kwargs = dict(input_kwargs)
        backend._param_address_by_id = dict(tree.param_address_by_id) if tree else {}
        backend._source_leaves = PaddleSourceLeaves.capture(
            prepared.model, backend.paddle, backend._param_address_by_id, PaddleOpCapture
        )
        prepare_model_session(session, prepared.model, tree)
        try:
            wrap_paddle(backend)
            backend._label_source_tensors(session, args, kwargs)
            with _state.active_logging(session):
                return cast(Any, prepared.model)(*args, **kwargs)
        except _StopRecording:
            return None
        finally:
            try:
                cleanup_model_session(session, prepared.model, tree)
            finally:
                unwrap_paddle()
                backend.tensor_store.clear()


def _scratch_session(model: Any) -> Trace:
    """Open the unmaterialized session the Paddle wrappers read during one pass."""

    session = Trace(model_class_name=type(model).__name__)
    fill_unset_declared_fields(session)
    session.capture_events = CaptureEvents()
    session._paddle_capture_depth = 0
    session._paddle_module_stack = []
    session._paddle_op_captures = []
    session._paddle_alias_annotations = []
    session._paddle_capture_gap_markers = []
    session._paddle_intervention_runtime = None
    return session


_NO_LINEAGE: tuple[tuple[str, ...], tuple[Any, ...]] = ((), ())
"""The (parent labels, module frames) of an input or source leaf: neither."""


class _RecordingPaddleBackend(PaddleBackend):
    """A Paddle backend whose wrapper events go to a record sink, not a trace."""

    def __init__(self, sink: Any, producer: PaddleRecordProducer) -> None:
        """Bind the sink the wrappers' events go to."""

        super().__init__()
        self._sink = sink
        self._producer = producer

    def _append_source(self, trace: Trace, label: str, tensor: object) -> None:
        """Label one input leaf and emit it as an ``input`` event."""

        self.tensor_store.set_label(tensor, label)
        events = trace.capture_events
        events.raw_layer_counter += 1
        index = events.raw_layer_counter
        site = ReservedLabel(label, label, index, index, "input", label)
        self._emit(site, "input", tensor, _NO_LINEAGE, kind="input")

    def emit_paddle_operation(
        self,
        trace: Trace,
        op_name: str,
        func: object,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: object,
        *,
        module_stack: tuple[ModuleFrame, ...] | None = None,
    ) -> object:
        """Emit one event per non-alias tensor output of a wrapped call; return ``output``."""

        del func
        outputs = tuple(self._iter_tensors_with_paths(output))
        if not outputs:
            return output
        events = trace.capture_events
        first_new = len(events.op_events)
        record_source_leaves(self, trace, events, args, kwargs)
        for event in events.op_events[first_new:]:
            self._emit_source_leaf(event)
        reserved = events.reserve_label_block(paddle_layer_type(op_name), len(outputs))
        events.func_call_id_counter += 1
        input_ids = {id(tensor) for _leaf, tensor in self._iter_input_leaf_records(args, kwargs)}
        aliases_allowed = is_alias_allowed_op(op_name)
        parents = tuple(edge.parent_label_raw for edge in self._parent_edges(args, kwargs)[0])
        frames = _stack_frames(module_stack or tuple(trace._paddle_module_stack))
        for index, (_path, tensor) in enumerate(outputs):
            if aliases_allowed and id(tensor) in input_ids:
                continue
            self._emit(reserved[index], op_name, tensor, (parents, frames))
            self.tensor_store.set_label_if_unlabeled(tensor, reserved[index].label_raw)
        return output

    def _emit_source_leaf(self, event: Any) -> None:
        """Emit a buffer or closure-constant leaf ``record_source_leaves`` just appended."""

        tensor = event.output.tensor.payload
        site = ReservedLabel(
            event.label_raw,
            event.label_raw,
            event.raw_index,
            event.type_index,
            event.layer_type,
            event.label_raw,
        )
        kind = "buffer" if event.layer_type == PADDLE_BUFFER_OP else "op"
        self._emit(site, event.layer_type, tensor, _NO_LINEAGE, kind=kind)

    def _emit(
        self,
        site: ReservedLabel,
        func_name: str,
        tensor: Any,
        lineage: tuple[tuple[str, ...], tuple[Any, ...]],
        *,
        kind: str = "op",
    ) -> None:
        """Emit one event, deliver its value when requested, unwind when told to stop.

        ``lineage`` is the call's (parent labels, module frames) pair, shared by
        every output of one call; source leaves have neither.
        """

        parents, frames = lineage

        # Deferred: the record interface is optional at import time and imports torch.
        from ...recording._backend_events import BackendOpEvent

        event = BackendOpEvent(
            raw_label=site.label_raw,
            layer_type=site.layer_type,
            type_index=site.type_index,
            raw_index=site.raw_index,
            parent_labels=parents,
            func_name=func_name,
            shape=self._shape(tensor),
            dtype=DtypeRef(backend="paddle", name=str(self._dtype(tensor))),
            device=_device_ref_from_paddle_place(self._device(tensor)),
            requires_grad=not bool(getattr(tensor, "stop_gradient", True)),
            module_stack=frames,
            func_call_id=site.raw_index,
            kind=kind,
        )
        if kind == "op":
            self._producer.ops_emitted += 1
        with _state.pause_logging():
            decision = self._sink.emit(event)
            if decision.request is not None:
                self._sink.deliver(decision.request, tensor)
        if decision.stop:
            raise _StopRecording


def _stack_frames(module_stack: tuple[ModuleFrame, ...]) -> tuple[ModuleStackFrame, ...]:
    """Return the predicate-context frames of a Paddle module stack, outermost first."""

    return tuple(
        ModuleStackFrame(
            address=frame.address,
            module_type=frame.module_type,
            module_id=0,
            pass_number=int(frame.call_index),
        )
        for frame in module_stack
    )


def paddle_record_producer() -> PaddleRecordProducer:
    """Return a fresh Paddle record producer (the ``fastlog`` capability binding)."""

    return PaddleRecordProducer()
