"""The JAX ``tl.record`` producer: lean per-equation events, values only when selected.

The producer satisfies :class:`torchlens.recording._backend_events.BackendRecordProducer`.
Each pass derives the closed jaxpr the way ``tl.trace(..., backend="jax")`` does
and interprets it with an equation hook. For every source leaf and every
interpreted equation the hook emits one metadata-only ``BackendOpEvent`` whose
raw label is the label ``tl.trace`` gives that op (same reservation order:
argument leaves, closure constants, then captures in interpreter order). A value
reaches the sink only when the sink requests it. When a decision says stop, the
hook unwinds the interpreter, so no equation after the stop frontier binds.

No ``Trace`` is built: the producer never finalizes, labels or validates a graph,
and no capture keeps its values (``_retention.KEEP_NONE``), so the interpreter
frees each intermediate after its last use; only values the sink selected stay.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..._ir.capture_events import CaptureEvents
from ...recording._backend_events import BackendOpEvent, RecordSink
from ._edits import _REGION_LAYER_TYPES
from ._input_kwargs import bind_input_kwargs
from ._interp_hooks import InterpreterHooks
from ._module_frames import jax_event_module_frames
from ._retention import KEEP_NONE, set_retention
from .jaxpr import flatten_dynamic_args, interpret_closed_jaxpr_with_inlining
from .modules import discover_equinox_module_tree, discover_nnx_module_tree

_JAX_CONTROL_FLOW = "unroll"
_JAX_MAX_CONTROL_FLOW_UNROLL = 64


class _StopRecording(Exception):
    """Raised inside the equation hook to unwind the interpreter at a stop."""


class _PassRetention:
    """Retention key of one record pass: no capture keeps its values.

    The hook hands every selected value to the sink before the capture is
    stored, so the interpreter may free each intermediate after its last use.
    """


class JaxPayloadCodec:
    """Payload conversions for selected JAX values."""

    def ram_copy(self, value: Any) -> Any:
        """Return ``value``: JAX arrays are immutable, so the array is its own copy."""

        return value

    def to_disk_tensor(self, value: Any) -> torch.Tensor:
        """Return a CPU ``torch.Tensor`` holding the same numbers as ``value``.

        Parameters
        ----------
        value
            One JAX array (a multi-result equation's tuple is refused).

        Returns
        -------
        torch.Tensor
            CPU tensor with the matching dtype (bfloat16 round-trips through float32).
        """

        import numpy as np

        if isinstance(value, tuple):
            raise TypeError(
                "a multi-result JAX equation has a tuple value; the disk bundle stores one "
                "array per record"
            )
        host = np.asarray(value)
        if str(host.dtype) == "bfloat16":
            return torch.from_numpy(host.astype(np.float32)).to(torch.bfloat16)
        return torch.from_numpy(np.array(host, copy=True))


@dataclass(frozen=True)
class _Prepared:
    """A model discovered once per recorder."""

    model: Any
    module_tree: Any


class JaxRecordProducer:
    """``tl.record`` producer for JAX callables and Equinox or Flax NNX module roots."""

    backend = "jax"
    supported_options: frozenset[str] = frozenset()

    def __init__(self) -> None:
        """Set up the codec and the per-recorder counters tests read."""

        self.codec = JaxPayloadCodec()
        self.passes = 0
        self.equations_bound = 0

    def prepare(self, model: Any) -> _Prepared:
        """Discover the module tree of ``model`` once (``None`` for a raw function)."""

        from .backend import _reject_transformed_callable

        _reject_transformed_callable(model)
        tree = discover_equinox_module_tree(model)
        if tree is None:
            tree = discover_nnx_module_tree(model)
        return _Prepared(model=model, module_tree=tree)

    def run_pass(
        self,
        prepared: _Prepared,
        input_args: Any,
        input_kwargs: Mapping[str, Any],
        sink: RecordSink,
    ) -> Any:
        """Interpret one forward, emitting events; return the output, or ``None`` after a stop."""

        from . import _module_return_probe as return_probe
        from .backend import JAXBackend, _derive_and_run, _reject_tracer_inputs

        self.passes += 1
        backend = JAXBackend()
        args = backend._normalize_input_args(input_args)
        _reject_tracer_inputs(args)
        call_fn, args = bind_input_kwargs(prepared.model, args, dict(input_kwargs) or None)
        tree = prepared.module_tree
        closed_jaxpr, consts, output = _derive_and_run(call_fn, args, (), tree)
        flat_args, _treedef = flatten_dynamic_args(args, ())
        emitter = _EventEmitter(sink, self)
        retention = _PassRetention()
        set_retention(retention, KEEP_NONE)
        try:
            emitter.emit_sources(backend, args, consts)
            result = interpret_closed_jaxpr_with_inlining(
                return_probe.jaxpr_to_interpret(tree, closed_jaxpr),
                flat_args,
                jax_control_flow=_JAX_CONTROL_FLOW,
                jax_max_control_flow_unroll=_JAX_MAX_CONTROL_FLOW_UNROLL,
                equation_hook=emitter.hook,
                hooks=InterpreterHooks(session=retention),
            )
            emitter.emit_pending(result.captures)
        except _StopRecording:
            return None
        return output


class _EventEmitter:
    """Equation hook that reserves ``tl.trace`` labels and emits one event per op."""

    def __init__(self, sink: RecordSink, producer: JaxRecordProducer) -> None:
        """Bind the sink and a fresh label reservation."""

        self.sink = sink
        self.producer = producer
        self.labels = CaptureEvents()
        self.seen = 0

    def emit_sources(self, backend: Any, args: Sequence[Any], consts: Sequence[Any]) -> None:
        """Emit argument leaves as inputs and reserve closure-constant labels."""

        for _path, value in backend._tree_leaves_with_paths(tuple(args)):
            if backend.is_tensor(value):
                self._emit("input", "input", value, (), (), kind="input")
        for _value in consts:
            self.labels.reserve_label("constant")

    def hook(
        self,
        captures: Sequence[Any],
        eqn: Any,
        outputs: tuple[Any, ...],
        stacks: tuple[tuple[str, ...], tuple[tuple[str, int], ...]],
    ) -> tuple[Any, ...]:
        """Emit pending region captures, then this equation; unwind on a stop."""

        self.emit_pending(captures)
        self.seen += 1
        self.producer.equations_bound += 1
        self._emit(eqn.primitive.name, eqn.primitive.name, outputs, stacks, ())
        return outputs

    def emit_pending(self, captures: Sequence[Any]) -> None:
        """Emit the captures the interpreter appended without calling the hook (regions)."""

        for capture in captures[self.seen :]:
            layer_type = _REGION_LAYER_TYPES.get(capture.kind, capture.primitive)
            capture_stacks = (capture.module_stack, capture.module_call_stack)
            self._emit(layer_type, capture.primitive, capture.output_values, capture_stacks, ())
        self.seen = max(self.seen, len(captures))

    def _emit(
        self,
        layer_type: str,
        func_name: str,
        value: Any,
        stacks: Any,
        parents: tuple[str, ...],
        *,
        kind: str = "op",
    ) -> None:
        """Emit one event, deliver its value when requested, unwind when told to stop."""

        reserved = self.labels.reserve_label(layer_type)
        if isinstance(value, tuple):
            value = value[0] if len(value) == 1 else value
        frames = jax_event_module_frames(*stacks)[1] if stacks else ()
        shape = getattr(value, "shape", None)
        event = BackendOpEvent(
            raw_label=reserved.label_raw,
            layer_type=reserved.layer_type,
            type_index=reserved.type_index,
            raw_index=reserved.raw_index,
            parent_labels=parents,
            func_name=func_name,
            shape=None if shape is None else tuple(shape),
            dtype=getattr(value, "dtype", None),
            device=getattr(value, "device", None),
            module_stack=tuple(frames),
            func_call_id=reserved.raw_index,
            kind=kind,
        )
        decision = self.sink.emit(event)
        if decision.request is not None:
            self.sink.deliver(decision.request, value)
        if decision.stop:
            raise _StopRecording


def jax_record_producer() -> JaxRecordProducer:
    """Return a fresh JAX record producer (the ``fastlog`` capability binding)."""

    return JaxRecordProducer()
