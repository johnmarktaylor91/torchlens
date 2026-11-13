"""Pre-forward source leaves of a Paddle capture: registered buffers and closure constants.

An op input that is not an input leaf, not a parameter and not the output of a
recorded op is one of two kinds of pre-forward tensor:

* a registered buffer (``Layer.register_buffer``) of the captured layer tree.
  It is recorded the way torch records a buffer: one parentless source op of
  type ``buffer`` with a saved payload, whose buffer address is the
  ``named_buffers`` name (``block.scale``). The tensor is labelled, so every
  later read of the same buffer is an edge from that one node, and the trace
  lists it in ``Trace.buffers``. A buffer is credited only while it still
  holds its pre-forward value: a write the capture did not see (``set_value``,
  ``paddle.assign``) leaves the read unlabelled, so validation fails closed on
  it, and the capture adds a trace marker for every buffer the forward changed.
* a closure constant (``constants_rule``), recorded as before: a ``constant``
  op with a saved payload and a validation record.

This module imports no Paddle at import time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from ..._ir.capture_events import CaptureEvents
from ..._ir.intervention import FunctionEventInput
from ._param_writes import PARAMETER_WRITE_MARKER, ParameterWriteGuard
from .constants_rule import PADDLE_CONSTANT_OP, find_closure_tensors

__all__ = ["PADDLE_BUFFER_OP", "PaddleSourceLeaves", "record_source_leaves"]

#: Layer type of a recorded registered buffer (torch's ``buffer`` source role).
PADDLE_BUFFER_OP = "buffer"


@dataclass
class PaddleSourceLeaves:
    """The pre-forward tensors one capture may record as source leaves.

    Attributes
    ----------
    closure:
        Closure tensors proven to exist before the forward, keyed by ``id``.
    buffers:
        ``id(buffer) -> (address, buffer)`` for the captured tree's registered
        buffers; held strongly so the ids stay valid for the capture.
    buffer_guard:
        Pre-forward fingerprints of those buffers.
    initial_values:
        ``address -> pre-forward copy`` of each buffer, taken before the
        wrappers are installed (``Trace._buffer_initial_values`` channel).
    capture_cls:
        ``PaddleOpCapture``, the validation record type of a closure
        constant, passed in so this module does not import the backend
        module that imports it.
    """

    closure: dict[int, Any] = field(default_factory=dict)
    buffers: dict[int, tuple[str, Any]] = field(default_factory=dict)
    buffer_guard: ParameterWriteGuard = field(default_factory=ParameterWriteGuard)
    initial_values: dict[str, Any] = field(default_factory=dict)
    capture_cls: type | None = None

    @classmethod
    def capture(
        cls,
        model: object,
        paddle: Any,
        param_address_by_id: Mapping[int, str],
        capture_cls: type,
    ) -> PaddleSourceLeaves:
        """Find the source leaves of ``model`` before the forward runs."""

        layer_type = getattr(getattr(paddle, "nn", None), "Layer", None)
        buffers: dict[int, tuple[str, Any]] = {}
        named_buffers = getattr(model, "named_buffers", None)
        if layer_type is not None and isinstance(model, layer_type) and callable(named_buffers):
            for address, buffer in named_buffers():
                if isinstance(buffer, paddle.Tensor) and id(buffer) not in param_address_by_id:
                    buffers.setdefault(id(buffer), (address, buffer))
        closure = find_closure_tensors(
            model, paddle.Tensor, layer_type, exclude_ids={*param_address_by_id, *buffers}
        )
        initial_values = {address: buffer.clone() for address, buffer in buffers.values()}
        guard = ParameterWriteGuard.of_tensors(buffers)
        return cls(closure, buffers, guard, initial_values, capture_cls)

    def written_markers(self, trace: Any) -> list[str]:
        """Trace markers for every registered buffer the forward changed."""

        return [
            f"{PARAMETER_WRITE_MARKER}: buffer {address}"
            for address in self.buffer_guard.written_addresses(trace)
        ]


def record_source_leaves(
    backend: Any,
    trace: Any,
    events: CaptureEvents,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Record each unlabeled buffer or proven closure constant among an op's inputs.

    Parameters
    ----------
    backend:
        The active ``PaddleBackend`` (its ``_source_leaves``, label store and
        event builder).
    trace:
        The live trace.
    events:
        The capture's event accumulator.
    args, kwargs:
        The op call's arguments.
    """

    leaves: PaddleSourceLeaves = backend._source_leaves
    if not (leaves.closure or leaves.buffers):
        return
    for _path, tensor in (
        *backend._iter_tensors_with_paths(args, root=("args",)),
        *backend._iter_tensors_with_paths(kwargs, root=("kwargs",)),
    ):
        if backend.tensor_store.get_label(tensor) is not None:
            continue
        buffer = leaves.buffers.get(id(tensor))
        if buffer is not None and buffer[1] is tensor:
            if not leaves.buffer_guard.was_written(tensor, trace):
                _emit_buffer(backend, trace, events, buffer[0], tensor)
            continue
        if leaves.closure.get(id(tensor)) is not tensor:
            continue
        if id(tensor) in backend._param_address_by_id:
            continue
        _emit_constant(backend, trace, events, tensor, leaves.capture_cls)


def _event_input(
    func_name: str, tensor: object, func_call_id: int, module_stack: tuple[Any, ...]
) -> FunctionEventInput:
    """The function-event input of a parentless source leaf."""

    return FunctionEventInput(
        func=None,
        func_name=func_name,
        func_qualname=None,
        args=(),
        kwargs={},
        raw_output=tensor,
        arg_copies=None,
        kwarg_copies=None,
        module_stack=module_stack,
        is_bottom_level_func=True,
        func_call_id=func_call_id,
        expected_output_count=1,
    )


def _emit_buffer(
    backend: Any, trace: Any, events: CaptureEvents, address: str, tensor: object
) -> None:
    """Append one registered-buffer source event and label the buffer tensor."""

    site = events.reserve_label_block(PADDLE_BUFFER_OP, 1)[0]
    func_call_id = events.func_call_id_counter + 1
    events.func_call_id_counter = func_call_id
    event = backend._build_event(
        session=trace,
        kind="source",
        reserved=site,
        func_event_input=_event_input(PADDLE_BUFFER_OP, tensor, func_call_id, ()),
        output=tensor,
        parents=(),
        parent_arg_positions={"args": {}, "kwargs": {}},
        edge_uses=(),
        policy=backend._capture_policy(trace),
        is_input=False,
        container_path=(),
    )
    # The shared buffer-address join reads the recorded address from the
    # equivalence class (``buffer_{address}``; a buffer source has no module
    # stack, so no suffix) and keeps it when it names a registered buffer.
    events.extend((replace(event, equivalence_class=f"buffer_{address}"),))
    backend.tensor_store.set_label(tensor, site.label_raw)
    trace._buffer_initial_values.setdefault(address, backend._source_leaves.initial_values[address])


def _emit_constant(
    backend: Any, trace: Any, events: CaptureEvents, tensor: object, capture_cls: Any
) -> None:
    """Append one closure-constant op event and its validation record."""

    site = events.reserve_label_block(PADDLE_CONSTANT_OP, 1)[0]
    func_call_id = events.func_call_id_counter + 1
    events.func_call_id_counter = func_call_id
    module_stack = tuple(getattr(trace, "_paddle_module_stack", ()))
    event = backend._build_event(
        session=trace,
        kind="op",
        reserved=site,
        func_event_input=_event_input(PADDLE_CONSTANT_OP, tensor, func_call_id, module_stack),
        output=tensor,
        parents=(),
        parent_arg_positions={"args": {}, "kwargs": {}},
        edge_uses=(),
        policy=backend._capture_policy(trace),
        is_input=False,
        container_path=(),
    )
    trace._paddle_op_captures.append(
        capture_cls(
            func=None,
            op_name=PADDLE_CONSTANT_OP,
            label_raw=site.label_raw,
            args_template=(),
            kwargs_template={},
            tensor_inputs=(),
            output_leaf_paths=((),),
            producer_labels=frozenset(),
        )
    )
    events.extend((event,))
    backend.tensor_store.set_label(tensor, site.label_raw)
