"""Native backward capture on TensorFlow.

The capture holds one persistent ``tf.GradientTape`` over the eager forward
(:func:`backward_ready_scope`, which the TensorFlow capture enters around its
op-callback session) and keeps it on the trace (:func:`retain_backward_state`).
:func:`record_backward` then runs ``tape.gradient`` once with two observers
installed for that call only: an op callback that sees every raw backward op,
and a wrapper around the tape's Python gradient entry, so each gradient rule
the tape executes is one frame (``_tf_frames``). Each frame is linked to the
forward op it differentiates and journaled as the events PyTorch's backward
capture emits; ``backends/_backward_neutral.py`` folds them into ``GradFn``,
``GradFnCall``, ``BackwardPass``, ``Op.grad`` and ``Param`` gradients.

Meanings follow PyTorch: one ``GradFn`` per executed gradient rule, named by
TensorFlow's own name for the rule (the forward op type, or the name a
``tf.custom_gradient`` function was recorded under); one ``GradFnCall`` per
executed rule, with the incoming gradient per forward output
(``grad_outputs``) and the outgoing gradient per parent occurrence of the op
(``grad_inputs``); ``Op.grad`` is the gradient that reached the op's output.
Nothing unobserved is filled: no peak memory, no engine flags.

The tape's gradient entry is private TensorFlow API (``backprop._gradient_function``
re-registered through ``pywrap_tfe.TFE_Py_RegisterGradientFunction``); it is
found by feature detection in :func:`_private_bindings`, and a missing binding
refuses typed, naming it. TensorFlow is imported inside functions only.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from ..._deprecations import MISSING
from ..._ir.events import (
    BackwardPassEnd,
    BackwardPassStart,
    GradFnDiscovered,
    GradFnFired,
    OpGradObserved,
    ParamGradObserved,
)
from ...errors import RunCapabilityUnavailableError
from .._backward_neutral import journal_backward_event, next_backward_pass_index, set_pass_roots
from ._tf_frames import (
    BackwardRecorder,
    ForwardCall,
    Frame,
    flowed_ids,
    frame_consumers,
    link_frames,
    tensor_id,
)

__all__ = ("backward_ready_scope", "record_backward", "retain_backward_state")

_STATE_KEY = "_tf_backward_state"
_SESSION_TAPE = "_tl_backward_tape"
_CODE = "run_capability_unavailable"  # RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE
_DETECTION_STAGE = "native_backward_capture"
_PREFIX = "TensorFlow backward capture is unavailable: "
_SOURCE_FIELDS = (
    "class_source_file",
    "class_source_line",
    "init_source_file",
    "init_source_line",
    "forward_source_file",
    "forward_source_line",
    "backward_source_file",
    "backward_source_line",
)

#: The recorder of the backward pass running now, read by Python backward
#: functions recorded during the forward (``tf.custom_gradient``).
_ACTIVE: list[BackwardRecorder] = []


# --- private TensorFlow bindings (feature detection) ------------------------


def _private_bindings() -> tuple[SimpleNamespace | None, str | None]:
    """Return the private bindings backward capture needs, or the first missing one.

    Feature detection only, never version parsing: the tape's gradient entry
    (``backprop._gradient_function``), its registration
    (``pywrap_tfe.TFE_Py_RegisterGradientFunction``), Python-recorded backward
    functions (``record.record_operation``) and eager op callbacks.
    """

    try:
        from tensorflow.python import pywrap_tfe
        from tensorflow.python.eager import backprop, record
        from tensorflow.python.framework import op_callbacks
    except ImportError as exc:
        return None, f"tensorflow.python private modules ({exc})"
    wanted = (
        (backprop, "_gradient_function", "backprop._gradient_function"),
        (pywrap_tfe, "TFE_Py_RegisterGradientFunction", "pywrap_tfe.TFE_Py_RegisterGradientFunction"),
        (record, "record_operation", "record.record_operation"),
        (op_callbacks, "add_op_callback", "op_callbacks.add_op_callback"),
        (op_callbacks, "remove_op_callback", "op_callbacks.remove_op_callback"),
    )  # fmt: skip
    for module, attribute, name in wanted:
        if not hasattr(module, attribute):
            return None, name
    bindings = SimpleNamespace(
        backprop=backprop, pywrap_tfe=pywrap_tfe, record=record, op_callbacks=op_callbacks
    )
    return bindings, None


def _require_bindings() -> SimpleNamespace:
    """The private bindings, or a typed refusal naming the missing one."""

    bindings, missing = _private_bindings()
    if bindings is None:
        raise RunCapabilityUnavailableError(
            _PREFIX + f"this TensorFlow does not expose {missing}, which it needs.",
            code=_CODE,
            detection_stage=_DETECTION_STAGE,
        )
    return bindings


# --- forward: the persistent tape -------------------------------------------


def _float_tensors(tf: Any, value: Any) -> list[Any]:
    """Floating-point TensorFlow tensors in ``value`` (nested lists, tuples, dicts)."""

    found: list[Any] = []
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, Mapping):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
        elif tf.is_tensor(item) and getattr(item.dtype, "is_floating", False):
            found.append(item)
    return found


def _framed_backward(
    op_type: Any, inputs: Sequence[Any], outputs: Sequence[Any], backward: Callable[..., Any]
) -> Callable[..., Any]:
    """A Python backward function that runs as a frame while a pass is recorded."""

    def framed(*grads: Any, **kwargs: Any) -> Any:
        recorder = _ACTIVE[-1] if _ACTIVE else None
        if recorder is None or not recorder.owns_thread():
            return backward(*grads, **kwargs)
        return recorder.run_frame(
            "record_operation", op_type, inputs, outputs, grads, lambda: backward(*grads, **kwargs)
        )

    return framed


@contextlib.contextmanager
def backward_ready_scope(session: Any) -> Iterator[Any]:
    """Hold a persistent gradient tape over the forward ``session`` runs.

    Trainable variables are watched as the forward reads them, and every
    floating-point input tensor is watched, as TensorFlow's derived gradients
    do. Python backward functions recorded during the forward
    (``tf.custom_gradient``) are wrapped so a recorded pass sees them as frames.
    On success the tape is left on ``session`` for :func:`retain_backward_state`.
    """

    bindings = _require_bindings()
    tf = session.tf
    record = bindings.record
    original = record.record_operation

    def recording(
        op_type: Any, outputs: Any, inputs: Any, backward: Any, *rest: Any, **kw: Any
    ) -> Any:
        framed = _framed_backward(op_type, list(inputs), list(outputs), backward)
        return original(op_type, outputs, inputs, framed, *rest, **kw)

    tape = tf.GradientTape(persistent=True, watch_accessed_variables=True)
    record.record_operation = recording
    try:
        with tape:
            for tensor in _float_tensors(tf, (session.args, session.kwargs)):
                tape.watch(tensor)
            yield tape
    finally:
        record.record_operation = original
    setattr(session, _SESSION_TAPE, tape)


@dataclass
class TFBackwardState:
    """What a backward-ready TensorFlow capture keeps for later backward passes.

    Attributes
    ----------
    tf:
        The TensorFlow module.
    tape:
        The persistent tape that recorded the forward.
    sources:
        Raw label of each recorded source (input or constant) and its tensor.
    nodes:
        Stable identity objects of gradient rules no captured op owns.
    losses:
        Loss tensor of each recorded pass, by pass number.
    """

    tf: Any
    tape: Any
    sources: dict[str, Any]
    nodes: dict[tuple[Any, ...], object] = field(default_factory=dict)
    losses: dict[int, Any] = field(default_factory=dict)


def retain_backward_state(trace: Any, session: Any) -> None:
    """Keep the forward's tape and source tensors on ``trace`` (no-op without a tape)."""

    tape = getattr(session, _SESSION_TAPE, None)
    if tape is None:
        return
    sources = {
        record.label_raw: record.tensor
        for record in getattr(session, "source_records", ())
        if getattr(record, "label_raw", None)
    }
    trace.__dict__[_STATE_KEY] = TFBackwardState(tf=session.tf, tape=tape, sources=sources)


def _state_of(trace: Any) -> TFBackwardState:
    """The trace's backward state, or a typed refusal."""

    state = trace.__dict__.get(_STATE_KEY)
    if not isinstance(state, TFBackwardState):
        raise RunCapabilityUnavailableError(
            _PREFIX + "this trace kept no gradient tape. Capture it with backward_ready=True to "
            "call backward on it.",
            code=_CODE,
            detection_stage=_DETECTION_STAGE,
        )
    return state


# --- one backward pass -------------------------------------------------------


def _retains(policy: Any) -> bool:
    """Whether gradient values are kept under ``policy`` (all or none on TensorFlow)."""

    if policy is None or policy is False or policy == "none" or policy == []:
        return False
    if policy is True or policy == "all":
        return True
    raise RunCapabilityUnavailableError(
        _PREFIX + f"save_grads={policy!r} selects ops, which TensorFlow backward capture does not "
        "support yet. Pass save_grads=True to keep every gradient, or False to keep none.",
        code=_CODE,
        detection_stage=_DETECTION_STAGE,
    )


def _loss_tensor(trace: Any, state: TFBackwardState, loss: Any) -> Any:
    """The scalar to differentiate: ``loss`` itself, or ``loss(trace.output)`` on the tape."""

    tf = state.tf
    if callable(loss) and not tf.is_tensor(loss):
        with state.tape:
            value = loss(trace.output)
    else:
        value = loss
    if not tf.is_tensor(value):
        raise RunCapabilityUnavailableError(
            _PREFIX + f"the loss must be a TensorFlow tensor; got {type(value).__name__}.",
            code=_CODE,
            detection_stage=_DETECTION_STAGE,
        )
    if value.shape.num_elements() != 1:
        raise RunCapabilityUnavailableError(
            _PREFIX + f"the loss must be a scalar (one element); got shape {tuple(value.shape)}. "
            "Reduce it first, as a backward pass starts from one value.",
            code=_CODE,
            detection_stage=_DETECTION_STAGE,
        )
    return value


def _variable_of(ref: Any) -> Any | None:
    """The ``tf.Variable`` behind a parameter reference (Keras 3 wraps it in ``.value``).

    Same reading as the TensorFlow derived-gradient leaves.
    """

    inner = getattr(ref, "value", None)
    if inner is not None and not callable(inner) and hasattr(inner, "dtype"):
        return inner
    return ref if ref is not None and hasattr(ref, "dtype") else None


def _param_targets(trace: Any, state: TFBackwardState) -> list[tuple[str, Any]]:
    """``(address, variable)`` of each parameter the tape watched."""

    watched = {id(variable) for variable in state.tape.watched_variables()}
    targets = []
    for address, param in (trace._param_logs or {}).items():
        variable = _variable_of(getattr(param, "_param_ref", None))
        if variable is not None and id(variable) in watched:
            targets.append((str(address), variable))
    return targets


def _input_targets(state: TFBackwardState) -> list[tuple[str, Any]]:
    """``(raw label, tensor)`` of each floating-point source the tape watched."""

    tf = state.tf
    return [
        (label, tensor)
        for label, tensor in state.sources.items()
        if tf.is_tensor(tensor) and getattr(tensor.dtype, "is_floating", False)
    ]


@contextlib.contextmanager
def _observing(bindings: SimpleNamespace, recorder: BackwardRecorder) -> Iterator[None]:
    """Install the op callback and the framed gradient entry for one ``tape.gradient``."""

    original = bindings.backprop._gradient_function

    def framed(*args: Any) -> Any:
        if not recorder.owns_thread():
            return original(*args)
        op_name, _attrs, _num_inputs, inputs, outputs, out_grads = args[:6]
        return recorder.run_frame(
            "gradient_function", op_name, inputs, outputs, out_grads, lambda: original(*args)
        )

    bindings.op_callbacks.add_op_callback(recorder.callback)
    bindings.pywrap_tfe.TFE_Py_RegisterGradientFunction(framed)
    _ACTIVE.append(recorder)
    try:
        yield
    finally:
        _ACTIVE.remove(recorder)
        bindings.pywrap_tfe.TFE_Py_RegisterGradientFunction(original)
        bindings.op_callbacks.remove_op_callback(recorder.callback)


def record_backward(
    trace: Any, loss: Any, *, save_grads: Any = MISSING, **backward_kwargs: Any
) -> None:
    """Run one TensorFlow backward pass from ``loss`` and journal what it executed.

    Parameters
    ----------
    trace:
        A TensorFlow trace captured backward-ready.
    loss:
        A scalar tensor computed on the capture's tape (a model output), or a
        callable mapping ``trace.output`` to that scalar; the callable runs on
        the capture's tape.
    save_grads:
        ``True``/``"all"`` keeps every gradient value, ``False``/``None`` keeps
        none (metadata only); defaults to the trace's ``save_grads``.
    **backward_kwargs:
        Refused: PyTorch's backward keywords have no TensorFlow meaning.
    """

    if backward_kwargs:
        raise RunCapabilityUnavailableError(
            _PREFIX + f"backward keyword(s) {sorted(backward_kwargs)} have no TensorFlow meaning. "
            "Call backward(loss) with save_grads= only.",
            code=_CODE,
            detection_stage=_DETECTION_STAGE,
        )
    state = _state_of(trace)
    bindings = _require_bindings()
    policy = trace._options_save_grads if save_grads is MISSING else save_grads
    retain = _retains(policy)
    _run_pass(trace, state, bindings, _loss_tensor(trace, state, loss), policy, retain)


def _run_pass(
    trace: Any,
    state: TFBackwardState,
    bindings: SimpleNamespace,
    loss_tensor: Any,
    policy: Any,
    retain: bool,
) -> None:
    """Run ``tape.gradient`` once under the observers and journal the pass."""

    seed = state.tf.ones_like(loss_tensor)
    inputs, params = _input_targets(state), _param_targets(trace, state)
    pass_index = next_backward_pass_index(trace)
    journal_backward_event(trace, _start_event(pass_index, loss_tensor, policy))
    recorder = BackwardRecorder()
    started = time.time()
    status = "error"
    try:
        with _observing(bindings, recorder):
            grads = state.tape.gradient(
                loss_tensor,
                [tensor for _, tensor in inputs] + [variable for _, variable in params],
                output_gradients=seed,
            )
        if not recorder.frames:
            raise RunCapabilityUnavailableError(
                _PREFIX + "the loss is not connected to the captured forward on its tape. Pass a "
                "callable of trace.output (it runs on the tape), or a scalar model output.",
                code=_CODE,
                detection_stage=_DETECTION_STAGE,
            )
        state.losses[pass_index] = loss_tensor
        journal = _PassJournal(trace, state, recorder, pass_index, retain, (loss_tensor, seed))
        journal.write(inputs, params, list(grads))
        status = "ok"
    finally:
        journal_backward_event(
            trace,
            BackwardPassEnd(
                pass_index=pass_index,
                duration=time.time() - started,
                peak_memory=None,
                status="ok" if status == "ok" else "error",
                order_attribution_coverage=None,
            ),
        )


def _start_event(pass_index: int, loss: Any, policy: Any) -> BackwardPassStart:
    """The pass's start event; fields TensorFlow does not observe stay ``None``."""

    return BackwardPassStart(
        pass_index=pass_index,
        trigger="backward",
        implicit=False,
        outer_context=None,
        call_context_ref=None,
        root_meta=(
            {
                "shape": tuple(int(dim) for dim in loss.shape),
                "dtype": loss.dtype.name,
                "device": str(loss.device),
            },
        ),
        root_grad_arguments=None,
        inputs_subset=(),
        order=None,
        origin_backward_pass=None,
        save_grads_policy_repr=repr(policy),
        engine_flags=None,
        forward_op_count_at_trigger=None,
        timestamp=time.time(),
    )


# --- journaling ---------------------------------------------------------------


def forward_calls(trace: Any) -> list[ForwardCall]:
    """Captured forward calls, one per raw TensorFlow op call, in execution order.

    The capture keeps one record per output; outputs of one call share the
    call's input records and count up from output slot 0.
    """

    calls: list[ForwardCall] = []
    group: list[Any] = []

    def close() -> None:
        if group:
            inputs = sorted(group[0].inputs, key=lambda item: item.input_index)
            calls.append(
                ForwardCall(
                    index=len(calls),
                    op_type=group[0].op_type,
                    input_ids=tuple(tensor_id(item.tensor) for item in inputs),
                    output_ids=tuple(tensor_id(c.output_tensor) for c in group),
                    captures=tuple(group),
                )
            )
            group.clear()

    for capture in getattr(trace, "_tf_op_captures", ()) or ():
        continues = (
            group
            and capture.output_index == len(group)
            and capture.inputs is group[0].inputs
            and capture.op_type == group[0].op_type
        )
        if not continues:
            close()
        group.append(capture)
    close()
    return calls


def _final_label(trace: Any, raw: str) -> str:
    """The op label a raw capture label finalized to."""

    for name in ("_raw_to_final_layer_labels", "_raw_to_final_op_labels"):
        mapping = getattr(trace, name, None)
        if isinstance(mapping, dict) and raw in mapping:
            return str(mapping[raw])
    return raw


def _parent_slots(call: ForwardCall) -> list[int]:
    """Callback input slot of each parent occurrence (recorded producer or source)."""

    capture = call.captures[0]
    ordered = sorted(capture.inputs, key=lambda item: item.input_index)
    return [
        item.input_index for item in ordered if item.producer_label_raw or item.source_label_raw
    ]


class _PassJournal:
    """Turns one pass's frames into backward events on the trace's stream."""

    def __init__(
        self,
        trace: Any,
        state: TFBackwardState,
        recorder: BackwardRecorder,
        pass_index: int,
        retain: bool,
        seeded: tuple[Any, Any],
    ) -> None:
        self.trace = trace
        self.state = state
        self.frames = recorder.frames
        self.pass_index = pass_index
        self.retain = retain
        loss_id, seed_id = (tensor_id(item) for item in seeded)
        self.seed_ids = {seed_id} if seed_id is not None else set()
        seeds = {loss_id: seed_id} if loss_id is not None and seed_id is not None else {}
        self.calls = forward_calls(trace)
        self.links = link_frames(self.calls, self.frames, recorder.events, seeds)
        self.flowed = flowed_ids(self.frames, recorder.events, self.seed_ids)
        self.consumers = frame_consumers(self.frames, recorder.events)
        self.node_ids = [self._node_id(frame) for frame in self.frames]

    def _node_id(self, frame: Frame) -> int:
        """Identity of the rule's node: stable per forward op across passes."""

        linked = self.links[frame.index]
        if linked is not None:
            return id(self.calls[linked].captures[0])
        key = (frame.kind, frame.op_type, frame.input_ids, frame.output_ids)
        return id(self.state.nodes.setdefault(key, object()))

    def _snapshot(self, value: Any) -> Any:
        """A kept gradient value (NumPy, as TensorFlow payloads are), or ``None``."""

        if value is None or not self.retain:
            return None
        return self.state.tf.convert_to_tensor(value).numpy().copy()

    def write(
        self,
        inputs: list[tuple[str, Any]],
        params: list[tuple[str, Any]],
        grads: list[Any],
    ) -> None:
        """Journal discoveries, calls, op and parameter gradients, and the roots."""

        for frame in self.frames:
            journal_backward_event(self.trace, self._discovery(frame))
        for frame in self.frames:
            journal_backward_event(self.trace, self._fired(frame))
            self._op_grads(frame)
        stamp = time.time()
        for (label, _tensor), grad in zip(inputs, grads[: len(inputs)], strict=True):
            self._op_grad_event(_final_label(self.trace, label), grad, stamp)
        for (address, _variable), grad in zip(params, grads[len(inputs) :], strict=True):
            self._param_grad_event(address, grad, stamp)
        roots = [
            self.node_ids[frame.index]
            for frame in self.frames
            if any(item in self.seed_ids for item in frame.out_grad_ids)
        ]
        set_pass_roots(self.trace, self.pass_index, roots)

    def _discovery(self, frame: Frame) -> GradFnDiscovered:
        linked = self.links[frame.index]
        op_label = (
            None
            if linked is None
            else _final_label(self.trace, self.calls[linked].captures[0].label_raw)
        )
        topology = tuple(
            self.node_ids[self.consumers[item]]
            for item in frame.result_ids
            if item is not None and item in self.consumers
        )
        return GradFnDiscovered(
            object_id=self.node_ids[frame.index],
            class_name=frame.op_type,
            class_qualname=frame.op_type,
            is_custom=frame.kind == "record_operation",
            op_label=op_label,
            param_ref=None,
            created_in_pass=None,
            creator_object_id=None,
            source=dict.fromkeys(_SOURCE_FIELDS),
            topology=topology,
        )

    def _fired(self, frame: Frame) -> GradFnFired:
        linked = self.links[frame.index]
        results = frame.results
        if linked is not None:
            slots = _parent_slots(self.calls[linked])
            results = tuple(results[slot] if slot < len(results) else None for slot in slots)
        return GradFnFired(
            object_id=self.node_ids[frame.index],
            pass_index=self.pass_index,
            grad_input_refs=tuple(self._snapshot(item) for item in results)
            if self.retain
            else None,
            grad_output_refs=(
                tuple(self._snapshot(item) for item in frame.out_grads) if self.retain else None
            ),
            fire_record=None,
            timestamp=frame.timestamp,
            fire_started_monotonic=frame.started,
            fire_finished_monotonic=frame.finished,
        )

    def _op_grads(self, frame: Frame) -> None:
        """``Op.grad`` of each output of the frame's forward op that a gradient reached."""

        linked = self.links[frame.index]
        if linked is None:
            return
        for slot, capture in enumerate(self.calls[linked].captures):
            if slot >= len(frame.out_grads):
                break
            grad = frame.out_grads[slot]
            if tensor_id(grad) in self.flowed:
                self._op_grad_event(
                    _final_label(self.trace, capture.label_raw), grad, frame.timestamp
                )

    def _op_grad_event(self, label: str, grad: Any, stamp: float) -> None:
        if grad is None:
            return
        tensor = self.state.tf.convert_to_tensor(grad)
        journal_backward_event(
            self.trace,
            OpGradObserved(
                op_label=label,
                pass_index=self.pass_index,
                payload_ref=self._snapshot(tensor),
                transformed_payload_ref=None,
                shape=tuple(int(dim) for dim in tensor.shape),
                dtype=tensor.dtype.name,
                memory=int(tensor.shape.num_elements() or 0) * tensor.dtype.size,
                timestamp=stamp,
            ),
        )

    def _param_grad_event(self, address: str, grad: Any, stamp: float) -> None:
        if grad is None:
            return
        tensor = self.state.tf.convert_to_tensor(grad)
        journal_backward_event(
            self.trace,
            ParamGradObserved(
                param_address=address,
                pass_index=self.pass_index,
                payload_ref=self._snapshot(tensor),
                shape=tuple(int(dim) for dim in tensor.shape),
                dtype=tensor.dtype.name,
                memory=int(tensor.shape.num_elements() or 0) * tensor.dtype.size,
                timestamp=stamp,
            ),
        )
