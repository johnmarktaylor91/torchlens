"""Native backward capture on Paddle: the dygraph grad-node graph, walked from the loss.

Paddle's eager mode builds one grad node per differentiable call
(``Tensor.grad_fn``: ``name()``, a stable ``node_ptr()`` and ``next_functions``
in gradient-slot order). Before the backward runs, the graph is walked from the
loss; during it, ``GradNodeBase._register_post_hook`` reports each executed node
once with (outgoing gradient per input slot, incoming gradient per output slot),
and tensor hooks report the summed gradient of every recorded tensor.

Records keep PyTorch's meanings (``backends/_backward_neutral.py``): one
``GradFn`` per grad node, named by Paddle's own rule name (``MatmulGradNode``,
``GradNodeAccumulation``, ``GradNodePyLayer_<name>``); one ``GradFnCall`` per
observed execution; ``Op.grad`` from the tensor hooks; parameter gradients by
address. Each recorded op's call carries, as the backward validator reads it
(``validation/_backward_neutral.py``), the incoming gradient per output slot
and the outgoing gradient per labelled parent occurrence. A forward op whose
Paddle backward spans several nodes (``F.linear`` runs matmul and add) keeps one
``GradFn`` per node; the op's own node carries the op's outgoing gradients,
read at the inner node that sends each one. Timing, peak memory and engine
flags that Paddle does not report stay ``None``.

Paddle 3.3 cannot return a list-valued input slot from a post hook (``concat``,
``stack``, ``add_n``), so those nodes get no post hook: their incoming gradients
come from their output tensors' hooks and their outgoing gradient to a list
element from that element's tensor hook, which is exact when the element has no
other consumer. Otherwise the pass is refused, naming the binding.

Hook contract (the Paddle capture adds the calls; ``session`` is its capture
session, the trace under construction):

1. ``with backward_ready_scope(session) as scope:`` around the forward; it sets
   ``session.paddle_backward = scope``.
2. ``scope.note(label_raw, tensor)`` wherever the capture labels a tensor
   (``tensor_store.set_label``): model inputs and every emitted op output. It
   keeps the tensor's grad node and installs one tensor hook.
3. ``scope.output = output`` with the model's live return value, then, after the
   trace's ops are materialized, ``retain_backward_state(trace, session)``.
4. ``record_backward(trace, loss)``: ``loss`` is a scalar Paddle tensor computed
   from the live output, a callable mapping that output to one, or ``None`` when
   the output is the scalar loss.
"""

from __future__ import annotations

import time
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from ... import _state
from ..._ir.events import (
    BackwardPassEnd,
    BackwardPassStart,
    GradFnDiscovered,
    GradFnFired,
    OpGradObserved,
    ParamGradObserved,
)
from ..._runnable import RunnableErrorCode
from ...errors import RunCapabilityUnavailableError
from .._backward_neutral import (
    PayloadPolicy,
    journal_backward_event,
    next_backward_pass_index,
    set_pass_roots,
)
from .._replay_adapters import paddle as adapter

__all__ = [
    "PAYLOADS",
    "BackwardScope",
    "backward_ready_scope",
    "record_backward",
    "retain_backward_state",
]

STATE_KEY = "_paddle_backward_state"
"""``trace.__dict__`` key of the retained backward state."""

ACCUMULATION = "GradNodeAccumulation"
"""Paddle's name for the leaf node that accumulates a leaf tensor's gradient."""

NODE_BINDINGS = ("name", "next_functions", "node_ptr", "_register_post_hook")
"""Grad-node methods the record needs (feature detection, never version parsing)."""

LIST_SLOT_NODES = frozenset(
    {
        "ConcatGradNode",
        "StackGradNode",
        "AddNGradNode",
        "EinsumGradNode",
        "MeshgridGradNode",
        "BroadcastTensorsGradNode",
        "MultiDotGradNode",
        "UnstackGradNode",
    }
)
"""Nodes with a list-valued input slot, for nodes no recorded op describes."""

_BACKWARD_KWARGS = frozenset({"retain_graph"})


def _nbytes(value: Any) -> int | None:
    """Bytes of a Paddle tensor (it has no ``nbytes``)."""

    numel, size = getattr(value, "numel", None), getattr(value, "element_size", None)
    if not callable(numel) or not callable(size):
        return None
    return int(numel()) * int(size())


def _retained_refs(raw: Any, transformed: Any, *, raw_memory: int | None) -> list[tuple[Any, int]]:
    """Charge each retained gradient payload once, keyed by identity."""

    refs: list[tuple[Any, int]] = []
    for value, memory in ((raw, raw_memory), (transformed, None)):
        if value is not None:
            refs.append((("id", id(value)), int(memory or _nbytes(value) or 0)))
    return refs


PAYLOADS = PayloadPolicy(memory_of=_nbytes, retained_refs=_retained_refs)
"""How the neutral fold reads Paddle gradient payloads."""


class _Unavailable(Exception):
    """The pass cannot be recorded; the message is the refusal reason."""


def _ptr(node: Any) -> int:
    return int(node.node_ptr())


def _name(node: Any) -> str:
    return str(node.name())


def _next(node: Any) -> list[Any]:
    return list(node.next_functions or ())


def _copy(value: Any) -> Any:
    """A detached copy of one gradient (``None`` stays ``None``)."""

    if value is None or not hasattr(value, "detach"):
        return None
    with _state.pause_logging():
        return value.detach().clone()


def _copy_group(values: Any) -> tuple[Any, ...]:
    """Copy one hook argument: a slot is a tensor, ``None`` or a list of tensors."""

    return tuple(
        tuple(_copy(item) for item in slot) if isinstance(slot, (list, tuple)) else _copy(slot)
        for slot in values
    )


def _is_differentiable(tensor: Any) -> bool:
    import paddle

    if bool(getattr(tensor, "stop_gradient", True)):
        return False
    return bool(paddle.is_floating_point(tensor) or paddle.is_complex(tensor))


@dataclass
class _Recorder:
    """What one backward pass reported through the hooks."""

    tensor_grads: dict[str, Any] = field(default_factory=dict)
    tensor_times: dict[str, float] = field(default_factory=dict)
    param_grads: dict[str, Any] = field(default_factory=dict)
    param_times: dict[str, float] = field(default_factory=dict)
    fires: dict[int, tuple[float, tuple[Any, ...], tuple[Any, ...]]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass(eq=False)
class BackwardScope:
    """Grad nodes and tensor hooks of one Paddle capture.

    Attributes
    ----------
    output:
        The model's live return value, set by the capture.
    nodes:
        Raw label to the grad node of the tensor carrying it (``None`` for a
        tensor Paddle does not differentiate, or a leaf).
    leaves:
        Raw labels of differentiable leaf tensors (model inputs).
    """

    output: Any = None
    nodes: dict[str, Any] = field(default_factory=dict)
    leaves: set[str] = field(default_factory=set)
    handles: list[Any] = field(default_factory=list)
    hooked_nodes: set[int] = field(default_factory=set)
    recorder: _Recorder | None = None

    def note(self, label_raw: str, tensor: Any) -> None:
        """Keep ``tensor``'s grad node and hook its gradient, once per label."""

        if label_raw in self.nodes or not hasattr(tensor, "register_hook"):
            return
        node = getattr(tensor, "grad_fn", None)
        self.nodes[label_raw] = node
        if not _is_differentiable(tensor):
            return
        if node is None or _name(node) == ACCUMULATION:
            self.leaves.add(label_raw)
        with _state.pause_logging():
            self.handles.append(tensor.register_hook(self._tensor_hook(label_raw, "tensor")))

    def _tensor_hook(self, key: str, kind: str) -> Callable[[Any], None]:
        ref = weakref.ref(self)

        def hook(grad: Any) -> None:
            scope = ref()
            recorder = None if scope is None else scope.recorder
            if recorder is not None:
                grads = recorder.tensor_grads if kind == "tensor" else recorder.param_grads
                times = recorder.tensor_times if kind == "tensor" else recorder.param_times
                grads[key] = _copy(grad)
                times[key] = time.time()

        return hook

    def hook_node(self, node: Any) -> None:
        """Install the post hook on ``node`` once; it reports only while recording."""

        ptr = _ptr(node)
        if ptr in self.hooked_nodes:
            return
        ref = weakref.ref(self)

        def hook(grad_in: Any, grad_out: Any) -> list[Any]:
            scope = ref()
            recorder = None if scope is None else scope.recorder
            if recorder is not None:
                try:
                    recorder.fires.setdefault(
                        ptr, (time.time(), _copy_group(grad_in), _copy_group(grad_out))
                    )
                except Exception as error:  # noqa: BLE001 - a raise would break run_backward
                    recorder.errors.append(f"{_name(node)}: {error!r}")
            return list(grad_in)

        node._register_post_hook(hook)
        self.hooked_nodes.add(ptr)


@contextmanager
def backward_ready_scope(session: Any) -> Iterator[BackwardScope]:
    """Open the backward scope for one capture; the capture calls ``scope.note``."""

    scope = BackwardScope()
    session.paddle_backward = scope
    yield scope


@dataclass
class _Site:
    """One recorded call: its output slot labels and tensor input leaves."""

    capture: Any
    slots: tuple[str, ...]


@dataclass
class _Retained:
    """The capture's backward state, kept on the trace."""

    scope: BackwardScope
    sites: dict[int, _Site]
    labelled_uses: dict[str, int]
    params: dict[str, Any]
    final_labels: dict[str, str]


def _sites(trace: Any, scope: BackwardScope) -> tuple[dict[int, _Site], dict[str, int]]:
    """Map each recorded call's grad node to its site; count each label's uses."""

    sites: dict[int, _Site] = {}
    uses: dict[str, int] = {}
    for capture in getattr(trace, "_paddle_op_captures", ()) or ():
        for leaf in capture.tensor_inputs:
            if leaf.label is not None:
                uses[leaf.label] = uses.get(leaf.label, 0) + 1
        site = adapter._site(trace, capture)
        node = None if site is None else scope.nodes.get(site.slot_labels[0])
        if node is None or _name(node) == ACCUMULATION:
            continue
        sites.setdefault(_ptr(node), _Site(capture, site.slot_labels))
    for op in trace._layer_list:
        if getattr(op, "is_output", False):
            raw = str(getattr(op, "_label_raw", op.layer_label))
            uses[raw] = uses.get(raw, 0) + 1
    return sites, uses


def _params(trace: Any) -> dict[str, Any]:
    """Live parameter tensors by address, as the captures hold them."""

    params: dict[str, Any] = {}
    for capture in getattr(trace, "_paddle_op_captures", ()) or ():
        for leaf in capture.tensor_inputs:
            live = capture.param_inputs.get(tuple(leaf.path))
            if leaf.param_address is not None and live is not None:
                params.setdefault(leaf.param_address, live)
    return params


def retain_backward_state(trace: Any, session: Any) -> None:
    """Keep the capture's grad nodes and live output on ``trace`` for :func:`record_backward`."""

    scope = getattr(session, "paddle_backward", None)
    if scope is None:
        return
    sites, uses = _sites(trace, scope)
    finals = {
        str(getattr(op, "_label_raw", op.layer_label)): op.layer_label for op in trace._layer_list
    }
    trace.__dict__[STATE_KEY] = _Retained(scope, sites, uses, _params(trace), finals)


def _loss_tensor(state: _Retained, loss: Any) -> Any:
    """``loss`` as given, the live output for ``None``, or the callable applied to it."""

    import paddle

    if loss is None:
        return state.scope.output
    if callable(loss) and not isinstance(loss, paddle.Tensor):
        if state.scope.output is None:
            raise _Unavailable("the capture kept no live output to pass to the loss callable")
        return loss(state.scope.output)
    return loss


def _resolve_loss(state: _Retained, loss: Any) -> Any:
    """The scalar Paddle loss tensor connected to the captured graph."""

    import paddle

    loss = _loss_tensor(state, loss)
    scalar = isinstance(loss, paddle.Tensor) and int(loss.numel()) == 1
    if not scalar or not _is_differentiable(loss):
        raise _Unavailable(
            "the loss must be a differentiable float scalar Paddle tensor, or a callable "
            "mapping the model's output to one"
        )
    root = loss.grad_fn
    if root is None:
        raise _Unavailable(
            "the loss has no grad node, so it is not connected to the captured graph"
        )
    missing = [name for name in NODE_BINDINGS if not hasattr(root, name)]
    if missing:
        raise _Unavailable(
            f"this Paddle's grad nodes lack {', '.join(missing)} "
            "(paddle.base.libpaddle.GradNodeBase), so the backward graph cannot be walked"
        )
    return loss


def _walk(root: Any) -> dict[int, Any]:
    """Every grad node reachable from ``root``, in breadth-first discovery order."""

    found: dict[int, Any] = {}
    queue = [root]
    while queue:
        node = queue.pop(0)
        if _ptr(node) in found:
            continue
        found[_ptr(node)] = node
        queue.extend(item for item in _next(node) if item is not None)
    return found


def _has_list_slot(site: _Site | None, node: Any) -> bool:
    if site is None:
        return _name(node) in LIST_SLOT_NODES
    return any(len(tuple(leaf.path)) > 2 for leaf in site.capture.tensor_inputs)


@dataclass(frozen=True)
class _Edge:
    """One gradient slot leaving an op's backward: ``node``'s slot ``index`` to ``target``."""

    node: Any
    index: int
    target: Any


def _frontier(site_node: Any, owned: set[int]) -> tuple[list[_Edge], list[Any]]:
    """The op's outgoing slots in call order, and its inner nodes (expanded in place)."""

    edges: list[_Edge] = []
    inner: list[Any] = []

    def visit(node: Any) -> None:
        for index, target in enumerate(_next(node)):
            inner_node = (
                target is not None and _ptr(target) not in owned and _name(target) != ACCUMULATION
            )
            if inner_node and all(_ptr(target) != _ptr(seen) for seen in inner):
                inner.append(target)
                visit(target)
            else:
                edges.append(_Edge(node, index, target))

    visit(site_node)
    return edges, inner


def _expected(state: _Retained, site: _Site, leaf: Any) -> Any:
    """What grad node a tensor input should reach: a node, ``ACCUMULATION`` or ``None``."""

    if leaf.label is not None:
        if leaf.label in state.scope.leaves:
            return ACCUMULATION
        return state.scope.nodes.get(leaf.label)
    live = site.capture.param_inputs.get(tuple(leaf.path))
    if leaf.param_address is not None and live is not None and _is_differentiable(live):
        return ACCUMULATION
    return None


def _matches(expected: Any, target: Any) -> bool:
    if expected is None or target is None:
        return expected is None and target is None
    if expected == ACCUMULATION:
        return _name(target) == ACCUMULATION
    return _ptr(expected) == _ptr(target)


def _align(state: _Retained, site: _Site, edges: list[_Edge]) -> list[_Edge | None]:
    """One edge (or ``None``) per tensor input, in call order; refuses when they disagree."""

    aligned: list[_Edge | None] = []
    cursor = 0
    for leaf in site.capture.tensor_inputs:
        expected = _expected(state, site, leaf)
        edge = edges[cursor] if cursor < len(edges) else None
        if edge is not None and _matches(expected, edge.target):
            aligned.append(edge)
            cursor += 1
        elif expected is None:
            aligned.append(None)
        else:
            raise _Unavailable(
                f"the grad slots of {site.slots[0]!r} do not line up with its recorded inputs "
                f"(input {tuple(leaf.path)} has no matching slot)"
            )
    if cursor != len(edges):
        raise _Unavailable(f"{site.slots[0]!r} sends gradients to slots its record does not name")
    return aligned


@dataclass
class _Plan:
    """The walked graph and how each node maps to the trace."""

    nodes: dict[int, Any]
    op_nodes: dict[int, _Site] = field(default_factory=dict)
    aligned: dict[int, list[_Edge | None]] = field(default_factory=dict)
    leaf_of: dict[int, tuple[str, str]] = field(default_factory=dict)


def _note_leaves(plan: _Plan, site: _Site, aligned: list[_Edge | None]) -> None:
    """Name each accumulation node the op reaches by the input or parameter it belongs to."""

    for leaf, edge in zip(site.capture.tensor_inputs, aligned, strict=True):
        target = None if edge is None else edge.target
        if target is None or _name(target) != ACCUMULATION:
            continue
        key = ("param", str(leaf.param_address)) if leaf.label is None else ("input", leaf.label)
        plan.leaf_of.setdefault(_ptr(target), key)


def _plan(state: _Retained, root: Any) -> _Plan:
    """Walk from the loss, align every recorded op's slots and install the post hooks."""

    plan = _Plan(_walk(root))
    order = [ptr for ptr in plan.nodes if ptr in state.sites]
    owned = set(order)
    for ptr in order:
        site = state.sites[ptr]
        edges, _inner = _frontier(plan.nodes[ptr], owned)
        aligned = _align(state, site, edges)
        plan.op_nodes[ptr], plan.aligned[ptr] = site, aligned
        _note_leaves(plan, site, aligned)
    for ptr, node in plan.nodes.items():
        if _name(node) != ACCUMULATION and not _has_list_slot(plan.op_nodes.get(ptr), node):
            state.scope.hook_node(node)
    return plan


def _slot_value(recorder: _Recorder, edge: _Edge) -> Any:
    fire = recorder.fires.get(_ptr(edge.node))
    if fire is None:
        return None
    grad_in = fire[1]
    return grad_in[edge.index] if edge.index < len(grad_in) else None


def _outgoing(state: _Retained, plan: _Plan, recorder: _Recorder, ptr: int) -> tuple[Any, ...]:
    """The op's outgoing gradient per labelled parent occurrence."""

    site = plan.op_nodes[ptr]
    list_slot = _has_list_slot(site, plan.nodes[ptr])
    values: list[Any] = []
    for leaf, edge in zip(site.capture.tensor_inputs, plan.aligned[ptr], strict=True):
        if leaf.label is None:
            continue
        if edge is None or edge.target is None:
            values.append(None)
        elif list_slot and _ptr(edge.node) == ptr:
            if state.labelled_uses.get(leaf.label, 0) != 1:
                raise _Unavailable(
                    f"Paddle's GradNodeBase._register_post_hook cannot return the list input of "
                    f"{site.slots[0]!r}, and {leaf.label!r} feeds other ops too, so the gradient "
                    "this op sends to it is not observable"
                )
            values.append(recorder.tensor_grads.get(leaf.label))
        else:
            values.append(_slot_value(recorder, edge))
    return tuple(values)


def _discoveries(state: _Retained, plan: _Plan) -> list[GradFnDiscovered]:
    events = []
    for ptr, node in plan.nodes.items():
        site = plan.op_nodes.get(ptr)
        leaf = plan.leaf_of.get(ptr)
        name = _name(node)
        events.append(
            GradFnDiscovered(
                object_id=ptr,
                class_name=name,
                class_qualname=name,
                is_custom=name.startswith("GradNodePyLayer"),
                op_label=None if site is None else state.final_labels.get(site.slots[0]),
                param_ref=leaf[1] if leaf is not None and leaf[0] == "param" else None,
                created_in_pass=None,
                creator_object_id=None,
                source={},
                topology=tuple(_ptr(item) for item in _next(node) if item is not None),
            )
        )
    return events


def _fire(ptr: int, pass_index: int, stamp: float, outgoing: Any, incoming: Any) -> GradFnFired:
    return GradFnFired(
        object_id=ptr,
        pass_index=pass_index,
        grad_input_refs=tuple(outgoing),
        grad_output_refs=tuple(incoming),
        fire_record=None,
        timestamp=stamp,
    )


def _op_fire(
    state: _Retained, plan: _Plan, recorder: _Recorder, ptr: int, pass_index: int
) -> GradFnFired | None:
    site = plan.op_nodes[ptr]
    incoming = tuple(recorder.tensor_grads.get(label) for label in site.slots)
    fire = recorder.fires.get(ptr)
    if fire is None:
        times = [
            recorder.tensor_times[label] for label in site.slots if label in recorder.tensor_times
        ]
        if not times:
            return None
        stamp = min(times)
    else:
        stamp = fire[0]
    return _fire(ptr, pass_index, stamp, _outgoing(state, plan, recorder, ptr), incoming)


def _leaf_fire(plan: _Plan, recorder: _Recorder, ptr: int, pass_index: int) -> GradFnFired | None:
    kind, key = plan.leaf_of[ptr]
    grads = recorder.tensor_grads if kind == "input" else recorder.param_grads
    times = recorder.tensor_times if kind == "input" else recorder.param_times
    if key not in grads:
        return None
    return _fire(ptr, pass_index, times[key], (), (grads[key],))


def _fires(
    state: _Retained, plan: _Plan, recorder: _Recorder, pass_index: int
) -> list[GradFnFired]:
    """One fire per node the pass observed executing."""

    fires: list[GradFnFired | None] = []
    for ptr in plan.nodes:
        if ptr in plan.op_nodes:
            fires.append(_op_fire(state, plan, recorder, ptr, pass_index))
        elif ptr in plan.leaf_of:
            fires.append(_leaf_fire(plan, recorder, ptr, pass_index))
        elif ptr in recorder.fires:
            stamp, grad_in, grad_out = recorder.fires[ptr]
            fires.append(_fire(ptr, pass_index, stamp, grad_in, grad_out))
    return [fire for fire in fires if fire is not None]


def _grad_events(state: _Retained, recorder: _Recorder, pass_index: int, stamp: float) -> list[Any]:
    def meta(value: Any) -> dict[str, Any]:
        return {
            "shape": tuple(int(dim) for dim in value.shape),
            "dtype": str(value.dtype),
            "memory": _nbytes(value),
        }

    ops = [
        OpGradObserved(
            op_label=label,
            pass_index=pass_index,
            payload_ref=value,
            transformed_payload_ref=None,
            timestamp=recorder.tensor_times.get(label, stamp),
            **meta(value),
        )
        for label, value in recorder.tensor_grads.items()
        if value is not None and label in state.final_labels
    ]
    params = [
        ParamGradObserved(
            param_address=address,
            pass_index=pass_index,
            payload_ref=value,
            timestamp=recorder.param_times.get(address, stamp),
            **meta(value),
        )
        for address, value in recorder.param_grads.items()
        if value is not None
    ]
    return [*ops, *params]


def _start(pass_index: int, stamp: float) -> BackwardPassStart:
    return BackwardPassStart(
        pass_index=pass_index,
        trigger="backward",
        implicit=False,
        outer_context=None,
        call_context_ref=None,
        root_meta=(),
        root_grad_arguments=None,
        inputs_subset=(),
        order=None,
        origin_backward_pass=None,
        save_grads_policy_repr=None,
        engine_flags=None,
        forward_op_count_at_trigger=None,
        timestamp=stamp,
    )


def _run(state: _Retained, plan: _Plan, loss: Any, retain_graph: bool) -> tuple[_Recorder, float]:
    """Run Paddle's backward with the hooks reporting; returns what they saw and its duration."""

    recorder = _Recorder()
    handles = []
    with _state.pause_logging():
        for address, param in state.params.items():
            if _is_differentiable(param):
                handles.append(param.register_hook(state.scope._tensor_hook(address, "param")))
    state.scope.recorder = recorder
    started = time.monotonic()
    try:
        loss.backward(retain_graph=retain_graph)
    finally:
        state.scope.recorder = None
        for handle in handles:
            handle.remove()
    duration = time.monotonic() - started
    if recorder.errors:
        raise _Unavailable(f"a grad-node hook failed: {recorder.errors[0]}")
    return recorder, duration


def _record(trace: Any, loss: Any, backward_kwargs: dict[str, Any]) -> tuple[list[Any], int]:
    """Run the pass and build its events; raises :class:`_Unavailable` with the reason."""

    state = trace.__dict__.get(STATE_KEY)
    if state is None:
        raise _Unavailable(
            "the capture kept no grad nodes; capture with "
            "capture=tl.options.CaptureOptions(backward_ready=True) first"
        )
    extra = sorted(set(backward_kwargs) - _BACKWARD_KWARGS)
    if extra:
        raise _Unavailable(f"Paddle backward takes only retain_graph (got {extra})")
    tensor = _resolve_loss(state, loss)
    plan = _plan(state, tensor.grad_fn)
    stamp = time.time()
    recorder, duration = _run(state, plan, tensor, bool(backward_kwargs.get("retain_graph")))
    pass_index = next_backward_pass_index(trace)
    end = BackwardPassEnd(
        pass_index=pass_index,
        duration=duration,
        peak_memory=None,
        status="ok",
        order_attribution_coverage=None,
    )
    events = [
        _start(pass_index, stamp),
        *_discoveries(state, plan),
        *_fires(state, plan, recorder, pass_index),
        *_grad_events(state, recorder, pass_index, stamp),
        end,
    ]
    return events, _ptr(tensor.grad_fn)


def record_backward(trace: Any, loss: Any = None, **backward_kwargs: Any) -> None:
    """Run one Paddle backward pass from ``loss`` and journal what it executed.

    Parameters
    ----------
    trace:
        A Paddle trace captured inside :func:`backward_ready_scope`.
    loss:
        A scalar Paddle tensor computed from the live output, a callable mapping
        the model's output to one, or ``None`` when the output is the scalar loss.
    **backward_kwargs:
        ``retain_graph`` only, with ``Tensor.backward``'s meaning.

    Raises
    ------
    RunCapabilityUnavailableError
        When the capture was not backward-ready, the loss is not connected, a
        needed Paddle binding is missing, or a gradient the record needs is not
        observable.
    """

    try:
        events, root = _record(trace, loss, backward_kwargs)
    except _Unavailable as error:
        raise RunCapabilityUnavailableError(
            f"backward() is unavailable on this Paddle trace: {error}.",
            code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
            detection_stage="native_backward_capture",
        ) from None
    for event in events:
        journal_backward_event(trace, event)
    set_pass_roots(trace, events[0].pass_index, [root])
