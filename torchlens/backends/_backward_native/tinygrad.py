"""Native backward capture on tinygrad: gradient UOps joined to the capture's single realize.

tinygrad differentiates its lazy UOp graph itself: ``compute_gradient`` walks
the graph from the loss and applies one rule per UOp (``pm_gradient``). The
capture records the forward as the UOps the forward built, realized once; the
backward follows the same rule. Before that one realize, the gradient of the
loss is built on the forward's own pre-realize UOps, and every gradient value
(each recorded op's incoming gradient and each node's gradient to each parent
occurrence) is listed in the same ``realize_once`` call, wrapped in
``CONTIGUOUS`` as the forward values are, so no forward value is computed
twice and random state is drawn once. Realizing the gradient after the
capture would recompute the forward from regenerated UOps instead.

The records keep PyTorch's meanings (``backends/_backward_neutral.py``): one
``GradFn`` per recorded op that sent a gradient to a recorded parent, named by
tinygrad's own rule (the UOp op name, for example ``MUL``); one ``GradFnCall``
per node, since the realize ran each rule once; ``Op.grad`` for every recorded
op and input that received a gradient; parameter gradients by address. tinygrad
has no gradient-accumulation node, so leaves get gradients and no node. Timing,
peak memory and engine flags are not observed and stay ``None``.

Hook contract (the tinygrad capture adds the calls; ``session`` is its own
capture session object):

1. ``with backward_ready_scope(session) as scope:`` around the forward. The
   scope reads ``session.loss_fn`` (the ``GradOptions.loss_fn`` callable, or
   ``None`` when the output itself is the scalar loss) and sets
   ``session.tinygrad_backward = scope``.
2. After the forward and before the single realize, inside the capture's
   execution guard (``_reject_mid_capture_execution``, so building the loss
   cannot run a kernel): ``extra = scope.gradient_targets(plan, source_uops,
   raw_output)``; the capture passes ``extra`` to the same ``realize_once``
   call as its recorded values and outputs. A loss that cannot be built is kept as the reason
   ``record_backward`` refuses with; the capture itself goes on.
3. After the trace's ops are emitted: ``retain_backward_state(trace, session)``
   with ``session.uop_labels`` (``id()`` of a pre-realize UOp to its raw label,
   the capture's own map) and, when an object module was traced,
   ``session.param_addresses`` (``id()`` of a parameter UOp to its address).
4. ``record_backward(trace, loss)`` journals that one executed pass.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from ..._ir.capture_events import CaptureEvents
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

__all__ = [
    "PAYLOADS",
    "BackwardScope",
    "backward_ready_scope",
    "record_backward",
    "retain_backward_state",
]

STATE_KEY = "_tinygrad_backward_state"
"""``trace.__dict__`` key of the retained backward state."""

RULE_QUALNAME_PREFIX = "tinygrad.gradient.pm_gradient."
"""Qualified-name prefix of tinygrad's per-UOp gradient rules."""

_CALL_UOPS = frozenset({"FUNCTION", "CALL", "GETTUPLE"})


def _nbytes(value: Any) -> int | None:
    """Bytes of a tinygrad tensor (``Tensor.nbytes`` is a method there)."""

    nbytes = getattr(value, "nbytes", None)
    if callable(nbytes):
        nbytes = nbytes()
    return None if nbytes is None else int(nbytes)


def _retained_refs(raw: Any, transformed: Any, *, raw_memory: int | None) -> list[tuple[Any, int]]:
    """Charge each retained gradient payload once, keyed by identity."""

    refs: list[tuple[Any, int]] = []
    for value, memory in ((raw, raw_memory), (transformed, None)):
        if value is not None:
            refs.append((("id", id(value)), int(memory or _nbytes(value) or 0)))
    return refs


PAYLOADS = PayloadPolicy(memory_of=_nbytes, retained_refs=_retained_refs)
"""How the neutral fold reads tinygrad gradient payloads."""


class _Unavailable(Exception):
    """The backward pass cannot be prepared; the message is the refusal reason."""


def _uop_name(uop: Any) -> str:
    op = getattr(uop, "op", None)
    return str(getattr(op, "name", op)).removeprefix("Ops.")


def _is_float(dtype: Any) -> bool:
    from tinygrad import dtypes

    return bool(dtypes.is_float(dtype))


def _gradient_api() -> tuple[Callable[..., Any], Any]:
    """tinygrad's ``compute_gradient`` and ``pm_gradient``, found by feature detection."""

    from tinygrad import gradient as tg_gradient

    compute = getattr(tg_gradient, "compute_gradient", None)
    rules = getattr(tg_gradient, "pm_gradient", None)
    if compute is None or rules is None:
        raise _Unavailable(
            "this tinygrad has no compute_gradient or pm_gradient in tinygrad.gradient, "
            "so its backward graph cannot be built"
        )
    return compute, rules


def _flat_tensors(value: Any) -> list[Any]:
    """tinygrad tensors in a nested output, in container order."""

    from tinygrad import Tensor

    if isinstance(value, Tensor):
        return [value]
    if isinstance(value, Mapping):
        return [leaf for child in value.values() for leaf in _flat_tensors(child)]
    if isinstance(value, (list, tuple)):
        return [leaf for child in value for leaf in _flat_tensors(child)]
    return []


def _walked(root: Any, targets: set[Any]) -> set[int]:
    """``id()`` of the UOps whose rule ``compute_gradient`` applies.

    tinygrad differentiates a UOp only when one of its sources leads to a
    target, and never through ``DETACH``: the gradient a ``DETACH`` receives
    stops there. A recorded op outside this set gets its gradient recorded
    and no backward node, as a PyTorch tensor without a ``grad_fn``.
    """

    on_path: set[int] = set()
    walked: set[int] = set()
    for uop in root.toposort():
        if any(id(src) in on_path or src in targets for src in uop.src):
            on_path.add(id(uop))
            if _uop_name(uop) != "DETACH":
                walked.add(id(uop))
    return walked


@dataclass(frozen=True)
class _PreparedNode:
    """One gradient rule before the realize: its forward UOp and outgoing gradients."""

    uop: Any
    outgoing: tuple[tuple[int, Any], ...]
    """``(source position, gradient UOp or None)`` per labelled parent occurrence."""


@dataclass
class BackwardScope:
    """Gradient UOps of one capture, built on the forward's pre-realize UOps.

    Attributes
    ----------
    loss_fn:
        Callable mapping the model's output to the scalar loss, or ``None``.
    unavailable:
        Why the pass could not be prepared, or ``None``.
    """

    loss_fn: Callable[[Any], Any] | None
    unavailable: str | None = None
    op_grads: list[tuple[Any, Any]] = field(default_factory=list)
    nodes: list[_PreparedNode] = field(default_factory=list)
    root_uops: list[Any] = field(default_factory=list)
    targets: dict[int, tuple[Any, Any, bool]] = field(default_factory=dict)

    def gradient_targets(
        self, plan: Any, source_uops: Sequence[Any], raw_output: Any
    ) -> tuple[Any, ...]:
        """Build the gradient UOps; return the tensors to add to the single realize.

        Parameters
        ----------
        plan:
            The capture plan (``tinygrad._realize_once.plan_capture``).
        source_uops:
            UOps of the traced input leaves, read before the realize.
        raw_output:
            The model's return value (lazy tinygrad tensors).

        Returns
        -------
        tuple[Any, ...]
            ``CONTIGUOUS`` gradient tensors; empty when the pass is unavailable.
        """

        try:
            self._prepare(tuple(plan.recorded), tuple(source_uops), raw_output)
        except Exception as error:  # noqa: BLE001 - kept as the typed refusal reason
            self.op_grads, self.nodes, self.root_uops, self.targets = [], [], [], {}
            detail = str(error) if isinstance(error, _Unavailable) else repr(error)
            self.unavailable = f"tinygrad could not build the backward graph: {detail}"
            return ()
        return tuple(tensor for _uop, tensor, _owned in self.targets.values())

    def _loss(self, raw_output: Any) -> Any:
        """The scalar loss on the lazy outputs (the capture's execution guard is active)."""

        from tinygrad import Tensor

        loss = raw_output if self.loss_fn is None else self.loss_fn(raw_output)
        if not isinstance(loss, Tensor) or tuple(loss.shape) != () or not _is_float(loss.dtype):
            raise _Unavailable(
                "the loss must be a float scalar tinygrad Tensor; pass "
                "grad_options=GradOptions(loss_fn=...) mapping the output to one"
            )
        return loss

    def _target(self, uop: Any) -> Any:
        """List ``uop`` once in the realize as ``CONTIGUOUS(uop)``; return ``uop``."""

        from tinygrad import Tensor

        if id(uop) not in self.targets:
            contiguous = uop.contiguous()
            self.targets[id(uop)] = (uop, Tensor(contiguous), contiguous is not uop)
        return uop

    def _prepare(
        self, recorded: tuple[Any, ...], sources: tuple[Any, ...], raw_output: Any
    ) -> None:
        from tinygrad import Tensor

        compute_gradient, rules = _gradient_api()
        loss = self._loss(raw_output)
        labelled = {id(uop) for uop in (*sources, *recorded)}
        ordered = list({id(uop): uop for uop in (*sources, *recorded)}.values())
        wanted = {uop for uop in ordered if _is_float(uop.dtype)}
        seed = Tensor(1.0, dtype=loss.dtype, device=loss.device)
        grads = compute_gradient(loss.uop, seed.uop, wanted)
        walked = _walked(loss.uop, wanted)
        for uop in ordered:
            grad = grads.get(uop) if uop in wanted else None
            if grad is None or _uop_name(grad) == "NOOP":
                continue
            self.op_grads.append((uop, self._target(grad)))
            if id(uop) in walked:
                self._prepare_node(uop, grad, rules, labelled)
        output_ids = {id(tensor.uop) for tensor in _flat_tensors(raw_output)}
        self.root_uops = [node.uop for node in self.nodes if id(node.uop) in output_ids]

    def _prepare_node(self, uop: Any, grad: Any, rules: Any, labelled: set[int]) -> None:
        """The rule's gradient to each labelled parent occurrence of ``uop``."""

        positions = [pos for pos, src in enumerate(uop.src) if id(src) in labelled]
        if not positions:
            return
        if _uop_name(uop) in _CALL_UOPS:
            raise _Unavailable(f"a recorded {_uop_name(uop)} UOp has no per-op gradient rule")
        local = rules.rewrite(uop, ctx=grad)
        if local is None or len(local) != len(uop.src):
            raise _Unavailable(f"tinygrad has no gradient rule for UOp {_uop_name(uop)}")
        outgoing = []
        for pos in positions:
            value = local[pos]
            usable = value is not None and _uop_name(value) != "NOOP"
            usable = usable and _is_float(uop.src[pos].dtype)
            outgoing.append((pos, self._target(value) if usable else None))
        self.nodes.append(_PreparedNode(uop=uop, outgoing=tuple(outgoing)))


@contextmanager
def backward_ready_scope(session: Any) -> Iterator[BackwardScope]:
    """Prepare a backward pass for the capture around which this scope runs.

    Parameters
    ----------
    session:
        The capture's session; ``session.loss_fn`` is read and
        ``session.tinygrad_backward`` is set to the yielded scope.

    Yields
    ------
    BackwardScope
        The capture calls :meth:`BackwardScope.gradient_targets` before its realize.
    """

    scope = BackwardScope(loss_fn=getattr(session, "loss_fn", None))
    session.tinygrad_backward = scope
    yield scope


@dataclass(frozen=True)
class _Node:
    object_id: int
    class_name: str
    op_label: str
    incoming: Any
    outgoing: tuple[Any, ...]
    topology: tuple[int, ...]


@dataclass
class _Retained:
    """One realized backward pass, waiting for ``record_backward``."""

    loss_fn: Callable[[Any], Any] | None
    unavailable: str | None
    timestamp: float
    nodes: tuple[_Node, ...] = ()
    op_grads: tuple[tuple[str, Any], ...] = ()
    param_grads: tuple[tuple[str, Any], ...] = ()
    roots: tuple[int, ...] = ()
    pinned: tuple[Any, ...] = ()
    recorded_pass: int | None = None


def _host_copy(tensor: Any) -> Any:
    """A realized copy of ``tensor`` in its own buffer (the backend's host-copy rule)."""

    import numpy as np
    from tinygrad import Tensor

    values = tensor.tolist()
    if str(tensor.device).startswith("NPY"):
        return Tensor(np.asarray(values), dtype=tensor.dtype, device=tensor.device)
    return Tensor(values, dtype=tensor.dtype, device=tensor.device).realize()


def _payloads(scope: BackwardScope) -> dict[int, Any]:
    """Realized gradient values by gradient-UOp id; a shared buffer is copied."""

    return {
        key: tensor if owned else _host_copy(tensor)
        for key, (_uop, tensor, owned) in scope.targets.items()
    }


def _final_labels(trace: Any, uop_labels: Mapping[int, str]) -> dict[int, str]:
    """``id()`` of a pre-realize UOp to the final label of the op recording it."""

    finals = {getattr(op, "_label_raw", None): op.layer_label for op in trace._layer_list}
    return {key: finals[raw] for key, raw in uop_labels.items() if raw in finals}


def _retain(
    trace: Any, scope: BackwardScope, uop_labels: Mapping[int, str], params: Mapping[int, str]
) -> _Retained:
    """Map the realized gradient values to the trace's ops and parameters."""

    retained = _Retained(scope.loss_fn, scope.unavailable, time.time())
    if scope.unavailable is not None:
        return retained
    labels = _final_labels(trace, uop_labels)
    values = _payloads(scope)
    missing = [uop for uop, _grad in scope.op_grads if id(uop) not in labels]
    if missing:
        retained.unavailable = (
            f"{len(missing)} tinygrad gradient values belong to UOps the trace does not "
            f"record (first: {_uop_name(missing[0])}), so the backward records would be partial"
        )
        return retained
    retained.op_grads = tuple((labels[id(uop)], values[id(grad)]) for uop, grad in scope.op_grads)
    retained.param_grads = tuple(
        (params[id(uop)], values[id(grad)]) for uop, grad in scope.op_grads if id(uop) in params
    )
    incoming = {id(uop): values[id(grad)] for uop, grad in scope.op_grads}
    node_ids = {id(node.uop) for node in scope.nodes}
    retained.nodes = tuple(
        _Node(
            object_id=id(node.uop),
            class_name=_uop_name(node.uop),
            op_label=labels[id(node.uop)],
            incoming=incoming[id(node.uop)],
            outgoing=tuple(
                None if grad is None else values[id(grad)] for _p, grad in node.outgoing
            ),
            topology=tuple(
                id(node.uop.src[pos])
                for pos, _g in node.outgoing
                if id(node.uop.src[pos]) in node_ids
            ),
        )
        for node in scope.nodes
    )
    retained.roots = tuple(id(uop) for uop in scope.root_uops)
    retained.pinned = tuple(uop for uop, _grad in scope.op_grads)
    return retained


def retain_backward_state(trace: Any, session: Any) -> None:
    """Keep the capture's realized backward pass on ``trace`` for :func:`record_backward`.

    Parameters
    ----------
    trace:
        The trace the capture just emitted.
    session:
        The capture's session (``tinygrad_backward``, ``uop_labels``,
        optional ``param_addresses``). Without a backward scope nothing is kept.
    """

    scope = getattr(session, "tinygrad_backward", None)
    if scope is None:
        return
    uop_labels = getattr(session, "uop_labels", None) or {}
    params = getattr(session, "param_addresses", None) or {}
    trace.__dict__[STATE_KEY] = _retain(trace, scope, uop_labels, params)
    if getattr(trace, "_capture_events", None) is None:
        # The capture releases its forward stream after materializing; the
        # backward pass journals into a fresh stream extending the projection.
        trace._capture_events = CaptureEvents.detached_from(trace)


def _refusal_reason(
    state: _Retained | None, loss: Any, backward_kwargs: Mapping[str, Any]
) -> str | None:
    """Why ``record_backward`` cannot journal a pass, or ``None``."""

    if state is None:
        return (
            "tinygrad computes the backward pass inside the capture's single realize; capture "
            "with capture=tl.options.CaptureOptions(backward_ready=True) and "
            "grad_options=GradOptions(loss_fn=...) first"
        )
    if state.unavailable is not None:
        return state.unavailable
    if backward_kwargs:
        return f"tinygrad backward takes no engine options (got {sorted(backward_kwargs)})"
    if state.recorded_pass is not None:
        return (
            f"the capture's single realize ran one backward pass, already recorded as pass "
            f"{state.recorded_pass}; capture again for another pass"
        )
    if loss is None or loss is state.loss_fn:
        return None
    if callable(loss):
        return (
            "tinygrad differentiates inside the capture's single realize, so the loss must be "
            "the loss_fn the capture prepared; capture again with "
            "grad_options=GradOptions(loss_fn=<this callable>) to differentiate it"
        )
    return (
        "a tinygrad value computed after the capture is not connected to the captured graph; "
        "pass the loss_fn callable the capture prepared"
    )


def _refuse(problem: str) -> RunCapabilityUnavailableError:
    return RunCapabilityUnavailableError(
        f"backward() is unavailable on this tinygrad trace: {problem}.",
        code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
        detection_stage="native_backward_capture",
    )


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


def _node_events(state: _Retained, pass_index: int) -> list[Any]:
    """Discoveries in forward order, then fires in the order a reverse pass runs them."""

    discovered = [
        GradFnDiscovered(
            object_id=node.object_id,
            class_name=node.class_name,
            class_qualname=RULE_QUALNAME_PREFIX + node.class_name,
            is_custom=False,
            op_label=node.op_label,
            param_ref=None,
            created_in_pass=None,
            creator_object_id=None,
            source={},
            topology=node.topology,
        )
        for node in state.nodes
    ]
    fired = [
        GradFnFired(
            object_id=node.object_id,
            pass_index=pass_index,
            grad_input_refs=node.outgoing,
            grad_output_refs=(node.incoming,),
            fire_record=None,
            timestamp=state.timestamp,
        )
        for node in reversed(state.nodes)
    ]
    return [*discovered, *fired]


def _grad_events(state: _Retained, pass_index: int) -> list[Any]:
    ops = [
        OpGradObserved(
            op_label=label,
            pass_index=pass_index,
            payload_ref=value,
            transformed_payload_ref=None,
            shape=tuple(int(dim) for dim in value.shape),
            dtype=str(value.dtype),
            memory=_nbytes(value),
            timestamp=state.timestamp,
        )
        for label, value in state.op_grads
    ]
    params = [
        ParamGradObserved(
            param_address=address,
            pass_index=pass_index,
            payload_ref=value,
            shape=tuple(int(dim) for dim in value.shape),
            dtype=str(value.dtype),
            memory=_nbytes(value),
            timestamp=state.timestamp,
        )
        for address, value in state.param_grads
    ]
    return [*ops, *params]


def record_backward(trace: Any, loss: Any = None, **backward_kwargs: Any) -> None:
    """Journal the backward pass the capture's single realize computed.

    Parameters
    ----------
    trace:
        A tinygrad trace captured inside :func:`backward_ready_scope`.
    loss:
        The capture's ``loss_fn`` (or ``None`` for it).
    **backward_kwargs:
        Refused: tinygrad has no backward engine options.

    Raises
    ------
    RunCapabilityUnavailableError
        When the capture prepared no pass, the pass could not be built, the
        loss is not the prepared one, or the pass was already recorded.
    """

    state = trace.__dict__.get(STATE_KEY)
    reason = _refusal_reason(state, loss, backward_kwargs)
    if reason is not None:
        raise _refuse(reason)
    pass_index = next_backward_pass_index(trace)
    events = [
        _start(pass_index, state.timestamp),
        *_node_events(state, pass_index),
        *_grad_events(state, pass_index),
        BackwardPassEnd(
            pass_index=pass_index,
            duration=None,
            peak_memory=None,
            status="ok",
            order_attribution_coverage=None,
        ),
    ]
    for event in events:
        journal_backward_event(trace, event)
    set_pass_roots(trace, pass_index, state.roots)
    state.recorded_pass = pass_index
