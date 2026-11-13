"""Native backward capture on JAX.

``record_backward(trace, loss)`` runs one backward pass of JAX's own pullback
and journals what it computed as the neutral backward events
(``backends/_backward_neutral.py``), with PyTorch's meanings:

- one ``GradFn`` per recorded equation whose rule JAX applied: every reached
  equation op with a differentiable parent. ``class_name`` is JAX's own name
  for the rule, ``transpose(jvp(<primitive>))``, the form JAX writes in the
  backward equations' name stacks. A rule that emits no backward equation
  (``add`` passes its cotangent through; ``floor`` has a zero derivative) is
  still applied by JAX's transposition and is recorded with the values it
  passed on;
- one ``GradFnCall`` per pass for each of them, with the incoming gradient per
  output slot and the outgoing gradient per parent occurrence, read at the
  node's boundaries from the backward program that ran;
- ``Op.grad`` for every reached op, including sources and output ops.

The program comes from ``_jax_program`` (the kept captures bound again inside
``tlb_`` scopes). Every backward equation must carry the scope of a recorded
node; anything else refuses the pass rather than recording a guess. Per-call
timing and engine fields stay ``None``: JAX runs the backward program as one
evaluation, so only the pass duration is observed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
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
from ...errors import RunCapabilityUnavailableError
from .._backward_neutral import journal_backward_event, next_backward_pass_index, set_pass_roots
from .._replay_adapters import jax as jax_adapter
from .._reverse_engine import EngineNode, ReplayGraph
from ._jax_program import (
    BackwardRun,
    Program,
    ProgramNode,
    build_program,
    run_backward,
    scope_groups,
    seed_dependent,
)

__all__ = (
    "backward_ready_scope",
    "record_backward",
    "retain_backward_state",
    "ValidationInputs",
    "validation_inputs",
)

_EMPTY_SOURCE: Mapping[str, object | None] = MappingProxyType({})
_CODE = "run_capability_unavailable"  # RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE
_DETECTION_STAGE = "native_backward_capture"
_STREAM_NAMES = ("event_stream", "_capture_events", "capture_events")


@contextmanager
def backward_ready_scope(session: Any) -> Iterator[None]:
    """Context around a backward-ready JAX forward.

    Nothing changes during the forward: the backward program is rebuilt from
    the kept captures with its own ``tlb_`` scopes, so the forward's name
    stacks (which carry module attribution) stay exactly as in a plain capture.
    """

    del session
    yield


def retain_backward_state(trace: Any, session: Any) -> None:
    """Keep what a later ``record_backward`` needs: the captures and an event stream.

    The capture calls this at the end of a backward-ready forward, before it
    releases its event stream. The equation captures it pins are the ones the
    validator already holds; the stream is where backward events are journaled.
    """

    del session
    captures = tuple(getattr(trace, "jax_equation_captures", None) or ())
    trace.__dict__["_jax_backward_captures"] = captures
    stream = getattr(trace, "capture_events", None)
    if stream is not None:
        trace._capture_events = stream


def _stream(trace: Any) -> Any:
    for name in _STREAM_NAMES:
        stream = getattr(trace, name, None)
        if stream is not None:
            return stream
    return None


def _ensure_stream(trace: Any) -> None:
    """Give a trace whose capture released its stream a fresh backward journal.

    The forward records are already built, and backward records are built
    from backward events only, so an empty stream loses nothing. Only called
    on a trace with no backward pass yet (see ``_call_problem``).
    """

    if _stream(trace) is None:
        trace._capture_events = CaptureEvents()


def _captures_present(trace: Any) -> bool:
    kept = trace.__dict__.get("_jax_backward_captures")
    if kept and not getattr(trace, "jax_equation_captures", None):
        trace.jax_equation_captures = kept
    return bool(getattr(trace, "jax_equation_captures", None))


def _call_problem(trace: Any, loss: Any, backward_kwargs: Mapping[str, Any]) -> str | None:
    """Why this ``backward`` call cannot run, or ``None``."""

    if backward_kwargs:
        names = ", ".join(sorted(backward_kwargs))
        return f"it takes no keyword arguments (got {names}): JAX's pullback has no engine options"
    if not callable(loss):
        return (
            "loss must be a callable mapping trace.output to a scalar: a JAX array computed "
            "after capture has no link to the captured program"
        )
    if not _captures_present(trace):
        return (
            "its equation captures were not kept, so the program cannot be rebuilt. Capture "
            "the trace again in this session"
        )
    if _stream(trace) is None and int(trace._num_backward_passes or 0):
        return "its event stream was released after earlier backward passes. Capture it again"
    return None


@dataclass(frozen=True)
class _Reached:
    """Which ops and output elements a pass's cotangent reached."""

    elements: Mapping[str, frozenset[int | None]]

    def has(self, label: str) -> bool:
        return label in self.elements


def _reached(program: Program, run: BackwardRun) -> _Reached:
    """Ops reached from the seeded outputs through differentiable occurrences.

    The seeded outputs are the output ops whose cotangent JAX propagated from
    the loss (a symbolic zero, an output the loss ignores, seeds nothing).
    """

    elements: dict[str, set[int | None]] = {}
    for (label, element), seeded in zip(program.out_tap_slots, run.out_seeded, strict=True):
        if seeded and label in program.output_labels:
            elements.setdefault(label, set()).add(element)
    for node in reversed(program.nodes):
        if node.label not in elements:
            continue
        for occ in node.occurrences:
            if occ.tap is not None:
                elements.setdefault(occ.parent, set()).add(occ.element)
    return _Reached({label: frozenset(items) for label, items in elements.items()})


def _op_grad(program: Program, run: BackwardRun, reached: _Reached, node: ProgramNode) -> Any:
    """The node's recorded gradient: per output, ``None`` where nothing reached it."""

    hit = reached.elements.get(node.label, frozenset())
    slots = {slot: index for index, slot in enumerate(program.out_tap_slots)}
    values = []
    for element, tap in enumerate(node.out_taps):
        key = None if not isinstance(node.op.out, tuple | list) else element
        reached_here = key in hit and tap is not None
        values.append(run.out_cts[slots[(node.label, key)]] if reached_here else None)
    return tuple(values) if isinstance(node.op.out, tuple | list) else values[0]


def _outgoing(program: Program, run: BackwardRun, node: ProgramNode) -> tuple[Any, ...]:
    """One outgoing gradient per parent occurrence (adapter convention for multi-result)."""

    outgoing = []
    for occ in node.occurrences:
        grad = None if occ.tap is None else run.edge_cts[occ.tap]
        parent_out = program.node(occ.parent).op.out
        if occ.element is not None and isinstance(parent_out, tuple | list):
            grad = tuple(grad if k == occ.element else None for k in range(len(parent_out)))
        outgoing.append(grad)
    return tuple(outgoing)


def _fired(program: Program, reached: _Reached) -> tuple[ProgramNode, ...]:
    """Reached equation nodes with a differentiable parent occurrence, in forward order."""

    return tuple(
        node
        for node in program.nodes
        if node.kind == "equation"
        and reached.has(node.label)
        and any(occ.tap is not None for occ in node.occurrences)
    )


def _link_problem(fired: tuple[ProgramNode, ...], run: BackwardRun) -> str | None:
    """Every backward equation group must belong to a fired node; else why not."""

    groups, problems = scope_groups(run.backward_jaxpr)
    known = {node.capture.index for node in fired}
    stray = sorted(set(groups) - known)
    if not (problems or stray):
        return None
    detail = "; ".join(problems) or f"equations scoped to captures {stray}"
    return (
        "JAX's backward program does not link one to one with the recorded ops "
        f"({detail}), so its records would be a guess"
    )


def _rule_name(node: ProgramNode) -> str:
    return f"transpose(jvp({node.capture.primitive}))"


def _discoveries(fired: tuple[ProgramNode, ...]) -> list[GradFnDiscovered]:
    ids = {node.label: node.capture.index for node in fired}
    events = []
    for node in reversed(fired):
        topology = tuple(ids[occ.parent] for occ in node.occurrences if occ.parent in ids)
        events.append(
            GradFnDiscovered(
                object_id=node.capture.index,
                class_name=_rule_name(node),
                class_qualname=f"jax.{_rule_name(node)}",
                is_custom=False,
                op_label=node.label,
                param_ref=None,
                created_in_pass=None,
                creator_object_id=None,
                source=_EMPTY_SOURCE,
                topology=topology,
            )
        )
    return events


def _shape_dtype_memory(value: Any) -> tuple[Any, Any, Any]:
    if isinstance(value, tuple):
        return None, None, sum(int(getattr(v, "nbytes", 0) or 0) for v in value if v is not None)
    shape = tuple(int(dim) for dim in value.shape)
    return shape, str(value.dtype), int(value.nbytes)


@dataclass(frozen=True)
class _Solved:
    """One backward pass's program, JAX run and reach, read together by the journal."""

    program: Program
    run: BackwardRun
    reached: _Reached


def _op_grad_events(solved: _Solved, pass_index: int, stamp: float) -> list[OpGradObserved]:
    program, run, reached = solved.program, solved.run, solved.reached
    events = []
    for node in program.nodes:
        if not reached.has(node.label):
            continue
        grad = _op_grad(program, run, reached, node)
        shape, dtype, memory = _shape_dtype_memory(grad)
        events.append(
            OpGradObserved(
                op_label=node.label,
                pass_index=pass_index,
                payload_ref=grad,
                transformed_payload_ref=None,
                shape=shape,
                dtype=dtype,
                memory=memory,
                timestamp=stamp,
            )
        )
    return events


def _param_address(trace: Any, node: ProgramNode) -> str | None:
    """The ``trace.params`` address of a source op that is a leaf of the params argument.

    Uses the JAX capture's own rule: an input leaf at container path ``0.<address>``
    is the parameter ``<address>`` when ``trace.params`` holds it.
    """

    if node.kind != "source" or trace._param_logs is None:
        return None
    annotations = getattr(node.op, "annotations", None) or {}
    path = annotations.get("jax_container_path")
    if not isinstance(path, str) or not path.startswith("0."):
        return None
    address = path.removeprefix("0.")
    return address if address in trace._param_logs else None


def _param_grad_events(
    trace: Any, solved: _Solved, pass_index: int, stamp: float
) -> list[ParamGradObserved]:
    program, run, reached = solved.program, solved.run, solved.reached
    events = []
    for node in program.nodes:
        address = _param_address(trace, node)
        if address is None or not reached.has(node.label):
            continue
        grad = _op_grad(program, run, reached, node)
        shape, dtype, memory = _shape_dtype_memory(grad)
        events.append(
            ParamGradObserved(
                param_address=address,
                pass_index=pass_index,
                payload_ref=grad,
                shape=shape,
                dtype=dtype,
                memory=memory,
                timestamp=stamp,
            )
        )
    return events


def _journal(trace: Any, solved: _Solved, fired: tuple[ProgramNode, ...], started: float) -> None:
    program, run, reached = solved.program, solved.run, solved.reached
    _ensure_stream(trace)
    pass_index = next_backward_pass_index(trace)
    stamp = time.time()
    events: list[Any] = [_start_event(pass_index, started)]
    events.extend(_discoveries(fired))
    events.extend(
        GradFnFired(
            object_id=node.capture.index,
            pass_index=pass_index,
            grad_input_refs=_outgoing(program, run, node),
            grad_output_refs=(_op_grad(program, run, reached, node),),
            fire_record=None,
            timestamp=stamp,
        )
        for node in reversed(fired)
    )
    events.extend(_op_grad_events(solved, pass_index, stamp))
    events.extend(_param_grad_events(trace, solved, pass_index, stamp))
    events.append(_end_event(pass_index, run.duration))
    for event in events:
        journal_backward_event(trace, event)
    set_pass_roots(trace, pass_index, _roots(program, fired))


def _roots(program: Program, fired: tuple[ProgramNode, ...]) -> list[int]:
    """Fired nodes that feed an output op directly (they receive the loss's cotangent)."""

    feeding = {
        occ.parent for label in program.output_labels for occ in program.node(label).occurrences
    }
    return [node.capture.index for node in fired if node.label in feeding]


def _start_event(pass_index: int, started: float) -> BackwardPassStart:
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
        timestamp=started,
    )


def _end_event(pass_index: int, duration: float) -> BackwardPassEnd:
    return BackwardPassEnd(
        pass_index=pass_index,
        duration=duration,
        peak_memory=None,
        status="ok",
        order_attribution_coverage=None,
    )


def _forward_problem(program: Program, run: BackwardRun) -> str | None:
    for node in program.nodes:
        if not jax_adapter.close(node.op.out, run.forward[node.label]):
            return (
                f"the rebuilt program does not reproduce op {node.label!r}, so its backward "
                "would not be the captured program's"
            )
    return None


def _record(trace: Any, loss: Any, backward_kwargs: Mapping[str, Any]) -> str | None:
    """Run and journal one pass; return why it cannot be recorded, or ``None``."""

    problem = _call_problem(trace, loss, backward_kwargs)
    if problem is not None:
        return problem
    started = time.time()
    program = build_program(trace)
    if isinstance(program, str):
        return program
    run = run_backward(program, loss)
    if isinstance(run, str):
        return run
    reached = _reached(program, run)
    fired = _fired(program, reached)
    problem = _forward_problem(program, run) or _link_problem(fired, run)
    if problem is not None:
        return problem
    _journal(trace, _Solved(program, run, reached), fired, started)
    return None


def record_backward(trace: Any, loss: Callable[[Any], Any], **backward_kwargs: Any) -> None:
    """Run one JAX backward pass from ``loss(trace.output)`` and journal it.

    Parameters
    ----------
    trace:
        A JAX trace that still holds its equation captures.
    loss:
        Callable mapping the model's return value (``trace.output``) to a real
        scalar, with ``loss_fn``'s meaning.

    Raises
    ------
    RunCapabilityUnavailableError
        For keyword arguments, a non-callable loss, dropped captures, an op the
        program cannot rebuild, a non-scalar loss, or a backward program that
        does not link one to one with the recorded ops.
    """

    problem = _record(trace, loss, backward_kwargs)
    if problem is not None:
        raise RunCapabilityUnavailableError(
            f"backward() on this JAX trace is unavailable: {problem}. No backward pass was "
            "recorded.",
            code=_CODE,
            detection_stage=_DETECTION_STAGE,
        )


_LOSS_LABEL = "<loss>"


@dataclass(frozen=True)
class _LossConsumer:
    """The loss callable as the consumer of the output ops, for the validator.

    The loss's own cotangent to each output op is where a backward pass starts.
    An output op that other ops also consume receives it beside their
    contributions, so the validator sums it in like any other consumer.
    """

    loss: Callable[[Any], Any]
    treedef: Any

    def cotangents(self, outputs: Sequence[Any]) -> tuple[Any, ...]:
        """JAX's cotangent of the loss per output (``None`` where the loss ignores it)."""

        import jax
        import jax.numpy as jnp

        def loss_of(*leaves: Any) -> Any:
            return self.loss(jax.tree.unflatten(self.treedef, list(leaves)))

        value, pullback = jax.vjp(loss_of, *outputs)
        seed = jnp.ones_like(value)
        live = seed_dependent(jax.make_jaxpr(pullback)(seed))
        return tuple(ct if dep else None for ct, dep in zip(pullback(seed), live, strict=True))


def _graph(program: Program, consumer: _LossConsumer) -> tuple[ReplayGraph, Any]:
    """The validator's graph (equation, output and loss nodes in forward order), and the loss call."""

    import jax
    import jax.numpy as jnp

    values = {node.label: node.op.out for node in program.nodes}
    outputs = [values[label] for label in program.output_labels]
    loss_value = consumer.loss(jax.tree.unflatten(program.output_treedef, outputs))
    values[_LOSS_LABEL] = loss_value
    nodes = tuple(
        EngineNode(
            op=node.op,
            parents=tuple(occ.parent for occ in node.occurrences),
            outputs=(node.label,),
        )
        for node in program.nodes
        if node.kind != "source"
    )
    loss_node = EngineNode(op=consumer, parents=program.output_labels, outputs=(_LOSS_LABEL,))
    call = ((jnp.ones_like(loss_value),), consumer.cotangents(outputs))
    return ReplayGraph(nodes=(*nodes, loss_node), value=values.__getitem__), call


@dataclass(frozen=True)
class _MarkerAdapter:
    """The JAX replay adapter, plus the derivatives of the two non-equation consumers.

    An output marker is not a JAX equation, so the JAX adapter does not replay
    it; its derivative is the identity by definition (it returns its parent's
    value). The loss consumer's derivative is JAX's derivative of the loss
    callable. Every other op goes to the JAX adapter unchanged.
    """

    adapter: Any
    markers: frozenset[str]

    def can_replay(self, trace: Any, op: Any) -> str | None:
        if isinstance(op, _LossConsumer) or jax_adapter._parent_label(op) in self.markers:
            return None
        return self.adapter.can_replay(trace, op)

    def vjp(self, trace: Any, op: Any, parent_values: Any, cotangents: Any) -> tuple[Any, ...]:
        if isinstance(op, _LossConsumer):
            return op.cotangents(parent_values)
        if jax_adapter._parent_label(op) in self.markers:
            return (cotangents[0],)
        return tuple(self.adapter.vjp(trace, op, parent_values, cotangents))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.adapter, name)


@dataclass(frozen=True)
class ValidationInputs:
    """What ``validation._backward_neutral.check_backward_records`` takes for one pass."""

    graph: ReplayGraph
    calls: dict[str, Any]
    grads: dict[str, Any]
    adapter: Any


def _pass_grads(trace: Any, pass_number: int) -> dict[str, Any]:
    """Recorded ``Op.grad`` per op ``label`` (pass-qualified, as the graph names ops)."""

    grads: dict[str, Any] = {}
    for op in trace._layer_list:
        for record in op._slot("_grad_records") or ():
            if record.backward_pass_number == pass_number and record.grad is not None:
                grads[jax_adapter._parent_label(op)] = record.grad
    return grads


def validation_inputs(
    trace: Any, recorded: Any, pass_number: int, loss: Callable[[Any], Any]
) -> ValidationInputs:
    """The graph, calls, gradients and adapter to check one recorded pass with.

    ``recorded`` is ``recorded_backward_from_trace(trace, pass_number)``; its
    calls are keyed by the ``GradFn``'s op label. Gradients are read by the
    pass-qualified op label the graph uses, so ops recorded in several forward
    passes keep one gradient each. ``loss`` is the callable the pass
    differentiated: the graph ends in one loss node that consumes the output
    ops, and its call carries JAX's cotangent of the loss per output, so an
    output op that other ops also consume is checked against the sum of the
    loss's cotangent and theirs. An output marker (TorchLens's node for a
    returned value) passes its gradient to its producer unchanged: its
    recorded ``Op.grad`` is entered as that identity call. The adapter
    returned checks both with their own derivatives.
    """

    program = build_program(trace)
    if isinstance(program, str):
        no_values: dict[str, Any] = {}
        return ValidationInputs(ReplayGraph((), no_values.__getitem__), {}, {}, jax_adapter)
    grads = _pass_grads(trace, pass_number)
    calls = dict(recorded.calls)
    markers = frozenset(n.label for n in program.nodes if n.kind == "passthrough")
    for label in markers:
        if grads.get(label) is not None:
            calls[label] = ((grads[label],), (grads[label],))
    graph, calls[_LOSS_LABEL] = _graph(program, _LossConsumer(loss, program.output_treedef))
    return ValidationInputs(graph, calls, grads, _MarkerAdapter(jax_adapter, markers))
