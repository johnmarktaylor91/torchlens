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
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..._ir.events import (
    BackwardPassEnd,
    BackwardPassStart,
    GradFnDiscovered,
    GradFnFired,
    OpGradObserved,
    ParamGradObserved,
)
from .._backward_neutral import journal_backward_event, next_backward_pass_index, set_pass_roots
from .._replay_adapters import jax as jax_adapter
from .._reverse_engine import EngineNode, ReplayGraph
from . import _refuse
from ._jax_program import (
    BackwardRun,
    Program,
    ProgramNode,
    ProgramRefusal,
    build_program,
    run_backward,
    scope_groups,
)

__all__ = (
    "backward_ready_scope",
    "record_backward",
    "retain_backward_state",
    "validation_inputs",
)

_EMPTY_SOURCE: Mapping[str, object | None] = MappingProxyType({})


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
    """Keep the equation captures a later ``record_backward`` rebuilds the program from.

    A capture that drops captures to save memory calls this at the end of a
    backward-ready forward; the captures it pins are the ones the validator
    already holds for a full capture.
    """

    del session
    captures = tuple(getattr(trace, "jax_equation_captures", None) or ())
    trace.__dict__["_jax_backward_captures"] = captures


def _captures_present(trace: Any) -> bool:
    kept = trace.__dict__.get("_jax_backward_captures")
    if kept and not getattr(trace, "jax_equation_captures", None):
        trace.jax_equation_captures = kept
    return bool(getattr(trace, "jax_equation_captures", None))


def _check_call(trace: Any, loss: Any, backward_kwargs: Mapping[str, Any]) -> None:
    if backward_kwargs:
        names = ", ".join(sorted(backward_kwargs))
        raise _refuse(
            f"backward() on a JAX trace takes no keyword arguments (got {names}): JAX's "
            "pullback has no engine options."
        )
    if not callable(loss):
        raise _refuse(
            "backward() on a JAX trace needs loss as a callable mapping trace.output to a "
            "scalar: a JAX array computed after capture has no link to the captured program."
        )
    if not _captures_present(trace):
        raise _refuse(
            "backward() on this JAX trace is unavailable: its equation captures were not "
            "kept, so the program cannot be rebuilt. Capture the trace again in this session."
        )


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


def _check_links(fired: tuple[ProgramNode, ...], run: BackwardRun) -> dict[int, list[str]]:
    """Every backward equation group must belong to a fired node; returns the groups."""

    groups, problems = scope_groups(run.backward_jaxpr)
    known = {node.capture.index for node in fired}
    stray = sorted(set(groups) - known)
    if problems or stray:
        detail = "; ".join(problems) or f"equations scoped to captures {stray}"
        raise _refuse(
            "JAX's backward program does not link one to one with the recorded ops "
            f"({detail}), so its records would be a guess. No backward pass was recorded."
        )
    return groups


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


def _op_grad_events(
    program: Program, run: BackwardRun, reached: _Reached, pass_index: int, stamp: float
) -> list[OpGradObserved]:
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
    trace: Any,
    program: Program,
    run: BackwardRun,
    reached: _Reached,
    pass_index: int,
    stamp: float,
) -> list[ParamGradObserved]:
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


def _journal(trace: Any, program: Program, run: BackwardRun, started: float) -> None:
    pass_index = next_backward_pass_index(trace)
    reached = _reached(program, run)
    fired = _fired(program, reached)
    _check_links(fired, run)
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
    events.extend(_op_grad_events(program, run, reached, pass_index, stamp))
    events.extend(_param_grad_events(trace, program, run, reached, pass_index, stamp))
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


def _forward_unchanged(program: Program, run: BackwardRun) -> None:
    for node in program.nodes:
        if not jax_adapter.close(node.op.out, run.forward[node.label]):
            raise _refuse(
                f"the rebuilt JAX program does not reproduce op {node.label!r}, so its "
                "backward would not be the captured program's. No backward pass was recorded."
            )


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

    _check_call(trace, loss, backward_kwargs)
    started = time.time()
    try:
        program = build_program(trace)
        run = run_backward(program, trace, loss)
    except ProgramRefusal as exc:
        raise _refuse(f"backward() on this JAX trace is unavailable: {exc}") from exc
    _forward_unchanged(program, run)
    _journal(trace, program, run, started)


def _graph(program: Program) -> ReplayGraph:
    """The validator's graph: equation and output nodes in forward order."""

    values = {node.label: node.op.out for node in program.nodes}
    nodes = tuple(
        EngineNode(
            op=node.op,
            parents=tuple(occ.parent for occ in node.occurrences),
            outputs=(node.label,),
        )
        for node in program.nodes
        if node.kind != "source"
    )
    return ReplayGraph(nodes=nodes, value=values.__getitem__)


def validation_inputs(trace: Any, recorded: Any) -> tuple[ReplayGraph, dict[str, Any]]:
    """The graph and calls ``validation._backward_neutral.check_backward_records`` takes.

    ``recorded`` is ``recorded_backward_from_trace(trace, pass_number)``. An
    output op is TorchLens's marker for a returned value: its gradient is the
    loss's cotangent, which it passes to its producer unchanged. Its recorded
    ``Op.grad`` is entered as that identity call, so the producer's gradient is
    checked as the sum of the loss's cotangent and its other consumers'
    contributions. Returns the graph and the calls to check.
    """

    program = build_program(trace)
    calls = dict(recorded.calls)
    for label in program.output_labels:
        grad = recorded.grads.get(label)
        if grad is not None and program.node(label).kind == "passthrough":
            calls[label] = ((grad,), (grad,))
    return _graph(program), calls
