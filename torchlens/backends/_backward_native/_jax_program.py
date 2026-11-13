"""The interpreted JAX program behind native backward capture.

The forward is rebuilt from the trace's kept equation captures: every captured
equation is bound again through the validator's own replay
(``jax.jaxpr.replay_equation``) inside ``jax.named_scope("tlb_<capture index>")``.
``jax.vjp`` of that program gives JAX's pullback; ``jax.make_jaxpr`` of the
pullback is the backward program, and JAX's own jaxpr evaluator
(``jax.extend.core.jaxpr_as_fun``) runs exactly that program. Each backward equation carries the scope of the
forward equation whose rule emitted it (``transpose(jvp(tlb_7))``), which links
it to its forward op.

Gradient values at node boundaries are read without changing the backward
program: each differentiable value and each differentiable parent occurrence
passes through the backend's zero tap (``_jax_zero_tap``, an identity whose
JVP adds the tap's tangent), so the cotangent JAX's backward computes at that
point comes out as the tap's gradient. A tap adds no backward equation (its
linear part is an ``add_any`` whose transpose fans out).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..._utils._record_group_table import record_member
from .._replay_adapters import jax as jax_adapter
from ..jax.jaxpr import _jax_zero_tap, replay_equation

__all__ = (
    "BackwardRun",
    "Occurrence",
    "Program",
    "ProgramNode",
    "build_program",
    "program_function",
    "run_backward",
    "scope_groups",
    "seed_dependent",
)

TLB_SCOPE = re.compile(r"tlb_(\d+)")
"""Scope tag of a rebuilt forward equation, by capture index."""


@dataclass(frozen=True)
class Occurrence:
    """One parent occurrence of a node, in argument-position order.

    ``element`` selects one output of a multi-result parent; ``tap`` indexes the
    edge tap of a differentiable occurrence, ``None`` for one that is not.
    """

    parent: str
    position: int
    element: int | None
    tap: int | None


@dataclass(frozen=True)
class ProgramNode:
    """One recorded op in forward order: an equation, a source or an output marker."""

    op: Any
    label: str
    kind: str
    capture: Any
    occurrences: tuple[Occurrence, ...]
    out_taps: tuple[int | None, ...]


@dataclass(frozen=True)
class Program:
    """The rebuilt forward: nodes, differentiated sources, output wiring and tap counts."""

    nodes: tuple[ProgramNode, ...]
    leaves: tuple[str, ...]
    output_labels: tuple[str, ...]
    output_treedef: Any
    out_tap_slots: tuple[tuple[str, int | None], ...]
    edge_tap_count: int
    by_label: Mapping[str, ProgramNode]

    def node(self, label: str) -> ProgramNode:
        """Return the node with ``label``."""

        return self.by_label[label]


def _is_inexact(value: Any) -> bool:
    import jax.numpy as jnp

    dtype = getattr(value, "dtype", None)
    return dtype is not None and bool(jnp.issubdtype(dtype, jnp.inexact))


def _elements(value: Any) -> tuple[tuple[int | None, Any], ...]:
    """``(element index, value)`` per output; a single value has index ``None``."""

    if isinstance(value, tuple | list):
        return tuple(enumerate(value))
    return ((None, value),)


def _select(value: Any, element: int | None) -> Any:
    return value if element is None else value[element]


class _TapCounter:
    """Hands out tap indexes in program order."""

    def __init__(self) -> None:
        self.out_slots: list[tuple[str, int | None]] = []
        self.edges = 0

    def out_taps(self, label: str, value: Any) -> tuple[int | None, ...]:
        taps: list[int | None] = []
        for element, item in _elements(value):
            if _is_inexact(item):
                taps.append(len(self.out_slots))
                self.out_slots.append((label, element))
            else:
                taps.append(None)
        return tuple(taps)

    def edge(self, value: Any) -> int | None:
        if not _is_inexact(value):
            return None
        self.edges += 1
        return self.edges - 1


def _occurrences(plan: Any, values: Mapping[str, Any], taps: _TapCounter) -> tuple[Occurrence, ...]:
    occurrences = []
    for position, parent, element in zip(
        plan.positions, plan.parents, plan.element_index, strict=True
    ):
        label = jax_adapter._parent_label(parent)
        tap = taps.edge(_select(values[label], element))
        occurrences.append(Occurrence(label, position, element, tap))
    return tuple(occurrences)


def _element_of(parent_value: Any, value: Any) -> int | None:
    """Which output of a multi-result parent ``value`` is (``None``: a single result)."""

    if not isinstance(parent_value, tuple | list):
        return None
    return next((k for k, item in enumerate(parent_value) if item is value), None)


def _leaf_node(
    op: Any, label: str, values: dict[str, Any], taps: _TapCounter, aliases: Mapping[str, str]
) -> ProgramNode | str | None:
    """A source (no parents) or an output marker (passes its one parent through).

    Both are TorchLens graph entries, not JAX equations: a source's value is
    given, and an output marker is the identity on the value the program returns.
    Returns ``None`` for any other non-equation op.
    """

    parents = tuple(str(p) for p in record_member(op, "parents", None) or ())
    if not parents:
        if op.out is None:
            return f"source op {label!r} has no saved value to differentiate"
        values[label] = op.out
        return ProgramNode(op, label, "source", None, (), taps.out_taps(label, op.out))
    if not record_member(op, "is_output", False) or len(parents) != 1 or parents[0] not in aliases:
        return None
    parent = aliases[parents[0]]
    element = _element_of(values[parent], op.out)
    occurrence = Occurrence(parent, 0, element, taps.edge(_select(values[parent], element)))
    values[label] = op.out
    return ProgramNode(op, label, "passthrough", None, (occurrence,), taps.out_taps(label, op.out))


def _node_for(
    trace: Any, op: Any, values: dict[str, Any], taps: _TapCounter, aliases: Mapping[str, str]
) -> ProgramNode | str:
    """The program node of one op, or why it cannot be rebuilt."""

    plan = jax_adapter._plan_or_reason(trace, op)
    label = jax_adapter._parent_label(op)
    if isinstance(plan, str):
        leaf = _leaf_node(op, label, values, taps, aliases)
        return f"op {label!r} cannot be rebuilt for backward: {plan}" if leaf is None else leaf
    values[label] = op.out
    occurrences = _occurrences(plan, values, taps)
    return ProgramNode(
        op, label, "equation", plan.capture, occurrences, taps.out_taps(label, op.out)
    )


def _aliases(op: Any, label: str) -> dict[str, str]:
    """Both spellings a child may use for ``op`` (the bare one only for a single pass)."""

    names = {label: label}
    layer_label = getattr(op, "layer_label", None)
    if layer_label and int(getattr(op, "num_passes", 1) or 1) == 1:
        names[str(layer_label)] = label
    return names


def _output_treedef(trace: Any, output_ops: Sequence[Any]) -> Any:
    """The tree structure of ``trace.output`` (leaves matched to the output ops), or why not."""

    import jax

    leaves, treedef = jax.tree.flatten(trace.output)
    if len(leaves) != len(output_ops) or any(
        leaf is not op.out and not jax_adapter.close(leaf, op.out)
        for leaf, op in zip(leaves, output_ops, strict=False)
    ):
        return (
            "the trace's output leaves do not match its output ops one to one, so a loss "
            "callable of trace.output cannot be differentiated"
        )
    return treedef


def build_program(trace: Any) -> Program | str:
    """Rebuild ``trace``'s forward from its kept equation captures.

    Returns the program, or why it cannot be rebuilt: an op the adapter cannot
    plan (opaque region, missing capture, effect) or output leaves that do not
    match the output ops.
    """

    taps = _TapCounter()
    values: dict[str, Any] = {}
    aliases: dict[str, str] = {}
    nodes = []
    for op in trace._layer_list:
        node = _node_for(trace, op, values, taps, aliases)
        if isinstance(node, str):
            return node
        nodes.append(node)
        aliases.update(_aliases(op, node.label))
    output_ops = tuple(trace.output_ops)
    treedef = _output_treedef(trace, output_ops)
    if isinstance(treedef, str):
        return treedef
    return Program(
        nodes=tuple(nodes),
        leaves=tuple(n.label for n in nodes if n.kind == "source" and n.out_taps[0] is not None),
        output_labels=tuple(jax_adapter._parent_label(op) for op in output_ops),
        output_treedef=treedef,
        out_tap_slots=tuple(taps.out_slots),
        edge_tap_count=taps.edges,
        by_label={node.label: node for node in nodes},
    )


def _node_inputs(node: ProgramNode, env: Mapping[str, Any], edge_taps: Sequence[Any]) -> list[Any]:
    inputs = list(node.capture.input_values) if node.capture is not None else [None]
    for occ in node.occurrences:
        value = _select(env[occ.parent], occ.element)
        if occ.tap is not None:
            value = _jax_zero_tap(value, edge_taps[occ.tap])
        inputs[occ.position] = value
    return inputs


def _node_outputs(node: ProgramNode, env: Mapping[str, Any], edge_taps: Sequence[Any]) -> list[Any]:
    """Bind one node; an equation binds inside its ``tlb_`` scope."""

    import jax

    if node.kind == "source":
        return [item for _element, item in _elements(env[node.label])]
    inputs = _node_inputs(node, env, edge_taps)
    if node.kind == "passthrough":
        return [item for _element, item in _elements(inputs[0])]
    with jax.named_scope(f"tlb_{node.capture.index}"):
        return list(replay_equation(node.capture, inputs))


def _tapped(node: ProgramNode, outputs: list[Any], out_taps: Sequence[Any]) -> Any:
    """The node's value with each differentiable output passed through its tap."""

    tapped = [
        item if tap is None else _jax_zero_tap(item, out_taps[tap])
        for item, tap in zip(outputs, node.out_taps, strict=True)
    ]
    return tuple(tapped) if isinstance(node.op.out, tuple | list) else tapped[0]


def program_function(
    program: Program, sources: Mapping[str, Any], loss: Callable[[Any], Any]
) -> Callable[..., tuple[Any, dict[str, Any]]]:
    """``f(leaves, out_taps, edge_taps) -> (loss, forward values)`` for ``jax.vjp``."""

    import jax

    def run(
        leaves: Sequence[Any], out_taps: Sequence[Any], edge_taps: Sequence[Any]
    ) -> tuple[Any, dict[str, Any]]:
        env: dict[str, Any] = dict(sources)
        env.update(zip(program.leaves, leaves, strict=True))
        for node in program.nodes:
            env[node.label] = _tapped(node, _node_outputs(node, env, edge_taps), out_taps)
        raw = jax.tree.unflatten(program.output_treedef, [env[k] for k in program.output_labels])
        return loss(raw), {node.label: env[node.label] for node in program.nodes}

    return run


@dataclass(frozen=True)
class BackwardRun:
    """One evaluated backward program and what it computed.

    Attributes
    ----------
    loss:
        The scalar the pass differentiated.
    forward:
        Rebuilt forward value per op label (for the unchanged-forward check).
    leaf_cts, out_cts, edge_cts:
        Gradient per differentiated source, per tapped output slot, per edge tap.
    out_seeded:
        Per tapped output slot: whether its cotangent depends on the seed
        (``False`` is a symbolic zero JAX never propagated).
    backward_jaxpr:
        The closed backward jaxpr that ran.
    duration:
        Wall time of evaluating it, in seconds.
    """

    loss: Any
    forward: Mapping[str, Any]
    leaf_cts: tuple[Any, ...]
    out_cts: tuple[Any, ...]
    edge_cts: tuple[Any, ...]
    out_seeded: tuple[bool, ...]
    backward_jaxpr: Any
    duration: float


def _zeros(values: Sequence[Any]) -> list[Any]:
    import jax.numpy as jnp

    return [jnp.zeros_like(value) for value in values]


def _primals(program: Program, sources: Mapping[str, Any]) -> tuple[list[Any], ...]:
    edge_values: list[Any] = [None] * program.edge_tap_count
    for node in program.nodes:
        for occ in node.occurrences:
            if occ.tap is not None:
                edge_values[occ.tap] = _select(_saved(program, sources, occ.parent), occ.element)
    out_values = [
        _select(_saved(program, sources, label), element)
        for label, element in program.out_tap_slots
    ]
    leaves = [sources[label] for label in program.leaves]
    return list(leaves), _zeros(out_values), _zeros(edge_values)


def _saved(program: Program, sources: Mapping[str, Any], label: str) -> Any:
    return sources[label] if label in sources else program.node(label).op.out


def seed_dependent(closed: Any) -> list[bool]:
    """Per outvar of ``closed``: whether it depends on the jaxpr's inputs."""

    jaxpr = closed.jaxpr
    live = {id(var) for var in jaxpr.invars}
    for eqn in jaxpr.eqns:
        if any(id(var) in live for var in eqn.invars if type(var).__name__ != "Literal"):
            live.update(id(var) for var in eqn.outvars)
    return [type(var).__name__ != "Literal" and id(var) in live for var in jaxpr.outvars]


def run_backward(program: Program, loss: Callable[[Any], Any]) -> BackwardRun | str:
    """Differentiate ``loss(trace.output)`` through the rebuilt program and run its pullback.

    Returns what ran, or why it cannot run (the loss is not a real scalar).
    """

    import jax
    import jax.numpy as jnp
    from jax.extend.core import jaxpr_as_fun

    sources = {n.label: n.op.out for n in program.nodes if n.kind == "source"}
    function = program_function(program, sources, loss)
    value, pullback, forward = jax.vjp(function, *_primals(program, sources), has_aux=True)
    if getattr(value, "shape", None) != () or not _is_inexact(value):
        return "the loss callable must return a real floating-point scalar"
    seed = jnp.ones_like(value)
    closed, out_shape = jax.make_jaxpr(pullback, return_shape=True)(seed)
    structure = jax.tree.structure(out_shape)
    started = time.perf_counter()
    flat = jax.block_until_ready(jaxpr_as_fun(closed)(seed))
    duration = time.perf_counter() - started
    leaf_cts, out_cts, edge_cts = jax.tree.unflatten(structure, list(flat))
    _leaf_dep, out_dep, _edge_dep = jax.tree.unflatten(structure, seed_dependent(closed))
    return BackwardRun(
        loss=value,
        forward=forward,
        leaf_cts=tuple(leaf_cts),
        out_cts=tuple(out_cts),
        edge_cts=tuple(edge_cts),
        out_seeded=tuple(bool(flag) for flag in out_dep),
        backward_jaxpr=closed,
        duration=duration,
    )


def _eqn_scopes(eqn: Any) -> set[int]:
    return {int(match) for match in TLB_SCOPE.findall(str(eqn.source_info.name_stack))}


def _eqn_group(eqn: Any, group_of_var: Mapping[int, int | None]) -> tuple[int | None, str | None]:
    """The forward capture index one backward equation belongs to, and any problem."""

    scopes = _eqn_scopes(eqn)
    if len(scopes) > 1:
        return None, f"{eqn.primitive.name} carries scopes {sorted(scopes)}"
    group = next(iter(scopes), None)
    if group is not None and eqn.primitive.name == "add_any":
        sources = {group_of_var.get(id(var), -1) for var in eqn.invars}
        if sources != {group}:
            return None, None
    return group, None


def scope_groups(closed: Any) -> tuple[dict[int, list[str]], list[str]]:
    """Backward primitives per forward capture index, and the problems found.

    An ``add_any`` whose inputs come from more than one group is the engine
    summing contributions (PyTorch does that outside any ``GradFn``), so it
    belongs to no node even though JAX tags it with one consumer's scope. An
    equation tagged with more than one forward scope is a problem.
    """

    groups: dict[int, list[str]] = {}
    group_of_var: dict[int, int | None] = {}
    problems: list[str] = []
    for eqn in closed.jaxpr.eqns:
        group, problem = _eqn_group(eqn, group_of_var)
        if problem is not None:
            problems.append(problem)
        for var in eqn.outvars:
            group_of_var[id(var)] = group
        if group is not None:
            groups.setdefault(group, []).append(eqn.primitive.name)
    return groups, problems
