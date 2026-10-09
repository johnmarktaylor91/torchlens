"""Regions: derived admissibility on the existing verbs (F01, surgery memo 3.4).

A region is a user-selected contiguous chunk of the executed graph treated as
ONE unit, with entry edges (values flowing in) and exit edges (values flowing
out). Regions land on the verbs that exist -- ``fork.do(region, edit)`` --
with zero new vocabulary; admissibility is DERIVED by TorchLens, never
asserted by the user:

1. **Complete exits, derived.** The exit set is ALL edges leaving the region;
   a user-supplied exit list that leaks refuses, naming the leaked edges.
2. **Convexity.** No path may leave the region and re-enter it (checked in
   the shipped address algebra as ``between(R, R) <= R``); the refusal
   prints one offending leave-and-re-enter path.
3. **Pass closure.** Members are pass-qualified op labels throughout; the
   region's replacement function is evaluated ONCE per region instance (a
   weakly-connected component of the internal dataflow -- one traversal =
   one instance), and all of an instance's exits commit atomically or none
   do (with a stochastic replacement, per-exit re-evaluation would be
   silently wrong).
4. **Effect closure -- replay lane only, by delegation.** The check consults
   evidence the capture already records (buffer write kinds and
   value-change evidence, collective-boundary annotations, in-place-op
   convention) -- it cites that ledger, never stands up a second authority.
   A known outward effect crossing the boundary refuses and names the
   channel and op; for channels TorchLens cannot observe the audit says
   "effect closure not certified", never "effect-free".

MECHANISM HONESTY (foldA D17): the replay lowering substitutes the region's
EXIT values and re-executes the consumers -- the interior is NOT replayed --
so every region firing disclosure carries the engine-set closed
``execution_effect`` value ``"exits_substituted_interior_not_replayed"``.
Execution removal exists in no lane.

``splice_module`` LOWERS to a region edit here: one code path, one audit
shape, one set of closure checks. A region target is one typed, ordered
boundary object (:class:`RegionBoundary`) naming every derived entry and
exit role, so module-call inputs and arbitrary dataflow entries are never
silently conflated.

Every spelling DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

from .errors import RegionError
from .types import HelperSpec

#: The engine-set execution-effect value for the region replay lowering.
_REGION_EXECUTION_EFFECT = "exits_substituted_interior_not_replayed"

#: The effect-closure channels the delegation CAN certify (recorded evidence).
_CERTIFIED_CHANNELS = ("buffer_writes", "collectives", "inplace_ops")

#: The honest disclosure for everything else (never "effect-free").
_UNCERTIFIED_NOTE = (
    "effect closure not certified beyond recorded evidence channels "
    f"{_CERTIFIED_CHANNELS}: unobserved side channels (host state, RNG "
    "consumption by uncaptured code, I/O) are not claimable"
)


@dataclass(frozen=True)
class RegionEdge:
    """One typed boundary edge of a region.

    Parameters
    ----------
    role:
        ``"entry"`` (value flows into the region) or ``"exit"`` (value flows
        out of it).
    parent:
        Pass-qualified label of the producing op.
    child:
        Pass-qualified label of the consuming op.
    addresses:
        The canonical edge occurrence addresses
        ``(child_func_call_id, arg_kind, arg_path)`` this dataflow edge
        resolves to (one value may be consumed at several argument
        positions).
    nested:
        Whether any occurrence address has a nested (container) argument
        path -- disclosed because the splice must extend into the container.
    """

    role: str
    parent: str
    child: str
    addresses: tuple[tuple[Any, ...], ...]
    nested: bool


@dataclass(frozen=True)
class RegionBoundary:
    """The typed, ordered boundary object of one region.

    ``entries`` and ``exits`` are ordered by (consuming-op execution order,
    occurrence order), so module-call inputs and arbitrary dataflow entries
    are named individually, never silently conflated.
    """

    entries: tuple[RegionEdge, ...]
    exits: tuple[RegionEdge, ...]


@dataclass(frozen=True)
class RegionInstance:
    """One pass instance of a region (a weakly-connected member component).

    The replacement function is evaluated ONCE per instance and all of the
    instance's exits commit atomically.
    """

    index: int
    members: tuple[str, ...]
    exit_ops: tuple[str, ...]
    exits: tuple[RegionEdge, ...]
    entries: tuple[RegionEdge, ...]


class RegionTarget:
    """A frozen admissible region: members + typed boundary + instances.

    Built by :func:`region` (or ``TraceSlice.as_region``); every
    admissibility fact was DERIVED at construction -- complete exits,
    convexity, pass-instance partition. Effect closure is checked at
    ``do()`` time on the replay lane (live lanes carry no effect gate).
    """

    __slots__ = ("_boundary", "_digest", "_instances", "_slice")

    _boundary: RegionBoundary
    _digest: str
    _instances: tuple[RegionInstance, ...]
    _slice: Any

    def __init__(
        self, slice_: Any, boundary: RegionBoundary, instances: tuple[RegionInstance, ...]
    ):
        """Freeze one derived region (internal; use :func:`region`)."""

        object.__setattr__(self, "_slice", slice_)
        object.__setattr__(self, "_boundary", boundary)
        object.__setattr__(self, "_instances", instances)
        payload = (
            "|".join(slice_.labels)
            + "||"
            + "|".join(repr(address) for edge in boundary.exits for address in edge.addresses)
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        object.__setattr__(self, "_digest", f"region-{digest}")

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse mutation after freeze."""

        raise AttributeError("RegionTarget is frozen; derive a new region instead.")

    @property
    def source_slice(self) -> Any:
        """The ``TraceSlice`` this region was derived from."""

        return self._slice

    @property
    def trace(self) -> Any:
        """The trace the region is bound to."""

        return self._slice.source_trace

    @property
    def members(self) -> tuple[str, ...]:
        """Pass-qualified member op labels in execution order."""

        return self._slice.labels

    @property
    def boundary(self) -> RegionBoundary:
        """The typed, ordered entry/exit boundary object."""

        return self._boundary

    @property
    def instances(self) -> tuple[RegionInstance, ...]:
        """The pass instances (one replacement evaluation each)."""

        return self._instances

    @property
    def region_digest(self) -> str:
        """Stable digest over members + exit addresses (audit identity)."""

        return self._digest

    def __repr__(self) -> str:
        """Disclosure-first repr: members, instances, boundary sizes."""

        return (
            f"<region {self._digest}: {len(self.members)} member op(s), "
            f"{len(self._instances)} instance(s), "
            f"{len(self._boundary.entries)} entry / {len(self._boundary.exits)} exit "
            "edge(s); replay do() substitutes exits, interior not replayed>"
        )


# ---------------------------------------------------------------------------
# construction: derived admissibility
# ---------------------------------------------------------------------------


def _edge_records_by_call(trace: Any) -> dict[int, list[Any]]:
    """Index the trace's EdgeUseRecords by consuming ``child_func_call_id``."""

    from .edge_substitution import _trace_edge_records

    by_call: dict[int, list[Any]] = {}
    for record in _trace_edge_records(trace):
        by_call.setdefault(record.child_func_call_id, []).append(record)
    return by_call


def _resolve_boundary_edge(
    role: str,
    parent_label: str,
    child_label: str,
    graph: Any,
    by_call: dict[int, list[Any]],
) -> RegionEdge:
    """Resolve one boundary dataflow edge to its occurrence addresses.

    An exit edge the pass-exact edge family cannot address gets EMPTY
    ``addresses`` here; the ``do()`` lowering refuses it typed AFTER the
    effect-closure delegation runs, so an outward-effect member (the more
    fundamental violation) is named first.
    """

    parent_op = graph.ops[parent_label]
    child_op = graph.ops[child_label]
    candidates = by_call.get(child_op.func_call_id, [])
    matches = []
    for record in candidates:
        if record.parent_func_call_id is not None:
            if record.parent_func_call_id == parent_op.func_call_id:
                matches.append(record)
        elif record.parent_label == parent_op.layer_label:
            matches.append(record)
    addresses = tuple(
        (record.child_func_call_id, record.arg_kind, tuple(record.arg_path)) for record in matches
    )
    nested = any(len(address[2]) != 1 for address in addresses)
    return RegionEdge(
        role=role, parent=parent_label, child=child_label, addresses=addresses, nested=nested
    )


def _connected_instances(
    members: tuple[str, ...], internal_edges: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, ...], ...]:
    """Partition members into weakly-connected components (pass instances)."""

    neighbors: dict[str, set[str]] = {label: set() for label in members}
    for parent, child in internal_edges:
        neighbors[parent].add(child)
        neighbors[child].add(parent)
    seen: set[str] = set()
    components: list[tuple[str, ...]] = []
    for label in members:  # members are already execution-ordered
        if label in seen:
            continue
        stack, component = [label], []
        seen.add(label)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in neighbors[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        order = {member: position for position, member in enumerate(members)}
        components.append(tuple(sorted(component, key=order.__getitem__)))
    return tuple(components)


def _offending_reentry_path(graph: Any, members: frozenset[str], closure: frozenset[str]) -> str:
    """Find one leave-and-re-enter path witnessing a convexity violation."""

    outside = closure - members
    for start in sorted(outside):
        # walk backward to a member, forward to a member: start sits on a
        # member -> outside -> member path by construction of the closure.
        back: list[str] = [start]
        node = start
        while node not in members:
            inside_parents = [p for p in sorted(graph.parents.get(node, ())) if p in closure]
            if not inside_parents:
                break
            node = inside_parents[0]
            back.append(node)
        if node not in members:
            continue
        forward: list[str] = []
        node = start
        while node not in members:
            inside_children = [c for c in sorted(graph.children.get(node, ())) if c in closure]
            if not inside_children:
                break
            node = inside_children[0]
            forward.append(node)
        if node not in members:
            continue
        return " -> ".join(reversed(back)) + " -> " + " -> ".join(forward)
    return "(path reconstruction unavailable; offending ops: " + ", ".join(sorted(outside)) + ")"


def _validate_declared_exits(exit_edges: tuple[RegionEdge, ...], exits: Any) -> None:
    """Check a user-declared exit list against the DERIVED exit set.

    Completeness is checked, never trusted: malformed entries and non-exit
    edges refuse ``region_exits_unknown``; a list that misses a derived exit
    edge LEAKS and refuses ``region_exits_incomplete`` naming the leaked
    edges.
    """

    declared: set[Any] = set()
    for item in exits:
        if isinstance(item, tuple):
            declared.add(tuple(item))
        elif isinstance(item, str):
            declared.add(item)
        else:
            raise RegionError(
                f"exits= entries must be (parent, child) pairs or parent "
                f"labels; got {type(item).__name__}",
                code="region_exits_unknown",
                remedy="declare exits as (parent_label, child_label) "
                "pairs or bare parent op labels",
            )
    derived_pairs = {(edge.parent, edge.child) for edge in exit_edges}
    derived_parents = {edge.parent for edge in exit_edges}
    unknown = {
        item
        for item in declared
        if (item not in derived_pairs if isinstance(item, tuple) else item not in derived_parents)
    }
    if unknown:
        raise RegionError(
            f"declared exits {sorted(map(repr, unknown))} are not exit edges of this region",
            code="region_exits_unknown",
            remedy="declare only edges that actually leave the region; "
            "read region(slice).boundary.exits for the derived set",
        )
    leaked = [
        edge
        for edge in exit_edges
        if (edge.parent, edge.child) not in declared and edge.parent not in declared
    ]
    if leaked:
        names = ", ".join(f"{edge.parent!r} -> {edge.child!r}" for edge in leaked)
        raise RegionError(
            f"the declared exit list LEAKS: derived exit edges [{names}] "
            "are not covered; the exit set is ALL edges leaving the "
            "region, derived, never asserted",
            code="region_exits_incomplete",
            remedy="cover every derived exit edge (or drop exits= and let "
            "TorchLens derive the complete set)",
        )


def region(slice_: Any, *, exits: Any = None) -> RegionTarget:
    """Derive an admissible region from a ``TraceSlice`` (F01, memo 3.4).

    Parameters
    ----------
    slice_:
        The ``TraceSlice`` naming the member ops (``trace.between(...)`` /
        ``trace.subgraph(...)``).
    exits:
        Optional user-declared exit list -- ``(parent_label, child_label)``
        pairs or bare parent labels. Completeness is CHECKED, never trusted:
        a declared list that misses a derived exit edge refuses, naming the
        leaked edges.

    Raises
    ------
    RegionError
        ``region_empty`` / ``region_not_convex`` / ``region_exits_unknown``
        / ``region_exits_incomplete`` / ``region_exit_address_underivable``.
    """

    from ..trace_slice import TraceSlice

    if not isinstance(slice_, TraceSlice):
        raise RegionError(
            f"region() derives from a TraceSlice; received {type(slice_).__name__}",
            code="region_target_invalid",
            remedy="build the member set first: trace.between(sources, sinks) "
            "or trace.subgraph(selection), then region(slice)",
        )
    if slice_.empty:
        raise RegionError(
            "region() refuses an empty slice: a region with no members has no "
            "boundary to operate on",
            code="region_empty",
            remedy="check the between()/subgraph() endpoints; emptiness of the "
            "slice itself is disclosure, but an empty REGION edit is vacuous",
        )
    graph = slice_._graph
    members = slice_._member_set
    trace = slice_.source_trace

    # Convexity: between(R, R) <= R, in the shipped address algebra.
    from ..selection_graph import _between_label_set

    closure = _between_label_set(graph, members, members)
    if not (closure <= members):
        path = _offending_reentry_path(graph, members, closure)
        raise RegionError(
            "the region is not convex: a dataflow path leaves it and "
            f"re-enters it ({path}); no single entries-to-exits function can "
            "exist for a re-entrant region",
            code="region_not_convex",
            remedy="grow the region to include the path (e.g. widen the "
            "between() endpoints), or split the edit into two regions",
        )

    by_call = _edge_records_by_call(trace)
    # A boundary crossing into a VALUE-UNCHANGED buffer sink is not an exit:
    # no value leaves (the write re-stores the same bytes; D18's eval-mode
    # BatchNorm evidence class). Value-changing / unproven buffer sinks stay
    # in the exit set so the do()-time effect-closure delegation refuses
    # them by name.
    out_edges = []
    for parent, child in slice_.boundary_out_edges:
        child_op = graph.ops[child]
        if (
            getattr(child_op, "layer_type", None) == "buffer"
            and getattr(child_op, "buffer_value_changed", None) is False
        ):
            continue
        out_edges.append((parent, child))
    exit_edges = tuple(
        _resolve_boundary_edge("exit", parent, child, graph, by_call) for parent, child in out_edges
    )
    entry_edges = tuple(
        _resolve_boundary_edge("entry", parent, child, graph, by_call)
        for parent, child in slice_.boundary_in_edges
    )

    if exits is not None:
        _validate_declared_exits(exit_edges, exits)

    instance_members = _connected_instances(slice_.labels, slice_.edges)
    instances = []
    for index, component in enumerate(instance_members, start=1):
        component_set = set(component)
        component_exits = tuple(edge for edge in exit_edges if edge.parent in component_set)
        component_entries = tuple(edge for edge in entry_edges if edge.child in component_set)
        exit_ops = tuple(dict.fromkeys(edge.parent for edge in component_exits))
        instances.append(
            RegionInstance(
                index=index,
                members=component,
                exit_ops=exit_ops,
                exits=component_exits,
                entries=component_entries,
            )
        )
    boundary = RegionBoundary(entries=entry_edges, exits=exit_edges)
    return RegionTarget(slice_, boundary, tuple(instances))


# ---------------------------------------------------------------------------
# effect closure (replay lane only, by delegation)
# ---------------------------------------------------------------------------


def _check_effect_closure(region_target: RegionTarget) -> dict[str, Any]:
    """Delegate the effect-closure check to recorded capture evidence.

    Consults, per member op: buffer write kinds + value-change evidence,
    collective-boundary annotations, and the in-place-op convention. A known
    outward effect refuses naming the channel and op; the returned disclosure
    carries the certified channels and the honest uncertified note.

    Raises
    ------
    RegionError
        ``region_effect_closure_violated``.
    """

    graph = region_target.source_slice._graph
    members = region_target.source_slice._member_set

    def _refuse_buffer(label: str, write_kind: Any, changed: Any) -> None:
        """Refuse one outward buffer write, citing the recorded evidence."""

        evidence = "value changed" if changed else "write evidence unproven"
        raise RegionError(
            f"region member {label!r} writes a buffer (write kind "
            f"{write_kind!r}, {evidence}); a region with an outward buffer "
            "effect cannot honestly be replayed away by substituting its "
            "exit tensors (channel: buffer_writes)",
            code="region_effect_closure_violated",
            remedy="exclude the buffer-writing op from the region (or run in "
            "eval mode so the write evidence proves value-unchanged), or run "
            "the edit on a live lane (intervene= / bind), where the changed "
            "side effects ARE the experiment",
        )

    # Buffer-write evidence rides the buffer SINK nodes: a region exit edge
    # into a value-changing (or unproven) buffer sink IS an outward buffer
    # effect of the writing member.
    for edge in region_target.boundary.exits:
        child_op = graph.ops[edge.child]
        if getattr(child_op, "layer_type", None) == "buffer":
            changed = getattr(child_op, "buffer_value_changed", None)
            if changed is not False:
                _refuse_buffer(edge.parent, getattr(child_op, "buffer_write_kind", None), changed)
    for label in region_target.members:
        op = graph.ops[label]
        write_kind = getattr(op, "buffer_write_kind", None)
        if write_kind is not None:
            changed = getattr(op, "buffer_value_changed", None)
            if changed is not False:
                _refuse_buffer(label, write_kind, changed)
        _check_member_side_channels(graph, members, label, op)
    return {
        "certified_channels": list(_CERTIFIED_CHANNELS),
        "note": _UNCERTIFIED_NOTE,
    }


def _check_member_side_channels(graph: Any, members: frozenset[str], label: str, op: Any) -> None:
    """Refuse one member's collective / visible in-place effect channels.

    Raises
    ------
    RegionError
        ``region_effect_closure_violated`` naming the channel and op.
    """

    annotations = getattr(op, "annotations", None) or {}
    if isinstance(annotations, dict) and annotations.get("collective") is not None:
        raise RegionError(
            f"region member {label!r} is a collective boundary op; a "
            "region that joins a collective cannot be replayed away "
            "(channel: collectives)",
            code="region_effect_closure_violated",
            remedy="exclude the collective op from the region",
        )
    func_name = getattr(op, "func_name", "") or ""
    if func_name.endswith("_") and not func_name.endswith("__"):
        for parent in graph.parents.get(label, frozenset()):
            external_consumers = [
                child for child in graph.children.get(parent, frozenset()) if child not in members
            ]
            if parent not in members or external_consumers:
                raise RegionError(
                    f"region member {label!r} mutates its input in place "
                    f"({func_name}) and the mutated value is visible "
                    "outside the region (channel: inplace_ops)",
                    code="region_effect_closure_violated",
                    remedy="use the out-of-place op inside regions, or "
                    "grow the region to contain every consumer of the "
                    "mutated value",
                )


# ---------------------------------------------------------------------------
# the replay lowering: coordinated atomic exit substitution
# ---------------------------------------------------------------------------


def _replace_at_path(container: Any, path: tuple[Any, ...], value: Any) -> Any:
    """Rebuild ``container`` with ``value`` at the nested ``path``.

    The container-consumed exit EXTENSION (memo 3.4): tuple/list/dict
    positions extend at any depth; an unsupported container kind refuses
    typed rather than guessing.
    """

    if not path:
        return value
    key, rest = path[0], path[1:]
    if isinstance(container, tuple):
        index = int(key)
        return (
            container[:index]
            + (_replace_at_path(container[index], rest, value),)
            + container[index + 1 :]
        )
    if isinstance(container, list):
        index = int(key)
        rebuilt = list(container)
        rebuilt[index] = _replace_at_path(container[index], rest, value)
        return rebuilt
    if isinstance(container, dict):
        rebuilt_dict = dict(container)
        rebuilt_dict[key] = _replace_at_path(container[key], rest, value)
        return rebuilt_dict
    raise RegionError(
        f"a region exit is consumed inside a {type(container).__name__} "
        "container position the splice cannot rebuild",
        code="region_exit_container_unsupported",
        remedy="reshape the consumer to take the tensor as a plain argument, "
        "or exclude this consumer from the region's exits by growing the region",
    )


def _splice_occurrence(
    args: tuple[Any, ...], kwargs: dict[str, Any], address: tuple[Any, ...], value: Any
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Splice one substituted value at one occurrence address (nested OK)."""

    _child_call_id, arg_kind, arg_path = address
    if arg_kind == "positional":
        position = int(arg_path[0])
        spliced = _replace_at_path(args[position], tuple(arg_path[1:]), value)
        return args[:position] + (spliced,) + args[position + 1 :], kwargs
    kwargs = dict(kwargs)
    kwargs[arg_path[0]] = _replace_at_path(kwargs[arg_path[0]], tuple(arg_path[1:]), value)
    return args, kwargs


def resplice_region_entry(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    arg_kind: Any,
    arg_path: Any,
    value: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Re-splice one region tier-(ii) entry at cone recomputation.

    Region exit substitutions re-splice like param entries (a later push
    must never silently revert the region edit), nested container paths
    included; called by the replay engine's reconstruction seam.
    """

    return _splice_occurrence(args, kwargs, (None, arg_kind, tuple(arg_path)), value)


def _entry_values(trace: Any, graph: Any, instance: RegionInstance) -> list[torch.Tensor]:
    """Collect the ordered values flowing INTO one region instance."""

    from .edge_substitution import _consumed_value

    values = []
    for edge in instance.entries:
        parent_op = graph.ops[edge.parent]
        child_op = graph.ops[edge.child]
        values.append(_consumed_value(trace, parent_op, child_op))
    return values


def _evaluate_instance(
    trace: Any,
    graph: Any,
    region_target: RegionTarget,
    instance: RegionInstance,
    edit: Any,
) -> dict[str, torch.Tensor]:
    """Evaluate the replacement ONCE for one instance; map exit op -> value.

    Single-exit instances accept any ordinary value edit (helper spec,
    callable, replacement tensor). Multi-exit instances need a coordinated
    replacement: ``splice_module`` (entries in, exits out) or a callable
    taking the ordered exit-value tuple and returning a same-length tuple.
    """

    from .edge_substitution import _consumed_value, _edit_hook
    from .hooks import make_hook_context
    from .runtime import validate_hook_output

    exit_values: dict[str, torch.Tensor] = {}
    for label in instance.exit_ops:
        edge = next(e for e in instance.exits if e.parent == label)
        exit_values[label] = _consumed_value(trace, graph.ops[edge.parent], graph.ops[edge.child])

    is_splice = isinstance(edit, HelperSpec) and edit.helper_name == "splice_module"
    if is_splice and dict(edit.metadata).get("input") == "in":
        module = edit.args[0]
        entries = _entry_values(trace, graph, instance)
        if not entries:
            raise RegionError(
                f"splice_module(input='in') on region instance {instance.index} "
                "has no derivable entry values",
                code="region_splice_arity_mismatch",
                remedy="use splice_module(input='out') to transform the exit value instead",
            )
        from .._state import pause_logging

        with pause_logging():
            result = module(*entries)
        results = list(result) if isinstance(result, (tuple, list)) else [result]
        if len(results) != len(instance.exit_ops):
            raise RegionError(
                f"splice_module returned {len(results)} value(s) for region "
                f"instance {instance.index} with {len(instance.exit_ops)} exit "
                f"op(s) ({', '.join(map(repr, instance.exit_ops))})",
                code="region_splice_arity_mismatch",
                remedy="return exactly one tensor per exit op, in exit order",
            )
        splice_context = make_hook_context(
            name="splice_module",
            timing="post",
            direction="forward",
            layer_log=graph.ops[instance.exit_ops[0]],
            run_ctx={"region": region_target.region_digest, "instance": instance.index},
        )
        return _validated_exit_values(
            instance, exit_values, results, validate_hook_output, splice_context
        )

    if len(instance.exit_ops) == 1:
        label = instance.exit_ops[0]
        hook_callable, _helper, helper_name = _edit_hook(edit, label)
        context = make_hook_context(
            name=helper_name,
            timing="post",
            direction="forward",
            layer_log=graph.ops[label],
            run_ctx={"region": region_target.region_digest, "instance": instance.index},
        )
        replaced = hook_callable(exit_values[label], hook=context)
        # The SAME payload gate every other edit door runs (node-level do,
        # intervene=, spec.bind): a region exit value feeds the recorded
        # consumers, so a shape/dtype/device change is a downstream lie, not
        # an edit (AUD-CODE 3.7c).
        return {label: validate_hook_output(replaced, exit_values[label], hook_context=context)}

    if isinstance(edit, (HelperSpec, torch.Tensor)) or not callable(edit):
        raise RegionError(
            f"region instance {instance.index} has {len(instance.exit_ops)} "
            "exit ops; a per-value edit cannot coordinate a multi-exit "
            "replacement (one evaluation per pass, all exits atomic)",
            code="region_edit_multi_exit_unsupported",
            remedy="pass a region callable fn(exit_values_tuple, *, hook) "
            "returning one tensor per exit op, or tl.splice_module(module, "
            "input='in') mapping entries to exits",
        )
    context = make_hook_context(
        name=getattr(edit, "__name__", "region_edit"),
        timing="post",
        direction="forward",
        layer_log=graph.ops[instance.exit_ops[0]],
        run_ctx={"region": region_target.region_digest, "instance": instance.index},
    )
    ordered = tuple(exit_values[label] for label in instance.exit_ops)
    result = edit(ordered, hook=context)
    if not isinstance(result, (tuple, list)) or len(result) != len(instance.exit_ops):
        raise RegionError(
            "a multi-exit region callable must return one tensor per exit op "
            f"(expected {len(instance.exit_ops)}, got "
            f"{type(result).__name__ if not isinstance(result, (tuple, list)) else len(result)})",
            code="region_splice_arity_mismatch",
            remedy="return a tuple of len(region.instances[i].exit_ops) tensors",
        )
    return _validated_exit_values(
        instance, exit_values, list(result), validate_hook_output, context
    )


def _validated_exit_values(
    instance: RegionInstance,
    exit_values: dict[str, torch.Tensor],
    results: list[Any],
    validate: Any,
    context: Any,
) -> dict[str, torch.Tensor]:
    """Run every replacement through the hook payload gate against its exit value."""

    return {
        label: validate(value, exit_values[label], hook_context=context)
        for label, value in zip(instance.exit_ops, results, strict=True)
    }


def _refuse_inadmissible_region_do(
    trace: Any, region_target: RegionTarget, edit: Any, *, engine: str
) -> None:
    """Refuse do(region, edit) operand/engine/binding/address violations typed."""

    if edit is None:
        raise RegionError(
            "do(region, edit) requires an edit: pass an edit helper, a hook "
            "callable, a replacement tensor, or a region callable",
            code="region_edit_invalid",
            remedy="pass the replacement as the second do() argument",
        )
    if engine != "replay":
        raise RegionError(
            "region edits ship on the replay/push engine only: the lowering "
            "substitutes exit values without re-running the interior, which "
            f"engine={engine!r} cannot express (live lanes run the model for "
            "real -- use intervene= or spec.bind there)",
            code="region_engine_unsupported",
            remedy="call do(region, edit) with engine='replay' (the default on a fork)",
        )
    if region_target.trace is not trace:
        raise RegionError(
            "this region was derived on a different trace; regions bind to "
            "the trace whose slice derived them",
            code="region_trace_mismatch",
            remedy="derive the region on the trace you edit: region(fork.between(sources, sinks))",
        )
    from .edge_substitution import _require_edge_provenance

    _require_edge_provenance(trace)


def _refuse_unaddressable_exits(region_target: RegionTarget) -> None:
    """Refuse exit edges the pass-exact edge family cannot address.

    Runs AFTER the effect-closure delegation: an outward-effect member is
    the more fundamental violation and is named first.
    """

    unaddressable = [edge for edge in region_target.boundary.exits if not edge.addresses]
    graph_ops = region_target.source_slice._graph.ops
    into_output = [
        edge.child
        for edge in unaddressable
        if bool(getattr(graph_ops.get(edge.child), "is_output", False))
    ]
    if into_output:
        names = ", ".join(repr(label) for label in into_output)
        raise RegionError(
            f"region exit edge(s) into the model-output alias node(s) [{names}] cannot "
            "be substituted: an output alias runs no function, so the exit "
            "substitution has no argument occurrence to splice, and leaving it "
            "would return the unedited value from the model",
            code="region_exit_address_underivable",
            remedy="end the region before the op whose value the model returns, "
            "or edit that op directly with fork.do(tl.module(...), edit)",
        )
    if unaddressable:
        names = ", ".join(f"{e.parent!r} -> {e.child!r}" for e in unaddressable)
        raise RegionError(
            f"region exit edge(s) [{names}] have no derivable occurrence "
            "address in the dataflow edge family; the coordinated exit "
            "substitution cannot address these crossings",
            code="region_exit_address_underivable",
            remedy="re-capture with CaptureOptions(intervention_ready=True) so "
            "argument provenance is recorded, or reshape the region so its "
            "exits are ordinary tensor arguments",
        )


def _evaluate_all_instances(
    trace: Any, graph: Any, region_target: RegionTarget, edit: Any
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Phase A evaluation: ONE edit evaluation per instance, tensors enforced."""

    replacements: dict[str, torch.Tensor] = {}
    per_instance: list[dict[str, Any]] = []
    for instance in region_target.instances:
        values = _evaluate_instance(trace, graph, region_target, instance, edit)
        for label, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise RegionError(
                    f"the region replacement for exit op {label!r} is a "
                    f"{type(value).__name__}, not a tensor",
                    code="region_edit_invalid",
                    remedy="return tensors from the region edit",
                )
        replacements.update(values)
        per_instance.append(
            {
                "instance": instance.index,
                "members": len(instance.members),
                "exit_ops": list(instance.exit_ops),
                "evaluations": 1,
            }
        )
    return replacements, per_instance


def _commit_region_occurrence(
    region_target: RegionTarget,
    graph: Any,
    child_op: Any,
    occurrence: tuple[tuple[Any, ...], torch.Tensor, RegionEdge],
    edit: Any,
) -> tuple[Any, dict[str, Any], tuple[Any, ...]]:
    """Phase B, one occurrence: tier-(ii) store + stamp + fire disclosure."""

    from .audit import build_fire_record
    from .edge_substitution import _value_digest

    address, value, edge = occurrence
    _call_id, arg_kind, arg_path = address
    store_key = (arg_kind, tuple(arg_path))
    store = dict(getattr(child_op, "edge_substitutions", None) or {})
    store[store_key] = {
        "value": value.detach().clone(),
        "parent_label": graph.ops[edge.parent].layer_label,
        "resolve_digest": region_target.region_digest,
        "helper_name": "region",
        "substitution_kind": "region",
    }
    stamps = dict(getattr(child_op, "edge_replacement_stamps", None) or {})
    stamps[store_key] = {
        "verdict": True,
        "value_digest": _value_digest(value),
        "resolve_digest": region_target.region_digest,
    }
    child_op._internal_set("edge_substitutions", store)
    child_op._internal_set("edge_replacement_stamps", stamps)
    fire_record = build_fire_record(
        target_label=child_op.layer_label,
        call_label=child_op.label,
        func_call_id=child_op.func_call_id,
        container_path=tuple(child_op.container_path or ()),
        engine="replay",
        helper=None,
        site_label=child_op.layer_label,
        timing="post",
        direction="forward",
        helper_name=(
            edit.helper_name
            if isinstance(edit, HelperSpec)
            else getattr(edit, "__name__", "region_edit")
        ),
        replaced=False,
        edge_address=tuple(address),
    )
    disclosure = {
        "edge_address": repr(tuple(address)),
        "parent": edge.parent,
        "child": edge.child,
        "value_digest": stamps[store_key]["value_digest"],
    }
    return fire_record, disclosure, store_key


def apply_region_do(
    trace: Any,
    region_target: RegionTarget,
    edit: Any,
    *,
    engine: str,
    strict: bool,
) -> str:
    """Apply one region edit on the replay engine (exit substitution).

    ONE transaction: every instance evaluates its replacement once, every
    exit occurrence is spliced and its consumer re-executed (phase A,
    nothing committed), then all tier-(ii) entries + fire records + replay
    updates commit and downstream cones push (phase B, rolled back on
    failure). The canonical REGION audit row (the renderer-neutral fact
    F43 reads) carries the engine-set ``execution_effect``.

    Returns the do() mutation kind (``"region_replayed"``).
    """

    _refuse_inadmissible_region_do(trace, region_target, edit, engine=engine)
    effect_disclosure = _check_effect_closure(region_target)
    _refuse_unaddressable_exits(region_target)
    graph = region_target.source_slice._graph

    import importlib

    replay_module = importlib.import_module("torchlens.intervention.replay")

    # ---- phase A: evaluate + re-execute consumers; commit NOTHING --------
    replacements, per_instance = _evaluate_all_instances(trace, graph, region_target, edit)

    # group every exit occurrence by consuming child op
    by_child: dict[str, list[tuple[tuple[Any, ...], torch.Tensor, RegionEdge]]] = {}
    for edge in region_target.boundary.exits:
        for address in edge.addresses:
            by_child.setdefault(edge.child, []).append((address, replacements[edge.parent], edge))

    new_outs: dict[str, Any] = {}
    for child_label, occurrences in by_child.items():
        child_op = graph.ops[child_label]
        template = replay_module._template_for_site(child_op)
        args, kwargs = replay_module._reconstruct_args_from_template(
            template, child_op, trace, {}, strict=strict
        )
        args, kwargs = replay_module._splice_param_substitutions([child_op], args, kwargs)
        for address, value, _edge in occurrences:
            args, kwargs = _splice_occurrence(args, kwargs, address, value)
        output = replay_module._execute_replay_func_strict(child_op, args, kwargs)
        new_outs[child_label] = replay_module._slice_output_by_path(
            output, replay_module._replay_container_path(child_op, trace)
        )

    # ---- phase B: commit atomically, then push -----------------------------
    committed: list[tuple[Any, tuple[Any, ...]]] = []
    applied: list[dict[str, Any]] = []
    try:
        for child_label, occurrences in by_child.items():
            child_op = graph.ops[child_label]
            fire_records = []
            for occurrence in occurrences:
                fire_record, disclosure, store_key = _commit_region_occurrence(
                    region_target, graph, child_op, occurrence, edit
                )
                committed.append((child_op, store_key))
                fire_records.append(fire_record)
                applied.append(disclosure)
            replay_module._commit_replay_updates(
                trace,
                {child_op.layer_label: new_outs[child_label]},
                {child_op.layer_label: fire_records},
            )
        execution_order = {label: index for index, label in enumerate(graph.order)}
        for child_label in sorted(by_child, key=execution_order.__getitem__):
            replay_module.push_from(trace, graph.ops[child_label])
    except BaseException:
        for child_op, store_key in committed:
            store = dict(getattr(child_op, "edge_substitutions", None) or {})
            stamps = dict(getattr(child_op, "edge_replacement_stamps", None) or {})
            store.pop(store_key, None)
            stamps.pop(store_key, None)
            child_op._internal_set("edge_substitutions", store or None)
            child_op._internal_set("edge_replacement_stamps", stamps or None)
        raise

    # The renderer-neutral region fact row (F43 reads, never writes): it
    # rides the persisted FREE-FORM state_history stream, not the closed
    # intervention_audit grammar -- a new audit-row KIND is a C07-owned
    # schema amendment, and the do() transaction envelope (EVENT row)
    # already lands in intervention_audit through the one chokepoint.
    trace._record_operation(
        "region_do",
        region_digest=region_target.region_digest,
        members=list(region_target.members),
        entries=[
            {"parent": e.parent, "child": e.child, "nested": e.nested}
            for e in region_target.boundary.entries
        ],
        exits=applied,
        instances=per_instance,
        edit=(
            edit.helper_name if isinstance(edit, HelperSpec) else getattr(edit, "__name__", "value")
        ),
        execution_effect=_REGION_EXECUTION_EFFECT,
        effect_closure=effect_disclosure,
        source="current_transaction",
    )
    return "region_replayed"


__all__ = [
    "RegionBoundary",
    "RegionEdge",
    "RegionInstance",
    "RegionTarget",
    "apply_region_do",
    "region",
]
