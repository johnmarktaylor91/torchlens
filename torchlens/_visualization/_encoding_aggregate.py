"""Exact aggregates for encoding channels on collapsed module boxes and segments.

A collapsed module box (``collapse="auto"``, ``depth=N``) or a condensed
segment capsule stands for a set of member ops. ``color_by`` / ``size_by``
encode such a node only when the source has an EXACT aggregate over those
ops: the sum of an additive per-op quantity (time, activation bytes, FLOPs,
joined device time). Every other source leaves the node unencoded and says
so in the legend; a legend never implies a value the members cannot prove.

Leaf module by design: it imports nothing from the visualization package at
runtime, so it adds no edge to the package's import cycle. Callers pass the
trace, the state, and any resolver they own.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

#: Key prefix separating aggregate-node keys from op/layer keys in the
#: shared per-draw encoding maps (an op label can never start with it).
AGGREGATE_KEY_PREFIX = "aggregate::"

#: Builtin tokens with an additive per-op field behind them.
BUILTIN_ADDITIVE_FIELDS = {
    "time": "func_duration",
    "bytes": "activation_memory",
    "flops": "flops_forward",
}

#: Per-op fields whose total over a set of ops is the exact sum of the
#: member values (durations, byte counts, operation counts).
ADDITIVE_OP_FIELDS = frozenset(
    {
        "func_duration",
        "activation_memory",
        "transformed_activation_memory",
        "flops_forward",
        "flops_backward",
        "flops_total",
        "macs_forward",
    }
)

#: Fields where an op that ran no function (an input, buffer or other
#: source tensor) contributes exactly zero rather than a missing value.
_ZERO_WITHOUT_FUNCTION = frozenset(
    {"func_duration", "flops_forward", "flops_backward", "flops_total", "macs_forward"}
)

#: The joined device-time builtin; summed from the join's per-op nanoseconds.
DEVICE_TIME_TOKEN = "device_time"

LINE_SUM = "module boxes: sum over member ops"
LINE_SUM_ALL_CALLS = "module boxes: sum over member ops across all calls"
LINE_SEGMENT_SUM = "segments: sum over member ops"
NOTE_FOLD_REPRESENTATIVE = "repeat-fold boxes: representative module only"
NOTE_INCOMPLETE = "aggregate nodes with a member op lacking a value -- unencoded"


@dataclass(frozen=True)
class AggregateTarget:
    """One rendered node that stands for a set of member ops.

    Parameters
    ----------
    name:
        Graphviz node name of the box or segment.
    kind:
        ``"module_box"`` or ``"segment"``.
    op_labels:
        Pass-qualified labels of the member ops the node represents.
    label_lines:
        Label rows when known before spec resolution (segments), else ``()``.
    all_calls:
        Whether the scope spans every call of a module (rolled boxes).
    fold_representative:
        Whether the box is a repeat-fold representative.
    """

    name: str
    kind: str
    op_labels: tuple[str, ...]
    label_lines: tuple[str, ...] = ()
    all_calls: bool = False
    fold_representative: bool = False

    @property
    def key(self) -> str:
        """Return the encoding-map key for this node."""

        return aggregate_key(self.name)


@dataclass(frozen=True)
class AggregateTargets:
    """Aggregate nodes of one draw plus the units they hide.

    Parameters
    ----------
    by_unit:
        Target per visible universe unit id (module boxes).
    segments:
        Segment targets in first-seen order.
    hidden_unit_ids:
        Unit ids whose ops render inside a segment rather than on their own.
    """

    by_unit: dict[str, AggregateTarget]
    segments: tuple[AggregateTarget, ...] = ()
    hidden_unit_ids: frozenset[str] = frozenset()


def aggregate_key(name: str) -> str:
    """Return the encoding-map key for an aggregate node name."""

    return f"{AGGREGATE_KEY_PREFIX}{name}"


def aggregate_field(spec: Any) -> str | None:
    """Return the additive per-op field behind ``spec``, or ``None``.

    ``None`` means the source has no exact sum over ops (callables, payload
    builtins such as magnitude, ``dims``, and non-additive fields).
    """

    if spec.source_kind == "builtin":
        if spec.source == DEVICE_TIME_TOKEN:
            return DEVICE_TIME_TOKEN
        return BUILTIN_ADDITIVE_FIELDS.get(spec.source)
    if spec.source_kind == "field" and spec.source in ADDITIVE_OP_FIELDS:
        return str(spec.source)
    return None


def unaggregatable_note(spec: Any) -> str:
    """Return the legend note for a source with no exact aggregate."""

    return f"collapsed nodes unencoded: {spec.display_name} has no exact sum over ops"


def _op_value(op: Any, field_name: str) -> float | None:
    """Return one member op's additive value, or ``None`` when it is missing."""

    value = getattr(op, field_name, None)
    if value is None:
        if field_name in _ZERO_WITHOUT_FUNCTION and getattr(op, "func", None) is None:
            return 0.0
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    as_float = float(value)
    return as_float if math.isfinite(as_float) else None


def sum_member_ops(trace: Any, op_labels: tuple[str, ...], field_name: str) -> float | None:
    """Return the exact sum of ``field_name`` over the member ops.

    Returns ``None`` when the scope is empty or any member lacks a value: a
    partial sum is a lower bound, not the node's value.
    """

    if not op_labels:
        return None
    total = 0.0
    for label in op_labels:
        try:
            op = trace.ops[label]
        except (KeyError, IndexError):
            return None
        value = _op_value(op, field_name)
        if value is None:
            return None
        total += value
    return total


def sum_device_ns(op_labels: tuple[str, ...], device_ns: Callable[[str], Any]) -> float | None:
    """Sum joined device nanoseconds over member ops (the per-node rule).

    Ops the join attributed no kernels to contribute nothing, exactly as on
    an op node; a scope with no attributed kernel at all reads n/a.
    """

    found = [value for value in (device_ns(str(label)) for label in op_labels) if value is not None]
    return float(sum(int(value) for value in found)) if found else None


def target_value(
    trace: Any,
    target: AggregateTarget,
    spec: Any,
    record: Callable[[str, bool], None],
    device_ns: Callable[[str], Any] | None = None,
) -> float | None:
    """Resolve one channel value for an aggregate node.

    ``record(text, is_aggregation_line)`` files the legend disclosure on the
    caller's channel (aggregation lines and notes are kept apart).
    """

    field_name = aggregate_field(spec)
    if field_name is None:
        record(unaggregatable_note(spec), False)
        return None
    if field_name == DEVICE_TIME_TOKEN:
        value = None if device_ns is None else sum_device_ns(target.op_labels, device_ns)
    else:
        value = sum_member_ops(trace, target.op_labels, field_name)
    if value is None:
        record(NOTE_INCOMPLETE, False)
        return None
    if target.kind == "segment":
        record(LINE_SEGMENT_SUM, True)
    else:
        record(LINE_SUM_ALL_CALLS if target.all_calls else LINE_SUM, True)
    if target.fold_representative:
        record(NOTE_FOLD_REPRESENTATIVE, False)
    return value


def segment_encoding_attrs(state: Any, name: str) -> dict[str, str]:
    """Return Graphviz fill and size attrs for one encoded segment node.

    Segment capsules are emitted as raw node args (no NodeSpec funnel), so
    this is their one channel writer. Size is a minimum (fixedsize=false):
    the capsule can only grow from its label.
    """

    if state is None:
        return {}
    key = aggregate_key(name)
    attrs: dict[str, str] = {}
    fill = state.colors.get(key)
    if fill is not None:
        attrs["fillcolor"] = fill
    size = state.final_sizes.get(key) or state.sizes.get(key)
    if size is not None:
        attrs.update({"width": str(size[0]), "height": str(size[1]), "fixedsize": "false"})
    return attrs


def box_channel_spec(state: Any, name: str, spec: Any) -> Any:
    """Apply an encoded module box's fill and size minimums to its NodeSpec.

    Runs in the encoding slot (after the mode preset, before the user callback,
    which still sees and may override them). Unencoded boxes pass through.
    """

    if state is None:
        return spec
    key = aggregate_key(name)
    fill = state.colors.get(key)
    if fill is not None:
        spec = spec.replace(fillcolor=fill)
    size = state.sizes.get(key)
    if size is not None:
        spec = spec.replace(width=size[0], height=size[1], fixedsize="false")
    return spec


@dataclass(frozen=True)
class BoxScopeHelpers:
    """Renderer predicates the box-scope rule needs, injected by the renderer.

    Injection keeps this module a leaf of the import graph.
    """

    run_fold_for_address: Callable[..., Any]
    should_show_remainder: Callable[..., bool]
    surfaced_own_output_ops: Callable[..., Any]
    segment_for_node: Callable[..., Any]


def collapsed_box_scope(
    trace: Any,
    node: Any,
    mode_depth: tuple[str, int],
    collapse: tuple[str | None, Any, Any],
    helpers: BoxScopeHelpers,
) -> AggregateTarget:
    """Return the box name and the pass-qualified ops a collapsed box stands for.

    THE one scope rule, shared by the box builder (mode-preset rows) and the
    encoding prepass (channel aggregates): one call's ops for an unrolled box,
    every call's ops for a rolled box, minus ops surfaced as their own nodes
    (keeping them would double-count them). A fold representative's scope is
    the representative only. ``mode_depth`` is ``(vis_mode, vis_call_depth)``;
    ``collapse`` is ``(collapse_address, repeat_folds, collapse_fn)``.
    """

    vis_mode, vis_call_depth = mode_depth
    collapse_address, repeat_folds, collapse_fn = collapse
    address_w_pass = (
        collapse_address if collapse_address is not None else node.modules[vis_call_depth - 1]
    )
    address, call_index = address_w_pass.rsplit(":", 1)
    fold = helpers.run_fold_for_address(address, repeat_folds)
    if fold is not None:
        address = fold.representative
    module = trace.modules[address]
    rolled = vis_mode != "unrolled"
    calls = tuple(module.ops.values()) if rolled else (module.ops[int(call_index) - 1],)
    member_labels = tuple(module.layer_labels) if rolled else tuple(calls[0].ops)
    labels = tuple(op_label for call in calls for op_label in call.ops)
    if fold is None and helpers.should_show_remainder(
        trace,
        address,
        member_labels,
        collapse_fn,
        vis_mode=vis_mode,
        max_module_depth=vis_call_depth,
    ):
        # Rolled: the surfaced-exit predicate is per layer, so every pass of
        # a surfaced layer renders outside the box; drop the layer base.
        def base(label: str) -> str:
            return label.rsplit(":", 1)[0] if rolled else label

        surfaced = {
            base(str(op.layer_label))
            for op in helpers.surfaced_own_output_ops(trace, address, member_labels)
        }
        labels = tuple(label for label in labels if base(label) not in surfaced)
    return AggregateTarget(
        name=address if rolled else f"{address}pass{call_index}",
        kind="module_box",
        op_labels=labels,
        all_calls=rolled,
        fold_representative=fold is not None,
    )


def encoding_aggregate_targets(
    trace: Any, universe: Any, scope: tuple[Any, ...]
) -> AggregateTargets:
    """Return the draw's module boxes and segments for the encoding prepass.

    ``scope`` is ``(context, repeat_folds, segment_lookup, helpers)``. Units
    whose node renders inside a segment are hidden; the segment is collected
    once.
    """

    context, repeat_folds, segment_lookup, helpers = scope
    by_unit: dict[str, AggregateTarget] = {}
    segments: dict[str, AggregateTarget] = {}
    hidden: set[str] = set()
    for unit in universe.units:
        emission = unit.emission
        if emission.node is None or emission.kind not in ("raw_op", "module_box"):
            continue
        segment = helpers.segment_for_node(emission.node, segment_lookup)
        if segment is not None:
            hidden.add(unit.unit_id)
            segments.setdefault(
                segment.name,
                AggregateTarget(segment.name, "segment", tuple(segment.ops), (segment.label,)),
            )
        elif emission.kind == "module_box":
            by_unit[unit.unit_id] = collapsed_box_scope(
                trace,
                emission.node,
                (context.vis_mode, context.vis_call_depth),
                (emission.call, repeat_folds, context.collapse_fn),
                helpers,
            )
    return AggregateTargets(by_unit, tuple(segments.values()), frozenset(hidden))


def collect_aggregate(
    state: Any,
    trace: Any,
    target: AggregateTarget,
    raw_values: dict[str, float],
    join_result: Callable[[Any, str], Any],
) -> None:
    """Resolve both channels on one box or segment (exact member sums only).

    ``join_result(trace, argument)`` returns the Kineto join for the
    device-time builtin (refusing typed when the trace has none).
    """

    state.eligible_count += 1
    for spec, values, lines, record_note in (
        (state.spec, raw_values, state.aggregation_lines, state.note),
        (state.size_spec, state.size_values, state.size_aggregation_lines, state.size_note),
    ):
        if spec is None:
            continue
        device_ns = None
        if spec.source == DEVICE_TIME_TOKEN and spec.source_kind == "builtin":
            argument = "size_by" if spec is state.size_spec else "color_by"
            device_ns = join_result(trace, argument).op_device_ns.get

        def record(
            text: str, is_line: bool, lines: list[str] = lines, note: Any = record_note
        ) -> None:
            """File one aggregate disclosure on this channel's legend."""

            if not is_line:
                note(text)
            elif text not in lines:
                lines.append(text)

        value = target_value(trace, target, spec, record, device_ns)
        if value is not None:
            values[target.key] = value
