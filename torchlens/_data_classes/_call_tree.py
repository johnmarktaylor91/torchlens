"""ModuleCall tree display and call-scope resolution helpers.

Split out of ``module.py`` under the R43 file-size ratchet (T74b bounce):
the ``show_call_tree`` ASCII printer with its display/visibility helpers,
plus the call-scope helpers that resolve a ModuleCall's op-slot labels and
count its scoped graph edges. ``module.py`` consumes
``_edge_counts_for_scope``, ``_print_call_tree_from_root``,
``_resolve_call_ops``, ``_sum_call_ops_field``, ``_live_call_trace`` and
``_bound_call_trace`` (``_conditional_arms`` also reads ``_live_call_trace``);
the display-only helpers are internal to the printer.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, TextIO, cast

from .._errors import RecordBindingError
from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .module import ModuleCall
    from .op import Op
    from .trace import Trace


def _display_call_label(call: "ModuleCall", show_call_index: bool) -> str:
    """Return the display label for a ModuleCall.

    Parameters
    ----------
    call:
        ModuleCall to display.
    show_call_index:
        Whether to include the ``:N`` call suffix.

    Returns
    -------
    str
        Display-ready call label.
    """

    if show_call_index:
        return call.label
    return call.label.rsplit(":", 1)[0]


def _module_call_summary(call: "ModuleCall") -> str:
    """Return a compact optional summary for a ModuleCall display row.

    Parameters
    ----------
    call:
        ModuleCall to summarize.

    Returns
    -------
    str
        Parenthesized summary, or an empty string.
    """

    duration = getattr(call, "forward_duration", None)
    if duration is None:
        return ""
    return f" (forward_duration: {duration})"


def _is_atomic_module_call(call: "ModuleCall") -> bool:
    """Return whether a ModuleCall represents an atomic module leaf.

    Parameters
    ----------
    call:
        ModuleCall to inspect.

    Returns
    -------
    bool
        ``True`` when one of the call's output Ops is marked atomic.
    """

    trace = call._source_trace
    if trace is None:
        return False
    return any(
        record_member(op, "is_atomic_module", False)
        for op_label in call.output_ops
        for op in _resolve_call_ops(trace, op_label)
    )


def _sum_call_ops_field(trace: "Trace | None", calls: "Iterable[ModuleCall]", field: str) -> int:
    """Sum a numeric Op field once per Op over the Ops the given calls ran.

    A Module called twice counts both passes of a Layer it runs, never only
    the Layer's representative pass; ``None`` values are skipped.
    """

    if trace is None:
        return 0
    seen: set[str] = set()
    total = 0
    for call in calls:
        for label in call.ops:
            for op in _resolve_call_ops(trace, label):
                value = record_member(op, field, None)
                if op.label not in seen and value is not None:
                    total += value
                seen.add(op.label)
    return total


def _resolve_call_ops(trace: "Trace", label: str) -> list["Op"]:
    """Resolve a ModuleCall op-slot label to its Op object(s).

    Submodule calls store pass-qualified Op labels, each resolving to exactly
    one Op (unchanged behavior). The root ``self:1`` call stores bare Layer
    labels (per the function-root-module invariant), and a recurrent
    (multi-pass) Layer label resolves to EVERY one of its pass Ops. Aggregate
    ModuleCall properties sum/iterate over the resolved Ops so the root
    aggregate correctly spans all passes without a caller ever hitting the
    ``AmbiguousOpLookupError`` that plain ``trace.ops[layer_label]`` raises.
    """

    return trace.ops.resolve_all(label)


def _edge_counts_for_scope(trace: "Trace | None", op_labels: list[str]) -> tuple[int, int, int]:
    """Count internal, input, and output graph edges for a scoped op set.

    Parameters
    ----------
    trace:
        Owning Trace used to resolve Op labels.
    op_labels:
        Pass-qualified labels for Ops in the scope.

    Returns
    -------
    tuple[int, int, int]
        ``(internal_edges, input_edges, output_edges)`` with distinct
        ``(parent, child)`` pairs.
    """

    if trace is None:
        return 0, 0, 0

    scope = {op.label for label in op_labels for op in _resolve_call_ops(trace, label)}
    internal_edges: set[tuple[str, str]] = set()
    input_edges: set[tuple[str, str]] = set()
    output_edges: set[tuple[str, str]] = set()

    for parent_label in scope:
        parent = trace.ops[parent_label]
        for child_label in parent.children:
            canonical_child_label = trace.ops[child_label].label
            edge = (parent_label, canonical_child_label)
            if canonical_child_label in scope:
                internal_edges.add(edge)
            else:
                output_edges.add(edge)
        for inbound_label in parent.parents:
            canonical_inbound_label = trace.ops[inbound_label].label
            if canonical_inbound_label not in scope:
                input_edges.add((canonical_inbound_label, parent_label))

    return len(internal_edges), len(input_edges), len(output_edges)


def _visible_call_children(call: "ModuleCall", include_atomic: bool) -> list["ModuleCall"]:
    """Return displayable child ModuleCalls for ``call``.

    Parameters
    ----------
    call:
        Parent ModuleCall.
    include_atomic:
        Whether to include atomic module leaves.

    Returns
    -------
    list[ModuleCall]
        Displayable children in call-tree order.
    """

    trace = call._source_trace
    if trace is None:
        raise RecordBindingError(
            "ModuleCall not bound to a Trace",
            code="record_not_bound",
            remedy="keep the owning Trace alive and read records through it",
        )
    children = [trace.module_calls[child_label] for child_label in call.call_children]
    if include_atomic:
        return children
    return [child for child in children if not _is_atomic_module_call(child)]


def _print_call_tree_from_root(
    call: "ModuleCall",
    *,
    max_depth: int | None,
    include_atomic: bool,
    show_call_index: bool,
    file: TextIO | None,
) -> None:
    """Print a ModuleCall subtree as an ASCII tree.

    Parameters
    ----------
    call:
        Root ModuleCall to print.
    max_depth:
        Maximum descendant depth to print, or ``None`` for no limit.
    include_atomic:
        Whether to include atomic module leaves.
    show_call_index:
        Whether to include the ``:N`` call suffix.
    file:
        Optional output stream.
    """

    def print_node(node: "ModuleCall", prefix: str, is_last: bool, depth: int) -> None:
        """Print one node and its descendants."""

        # ASCII rails: emitted text is ASCII-canonical (lovely bug 10 /
        # summary-memo string contract); unicode belongs to explicit display
        # boundaries only, which this printer does not verify.
        connector = "" if depth == 0 else ("`-- " if is_last else "|-- ")
        label = _display_call_label(node, show_call_index)
        print(f"{prefix}{connector}{label}{_module_call_summary(node)}", file=file)

        children = _visible_call_children(node, include_atomic)
        if max_depth is not None and depth >= max_depth:
            if children:
                child_prefix = prefix if depth == 0 else prefix + ("    " if is_last else "|   ")
                print(f"{child_prefix}`-- ...", file=file)
            return

        child_prefix = prefix if depth == 0 else prefix + ("    " if is_last else "|   ")
        for index, child in enumerate(children):
            print_node(child, child_prefix, index == len(children) - 1, depth + 1)

    print_node(call, "", True, 0)


def root_call(trace: "Trace") -> "ModuleCall":
    """Body of ``Trace._root_call``; the record keeps the thin spelling and docstring."""

    for call in trace.module_calls:
        if call.call_parent is None:
            return cast("ModuleCall", call)
    raise RuntimeError("Trace has no root ModuleCall")


def walk_calls(trace: "Trace") -> Iterator["ModuleCall"]:
    """Body of ``Trace.walk_calls``; the record keeps the thin spelling and docstring."""

    if not trace.module_calls:
        return
    yield from root_call(trace).walk_descendants(include_self=True)


def show_call_tree(
    trace: "Trace",
    max_depth: int | None = None,
    include_atomic: bool = True,
    show_call_index: bool = True,
    file: TextIO | None = None,
) -> None:
    """Body of ``Trace.show_call_tree``; the record keeps the thin spelling and docstring."""

    if not trace.module_calls:
        return
    root_call(trace).show_call_tree(
        max_depth=max_depth,
        include_atomic=include_atomic,
        show_call_index=show_call_index,
        file=file,
    )


def _live_call_trace(
    call: Any, *, slot: str = "_source_trace_ref", record: str = "ModuleCall"
) -> "Trace | None":
    """Return the ModuleCall's owning Trace, or ``None`` for a call never bound to one.

    A call holds its Trace weakly, so a dead reference means the user dropped the
    Trace: that refuses ``trace_reference_collected`` instead of reading as an
    empty result. ``ConditionalArm`` reads its own weak reference through here
    too (``slot`` names the instance-dict key holding the reference, ``record``
    the record named in the refusal), so both share one refusal site.
    """

    ref = call.__dict__.get(slot)
    if ref is None:
        return None
    trace = ref()
    if trace is None:
        raise RecordBindingError(
            f"the Trace this {record} was read from has been garbage-collected: a {record} holds its Trace weakly, so a chained read on a temporary (tl.trace(...)[key].out, t.fork().do(...)[key].out) loses it as soon as the record is taken",
            code="trace_reference_collected",
            remedy="keep the Trace (t = tl.trace(...)) or fork (f = t.fork().do(...)) in a variable while reading its records",
        )
    return cast("Trace", trace)


def _bound_call_trace(call: "ModuleCall") -> "Trace":
    """Return the ModuleCall's owning Trace; refuse when unbound or collected."""

    trace = _live_call_trace(call)
    if trace is None:
        raise RecordBindingError(
            "ModuleCall not bound to a Trace",
            code="record_not_bound",
            remedy="keep the owning Trace alive and read records through it",
        )
    return trace
