"""ModuleCall tree display and call-scope resolution helpers.

Split out of ``module.py`` under the R43 file-size ratchet (T74b bounce):
the ``show_call_tree`` ASCII printer with its display/visibility helpers,
plus the call-scope helpers that resolve a ModuleCall's op-slot labels and
count its scoped graph edges. ``module.py`` consumes
``_edge_counts_for_scope``, ``_print_call_tree_from_root``, and
``_resolve_call_ops``; the display-only helpers are internal to the printer.
"""

from typing import TYPE_CHECKING, TextIO

from .._errors import RecordBindingError

if TYPE_CHECKING:
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
        return call.call_label
    return call.call_label.rsplit(":", 1)[0]


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
        getattr(op, "is_atomic_module", False)
        for op_label in call.output_ops
        for op in _resolve_call_ops(trace, op_label)
    )


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
