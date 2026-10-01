"""Segment descriptor construction and label derivation for collapse plans.

Extracted from ``collapse_optimizer.py`` (C05 fix cycle, R43 size ratchet):
the family that turns a legal segment run into its renderer-facing
``SegmentDescriptor`` -- injective Graphviz-safe addresses, owner keys,
spanned-module derivation, range text, and the honest ``(xN)`` labels.
Pure functions over ``Trace``/``RenderContext``/plan vocabulary; no
optimizer state enters here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .collapse_plan import RenderContext, SegmentDescriptor

if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from .auto_collapse import CollapseAnalysis


def _encode_segment_address(address: str) -> str:
    """Return an injective Graphviz-safe encoding of one module address.

    Escaping literal underscores before mapping dots keeps distinct addresses
    distinct: ``a.b0`` becomes ``a_b0`` while ``a_b0`` becomes ``a__b0``.
    """

    return address.replace("_", "__").replace(".", "_")


def _make_child_segment_descriptor(
    trace: Trace,
    context: RenderContext,
    addresses: tuple[str, ...],
    covered_ops: tuple[str, ...],
) -> SegmentDescriptor:
    """Build a renderer descriptor for one child segment."""

    if covered_ops:
        covered_entries = tuple(_trace_op_for_concrete_label(trace, label) for label in covered_ops)
        if context.vis_mode == "rolled":
            seen_bases: set[str] = set()
            num_layers = 0
            num_buffers = 0
            for label, entry in zip(covered_ops, covered_entries, strict=True):
                base = str(label).rsplit(":", 1)[0]
                if base in seen_bases:
                    continue
                seen_bases.add(base)
                num_layers += 1
                num_buffers += bool(entry.is_buffer)
        else:
            num_layers = len(covered_entries)
            num_buffers = sum(bool(entry.is_buffer) for entry in covered_entries)
    else:
        num_layers = sum(
            int(getattr(trace.modules[address], "num_layers", 0) or 0) for address in addresses
        )
        buffer_labels = {
            label
            for address in addresses
            for label in (getattr(trace.modules[address], "buffer_layers", ()) or ())
        }
        num_buffers = len(buffer_labels)
    num_ops = max(0, num_layers - num_buffers)
    num_params = sum(
        int(getattr(trace.modules[address], "num_params", 0) or 0) for address in addresses
    )
    owner = _segment_owner_key(trace, addresses, context.vis_mode)
    label = _child_segment_label(addresses, num_layers, num_buffers, num_params)
    name = (
        f"{_encode_segment_address(addresses[0])}__segment__"
        f"{_encode_segment_address(addresses[-1])}pass1"
    )
    return SegmentDescriptor(
        name=name,
        kind="child",
        label=label,
        members=addresses,
        ops=covered_ops,
        owner=owner,
        num_ops=num_ops,
        num_buffers=num_buffers,
        num_params=num_params,
    )


def _child_segment_covered_ops(
    analysis: CollapseAnalysis,
    addresses: tuple[str, ...],
) -> tuple[str, ...]:
    """Return concrete pass-qualified op labels covered by a child segment."""

    labels: list[str] = []
    for address in addresses:
        signal = analysis.signals.get(address)
        if signal is None:
            continue
        labels.extend(str(label) for label in signal.subtree_ops)
    return tuple(dict.fromkeys(labels))


def _make_op_segment_descriptor(
    trace: Trace,
    context: RenderContext,
    labels: tuple[str, ...],
    concrete: tuple[str, ...] | None = None,
) -> SegmentDescriptor:
    """Build a renderer descriptor for one operation segment.

    Parameters
    ----------
    trace:
        Trace being optimized.
    context:
        Rendering context.
    labels:
        Pass-free plan labels in segment order.
    concrete:
        Pass-qualified op labels matching ``labels``. Falls back to the plan
        labels when concrete attribution is unavailable.
    """

    resolved = tuple(concrete) if concrete else tuple(labels)
    owner = _op_segment_owner_key(trace, resolved, context.vis_mode)
    label = _op_segment_label(trace, labels, resolved)
    spanned = _op_segment_spanned_modules(trace, resolved, owner, context.vis_mode)
    if spanned:
        # r-b6 R19-5: a segment placed above its ops' module homes (top level
        # or a shared ancestor) must DISCLOSE the modules it spans — the box
        # otherwise silently strips module containment from every hidden op.
        shown = ", ".join(f"@{module}" for module in spanned[:3])
        if len(spanned) > 3:
            shown += f", +{len(spanned) - 3} more"
        label = f"{label} -- spans {shown}"
    name = f"{resolved[0].replace(':', 'pass')}__segment__{resolved[-1].replace(':', 'pass')}"
    return SegmentDescriptor(
        name=name,
        kind="op",
        label=label,
        ops=resolved,
        owner=owner,
        num_ops=len(resolved),
        num_params=0,
    )


def _effective_render_module_stack(op: Op) -> tuple[str, ...]:
    """Return the call-qualified module stack that clusters a visible op.

    Mirrors the renderer's ``_raw_render_node_owner_key``: an atomic module's
    own exit op stays visible while its innermost atomic box is dropped, so
    that op clusters under the atomic module's parent call.
    """

    modules = [str(module) for module in getattr(op, "modules", ()) or ()]
    if getattr(op, "is_atomic_module", False) and modules:
        modules = modules[:-1]
    return tuple(modules)


def _crosses_module_call_boundary(
    previous: tuple[str, ...],
    current: tuple[str, ...],
) -> bool:
    """Return whether two adjacent rendered stacks cross a module-CALL reuse boundary.

    A boundary is crossed exactly when some shared stack level holds two
    different CALLS of the same module address (``core:1`` -> ``core:2``).
    Levels holding genuinely different sibling modules are not boundaries:
    merging across siblings is honest because the segment then owns their
    exact call-qualified LCA.
    """

    for prev_entry, entry in zip(previous, current):
        if prev_entry == entry:
            continue
        if prev_entry.rsplit(":", 1)[0] == entry.rsplit(":", 1)[0]:
            return True
    return False


def _op_segment_owner_key(
    trace: Trace,
    labels: tuple[str, ...],
    vis_mode: str = "unrolled",
) -> str | None:
    """Return the lowest common rendered module cluster for operation labels.

    Unrolled clusters are per module CALL, so commonality is exact
    call-qualified stack equality: a segment whose ops span several calls of
    one reused module has no common call-level entry and owns the honest LCA
    (the shared parent call, or ``None`` for top level) instead of falsely
    claiming the first call (round-24 C1). Rolled clusters merge passes per
    address, so rolled commonality stays pass-free AND the returned owner
    must be the pass-free address itself: the rolled cluster flush drains
    buckets by pass-free address, so a pass-qualified owner posts the labeled
    segment node into a bucket no rolled cluster ever drains and Graphviz
    materializes an unlabeled default ellipse instead (round-27).
    """

    module_stacks: list[tuple[str, ...]] = []
    for label in labels:
        op = _trace_op_for_concrete_label(trace, label)
        module_stacks.append(_effective_render_module_stack(op))
    if not module_stacks or any(not stack for stack in module_stacks):
        return None
    common: str | None = None
    for values in zip(*module_stacks, strict=False):
        if vis_mode == "rolled":
            keys = {value.rsplit(":", 1)[0] for value in values}
        else:
            keys = set(values)
        if len(keys) != 1:
            break
        common = next(iter(keys))
    return common


def _trace_op_for_render_label(trace: Trace, label: str) -> Op:
    """Return the trace op for a pass-free rendered label."""

    try:
        return trace.ops[f"{label}:1"]
    except (KeyError, IndexError):
        return trace.ops[label]


def _op_segment_spanned_modules(
    trace: Trace,
    labels: tuple[str, ...],
    owner: str | None,
    vis_mode: str,
) -> list[str]:
    """Return the distinct module homes an op segment spans below its owner.

    r-b6 R19-5: an op segment owns the honest LCA of its members, which can
    sit ABOVE the modules the ops actually live in (top level when the
    members straddle sibling modules). The rendered box then carries no
    module containment at all, so the label must name the spanned modules.
    Returns the ordered distinct immediate homes below ``owner`` when the
    placement actually loses containment information, else ``[]``.

    T9 (grind-p3): the walk reads the op's FULL module stack, not the
    renderer's effective stack. The effective stack drops an atomic
    module's own innermost level — a presentation choice (the renderer
    keeps the op and drops the box) — and inheriting that drop here made
    the disclosure silently omit hidden atomic module calls from the
    ``spans @...`` list.
    """

    homes: list[str] = []
    has_direct_member = False
    for label in labels:
        op = _trace_op_for_concrete_label(trace, label)
        stack: tuple[str, ...] = tuple(str(module) for module in getattr(op, "modules", ()) or ())
        if vis_mode == "rolled":
            stack = tuple(value.rsplit(":", 1)[0] for value in stack)
        if owner is None or owner not in stack:
            home = stack[0] if stack else None
        else:
            owner_depth = stack.index(owner)
            home = stack[owner_depth + 1] if owner_depth + 1 < len(stack) else None
        if home is None:
            has_direct_member = True
        elif home not in homes:
            homes.append(home)
    if not homes:
        return []
    if len(homes) > 1 or owner is None or has_direct_member:
        return homes
    return []


def _trace_op_for_concrete_label(trace: Trace, label: str) -> Op:
    """Return the trace op for a concrete or legacy pass-free label.

    Pass-qualified labels are exact accessor keys and resolve without any
    fuzzy lookup; legacy pass-free labels fall back to the render-label
    helper.
    """

    if ":" in label:
        return trace.ops[label]
    return _trace_op_for_render_label(trace, label)


def _op_segment_label(
    trace: Trace,
    labels: tuple[str, ...],
    concrete: tuple[str, ...],
) -> str:
    """Return a class-free range label for an operation segment.

    Endpoints stay pass-free for single-pass layers and show the concrete
    pass-qualified label when the layer runs multiple passes, so two per-pass
    segments of a reused block are visually distinguishable and each label
    stands for exactly its own hidden ops.
    """

    # Exact-key membership only: probing missing keys through the public
    # accessor would enter fuzzy lookup, which canonical collapse work must
    # never do.
    valid = {str(op.label) for op in trace.ops}

    def endpoint(base: str, resolved: str) -> str:
        """Render a range endpoint, keeping its pass suffix only for a multi-pass label."""

        multipass = f"{base}:2" in valid
        return resolved if ":" in resolved and multipass else base

    first = endpoint(labels[0], concrete[0])
    last = endpoint(labels[-1], concrete[-1])
    return f"{first} ... {last} -- {len(labels)} ops"


def _segment_owner_key(
    trace: Trace,
    addresses: tuple[str, ...],
    vis_mode: str = "unrolled",
) -> str | None:
    """Return the lowest rendered module cluster that owns ``addresses``.

    Child segments replace consecutive single-call child BOXES (multi-call
    members are refused at run construction), and the renderer places those
    boxes with the lexical parent plus the member's own call index
    (``_collapsed_module_owner_key``) -- always ``parent:1`` here. The
    segment owner mirrors that exact box rule so a segment always renders in
    the same cluster the boxes it replaces would have; deriving it from call
    nesting instead would split a segment from its unabsorbed sibling boxes.

    Rolled clusters are keyed by pass-FREE address (mirroring
    ``_collapsed_module_owner_key``'s rolled branch), so the rolled owner is
    the bare parent address: a pass-qualified owner lands in a bucket the
    rolled cluster flush never drains and the labeled segment node is
    silently dropped (round-27).
    """

    parent = addresses[0].rsplit(".", 1)[0] if "." in addresses[0] else "self"
    if parent == "self" or parent not in trace.modules:
        return None
    if vis_mode == "rolled":
        return parent
    parent_key = f"{parent}:1"
    return parent_key if parent_key in trace.modules else parent


def _members_are_name_consecutive(addresses: tuple[str, ...]) -> bool:
    """Return whether member leaf names form an ascending consecutive range.

    Segment legality is flow-based, so members can be name-noncontiguous or
    name-descending; a ``first-last`` interval label is only honest when the
    leaf names share one parent and one stem and count up by exactly one.
    """

    parents = {address.rsplit(".", 1)[0] if "." in address else "" for address in addresses}
    if len(parents) != 1:
        return False
    stems: list[str] = []
    values: list[int] = []
    for address in addresses:
        leaf = address.rsplit(".", 1)[-1]
        # Manual trailing-digit split, equivalent to
        # ``re.fullmatch(r"(.*?)(\d+)", leaf)`` (lazy stem = maximal digit
        # suffix): that pattern backtracks QUADRATICALLY on
        # artifact-supplied leaves like "9"*n + "x" (measured 9s at 40k
        # chars). ``str.isdecimal`` is exactly the ``\d`` character class.
        cut = len(leaf)
        while cut > 0 and leaf[cut - 1].isdecimal():
            cut -= 1
        if cut == len(leaf):
            return False
        stems.append(leaf[:cut])
        values.append(int(leaf[cut:]))
    if len(set(stems)) != 1:
        return False
    return all(right == left + 1 for left, right in zip(values, values[1:]))


def _child_segment_range_text(addresses: tuple[str, ...]) -> str:
    """Return an honest member summary for a child-segment label.

    Name-consecutive runs keep the compact ``prefix.first-last`` interval.
    Flow-legal but name-noncontiguous runs enumerate their members IN FULL,
    regardless of count (R19): an elided ``first, second, ..., last`` reads
    as a complete interval and overstates hidden membership whenever the
    run skips names that render as separate visible boxes.
    """

    first = addresses[0]
    prefix = first.rsplit(".", 1)[0] if "." in first else ""
    leaves = [address.rsplit(".", 1)[-1] for address in addresses]
    if _members_are_name_consecutive(addresses):
        range_text = f"{leaves[0]}-{leaves[-1]}"
        return f"{prefix}.{range_text}" if prefix else range_text
    listed = ", ".join(leaves)
    return f"{prefix}.{{{listed}}}" if prefix else f"{{{listed}}}"


def _child_segment_label(
    addresses: tuple[str, ...],
    num_layers: int,
    num_buffers: int,
    num_params: int,
) -> str:
    """Return a class-free range label for a child segment."""

    from ._render_common import format_collapsed_module_contents

    range_text = _child_segment_range_text(addresses)
    contents = format_collapsed_module_contents(num_layers, num_buffers)
    return (
        f"{range_text} -- {len(addresses)} blocks, {contents}, "
        f"{_format_param_count(num_params)} params"
    )


def _format_param_count(value: int) -> str:
    """Return a compact parameter count for segment labels."""

    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)
