"""Percent columns and the cost tree view (F09; costreport items 7-8).

D5: additivity is a property of the (row, column) PAIR. One row per
accounting unit; two column families -- ``self`` (exclusive, additive,
sums to the partition total) and ``subtree`` (inclusive, non-additive,
marked) -- each column declares ``additive: bool`` and every row carries
its role (OWNER / SUBTOTAL / REMAINDER) into every export so nobody sums
both. The root row carries self=0, subtree=partition total (T-ROOT; the
measured 3.878x naive module sum is the forbidden regression).

D4: exclusive percents sum to 100% OF THE KNOWN TOTAL; the unknown
disclosure is an EVENT count plus the named ledger, never a FLOP
percentage.

D6: sort/filter/top_k/depth are VIEW operations -- the denominator stays
the whole-capture partition total, invariant, with a display receipt
naming the hidden share.

D8: the tree is a LAYOUT over the same row graph, never a second
traversal: recorded call-parent identity, named model root, strict
sibling folding, pass annotation, top-k owners with ancestors as
uncounted context and deterministic remainder rows, exact conservation
under depth cutoff.

Live-only derivations over the C02 aggregation face (fence chain
A07 -> C02 -> F09); spellings DOCUMENTED-UNSTABLE pending naming-session
ratification.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NamedTuple

from .._errors import InvalidArgumentError
from ._compute_truth import ComputeAggregation, compute_aggregation

if TYPE_CHECKING:
    from ..data_classes.trace import Trace


class _RemainderMass(NamedTuple):
    """The conserved mass one REMAINDER row folds in (D5/D6)."""

    flops: int
    n_ops: int
    n_unknown: int
    passes: int | None


#: Row roles (D5, closed): an OWNER row's self family is additive mass; a
#: SUBTOTAL row aggregates owners below it; a REMAINDER row is synthesized
#: to conserve totals under a view (top_k / depth cutoff / root-direct ops).
COST_ROW_ROLES: tuple[str, ...] = ("OWNER", "SUBTOTAL", "REMAINDER")

#: Column additivity declarations (D5): the metadata every export carries.
COST_COLUMN_ADDITIVITY: dict[str, bool] = {
    "self_flops": True,
    "self_pct": True,
    "subtree_flops": False,
    "subtree_pct": False,
}


@dataclass(frozen=True)
class CostTreeRow:
    """One accounting-unit row of the cost tree (D5/D8).

    Parameters
    ----------
    row_id:
        Stable identity (call label, or a synthesized remainder id).
    label:
        Display label (module address / model root name / remainder note).
    kind:
        ``"root"`` / ``"call"`` / ``"remainder"``.
    role:
        One of :data:`COST_ROW_ROLES`.
    depth:
        Tree depth (root = 0).
    parent_id:
        ``row_id`` of the parent row (``None`` on the root).
    self_flops:
        Exclusive known-FLOPs mass owned directly by this row (additive).
    subtree_flops:
        Inclusive known-FLOPs mass of this row and everything below it
        (NON-additive; never sum this column).
    self_pct / subtree_pct:
        The same families as percents of the whole-capture KNOWN partition
        total (D4/D6) -- never of a visible-row sum.
    n_ops:
        Compute ops owned directly (self scope).
    n_unknown_ops:
        Directly-owned ops with no cost rule (an EVENT count, D4).
    passes:
        Distinct pass count over directly-owned ops (annotated, never
        conflated with op count).
    folded:
        Number of sibling calls this row represents (strict fold; 1 =
        unfolded).
    """

    row_id: str
    label: str
    kind: str
    role: str
    depth: int
    parent_id: str | None
    self_flops: int | None
    subtree_flops: int | None
    self_pct: float | None
    subtree_pct: float | None
    n_ops: int
    n_unknown_ops: int
    passes: int | None
    folded: int


@dataclass(frozen=True)
class CostTree:
    """The cost tree: rows in render order plus the invariant denominators.

    ``partition_total`` is the whole-capture known-FLOPs partition total
    (D6: invariant under every view operation); ``hidden_flops`` is the
    display receipt -- the known mass folded into REMAINDER rows by the
    current view (already INCLUDED in the conservation identity through
    those rows, disclosed so a truncated view never reads as complete).
    """

    rows: tuple[CostTreeRow, ...]
    partition_total: int
    unknown_events: int
    hidden_flops: int
    column_additivity: dict[str, bool]

    def __repr__(self) -> str:
        """Bounded identity card (F10/D31): rows point, never dump."""

        return (
            f"CostTree({len(self.rows)} rows, partition_total={self.partition_total}, "
            f"unknown_events={self.unknown_events}, hidden_flops={self.hidden_flops}; "
            f"print() for the table, read .rows)"
        )

    def __str__(self) -> str:
        """Render the tree as an ASCII table."""

        return format_cost_tree(self)


def _pct(value: int | None, total: int) -> float | None:
    """Percent of the known partition total (D4); None-safe."""

    if value is None or total <= 0:
        return None
    return 100.0 * value / total


def _innermost_call(op_row_label: str, label_to_call: dict[str, str]) -> str | None:
    """The innermost containing call label for one op label."""

    return label_to_call.get(op_row_label)


def _structure_digest(
    row_id: str,
    self_flops: int | None,
    children: dict[str, list[str]],
    digests: dict[str, str],
) -> str:
    """Bottom-up structural digest used by the strict fold signature.

    Two sibling calls fold ONLY when their entire subtrees agree on
    structure AND cost (byte-equal digests) -- the summary memo's strict
    fold doctrine: ``(xN)`` must never hide a sibling that differs.
    """

    child_digests = ",".join(digests[child] for child in children.get(row_id, ()))
    return hashlib.sha256(f"{self_flops}|{child_digests}".encode()).hexdigest()[:16]


def build_cost_tree(
    trace: Trace,
    *,
    top_k: int | None = None,
    max_depth: int | None = None,
    fold_repeats: bool = True,
    aggregation: ComputeAggregation | None = None,
) -> CostTree:
    """Build the module-call cost tree from the ONE aggregation (D8).

    Parameters
    ----------
    trace:
        Finished trace.
    top_k:
        Keep the ``top_k`` largest OWNER/SUBTOTAL rows per parent (by
        subtree mass); hidden siblings collapse into one deterministic
        REMAINDER row per parent. Denominators are unaffected (D6).
    max_depth:
        Fold rows deeper than this depth into their ancestor's REMAINDER
        row; conservation stays exact.
    fold_repeats:
        Fold structurally identical sibling subtrees under a strict
        byte-equal signature, annotated ``(xN)``.
    aggregation:
        An existing :class:`ComputeAggregation` to project (the tree is a
        layout over the same rows, never a second traversal); built once
        from the trace when omitted.
    """

    if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1):
        raise InvalidArgumentError(
            f"top_k must be a positive integer or None; got {top_k!r}.",
            code="cost_tree_top_k_invalid",
            remedy="Pass top_k >= 1 to keep the k largest rows per parent, or None for all.",
        )
    if max_depth is not None and (
        not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0
    ):
        raise InvalidArgumentError(
            f"max_depth must be a non-negative integer or None; got {max_depth!r}.",
            code="cost_tree_depth_invalid",
            remedy="Pass max_depth >= 0 (0 keeps only the root), or None for the full tree.",
        )

    agg = compute_aggregation(trace) if aggregation is None else aggregation
    builder = _TreeBuilder(
        trace,
        agg,
        top_k=top_k,
        max_depth=max_depth,
        fold_repeats=fold_repeats,
    )
    return builder.build()


class _TreeBuilder:
    """Stateful assembly of one cost tree (D8's phases as named steps).

    The phases are genuinely sequential -- ownership accumulation, the
    call forest, bottom-up inclusive totals, then the view-shaped emit
    walk -- and every REMAINDER row goes through ONE constructor so the
    conservation arithmetic lives in one place.
    """

    def __init__(
        self,
        trace: Any,
        agg: ComputeAggregation,
        *,
        top_k: int | None,
        max_depth: int | None,
        fold_repeats: bool,
    ) -> None:
        """Capture the view arguments and derive the shared indexes."""

        self.trace = trace
        self.agg = agg
        self.top_k = top_k
        self.max_depth = max_depth
        self.fold_repeats = fold_repeats
        self.partition_total = int(agg.partition_total)
        self.calls = {str(call.call_label): call for call in trace.module_calls.values()}
        self.model_root_name = str(getattr(trace, "model_class_name", None) or "model")
        self.rows: list[CostTreeRow] = []
        self.hidden_flops = 0
        self._ownership()
        self._forest()

    # -- phase 1: op ownership (innermost recorded call) ------------------

    def _ownership(self) -> None:
        """Accumulate per-owner self mass from the aggregation rows."""

        op_owner: dict[str, str | None] = {}
        pass_by_label: dict[str, int] = {}
        for op in self.trace.layer_list:
            label = str(getattr(op, "label", None) or getattr(op, "layer_label", ""))
            stack = tuple(getattr(op, "module_call_stack", ()) or ())
            op_owner[label] = str(stack[-1]) if stack else None
            pass_by_label[label] = int(getattr(op, "pass_index", 1) or 1)
        self.self_flops: dict[str | None, int] = {}
        self.self_ops: dict[str | None, int] = {}
        self.self_unknown: dict[str | None, int] = {}
        self.self_passes: dict[str | None, set[int]] = {}
        for row in self.agg.rows:
            if row.kind != "op":
                continue
            row_key = row.op_label or row.label
            owner = op_owner.get(row_key)
            if owner is not None and owner not in self.calls:
                owner = None
            self.self_ops[owner] = self.self_ops.get(owner, 0) + 1
            self.self_passes.setdefault(owner, set()).add(pass_by_label.get(row_key, 1))
            if row.coverage_class == "unknown":
                self.self_unknown[owner] = self.self_unknown.get(owner, 0) + 1
            elif row.flops_fma2 is not None:
                self.self_flops[owner] = self.self_flops.get(owner, 0) + int(row.flops_fma2)

    # -- phase 2: the call forest + inclusive totals -----------------------

    def _forest(self) -> None:
        """Build children/roots from recorded call-parent identity (D8)."""

        self.children: dict[str, list[str]] = {label: [] for label in self.calls}
        self.roots: list[str] = []
        for label, call in self.calls.items():
            parent = getattr(call, "call_parent", None)
            if parent is None or str(parent) not in self.calls:
                self.roots.append(label)
            else:
                self.children[str(parent)].append(label)
        for siblings in self.children.values():
            siblings.sort(key=lambda label: int(self.calls[label].ordinal_index))
        self.roots.sort(key=lambda label: int(self.calls[label].ordinal_index))
        self.subtree_flops: dict[str, int] = {}
        self.subtree_unknown: dict[str, int] = {}
        for root_label in self.roots:
            self._accumulate(root_label)

    def _accumulate(self, label: str) -> tuple[int, int]:
        """Bottom-up inclusive totals (one traversal over the same rows)."""

        flops = self.self_flops.get(label, 0)
        unknown = self.self_unknown.get(label, 0)
        for child in self.children[label]:
            child_flops, child_unknown = self._accumulate(child)
            flops += child_flops
            unknown += child_unknown
        self.subtree_flops[label] = flops
        self.subtree_unknown[label] = unknown
        return flops, unknown

    # -- row constructors ---------------------------------------------------

    def _remainder(
        self,
        row_id: str,
        label: str,
        position: tuple[int, str],
        mass: _RemainderMass,
    ) -> None:
        """Append one REMAINDER row (the ONE conservation constructor).

        ``position`` is the (depth, parent_id) pair; ``mass`` is the
        conserved (flops, n_ops, n_unknown, passes) being folded in.
        """

        depth, parent_id = position
        flops, n_ops, n_unknown, passes = mass

        self.rows.append(
            CostTreeRow(
                row_id=row_id,
                label=label,
                kind="remainder",
                role="REMAINDER",
                depth=depth,
                parent_id=parent_id,
                self_flops=flops,
                subtree_flops=flops,
                self_pct=_pct(flops, self.partition_total),
                subtree_pct=_pct(flops, self.partition_total),
                n_ops=n_ops,
                n_unknown_ops=n_unknown,
                passes=passes,
                folded=1,
            )
        )

    # -- phase 3: the view-shaped emit walk ---------------------------------

    def _emit_call(self, label: str, depth: int, parent_id: str | None, folded: int) -> None:
        """Emit one call row and its children under the current view."""

        is_root = parent_id is None
        display = self.model_root_name if is_root else _call_display(self.calls[label])
        own = self.self_flops.get(label, 0)
        own_ops = self.self_ops.get(label, 0)
        # T-ROOT: the root row's self cells are 0; root-direct ops move to
        # an explicit REMAINDER child so conservation stays exact.
        row_self = 0 if is_root else own * folded
        self.rows.append(
            CostTreeRow(
                row_id=label,
                label=display + (f" (x{folded})" if folded > 1 else ""),
                kind="root" if is_root else "call",
                role="SUBTOTAL",
                depth=depth,
                parent_id=parent_id,
                self_flops=row_self,
                subtree_flops=self.subtree_flops.get(label, 0) * folded,
                self_pct=_pct(row_self, self.partition_total),
                subtree_pct=_pct(self.subtree_flops.get(label, 0) * folded, self.partition_total),
                n_ops=own_ops * folded,
                n_unknown_ops=self.self_unknown.get(label, 0) * folded,
                passes=(len(self.self_passes.get(label, ())) or None),
                folded=folded,
            )
        )
        if is_root and (own > 0 or own_ops > 0):
            self._remainder(
                f"{label}::direct",
                "(root-direct ops)",
                (depth + 1, label),
                _RemainderMass(
                    own,
                    own_ops,
                    self.self_unknown.get(label, 0),
                    len(self.self_passes.get(label, ())) or None,
                ),
            )
        if self.max_depth is not None and depth + 1 > self.max_depth:
            self._emit_depth_remainder(label, depth, folded)
            return
        visible, dropped = self._select_top_k(self._fold_groups(self.children[label]))
        for child, fold_count in visible:
            self._emit_call(child, depth + 1, label, fold_count * folded)
        if dropped:
            self._emit_topk_remainder(label, depth, folded, dropped)

    def _emit_depth_remainder(self, label: str, depth: int, folded: int) -> None:
        """Fold everything below the depth cutoff into one honest row."""

        kids = self.children[label]
        if not kids:
            return
        hidden = sum(self.subtree_flops.get(child, 0) for child in kids) * folded
        self.hidden_flops += hidden
        self._remainder(
            f"{label}::depth",
            f"(+ {len(kids)} calls below depth {self.max_depth})",
            (depth + 1, label),
            _RemainderMass(
                hidden,
                sum(self.self_ops.get(child, 0) for child in kids),
                sum(self.subtree_unknown.get(child, 0) for child in kids),
                None,
            ),
        )

    def _emit_topk_remainder(
        self, label: str, depth: int, folded: int, dropped: list[tuple[str, int]]
    ) -> None:
        """Aggregate the top-k-hidden siblings into one deterministic row."""

        dropped_flops = sum(self.subtree_flops.get(item[0], 0) * item[1] for item in dropped)
        self.hidden_flops += dropped_flops * folded
        self._remainder(
            f"{label}::topk",
            f"(+ {sum(item[1] for item in dropped)} more calls)",
            (depth + 1, label),
            _RemainderMass(
                dropped_flops * folded,
                sum(self.self_ops.get(item[0], 0) * item[1] for item in dropped),
                sum(self.subtree_unknown.get(item[0], 0) * item[1] for item in dropped),
                None,
            ),
        )

    def _fold_groups(self, ordered: list[str]) -> list[tuple[str, int]]:
        """Group consecutive identical siblings under the strict signature."""

        if not (self.fold_repeats and ordered):
            return [(child, 1) for child in ordered]
        digests: dict[str, str] = {}

        def _digest(node: str) -> str:
            """Strict bottom-up fold signature for one subtree."""

            if node not in digests:
                for child in self.children[node]:
                    _digest(child)
                digests[node] = _structure_digest(
                    node, self.self_flops.get(node, 0), self.children, digests
                )
            return digests[node]

        def _base_name(node: str) -> str:
            """Sibling family name (address sans trailing ordinal digits)."""

            return _call_display(self.calls[node]).rsplit(":", 1)[0].rstrip("0123456789")

        groups: list[tuple[str, int]] = []
        run_start = 0
        while run_start < len(ordered):
            run_end = run_start + 1
            while (
                run_end < len(ordered)
                and _digest(ordered[run_end]) == _digest(ordered[run_start])
                and _base_name(ordered[run_end]) == _base_name(ordered[run_start])
            ):
                run_end += 1
            groups.append((ordered[run_start], run_end - run_start))
            run_start = run_end
        return groups

    def _select_top_k(
        self, groups: list[tuple[str, int]]
    ) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
        """Split sibling groups into visible and top-k-hidden (D6 view op)."""

        if self.top_k is None or len(groups) <= self.top_k:
            return groups, []
        ranked = sorted(
            groups, key=lambda item: self.subtree_flops.get(item[0], 0) * item[1], reverse=True
        )
        kept = {id(item) for item in ranked[: self.top_k]}
        visible = [item for item in groups if id(item) in kept]
        dropped = [item for item in groups if id(item) not in kept]
        return visible, dropped

    # -- assembly -------------------------------------------------------------

    def _synthetic_root(self) -> None:
        """One synthetic root for a trace with no module calls."""

        direct = self.self_flops.get(None, 0)
        self.rows.append(
            CostTreeRow(
                row_id="::root",
                label=self.model_root_name,
                kind="root",
                role="SUBTOTAL",
                depth=0,
                parent_id=None,
                self_flops=0,
                subtree_flops=self.partition_total,
                self_pct=_pct(0, self.partition_total),
                subtree_pct=_pct(self.partition_total, self.partition_total),
                n_ops=0,
                n_unknown_ops=0,
                passes=None,
                folded=1,
            )
        )
        if direct or self.self_ops.get(None, 0):
            self._remainder(
                "::root::direct",
                "(root-direct ops)",
                (1, "::root"),
                _RemainderMass(
                    direct,
                    self.self_ops.get(None, 0),
                    self.self_unknown.get(None, 0),
                    len(self.self_passes.get(None, ())) or None,
                ),
            )

    def _attach_outside_ops(self) -> None:
        """Ops outside every recorded call: one remainder under the first root."""

        if not (self.self_flops.get(None, 0) or self.self_ops.get(None, 0)):
            return
        outside = self.self_flops.get(None, 0)
        self._remainder(
            "::outside",
            "(ops outside recorded calls)",
            (1, self.roots[0]),
            _RemainderMass(
                outside,
                self.self_ops.get(None, 0),
                self.self_unknown.get(None, 0),
                len(self.self_passes.get(None, ())) or None,
            ),
        )
        for index, tree_row in enumerate(self.rows):
            if tree_row.row_id == self.roots[0]:
                self.rows[index] = replace(
                    tree_row,
                    subtree_flops=(tree_row.subtree_flops or 0) + outside,
                    subtree_pct=_pct((tree_row.subtree_flops or 0) + outside, self.partition_total),
                )
                break

    def build(self) -> CostTree:
        """Run the emit walk and freeze the tree."""

        if self.roots:
            for root_label in self.roots:
                self._emit_call(root_label, 0, None, 1)
            self._attach_outside_ops()
        else:
            self._synthetic_root()
        return CostTree(
            rows=tuple(self.rows),
            partition_total=self.partition_total,
            unknown_events=self.agg.coverage.unknown,
            hidden_flops=self.hidden_flops,
            column_additivity=dict(COST_COLUMN_ADDITIVITY),
        )


def _call_display(call: Any) -> str:
    """Display label for one module call (address:pass spelling kept)."""

    return str(getattr(call, "call_label", "?"))


def format_cost_tree(tree: CostTree) -> str:
    """Render the cost tree as an ASCII table with honest receipts.

    The header names both column families and their additivity; the
    footer prints the invariant denominator, the unknown EVENT count
    (never a FLOP percentage, D4), and the hidden-share display receipt
    when a view dropped rows (D6).
    """

    lines = [
        "cost tree (self = exclusive, additive; subtree = inclusive, NON-additive)",
        f"{'call':<44} {'self FLOPs':>14} {'self%':>7} {'subtree FLOPs':>14} {'subtree%':>9}",
    ]
    for row in tree.rows:
        indent = "  " * row.depth
        label = f"{indent}{row.label}"
        if row.passes is not None and row.passes > 1:
            label += f" [{row.passes} passes]"
        self_cell = "-" if row.self_flops is None else str(row.self_flops)
        self_pct = "-" if row.self_pct is None else f"{row.self_pct:.1f}%"
        sub_cell = "-" if row.subtree_flops is None else str(row.subtree_flops)
        sub_pct = "-" if row.subtree_pct is None else f"{row.subtree_pct:.1f}%"
        lines.append(f"{label:<44} {self_cell:>14} {self_pct:>7} {sub_cell:>14} {sub_pct:>9}")
    lines.append(f"partition total (known forward FLOPs, whole capture): {tree.partition_total}")
    if tree.unknown_events:
        lines.append(
            f"unknown-cost ops: {tree.unknown_events} (an event count; see unknown_flop_ops)"
        )
    if tree.hidden_flops:
        lines.append(
            f"view receipt: {tree.hidden_flops} known FLOPs folded into remainder rows "
            "by top_k/depth"
        )
    return "\n".join(lines)
