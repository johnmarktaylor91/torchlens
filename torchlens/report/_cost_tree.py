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
from dataclasses import dataclass, field, replace
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
    forward_mac: int = 0
    backward: int | None = None


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
    "self_forward_mac_flops": True,
    "self_backward_flops": True,
    "self_total_flops": True,
    "subtree_forward_mac_flops": False,
    "subtree_backward_flops": False,
    "subtree_total_flops": False,
}

#: The counting convention of each FLOPs column family. The forward
#: ``*_flops`` columns count every op kind with a rule; the executed backward
#: counts only the matmul and convolution families, so ``*_total_flops``
#: adds the backward to the forward's multiply-accumulate work (2 x true
#: MACs), never to the elementwise-inclusive forward column.
COST_COLUMN_CONVENTIONS: dict[str, str] = {
    "flops": "forward, every op kind with a cost rule (fma2)",
    "forward_mac_flops": "forward multiply-accumulate work only: 2 x true MACs (fma2)",
    "backward_flops": (
        "executed backward, matmul and convolution families (fma2, FlopCounterMode's "
        "convention); other backward calls are named in backward_status, never counted"
    ),
    "total_flops": "forward_mac_flops + backward_flops (one convention)",
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
    self_forward_mac_flops / subtree_forward_mac_flops:
        The forward's multiply-accumulate work (2 x true MACs), the
        convention the executed backward is counted in.
    self_backward_flops / subtree_backward_flops:
        Executed backward FLOPs charged to this row's ops through their
        GradFn calls (matmul and convolution families, fma2), recompute
        included; ``None`` when no backward was counted (the tree's
        ``backward_state`` says whether it was not run or is unknown).
    self_total_flops / subtree_total_flops:
        Forward MAC work plus executed backward; ``None`` with the backward.
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
    self_forward_mac_flops: int | None = None
    subtree_forward_mac_flops: int | None = None
    self_backward_flops: int | None = None
    subtree_backward_flops: int | None = None
    self_total_flops: int | None = None
    subtree_total_flops: int | None = None


@dataclass(frozen=True)
class CostTree:
    """The cost tree: rows in render order plus the invariant denominators.

    ``partition_total`` is the whole-capture known-FLOPs partition total
    (D6: invariant under every view operation); ``hidden_flops`` is the
    display receipt -- the known mass folded into REMAINDER rows by the
    current view (already INCLUDED in the conservation identity through
    those rows, disclosed so a truncated view never reads as complete).

    ``backward_state`` is the three-state backward answer
    (``tl.report.backward_status``); ``backward_total`` is the executed
    backward count when one was counted (the self backward column sums to
    it exactly, unattributed calls included as their own row), else
    ``None``; ``column_conventions`` names each FLOPs column's convention.
    """

    rows: tuple[CostTreeRow, ...]
    partition_total: int
    unknown_events: int
    hidden_flops: int
    column_additivity: dict[str, bool]
    backward_state: str = "grad_disabled"
    backward_total: int | None = None
    forward_mac_total: int = 0
    column_conventions: dict[str, str] = field(default_factory=dict)

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
    extra: str = "",
) -> str:
    """Bottom-up structural digest used by the strict fold signature.

    Two sibling calls fold ONLY when their entire subtrees agree on
    structure AND cost (byte-equal digests) -- the summary memo's strict
    fold doctrine: ``(xN)`` must never hide a sibling that differs.
    """

    child_digests = ",".join(digests[child] for child in children.get(row_id, ()))
    return hashlib.sha256(f"{self_flops}{extra}|{child_digests}".encode()).hexdigest()[:16]


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
        self._backward_ownership()
        self._forest()

    # -- phase 1: op ownership (innermost recorded call) ------------------

    def _ownership(self) -> None:
        """Accumulate per-owner self mass from the aggregation rows."""

        op_owner: dict[str, str | None] = {}
        pass_by_label: dict[str, int] = {}
        self.owner_by_any_label: dict[str, str | None] = {}
        for op in self.trace.layer_list:
            label = str(getattr(op, "label", None) or getattr(op, "layer_label", ""))
            stack = tuple(getattr(op, "module_call_stack", ()) or ())
            op_owner[label] = str(stack[-1]) if stack else None
            pass_by_label[label] = int(getattr(op, "pass_index", 1) or 1)
            for spelling in _label_spellings(op):
                self.owner_by_any_label.setdefault(spelling, op_owner[label])
        self.self_flops: dict[str | None, int] = {}
        self.self_mac: dict[str | None, int] = {}
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
                if row.fma_macs:
                    self.self_mac[owner] = self.self_mac.get(owner, 0) + 2 * int(row.fma_macs)

    def _backward_ownership(self) -> None:
        """Charge each counted backward row to the owner of its forward op.

        Rows whose node has no forward op, and the unattributed bucket, are
        kept apart and rendered as their own REMAINDER rows so the self
        backward column sums to the executed count exactly.
        """

        from ._cost_backward import backward_status

        status = backward_status(self.trace)
        self.backward_state = status.state
        self.backward_known = (
            status.state == "backward_observed" and status.executed_flops is not None
        )
        self.backward_total = status.executed_flops if self.backward_known else None
        self.self_bwd: dict[str | None, int] = {}
        self.bwd_no_op = 0
        self.bwd_unattributed = int(status.unattributed_flops or 0)
        if not self.backward_known:
            return
        for attribution in status.attributions:
            for row in attribution.rows:
                if row.op_label is None:
                    self.bwd_no_op += int(row.flops or 0)
                    continue
                owner = self.owner_by_any_label.get(str(row.op_label))
                if owner is not None and owner not in self.calls:
                    owner = None
                self.self_bwd[owner] = self.self_bwd.get(owner, 0) + int(row.flops or 0)

    def _bwd(self, value: int) -> int | None:
        """A backward cell: the value when the backward was counted, else None."""

        return value if self.backward_known else None

    def _total(self, mac: int, bwd: int) -> int | None:
        """Forward MAC work plus the counted backward (one convention)."""

        return mac + bwd if self.backward_known else None

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
        self.subtree_mac: dict[str, int] = {}
        self.subtree_bwd: dict[str, int] = {}
        for root_label in self.roots:
            self._accumulate(root_label)

    def _accumulate(self, label: str) -> tuple[int, int]:
        """Bottom-up inclusive totals (one traversal over the same rows)."""

        flops = self.self_flops.get(label, 0)
        unknown = self.self_unknown.get(label, 0)
        mac = self.self_mac.get(label, 0)
        bwd = self.self_bwd.get(label, 0)
        for child in self.children[label]:
            child_flops, child_unknown = self._accumulate(child)
            flops += child_flops
            unknown += child_unknown
            mac += self.subtree_mac[child]
            bwd += self.subtree_bwd[child]
        self.subtree_flops[label] = flops
        self.subtree_unknown[label] = unknown
        self.subtree_mac[label] = mac
        self.subtree_bwd[label] = bwd
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
        flops, n_ops, n_unknown, passes, mac, bwd_mass = mass
        bwd = self._bwd(bwd_mass or 0)
        total = self._total(mac, bwd_mass or 0)

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
                self_forward_mac_flops=mac,
                subtree_forward_mac_flops=mac,
                self_backward_flops=bwd,
                subtree_backward_flops=bwd,
                self_total_flops=total,
                subtree_total_flops=total,
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
        own_mac = self.self_mac.get(label, 0)
        own_bwd = self.self_bwd.get(label, 0)
        self_mac = 0 if is_root else own_mac * folded
        self_bwd = 0 if is_root else own_bwd * folded
        sub_mac = self.subtree_mac.get(label, 0) * folded
        sub_bwd = self.subtree_bwd.get(label, 0) * folded
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
                self_forward_mac_flops=self_mac,
                subtree_forward_mac_flops=sub_mac,
                self_backward_flops=self._bwd(self_bwd),
                subtree_backward_flops=self._bwd(sub_bwd),
                self_total_flops=self._total(self_mac, self_bwd),
                subtree_total_flops=self._total(sub_mac, sub_bwd),
            )
        )
        if is_root and (own > 0 or own_ops > 0 or own_bwd > 0):
            self._remainder(
                f"{label}::direct",
                "(root-direct ops)",
                (depth + 1, label),
                _RemainderMass(
                    own,
                    own_ops,
                    self.self_unknown.get(label, 0),
                    len(self.self_passes.get(label, ())) or None,
                    own_mac,
                    own_bwd,
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
                sum(self.subtree_mac.get(child, 0) for child in kids) * folded,
                sum(self.subtree_bwd.get(child, 0) for child in kids) * folded,
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
                sum(self.subtree_mac.get(item[0], 0) * item[1] for item in dropped) * folded,
                sum(self.subtree_bwd.get(item[0], 0) * item[1] for item in dropped) * folded,
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
                # Folding multiplies one representative's cells, so siblings
                # fold only when their MAC and backward cells agree too.
                digests[node] = _structure_digest(
                    node,
                    self.self_flops.get(node, 0),
                    self.children,
                    digests,
                    extra=f"|{self.self_mac.get(node, 0)}|{self.self_bwd.get(node, 0)}",
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
        mac_total = sum(self.self_mac.values())
        bwd_total = sum(self.self_bwd.values()) + self.bwd_no_op + self.bwd_unattributed
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
                self_forward_mac_flops=0,
                subtree_forward_mac_flops=mac_total,
                self_backward_flops=self._bwd(0),
                subtree_backward_flops=self._bwd(bwd_total),
                self_total_flops=self._total(0, 0),
                subtree_total_flops=self._total(mac_total, bwd_total),
            )
        )
        if direct or self.self_ops.get(None, 0) or self.self_bwd.get(None, 0):
            self._remainder(
                "::root::direct",
                "(root-direct ops)",
                (1, "::root"),
                _RemainderMass(
                    direct,
                    self.self_ops.get(None, 0),
                    self.self_unknown.get(None, 0),
                    len(self.self_passes.get(None, ())) or None,
                    self.self_mac.get(None, 0),
                    self.self_bwd.get(None, 0),
                ),
            )
        self._backward_buckets(1, "::root")

    def _attach_outside_ops(self) -> None:
        """Ops outside every recorded call: one remainder under the first root."""

        extra_bwd = self._backward_buckets(1, self.roots[0])
        outside_bwd = self.self_bwd.get(None, 0)
        if not (self.self_flops.get(None, 0) or self.self_ops.get(None, 0) or outside_bwd):
            self._grow_first_root(0, 0, extra_bwd)
            return
        outside = self.self_flops.get(None, 0)
        outside_mac = self.self_mac.get(None, 0)
        self._remainder(
            "::outside",
            "(ops outside recorded calls)",
            (1, self.roots[0]),
            _RemainderMass(
                outside,
                self.self_ops.get(None, 0),
                self.self_unknown.get(None, 0),
                len(self.self_passes.get(None, ())) or None,
                outside_mac,
                outside_bwd,
            ),
        )
        self._grow_first_root(outside, outside_mac, outside_bwd + extra_bwd)

    def _backward_buckets(self, depth: int, parent_id: str) -> int:
        """REMAINDER rows for backward work with no forward-op owner; returns their sum.

        Node calls with no forward op (a reentrant checkpoint node, for
        example) and the unattributed calls each get one row, so the self
        backward column sums to the executed count exactly.
        """

        if not self.backward_known:
            return 0
        for suffix, label, value in (
            ("no_op", "(backward nodes with no forward op)", self.bwd_no_op),
            ("unattributed", "(unattributed backward calls)", self.bwd_unattributed),
        ):
            if value:
                self._remainder(
                    f"::backward::{suffix}",
                    label,
                    (depth, parent_id),
                    _RemainderMass(0, 0, 0, None, 0, value),
                )
        return self.bwd_no_op + self.bwd_unattributed

    def _grow_first_root(self, flops: int, mac: int, bwd: int) -> None:
        """Add mass attached under the first root to that root's subtree cells."""

        if not (flops or mac or bwd):
            return
        for index, tree_row in enumerate(self.rows):
            if tree_row.row_id == self.roots[0]:
                sub_flops = (tree_row.subtree_flops or 0) + flops
                sub_mac = (tree_row.subtree_forward_mac_flops or 0) + mac
                sub_bwd = self.subtree_bwd.get(self.roots[0], 0) + bwd
                self.rows[index] = replace(
                    tree_row,
                    subtree_flops=sub_flops,
                    subtree_pct=_pct(sub_flops, self.partition_total),
                    subtree_forward_mac_flops=sub_mac,
                    subtree_backward_flops=self._bwd(sub_bwd),
                    subtree_total_flops=self._total(sub_mac, sub_bwd),
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
            backward_state=self.backward_state,
            backward_total=self.backward_total,
            forward_mac_total=sum(self.self_mac.values()),
            column_conventions=dict(COST_COLUMN_CONVENTIONS),
        )


def _label_spellings(op: Any) -> tuple[str, ...]:
    """The label spellings a GradFn's ``op_label`` may use for one op."""

    spellings = []
    for name in ("label", "layer_label", "layer_label_w_pass", "layer_label_no_pass"):
        value = getattr(op, name, None)
        if value:
            spellings.append(str(value))
    return tuple(spellings)


def _call_display(call: Any) -> str:
    """Display label for one module call (address:pass spelling kept)."""

    return str(getattr(call, "call_label", "?"))


def format_cost_tree(tree: CostTree) -> str:
    """Render the cost tree as an ASCII table with honest receipts.

    The header names both column families and their additivity; the
    footer prints the invariant denominator, the unknown EVENT count
    (never a FLOP percentage, D4), and the hidden-share display receipt
    when a view dropped rows (D6). When an executed backward was counted,
    three self columns follow: forward MAC work, executed backward, and
    their total (one convention); otherwise the footer states the backward
    as not run or unknown, never 0.
    """

    counted = tree.backward_total is not None
    head = f"{'call':<44} {'self FLOPs':>14} {'self%':>7} {'subtree FLOPs':>14} {'subtree%':>9}"
    if counted:
        head += f" {'fwd MAC':>12} {'bwd':>12} {'fwd MAC+bwd':>12}"
    lines = ["cost tree (self = exclusive, additive; subtree = inclusive, NON-additive)", head]
    for row in tree.rows:
        indent = "  " * row.depth
        label = f"{indent}{row.label}"
        if row.passes is not None and row.passes > 1:
            label += f" [{row.passes} passes]"
        self_cell = "-" if row.self_flops is None else str(row.self_flops)
        self_pct = "-" if row.self_pct is None else f"{row.self_pct:.1f}%"
        sub_cell = "-" if row.subtree_flops is None else str(row.subtree_flops)
        sub_pct = "-" if row.subtree_pct is None else f"{row.subtree_pct:.1f}%"
        line = f"{label:<44} {self_cell:>14} {self_pct:>7} {sub_cell:>14} {sub_pct:>9}"
        if counted:
            line += " " + " ".join(
                f"{'-' if value is None else value:>12}"
                for value in (
                    row.self_forward_mac_flops,
                    row.self_backward_flops,
                    row.self_total_flops,
                )
            )
        lines.append(line)
    lines.append(f"partition total (known forward FLOPs, whole capture): {tree.partition_total}")
    lines.append(_backward_footer(tree))
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


def _backward_footer(tree: CostTree) -> str:
    """One line on the backward columns: counted with its convention, or why absent."""

    if tree.backward_total is not None:
        return (
            f"executed backward: {tree.backward_total} FLOPs (matmul and convolution "
            "families, fma2); fwd MAC = 2 x true MACs, the same convention"
        )
    if tree.backward_state == "grad_disabled":
        return "executed backward: not run (autograd recorded no graph for this capture)"
    return (
        "executed backward: unknown (no counted backward on this object; "
        "see tl.report.backward_status)"
    )
