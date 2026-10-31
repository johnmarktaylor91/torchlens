"""Executed-backward FLOPs attributed to the node and forward op that did the work.

A captured backward records every ATen call it dispatched
(``BackwardPass.aten_ops``), each tagged with the GradFn call it ran
under. This module charges every counted call to that GradFn call and to the
forward op the node differentiates (``GradFn.op_label``). A call that ran
under no recorded node (the engine's seed gradient, nodes outside the walk)
goes to the ``unattributed`` bucket, which names each call: it is never
dropped and never spread over the rows. The rows plus the bucket equal the
pass's executed count exactly (integers).

In a backward that several traces share (one engine run reaching nodes of
two captures), each trace's census sees the whole run; a call that ran on
another trace's node belongs to that trace and is left out of this trace's
view (counted in ``other_trace_ops``), and a call on nobody's node is
unattributed in every view.

Recompute. Under activation checkpointing the backward replays forward
kernels to rebuild what the forward did not keep. The engine runs backward
kernels with grad mode off unless the pass builds a graph, while checkpoint
replays run with grad mode on, so a counted call that dispatched with grad
mode on, inside a pass without ``create_graph``, on a trace whose checkpoint
witness saw checkpointing, is labeled recompute on the row it ran under. The
label splits a row's backward count; it never moves a FLOP between rows.
Where the evidence cannot separate the two (a ``create_graph`` pass that
checkpoints, or a trace with no checkpoint evidence) the recompute share is
``None`` (unknown), never a guess.

The attribution of a pass is written onto the pass as a small closed record
(``BackwardPass.executed_flops_record``) so a saved and loaded trace reads
the counted answer; the per-call census itself is session-time.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .._utils import _torch_compat
from ..capture.flops import DISPATCH_COUNTING_CONVENTION

#: Closed key set of the persisted per-pass record (SCHEMA-REQUESTS row 15).
EXECUTED_FLOPS_RECORD_KEYS: frozenset[str] = frozenset(
    {"convention", "executed_flops", "uncounted_ops", "rows", "unattributed"}
)

#: Longest string the persisted record keeps (labels and operator names).
_RECORD_STRING_LIMIT = 200

#: Recompute evidence states of one pass's attribution.
RECOMPUTE_STATES: tuple[str, ...] = ("labeled", "none_observed", "unknown")


@dataclass(frozen=True)
class BackwardFlopsRow:
    """Executed backward FLOPs charged to one GradFn call (DOCUMENTED-UNSTABLE).

    Attributes
    ----------
    pass_index:
        One-based backward pass the call ran in.
    grad_fn_label:
        Label of the backward node (``"addmm_back_1_3"``).
    call_index:
        One-based call of that node.
    op_label:
        The forward op the node differentiates, or ``None`` for a node with
        no forward op (``AccumulateGrad``, a reentrant checkpoint node).
    flops:
        Counted fma2 FLOPs of the calls that ran under this node (matmul and
        convolution families), recompute included; ``None`` when a counted
        call's shapes could not be read.
    recompute_flops:
        The part of ``flops`` that replayed forward kernels under activation
        checkpointing; ``0`` when none ran, ``None`` when the evidence cannot
        tell (see the module docstring).
    """

    pass_index: int
    grad_fn_label: str
    call_index: int
    op_label: str | None
    flops: int | None
    recompute_flops: int | None


@dataclass(frozen=True)
class BackwardFlopsAttribution:
    """One backward pass's executed FLOPs, row by row (DOCUMENTED-UNSTABLE).

    ``sum(row.flops for row in rows) + unattributed_flops == executed_flops``
    exactly whenever ``executed_flops`` is known.

    Attributes
    ----------
    pass_index:
        One-based backward pass index.
    convention:
        The counting convention (fma2, matmul and convolution families).
    executed_flops:
        This trace's executed backward count for the pass, or ``None`` when
        a counted call could not be read or the pass raised before it
        finished (a backward that did not run to completion has no count).
    rows:
        One row per GradFn call that dispatched at least one call, in
        first-run order.
    unattributed_flops:
        Counted FLOPs of calls that ran under no recorded node.
    unattributed_ops:
        Every unattributed call as ``(operator, flops)``, in run order.
    uncounted_ops:
        Call kinds outside the counted families, with call counts (named,
        never zeroed).
    recompute:
        ``"labeled"``, ``"none_observed"`` or ``"unknown"``.
    other_trace_ops:
        Calls that ran on another trace's nodes in a shared backward (left
        out of this view); ``None`` when read from a saved record.
    source:
        ``"census"`` (this session's recorded calls) or ``"saved record"``.
    """

    pass_index: int
    convention: str
    executed_flops: int | None
    rows: tuple[BackwardFlopsRow, ...]
    unattributed_flops: int | None
    unattributed_ops: tuple[tuple[str, int | None], ...]
    uncounted_ops: tuple[tuple[str, int], ...]
    recompute: str
    other_trace_ops: int | None
    source: str

    def __repr__(self) -> str:
        """Bounded identity card: totals lead, rows are counted, never dumped."""

        return (
            f"BackwardFlopsAttribution(pass {self.pass_index}: executed_flops="
            f"{self.executed_flops}, {len(self.rows)} row(s), unattributed_flops="
            f"{self.unattributed_flops} over {len(self.unattributed_ops)} call(s), "
            f"recompute={self.recompute}; read .rows)"
        )

    @property
    def recompute_flops(self) -> int | None:
        """Total recompute FLOPs of the pass (``None`` when any row is unknown)."""

        total = 0
        for row in self.rows:
            if row.recompute_flops is None:
                return None
            total += row.recompute_flops
        return total

    def flops_by_op(self) -> dict[str | None, int | None]:
        """Backward FLOPs per forward op label (``None`` keys nodes with no op)."""

        totals: dict[str | None, int | None] = {}
        for row in self.rows:
            previous = totals.get(row.op_label, 0)
            totals[row.op_label] = (
                None if previous is None or row.flops is None else previous + row.flops
            )
        return totals


class _RowAccumulator:
    """Mutable per-row totals while one pass's calls are folded."""

    __slots__ = ("flops", "recompute", "key", "label", "op_label")

    def __init__(self, key: tuple[int, int], label: str, op_label: str | None) -> None:
        """Start an empty row for one GradFn call."""

        self.key = key
        self.label = label
        self.op_label = op_label
        self.flops: int | None = 0
        self.recompute: int | None = 0

    def add(self, op: Any, builds_graph: bool | None, checkpoint: str) -> bool:
        """Fold one counted call into the row; whether its recompute share is unknown."""

        self.flops = _add(self.flops, op.flops)
        if op.flops is None:
            self.recompute = None
            return False
        share = _recompute_share(op, builds_graph, checkpoint)
        self.recompute = _add(self.recompute, share)
        return share is None


class _OwnedElsewhere:
    """Whether a node this trace walked differentiates another trace's forward op.

    In a backward shared by several captures each capture walks every
    reachable node; a node belongs to the capture whose forward recorded its
    op (``has_op`` there). A node with no forward op in any capture (a
    parameter's ``AccumulateGrad``, the user's loss reduction) stays a row in
    every capture that walked it.
    """

    def __init__(self, trace: Any, pass_index: int) -> None:
        """Collect the other captures that shared this pass's engine run."""

        from ..backends.torch._backward_dispatch import shared_backward_partners

        self._partners = shared_backward_partners(trace, pass_index)
        self._cache: dict[int, bool] = {}

    def __call__(self, grad_fn: Any, object_id: int) -> bool:
        """True when another capture's forward recorded this node's op."""

        if getattr(grad_fn, "has_op", False) or not self._partners:
            return False
        if object_id not in self._cache:
            self._cache[object_id] = any(
                _has_op_in_pass(other, object_id, other_pass)
                for other, other_pass in self._partners
            )
        return self._cache[object_id]


def _has_op_in_pass(trace: Any, object_id: int, pass_index: int) -> bool:
    """Whether ``trace`` recorded this node with a forward op and a call in ``pass_index``."""

    grad_fn = (getattr(trace, "grad_fn_logs", {}) or {}).get(object_id)
    if grad_fn is None or not getattr(grad_fn, "has_op", False):
        return False
    calls = getattr(grad_fn, "calls", None) or {}
    values = calls.values() if hasattr(calls, "values") else calls
    return any(getattr(call, "backward_pass_index", None) == pass_index for call in values)


def _checkpoint_evidence(trace: Any) -> str:
    """``"observed"``, ``"none_observed"`` or ``"unknown"`` from the witness."""

    from ._cost_backward import recompute_observation

    return recompute_observation(trace)


def _pass_builds_graph(backward_pass: Any) -> bool | None:
    """Whether the pass ran with ``create_graph`` (``None`` when unrecorded)."""

    flags = getattr(backward_pass, "engine_flags", None)
    if not isinstance(flags, Mapping):
        return None
    return bool(flags.get("create_graph", False))


def _resolve_call_index(grad_fn: Any, ref: tuple[int, int, int], pass_index: int) -> int:
    """The node's call index in this trace for the pass the call ran in.

    The marker carries the call index its pushing trace predicted; when two
    traces share a node the marker may be the other trace's, so the call is
    looked up by pass when that is unambiguous.
    """

    calls = getattr(grad_fn, "calls", None) or {}
    values = calls.values() if hasattr(calls, "values") else calls
    in_pass = [
        int(call.call_index)
        for call in values
        if getattr(call, "backward_pass_index", None) == pass_index
    ]
    if len(in_pass) == 1:
        return in_pass[0]
    return int(ref[1])


def _recompute_share(op: Any, builds_graph: bool | None, checkpoint: str) -> int | None:
    """The recompute part of one counted call: its FLOPs, 0, or unknown."""

    if not op.grad_enabled or checkpoint == "none_observed":
        return 0
    if builds_graph is False and checkpoint == "observed":
        return int(op.flops)
    return None


def pass_failed(backward_pass: Any) -> bool:
    """Whether the engine raised before the pass finished."""

    return str(getattr(backward_pass, "status", "ok") or "ok") != "ok"


def node_resolver_missing() -> bool:
    """Whether torch lacks the executing-node resolver (``HAS_CURRENT_AUTOGRAD_NODE``).

    Without it a recorded call cannot be matched to the node that ran it, so
    the census charges no call to a node and the pass has no executed count.
    """

    return not _torch_compat.HAS_CURRENT_AUTOGRAD_NODE


def _add(left: int | None, right: int | None) -> int | None:
    """None-propagating sum."""

    return None if left is None or right is None else left + right


def _census_executed(
    backward_pass: Any,
    rows: tuple[BackwardFlopsRow, ...],
    unattributed_flops: int | None,
) -> int | None:
    """The pass's executed count: rows plus the unattributed bucket."""

    if pass_failed(backward_pass) or node_resolver_missing():
        # A pass that raised did not run to completion, and without the
        # executing-node resolver no call can be charged to its node: the
        # calls stay as evidence, but the pass has no executed count.
        return None
    executed = unattributed_flops
    for row in rows:
        executed = _add(executed, row.flops)
    return executed


def _census_recompute_state(checkpoint: str, any_unknown: bool) -> str:
    """The pass's recompute state from the witness and the per-call shares."""

    if checkpoint == "none_observed":
        return "none_observed"
    if any_unknown or checkpoint == "unknown":
        return "unknown"
    return "labeled"


def attribute_census(
    trace: Any,
    backward_pass: Any,
    ops: Iterable[Any],
) -> BackwardFlopsAttribution:
    """Attribute one pass's recorded calls to this trace's GradFn calls.

    Parameters
    ----------
    trace
        The trace whose view is built.
    backward_pass
        The pass record (its index and engine flags).
    ops
        The pass's recorded ``DispatchedOp`` calls, in run order.
    """

    pass_index = int(backward_pass.pass_index)
    grad_fn_logs = getattr(trace, "grad_fn_logs", {}) or {}
    builds_graph = _pass_builds_graph(backward_pass)
    checkpoint = _checkpoint_evidence(trace)
    rows: dict[tuple[int, int], _RowAccumulator] = {}
    unattributed: list[tuple[str, int | None]] = []
    unattributed_flops: int | None = 0
    uncounted: dict[str, int] = {}
    other_trace = 0
    any_unknown_recompute = False
    owned_elsewhere = _OwnedElsewhere(trace, pass_index)
    for op in ops:
        ref = op.grad_fn_call_ref
        grad_fn = None if ref is None else grad_fn_logs.get(ref[0])
        if ref is not None and (grad_fn is None or owned_elsewhere(grad_fn, ref[0])):
            other_trace += 1
            continue
        if op.flops is None and op.family == "uncounted":
            uncounted[op.base_name] = uncounted.get(op.base_name, 0) + 1
        if ref is None:
            unattributed.append((op.operator, op.flops))
            if op.family != "uncounted":
                unattributed_flops = _add(unattributed_flops, op.flops)
            continue
        call_index = _resolve_call_index(grad_fn, ref, pass_index)
        key = (int(ref[0]), call_index)
        row = rows.get(key)
        if row is None:
            row = _RowAccumulator(
                key, str(getattr(grad_fn, "label", "")), getattr(grad_fn, "op_label", None)
            )
            rows[key] = row
        if op.family != "uncounted" and row.add(op, builds_graph, checkpoint):
            any_unknown_recompute = True
    built = tuple(
        BackwardFlopsRow(
            pass_index=pass_index,
            grad_fn_label=row.label,
            call_index=row.key[1],
            op_label=None if row.op_label is None else str(row.op_label),
            flops=row.flops,
            recompute_flops=row.recompute,
        )
        for row in rows.values()
    )
    return BackwardFlopsAttribution(
        pass_index=pass_index,
        convention=DISPATCH_COUNTING_CONVENTION,
        executed_flops=_census_executed(backward_pass, built, unattributed_flops),
        rows=built,
        unattributed_flops=unattributed_flops,
        unattributed_ops=tuple(unattributed),
        uncounted_ops=tuple(uncounted.items()),
        recompute=_census_recompute_state(checkpoint, any_unknown_recompute),
        other_trace_ops=other_trace,
        source="census",
    )


def _saved_recompute_state(rows: tuple[BackwardFlopsRow, ...], checkpoint: str) -> str:
    """The recompute state of a saved record: its rows first, then the witness.

    A positive recompute share was labeled at save time, which needed the
    checkpoint witness; all-zero rows defer to the persisted witness.
    """

    shares = [row.recompute_flops for row in rows]
    if any(share is None for share in shares):
        return "unknown"
    if any(shares):
        return "labeled"
    if checkpoint == "none_observed":
        return "none_observed"
    return "labeled" if checkpoint == "observed" else "unknown"


def executed_flops_record(attribution: BackwardFlopsAttribution) -> dict[str, Any]:
    """The closed persisted form of one pass's attribution (ints, None, short strings)."""

    return {
        "convention": _short(attribution.convention),
        "executed_flops": attribution.executed_flops,
        "uncounted_ops": [[_short(name), int(count)] for name, count in attribution.uncounted_ops],
        "rows": [
            [
                _short(row.grad_fn_label),
                int(row.call_index),
                None if row.op_label is None else _short(row.op_label),
                row.flops,
                row.recompute_flops,
            ]
            for row in attribution.rows
        ],
        "unattributed": [[_short(name), flops] for name, flops in attribution.unattributed_ops],
    }


def _short(value: str) -> str:
    """Bound one persisted string."""

    return str(value)[:_RECORD_STRING_LIMIT]


def attribution_from_record(
    trace: Any, pass_index: int, record: Mapping[str, Any]
) -> BackwardFlopsAttribution:
    """Rebuild one pass's attribution from its persisted record (validated at load).

    The recompute state is re-read from the trace's persisted checkpoint
    witness; the count of calls left to other traces is not persisted.
    """

    rows = tuple(
        BackwardFlopsRow(
            pass_index=int(pass_index),
            grad_fn_label=str(label),
            call_index=int(call_index),
            op_label=None if op_label is None else str(op_label),
            flops=flops,
            recompute_flops=recompute,
        )
        for label, call_index, op_label, flops, recompute in record["rows"]
    )
    unattributed = tuple((str(name), flops) for name, flops in record["unattributed"])
    executed = record["executed_flops"]
    # With a known total every None entry is a call outside the counted
    # families; with an unknown total the bucket is unknown too.
    unattributed_flops: int | None = (
        None if executed is None else sum(flops for _name, flops in unattributed if flops)
    )
    recompute = _saved_recompute_state(rows, _checkpoint_evidence(trace))
    return BackwardFlopsAttribution(
        pass_index=int(pass_index),
        convention=str(record["convention"]),
        executed_flops=executed,
        rows=rows,
        unattributed_flops=unattributed_flops,
        unattributed_ops=unattributed,
        uncounted_ops=tuple((str(name), int(count)) for name, count in record["uncounted_ops"]),
        recompute=recompute,
        other_trace_ops=None,
        source="saved record",
    )


def _is_count(value: Any, *, minimum: int = 0) -> bool:
    """A non-boolean int at or above ``minimum``."""

    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_label(value: Any) -> bool:
    """A short string."""

    return isinstance(value, str) and len(value) <= _RECORD_STRING_LIMIT


def _entries_problem(entries: Any, field: str, check: Any, width: int) -> tuple[str, str] | None:
    """Validate one list of fixed-width entries with a per-entry predicate."""

    if not isinstance(entries, (list, tuple)):
        return f"{field} must be a list", f"{field}_type"
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != width or not check(entry):
            return f"{field} entry {entry!r} is malformed", f"{field}_entry"
    return None


def _row_ok(entry: Any) -> bool:
    """Whether one saved row holds a label, a call index and known-or-None counts."""

    label, call_index, op_label, flops, recompute = entry
    return (
        _is_label(label)
        and _is_count(call_index, minimum=1)
        and (op_label is None or _is_label(op_label))
        and (flops is None or _is_count(flops))
        and (recompute is None or _is_count(recompute))
    )


def _record_header_problem(record: Any) -> tuple[str, str] | None:
    """The record's type, key set, convention and executed count."""

    if not isinstance(record, Mapping):
        return "the record must be a mapping or None", "record_type"
    if set(record) != EXECUTED_FLOPS_RECORD_KEYS:
        return (
            f"keys {sorted(map(str, record))} differ from {sorted(EXECUTED_FLOPS_RECORD_KEYS)}",
            "record_keys",
        )
    if not _is_label(record["convention"]):
        return "convention must be a short string", "convention_type"
    executed = record["executed_flops"]
    if executed is not None and not _is_count(executed):
        return f"executed_flops {executed!r} must be a non-negative int or None", "count_type"
    return None


def _record_coherence_problem(record: Mapping[str, Any]) -> tuple[str, str] | None:
    """Whether rows plus unattributed equal a known executed count."""

    executed = record["executed_flops"]
    if executed is None:
        return None
    row_flops = [entry[3] for entry in record["rows"]]
    if any(flops is None for flops in row_flops):
        return "a known executed_flops needs every row's flops", "rows_incoherent"
    bucket = sum(flops for _name, flops in record["unattributed"] if flops)
    if sum(row_flops) + bucket != executed:
        return "rows plus unattributed differ from executed_flops", "rows_incoherent"
    return None


def executed_flops_record_problem(record: Any) -> tuple[str, str] | None:
    """Why a persisted ``executed_flops_record`` is malformed, or ``None`` when sound.

    Returns ``(message, reason)``: the closed key set, ints and short strings
    only, and rows plus unattributed equal to a known executed count.
    """

    problem = _record_header_problem(record)
    if problem is not None:
        return problem
    for field, check, width in (
        ("uncounted_ops", lambda e: _is_label(e[0]) and _is_count(e[1], minimum=1), 2),
        ("rows", _row_ok, 5),
        ("unattributed", lambda e: _is_label(e[0]) and (e[1] is None or _is_count(e[1])), 2),
    ):
        problem = _entries_problem(record[field], field, check, width)
        if problem is not None:
            return problem
    return _record_coherence_problem(record)
