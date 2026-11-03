"""Graph lineage, comparison, and activation-analysis helpers."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from torchlens._data_classes.op import Op
    from torchlens._data_classes.trace import Trace

from .._debug_report import Report, at_source, table_report, trace_subject
from .._utils._record_group_table import record_member
from ._common import (
    _compute_ops,
    _op_from_label,
    _op_label,
    _require_pandas,
    _resolve_op,
    _safe_out,
    _shape_dtype,
    _source_line,
    _tensor_unavailable_reason,
)
from ._unit_axis import unit_axis

LineageDirection = Literal["ancestors", "descendants", "both"]


@dataclass(frozen=True)
class LineageReport(Report):
    """Result returned by :func:`lineage`.

    Parameters
    ----------
    start_label:
        Resolved starting op label, or the requested label when lookup failed.
    direction:
        Traversal direction.
    nodes:
        Tuples of ``(label, depth, source_line, shape, dtype)``.
    message:
        Human-readable status.

    Examples
    --------
    >>> run = tl.trace(model, x)
    >>> report = tl.debug.lineage(run, "conv2d_2_3")
    >>> isinstance(report, LineageReport), report.direction
    (True, 'ancestors')
    >>> len(report.nodes) > 1
    True
    """

    start_label: str | None
    direction: str
    nodes: list[tuple[str, int, str | None, tuple[int, ...] | None, str | None]]
    message: str

    def _report_subject(self) -> str:
        """What this report examined (the card's subject)."""

        return f"{self.start_label} ({self.direction})"

    def _report_status(self) -> str:
        """The verdict token on the card's lead line."""

        return f"{len(self.nodes)} nodes" if self.nodes else "empty"

    def _report_findings(self) -> list[str]:
        """One-line findings for the card, most severe first."""

        findings = [self.message] if self.message else []
        findings.extend(f"{node[0]} (depth {node[1]}){at_source(node[2])}" for node in self.nodes)
        return findings

    def _report_rows(self) -> list[dict[str, Any]] | None:
        """The natural table rows behind ``to_pandas()``.

        Columns follow the node tuple: ``label``, ``depth``, ``source_line``,
        ``shape``, ``dtype``.
        """

        names = ("label", "depth", "source_line", "shape", "dtype")
        return [dict(zip(names, node, strict=False)) for node in self.nodes]


def lineage(
    trace: Trace,
    op_or_label: Any,
    *,
    direction: str = "ancestors",
    max_depth: int | None = None,
) -> LineageReport:
    """Walk graph lineage for a completed trace.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.
    op_or_label:
        Start op or lookup key accepted by ``trace.__getitem__``.
    direction:
        ``"ancestors"``, ``"descendants"``, or ``"both"``.
    max_depth:
        Optional maximum graph distance from the start op.

    Returns
    -------
    LineageReport
        Traversal result with unavailable lookups reported in ``message``.

    Examples
    --------
    >>> run = tl.trace(model, x)
    >>> report = lineage(run, "conv2d_2_3", max_depth=1)
    >>> report.start_label.startswith("conv2d_2_3"), len(report.nodes)
    (True, 2)
    >>> len(lineage(run, "conv2d_1_1", direction="descendants").nodes) > 1
    True
    """

    start_op, error = _resolve_op(trace, op_or_label)
    if direction not in {"ancestors", "descendants", "both"}:
        requested = str(getattr(op_or_label, "label", op_or_label))
        return LineageReport(
            start_label=requested,
            direction=direction,
            nodes=[],
            message="unavailable: direction must be 'ancestors', 'descendants', or 'both'",
        )
    if start_op is None:
        return LineageReport(
            start_label=str(op_or_label),
            direction=direction,
            nodes=[],
            message=error or "unavailable",
        )

    def neighbor_labels(op: Op) -> Iterable[str]:
        """Return graph-neighbor labels for the requested direction."""

        if direction == "ancestors":
            return getattr(op, "parents", ()) or ()
        if direction == "descendants":
            return getattr(op, "children", ()) or ()
        return tuple(getattr(op, "parents", ()) or ()) + tuple(getattr(op, "children", ()) or ())

    start_label = _op_label(start_op)
    # deque: list.pop(0) shifts the whole queue per visit, Theta(V^2) on
    # wide graphs for a linear BFS (R52, 9.3x measured on the FIFO class).
    queue: deque[tuple[Op, int]] = deque([(start_op, 0)])
    visited = {start_label}
    nodes: list[tuple[str, int, str | None, tuple[int, ...] | None, str | None]] = []
    while queue:
        op, depth = queue.popleft()
        shape, dtype = _shape_dtype(op)
        nodes.append((_op_label(op), depth, _source_line(op), shape, dtype))
        if max_depth is not None and depth >= max_depth:
            continue
        for label in neighbor_labels(op):
            child_op = _op_from_label(trace, str(label))
            visit_label = _op_label(child_op) if child_op is not None else str(label)
            if visit_label in visited:
                continue
            visited.add(visit_label)
            if child_op is None:
                nodes.append((visit_label, depth + 1, None, None, None))
                continue
            queue.append((child_op, depth + 1))

    return LineageReport(
        start_label=start_label,
        direction=direction,
        nodes=nodes,
        message=f"{len(nodes)} node(s)",
    )


def _activation_row(
    label: str, status: str, reason: str, source_line: str | None = None
) -> dict[str, Any]:
    """Build a minimal activation-unavailable comparison row.

    Parameters
    ----------
    label:
        Op label.
    status:
        Presence status.
    reason:
        Unavailable reason.
    source_line:
        ``"file:line"`` of the user code that ran the op, when known.

    Returns
    -------
    dict[str, Any]
        Row dictionary.
    """

    return {
        "op": label,
        "status": status,
        "shape_match": None,
        "dtype_match": None,
        "max_abs": None,
        "mean_abs": None,
        "allclose": None,
        "reason": reason,
        "source_line": source_line,
    }


def _align_for_compare(
    out_a: torch.Tensor, out_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, str] | None:
    """Bring two activations onto one device and dtype so they can be compared.

    A CPU run against a GPU run, or a float32 run against a float64 run, is
    exactly the comparison a user wants to see; the values are moved to the
    CPU and compared in the wider dtype (``torch.promote_types``). Only a
    shape mismatch truly cannot be compared.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, str] | None
        The aligned pair and a note saying what was aligned (``""`` when
        nothing was), or ``None`` when the shapes differ.
    """

    if out_a.shape != out_b.shape:
        return None
    notes = []
    a, b = out_a.detach(), out_b.detach()
    if a.device != b.device:
        notes.append(f"devices differ ({a.device} vs {b.device}): compared on cpu")
        a, b = a.cpu(), b.cpu()
    if a.dtype != b.dtype:
        common = torch.promote_types(a.dtype, b.dtype)
        notes.append(
            f"dtypes differ ({str(a.dtype).removeprefix('torch.')} vs "
            f"{str(b.dtype).removeprefix('torch.')}): compared in "
            f"{str(common).removeprefix('torch.')}"
        )
        a, b = a.to(common), b.to(common)
    return a, b, "; ".join(notes)


def _compare_unavailable_reason(value: Any) -> str | None:
    """Return why a payload cannot be value-compared, or ``None``.

    Wider than :func:`_tensor_unavailable_reason`: integer, boolean and complex
    tensors compare too (integers and booleans exactly), so an index or mask op
    where two runs part ways is reported instead of skipped.
    """

    reason = _tensor_unavailable_reason(value)
    if reason in ("non-floating", "complex"):
        return None
    return reason


def _value_delta(
    cmp_a: torch.Tensor, cmp_b: torch.Tensor, *, rtol: float, atol: float
) -> tuple[float, float, bool]:
    """Return ``(max_abs, mean_abs, allclose)`` for an aligned pair.

    Floating and complex payloads use ``torch.allclose``; integer and boolean
    payloads must match exactly (a tolerance on an index is meaningless).
    """

    exact = not (torch.is_floating_point(cmp_a) or torch.is_complex(cmp_a))
    wide = torch.complex128 if torch.is_complex(cmp_a) else torch.float64
    delta = torch.abs(cmp_a.to(wide) - cmp_b.to(wide)) if exact else torch.abs(cmp_a - cmp_b)
    max_abs = float(delta.max().item()) if delta.numel() else 0.0
    mean_abs = float(delta.mean().item()) if delta.numel() else 0.0
    if exact:
        return max_abs, mean_abs, bool(torch.equal(cmp_a, cmp_b))
    return max_abs, mean_abs, bool(torch.allclose(cmp_a, cmp_b, rtol=rtol, atol=atol))


def _compare_coverage(rows: list[dict[str, Any]]) -> str:
    """Return the compare card's coverage line: compared vs skipped, and why."""

    compared = sum(1 for row in rows if row["allclose"] is not None)
    aligned = sum(1 for row in rows if row["allclose"] is not None and row["reason"])
    skipped: dict[str, int] = {}
    for row in rows:
        if row["allclose"] is None:
            reason = str(row["reason"] or row["status"])
            skipped[reason] = skipped.get(reason, 0) + 1
    text = f"{compared} of {len(rows)} ops compared"
    if aligned:
        notes = sorted(
            {
                str(note)
                for row in rows
                if row["allclose"] is not None and row["reason"]
                for note in str(row["reason"]).split("; ")
            }
        )
        text += f" ({aligned} after alignment: {'; '.join(notes)})"
    if skipped:
        detail = ", ".join(f"{reason}: {count}" for reason, count in skipped.items())
        text += f"; {sum(skipped.values())} not compared ({detail})"
    return text


def _compare_rows(
    trace_a: Trace,
    trace_b: Trace,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pandas-free core of :func:`compare` (agent stage-0 item 4).

    The wire path for agent/MCP consumers: plain row dicts plus the attrs
    mapping, importable and runnable without pandas. :func:`compare` is the
    optional DataFrame view over this core.

    Parameters
    ----------
    trace_a:
        First completed TorchLens trace.
    trace_b:
        Second completed TorchLens trace.
    rtol:
        Relative tolerance for ``torch.allclose``.
    atol:
        Absolute tolerance for ``torch.allclose``.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, Any]]
        One row dict per pass-qualified op IN EXECUTION ORDER (trace_a's
        order, with trace_b-only ops appended in trace_b's order), and the
        summary attrs (counts, tolerances, per-trace capture-honesty facts).
        The first row with ``allclose=False`` is therefore the first
        divergence the forward actually executed; :func:`first_divergence`
        returns it directly.
    """

    from torchlens._runnable import refuse_poisoned_trace

    refuse_poisoned_trace(trace_a, "faithful comparison")
    refuse_poisoned_trace(trace_b, "faithful comparison")
    ops_a = {_op_label(op): op for op in _compute_ops(trace_a)}
    ops_b = {_op_label(op): op for op in _compute_ops(trace_b)}
    # EXECUTION order, never alphabetical (observe memo item 2): the published
    # first-diverging-op recipe reads the first row, and a sorted() walk named
    # whichever op sorts first (op 67 of 70 on resnet18). Rows follow trace_a's
    # execution order; ops only present in trace_b append afterwards in
    # trace_b's execution order (disclosed by their "only-b" status).
    labels = list(ops_a) + [label for label in ops_b if label not in ops_a]
    rows: list[dict[str, Any]] = []
    summary = {
        "matched": 0,
        "shape_mismatch": 0,
        "only_a": 0,
        "only_b": 0,
        "value_diverged": 0,
        "activation_unavailable": 0,
    }

    for label in labels:
        op_a = ops_a.get(label)
        op_b = ops_b.get(label)
        if op_a is None:
            summary["only_b"] += 1
            rows.append(_activation_row(label, "only-b", "only-b", _source_line(ops_b[label])))
            continue
        if op_b is None:
            summary["only_a"] += 1
            rows.append(_activation_row(label, "only-a", "only-a", _source_line(op_a)))
            continue

        out_a, reason_a = _safe_out(op_a)
        out_b, reason_b = _safe_out(op_b)
        reason_a = reason_a or _compare_unavailable_reason(out_a)
        reason_b = reason_b or _compare_unavailable_reason(out_b)
        shape_match = getattr(op_a, "shape", None) == getattr(op_b, "shape", None)
        dtype_match = getattr(op_a, "dtype", None) == getattr(op_b, "dtype", None)
        row = {
            "op": label,
            "status": "present-in-both",
            "shape_match": shape_match,
            "dtype_match": dtype_match,
            "max_abs": None,
            "mean_abs": None,
            "allclose": None,
            "reason": "",
            "source_line": _source_line(op_a) or _source_line(op_b),
        }
        if not shape_match:
            summary["shape_mismatch"] += 1
        if reason_a is not None or reason_b is not None:
            summary["activation_unavailable"] += 1
            row["reason"] = f"a={reason_a or 'ok'}; b={reason_b or 'ok'}"
            rows.append(row)
            continue
        if not isinstance(out_a, torch.Tensor) or not isinstance(out_b, torch.Tensor):
            summary["activation_unavailable"] += 1
            row["reason"] = "non-tensor/container"
            rows.append(row)
            continue
        aligned = _align_for_compare(out_a, out_b)
        if aligned is None:
            summary["activation_unavailable"] += 1
            row["reason"] = "shape-mismatch"
            rows.append(row)
            continue
        cmp_a, cmp_b, row["reason"] = aligned

        max_abs, mean_abs, allclose = _value_delta(cmp_a, cmp_b, rtol=rtol, atol=atol)
        if allclose:
            summary["matched"] += 1
        else:
            summary["value_diverged"] += 1
        row.update({"max_abs": max_abs, "mean_abs": mean_abs, "allclose": allclose})
        rows.append(row)

    from .._capture_honesty import capture_honesty_facts

    attrs: dict[str, Any] = {
        **summary,
        "rtol": rtol,
        "atol": atol,
        "torchlens_capture_honesty": {
            "trace_a": capture_honesty_facts(trace_a),
            "trace_b": capture_honesty_facts(trace_b),
        },
    }
    return rows, attrs


def compare(
    trace_a: Trace,
    trace_b: Trace,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Report:
    """Compare saved tensor activations across two traces, op by op.

    Parameters
    ----------
    trace_a:
        First completed TorchLens trace.
    trace_b:
        Second completed TorchLens trace.
    rtol:
        Relative tolerance for ``torch.allclose``.
    atol:
        Absolute tolerance for ``torch.allclose``.

    Returns
    -------
    Report
        Tabular ``tl.debug`` report; ``.to_pandas()`` returns the
        DataFrame described here.
        One row per pass-qualified op with summary counts in ``df.attrs``;
        ``source_line`` is the ``"file:line"`` of the user code that ran the
        op. Activations on different devices are compared on the CPU, and
        different dtypes in the wider dtype (the row's ``reason`` says so, and
        the card's coverage line names every alignment); integer and boolean
        activations must match exactly. The coverage line also counts the ops
        that could not be compared and why. The optional view over :func:`_compare_rows` (the
        pandas-free core).

    Examples
    --------
    >>> # requires: pandas
    >>> first, second = tl.trace(model, x), tl.trace(model, x)
    >>> frame = compare(first, second).to_pandas()
    >>> frame.attrs["matched"] > 0, frame.attrs["value_diverged"]
    (True, 0)
    """

    rows, attrs = _compare_rows(trace_a, trace_b, rtol=rtol, atol=atol)
    pd = _require_pandas()
    frame = pd.DataFrame(
        rows,
        columns=[
            "op",
            "status",
            "shape_match",
            "dtype_match",
            "max_abs",
            "mean_abs",
            "allclose",
            "reason",
            "source_line",
        ],
    )
    frame.attrs.update(attrs)
    return table_report(
        "compare",
        frame,
        subject=f"{trace_subject(trace_a)} vs {trace_subject(trace_b)}",
        finding_columns=("op", "status", "max_abs", "reason", "source_line"),
        coverage=_compare_coverage(rows),
    )


def first_divergence(
    trace_a: Trace,
    trace_b: Trace,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> dict[str, Any] | None:
    """Return the first EXECUTION-ORDER row where the two traces separate.

    A divergence row is one whose compared payloads crossed tolerance
    (``allclose=False``), mismatched shape, or exists in only one trace.
    Payloads on different devices (a CPU run against a GPU run) are compared
    on the CPU, and different dtypes in the wider dtype, so those runs
    diverge where their values really separate; integer and boolean payloads
    (indices, masks) must match exactly. Rows whose payloads could not be
    compared at all (unsaved, non-tensor, sparse, meta, quantized) are NOT
    divergences; ``print(tl.debug.compare(trace_a, trace_b))`` counts them
    on its coverage line, and a divergence inside such a region cannot be
    excluded. The row's ``source_line`` is the ``"file:line"`` of the user
    code that ran the op.

    Parameters
    ----------
    trace_a:
        First completed TorchLens trace.
    trace_b:
        Second completed TorchLens trace.
    rtol:
        Relative tolerance for ``torch.allclose``.
    atol:
        Absolute tolerance for ``torch.allclose``.

    Returns
    -------
    dict[str, Any] | None
        The first diverging row in execution order, or ``None`` when no
        compared row diverged.

    Examples
    --------
    >>> first, shifted = tl.trace(model, x), tl.trace(model, x + 1)
    >>> first_divergence(first, shifted) is not None
    True
    >>> first_divergence(first, first) is None
    True
    """

    rows, _attrs = _compare_rows(trace_a, trace_b, rtol=rtol, atol=atol)
    for row in rows:
        if row["status"] in ("only-a", "only-b"):
            return row
        if row["shape_match"] is False:
            return row
        if row["allclose"] is False:
            return row
    return None


def dead_neurons_rows(
    trace: Trace, *, dim: int | None = None, max_activation: float = 0.0
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pandas-free core of :func:`dead_neurons` (agent stage-0 item 4).

    Parameters
    ----------
    trace:
        Completed TorchLens trace.
    dim:
        Unit axis forced on every op, or ``None`` to pick it per op (see
        :func:`dead_neurons`).
    max_activation:
        Largest output magnitude a dead unit may reach at any position.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, Any]]
        One row per layer-output op (see :func:`_layer_output_ops`) and the
        attrs mapping (skip count, parameters, the insufficient-sample note,
        capture-honesty facts).
    """

    rows: list[dict[str, Any]] = []
    skipped = 0
    axes: dict[str, int] = {}
    for op in _layer_output_ops(trace):
        label = _op_label(op)
        source_line = _source_line(op)
        out, reason = _safe_out(op)
        reason = reason or _tensor_unavailable_reason(out)
        if reason is None and not isinstance(out, torch.Tensor):
            skipped += 1
            continue
        feature_dim = None if reason else _feature_dim(trace, op, out, dim, axes)
        if feature_dim is None or not isinstance(out, torch.Tensor):
            skipped += 1
            rows.append(
                {
                    "op": label,
                    "total_units": None,
                    "dead_count": None,
                    "dead_frac": None,
                    "sample_dead_idx": [],
                    "reason": reason or "invalid-dim",
                    "source_line": source_line,
                }
            )
            continue
        reduce_dims = tuple(index for index in range(out.ndim) if index != feature_dim)
        magnitude = out.detach().abs()
        peak_by_unit = magnitude.amax(dim=reduce_dims) if reduce_dims else magnitude
        dead_mask = peak_by_unit <= max_activation
        dead_indices = torch.nonzero(dead_mask, as_tuple=False).flatten().tolist()
        total_units = int(out.shape[feature_dim])
        dead_count = len(dead_indices)
        rows.append(
            {
                "op": label,
                "total_units": total_units,
                "dead_count": dead_count,
                "dead_frac": 0.0 if total_units == 0 else dead_count / total_units,
                "sample_dead_idx": dead_indices[:10],
                "reason": "",
                "source_line": source_line,
            }
        )

    from .._capture_honesty import capture_honesty_facts

    attrs: dict[str, Any] = {
        "skipped": skipped,
        "max_activation": max_activation,
        "dim": dim,
        "note": "one trace is an insufficient sample for dataset-level death",
        "torchlens_capture_honesty": capture_honesty_facts(trace),
    }
    return rows, attrs


def _layer_output_ops(trace: Trace) -> list[Op]:
    """Return the compute ops whose output is a layer's output.

    A layer is a module call or the model itself, so the ops are those
    flagged ``is_module_output`` (the tensor a module call returned) or
    ``is_output_parent`` (a tensor the model's ``forward`` returned). The ops
    inside a composite activation (the ``tanh`` of a tanh-approximated GELU,
    the ``sigmoid`` of ``x * sigmoid(x)``) are not layer outputs and get no
    row: their saturation says nothing about whether the layer's units fire.
    The trace's output boundary nodes (``is_output``) are clones of the
    returned ops and are left out, so no tensor is counted twice.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.

    Returns
    -------
    list[Op]
        Layer-output ops in execution order.
    """

    return [
        op
        for op in _compute_ops(trace)
        if not bool(getattr(op, "is_output", False))
        and (
            bool(record_member(op, "is_module_output", False))
            or bool(record_member(op, "is_output_parent", False))
        )
    ]


def _feature_dim(
    trace: Trace, op: Op, out: Any, dim: int | None, axes: dict[str, int]
) -> int | None:
    """Return the unit axis for one activation tensor, or ``None`` when it is invalid."""

    if dim is None:
        return unit_axis(trace, op, out, axes)
    feature_dim = dim if dim >= 0 else out.ndim + dim
    return feature_dim if 0 <= feature_dim < out.ndim else None


#: Columns of the ``dead_neurons`` table, in order.
_DEAD_NEURON_COLUMNS: tuple[str, ...] = (
    "op",
    "total_units",
    "dead_count",
    "dead_frac",
    "sample_dead_idx",
    "reason",
    "source_line",
)


def dead_neurons(trace: Trace, *, dim: int | None = None, max_activation: float = 0.0) -> Report:
    """Find the dead units of every layer in one completed trace.

    A unit is dead when its output is identically zero at every position in
    the batch: every element of the unit's slice of the activation has a
    magnitude at or below ``max_activation`` (``0.0`` by default, so exactly
    zero). The units are a layer's OUTPUT units: one row per op that a module
    call returned (``Op.is_module_output``) or that the model's ``forward``
    returned (``Op.is_output_parent``). The ops inside a composite activation
    (the ``tanh`` of a tanh-approximated GELU, the ``sigmoid`` of SiLU) get no
    row: a saturated inner intermediate is not a dead output unit.

    A single trace is a single example: a unit that is zero on this input may
    fire on another. Aggregate multiple traces when deciding whether units
    are dead over a dataset.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.
    dim:
        Unit axis. ``None`` (the default) picks it per op: the last axis of a
        two- or three-dimensional activation (a linear layer's ``(batch,
        features)`` or a transformer block's ``(batch, tokens, features)``),
        the channel axis of a convolution, pooling or per-channel norm and of
        the same-shaped ops it feeds (an activation after a convolution), and
        axis 1 for four or more dimensions. Activations under two dimensions
        get ``reason="invalid-dim"``. An int forces that axis on every op.
    max_activation:
        Largest output magnitude a dead unit may reach at any position:
        absolute, on the layer's output, ``0.0`` by default.

    Returns
    -------
    Report
        Tabular ``tl.debug`` report; ``.to_pandas()`` returns the
        DataFrame described here.
        Columns are ``op``, ``total_units``, ``dead_count``, ``dead_frac``,
        ``sample_dead_idx``, ``reason``, and ``source_line`` (``"file:line"``
        of the user code that ran the op). The optional view over
        :func:`dead_neurons_rows` (the pandas-free core).

    Examples
    --------
    >>> # requires: pandas
    >>> frame = dead_neurons(tl.trace(model, x), dim=1).to_pandas()
    >>> {"op", "dead_count"}.issubset(frame.columns)
    True
    """

    rows, attrs = dead_neurons_rows(trace, dim=dim, max_activation=max_activation)
    pd = _require_pandas()
    frame = pd.DataFrame(rows, columns=list(_DEAD_NEURON_COLUMNS))
    frame.attrs.update(attrs)
    return table_report(
        "dead_neurons",
        frame,
        subject=trace_subject(trace),
        finding_columns=("op", "dead_count", "total_units", "dead_frac", "source_line"),
    )
