"""Graph lineage, comparison, and activation-analysis helpers."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch

from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    from torchlens._data_classes.op import Op
    from torchlens._data_classes.trace import Trace

from .._debug_report import Report, at_source, table_report, trace_subject
from .._io.payload_codec import get_payload_codec
from ..backends._backend_names import TORCH_BACKEND_NAME
from ._common import (
    _compared_site_ops,
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
from ._compare_guard import refuse_uncomparable_pair
from ._compare_pairing import paired_rows, row_source
from ._compare_values import (
    _activation_row,
    _align_for_compare,
    _compare_coverage,
    _compare_unavailable_reason,
    _value_delta,
)
from ._cross_backend import resolve_align
from ._unit_axis import unit_axis
from ._unit_axis_layout import backend_unit_axis
from ._xb_report import LABEL_COLUMNS, module_compare, module_first_divergence
from ._xb_tolerance import SAME_BACKEND_ATOL, SAME_BACKEND_RTOL

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


def _compare_rows(
    trace_a: Trace,
    trace_b: Trace,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    by_label: bool = False,
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
    by_label:
        Pair ops by label even when the graph fingerprints differ (``align="label"``).

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

    refuse_uncomparable_pair(trace_a, trace_b)
    ops_a = _compared_site_ops(trace_a)
    ops_b = _compared_site_ops(trace_b)
    # EXECUTION order, never sorted: first_divergence reads the first row (see Returns).
    rows: list[dict[str, Any]] = []
    summary = {
        "matched": 0,
        "shape_mismatch": 0,
        "only_a": 0,
        "only_b": 0,
        "value_diverged": 0,
        "activation_unavailable": 0,
    }

    pairs = paired_rows(trace_a, trace_b, ops_a, ops_b, by_label=by_label)
    for label, status, op_a, op_b, note in pairs:
        if status != "present-in-both":
            summary[status.replace("-", "_")] += 1
            rows.append(_activation_row(label, status, note, _source_line(row_source(op_a, op_b))))
            continue

        out_a, reason_a = _safe_out(op_a, get_payload_codec(str(trace_a._provenance_backend)))
        out_b, reason_b = _safe_out(op_b, get_payload_codec(str(trace_b._provenance_backend)))
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
            "reason": note,
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
        cmp_a, cmp_b, aligned_note = aligned
        row["reason"] = "; ".join(filter(None, (note, aligned_note)))

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
    rtol: float | None = None,
    atol: float | None = None,
    align: str = "auto",
    module_map: Mapping[str, str] | None = None,
) -> Report:
    """Compare saved activations across two traces, op by op or module by module.

    Parameters
    ----------
    trace_a, trace_b:
        Completed traces; ``trace_a`` is the reference run.
    rtol, atol:
        Tolerance overrides. One backend: ``torch.allclose``, default ``1e-5`` / ``1e-8``.
        Across backends: ``rms(a - b) <= atol + rtol * rms(a)`` per module output, default
        from the narrower dtype (float32 ``1e-4``, float16 ``5e-2``, bfloat16 ``0.75``; atol 0,
        or ``c * eps`` times the smallest input rms on an all-zero reference); overridable.
    align:
        ``"auto"`` on one backend pairs ops by label when the graph fingerprints
        (``trace.provenance.fingerprint``) are equal, else by graph structure, never by value
        (an inserted op is unpaired instead of shifting every later pair); across backends it
        is ``"structure"``. ``"label"`` always joins ops by label (one backend);
        ``"structure"`` pairs module calls by address and call order.
    module_map:
        Side-a to side-b module address prefixes for renamed ports; mapped pairs are
        ``"asserted"``.

    Returns
    -------
    Report
        Tabular ``tl.debug`` report; ``.to_pandas()`` returns the DataFrame described here.
        Op pairing: one row per pass-qualified op with summary counts in ``df.attrs``;
        ``source_line`` is the ``"file:line"`` of the user code that ran the op. Activations on
        different devices compare on the CPU, different dtypes in the wider dtype (``reason``
        and the coverage line say so); integer and boolean activations must match exactly. The
        coverage line counts the ops that could not be compared and why; ops whose structural
        twins differ in number stay unpaired (``ambiguous``). Module pairing adds per-row
        ``module_b``, ``alignment``, ``outcome``, ``rtol``, ``atol``, ``transform`` and
        ``inputs_agree``, ``df.attrs["coverage"]`` and a coverage line for both sides. A label
        join across backends, or a backend whose module recording is not admitted, raises
        :class:`~torchlens.errors.BackendUnsupportedError` (code ``backend_unsupported``).

    Examples
    --------
    >>> # requires: pandas
    >>> first, second = tl.trace(model, x), tl.trace(model, x)
    >>> frame = compare(first, second).to_pandas()
    >>> frame.attrs["matched"] > 0, frame.attrs["value_diverged"]
    (True, 0)
    """

    if resolve_align(trace_a, trace_b, align, module_map) == "structure":
        return module_compare(trace_a, trace_b, rtol=rtol, atol=atol, module_map=module_map)
    rows, attrs = _compare_rows(
        trace_a, trace_b, by_label=align == "label", **_label_tolerance(rtol, atol)
    )
    pd = _require_pandas()
    frame = pd.DataFrame(rows, columns=list(LABEL_COLUMNS))
    frame.attrs.update(attrs)
    return table_report(
        "compare",
        frame,
        subject=f"{trace_subject(trace_a)} vs {trace_subject(trace_b)}",
        finding_columns=("op", "status", "max_abs", "reason", "source_line"),
        coverage=_compare_coverage(rows),
    )


def _label_tolerance(rtol: float | None, atol: float | None) -> dict[str, float]:
    """The label join's ``torch.allclose`` tolerances, defaults filled in."""

    return {
        "rtol": SAME_BACKEND_RTOL if rtol is None else rtol,
        "atol": SAME_BACKEND_ATOL if atol is None else atol,
    }


def first_divergence(
    trace_a: Trace,
    trace_b: Trace,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    align: str = "auto",
    module_map: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Return the first row where the two traces separate, or ``None``.

    Op pairing (one backend): the first EXECUTION-ORDER row whose payloads crossed tolerance
    (``allclose=False``), mismatched shape, or exist in only one trace (an inserted op is
    reported as itself). Rows whose payloads could not be compared (unsaved, non-tensor,
    sparse, meta, quantized) are not divergences.

    Module pairing: the first row in module-exit order whose ``outcome`` is ``"different"`` on a ``"supported"`` or ``"asserted"`` pair; a
    module with no partner is a coverage gap, never a divergence. Its ``reason`` says
    "localized inside this module" when the module's inputs agree, else "first observed
    differing module output" with the computation skipped since the last agreeing module.
    ``None`` only when every paired module agrees. Refuses with
    :class:`~torchlens.errors.InvalidArgumentError` (code ``module_comparison_inconclusive``)
    when a side has no module boundaries, the model inputs differ, nothing could be
    compared, or a paired module could not be decided.

    Parameters
    ----------
    trace_a, trace_b:
        Completed traces; ``trace_a`` is the reference run.
    rtol, atol:
        Tolerance overrides. One backend: ``torch.allclose``, default ``1e-5`` / ``1e-8``.
        Across backends: ``rms(a - b) <= atol + rtol * rms(a)`` per module output, default
        from the narrower dtype (float32 ``1e-4``, float16 ``5e-2``, bfloat16 ``0.75``; atol 0,
        or ``c * eps`` times the smallest input rms on an all-zero reference); overridable.
    align:
        ``"auto"`` on one backend pairs ops by label when the graph fingerprints
        (``trace.provenance.fingerprint``) are equal, else by graph structure, never by value
        (an inserted op is unpaired instead of shifting every later pair); across backends it
        is ``"structure"``. ``"label"`` always joins ops by label (one backend);
        ``"structure"`` pairs module calls by address and call order.
    module_map:
        Side-a to side-b module address prefixes for renamed ports; mapped pairs are
        ``"asserted"``.

    Returns
    -------
    dict[str, Any] | None
        The first diverging row, or ``None``.

    Examples
    --------
    >>> first, shifted = tl.trace(model, x), tl.trace(model, x + 1)
    >>> first_divergence(first, shifted) is not None
    True
    >>> first_divergence(first, first) is None
    True
    """

    if resolve_align(trace_a, trace_b, align, module_map) == "structure":
        return module_first_divergence(
            trace_a, trace_b, rtol=rtol, atol=atol, module_map=module_map
        )
    rows, _attrs = _compare_rows(
        trace_a, trace_b, by_label=align == "label", **_label_tolerance(rtol, atol)
    )
    for row in rows:
        if row["status"] in ("only-a", "only-b") or False in (row["shape_match"], row["allclose"]):
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
        module = _output_module(op)
        row_key = {"op": _op_label(op), "module": module, "source_line": _source_line(op)}
        out, reason = _safe_out(op, get_payload_codec(str(trace._provenance_backend)))
        reason = reason or _tensor_unavailable_reason(out)
        if reason is None and not isinstance(out, torch.Tensor):
            skipped += 1
            continue
        feature_dim = None if reason else _feature_dim(trace, op, out, dim, axes)
        if feature_dim is None or not isinstance(out, torch.Tensor):
            skipped += 1
            rows.append(
                {
                    **row_key,
                    "total_units": None,
                    "dead_count": None,
                    "dead_frac": None,
                    "sample_dead_idx": [],
                    "reason": reason or _unit_axis_reason(trace, out, dim),
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
                **row_key,
                "total_units": total_units,
                "dead_count": dead_count,
                "dead_frac": 0.0 if total_units == 0 else dead_count / total_units,
                "sample_dead_idx": dead_indices[:10],
                "reason": "",
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
        if str(trace._provenance_backend) != TORCH_BACKEND_NAME and out.ndim > 2:
            return backend_unit_axis(trace, op, out, axes)
        return unit_axis(trace, op, out, axes)
    feature_dim = dim if dim >= 0 else out.ndim + dim
    return feature_dim if 0 <= feature_dim < out.ndim else None


def _unit_axis_reason(trace: Trace, out: Any, dim: int | None) -> str:
    """Explain spatial preview rows whose unit layout the trace does not specify."""

    if dim is None and isinstance(out, torch.Tensor) and out.ndim > 2:
        if str(trace._provenance_backend) != TORCH_BACKEND_NAME:
            return "unit axis unavailable for preview spatial output; pass dim explicitly"
    return "invalid-dim"


def _output_module(op: Op) -> str | None:
    """Return the innermost module address whose output ``op`` is, or ``None``.

    Parameters
    ----------
    op:
        TorchLens op.

    Returns
    -------
    str | None
        The deepest address in ``Op.output_of_modules``, or ``None`` when the op
        is no module's output.
    """

    addresses = list(record_member(op, "output_of_modules", None) or ())
    return max(addresses, key=lambda address: address.count("."), default=None)


#: Columns of the ``dead_neurons`` table, in order.
_DEAD_NEURON_COLUMNS: tuple[str, ...] = (
    "op",
    "module",
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
        Columns are ``op``, ``module`` (the innermost module whose output the
        op is, or ``None``), ``total_units``, ``dead_count``, ``dead_frac``,
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
