"""Module-level comparison of two traces, written for any backend pair.

The core behind ``tl.debug.compare`` and ``tl.debug.first_divergence`` when the two
runs are paired module by module (``align="structure"``, or ``align="auto"`` on two
traces from different backends). It follows one evidence rule: every row carries how
its pair was established (``alignment``) apart from what the values showed
(``outcome``), a value is called ``different`` only on a row whose pairing was made
without looking at values (``supported`` or ``asserted``), and a call with no partner
is a coverage gap, never a divergence. Rows follow side a's module-exit order, so the
first ``different`` row is the innermost module whose output separates, and a later
module that agrees again never hides it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from torchlens._runnable import refuse_poisoned_trace

from ._common import _op_from_label, _op_label, _ordered_ops, _source_line
from ._xb_calls import CallInfo, read_calls
from ._xb_pairing import ABSENT, ASSERTED, SUPPORTED, CallPair, pair_calls
from ._xb_refusals import refuse_align, refuse_backend
from ._xb_values import (
    DIFFERENT,
    EQUAL,
    NOT_EVALUATED,
    SHAPE_INCOMPATIBLE,
    UNAVAILABLE,
    ValueOptions,
    ValueResult,
    combine,
    compare_leaf,
    drew_random,
    input_scale,
    layout_justified,
    read_values,
)

if TYPE_CHECKING:
    from torchlens._data_classes.trace import Trace

#: Backends whose module recording passed the conformance cell. A backend joins
#: when its recorded module outputs pass the same checks (identity, tuple and nested
#: returns, duplicate leaves, returned inputs, repeated calls, save and load).
ADMITTED_BACKENDS: frozenset[str] = frozenset({"torch", "mlx"})
ALIGN_MODES = ("auto", "label", "structure")
MODULE_COLUMNS = ("module_b", "alignment", "outcome", "rtol", "atol", "transform", "inputs_agree")
_PAIRED = (SUPPORTED, ASSERTED)


def resolve_align(
    trace_a: Trace, trace_b: Trace, align: str, module_map: Mapping[str, str] | None
) -> str:
    """Return ``"label"`` or ``"structure"`` for a pair, refusing what cannot run.

    ``"label"`` means op pairing: ``"auto"`` on one backend (fingerprint label join, else
    graph-structure pairing) and ``"label"`` (always a label join); the caller tells them
    apart. ``"auto"`` across backends, or with a ``module_map``, pairs modules. ``"label"``
    across backends is refused later by the label path's own guard.
    """

    if align not in ALIGN_MODES:
        refuse_align(
            f"align={align!r} is not a pairing mode",
            remedy='pass align="auto", "label" or "structure"',
        )
    if module_map is not None and not (
        isinstance(module_map, Mapping)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in module_map.items())
    ):
        refuse_align(
            "module_map= must map side-a module address prefixes to side-b prefixes (str to str)",
            remedy='pass a dict such as {"blocks": "layers"}',
        )
    backend_a, backend_b = str(trace_a._provenance_backend), str(trace_b._provenance_backend)
    if align == "label" or (align == "auto" and backend_a == backend_b and not module_map):
        if module_map:
            refuse_align(
                'module_map= pairs module calls, but align="label" joins operations by label',
                remedy='pass align="structure" (or "auto") with module_map=',
            )
        return "label"
    for backend in (backend_a, backend_b):
        if backend not in ADMITTED_BACKENDS:
            refuse_backend(backend_a, backend_b, ADMITTED_BACKENDS)
    return "structure"


def _boundary(
    traces: tuple[Trace, Trace],
    labels: tuple[tuple[str, ...], tuple[str, ...]],
    *,
    justified: bool,
    options: ValueOptions,
) -> ValueResult:
    """Compare the recorded values at one paired boundary (outputs or inputs)."""

    values_a, values_b = read_values(traces[0], labels[0]), read_values(traces[1], labels[1])
    if not values_a or not values_b:
        side = "a" if not values_a else "b"
        return ValueResult(UNAVAILABLE, reason=f"boundary unrecorded ({side})")
    disclosure = ""
    if len(values_a) != len(values_b):
        disclosure = f"compared the first of {len(values_a)} vs {len(values_b)} returned values"
        values_a, values_b = values_a[:1], values_b[:1]
    results = []
    for (value_a, why_a), (value_b, why_b) in zip(values_a, values_b, strict=True):
        if value_a is None or value_b is None:
            results.append(ValueResult(UNAVAILABLE, reason=f"a={why_a or 'ok'}; b={why_b or 'ok'}"))
            continue
        results.append(compare_leaf(value_a, value_b, justified=justified, options=options))
    return combine(results, disclosure)


def _base_row(pair: CallPair, trace_a: Trace) -> dict[str, Any]:
    """The row of a pair before any value is read."""

    a, b = pair.a, pair.b
    status = (
        "present-in-both"
        if a is not None and b is not None
        else "only-a"
        if b is None
        else "only-b"
    )
    first_out = _op_from_label(trace_a, a.outputs[0]) if a is not None and a.outputs else None
    return {
        "op": a.label if a is not None else b.label if b is not None else "",
        "status": status,
        "shape_match": None,
        "dtype_match": None,
        "max_abs": None,
        "mean_abs": None,
        "allclose": None,
        "reason": pair.note,
        "source_line": _source_line(first_out) if first_out is not None else None,
        "module_b": b.label if b is not None else None,
        "alignment": pair.alignment,
        "outcome": NOT_EVALUATED,
        "rtol": None,
        "atol": None,
        "transform": "",
        "inputs_agree": None,
    }


def _row(
    pair: CallPair,
    traces: tuple[Trace, Trace],
    cross_backend: bool,
    tol: tuple[float | None, float | None],
) -> tuple[dict[str, Any], ValueResult | None]:
    """Build one row (values only for pairs established without values) and its input result."""

    row = _base_row(pair, traces[0])
    if pair.alignment not in _PAIRED or pair.a is None or pair.b is None:
        return row, None
    justified = layout_justified(pair.a.class_name, pair.b.class_name)
    options = ValueOptions(cross_backend, tol[0], tol[1], input_scale(traces[0], pair.a.inputs))
    out = _boundary(traces, (pair.a.outputs, pair.b.outputs), justified=justified, options=options)
    inputs = _boundary(traces, (pair.a.inputs, pair.b.inputs), justified=justified, options=options)
    row.update(
        {
            "dtype_match": out.dtype_match,
            "max_abs": out.max_abs,
            "mean_abs": out.mean_abs,
            "allclose": {EQUAL: True, DIFFERENT: False}.get(out.outcome),
            "shape_match": out.outcome != SHAPE_INCOMPATIBLE and not out.transform,
            "reason": "; ".join(part for part in (pair.note, out.reason) if part),
            "outcome": out.outcome,
            "rtol": out.rtol,
            "atol": out.atol,
            "transform": out.transform,
            "inputs_agree": {EQUAL: True, DIFFERENT: False}.get(inputs.outcome),
        }
    )
    return row, inputs


def _order_key(pair: CallPair) -> tuple[int, int, int, int]:
    """Side a's exit order first, then side-b-only calls in side b's exit order."""

    call = pair.a if pair.a is not None else pair.b
    if call is None:  # a pair always has at least one side
        return (2, 0, 0, 0)
    return (0 if pair.a is not None else 1, call.exit, -call.depth, call.enter)


def randomness_note(trace_a: Trace, trace_b: Trace, calls: list[CallInfo]) -> str:
    """Say when either run trained or ran a random producer, else ``""``."""

    training = sorted({call.label for call in calls if call.training})
    random_ops = [
        _op_label(op)
        for trace in (trace_a, trace_b)
        for op in _ordered_ops(trace)
        if drew_random(trace, op)
    ]
    parts = []
    if training:
        parts.append(f"modules in train mode ({len(training)}, e.g. {training[0]})")
    if random_ops:
        parts.append(f"random producers ran ({', '.join(random_ops[:3])})")
    if not parts:
        return ""
    return "randomness: " + "; ".join(parts) + "; compare in eval mode with fixed seeds"


def module_rows(
    trace_a: Trace,
    trace_b: Trace,
    *,
    rtol: float | None,
    atol: float | None,
    module_map: Mapping[str, str] | None,
) -> tuple[list[dict[str, Any]], list[CallPair], dict[str, Any]]:
    """Pair the module calls of two traces and compare their recorded outputs.

    Returns
    -------
    tuple[list[dict[str, Any]], list[CallPair], dict[str, Any]]
        Rows in order, the matching pairs, and facts (``cross_backend``, ``backends``,
        ``randomness``, per-row input results, the calls of each side).
    """

    refuse_poisoned_trace(trace_a, "faithful comparison")
    refuse_poisoned_trace(trace_b, "faithful comparison")
    backends = (str(trace_a._provenance_backend), str(trace_b._provenance_backend))
    cross = backends[0] != backends[1]
    calls_a, calls_b = read_calls(trace_a), read_calls(trace_b)
    pairs = sorted(pair_calls(calls_a, calls_b, module_map), key=_order_key)
    built = [_row(pair, (trace_a, trace_b), cross, (rtol, atol)) for pair in pairs]
    rows = [row for row, _inputs in built]
    facts = {
        "inputs": [inputs for _row_, inputs in built],
        "cross_backend": cross,
        "randomness": randomness_note(trace_a, trace_b, calls_a + calls_b),
        "calls": (calls_a, calls_b),
        "backends": backends,
    }
    return rows, pairs, facts


def coverage_counts(rows: list[dict[str, Any]], pairs: list[CallPair]) -> dict[str, dict[str, int]]:
    """Count module calls per side: paired, unpaired, boundary unrecorded, unresolved."""

    counts = {
        side: dict.fromkeys(("paired", "unpaired", "boundary_unrecorded", "unresolved"), 0)
        for side in "ab"
    }
    for row, pair in zip(rows, pairs, strict=True):
        for side, call in (("a", pair.a), ("b", pair.b)):
            if call is None:
                continue
            if row["alignment"] not in _PAIRED or row["alignment"] == ABSENT:
                counts[side]["unpaired"] += 1
                continue
            counts[side]["paired"] += 1
            if f"boundary unrecorded ({side})" in str(row["reason"]):
                counts[side]["boundary_unrecorded"] += 1
            elif row["outcome"] not in (EQUAL, DIFFERENT):
                counts[side]["unresolved"] += 1
    return counts
