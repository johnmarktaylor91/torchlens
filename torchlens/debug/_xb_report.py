"""The two public answers of the module-level comparison: a Report and a first row.

``compare`` shows every row with a coverage line on both sides (module calls paired,
unpaired, boundary unrecorded, unresolved; never a percentage). ``first_divergence``
returns the first ``different`` row in side a's module-exit order, ``None`` only when
every paired module agrees, and refuses when the question cannot be answered.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .._capture_honesty import capture_honesty_facts
from .._debug_report import Report, table_report, trace_subject
from ._common import _op_label, _ordered_ops, _require_pandas
from ._cross_backend import MODULE_COLUMNS, coverage_counts, module_rows
from ._xb_op_place import placement_reason
from ._xb_pairing import ASSERTED, SUPPORTED, CallPair
from ._xb_refusals import refuse_inconclusive
from ._xb_values import DIFFERENT, EQUAL, SHAPE_INCOMPATIBLE

if TYPE_CHECKING:
    from torchlens._data_classes.trace import Trace

LABEL_COLUMNS = (
    "op",
    "status",
    "shape_match",
    "dtype_match",
    "max_abs",
    "mean_abs",
    "allclose",
    "reason",
    "source_line",
)
_LOCALIZED = "localized inside this module"
_FIRST_OBSERVED = "first observed differing module output"
_REMEDY = "compare runs given the same inputs, in eval mode, with module outputs recorded"


def _coverage_line(counts: dict[str, dict[str, int]], randomness: str) -> str:
    """The card's coverage line: per-side module-call counts, then any randomness."""

    def side(name: str) -> str:
        """One side's counts as words."""

        c = counts[name]
        return (
            f"{name}: {c['paired']} paired, {c['unpaired']} unpaired, "
            f"{c['boundary_unrecorded']} boundary unrecorded, {c['unresolved']} unresolved"
        )

    text = f"module calls {side('a')}; {side('b')}"
    return f"{text}; {randomness}" if randomness else text


def _attrs(
    rows: list[dict[str, Any]], pairs: list[CallPair], facts: dict[str, Any], tol: tuple[Any, Any]
) -> dict[str, Any]:
    """Summary attrs: the label path's counts at module resolution, plus module-level keys.

    ``coverage``, the pairing mode that ran (``align``, always ``"structure"`` here) and
    each side's backend (``backends``).
    """

    def count(key: str, value: Any) -> int:
        """Rows whose ``key`` equals ``value``."""

        return sum(1 for row in rows if row[key] == value)

    return {
        "matched": count("outcome", EQUAL),
        "shape_mismatch": count("outcome", SHAPE_INCOMPATIBLE),
        "only_a": count("status", "only-a"),
        "only_b": count("status", "only-b"),
        "value_diverged": count("outcome", DIFFERENT),
        "activation_unavailable": count("outcome", "unavailable"),
        "rtol": tol[0],
        "atol": tol[1],
        "coverage": coverage_counts(rows, pairs),
        "align": "structure",
        "backends": facts["backends"],
    }


def module_compare(
    trace_a: Trace,
    trace_b: Trace,
    *,
    rtol: float | None,
    atol: float | None,
    module_map: Mapping[str, str] | None,
) -> Report:
    """``tl.debug.compare`` at module resolution (see that function's docstring)."""

    rows, pairs, facts = module_rows(trace_a, trace_b, rtol=rtol, atol=atol, module_map=module_map)
    _refuse_batch_mismatch(rows, pairs, facts)
    attrs = _attrs(rows, pairs, facts, (rtol, atol))
    attrs["torchlens_capture_honesty"] = {
        "trace_a": capture_honesty_facts(trace_a),
        "trace_b": capture_honesty_facts(trace_b),
    }
    pd = _require_pandas()
    frame = pd.DataFrame(rows, columns=[*LABEL_COLUMNS, *MODULE_COLUMNS])
    frame.attrs.update(attrs)
    return table_report(
        "compare",
        frame,
        subject=f"{trace_subject(trace_a)} vs {trace_subject(trace_b)}",
        finding_columns=("op", "module_b", "alignment", "outcome", "max_abs", "rtol", "reason"),
        coverage=_coverage_line(attrs["coverage"], facts["randomness"]),
    )


def _root_row(rows: list[dict[str, Any]], pairs: list[CallPair]) -> dict[str, Any] | None:
    """The paired root-call row, whose inputs are the model inputs."""

    for row, pair in zip(rows, pairs, strict=True):
        if pair.a is not None and pair.b is not None and pair.a.parent is None:
            return row
    return None


def _refuse_batch_mismatch(
    rows: list[dict[str, Any]], pairs: list[CallPair], facts: dict[str, Any]
) -> None:
    """Refuse at the input boundary when the model inputs' shapes cannot be aligned.

    A leading size-one axis on one side only is dropped and disclosed; every other
    batch or shape difference of the model inputs refuses.
    """

    for row, pair, inputs in zip(rows, pairs, facts["inputs"], strict=True):
        if pair.a is None or pair.b is None or pair.a.parent is not None or inputs is None:
            continue
        if inputs.outcome == SHAPE_INCOMPATIBLE:
            refuse_inconclusive(
                f"the model inputs differ in shape beyond a leading size-one axis ({inputs.reason}) "
                f"at {row['op']}, so the runs are not the same computation",
                remedy="give both runs the same batch; only a batch of one against no batch axis "
                "is aligned automatically",
            )


_Table = tuple[list[dict[str, Any]], list[CallPair]]


def _skipped_before(trace_a: Trace, pair: CallPair, table: _Table) -> str:
    """Side-a ops between the last agreeing paired module exit and this module's entry."""

    if pair.a is None:
        return ""
    exits = [
        p.a.exit
        for r, p in zip(*table, strict=True)
        if p.a is not None and r["outcome"] == EQUAL and p.a.exit < pair.a.enter
    ]
    start = max(exits, default=-1) + 1
    labels = [_op_label(op) for op in _ordered_ops(trace_a)[start : max(pair.a.enter, start)]]
    if not labels:
        return "nothing recorded between matched boundaries"
    shown = ", ".join(labels[:6])
    more = f" and {len(labels) - 6} more" if len(labels) > 6 else ""
    return f"skipped computation between matched boundaries: {shown}{more}"


def _worded(
    row: dict[str, Any], pair: CallPair, trace_a: Trace, table: _Table, randomness: str
) -> dict[str, Any]:
    """Prefix the row's reason with what the input check allows it to claim."""

    if row["inputs_agree"] is True:
        lead = f"{_LOCALIZED} (inputs agree, output differs)"
    else:
        state = "inputs already differ" if row["inputs_agree"] is False else "inputs not compared"
        lead = f"{_FIRST_OBSERVED} ({state}); {_skipped_before(trace_a, pair, table)}"
    parts = [lead, row["reason"], randomness]
    return {**row, "reason": "; ".join(part for part in parts if part)}


def module_first_divergence(
    trace_a: Trace,
    trace_b: Trace,
    *,
    rtol: float | None,
    atol: float | None,
    module_map: Mapping[str, str] | None,
) -> dict[str, Any] | None:
    """``tl.debug.first_divergence`` at module resolution (see that function's docstring)."""

    rows, pairs, facts = module_rows(trace_a, trace_b, rtol=rtol, atol=atol, module_map=module_map)
    _refuse_unanswerable(rows, pairs, facts)
    for row, pair in zip(rows, pairs, strict=True):
        if row["outcome"] == DIFFERENT:
            worded = _worded(row, pair, trace_a, (rows, pairs), facts["randomness"])
            placed = placement_reason(pair, (rows, pairs), facts, (trace_a, trace_b), (rtol, atol))
            return {**worded, "reason": f"{worded['reason']}; {placed}"}
    paired = [row for row in rows if row["alignment"] in (SUPPORTED, ASSERTED)]
    if not any(row["outcome"] == EQUAL for row in paired):
        refuse_inconclusive(
            "no paired module call could be value-compared, so no divergence can be named or excluded",
            remedy=_REMEDY,
        )
    unresolved = [row for row in paired if row["outcome"] != EQUAL]
    if unresolved:
        names = ", ".join(
            f"{row['op']} ({row['outcome']}: {row['reason']})" for row in unresolved[:3]
        )
        refuse_inconclusive(
            f"{len(unresolved)} paired module call(s) could not be decided, so agreement cannot be "
            f"claimed: {names}",
            remedy="print(tl.debug.compare(trace_a, trace_b)) lists every unresolved row",
        )
    return None


def _refuse_unanswerable(
    rows: list[dict[str, Any]], pairs: list[CallPair], facts: dict[str, Any]
) -> None:
    """Refuse when a side has no module boundaries or the model inputs differ."""

    calls_a, calls_b = facts["calls"]
    for name, calls in (("a", calls_a), ("b", calls_b)):
        if not [call for call in calls if call.parent is not None]:
            refuse_inconclusive(
                f"trace_{name} records no module boundaries below the root, so modules cannot be paired",
                remedy="capture a model built from submodules on both sides",
            )
    _refuse_batch_mismatch(rows, pairs, facts)
    root = _root_row(rows, pairs)
    if root is not None and root["inputs_agree"] is False:
        refuse_inconclusive(
            "the two runs were given different inputs, so every module downstream differs",
            remedy=_REMEDY,
        )
