"""Trace fact functions reached from the records through ``_data_classes._doors.report``.

Bodies of the ``Trace`` fact properties and methods (``factcore``,
``health_facts``, the FLOP and MAC rollups, ``profile``, ``explain``,
``bill_of_materials``, ``capabilities``, episode facts) live here as
``f(trace, ...)`` functions; the record keeps one thin method per name with
its public docstring and forwards to these through the door.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from .._capture._episode_coupling import attest_coupling
from .._capture._episode_ledger import capture_kind_for, episode_ledger_for
from .._capture.peak_memory import read_peak_pair
from .._capture_honesty import capture_advisories, poison_facts
from .._data_classes._trace_inventory import build_bill_of_materials
from .._quantities import Flops, Macs
from ._compute_truth import aggregate_forward_compute, compute_aggregation, unknown_op_ledger
from ._explain import explain as _explain
from ._factcore import factcore as _factcore
from ._health import health_facts as _health_facts, nonfinite_verdict as _nonfinite_verdict
from ._profile import profile as _profile
from ._registry import capability_card

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from ._explain import ExplainReport
    from ._profile import ProfileReport

__all__: list[str] = []


class _CallableDict(dict[Any, Any]):
    """Dict that returns a plain dict when called.

    This preserves legacy ``log.report_by_type()`` ergonomics for budgeted
    report properties that should not remain inspectable custom_methods.
    """

    def __call__(self) -> dict[Any, Any]:
        """Return a plain-dict copy of this report.

        Returns
        -------
        dict[Any, Any]
            Plain dict containing this report's items.
        """

        return dict(self)


def cost_forward_peak_memory_pair(trace: Trace) -> dict[str, Any] | None:
    """Body of ``Trace._cost_forward_peak_memory_pair``; the record keeps the thin spelling and docstring."""

    return read_peak_pair(trace)


def provenance_kind(trace: Trace) -> str:
    """Body of ``Trace._provenance_kind``; the record keeps the thin spelling and docstring."""

    return capture_kind_for(trace)


def episode(trace: Trace) -> Any | None:
    """Body of ``Trace.episode``; the record keeps the thin spelling and docstring."""

    return episode_ledger_for(trace)


def episode_coupling(trace: Trace) -> Any | None:
    """Body of ``Trace.episode_coupling``; the record keeps the thin spelling and docstring."""

    return attest_coupling(trace)


def cost_total_flops_forward(trace: Trace) -> Flops:
    """Body of ``Trace._cost_total_flops_forward``; the record keeps the thin spelling and docstring."""

    return aggregate_forward_compute(trace).flops_fma2


def cost_flops_by_op_type(trace: Trace) -> _CallableDict:
    """Body of ``Trace._cost_flops_by_op_type``; the record keeps the thin spelling and docstring."""

    result: dict[str, dict[str, int | Flops]] = {}
    rows_by_label = {row.label: row for row in compute_aggregation(trace).rows}
    for entry in trace._layer_list:
        lt = entry.type
        if lt not in result:
            result[lt] = {"forward": Flops(0), "backward": Flops(0), "count": 0}
        result[lt]["count"] += 1
        row = rows_by_label.get(str(entry.layer_label))
        if row is not None and row.flops_fma2 is not None:
            result[lt]["forward"] += Flops(row.flops_fma2)
        if entry.flops_backward is not None:
            result[lt]["backward"] += entry.flops_backward
    return _CallableDict(result)


def cost_total_macs_forward(trace: Trace) -> Macs:
    """Body of ``Trace._cost_total_macs_forward``; the record keeps the thin spelling and docstring."""

    return aggregate_forward_compute(trace).macs


def factcore(trace: Trace) -> Any:
    """Body of ``Trace.factcore``; the record keeps the thin spelling and docstring."""

    return _factcore(trace)


def health_facts(trace: Trace) -> Any:
    """Body of ``Trace.health_facts``; the record keeps the thin spelling and docstring."""

    return _health_facts(trace)


def nonfinite_verdict(trace: Trace) -> str:
    """Body of ``Trace.nonfinite_verdict``; the record keeps the thin spelling and docstring."""

    return _nonfinite_verdict(trace)


def unknown_flop_ops(trace: Trace) -> tuple[Any, ...]:
    """Body of ``Trace.unknown_flop_ops``; the record keeps the thin spelling and docstring."""

    return unknown_op_ledger(trace)


def compute_coverage(trace: Trace) -> dict[str, int]:
    """Body of ``Trace.compute_coverage``; the record keeps the thin spelling and docstring."""

    totals = aggregate_forward_compute(trace)
    return {
        "known": totals.known,
        "zero_by_rule": totals.zero_by_rule,
        "not_applicable": totals.not_applicable,
        "unknown": totals.unknown,
    }


def macs_unknown_split_ops(trace: Trace) -> tuple[str, ...]:
    """Body of ``Trace.macs_unknown_split_ops``; the record keeps the thin spelling and docstring."""

    return aggregate_forward_compute(trace).macs_unknown_split


def cost_macs_by_kind(trace: Trace) -> _CallableDict:
    """Body of ``Trace._cost_macs_by_kind``; the record keeps the thin spelling and docstring."""

    result: dict[str, dict[str, int | Macs | None]] = {}
    rows_by_label = {row.label: row for row in compute_aggregation(trace).rows}
    for entry in trace._layer_list:
        lt = entry.type
        if lt not in result:
            result[lt] = {"forward": Macs(0), "backward": None, "count": 0}
        bucket = result[lt]
        bucket["count"] = int(bucket["count"] or 0) + 1
        row = rows_by_label.get(str(entry.layer_label))
        if row is not None and row.fma_macs is not None:
            bucket["forward"] = Macs(int(bucket["forward"] or 0) + int(row.fma_macs))
    return _CallableDict(result)


def profile(
    trace: Trace,
    level: Literal["op", "module", "call"] = "op",
    *,
    sort_by: Literal["time", "flops", "activation_memory", "param_count"] = "time",
    ascending: bool = False,
    k: int | None = None,
    group_by: Literal["source_line"] | None = None,
) -> ProfileReport:
    """Body of ``Trace.profile``; the record keeps the thin spelling and docstring."""

    return _profile(
        trace,
        level=level,
        sort_by=sort_by,
        ascending=ascending,
        k=k,
        group_by=group_by,
    )


def bill_of_materials(trace: Trace) -> dict[str, Any]:
    """Body of ``Trace.bill_of_materials``; the record keeps the thin spelling and docstring."""

    return build_bill_of_materials(
        trace,
        core=_factcore(trace),
        poison_facts=poison_facts(trace),
        advisories=capture_advisories(trace),
    )


def explain(
    trace: Trace,
    *,
    audience: Literal["researcher", "practitioner", "auto"] = "auto",
    max_tokens: int | None = None,
) -> ExplainReport:
    """Body of ``Trace.explain``; the record keeps the thin spelling and docstring."""

    return _explain(trace, audience, max_tokens=max_tokens)


def capabilities(trace: Trace) -> Any:
    """Body of ``Trace.capabilities``; the record keeps the thin spelling and docstring."""

    return capability_card(trace)
