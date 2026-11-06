"""Trace-level counters derived from finalized layers and ops."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from .._quantities import Bytes

if TYPE_CHECKING:
    from .._data_classes.param import Param
    from .._data_classes.trace import Trace


def _count_trace_ops(trace: Trace) -> int:
    """Count the ops that are neither inputs, outputs nor buffers."""

    return sum(
        1
        for op_log in trace._layer_list
        if not (op_log.is_input or op_log.is_output or op_log._graph_is_buffer)
    )


def _stamp_param_totals(trace: Trace, params: Iterable[Param]) -> None:
    """Set the trace's parameter counters and byte total from its parameter records.

    Parameters
    ----------
    trace:
        Trace whose parameter counters are written.
    params:
        The trace's deduplicated parameter records (one per parameter object).

    Returns
    -------
    None
        ``_num_param_tensors``, ``num_params``, ``num_params_trainable``,
        ``num_params_frozen`` and ``_cost_total_param_memory`` are written.

    Notes
    -----
    Every preview backend stamps its totals here, so the byte total
    (``trace.cost.total_param_memory``, the drawing header's parameter size)
    can never be left at its zero default while the count is filled in. Each
    record's ``param_memory`` was read from the framework array at capture.
    """

    records = list(params)
    trace._num_param_tensors = len(records)
    trace.num_params = sum(param.num_params for param in records)
    trace.num_params_trainable = sum(param.num_params for param in records if param.is_trainable)
    trace.num_params_frozen = trace.num_params - trace.num_params_trainable
    trace._cost_total_param_memory = Bytes(sum(int(param.param_memory or 0) for param in records))


def _update_param_totals_from_layers(trace: Trace) -> None:
    """Recompute trace parameter totals on a parameter-identity basis.

    Parameters
    ----------
    trace:
        Trace whose ``param_logs`` registry is already populated.

    Returns
    -------
    None
        Trace-level parameter counters are updated when parameters are present.

    Notes
    -----
    Totals derive from ``trace._param_logs`` (the deduped-by-identity param
    registry), not from summing each op's own ``num_param_tensors`` across
    unique layer labels: a param reused by more than one op/layer (a tied
    weight, or a composite op decomposed into several primitive ops that
    each separately attach it) is counted once per referencing layer by the
    per-op sum, inflating the total past the true unique-param count and
    tripping the ``trace_self_consistency`` invariant's parameter-identity
    check (``num_param_tensors != len(param_logs)``).
    """

    if trace._provenance_param_source != "none":
        _stamp_param_totals(trace, trace._param_logs)
        trace._num_layers_with_params = len(
            {op.layer_label for op in trace._layer_list if op.has_params}
        )
