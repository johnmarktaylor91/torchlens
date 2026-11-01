"""Trace-level counters derived from finalized layers and ops."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def _count_trace_ops(trace: Trace) -> int:
    """Count the ops that are neither inputs, outputs nor buffers."""

    return sum(
        1
        for op_log in trace.layer_list
        if not (op_log.is_input or op_log.is_output or op_log.is_buffer)
    )


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
    Totals derive from ``trace.param_logs`` (the deduped-by-identity param
    registry), not from summing each op's own ``num_param_tensors`` across
    unique layer labels: a param reused by more than one op/layer (a tied
    weight, or a composite op decomposed into several primitive ops that
    each separately attach it) is counted once per referencing layer by the
    per-op sum, inflating the total past the true unique-param count and
    tripping the ``trace_self_consistency`` invariant's parameter-identity
    check (``num_param_tensors != len(param_logs)``).
    """

    if trace.param_source != "none":
        trace.num_param_tensors = len(trace.param_logs)
        trace.num_params = sum(param.num_params for param in trace.param_logs)
        trace.num_params_trainable = sum(
            param.num_params for param in trace.param_logs if param.is_trainable
        )
        trace.num_params_frozen = trace.num_params - trace.num_params_trainable
        trace.num_layers_with_params = len(
            {op.layer_label for op in trace.layer_list if op.has_params}
        )
