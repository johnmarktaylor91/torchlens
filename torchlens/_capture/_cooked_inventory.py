"""What a recording hands its cooked Trace besides the records themselves.

A predicate pass prepares a tensor-free parameter inventory (``Param`` logs
keyed by address) and a ``raw label -> call-site class`` table for every
recorded op whose call site the capture walk saw. Both are snapshotted into
the run core's projection facts when the pass ends and installed on the fresh
Trace shell before postprocessing, so a cooked recording resolves parameters
and groups by call site (``callsite_v1``) exactly as an exhaustive capture.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

__all__ = ("install_cooked_inventory", "install_run_facts", "recording_projection_inventory")


def recording_projection_inventory(
    recording: Any,
) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, str]]:
    """Return this pass's records, parameter logs and call-site classes.

    A Recorder shares one Recording across its passes, so its record list is
    cumulative; each pass's core keeps only the records that pass added, or
    the projector's per-core concatenation would repeat every earlier pass's
    records once per later pass.
    """

    records: tuple[Any, ...] = tuple(object.__getattribute__(recording, "records"))
    state = object.__getattribute__(recording, "_recording_state")
    param_logs = dict(getattr(state, "prepared_param_logs", None) or {})
    call_site_classes = dict(getattr(state, "call_site_classes", None) or {})
    pass_index = getattr(state, "pass_index", None)
    if pass_index is not None:
        records = tuple(record for record in records if record.ctx.pass_number == pass_index)
    return records, param_logs, call_site_classes


def install_run_facts(trace: Any, trace_facts: Mapping[str, Any]) -> None:
    """Copy the sealed run facts a cooked Trace needs before postprocessing."""

    trace._provenance_start_time = trace_facts["capture_start_time"]
    trace._provenance_setup_duration = trace_facts["setup_duration"]
    trace._cost_forward_duration = trace_facts["forward_duration"]
    trace._cost_forward_peak_memory = trace_facts["forward_peak_memory"]
    trace._cost_forward_peak_memory_reason = trace_facts.get("forward_peak_memory_reason")
    trace._provenance_forward_memory_backend = trace_facts["forward_memory_backend"]
    trace._source_model_ref = trace_facts["source_model_ref"]
    trace._options_seed = trace_facts["random_seed"]
    trace._raw_graph_ws.layer_counter = int(trace_facts["layer_counter"])


def install_cooked_inventory(
    trace: Any,
    trace_facts: Mapping[str, Any],
    *,
    param_accessor: Callable[[dict[str, Any]], Any],
) -> None:
    """Give a fresh cooked Trace shell its parameter logs and call-site classes.

    Each cook gets its own ``Param`` objects: postprocessing annotates them
    (ops, pass counts) and a Recording may be cooked more than once.
    ``param_accessor`` wraps them the way the Trace exposes ``params``. The
    call-site table goes to the raw-graph workspace side table step 7 reads,
    never into a public field.
    """

    param_logs = trace_facts.get("param_logs")
    if param_logs:
        trace._param_logs = param_accessor(
            {address: copy.copy(param_log) for address, param_log in param_logs.items()}
        )
    call_site_classes = trace_facts.get("call_site_classes")
    if call_site_classes:
        trace._raw_graph_ws._call_site_classes = dict(call_site_classes)
