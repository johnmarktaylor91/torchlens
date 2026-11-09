"""What a recording hands its cooked Trace besides the records themselves.

A predicate pass prepares a tensor-free parameter inventory (``Param`` logs
keyed by address) and a ``raw label -> call-site class`` table for every
recorded op whose call site the capture walk saw. Both are snapshotted into
the run core's projection facts when the pass ends and installed on the fresh
Trace shell before postprocessing, so a cooked recording resolves parameters
and groups by call site (``callsite_v2``) exactly as an exhaustive capture.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

__all__ = (
    "capture_identity_facts",
    "install_cooked_inventory",
    "install_run_facts",
    "recording_projection_inventory",
)

#: Capture-identity facts an exhaustive capture stamps on its Trace at setup
#: (``_run_setup._stamp_identity_and_sinks``); a cooked Trace carries the same
#: ones so the rerun/append identity gate and the artifact validators read the
#: capture's facts, never a cook-time guess.
IDENTITY_FACT_ATTRS: tuple[str, ...] = (
    "_provenance_model_object_id",
    "_source_class_qualname",
    "_provenance_param_hash_quick",
    "_provenance_param_hash_full",
    "_provenance_root_entry_point",
    "_provenance_input_object_id",
    "_provenance_input_signature_hash",
)


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


def capture_identity_facts(trace: Any) -> dict[str, Any]:
    """Snapshot the identity facts a capture stamped on its Trace shell (absent ones ``None``)."""

    return {name: trace.__dict__.get(name) for name in IDENTITY_FACT_ATTRS}


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
    for name, value in dict(trace_facts.get("identity_facts") or {}).items():
        setattr(trace, name, value)
    # No completeness witness ran over a predicate pass, so the cooked Trace
    # makes no verification claim (the same ``None`` a value-free exhaustive
    # capture carries), never the missing-attribute crash of an unset slot.
    trace._provenance_verified = None
    trace._provenance_verification_reason = None


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
    _install_layers_with_params(trace)


def _install_layers_with_params(trace: Any) -> None:
    """Group the cooked parameterized ops the way exhaustive capture does.

    Exhaustive capture appends each parameterized op's raw label to
    ``layers_with_params`` under its parameter-group key (the equivalence
    class before the module-stack suffix) as the op is logged
    (``_ops_activations``). A predicate pass logs nothing there, so the cook
    rebuilds the table from the op records already installed on
    ``trace.capture_events`` (``param_xrefs`` checks it against
    ``Param.used_by_layers``).
    """

    events = getattr(trace, "capture_events", None)
    if events is None:
        return
    table = trace.layers_with_params
    for event in events.amended_op_records():
        if not event.params:
            continue
        key = _parameter_group_key(event)
        if key is not None and event.label_raw not in table[key]:
            table[key].append(event.label_raw)


def _parameter_group_key(event: Any) -> str | None:
    """Return an op record's equivalence class without its module-stack suffix."""

    equivalence_class = event.equivalence_class
    if equivalence_class is None or not event.modules:
        return equivalence_class
    suffix = "_".join(address for address, _call_index in event.modules)
    if suffix and equivalence_class.endswith(suffix):
        return equivalence_class[: -len(suffix)]
    return equivalence_class
