"""Trace pickle state helpers moved off ``Trace.__getstate__`` and ``__setstate__``."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .._constants import MODEL_LOG_FIELD_ORDER
from .._io import (
    TLSPEC_VERSION,
    coerce_container_typed_state,
    default_fill_state,
    read_tlspec_version,
)
from .._quantities import Bytes, Duration
from .._trace_state import TraceState
from . import _doors, _trace_rehydrate
from ._transform_repr import _scrubbed_transform_repr
from .derived_grad import DerivedGradAccessor
from .module import ModuleAccessor

if TYPE_CHECKING:
    from .trace import Trace

__all__ = ["trace_getstate", "trace_setstate"]


# Typed container defaults for every non-Optional container field in
# `MODEL_LOG_FIELD_ORDER`. The blanket ``{field: None}`` base below is wrong for
# these: an absent (legacy/partial-state) container field would restore as
# ``None`` instead of its declared list/dict/set/tuple and then crash on first
# touch (``.values()``/iteration), or -- for ``layer_list``/``layer_logs`` --
# crash inside ``__setstate__`` itself. Plain builtin types are used
# deliberately: they let ``coerce_container_typed_state`` also repair a
# present-but-wrong-typed legacy value (e.g. ``None`` or a ``dict`` where a
# ``list`` is now declared). Fields whose declared type is a genuine
# ``OrderedDict``/``defaultdict`` at runtime restore correctly as a plain dict
# for legacy states (still ``.values()``/``.items()``-usable); fresh captures
# always carry the exact runtime container, so this only affects legacy fill.
_MODEL_LOG_CONTAINER_DEFAULTS: dict[str, Any] = {
    "_source_snapshots": [],
    "escape_diagnostics": [],
    "completeness_diagnostics": [],
    "_health_completeness_decompositions": [],
    "capture_guard_passes": [],
    "annotations": {},
    "_provenance_observer_spans": [],
    "_graph_manual_tensor_connections": [],
    "_source_frames": [],
    "_out_identity_cache": {},
    "_out_hash_cache": {},
    "_code_context_cache": {},
    "_grad_op_nums_to_save": [],
    "_provenance_input_annotations": {},
    "_source_code_blob": {},
    "state_history": [],
    "_append_history": [],
    "_last_hook_handle_ids": (),
    "_graph_relationship_evidence": {},
    "_replay_frontier": {},
    "_layer_list": [],
    "_layer_dict_main_keys": {},
    "_layer_dict_all_keys": {},
    "_layer_logs": {},
    "layer_labels": [],
    "op_labels": [],
    "_layer_num_calls": {},
    "_by_pass": {},
    "_raw_to_final_layer_labels": {},
    "_raw_to_final_parent_layer_labels": {},
    "_raw_to_final_op_labels": {},
    "_final_to_raw_layer_labels": {},
    "_lookup_keys_to_layer_num_dict": {},
    "_layer_num_to_lookup_keys_dict": {},
    "_ambiguous_lookup_keys": {},
    "input_layers": [],
    "output_layers": [],
    "buffer_layers": [],
    "_buffer_num_calls": {},
    "internal_source_ops": [],
    "internal_sink_ops": [],
    "internally_terminated_bool_ops": [],
    "_graph_conditional_branch_edges": [],
    "_graph_conditional_records": [],
    "_graph_conditional_arm_entry_edges": {},
    "_graph_conditional_edge_call_indices": {},
    "layers_with_params": {},
    "_graph_op_equivalence_classes": {},
    "_orphan_labels": [],
    "_orphan_logs": (),
    "_orphan_records": [],
    "_phase_timings": {},
    "_grad_fn_logs": {},
    "_autograd_grad_fn_order": [],
    "_backward_pass_logs": {},
    "_grad_fn_param_refs": {},
    "_autograd_backward_root_grad_fn_object_ids": [],
    "_cost_backward_durations": [],
}
_MODEL_LOG_DEFAULT_FILL: dict[str, Any] = {
    "_options_grouping": "structural",
    "_options_grouping_policy": None,
    "_provenance_distributed_scope": None,
    "name": None,
    "model_name": None,
    "_provenance_backend": "torch",
    "_options_module_identity_mode": "torch_module",
    "_provenance_param_source": "native-module",
    "_autograd_derived_grads": DerivedGradAccessor(),
    "_runnable": None,
    "_fast_run_session": None,
    "_distributed_plane_p": None,
    "_buffer_persistence": {},
    "is_replayable": False,
    "_options_save_arg_templates": False,
    "input": None,
    "_transform": None,
    "_options_save_raw_input": "small",
    "batch_render": "auto",
    "transformed_output": None,
    "_output_transform": None,
    "_options_save_transformed_output": "small",
    "layer_visualizers": None,
    "save_visualizations": False,
    "_visualizer_dir": None,
    "_parent_run": None,
    "_intervention_spec": None,
    "state_history": [],
    "last_run": None,
    "_append_history": [],
    "_has_direct_writes": False,
    "_warned_direct_write": False,
    "_warned_mutate_in_place": False,
    "_warned_once": set(),
    "_spec_revision": 0,
    "_out_recipe_revision": 0,
    "_annotation_blobs": None,
    "_append_sequence_id": 0,
    "_last_hook_handle_ids": (),
    "state": TraceState.PRISTINE,
    "_provenance_model_object_id": None,
    "_source_class_qualname": None,
    "_provenance_param_hash_quick": None,
    "_provenance_param_hash_full": None,
    "_provenance_root_entry_point": None,
    "_provenance_input_object_id": None,
    "_provenance_input_signature_hash": None,
    "_provenance_fingerprint": None,
    "_module_filter": None,
    "_options_emit_nvtx": False,
    "measure_python_peak_memory": False,
    "distributed_witness": "none",
    "save_budget": "auto",
    "_options_on_nonfinite": False,
    "track_nonfinite": False,
    "track_device_memory": False,
    "measure_host_peak_memory": False,
    "_options_structure_only": False,
    "intervention_audit": [],
    "_options_keep_orphans": False,
    "annotations": {},
    "_provenance_observer_spans": [],
    "_graph_manual_tensor_connections": [],
    "_source_forward_line": None,
    "_source_forward_file": None,
    "_source_class_file": None,
    "_source_class_line": None,
    "_source_init_file": None,
    "_source_init_line": None,
    "_source_class_docstring": None,
    "_source_init_signature": None,
    "_source_init_docstring": None,
    "_source_forward_signature": None,
    "_source_forward_docstring": None,
    "_source_frames": [],
    "_provenance_cache_hit": False,
    "_provenance_cache_key": None,
    "_provenance_cache_path": None,
    "_provenance_recording_kept": True,
    "facet_functions": None,
    "_out_dedup_mode": "identity",
    "_out_identity_cache": {},
    "_out_hash_cache": {},
    "_code_context_cache": {},
    "capture_tensor_grad_hooks": True,
    "_options_save_grads": None,
    "_options_inference_only": False,
    "_options_chunk_size": False,
    "_provenance_is_appended": False,
    "_graph_relationship_evidence": {},
    "_replay_frontier": {},
    "_ambiguous_lookup_keys": {},
    "_cost_total_grad_memory": 0,
    "_cost_total_backward_memory": 0,
    "_cost_saved_grad_memory": 0,
    "_cost_total_param_grad_memory": 0,
    "_cost_forward_peak_memory": 0,
    "_provenance_forward_memory_backend": "unknown",
    "_cost_forward_peak_memory_reason": None,
    "backward_peak_memory_reason": None,
    "_forward_peak_memory_pair": None,
    "_phase_timings": {},
    "_replay_arg_version_data_complete": True,
    "_grad_fn_param_refs": {},
}
_MODEL_LOG_DEFAULT_FILL = {
    **dict.fromkeys(MODEL_LOG_FIELD_ORDER),
    **_MODEL_LOG_CONTAINER_DEFAULTS,
    **_MODEL_LOG_DEFAULT_FILL,
}
_MODEL_LOG_DEFAULT_FILL["tlspec_version"] = TLSPEC_VERSION
_SESSION_BOOL_KNOBS = (
    "measure_python_peak_memory",
    "measure_host_peak_memory",
    "track_nonfinite",
    "track_device_memory",
)


def _normalize_conditional_arm_entry_edges(
    value: Any,
) -> dict[tuple[int, str], list[tuple[str, str]]]:
    """Return conditional arm edges in canonical flat-key form.

    Parameters
    ----------
    value:
        Stored conditional arm edge state from current or older portable bundles.

    Returns
    -------
    dict[tuple[int, str], list[tuple[str, str]]]
        Mapping from ``(conditional_id, arm_kind)`` to edge tuples.
    """

    normalized: dict[tuple[int, str], list[tuple[str, str]]] = {}
    if not isinstance(value, Mapping):
        return normalized
    for raw_key, raw_edges in value.items():
        if isinstance(raw_key, tuple) and len(raw_key) == 2 and isinstance(raw_key[1], str):
            normalized[(int(raw_key[0]), raw_key[1])] = list(raw_edges or [])
            continue
        if not isinstance(raw_edges, Mapping):
            continue
        conditional_id = int(raw_key)
        for arm_kind, arm_edges in raw_edges.items():
            normalized[(conditional_id, str(arm_kind))] = list(arm_edges or [])
    return normalized


def _drop_runtime_only_caches(state: dict[str, Any]) -> None:
    """Strip lazy accessor, geometry and render caches from pickle ``state`` in place."""

    # Lazy per-instance accessor caches (FieldPolicy.DROP) rebuild on
    # demand after restore; carrying them would make pickle bytes depend
    # on which accessors were touched (access-is-pure contract).
    state.pop("_op_accessor_cache", None)
    state.pop("_layer_accessor_cache", None)
    state.pop("_module_call_accessor", None)
    # Lazy influence-geometry caches (FieldPolicy.DROP) hold
    # mappingproxy-bearing solution objects after any receptive-field
    # access, so carrying them made pickle.dumps/copy.deepcopy crash with
    # "cannot pickle 'mappingproxy' object" while Trace.save succeeded on
    # the same trace. They rebuild on first access after restore.
    state.pop("_receptive_field_solution", None)
    state.pop("_rf_directional_solutions", None)
    # Render diagnostic from the last draw(): its EncodingChannelSpec can
    # hold the RAW user callable a `color_by=lambda ...` passed in, so
    # carrying it made pickle.dumps crash after an ordinary draw while
    # Trace.save succeeded on the same trace (the R10-7 raw-callable class).
    # Runtime-only either way; it rebuilds on the next draw.
    state.pop("_last_encoding_state", None)
    # The recorded save request can hold a raw user save callable.
    state.pop("_rerun_save_request", None)
    # Same runtime-only class: the last draw's layout-execution geometry
    # record (vizmech D24) rebuilds on the next draw and never serializes.
    state.pop("_last_render_geometry", None)
    state.pop("_container_ordinals_by_output_op_label", None)
    state.pop("_container_ordinals_by_input_func_call_id", None)


def trace_getstate(trace: Trace) -> dict[str, Any]:
    """Return pickle state with non-picklable weakref-backed accessors stripped."""
    state = trace.__dict__.copy()
    state["_pickle_module_accessor_state"] = trace.__dict__.get("_module_logs")
    # Event streams never serialize (FieldPolicy.DROP): strip the stream
    # AND the projection guard derived from it, or a restored trace would
    # claim a source-process revision/fold-state over a fresh empty stream
    # (silently dropped passes / stale-id partial folds on the advertised
    # restored-trace backward-capture path).
    state.pop("_capture_events", None)
    state.pop("_backward_projection_event_count", None)
    state.pop("_backward_projection_revision", None)
    state.pop("_backward_projection_fold_state", None)
    state.pop("_tl_materializing_backward_projection", None)
    state["_module_logs"] = None
    state["_buffer_accessor"] = None
    state["_source_model_ref"] = None
    state["_parent_run"] = None
    state["last_run"] = None
    state["_out_identity_cache"] = {}
    state["_out_hash_cache"] = {}
    state["_code_context_cache"] = {}
    _drop_runtime_only_caches(state)
    # B1-02: the semantic-output scratch never serializes. Plain pickle
    # legitimately carries most session-time DROP state (in-process
    # round-trips need `backward_ready`, `save_budget`, ...), but these
    # hold LIVE USER OBJECTS -- an HF tokenizer and a model-derived
    # metadata key. A surviving copy reconstructs the tokenizer on load and
    # bakes its vocab/merges into an artifact the user believes is a graph
    # (measured 47KB -> 238KB). Capture drops them at every settlement
    # path; this is the serialization boundary's own belt.
    state.pop("_output_tokenizer", None)
    state.pop("_semantic_output_metadata", None)
    # Capture-session predicate carriers (FieldPolicy.DROP) hold LIVE USER
    # CLOSURES when predicate capture/intervention/halt was used (a
    # ``tl.when(...)`` conditional is a local function), so pickling a
    # trace that ``Trace.save`` accepted failed on exactly these fields.
    # Serialize to the loaded-artifact form instead (absent / None), which
    # every post-capture consumer already tolerates -- same belt as the
    # semantic-output scratch above.
    state.pop("_stop_directive", None)
    state.pop("_capture_config", None)
    state["_predicate_save_options"] = None
    # R10-6: run(fast=True) binds a session holding weakrefs/compiled
    # binders; every other session workspace pops here but this one did
    # not, so pickle.dumps/copy.deepcopy after a fast run crashed with
    # "cannot pickle 'weakref.ReferenceType'". Session-time (FieldPolicy
    # DROP under its private name); a restored trace re-runs verified.
    state.pop("_fast_run_session", None)
    # Session-time plane-P dispatch journal (merge-ranks C2): tuples of
    # per-dispatch facts, meaningless outside the capture session.
    state.pop("_distributed_plane_p", None)
    # R10-7: the REPR is scrubbed below, but the RAW user callables stayed
    # in state, so a lambda transform= made pickle.dumps crash while
    # Trace.save succeeded on the same trace. Serialize to the loaded-artifact
    # form (None), which every post-load consumer already tolerates.
    state["_options_activation_transform"] = None
    state["_options_grad_transform"] = None
    state["_output_transform"] = None
    # R10-7b: the MODERN tl.trace(..., transform=...) kwarg populates
    # `_transform`, a fourth raw-callable holder the original fix missed
    # (it covered only the deprecated activation_transform= spelling).
    state["_transform"] = None
    # Facet recipe snapshot (FieldPolicy.DROP): its entries pickle builtin
    # recipe FUNCTIONS by import identity, so a later recipe rename made
    # every earlier plain pickle unloadable (gqa_sdpa_attention,
    # 2026-08-26). Serialize to the loaded-artifact form (None);
    # `_snapshot_for_record` falls back to the live registry exactly as it
    # does for every tlspec-loaded Trace.
    state["facet_functions"] = None
    state["_cost_pinned_autograd_memory"] = None  # a pickled copy pins nothing
    state.pop("_uncalled_module_addresses", None)  # session-time: unknown once restored
    state.pop("_raw_graph_ws", None)
    state.pop("_module_capture_ws", None)
    state.pop("_wrapper_runtime_ws", None)
    state.pop("_trace_core", None)
    state["_backward_gradfn_refs"] = []
    state["_tl_backward_hooked_tensor_keys"] = set()
    state.pop("_tl_grad_hook_owner_by_label", None)
    state["_pending_live_fire_records"] = []
    state["_last_hook_handle_ids"] = ()
    state["_activation_transform_repr"] = _scrubbed_transform_repr(
        trace._options_activation_transform
    )
    # Runnable traces bind state as immutable MappingProxyType views, which
    # cannot be pickled/deepcopied. Preserve tensor identity while replacing
    # only those mapping proxies with ordinary dictionaries.
    state["_runnable"] = _doors.runnable().seam.runnable_trace_state(trace).pickle_safe_copy()
    # The settled capture outcome persists as its STRING-ONLY payload dict
    # (tlspec v7): loads parse it against the closed vocabularies and the
    # status coherence matrix, so a stale or forged record can only ever
    # degrade to UNKNOWN, never upgrade. A pre-settlement live object
    # persists None and loads derive from the structural lattice.
    outcome = state.get("_capture_outcome")
    state["_capture_outcome"] = (
        outcome.to_payload() if outcome is not None and hasattr(outcome, "to_payload") else None
    )
    # grind-r5 P7 (b3 opus R10-1 / b6 opus R16-1): the producer's NEGATIVE
    # verification claim persists -- a capture TorchLens itself refused to
    # bless must not round-trip into "no claim". A POSITIVE claim never
    # persists (a tampered artifact could otherwise forge verified=True;
    # verdicts may only degrade across persistence). ``rescue_rerun``
    # stays session-time per the migration doc.
    if state.get("_provenance_verified") is False:
        reason = state.get("_provenance_verification_reason")
        state["_provenance_verification_reason"] = reason if isinstance(reason, str) else None
    else:
        state["_provenance_verified"] = None
        state["_provenance_verification_reason"] = None
    state["tlspec_version"] = TLSPEC_VERSION
    return state


def trace_setstate(trace: Trace, state: dict[str, Any], *, conditional_accessor: type) -> None:
    """Restore pickle state and rebuild weakref-backed links."""
    pickle_module_accessor_state = state.pop("_pickle_module_accessor_state", None)
    # P7/R10: restore the persisted NEGATIVE verification disclosure. The
    # row is string-only and can only ever WORSEN a verdict: anything but
    # the exact {"verified": False} shape is ignored (stays "no claim"), so
    # a forged row cannot bless a capture and a stripped row merely reverts
    # to the historical launder this closes for honest artifacts.
    verification_row = state.pop("_capture_verification", None)
    if (
        isinstance(verification_row, dict)
        and verification_row.get("verified") is False
        and isinstance(verification_row.get("reason"), (str, type(None)))
    ):
        state["_provenance_verified"] = False
        state["_provenance_verification_reason"] = verification_row.get("reason")
    for field_name in (
        *_doors.ir().workspaces.LEGACY_TRACE_BUILD_STATE_KEYS,
        # "_build_state" is the pre-M10 flat scratchpad key; the three
        # workspace keys are its dissolved successors. All transient,
        # all dropped on restore.
        "_build_state",
        "_raw_graph_ws",
        "_module_capture_ws",
        "_wrapper_runtime_ws",
        "_trace_core",
    ):
        state.pop(field_name, None)
    serialized_tlspec_version = read_tlspec_version(
        state, cls_name=type(trace).__name__, cls=type(trace)
    )
    containers_were_serialized = "_containers" in state and state["_containers"] is not None
    setstate_defaults = {
        **_MODEL_LOG_DEFAULT_FILL,
        "tlspec_version": serialized_tlspec_version,
        "_options_input_transform": None,
        "_decoded_output": None,
        "_provenance_output_postprocessor": None,
        "output_id2label": None,
        "output_num_classes": None,
        "_activation_transform_repr": None,
        "_options_module_identity_mode": "torch_module",
        "_provenance_param_source": "native-module",
        # Session-time (DROP): a restored trace carries no optimizer handle.
        "_optimizer": None,
        "_options_save_raw_activations": True,
        "_options_save_mode": "copy",
        "transformed_output": None,
        "_output_transform": None,
        "_options_save_transformed_output": "small",
        "layer_visualizers": None,
        "save_visualizations": False,
        "_visualizer_dir": None,
        "_provenance_input_annotations": {},
        "_options_grad_transform": None,
        "_options_save_raw_grads": True,
        "_options_save_grads": None,
        "capture_tensor_grad_hooks": True,
        "_grad_op_nums_to_save": [],
        "_autograd_has_backward_pass": False,
        "_grad_fn_logs": OrderedDict(),
        "_autograd_grad_fn_order": [],
        "_backward_pass_logs": OrderedDict(),
        "_autograd_backward_root_grad_fn_object_ids": [],
        "_cost_backward_durations": [],
        "_num_backward_passes": 0,
        "_cost_backward_peak_memory": 0,
        "_provenance_backward_memory_backend": "unknown",
        "backward_peak_memory_reason": None,
        "_cost_total_autograd_memory": None,
        "_cost_pinned_autograd_memory": None,
        "_buffer_accessor": None,
        "_module_logs": None,
        "_out_writer": None,
        "_keep_outs_in_memory": True,
        "_defer_streaming_bundle_finalization": False,
        "_out_sink": None,
        "_append_history": [],
        "_source_code_blob": {},
        "_source_model_ref": None,
        "_options_backward_ready": False,
        "_options_inference_only": False,
        "_options_chunk_size": False,
        "_module_filter": None,
        "_options_on_nonfinite": False,
        "track_nonfinite": False,
        "track_device_memory": False,
        "measure_host_peak_memory": False,
        "_options_structure_only": False,
        "intervention_audit": [],
        "_options_keep_orphans": False,
        "annotations": {},
        "_provenance_observer_spans": [],
        "_graph_manual_tensor_connections": [],
        "_source_forward_file": None,
        "_source_forward_line": None,
        "_source_class_file": None,
        "_source_class_line": None,
        "_source_init_file": None,
        "_source_init_line": None,
        "_source_class_docstring": None,
        "_source_init_signature": None,
        "_source_init_docstring": None,
        "_source_forward_signature": None,
        "_source_forward_docstring": None,
        "_source_frames": [],
        "_provenance_cache_hit": False,
        "_provenance_cache_key": None,
        "_provenance_cache_path": None,
        "_provenance_recording_kept": True,
        "_out_dedup_mode": "identity",
        "_out_identity_cache": {},
        "_out_hash_cache": {},
        "_code_context_cache": {},
        "_last_hook_handle_ids": (),
        "_graph_conditionals": conditional_accessor(),
        "_cost_total_grad_memory": 0,
        "_cost_saved_grad_memory": 0,
        "_cost_total_param_grad_memory": 0,
        "_cost_forward_peak_memory": 0,
        "_provenance_forward_memory_backend": "unknown",
        "_cost_forward_peak_memory_reason": None,
        "_postprocessing_active": False,
        # `_backward_gradfn_refs` is `FieldPolicy.DROP` (never part of
        # `MODEL_LOG_FIELD_ORDER`) and `__getstate__` always emits an
        # empty `list` for it -- but any artifact serialized by code
        # older than this line still has it baked in as a `dict`
        # (`__getstate__` used to hardcode `{}`). Listing it here
        # lets `coerce_container_typed_state` below fix that
        # present-but-wrong-typed legacy shape, not just absence.
        "_backward_gradfn_refs": [],
    }
    default_fill_state(state, defaults=setstate_defaults)
    runnable_state = _doors.runnable().seam.normalize_runnable_trace_state(state)
    coerce_container_typed_state(
        state,
        setstate_defaults,
        exclude={
            # `List[int] | str` -- a bare string sentinel (``"all"``) is
            # a legitimate value, not a type-mismatch bug; coercing it
            # to ``list("all")`` would silently corrupt real data.
            "_grad_op_nums_to_save",
        },
    )
    state["_cost_pinned_autograd_memory"] = None  # a restored trace pins nothing
    state.pop("_uncalled_module_addresses", None)
    if state.get("_intervention_spec") is None:
        state["_intervention_spec"] = _doors.intervention().types.InterventionSpec()
    if not state.get("_graph_relationship_evidence"):
        state["_graph_relationship_evidence"] = {
            "model": _doors.intervention().types.Relationship.UNKNOWN,
            "weights": _doors.intervention().types.Relationship.UNKNOWN,
            "input": _doors.intervention().types.Relationship.UNKNOWN,
            "graph": _doors.intervention().types.Relationship.UNKNOWN,
        }
    if state["_options_backward_ready"] is None:
        state["_options_backward_ready"] = False
    # Session-time DROP knobs (``track_nonfinite`` too: a loaded artifact never
    # re-records, so load restores the default and the queryable record falls
    # back to the saved-payload basis).
    for knob in _SESSION_BOOL_KNOBS:
        if state.get(knob) is None:
            state[knob] = False
    if state.get("distributed_witness") is None:
        state["distributed_witness"] = "none"
    # ``save_budget`` is FieldPolicy.DROP, so a portable artifact never
    # carries a real value; it arrives absent or None and is restored to the
    # default. A loaded trace retains nothing, so there is no ceiling to honor.
    if state.get("save_budget") is None:
        state["save_budget"] = "auto"
    if state["_options_inference_only"] is None:
        state["_options_inference_only"] = False
    if state["_options_chunk_size"] is None:
        state["_options_chunk_size"] = False
    if runnable_state.poisoned is None:
        runnable_state.poisoned = False
    for field_name in (
        "_provenance_setup_duration",
        "_cost_forward_duration",
        "_provenance_cleanup_duration",
        "_cost_func_calls_duration",
    ):
        state[field_name] = Duration(state.get(field_name) or 0.0)
    state["_cost_backward_durations"] = [
        Duration(duration) for duration in state.get("_cost_backward_durations", [])
    ]
    conditional_arm_entry_edges = _normalize_conditional_arm_entry_edges(
        state.get("_graph_conditional_arm_entry_edges") or {}
    )
    state["_graph_conditional_arm_entry_edges"] = conditional_arm_entry_edges
    for field_name in (
        "_cost_total_activation_memory",
        "_cost_total_grad_memory",
        "_cost_total_backward_memory",
        "_cost_saved_activation_memory",
        "_cost_saved_grad_memory",
        "_cost_total_param_memory",
        "_cost_total_param_grad_memory",
        "_cost_forward_peak_memory",
        "_cost_backward_peak_memory",
    ):
        if field_name.endswith("_peak_memory") and state.get(field_name, 0) is None:
            continue  # a recorded CPU None (with its reason) stays None
        state[field_name] = Bytes(state.get(field_name, 0) or 0)
    if state.get("_cost_total_autograd_memory") is not None:
        state["_cost_total_autograd_memory"] = Bytes(state["_cost_total_autograd_memory"])
    from .._io.state_keys import refuse_callable_shadowing_state_keys

    refuse_callable_shadowing_state_keys(type(trace), state)
    # Restore is keyed off the INCOMING state, never the reused object's
    # dict: a legacy artifact that baked in a stream is discarded, and a
    # reused object's prior stream must not survive a re-restore (it would
    # retain the previous trace's events and re-serialize them later).
    state.pop("_capture_events", None)
    trace.__dict__.update(state)
    # A persisted primitive-op profile (tlspec v8) is hostile artifact
    # input: validate its foreign keys whenever one is present.
    if trace.__dict__.get("_primitive_op_profile") is not None:
        from ..validation._invariants_primitive_ops import validate_loaded_primitive_profile

        validate_loaded_primitive_profile(trace)
    # Event streams never serialize (FieldPolicy.DROP), but a restored
    # trace remains a supported backward-capture target within the live
    # process, so restore installs a fresh stream EXPLICITLY here rather
    # than letting backward capture fabricate one silently on demand.
    # The stream is DETACHED, not blank: it records the restored
    # projection's pass-index base and cumulative baselines so a new
    # backward numbers itself after the preserved passes and a full
    # rebuild keeps (never silently erases) the pre-pickle projection.
    from torchlens._ir.capture_events import CaptureEvents

    trace.__dict__["_capture_events"] = CaptureEvents.detached_from(trace)
    # The projection guard is derived from the stream that just got
    # replaced; stale values from the source process (or a reused object)
    # would silently drop new passes or fold onto dead-process fold state.
    trace.__dict__.pop("_backward_projection_event_count", None)
    trace.__dict__.pop("_backward_projection_revision", None)
    trace.__dict__.pop("_backward_projection_fold_state", None)
    trace.__dict__.pop("_tl_materializing_backward_projection", None)
    if not containers_were_serialized:
        trace.__dict__.pop("_containers", None)
    if trace.__dict__.get("_module_logs") is None:
        trace._module_logs = ModuleAccessor({})
    if "_buffer_accessor" not in trace.__dict__:
        trace._buffer_accessor = None
    for layer_log in trace._layer_logs.values():
        layer_log.source_trace = trace
        for layer_pass in layer_log.ops.values():
            layer_pass.source_trace = trace
    for layer_pass in trace._layer_list:
        layer_pass.source_trace = trace
    for grad_fn_handle in trace._grad_fn_logs.values():
        grad_fn_handle.trace = trace
        if grad_fn_handle.op is not None:
            grad_fn_handle.op.grad_fn_handle = grad_fn_handle
            op_passes = getattr(grad_fn_handle.op, "ops", None)
            if op_passes is not None and hasattr(op_passes, "values"):
                for layer_pass in op_passes.values():
                    layer_pass.grad_fn_handle = grad_fn_handle
    # F44 injections stage 2: split persisted injected-op rows out of
    # layer_list BEFORE the persisted-claim validators run (the site-key
    # totality check would correctly refuse the partial family) and
    # before any accessor can observe them; forged rows refuse typed
    # inside the split.
    from .._io.injection_codec import split_restored_injected_rows

    split_restored_injected_rows(trace)
    # Persisted DROP-gated claim families become hostile artifact input at
    # this boundary. Validate them before any consumer can observe or bind
    # the restored values; wholly absent legacy families remain legal.
    from .._io.forgery_validation import validate_persisted_forgery_surfaces

    validate_persisted_forgery_surfaces(trace)
    # grind-r5 P7: load-side twin of the __getstate__ sanitation -- a
    # tampered artifact claiming capture_verified=True (or any non-bool)
    # degrades to None/no-claim; only the producer's NEGATIVE claim (with
    # a string-only reason) is adopted. Verdicts never improve across a
    # save/load cycle.
    if trace.__dict__.get("_provenance_verified") is not False:
        trace.__dict__["_provenance_verified"] = None
        trace.__dict__["_provenance_verification_reason"] = None
    elif not isinstance(trace.__dict__.get("_provenance_verification_reason"), (str, type(None))):
        trace.__dict__["_provenance_verification_reason"] = None
    # Resolve the settled capture outcome BEFORE core rehydration: adopt a
    # coherent persisted attestation, else derive from the structural
    # lattice (fail-closed to UNKNOWN on parse/coherence violations). The
    # helper reads only the restored __dict__ and never consults
    # rehydration side effects; rehydration keys on the same truthiness
    # convention independently.
    from .._capture.outcome import resolve_loaded_outcome

    trace.__dict__["_capture_outcome"] = resolve_loaded_outcome(trace.__dict__)
    # Grouping-policy stamp: adopt-or-degrade (L1). An absent stamp
    # (every pre-stamp v7 artifact) settles silently to the canonical
    # legacy settlement; an invalid one warns once and settles with the
    # violated rule's name. Monotonic: verdicts only worsen across
    # persistence, and the settled payload round-trips byte-stable. The
    # knob mirror normalizes first (a DROP-scrubbed field restores as an
    # explicit None, bypassing default fill).
    if trace.__dict__.get("_options_grouping") is None:
        trace.__dict__["_options_grouping"] = "structural"
    # L9: a DROP-scrubbed timing-provenance field restores as an explicit
    # None; normalize to the honest "unmeasured" state (a loaded pre-bump
    # artifact carries no timing evidence). The witness stays None.
    if trace.__dict__.get("grad_fn_timing_provenance") is None:
        trace.__dict__["grad_fn_timing_provenance"] = "unmeasured"
    from .._postprocess._grouping_stamp import settle_loaded_grouping_policy

    trace.__dict__["_options_grouping_policy"] = settle_loaded_grouping_policy(trace.__dict__)
    # F9: adopt the restored detached records into a fresh sealed core so
    # loaded traces rejoin the single-truth store (best-effort — an abort
    # preserves the coreless-island behavior; backward records stay
    # detached by design). See _data_classes/_trace_rehydrate.py.

    _trace_rehydrate.rehydrate_trace_core(trace)
    if pickle_module_accessor_state is not None:
        from .._io.accessor_rebuild import rebuild_trace_accessors

        rebuild_trace_accessors(
            trace,
            pickle_module_accessor_state._dict,
            pickle_module_accessor_state._list,
            pickle_module_accessor_state._pass_dict,
        )
    # Episode-ledger load validation (S7): an episode payload in the
    # restored annotations is validated fail-closed -- illegal attachment
    # refuses typed, incoherent geometry quarantines with one warning.
    if isinstance(trace.__dict__.get("annotations"), dict) and (
        "episode" in trace.__dict__["annotations"]
    ):
        from .._capture._episode_ledger import validate_loaded_episode_annotations

        validate_loaded_episode_annotations(trace)
    _doors.state().register_log(trace)
