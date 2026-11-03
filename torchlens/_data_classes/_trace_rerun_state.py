"""Trace rerun-state family: the saved inputs and options a rerun replays."""

from __future__ import annotations

import copy
import dataclasses
from typing import TYPE_CHECKING, Any

import torch

from .._constants import LAYER_PASS_LOG_FIELD_ORDER
from .._quantities import Bytes
from .._utils._record_group_table import layer_attr, op_attr, record_member
from .._utils.fail_open import fail_open
from ._state_adapter import state_items, state_restore
from ._trace_accessors import (
    _invalidate_trace_module_call_accessor_cache,
    _invalidate_trace_op_layer_accessor_caches,
)

if TYPE_CHECKING:
    from .trace import Trace


def stateful_run_warning_was_emitted(trace: Trace) -> bool:
    """Whether ``trace`` already carried its once-only stateful-run warning."""

    return bool(trace.__dict__.get("_stateful_run_warning_emitted", False))


def mark_stateful_run_warning_emitted(trace: Trace) -> None:
    """Record on ``trace`` that its once-only stateful-run warning was emitted."""

    trace.__dict__["_stateful_run_warning_emitted"] = True


__all__ = [
    "mark_stateful_run_warning_emitted",
    "stateful_run_warning_was_emitted",
    "replace_state_from",
    "refresh_matching_rerun_state_from",
    "refresh_rerun_op_from",
    "refresh_rerun_layer_logs_from",
    "copy_user_annotations",
    "merge_user_annotations",
    "refresh_rerun_trace_fields_from",
    "copy_rerun_value",
    "append_state_from",
    "is_append_buffer_side_effect_layer",
    "copy_append_last_chunk_fields",
    "copy_append_metadata_value",
    "refresh_appended_tensor_metadata",
    "discharge_against",
]


#: Values ``copy.deepcopy`` returns unchanged, so a rerun refresh hands them over as they are.
_ATOMIC_RERUN_VALUE_TYPES: frozenset[type] = frozenset(
    {type(None), bool, int, float, complex, str, bytes, torch.dtype, torch.device}
)


def replace_state_from(target: Trace, trace: Trace) -> None:
    """Atomically replace this log's run-state from another ``Trace``.

    This method is intended for intervention rerun. The rerun engine builds
    ``trace`` off to the side and calls this only after validation ops.
    The final state replacement uses one state-restore pass over the new fields
    to minimize torn-state windows. Concurrent reads during rerun are
    unsupported; no lock is taken.

    Parameters
    ----------
    target:
        The trace that receives the state.
    trace:
        Fully postprocessed fresh log whose graph, layer containers, accessors,
        output metadata, shape/hash fields and per-pass entries replace the
        target's current run-state in place.
    """

    preserved_fields = (
        "name",
        "_parent_run",
        "_intervention_spec",
        "_transform",
        "_options_save_raw_input",
        "batch_render",
        "_output_transform",
        "_options_save_transformed_output",
        "state_history",
        "_warned_direct_write",
        "_warned_mutate_in_place",
        "_provenance_model_object_id",
        "_source_class_qualname",
        "_provenance_param_hash_quick",
        "_provenance_param_hash_full",
        "_provenance_root_entry_point",
        "_provenance_input_object_id",
        "_provenance_input_signature_hash",
        "_provenance_is_appended",
        "_append_sequence_id",
        "_append_history",
        "_graph_relationship_evidence",
        "_source_model_ref",
        "_has_direct_writes",
        "_spec_revision",
        "_out_recipe_revision",
        "_provenance_input_annotations",
    )
    current_state = dict(state_items(target))
    preserved_trace_user_annotations = target._copy_user_annotations(
        current_state.get("annotations")
    )
    preserved_state = {field_name: current_state.get(field_name) for field_name in preserved_fields}
    # DERIVED annotation blobs (feature maps / RDM / MDS / scree, the
    # ``_DERIVED_ANNOTATION_BLOB_PREFIXES`` namespaces) are NOT preserved:
    # they were computed from the replaced run's activations and would
    # render stale data over the new run's stimuli. USER
    # ``annotate(data=...)`` blobs (bare ``layer:``/``op:`` keys) are the
    # user's own attachments and survive like the user annotations above.
    preserved_state["_annotation_blobs"] = target._user_annotation_blobs()
    replacement_state = dict(state_items(trace))
    replacement_state.update(preserved_state)
    replacement_state["annotations"] = target._merge_user_annotations(
        target._copy_rerun_value(getattr(trace, "annotations", {})),
        preserved_trace_user_annotations,
    )
    state_restore(target, replacement_state)
    target.__dict__.pop("_validation_replay_status", None)
    _invalidate_trace_op_layer_accessor_caches(target)
    _invalidate_trace_module_call_accessor_cache(target)
    target._rebind_fork_owner_refs()


def refresh_matching_rerun_state_from(trace: Trace, new_log: Trace) -> bool:
    """Refresh payload-bearing fields from a same-shape rerun.

    Parameters
    ----------
    trace:
        The owning trace.
    new_log:
        Fully captured and postprocessed rerun candidate.

    Returns
    -------
    bool
        True when the existing graph containers were refreshed in place.
        False means labels did not match closely enough and callers should
        use ``replace_state_from``.
    """

    old_raw_labels = tuple(layer._layer_label_raw for layer in trace._layer_list)
    new_raw_labels = tuple(layer._layer_label_raw for layer in new_log._layer_list)
    old_final_labels = tuple(layer.layer_label for layer in trace._layer_list)
    new_final_labels = tuple(layer.layer_label for layer in new_log._layer_list)
    if old_raw_labels != new_raw_labels or old_final_labels != new_final_labels:
        return False

    # Pair the two op sequences POSITIONALLY, never through a label -> op map.
    # Neither `_layer_label_raw` nor `layer_label` is pass-qualified, so every
    # pass of a multi-pass (recurrent) layer shares both keys: a dict keyed by
    # either collapses an N-pass layer to its last pass and then refreshes all
    # N existing passes from that one op, silently overwriting the earlier
    # passes' activations and pass labels. The label-sequence equality checked
    # just above is exactly the precondition that makes index i of one list the
    # same op as index i of the other.
    for layer, new_layer in zip(trace._layer_list, new_log._layer_list):
        refresh_rerun_op_from(trace, layer, new_layer)
    refresh_rerun_layer_logs_from(trace, new_log)
    refresh_rerun_trace_fields_from(trace, new_log)
    trace.__dict__.pop("_validation_replay_status", None)
    _invalidate_trace_op_layer_accessor_caches(trace)
    _invalidate_trace_module_call_accessor_cache(trace)
    trace._rebind_fork_owner_refs()
    return True


def refresh_rerun_op_from(trace: Trace, layer: Any, new_layer: Any) -> None:
    """Copy rerun fields into one existing ``Op``.

    Parameters
    ----------
    trace:
        The owning trace.
    layer:
        Existing operation record retained by the fast path.
    new_layer:
        Fresh rerun record supplying current payloads and per-call metadata.
    """

    preserved_fields = {
        "source_trace",
        "_source_trace_ref",
        "input_ops",
        "input_activations",
        "input_shapes",
        "input_dtypes",
        "input_memory",
        "num_inputs",
        "is_in_conditional_body",
    }
    new_layer_state = dict(state_items(new_layer))
    preserved_user_annotations = copy_user_annotations(trace, getattr(layer, "annotations", None))
    copies: dict[str, Any] = {}  # fields both loops below write are copied once
    for field_name in LAYER_PASS_LOG_FIELD_ORDER:
        if field_name in preserved_fields:
            continue
        value = copies[field_name] = copy_rerun_value(trace, new_layer_state.get(field_name))
        if field_name == "annotations":
            value = merge_user_annotations(trace, value, preserved_user_annotations)
        layer._internal_set(field_name, value)
    for field_name in (
        "out_ref",
        "grad_ref",
        "_pending_blob_id",
        "_pending_transformed_out_blob_id",
        "_pending_grad_blob_id",
        "_pending_transformed_grad_blob_id",
        "annotations",
        "edits",
        "container_spec",
        "args_template",
        "kwargs_template",
        "_edge_uses",
    ):
        if hasattr(new_layer, op_attr(field_name)):
            if field_name not in copies:
                copies[field_name] = copy_rerun_value(trace, new_layer_state.get(field_name))
            value = copies[field_name]
            if field_name == "annotations":
                value = merge_user_annotations(trace, value, preserved_user_annotations)
            layer._internal_set(
                field_name,
                value,
            )
    layer.source_trace = trace


def refresh_rerun_layer_logs_from(trace: Trace, new_log: Trace) -> None:
    """Refresh aggregate ``Layer`` records from a same-shape rerun.

    Parameters
    ----------
    trace:
        The owning trace.
    new_log:
        Fresh rerun trace with layer aggregates already postprocessed.
    """

    new_layer_logs = new_log._layer_logs or {}
    for label, layer_log in trace._layer_logs.items():
        new_layer_log = new_layer_logs.get(label)
        if new_layer_log is None:
            continue
        preserved_user_annotations = copy_user_annotations(
            trace, getattr(layer_log, "annotations", None)
        )
        for field_name, value in state_items(new_layer_log):
            if field_name in {"_source_trace_ref", "ops"}:
                continue
            copied_value = copy_rerun_value(trace, value)
            if field_name == "annotations":
                copied_value = merge_user_annotations(
                    trace,
                    copied_value,
                    preserved_user_annotations,
                )
            setattr(layer_log, layer_attr(field_name), copied_value)
        layer_log.source_trace = trace


def copy_user_annotations(trace: Trace, annotations: Any) -> dict[str, Any] | None:
    """Copy the reserved user annotation namespace from a mapping.

    Parameters
    ----------
    trace:
        The owning trace.
    annotations:
        Existing annotation mapping.

    Returns
    -------
    dict[str, Any] | None
        Copied user namespace, or ``None`` when absent.
    """

    if not isinstance(annotations, dict):
        return None
    user_annotations = annotations.get("user")
    if not isinstance(user_annotations, dict):
        return None
    return copy_rerun_value(trace, user_annotations)


def merge_user_annotations(
    trace: Trace,
    fresh_annotations: Any,
    preserved_user_annotations: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge preserved user annotations into fresh internal annotations.

    Parameters
    ----------
    trace:
        The owning trace.
    fresh_annotations:
        Annotation mapping from the fresh rerun.
    preserved_user_annotations:
        Previously stored ``annotations["user"]`` mapping.

    Returns
    -------
    dict[str, Any]
        Fresh annotations plus the preserved user namespace.
    """

    merged = fresh_annotations if isinstance(fresh_annotations, dict) else {}
    if preserved_user_annotations is not None:
        merged["user"] = copy_rerun_value(trace, preserved_user_annotations)
    return merged


def refresh_rerun_trace_fields_from(trace: Trace, new_log: Trace) -> None:
    """Refresh trace-level run fields without replacing graph containers.

    Parameters
    ----------
    trace:
        The owning trace.
    new_log:
        Fresh rerun trace supplying current run metadata.
    """

    # Share the fresh return itself; the metadata copier rebuilds containers.
    trace._output_slot = new_log._output_slot
    field_names = (
        "transformed_output",
        "_options_save_transformed_output",
        "_autograd_has_grads",
        "_options_seed",
        "_options_chunk_size",
        "_provenance_input_object_id",
        "_provenance_input_signature_hash",
        "_provenance_fingerprint",
        "_raw_event_shape_hash",
        "_raw_call_fingerprint",
        "_num_saved_ops",
        "_cost_saved_activation_memory",
        "_cost_total_activation_memory",
        "_cost_saved_grad_memory",
        "_cost_total_grad_memory",
        "_cost_total_backward_memory",
        "_cost_total_autograd_memory",
        "_cost_pinned_autograd_memory",
        "_cost_forward_peak_memory",
        "_cost_forward_peak_memory_reason",
        "_provenance_forward_memory_backend",
        "_forward_peak_memory_pair",
        "_cost_backward_peak_memory",
        "backward_peak_memory_reason",
        "output_layers",
        "output_layers_by_pass",
        "output_layers_by_module_call",
        "_output_container_specs_by_raw_label",
    )
    for field_name in field_names:
        if hasattr(new_log, field_name):
            setattr(trace, field_name, copy_rerun_value(trace, getattr(new_log, field_name)))
    # DERIVED annotation blobs (feature-map/RDM/MDS/scree namespaces) were
    # computed from the replaced activations and are invalidated; USER
    # ``annotate(data=...)`` blobs survive the same-shape refresh.
    trace._annotation_blobs = trace._user_annotation_blobs()
    trace.facet_functions = getattr(new_log, "facet_functions", None)


def copy_rerun_value(trace: Trace, value: Any) -> Any:
    """Copy rerun metadata while keeping tensor payload identities.

    Parameters
    ----------
    trace:
        The owning trace.
    value:
        Value copied from the fresh rerun trace.

    Returns
    -------
    Any
        Copied metadata container, or the original tensor/object when
        copying would be incorrect or unnecessary.
    """

    if isinstance(value, torch.Tensor) or type(value) in _ATOMIC_RERUN_VALUE_TYPES:
        return value
    if isinstance(value, list):
        return [copy_rerun_value(trace, item) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_rerun_value(trace, item) for item in value)
    if isinstance(value, dict):
        return {
            copy_rerun_value(trace, key): copy_rerun_value(trace, item)
            for key, item in value.items()
        }
    from ..intervention.types import REPLAY_TEMPLATE_TYPES

    if type(value) in REPLAY_TEMPLATE_TYPES:
        # Field by field, so a parameter literal keeps its identity (a deep copy cloned it).
        names = [field.name for field in dataclasses.fields(value)]
        return type(value)(**{n: copy_rerun_value(trace, getattr(value, n)) for n in names})
    return fail_open(lambda: copy.deepcopy(value), lambda _error: value)


def append_state_from(target: Trace, trace: Trace) -> None:
    """Merge compatible chunk outs from ``trace`` into this log.

    Parameters
    ----------
    target:
        The trace that receives the state.
    trace:
        Freshly captured append chunk whose topology and tensor metadata
        have already been validated against this log.
    """

    old_by_label = {
        key: layer
        for layer in target._layer_list
        for key in (layer._layer_label_raw, layer.layer_label)
    }
    # By position, not raw label: a reused module repeats its raw label across passes.
    for layer, new_layer in zip(target._layer_list, trace._layer_list, strict=True):
        if not (
            record_member(layer, "is_buffer", False)
            or is_append_buffer_side_effect_layer(target, layer, old_by_label)
        ):
            layer._append_tensor_from(new_layer, "out")
            layer._append_tensor_from(new_layer, "transformed_out")
        copy_append_last_chunk_fields(target, layer, new_layer)
        refresh_appended_tensor_metadata(target, layer)
    target._autograd_has_grads = target._autograd_has_grads or trace._autograd_has_grads
    target._options_seed = trace._options_seed
    target._provenance_input_object_id = trace._provenance_input_object_id
    target._provenance_input_signature_hash = trace._provenance_input_signature_hash
    # Concatenated captures have no single model-return object.
    target._output_slot = None
    target._rebind_fork_owner_refs()


def is_append_buffer_side_effect_layer(
    trace: Trace, layer: Any, layer_by_label: dict[str, Any]
) -> bool:
    """Return whether ``layer`` only feeds buffer version side effects.

    Parameters
    ----------
    trace:
        The owning trace.
    layer:
        Candidate layer being considered for append tensor concatenation.
    layer_by_label:
        Mapping from raw and final labels to layers in this trace.

    Returns
    -------
    bool
        True when every tracked child is a buffer version node created by a
        buffer write.
    """

    child_labels = list(getattr(layer, "children", []))
    if not child_labels:
        return False
    saw_buffer_write = False
    for child_label in child_labels:
        child_layer = layer_by_label.get(child_label)
        if child_layer is None:
            return False
        if not (
            record_member(child_layer, "is_buffer", False)
            and getattr(child_layer, "buffer_write_kind", None) is not None
        ):
            return False
        saw_buffer_write = True
    return saw_buffer_write


def copy_append_last_chunk_fields(trace: Trace, layer: Any, new_layer: Any) -> None:
    """Copy per-call metadata fields from the last appended chunk.

    Parameters
    ----------
    trace:
        The owning trace.
    layer:
        Existing accumulated layer pass.
    new_layer:
        New chunk layer pass supplying per-call state.
    """

    for field_name in (
        "func_duration",
        "flops_forward",
        "flops_backward",
        "func_rng_states",
        "func_autocast_state",
        "arg_names",
        "num_args_total",
        "num_pos_args",
        "num_kwargs",
        "non_tensor_pos_args",
        "non_tensor_kwargs",
        "func_non_tensor_args",
        "is_inplace",
        "grad_fn_class_name",
        "grad_fn_object_id",
        "edits",
        "annotations",
    ):
        attr = op_attr(field_name)
        if hasattr(new_layer, attr):
            layer._internal_set(
                field_name, copy_append_metadata_value(trace, getattr(new_layer, attr))
            )


def copy_append_metadata_value(trace: Trace, value: Any) -> Any:
    """Copy metadata from the last chunk without failing on non-leaf tensors.

    Parameters
    ----------
    trace:
        The owning trace.
    value:
        Metadata value from the new chunk.

    Returns
    -------
    Any
        Best-effort copied value.
    """

    if isinstance(value, torch.Tensor):
        from .._utils.tensor_utils import safe_copy

        return safe_copy(value, detach_tensor=True)
    if isinstance(value, list):
        return [copy_append_metadata_value(trace, item) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_append_metadata_value(trace, item) for item in value)
    if isinstance(value, dict):
        return {
            copy_append_metadata_value(trace, key): copy_append_metadata_value(trace, item)
            for key, item in value.items()
        }
    try:
        return copy.deepcopy(value)
    except RuntimeError:
        return value


def refresh_appended_tensor_metadata(trace: Trace, layer: Any) -> None:
    """Refresh shape, dtype, and memory fields after tensor concatenation.

    Parameters
    ----------
    trace:
        The owning trace.
    layer:
        Layer pass whose tensor fields may have been concatenated.
    """

    for tensor_field, shape_field, dtype_field, memory_field in (
        ("out", "shape", "dtype", "activation_memory"),
        (
            "transformed_out",
            "transformed_out_shape",
            "transformed_out_dtype",
            "transformed_activation_memory",
        ),
        ("grad", "grad_shape", "grad_dtype", "gradient_memory"),
        (
            "transformed_grad",
            "transformed_grad_shape",
            "transformed_grad_dtype",
            "transformed_gradient_memory",
        ),
    ):
        value = record_member(layer, tensor_field, None)
        if isinstance(value, torch.Tensor):
            from .._utils.tensor_utils import get_memory_amount

            layer._internal_set(shape_field, tuple(value.shape))
            layer._internal_set(dtype_field, value.dtype)
            layer._internal_set(memory_field, Bytes(get_memory_amount(value)))
        else:
            layer._internal_set(shape_field, None)
            layer._internal_set(dtype_field, None)
            layer._internal_set(memory_field, None)


def discharge_against(trace: Trace, real_trace: Trace) -> Any:
    """Discharge this structure-only trace's hypotheses against a real run.

    DOCUMENTED-UNSTABLE surface (L7a; no deprecation shim owed on
    rename). Only defined on a structure-only capture; ``real_trace``
    must be an ordinary settled COMPLETE capture of the same graph.
    Returns a frozen ``StructureDischarge`` record (per-claim table +
    overall verdict); neither trace is mutated — the verdict registers in
    a weak-keyed side table consulted by the structure-only capability
    chokepoint (a REFUTED discharge flips hypothesis consumers to typed
    refusals). See ``torchlens._capture.structure_only.discharge_against``
    for the full join/precondition contract.
    """

    from .._capture.structure_only import discharge_against as _discharge

    return _discharge(trace, real_trace)
