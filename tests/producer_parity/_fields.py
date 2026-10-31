"""Canonical field universe for the producer-unification migration.

Three checked-in tables live here:

* ``TIER_CLASSIFICATION`` — the closed Tier-F / Tier-R classification of every
  field of every dataclass reachable from ``OpEvent`` (the record universe this
  phase; ``OpRecord`` joins the walk in P1). The walker test fails on any
  unclassified field, so the inventory is mechanically closed, never
  hand-maintained-and-stale.
* ``VOLATILE_BUCKETS`` — the id-derived token buckets the parity comparator
  treats structurally (Check A) and attests within-leg (Check B), each with its
  declared independent observation root (or an explicit presence-only residual
  marker).
* ``VOLATILE_PATHS`` — non-token volatile paths (timestamps, durations, memory,
  RNG state) with their named canonicalizers.

Tier semantics (design-of-record section 2.4):

* ``F`` — portable fact: plain data, survives save/load unchanged.
* ``R`` — runtime-bearing: holds a live object, callable, payload, or arbitrary
  user value. Each R entry names its retention disposition.
"""

from __future__ import annotations

from torchlens.fastlog.types import CaptureSpec
from torchlens.ir.container import ContainerSpec
from torchlens.ir.events import (
    ArgTemplateRef,
    BlobRef,
    FunctionCallRef,
    ModuleFrame,
    OpEvent,
    OutputRef,
    ParentEdge,
)
from torchlens.ir.intervention import FireResult, InterventionTemplateRef
from torchlens.ir.op_record import (
    AncestryFacet,
    AnnotationsFacet,
    AutogradFacet,
    ControlFacet,
    GraphFacet,
    IngestExtras,
    InterventionFacet,
    ModulesFacet,
    OpAmendment,
    OpCore,
    OpRecord,
    ParamsFacet,
    PolicyFacet,
    RecordingFacet,
    TransformFacet,
)
from torchlens.ir.predicate import ModuleStackFrame, RecordContext
from torchlens.ir.refs import DeferredRef, DeviceRef, DtypeRef, ParamRef, TensorRef
from torchlens.ir.semantics import BackendSemantics, CapturePolicy

# Roots for the Tier walker: every dataclass the record graph can carry,
# including classes hidden behind ``object``-typed fields (record_context,
# capture_spec) that annotation walking alone cannot reach.
WALKER_ROOTS: tuple[type, ...] = (
    OpEvent,
    FunctionCallRef,
    OutputRef,
    TensorRef,
    DeferredRef,
    ArgTemplateRef,
    ParentEdge,
    ParamRef,
    ModuleFrame,
    BackendSemantics,
    CapturePolicy,
    FireResult,
    InterventionTemplateRef,
    RecordContext,
    CaptureSpec,
    ContainerSpec,
    BlobRef,
    ModuleStackFrame,
    DeviceRef,
    DtypeRef,
    # P1: the decomposed record graph joins the walk
    OpRecord,
    OpCore,
    GraphFacet,
    ModulesFacet,
    AncestryFacet,
    AutogradFacet,
    TransformFacet,
    ControlFacet,
    ParamsFacet,
    AnnotationsFacet,
    PolicyFacet,
    RecordingFacet,
    InterventionFacet,
    IngestExtras,
    OpAmendment,
)

# Namespace for resolving string annotations (events.py imports its ref types
# under TYPE_CHECKING, so ``typing.get_type_hints`` needs help at runtime).
ANNOTATION_NAMESPACE: dict[str, type] = {cls.__name__: cls for cls in WALKER_ROOTS}

# Retention dispositions for Tier-R fields.
R_JOURNAL = "R:journal-lifetime"  # retained while the journal lives (matches today)
R_SCRUBBED = "R:release-scrubbed"  # nulled by release_runtime_sidecars
R_COMPAT = "R:compat-OpEvent-only"  # never crosses to OpRecord; dies with OpEvent (S15)

TIER_CLASSIFICATION: dict[str, str] = {
    # ---- OpEvent -----------------------------------------------------------
    "OpEvent.kind": "F",
    "OpEvent.label_raw": "F",
    "OpEvent.layer_label_raw": "F",
    "OpEvent.layer_type": "F",
    "OpEvent.raw_index": "F",
    "OpEvent.type_index": "F",
    "OpEvent.step_index": "F",
    "OpEvent.source_trace": R_SCRUBBED,
    "OpEvent.source_trace_id": "F",
    "OpEvent.tracing_finished": "F",
    "OpEvent.construction_done": "F",
    "OpEvent.function": "F",  # nested; members classified below
    "OpEvent.output": "F",
    "OpEvent.templates": "F",
    "OpEvent.parents": "F",
    "OpEvent.parent_arg_positions": "F",
    "OpEvent._edge_uses": "F",
    "OpEvent.params": "F",
    "OpEvent.parent_params": R_SCRUBBED,
    "OpEvent.module_stack": "F",
    "OpEvent.modules": "F",
    "OpEvent.backend_semantics": "F",
    "OpEvent.policy": "F",
    "OpEvent.predicate_matched": "F",
    "OpEvent.pass_index": "F",
    "OpEvent.grad_fn_class_qualname": "F",
    "OpEvent.grad_fn_handle": R_COMPAT,  # index-owned after P2; scrub is a named fix
    "OpEvent.equivalence_class": "F",
    "OpEvent.is_transform": "F",
    "OpEvent.transform_kind": "F",
    "OpEvent.transform_chain": "F",
    "OpEvent.transform_config": R_JOURNAL,  # arbitrary user values; never recursively frozen
    "OpEvent.transform_fn_name": "F",
    "OpEvent.transform_fn_qualname": "F",
    "OpEvent.transform_fn_source": R_JOURNAL,
    "OpEvent.is_output_parent": "F",
    "OpEvent.has_internal_source_ancestor": "F",
    "OpEvent.internal_source_ancestors": "F",
    "OpEvent.input_ancestors": "F",
    "OpEvent.root_ancestors": "F",
    "OpEvent.func_call_id": "F",
    "OpEvent.is_bottom_level": "F",
    "OpEvent.is_scalar_bool": "F",
    "OpEvent.bool_value": "F",
    "OpEvent.intervention_fired": "F",
    "OpEvent.intervention_replaced": "F",
    "OpEvent.fire_results": "F",
    "OpEvent.intervention_template_ref": "F",
    "OpEvent.record_context": R_JOURNAL,
    "OpEvent.capture_spec": R_JOURNAL,
    "OpEvent.unattributed_tensor_args": "F",
    "OpEvent.dropped_edge_tensor_args": "F",
    "OpEvent.input_was_parameter": "F",
    "OpEvent.seq": "F",
    # ---- FunctionCallRef ---------------------------------------------------
    "FunctionCallRef.func": R_JOURNAL,
    "FunctionCallRef.func_name": "F",
    "FunctionCallRef.func_qualname": "F",
    "FunctionCallRef.func_call_id": "F",
    "FunctionCallRef.code_context": R_JOURNAL,
    "FunctionCallRef.func_duration": "F",
    "FunctionCallRef.flops_forward": "F",
    "FunctionCallRef.flops_backward": "F",
    "FunctionCallRef.func_rng_states": R_JOURNAL,
    "FunctionCallRef.func_autocast_state": R_JOURNAL,
    "FunctionCallRef.arg_names": "F",
    "FunctionCallRef.num_args_total": "F",
    "FunctionCallRef.num_pos_args": "F",
    "FunctionCallRef.num_kwargs": "F",
    "FunctionCallRef.non_tensor_pos_args": R_JOURNAL,
    "FunctionCallRef.non_tensor_kwargs": R_JOURNAL,
    "FunctionCallRef.func_non_tensor_args": R_JOURNAL,
    "FunctionCallRef.is_inplace": "F",
    "FunctionCallRef.func_config": R_JOURNAL,
    "FunctionCallRef.func_id": "F",
    # ---- OutputRef ---------------------------------------------------------
    "OutputRef.tensor": "F",
    "OutputRef.transformed_tensor": "F",
    "OutputRef.has_saved_activation": "F",
    "OutputRef.output_device": "F",
    "OutputRef.activation_transform": R_SCRUBBED,
    "OutputRef.detach_saved_activations": "F",
    "OutputRef.visualizer_path": "F",
    "OutputRef.multi_output_index": "F",
    "OutputRef.in_multi_output": "F",
    "OutputRef.container_path": "F",
    "OutputRef.container_spec": "F",
    "OutputRef.child_versions": "F",
    # ---- TensorRef ---------------------------------------------------------
    "TensorRef.label_raw": "F",
    "TensorRef.shape": "F",
    "TensorRef.dtype": "F",
    "TensorRef.device": "F",
    "TensorRef.requires_grad": "F",
    "TensorRef.memory": "F",
    "TensorRef.payload": R_SCRUBBED,
    "TensorRef.blob_ref": "F",
    "TensorRef.backend_handle_id": "F",  # id-derived token: volatile bucket, portable shape
    # ---- DeferredRef -------------------------------------------------------
    "DeferredRef.backend": "F",
    "DeferredRef.handle_id": "F",
    "DeferredRef.blob_ref": "F",
    "DeferredRef.inferred_shape": "F",
    "DeferredRef.inferred_dtype": "F",
    "DeferredRef.materialize_fn": R_JOURNAL,
    # ---- ArgTemplateRef ----------------------------------------------------
    "ArgTemplateRef.saved_args": R_SCRUBBED,
    "ArgTemplateRef.saved_kwargs": R_SCRUBBED,
    "ArgTemplateRef.args_template": R_SCRUBBED,
    "ArgTemplateRef.kwargs_template": R_SCRUBBED,
    "ArgTemplateRef.has_saved_args": "F",
    # ---- ParentEdge --------------------------------------------------------
    "ParentEdge.parent_label_raw": "F",
    "ParentEdge.arg_position": "F",
    "ParentEdge.edge_use": "F",
    # ---- ParamRef ----------------------------------------------------------
    "ParamRef.barcode": "F",  # random token: volatile bucket, portable shape
    "ParamRef.address": "F",
    "ParamRef.shape": "F",
    "ParamRef.dtype": "F",
    "ParamRef.trainable": "F",
    "ParamRef.module_address": "F",
    # ---- ModuleFrame -------------------------------------------------------
    "ModuleFrame.address": "F",
    "ModuleFrame.address_normalized": "F",
    "ModuleFrame.module_type": "F",
    "ModuleFrame.call_index": "F",
    "ModuleFrame.fx_qualpath": "F",
    "ModuleFrame.entry_argnames": "F",
    # ---- BackendSemantics --------------------------------------------------
    "BackendSemantics.backend_grad_handle": R_SCRUBBED,  # named release fix
    "BackendSemantics.grad_fn_class_name": "F",
    "BackendSemantics.autograd_memory": "F",
    "BackendSemantics.num_autograd_tensors": "F",
    "BackendSemantics.mutated_input_positions": "F",
    "BackendSemantics.aliased_output_inputs": "F",
    "BackendSemantics.unknown_aliasing": "F",
    "BackendSemantics.bytes_delta_at_call": "F",
    "BackendSemantics.bytes_peak_at_call": "F",
    # ---- CapturePolicy -----------------------------------------------------
    # R47-2: the six write-only fields (must_keep_topology, requires_isolation,
    # save_args, save_code, save_rng, stream) were removed with the dataclass;
    # only the consumed facts stay pinned.
    "CapturePolicy.save_payload": "F",
    "CapturePolicy.save_grad": "F",
    "CapturePolicy.save_mode": "F",
    # ---- FireResult --------------------------------------------------------
    "FireResult.plan_id": "F",
    "FireResult.site_label": "F",
    "FireResult.fired_at_capture_index": "F",
    "FireResult.pre_hook_shape": "F",
    "FireResult.post_hook_shape": "F",
    "FireResult.pre_hook_dtype": "F",
    "FireResult.post_hook_dtype": "F",
    "FireResult.replaced": "F",
    "FireResult.fire_record": R_JOURNAL,
    # ---- InterventionTemplateRef -------------------------------------------
    "InterventionTemplateRef.template_id": "F",
    "InterventionTemplateRef.spec_revision": "F",
    "InterventionTemplateRef.template_kind": "F",
    "InterventionTemplateRef.template_args": R_COMPAT,
    # ---- RecordContext -----------------------------------------------------
    "RecordContext.kind": "F",
    "RecordContext.label": "F",
    "RecordContext.raw_label": "F",
    "RecordContext.pass_index": "F",
    "RecordContext.event_index": "F",
    "RecordContext.step_index": "F",
    "RecordContext.layer_type": "F",
    "RecordContext.type_index": "F",
    "RecordContext.raw_index": "F",
    "RecordContext.func_name": "F",
    "RecordContext.address": "F",
    "RecordContext.module_type": "F",
    "RecordContext.module_pass_index": "F",
    "RecordContext.module_stack": "F",
    "RecordContext.recent_events": R_JOURNAL,
    "RecordContext.recent_ops": R_JOURNAL,
    "RecordContext.parent_labels": "F",
    "RecordContext.input_output_address": "F",
    "RecordContext.shape": "F",
    "RecordContext.dtype": "F",
    "RecordContext.tensor_device": "F",
    "RecordContext.tensor_requires_grad": "F",
    "RecordContext.output_index": "F",
    "RecordContext.is_bottom_level_func": "F",
    "RecordContext.time_since_pass_start": "F",
    "RecordContext.sample_id": "F",
    "RecordContext.label_raw": "F",
    "RecordContext.label_prefix": "F",
    "RecordContext.func_call_id": "F",
    "RecordContext.parent_labels_raw": "F",
    "RecordContext.is_output_parent": "F",
    "RecordContext.backend_requires_isolation": "F",
    "RecordContext.is_scalar_bool": "F",
    "RecordContext.bool_value": "F",
    "RecordContext.is_transform": "F",
    "RecordContext.transform_kind": "F",
    "RecordContext.window_miss": "F",
    "RecordContext.output_of_module_calls": "F",
    # ---- CaptureSpec -------------------------------------------------------
    "CaptureSpec.save_out": "F",
    "CaptureSpec.save_metadata": "F",
    "CaptureSpec.keep_grad": "F",
    "CaptureSpec.device": "F",
    "CaptureSpec.dtype": "F",
    "CaptureSpec.save_mode": "F",
    # ---- ContainerSpec -----------------------------------------------------
    "ContainerSpec.kind": "F",
    "ContainerSpec.length": "F",
    "ContainerSpec.keys": "F",
    "ContainerSpec.fields": "F",
    "ContainerSpec.type_module": "F",
    "ContainerSpec.type_qualname": "F",
    "ContainerSpec.child_specs": "F",
    "ContainerSpec.literal_value": R_JOURNAL,
    "ContainerSpec.aux_data": R_JOURNAL,
    "ContainerSpec.lossy_reconstruction": "F",
    # ---- container path keys (reachable via ContainerSpec) ------------------
    "DataclassField.name": "F",
    "DictKey.key": "F",
    "HFKey.key": "F",
    "NamedField.name": "F",
    "TupleIndex.index": "F",
    # ---- BlobRef -----------------------------------------------------------
    "BlobRef.uri": "F",
    "BlobRef.format": "F",
    "BlobRef.dtype": "F",
    "BlobRef.shape": "F",
    "BlobRef.byte_length": "F",
    "BlobRef.sha256": "F",
    # ---- ModuleStackFrame --------------------------------------------------
    "ModuleStackFrame.address": "F",
    "ModuleStackFrame.module_type": "F",
    "ModuleStackFrame.module_id": "F",  # id-derived token
    "ModuleStackFrame.pass_index": "F",
    # ---- DeviceRef / DtypeRef ----------------------------------------------
    "DeviceRef.backend": "F",
    "DeviceRef.name": "F",
    "DtypeRef.backend": "F",
    "DtypeRef.name": "F",
    # ---- OpRecord graph (P1) -------------------------------------------------
    "OpCore.seq": "F",
    "OpCore.kind": "F",
    "OpCore.label_raw": "F",
    "OpCore.layer_label_raw": "F",
    "OpCore.layer_type": "F",
    "OpCore.raw_index": "F",
    "OpCore.type_index": "F",
    "OpCore.step_index": "F",
    "OpCore.pass_index": "F",
    "OpCore.parents": "F",
    "OpCore.output": "F",
    "OpCore.is_bottom_level": "F",
    "OpCore.func_call_id": "F",
    "GraphFacet.parent_arg_positions": "F",
    "GraphFacet.edge_uses": "F",
    "GraphFacet.unattributed_tensor_args": "F",
    "GraphFacet.dropped_edge_tensor_args": "F",
    "GraphFacet.is_output_parent": "F",
    "GraphFacet.input_was_parameter": "F",
    "GraphFacet.equivalence_class": "F",
    "ModulesFacet.module_stack": "F",
    "ModulesFacet.modules": "F",
    "AncestryFacet.input_ancestors": "F",
    "AncestryFacet.internal_source_ancestors": "F",
    "AncestryFacet.root_ancestors": "F",
    "AncestryFacet.has_internal_source_ancestor": "F",
    "AutogradFacet.grad_fn_class_qualname": "F",
    "TransformFacet.is_transform": "F",
    "TransformFacet.transform_kind": "F",
    "TransformFacet.transform_chain": "F",
    "TransformFacet.transform_config": R_JOURNAL,
    "TransformFacet.transform_fn_name": "F",
    "TransformFacet.transform_fn_qualname": "F",
    "TransformFacet.transform_fn_source": R_JOURNAL,
    "TransformFacet.fn_code_location": "F",
    "ControlFacet.is_scalar_bool": "F",
    "ControlFacet.bool_value": "F",
    "ParamsFacet.params": "F",
    "ParamsFacet.parent_params": R_SCRUBBED,
    "AnnotationsFacet.annotations": R_JOURNAL,
    "PolicyFacet.backend_semantics": "F",
    "PolicyFacet.policy": "F",
    "PolicyFacet.predicate_matched": "F",
    "PolicyFacet.tracing_finished": "F",
    "PolicyFacet.construction_done": "F",
    "RecordingFacet.record_context": R_JOURNAL,
    "RecordingFacet.capture_spec": R_JOURNAL,
    "InterventionFacet.intervention_fired": "F",
    "InterventionFacet.intervention_replaced": "F",
    "InterventionFacet.fire_results": "F",
    "OpRecord.core": "F",
    "OpRecord.function": "F",
    "OpRecord.templates": "F",
    "OpRecord.graph": "F",
    "OpRecord.modules_facet": "F",
    "OpRecord.ancestry": "F",
    "OpRecord.autograd": "F",
    "OpRecord.transform": "F",
    "OpRecord.control": "F",
    "OpRecord.params_facet": "F",
    "OpRecord.annotations_facet": "F",
    "OpRecord.policy_facet": "F",
    "OpRecord.recording": "F",
    "OpRecord.intervention": "F",
    "IngestExtras.source_trace": R_JOURNAL,
    "IngestExtras.source_trace_id": "F",
    "IngestExtras.grad_fn_handle": R_COMPAT,
    "OpAmendment.seq": "F",
    "OpAmendment.run_nonce": "F",
    "OpAmendment.target_seq": "F",
    "OpAmendment.target_label_raw": "F",
    "OpAmendment.family": "F",
    "OpAmendment.patch": R_JOURNAL,
}


# --------------------------------------------------------------------------
# Volatile-token buckets for the two-check parity discharge (DoR section 6.2).
#
# ``independent_root`` names the live-runtime observation path Check B uses;
# it must NEVER route through the draft/record under test (review v4 note N1).
# ``attested=False`` buckets are the named presence-only residual class.
# ``retention_premise`` records what keeps Check A's partition sound against
# id() reuse (review v4 note N4).
# --------------------------------------------------------------------------

VOLATILE_BUCKETS: dict[str, dict[str, object]] = {
    "grad_fn_object_id": {
        "attested": True,
        "independent_root": (
            "the live output tensor's selected grad_fn at the semantic-selection "
            "boundary in ops.py (_select_user_grad_fn area, ops.py:5187): the shim "
            "records id() of the handle TorchLens SELECTS (tl_user_grad_fn when "
            "present, else out.grad_fn) BEFORE the tl_user_grad_fn marker is "
            "deleted. Rooted in the tensor object, never the draft."
        ),
        "retention_premise": (
            "grad_fn_handles_by_label_raw holds STRONG references for the whole "
            "run (plain dict), so grad_fn ids cannot be recycled while the "
            "comparator runs. A weak-ref handle index invalidates Check A for "
            "this bucket (test_attestation asserts the strong-ref premise)."
        ),
        "journal_paths": ("grad_fn_handle",),
        "store_fields": ("grad_fn_object_id",),
    },
    "backend_handle_id": {
        "attested": True,
        "independent_root": (
            "COHERENCE attestation: the stored token is definitionally "
            "str(id(<retained payload object>)) minted at freeze (ops.py "
            "TensorRef build / lookback retention), and there is no earlier "
            "independent mint site — so Check B for this bucket asserts the "
            "token equals id() of the payload object carried by the SAME "
            "record, while Check A's payload-digest tie anchors the payload "
            "content itself. A token swap without a payload swap breaks "
            "coherence (red); a joint token+payload swap moves payload bytes "
            "between anchors (red via digest ties). Records with no retained "
            "payload (payload=None, handle id present) are PRESENCE-ONLY "
            "residuals for this bucket, disclosed here."
        ),
        "retention_premise": (
            "the retained payload object is held by the journal record itself "
            "for the comparison window (payload not yet scrubbed when the "
            "harness snapshots)."
        ),
        "journal_paths": (
            "output.tensor.backend_handle_id",
            "output.transformed_tensor.backend_handle_id",
        ),
        "store_fields": (),
    },
    "tl_barcode": {
        "attested": True,
        "independent_root": (
            "the barcode attachment site: set_param_meta(param, barcode=...) "
            "(backends/torch/_tl.py) records (param address, barcode) at "
            "attachment; make_random_barcode is wrapped to prove every stored "
            "barcode was minted this run. Rooted in the live parameter object."
        ),
        "retention_premise": "parameters are model-owned and outlive the capture.",
        "journal_paths": ("params[*].barcode",),
        "store_fields": ("_param_barcodes",),
    },
    "id_embedded_names": {
        "attested": False,  # attested only via derivation from the buckets above;
        # the underivable remainder is the NAMED PRESENCE-ONLY RESIDUAL.
        "independent_root": (
            "derivation: an embedded hex/int id inside a name must equal the "
            "attested id recorded for the same anchor by an attested bucket; "
            "embeddings with no attested source are presence/shape-of-name only."
        ),
        "retention_premise": "n/a (strings)",
        "journal_paths": (),
        "store_fields": (),
    },
}

# Non-token volatile paths: canonicalized per path as presence + type (+ cheap
# invariants), never compared by value across legs (DoR: "not identity tokens").
VOLATILE_VALUE_PATHS: frozenset[str] = frozenset(
    {
        "function.func_duration",
        "function.func_rng_states",
        "function.func_autocast_state",
        "function.flops_forward",
        "function.flops_backward",
        "backend_semantics.autograd_memory",
        "backend_semantics.num_autograd_tensors",
        "backend_semantics.bytes_delta_at_call",
        "backend_semantics.bytes_peak_at_call",
        "output.tensor.memory",
        "output.transformed_tensor.memory",
    }
)
