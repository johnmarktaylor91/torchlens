"""Request resolution and Trace setup for the RUN stage entry.

The half of ``run_capture`` before the forward: alias refusal, device
placement, root identity, the rerun save request, the module-selector
lowerings and the hook plan (pure request resolution the PREPARE stage may
absorb verbatim), then the ``Trace`` construction and its entry stamps (save
request, identity facts, sinks, predicate recording options, streaming
writer), exactly in today's order.
"""

__tl_layer__ = "L4"

import copy
import weakref
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from torch import nn

from .._autoroute._builtin_output import semantic_output_cache_key
from .._capture_state_helpers import (
    _capture_output_metadata_from_model_config,
    _fingerprint_model_weights,
    _hash_input_signatures,
    _input_id_for_relationship_evidence,
    _move_tensors_to_device,
    _qualname_for_model,
)
from .._data_classes.trace import Trace
from .._io.streaming import BundleStreamWriter
from .._ir.selector_eval import selector_contains_kind
from .._registry._facet_snapshot import facet_snapshot
from .._selection.module_alias_guard import refuse_model_alias_spellings
from .._selection.selectors import BaseSelector
from .._trace_state import TraceState
from .._utils.introspection import _get_code_context
from ..backends.torch.bound_root import TLBoundMethodRoot
from ..intervention import injection as _injection
from ..intervention.hooks import normalize_hook_plan, site_universe as _site_universe
from ..intervention.types import InterventionDecision
from ..options import StorageOptions
from ..recording.options import _RecordingConfig
from ..recording.types import CaptureSpec
from ._episode_ledger import attach_episode_header
from .prepare.resolve import (
    _backward_intervention_spec_from_predicate,
    _intervention_spec_from_hook_plan,
    _merge_intervention_spec_hooks,
    activation_only_predicate as _activation_only_predicate,
)
from .stages import CaptureRequest
from .stop import StopDirective


def _resolved_request(request: CaptureRequest) -> SimpleNamespace:
    """Resolve a packed request into one namespace the run half reads.

    The namespace starts as the request's flat plan (the keyword arguments of
    ``_run_model_and_save_specified_outs``) plus ``model``, ``input_args`` and
    ``input_kwargs``; resolution rewrites the inputs (device placement), the
    save and intervene selectors, and adds the root identity facts, the rerun
    save request, the hook plan and the composed intervention spec.
    """

    r = SimpleNamespace(
        model=request.model,
        input_args=request.input_args,
        input_kwargs=request.input_kwargs,
        **dict(request.plan or {}),
    )
    # A shared module's alias spelling cannot select one call site: refuse before the forward.
    alias_sources = (
        r.save_predicate,
        r.intervene_predicate,
        r.halt_predicate,
        r.hooks,
        r.intervention_spec,
    )
    deferred = (
        r.normalized_hook_plan,
        r._deferred_retention_selector,
        r._deferred_gradient_selector,
    )
    refuse_model_alias_spellings(r.model, *alias_sources, *deferred)
    _place_inputs_on_model_device(r)
    _stamp_root_identity(r)
    # A rerun replays the REQUEST on its new graph: a staged edit's inserted op
    # shifts every later raw index, so the resolved save set does not transfer.
    r.rerun_save_request = {
        "layers_to_save": copy.copy(r.layers_to_save),
        "save_predicate": r.save_predicate,
        "lookback": r.lookback,
        "lookback_payload_policy": r.lookback_payload_policy,
        "retain_output_parents_for_layers_to_save": r.retain_output_parents_for_layers_to_save,
        "_deferred_retention_selector": r._deferred_retention_selector,
    }
    _lower_module_save_selector(r)
    _lower_module_intervene_selector(r)
    _compose_hook_plan(r)
    return r


def _place_inputs_on_model_device(r: SimpleNamespace) -> None:
    """Move the inputs to the model's device when it has a concrete one.

    Auto-detect the model device from its first parameter and move inputs to
    match: this prevents silent device-mismatch errors when the model is on
    CUDA but the user passes CPU tensors (a common mistake). A META first
    parameter is never a destination: offload-hooked models (accelerate
    device_map / cpu/disk offload, lane F37) hold meta params between forwards
    and their hooks place inputs on the real execution device themselves;
    moving inputs to meta would poison the forward ("Cannot copy out of meta
    tensor") that runs fine unlogged.
    """

    model_device = next((p.device for p in r.model.parameters()), None)
    if model_device is not None and model_device.type != "meta":
        r.input_args = _move_tensors_to_device(r.input_args, model_device)
        if r.input_kwargs is not None:
            r.input_kwargs = _move_tensors_to_device(r.input_kwargs, model_device)


def _stamp_root_identity(r: SimpleNamespace) -> None:
    """Derive the root identity facts and the input relationship evidence."""

    model = r.model
    if isinstance(model, TLBoundMethodRoot):
        # F41 bound-method root: the synthetic root's identity reads the
        # OWNER (type(owner).__name__ / its qualified class name), never the
        # TL-authored wrapper class or the string "method"; the entry-point
        # fact below discloses the bound_method invocation kind.
        r.model_class_name = model.tl_owner_class_name
        r.model_class_qualname = model.tl_owner_class_qualname
        r.root_entry_point = model.tl_root_entry_point
    else:
        r.model_class_name = str(type(model).__name__)
        r.model_class_qualname = _qualname_for_model(model)
        r.root_entry_point = f"module_call:{r.model_class_qualname}.forward"
    r.model_object_id = id(model)
    r.weight_fingerprint = _fingerprint_model_weights(model)
    r.input_object_id = _input_id_for_relationship_evidence(r.input_args)
    r.input_signature_hash = _hash_input_signatures(r.input_args, r.input_kwargs)


def _lower_module_save_selector(r: SimpleNamespace) -> None:
    """Lower a module-kind save selector onto the deferred retention selector."""

    r.module_save_selector = (
        r.save_predicate
        if isinstance(r.save_predicate, BaseSelector)
        and selector_contains_kind(r.save_predicate, "module")
        else None
    )
    if r.module_save_selector is None:
        return
    if r._deferred_retention_selector is None:
        r._deferred_retention_selector = r.module_save_selector
    elif isinstance(r._deferred_retention_selector, list):
        r._deferred_retention_selector = [*r._deferred_retention_selector, r.module_save_selector]
    else:
        r._deferred_retention_selector = [r._deferred_retention_selector, r.module_save_selector]
    if r.backward_ready and r._deferred_gradient_selector is None:
        r._deferred_gradient_selector = "all"
    r.save_predicate = None
    r.layers_to_save = "none"


def _lower_module_intervene_selector(r: SimpleNamespace) -> None:
    """Lower a module-kind activation intervene selector into hook-plan entries."""

    r.module_intervene_selector = None
    r.lowered_intervene_spec = None
    r.module_intervene_entries = []
    candidate_selector = getattr(r.intervene_predicate, "selector", None)
    candidate_decision = getattr(r.intervene_predicate, "decision", None)
    if not (
        isinstance(candidate_selector, BaseSelector)
        and selector_contains_kind(candidate_selector, "module")
        and isinstance(candidate_decision, InterventionDecision)
        and candidate_decision.hook is not None
        and _site_universe(candidate_selector) == "activation"
    ):
        return
    r.module_intervene_selector = candidate_selector
    # .selector is non-None only for single-rule specs, so rules[0] is
    # exact; the rule digest is unrecoverable downstream of this
    # lowering (W051-BIND 2.3b).
    spec_rules: Any = getattr(r.intervene_predicate, "rules", ())
    lowered_rule_id = spec_rules[0].rule_id
    r.module_intervene_entries = [
        replace(
            entry,
            metadata={
                **dict(entry.metadata),
                "created_by": "intervene_predicate",
                "zero_match_ledger": "intervene_selector",
                "rule_id": lowered_rule_id,
            },
        )
        for entry in normalize_hook_plan(candidate_selector, candidate_decision.hook)
    ]
    # Keep the original spec: the capture-door event must record the rules
    # payload the loader's injected-op anchor (W051-BIND 2.3c) vouches by.
    r.lowered_intervene_spec = r.intervene_predicate
    r.intervene_predicate = None


def _compose_hook_plan(r: SimpleNamespace) -> None:
    """Compose the live hook plan and the intervention spec it implies."""

    hook_plan = list(r.normalized_hook_plan) if r.normalized_hook_plan is not None else []
    intervention_spec = r.intervention_spec
    # A plan an internal engine pre-normalized FROM the caller's spec (rerun)
    # is already staged on that spec. Folding it back in re-appended the
    # spec's own hooks to the live spec object on every rerun, so the staged
    # plan doubled per call (1, 2, 4, 8) and each rerun fired it that often.
    spec_derived_count = len(hook_plan) if intervention_spec is not None else 0
    if intervention_spec is None:
        intervention_spec = _backward_intervention_spec_from_predicate(r.intervene_predicate)
    if hook_plan == [] and r.hooks:
        hook_plan = normalize_hook_plan(r.hooks)
    hook_plan.extend(r.module_intervene_entries)
    hook_plan_spec = _intervention_spec_from_hook_plan(hook_plan[spec_derived_count:])
    if intervention_spec is None:
        intervention_spec = hook_plan_spec
    elif hook_plan_spec is not None:
        intervention_spec = _merge_intervention_spec_hooks(intervention_spec, hook_plan_spec)
    r.hook_plan = hook_plan
    r.intervention_spec = intervention_spec


def _construct_trace(r: SimpleNamespace) -> Trace:
    """Build the Trace from the resolved request's option values."""

    return Trace(
        model_class_name=r.model_class_name,
        output_device=r.output_device,
        activation_transform=r.activation_transform,
        grad_transform=r.grad_transform,
        save_raw_activations=r.save_raw_activations,
        save_raw_grads=r.save_raw_grads,
        save_mode=r.save_mode,
        capture_tensor_grad_hooks=r.capture_tensor_grad_hooks,
        keep_orphans=r.keep_orphans,
        save_args=r.save_args,
        save_grads=r.grads_to_save if r.save_grads else None,
        detach_saved_activations=r.detach_saved_activations,
        mark_layer_depths=r.mark_layer_depths,
        source_context_lines=r.num_context_lines,
        optimizer=r.optimizer,
        save_source_context=r.save_source_context,
        save_rng=r.save_rng,
        recurrence_detection=r.recurrence_detection,
        verbose=r.verbose,
        backward_ready=r.backward_ready,
        inference_only=r.inference_only,
        emit_nvtx=r.emit_nvtx,
        measure_python_peak_memory=r.measure_python_peak_memory,
        distributed_witness=r.distributed_witness,
        save_budget=r.save_budget,
        input_transform=r.transform,
        input=r.raw_input,
        save_raw_input=r.save_raw_input,
        batch_render=r.batch_render,
        output_transform=r.output_transform,
        save_transformed_output=r.save_transformed_output,
        layer_visualizers=r.layer_visualizers,
        save_visualizations=r.save_visualizations,
        facet_functions=facet_snapshot(r.recipes),
    )


def _stamp_save_request(trace: Trace, r: SimpleNamespace) -> None:
    """Stamp the save request, the resolved and deferred selectors and output metadata."""

    model = r.model
    # One session field carries the capture's save request: the torch
    # backend reads its output-parent retention flag during the forward,
    # and a legacy rerun replays it unless this is a refresh capture,
    # whose resolved raw-index save set the projector rebinds instead.
    trace._rerun_save_request = {
        **r.rerun_save_request,
        "replayable": r._resolved_layer_nums_to_save is None,
    }
    if r._resolved_layer_nums_to_save is not None:
        trace._refresh_resolved_layer_nums_to_save = list(r._resolved_layer_nums_to_save)
    if r._resolved_grad_layer_nums_to_save is not None:
        trace._refresh_resolved_grad_layer_nums_to_save = r._resolved_grad_layer_nums_to_save
    if r._deferred_retention_selector is not None:
        trace._deferred_retention_selector = r._deferred_retention_selector
    if r._deferred_gradient_selector is not None:
        trace._deferred_gradient_selector = r._deferred_gradient_selector
    if r._refresh_projection_capture:
        trace._refresh_projection_capture = True
    _capture_output_metadata_from_model_config(trace, model)
    trace._output_tokenizer = getattr(model, "_torchlens_output_tokenizer", None)
    trace._semantic_output_metadata = semantic_output_cache_key(model)
    if r.intervention_spec is not None:
        trace._intervention_spec = r.intervention_spec
    trace.name = r.name
    trace._source_frames = _get_code_context(
        r.num_context_lines,
        source_loading_enabled=r.save_source_context,
    )
    forward_code = getattr(model.forward, "__code__", None)
    trace._source_forward_line = getattr(forward_code, "co_firstlineno", None)
    trace.is_replayable = r.replayable
    trace._capture_container_structure = r.capture_container_structure
    if r.hook_plan:
        trace.state = TraceState.LIVE_CAPTURED


def _stamp_identity_and_sinks(trace: Trace, r: SimpleNamespace) -> None:
    """Stamp the root identity facts, sinks, honesty flags and the stop directive."""

    model = r.model
    trace._provenance_model_object_id = r.model_object_id
    trace._source_class_qualname = r.model_class_qualname
    trace._provenance_param_hash_quick = r.weight_fingerprint
    trace._provenance_param_hash_full = r.weight_fingerprint
    # C07X (iv)/D10: unconditional write; F41's fail-closed gate reads it.
    # module_call for plain module roots, bound_method for the F41
    # TL-authored wrapper root (derived beside the identity fields above).
    trace._provenance_root_entry_point = r.root_entry_point
    trace._provenance_input_object_id = r.input_object_id
    trace._provenance_input_signature_hash = r.input_signature_hash
    trace._source_code_blob = capture_model_source_code(model)
    trace._source_model_ref = make_weak_model_ref(model)
    trace._out_sink = r.out_sink
    trace._keep_outs_in_memory = r.keep_outs_in_memory
    trace._grad_stream_retain_in_memory = r.retain_grads_in_memory
    trace._defer_streaming_bundle_finalization = r.grad_storage_path is not None
    trace._wrapper_runtime_ws.in_exhaustive_pass = True
    trace._options_on_nonfinite = r.raise_on_nan
    trace.track_nonfinite = r.track_nonfinite
    trace.track_device_memory = r.track_device_memory
    trace.measure_host_peak_memory = r.measure_host_peak_memory
    trace._options_aten_ops = "values" if r.aten_ops == "values" else bool(r.aten_ops)
    _injection.arm_injection_logging(trace, armed=r.log_injections)
    # L7a mode-marker prep (S2 SEAM, labeled): the flag DECLARES the mode
    # (memo sec 1.5) and stamps the mirror field here at entry. At S2
    # ratification the settlement-side stamp moves to the
    # _capture/outcome.py witness machinery in the S2 author's PR; this
    # entry write stays as the declared-mode source of truth.
    trace._options_structure_only = r.structure_only
    # Per-op argument templates are all saved-graph replay needs (trace.do(site, edit)
    # without model=), and they cost about a tenth of capture time with no retained
    # tensor bytes, so every eager capture records them. replayable=True still adds
    # the runnable apparatus on top. A structure-only capture has no values to replay.
    trace._options_save_arg_templates = not r.structure_only
    trace._stop_directive = StopDirective(
        halt_options=getattr(trace, "_predicate_save_options", None),
        raise_on_nan=r.raise_on_nan,
        forward_error_mode="raise",
        inference_only=r.inference_only,
        inference_only_implied=r.inference_only_implied,
    )


def _configure_predicate_options(trace: Trace, r: SimpleNamespace) -> None:
    """Arm the predicate recording options when any predicate is configured."""

    if r.save_predicate is None and r.intervene_predicate is None and r.halt_predicate is None:
        return
    predicate_history_size = r.lookback if r.lookback > 0 else 8
    default_save = r.save_predicate is None and r.layers_to_save == "all"
    default_op: bool | CaptureSpec = default_save
    if r.backward_ready:
        default_op = CaptureSpec(save_out=default_save, save_metadata=default_save, save_grad=True)
    streaming = (
        StorageOptions(
            path=r.save_outs_to,
            retain_in_memory=r.keep_outs_in_memory,
            include_custom_attributes=r.stream_custom_attributes,
            include_buffer_values=r.stream_buffer_values,
            async_writes=r.stream_async_writes,
            max_pending_bytes=r.stream_max_pending_bytes,
        )
        if r.save_outs_to is not None
        else None
    )
    trace._predicate_save_options = _RecordingConfig(
        keep_op=r.save_predicate,
        # The live forward sees only activation branches; gradient and
        # node branches armed as sticky backward hooks above.
        intervene=_activation_only_predicate(r.intervene_predicate),
        halt=r.halt_predicate,
        default_op=default_op,
        streaming=streaming,
        history_size=predicate_history_size,
        lookback=r.lookback,
        lookback_payload_policy=r.lookback_payload_policy,
        on_predicate_error="fail-fast",
        # Selector saves honour the capture's detach choice in both
        # directions, exactly like the save-all path.
        detach_saved_activations=bool(r.detach_saved_activations),
    )
    trace._halt_returns_partial_trace = r.halt_predicate is not None
    trace._stop_directive = StopDirective(
        halt_options=trace._predicate_save_options,
        raise_on_nan=r.raise_on_nan,
        forward_error_mode=trace._predicate_save_options.on_forward_error,
        inference_only=r.inference_only,
        inference_only_implied=r.inference_only_implied,
    )
    trace._predicate_history_size = predicate_history_size
    trace._predicate_lookback = r.lookback
    trace._predicate_lookback_payload_policy = r.lookback_payload_policy


def _open_streaming_writer(trace: Trace, r: SimpleNamespace) -> None:
    """Open the streaming bundle writer and attach the episode header."""

    bundle_path = r.grad_storage_path if r.grad_storage_path is not None else r.save_outs_to
    if bundle_path is not None:
        trace._out_writer = BundleStreamWriter(
            bundle_path,
            include_custom_attributes=r.stream_custom_attributes,
            include_buffer_values=r.stream_buffer_values,
        )
        # None = consumer default: trace captures overlap blob writes
        # with the forward; False is the explicit synchronous opt-out.
        if r.stream_async_writes is not False:
            trace._out_writer.arm_async_writes(r.stream_max_pending_bytes)
    if r.episode_resolved is not None:
        # Pre-capture episode declaration marker: rides the partial
        # product on failure and is visible to postprocess consumers; the
        # settlement-time writer replaces it with the finalized ledger.
        attach_episode_header(trace, r.episode_resolved)


def capture_model_source_code(model: nn.Module) -> dict[str, str]:
    """Capture the code-panel source text (deferred: the code panel sits above this layer)."""
    from .._visualization.code_panel import capture_model_source_code as _capture

    return _capture(model)


def make_weak_model_ref(model: nn.Module) -> "weakref.ReferenceType[nn.Module] | None":
    """Weakly reference the model for the code panel (deferred like the function above)."""
    from .._visualization.code_panel import make_weak_model_ref as _make_ref

    return _make_ref(model)
