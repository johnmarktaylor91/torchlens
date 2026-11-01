"""Run entry for the RUN stage: one capture forward from slot claim to ``Trace``.

Moved verbatim from ``torchlens._user_funcs`` by the capture seam.
"""

import copy
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from torchlens import _state

from .._autoroute._builtin_output import semantic_output_cache_key
from .._capture_honesty import record_host_tensor_source_advisory
from .._capture_state_helpers import (
    _capture_output_metadata_from_model_config,
    _fingerprint_model_weights,
    _hash_input_signatures,
    _input_id_for_relationship_evidence,
    _move_tensors_to_device,
    _qualname_for_model,
)
from .._data_classes.trace import (
    Trace,
)
from .._errors import (
    CaptureNotActiveError,
    InvalidArgumentError,
    UserTransformError,
)
from .._io import ArtifactError
from .._io.streaming import BundleStreamWriter
from .._ir import ParentEdge
from .._ir.op_record import amend_graph_edge_insertion
from .._ir.selector_eval import selector_contains_kind
from .._literals import (
    OutputDeviceLiteral,
)
from .._save_budget import SaveBudgetExceededError, SaveBudgetOption
from .._trace_state import TraceState
from .._utils.introspection import _get_code_context
from .._utils.tensor_utils import SaveMode
from .._visualization.code_panel import (
    capture_model_source_code,
    make_weak_model_ref,
)
from ..backends.torch._tl import get_tensor_label
from ..backends.torch.bound_root import (
    TLBoundMethodRoot,
)
from ..intervention import injection as _injection
from ..intervention._module_alias_guard import refuse_model_alias_spellings
from ..intervention.hooks import (
    normalize_hook_plan,
    site_universe as _site_universe,
)
from ..intervention.predicates import InterventionPredicate
from ..intervention.selectors import BaseSelector
from ..intervention.types import InterventionDecision
from ..options import (
    StorageOptions,
)
from ..recording.exceptions import PredicateError
from ..recording.options import HaltPredicateFn, PredicateFn, RecordingOptions
from ..recording.types import CaptureSpec
from ..types import _ActivationPostfunc, _GradientPostfunc
from ._episode_ledger import (
    attach_episode_header,
)
from .prepare.resolve import (
    _backward_intervention_spec_from_predicate,
    _intervention_spec_from_hook_plan,
    _merge_intervention_spec_hooks,
    _record_capture_intervention_event,
    _warn_zero_match_capture_selectors,
    activation_only_predicate as _activation_only_predicate,
)
from .stages import CaptureRequest, ForwardResult
from .stop import StopDirective


def register_tensor_connection(parent: torch.Tensor, child: torch.Tensor) -> None:
    """Register a manual parent-child tensor edge during capture.

    Parameters
    ----------
    parent:
        Parent tensor already tagged by TorchLens.
    child:
        Child tensor already tagged by TorchLens.

    Raises
    ------
    RuntimeError
        If no forward pass is being captured.
    ValueError
        If either tensor has not been tagged by TorchLens.
    """

    trace = _state._active_trace
    if trace is None:
        raise CaptureNotActiveError(
            "register_tensor_connection() was called without an active trace",
            code="capture_context_required",
            remedy=(
                "call register_tensor_connection() from model.forward() while tl.trace() is running"
            ),
            operation="register_tensor_connection",
        )
    parent_label = get_tensor_label(parent)
    child_label = get_tensor_label(child)
    if parent_label is None or child_label is None:
        raise InvalidArgumentError(
            "register_tensor_connection() received a tensor without a TorchLens capture label",
            code="tensor_connection_labels_missing",
            remedy="pass tensors produced by already-captured operations in the active trace",
            argument="parent/child",
        )
    trace.manual_tensor_connections.append((parent_label, child_label))
    _register_live_tensor_connection(trace, parent_label, child_label)


def _register_live_tensor_connection(
    trace: Trace,
    parent_label: str,
    child_label: str,
) -> None:
    """Register a parent-child edge on live capture records.

    Parameters
    ----------
    trace:
        Active trace receiving the manual edge.
    parent_label:
        Raw label for the parent tensor operation.
    child_label:
        Raw label for the child tensor operation.

    Returns
    -------
    None
        Mutates the live parent and child field mappings.
    """

    event = trace.capture_events.live_index.require_event(child_label)
    if parent_label in {edge.parent_label_raw for edge in event.parents}:
        return
    parent_arg_positions = copy.deepcopy(event.parent_arg_positions)
    parent_arg_positions.setdefault("args", {})[len(parent_arg_positions.get("args", {}))] = (
        parent_label
    )
    trace.capture_events.append_amendment(
        amend_graph_edge_insertion(
            event.seq,
            child_label,
            parents=(
                *event.parents,
                ParentEdge(parent_label_raw=parent_label, arg_position=None, edge_use="output"),
            ),
            parent_arg_positions=parent_arg_positions,
        )
    )


def _run_model_and_save_specified_outs(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None,
    layers_to_save: str | list[int | str] | None = "all",
    keep_orphans: bool = False,
    output_device: OutputDeviceLiteral = "same",
    activation_transform: _ActivationPostfunc | None = None,
    grad_transform: _GradientPostfunc | None = None,
    save_raw_activations: bool = True,
    save_raw_grads: bool = True,
    save_mode: SaveMode = "copy",
    capture_tensor_grad_hooks: bool = True,
    mark_layer_depths: bool = False,
    detach_saved_activations: bool = False,
    save_args: bool = False,
    save_grads: bool = False,
    grads_to_save: str | list[int | str] | None = "all",
    seed: int | None = None,
    num_context_lines: int = 7,
    optimizer: Any = None,
    save_source_context: bool = False,
    save_rng: bool = False,
    recurrence_detection: bool = True,
    save_outs_to: str | Path | None = None,
    keep_outs_in_memory: bool = True,
    stream_custom_attributes: bool = True,
    stream_buffer_values: bool = True,
    stream_async_writes: bool | None = None,
    stream_max_pending_bytes: int | None = None,
    grad_storage_path: str | Path | None = None,
    retain_grads_in_memory: bool = True,
    out_sink: Callable[[str, torch.Tensor], None] | None = None,
    intervention_ready: bool = False,
    capture_container_structure: bool = False,
    hooks: Any | None = None,
    intervention_spec: Any | None = None,
    normalized_hook_plan: Any | None = None,
    verbose: bool = False,
    backward_ready: bool = False,
    inference_only: bool = False,
    inference_only_implied: bool = False,
    name: str | None = None,
    module_filter: Callable[[Any], bool] | None = None,
    emit_nvtx: bool = False,
    measure_python_peak_memory: bool = False,
    distributed_witness: str = "none",
    save_budget: SaveBudgetOption = "auto",
    raise_on_nan: bool = False,
    track_nonfinite: bool = False,
    track_device_memory: bool = False,
    structure_only: bool = False,
    log_injections: bool = False,
    measure_host_peak_memory: bool = False,
    transform: Callable[[Any], Any] | None = None,
    raw_input: Any | None = None,
    save_raw_input: str | bool = "small",
    batch_render: str = "auto",
    output_transform: Callable[[Any], Any] | None = None,
    save_transformed_output: str | bool = "small",
    layer_visualizers: dict[Any, Callable[..., Any]] | None = None,
    save_visualizations: bool = False,
    recipes: list[Callable[[Any], dict[str, Any]]]
    | tuple[Callable[[Any], dict[str, Any]], ...]
    | None = None,
    save_predicate: PredicateFn | None = None,
    intervene_predicate: InterventionPredicate | None = None,
    halt_predicate: HaltPredicateFn | None = None,
    _halt_from_stop_after: bool = False,
    lookback: int = 0,
    lookback_payload_policy: str = "metadata_only",
    retain_output_parents_for_layers_to_save: bool = False,
    episode_resolved: Any | None = None,
    echo_options: Any | None = None,
    _selective_layers_to_save_request: object | None = None,
    _resolved_layer_nums_to_save: tuple[int, ...] | None = None,
    _resolved_grad_layer_nums_to_save: tuple[int, ...] | str | None = None,
    _deferred_retention_selector: Any = None,
    _deferred_gradient_selector: Any = None,
    _refresh_projection_capture: bool = False,
) -> Trace:
    """Run a forward pass with logging enabled, returning a populated Trace.

    This is the single internal entry point that creates a Trace, configures it,
    and delegates to ``Trace._run_and_log_inputs_through_model`` which handles
    model preparation, the exhaustive (and optionally fast) forward pass, and all
    postprocessing.

    Parameters
    ----------
    model:
        PyTorch model.
    input_args:
        Positional arguments to model.forward(); a single tensor or list.
    input_kwargs:
        Keyword arguments to model.forward().
    layers_to_save:
        Which layers to save outs for ('all', 'none'/None, or a list).
    keep_orphans:
        If True, island ops are retained in raw metadata and exposed via
        ``trace.graph.orphans`` while remaining hidden from the main graph.
    output_device:
        Device for saved tensors: 'same' (default), 'cpu', or 'cuda'.
        activation_transform: Optional transform applied to each out before storage
            (e.g., channel-wise averaging to reduce memory).
        grad_transform: Optional transform applied to each grad before storage.
        save_raw_activations: Whether raw outs are retained when ``activation_transform``
            is set. Metadata always describes the raw out.
        save_raw_grads: Whether raw grads are retained when ``grad_transform`` is set.
            Metadata always describes the raw grad.
        save_mode: Tensor retention mode for saved activation and gradient payloads.
            ``"copy"`` is the safe cloning default; ``"reference"`` preserves the
            captured value through in-place handling; ``"view"`` is a live alias that
            downstream in-place operations can mutate; and ``"cpu_async"`` clones to CPU.
        capture_tensor_grad_hooks: Whether forward tensors receive tensor-level
            backward hooks for implicit backward events and per-op gradient payloads.
        mark_layer_depths: Compute BFS distances from input/output layers.
            Expensive for large graphs - off by default.
        detach_saved_activations: If True, saved tensors are detached from the autograd graph.
        save_args: If True, store the non-tensor arguments to each function call.
            Required for validation replay (``validate_saved_outs``).
        save_grads: If True, register backward hooks to capture grads.
        grads_to_save: Which layer grads to save.
        seed: Fixed RNG seed for reproducibility (important for stochastic models).
            The capture snapshots the global RNG states (random/NumPy/torch CPU, plus
            CUDA when live), reseeds them for its forward, and restores the snapshot on
            every exit path, so code after the capture continues its own streams. When
            None the seed is drawn from a private entropy-seeded stream, so an outer
            ``torch.manual_seed`` does not make an unseeded capture reproducible. The
            seed used is recorded on ``trace.random_seed``.
        num_context_lines: Number of source-code context lines stored per function call.
        optimizer: Optional optimizer - used to tag which parameters have optimizers attached.
        recurrence_detection: If True (default), run full isomorphic subgraph expansion to
            detect repeated patterns (loops). Set this to False when the forward pass has
            more than about 1M operations and postprocessing speed matters; the False path
            skips the expensive expansion step and only groups operations that share the
            same parameters.
        save_outs_to: Optional portable bundle directory for streaming out save.
        stream_custom_attributes: Whether harvested module attributes are
            persisted in the streamed bundle (streaming counterpart of
            Trace.save's include_custom_attributes).
        stream_buffer_values: Whether captured pre-forward buffer values are
            persisted in the streamed bundle (streaming counterpart of
            Trace.save's include_buffer_values).
        stream_async_writes: Tri-state async-write routing for the streaming
            writer (StorageOptions.async_writes): None/True arm the bounded
            async pipeline, False keeps synchronous per-blob writes.
        stream_max_pending_bytes: Pending snapshot byte budget for the async
            pipeline (None uses the writer default).
        keep_outs_in_memory: Whether streamed outs should remain in memory
            after finalization.
        grad_storage_path: Optional portable bundle directory for streaming grad save.
        retain_grads_in_memory: Whether streamed grads should remain in memory after
            backward finalization.
        out_sink: Optional callback invoked with ``(label, tensor)`` for each
            saved out.
        intervention_ready: If True, capture replay-template metadata and mark the
            returned log as eligible for intervention mutators, replay, rerun, and
            intervention spec persistence.
        capture_container_structure: If True, persist input and output container
            structure without enabling intervention replay metadata.
        hooks: Optional live forward post-hook plan. Accepts the same shapes as
            ``Trace.attach_hooks`` and executes during this capture when supplied.
        intervention_spec: Active intervention spec to expose in runtime context.
        normalized_hook_plan: Optional pre-normalized hook entries for internal engines.
            When ``intervention_spec`` is also supplied they must be its normalization:
            they drive live dispatch and are never merged back into that spec.
        verbose: If True, print timed progress messages at each major pipeline stage.
        backward_ready: If True, keep saved outs attached to autograd for training.
        inference_only: If True, wrap the user forward in ``torch.no_grad()``.
        inference_only_implied: True when ``inference_only`` was implied by
            ``no_grad`` / ``inference_mode`` at entry rather than passed
            explicitly; the capture then clears it if the forward kept an
            autograd graph (``enable_grad`` inside the forward).
        name: User-facing log name. If omitted, generated by the public wrapper.
        emit_nvtx: If True, emit NVTX ranges around decorated torch operations.
            This is a profiling aid for CUDA/Nsight workflows and does not
            change graph construction or saved payloads.
        measure_python_peak_memory: If True, fold a ``tracemalloc``
            Python-allocation peak into the CPU/MPS ``forward_peak_memory``
            measurement. Off by default because the allocator hook taxes every
            traced operation.
        distributed_witness: Session-time witness level for collective boundary
            records captured under the distributed opt-in. ``"none"`` (default)
            records structure and correlation only; ``"digest"`` adds byte-exact
            SHA-256 contribution/destination digests (redundant merge evidence
            that can only demote, never rescue). ``"payload"`` is reserved.
        save_budget: Per-device ceiling on retained activation bytes. ``"auto"``
            (default) allows half of each device's available memory, a float sets
            another fraction, an int an absolute byte cap, and ``None`` disables
            the guard. Crossing it raises ``SaveBudgetExceededError`` naming the
            projected or committed footprint. The primary retained copy is admitted
            before allocation; this is not a general OOM-prevention guarantee.
        raise_on_nan: If True, stop capture at the first NaN or Inf tensor and raise
            ``CaptureError`` with the offending operation metadata.
        track_nonfinite: If True, record a per-op finiteness verdict for every
            committed op output, served by ``Trace.nonfinite_ops`` (session-time
            knob; never changes control flow).
        log_injections: Record intervention-hook torch calls as injected-op records.
        measure_host_peak_memory: Measure CPU peaks (resets the process ``VmHWM``).
        transform: Optional callable used to produce model-ready inputs from raw user input.
        raw_input: Original user input before ``transform`` was applied.
        save_raw_input: Portable save policy for the original raw input.
        batch_render: Raw-input batch rendering policy for visualization.
        output_transform: Optional callable used to produce human-readable
            output metadata from model output.
        save_transformed_output: Portable save policy for the transformed raw output.
        layer_visualizers: Optional mapping from selectors to thumbnail visualizer callables.
        save_visualizations: Whether rendered thumbnails should persist in portable bundles.
        recipes: Per-trace additive facet recipes captured into the trace-owned
            immutable registry snapshot.
        save_predicate: Optional in-flight predicate controlling saved activation
            payloads during the exhaustive pass.
        intervene_predicate: Optional in-flight predicate controlling current-op
            interventions during the exhaustive pass.
        halt_predicate: Optional in-flight predicate that finalizes a partial trace
            at the matching source, operation, or module boundary.
        lookback: Number of recent events available to predicate-window queries.
        lookback_payload_policy: Candidate payload retention policy for retroactive
            ``followed_by`` saves. Memory cost is bounded by ``lookback`` times
            the candidate payload size.
        retain_output_parents_for_layers_to_save: Whether this predicate capture
            originated from selective ``layers_to_save`` and must preserve the
            legacy output-parent payload rule.
        _resolved_layer_nums_to_save: Internal refresh-only raw operation numbers
            already resolved against the original Trace.
        _resolved_grad_layer_nums_to_save: Internal refresh-only gradient operation
            numbers already resolved against the original Trace.
        _refresh_projection_capture: Whether this run supplies a RefreshProjector.

    Returns

    -------
        Fully-populated Trace.
    """
    # A shared module's alias spelling cannot select one call site: refuse before the forward.
    alias_sources = (save_predicate, intervene_predicate, halt_predicate, hooks, intervention_spec)
    deferred = (normalized_hook_plan, _deferred_retention_selector, _deferred_gradient_selector)
    refuse_model_alias_spellings(model, *alias_sources, *deferred)
    # Auto-detect model device from its first parameter and move inputs to match.
    # This prevents silent device-mismatch errors when the model is on CUDA but
    # the user ops CPU tensors (a common mistake). A META first parameter is
    # never a destination: offload-hooked models (accelerate device_map /
    # cpu/disk offload, lane F37) hold meta params between forwards and their
    # hooks place inputs on the real execution device themselves — moving
    # inputs to meta would poison the forward ("Cannot copy out of meta
    # tensor") that runs fine unlogged.
    model_device = next((p.device for p in model.parameters()), None)
    if model_device is not None and model_device.type != "meta":
        input_args = _move_tensors_to_device(input_args, model_device)
        if input_kwargs is not None:
            input_kwargs = _move_tensors_to_device(input_kwargs, model_device)

    if isinstance(model, TLBoundMethodRoot):
        # F41 bound-method root: the synthetic root's identity reads the
        # OWNER (type(owner).__name__ / its qualified class name), never the
        # TL-authored wrapper class or the string "method"; the entry-point
        # fact below discloses the bound_method invocation kind.
        model_class_name = model.tl_owner_class_name
        model_class_qualname = model.tl_owner_class_qualname
        root_entry_point = model.tl_root_entry_point
    else:
        model_class_name = str(type(model).__name__)
        model_class_qualname = _qualname_for_model(model)
        root_entry_point = f"module_call:{model_class_qualname}.forward"
    model_object_id = id(model)
    weight_fingerprint = _fingerprint_model_weights(model)
    input_object_id = _input_id_for_relationship_evidence(input_args)
    input_signature_hash = _hash_input_signatures(input_args, input_kwargs)
    # A rerun replays the REQUEST on its new graph: a staged edit's inserted op
    # shifts every later raw index, so the resolved save set does not transfer.
    rerun_save_request = {
        "layers_to_save": copy.copy(layers_to_save),
        "save_predicate": save_predicate,
        "lookback": lookback,
        "lookback_payload_policy": lookback_payload_policy,
        "retain_output_parents_for_layers_to_save": retain_output_parents_for_layers_to_save,
        "_deferred_retention_selector": _deferred_retention_selector,
    }
    module_save_selector = (
        save_predicate
        if isinstance(save_predicate, BaseSelector)
        and selector_contains_kind(save_predicate, "module")
        else None
    )
    if module_save_selector is not None:
        if _deferred_retention_selector is None:
            _deferred_retention_selector = module_save_selector
        elif isinstance(_deferred_retention_selector, list):
            _deferred_retention_selector = [
                *_deferred_retention_selector,
                module_save_selector,
            ]
        else:
            _deferred_retention_selector = [
                _deferred_retention_selector,
                module_save_selector,
            ]
        if backward_ready and _deferred_gradient_selector is None:
            _deferred_gradient_selector = "all"
        save_predicate = None
        layers_to_save = "none"

    module_intervene_selector = None
    lowered_intervene_spec = None
    module_intervene_entries: list[Any] = []
    candidate_intervene_selector = getattr(intervene_predicate, "selector", None)
    candidate_intervene_decision = getattr(intervene_predicate, "decision", None)
    if (
        isinstance(candidate_intervene_selector, BaseSelector)
        and selector_contains_kind(candidate_intervene_selector, "module")
        and isinstance(candidate_intervene_decision, InterventionDecision)
        and candidate_intervene_decision.hook is not None
        and _site_universe(candidate_intervene_selector) == "activation"
    ):
        module_intervene_selector = candidate_intervene_selector
        # .selector is non-None only for single-rule specs, so rules[0] is
        # exact; the rule digest is unrecoverable downstream of this
        # lowering (W051-BIND 2.3b).
        spec_rules: Any = getattr(intervene_predicate, "rules", ())
        lowered_rule_id = spec_rules[0].rule_id
        module_intervene_entries = [
            replace(
                entry,
                metadata={
                    **dict(entry.metadata),
                    "created_by": "intervene_predicate",
                    "zero_match_ledger": "intervene_selector",
                    "rule_id": lowered_rule_id,
                },
            )
            for entry in normalize_hook_plan(
                candidate_intervene_selector,
                candidate_intervene_decision.hook,
            )
        ]
        # Keep the original spec: the capture-door event must record the rules
        # payload the loader's injected-op anchor (W051-BIND 2.3c) vouches by.
        lowered_intervene_spec = intervene_predicate
        intervene_predicate = None
    hook_plan = list(normalized_hook_plan) if normalized_hook_plan is not None else []
    # A plan an internal engine pre-normalized FROM the caller's spec (rerun)
    # is already staged on that spec. Folding it back in re-appended the
    # spec's own hooks to the live spec object on every rerun, so the staged
    # plan doubled per call (1, 2, 4, 8) and each rerun fired it that often.
    spec_derived_count = len(hook_plan) if intervention_spec is not None else 0
    if intervention_spec is None:
        intervention_spec = _backward_intervention_spec_from_predicate(intervene_predicate)
    if hook_plan == [] and hooks:
        hook_plan = normalize_hook_plan(hooks)
    hook_plan.extend(module_intervene_entries)
    hook_plan_spec = _intervention_spec_from_hook_plan(hook_plan[spec_derived_count:])
    if intervention_spec is None:
        intervention_spec = hook_plan_spec
    elif hook_plan_spec is not None:
        intervention_spec = _merge_intervention_spec_hooks(intervention_spec, hook_plan_spec)
    # r8 R54 (sol 2-thread repro): the process-global runtime-context reset +
    # configure below ran BEFORE any admission check, so a concurrent capture
    # destined for the typed refusal first RESET the admitted winner's live
    # context (clearing its replay-template flag and intervention plan
    # mid-capture). Claim the capture slot FIRST; the inner claim in
    # ``run_and_log_inputs_through_model`` passes through on the token. The
    # slot is released exactly once on every settlement path.
    _capture_slot = _state.capture_reservation()
    _reservation_token = _capture_slot.__enter__()
    _reservation_live = [True]

    def _release_capture_slot() -> None:
        """Release the early admission claim exactly once."""

        if _reservation_live[0]:
            _reservation_live[0] = False
            _capture_slot.__exit__(None, None, None)

    try:
        _state.reset_capture_runtime_context()
        _state.configure_capture_runtime_context(
            hook_plan=hook_plan,
            intervention_spec=intervention_spec,
            capture_replay_templates=intervention_ready,
            model_object_id=model_object_id,
            model_class_qualname=model_class_qualname,
            weight_fingerprint=weight_fingerprint,
            input_object_id=input_object_id,
            input_signature_hash=input_signature_hash,
        )
    except BaseException:
        _release_capture_slot()
        raise
    from .._snoop._entry import echo_forward_failure, finish_echo, open_echo_session

    echo_session: Any = None
    try:
        from .._semantic import facets as facets_mod

        trace = Trace(
            model_class_name=model_class_name,
            output_device=output_device,
            activation_transform=activation_transform,
            grad_transform=grad_transform,
            save_raw_activations=save_raw_activations,
            save_raw_grads=save_raw_grads,
            save_mode=save_mode,
            capture_tensor_grad_hooks=capture_tensor_grad_hooks,
            keep_orphans=keep_orphans,
            save_args=save_args,
            save_grads=grads_to_save if save_grads else None,
            detach_saved_activations=detach_saved_activations,
            mark_layer_depths=mark_layer_depths,
            source_context_lines=num_context_lines,
            optimizer=optimizer,
            save_source_context=save_source_context,
            save_rng=save_rng,
            recurrence_detection=recurrence_detection,
            verbose=verbose,
            backward_ready=backward_ready,
            inference_only=inference_only,
            module_filter=module_filter,
            emit_nvtx=emit_nvtx,
            measure_python_peak_memory=measure_python_peak_memory,
            distributed_witness=distributed_witness,
            save_budget=save_budget,
            input_transform=transform,
            raw_input=raw_input,
            save_raw_input=save_raw_input,
            batch_render=batch_render,
            output_transform=output_transform,
            save_transformed_output=save_transformed_output,
            layer_visualizers=layer_visualizers,
            save_visualizations=save_visualizations,
            facet_registry_snapshot=facets_mod.snapshot(recipes),
        )
        # One session field carries the capture's save request: the torch
        # backend reads its output-parent retention flag during the forward,
        # and a legacy rerun replays it unless this is a refresh capture,
        # whose resolved raw-index save set the projector rebinds instead.
        trace._rerun_save_request = {
            **rerun_save_request,
            "replayable": _resolved_layer_nums_to_save is None,
        }
        if _resolved_layer_nums_to_save is not None:
            trace._refresh_resolved_layer_nums_to_save = list(_resolved_layer_nums_to_save)
        if _resolved_grad_layer_nums_to_save is not None:
            trace._refresh_resolved_grad_layer_nums_to_save = _resolved_grad_layer_nums_to_save
        if _deferred_retention_selector is not None:
            trace._deferred_retention_selector = _deferred_retention_selector
        if _deferred_gradient_selector is not None:
            trace._deferred_gradient_selector = _deferred_gradient_selector
        if _refresh_projection_capture:
            trace._refresh_projection_capture = True
        _capture_output_metadata_from_model_config(trace, model)
        trace._output_tokenizer = getattr(model, "_torchlens_output_tokenizer", None)
        trace._semantic_output_metadata = semantic_output_cache_key(model)
        if intervention_spec is not None:
            trace._intervention_spec = intervention_spec
        trace.trace_label = name
        trace.code_context = _get_code_context(
            num_context_lines,
            source_loading_enabled=save_source_context,
        )
        forward_code = getattr(model.forward, "__code__", None)
        trace.forward_source_line = getattr(forward_code, "co_firstlineno", None)
        trace.intervention_ready = intervention_ready
        trace._capture_container_structure = capture_container_structure
        if hook_plan:
            trace.state = TraceState.LIVE_CAPTURED
        trace.model_object_id = model_object_id
        trace.model_class_qualname = model_class_qualname
        trace.param_hash_quick = weight_fingerprint
        trace.param_hash_full = weight_fingerprint
        # C07X (iv)/D10: unconditional write; F41's fail-closed gate reads it.
        # module_call for plain module roots, bound_method for the F41
        # TL-authored wrapper root (derived beside the identity fields above).
        trace.root_entry_point = root_entry_point
        trace.input_object_id = input_object_id
        trace.input_signature_hash = input_signature_hash
        trace._source_code_blob = capture_model_source_code(model)
        trace._source_model_ref = make_weak_model_ref(model)
        trace._out_sink = out_sink
        trace._keep_outs_in_memory = keep_outs_in_memory
        trace._grad_stream_retain_in_memory = retain_grads_in_memory
        trace._defer_streaming_bundle_finalization = grad_storage_path is not None
        trace._wrapper_runtime_ws.in_exhaustive_pass = True
        trace.raise_on_nan = raise_on_nan
        trace.track_nonfinite = track_nonfinite
        trace.track_device_memory = track_device_memory
        trace.measure_host_peak_memory = measure_host_peak_memory
        _injection.arm_injection_logging(trace, armed=log_injections)
        # L7a mode-marker prep (S2 SEAM, labeled): the flag DECLARES the mode
        # (memo sec 1.5) and stamps the mirror field here at entry. At S2
        # ratification the settlement-side stamp moves to the
        # _capture/outcome.py witness machinery in the S2 author's PR; this
        # entry write stays as the declared-mode source of truth.
        trace.structure_only = structure_only
        trace._stop_directive = StopDirective(
            halt_options=getattr(trace, "_predicate_save_options", None),
            raise_on_nan=raise_on_nan,
            forward_error_mode="raise",
            inference_only=inference_only,
            inference_only_implied=inference_only_implied,
        )
        if (
            save_predicate is not None
            or intervene_predicate is not None
            or halt_predicate is not None
        ):
            predicate_history_size = lookback if lookback > 0 else 8
            default_save = save_predicate is None and layers_to_save == "all"
            default_op: bool | CaptureSpec = default_save
            if backward_ready:
                default_op = CaptureSpec(
                    save_out=default_save,
                    save_metadata=default_save,
                    save_grad=True,
                )
            trace._predicate_save_options = RecordingOptions(
                keep_op=save_predicate,
                # The live forward sees only activation branches; gradient and
                # node branches armed as sticky backward hooks above.
                intervene=_activation_only_predicate(intervene_predicate),
                halt=halt_predicate,
                default_op=default_op,
                streaming=StorageOptions(
                    path=save_outs_to,
                    retain_in_memory=keep_outs_in_memory,
                    include_custom_attributes=stream_custom_attributes,
                    include_buffer_values=stream_buffer_values,
                    async_writes=stream_async_writes,
                    max_pending_bytes=stream_max_pending_bytes,
                )
                if save_outs_to is not None
                else None,
                history_size=predicate_history_size,
                lookback=lookback,
                lookback_payload_policy=lookback_payload_policy,  # type: ignore[arg-type]
                on_predicate_error="fail-fast",
                # Selector saves honour the capture's detach choice in both
                # directions, exactly like the save-all path.
                detach_saved_activations=bool(detach_saved_activations),
            )
            trace._halt_returns_partial_trace = halt_predicate is not None
            trace._stop_directive = StopDirective(
                halt_options=trace._predicate_save_options,
                raise_on_nan=raise_on_nan,
                forward_error_mode=trace._predicate_save_options.on_forward_error,
                inference_only=inference_only,
                inference_only_implied=inference_only_implied,
            )
            trace._predicate_history_size = predicate_history_size
            trace._predicate_lookback = lookback
            trace._predicate_lookback_payload_policy = lookback_payload_policy
        bundle_path = grad_storage_path if grad_storage_path is not None else save_outs_to
        if bundle_path is not None:
            trace._out_writer = BundleStreamWriter(
                bundle_path,
                include_custom_attributes=stream_custom_attributes,
                include_buffer_values=stream_buffer_values,
            )
            # None = consumer default: trace captures overlap blob writes
            # with the forward; False is the explicit synchronous opt-out.
            if stream_async_writes is not False:
                trace._out_writer.arm_async_writes(stream_max_pending_bytes)
        if episode_resolved is not None:
            # Pre-capture episode declaration marker: rides the partial
            # product on failure and is visible to postprocess consumers; the
            # settlement-time writer replaces it with the finalized ledger.
            attach_episode_header(trace, episode_resolved)
        # Echo narrator (snoop D1): one read-only observer per capture
        # (runtime-only state). Opened INSIDE the guarded region: the echo
        # sink opens its path eagerly, and a sink-open failure (missing
        # directory, unwritable path) used to escape between the slot claim
        # and the forward's own guard, leaking the capture reservation for as
        # long as the exception object lived (AUD-CODE 2.15) -- the next
        # tl.trace then refused reentrant_trace on an idle process.
        echo_session = open_echo_session(trace, echo_options)
    except BaseException:
        # A pre-forward setup failure (the Trace ctor, the echo sink open, or
        # any later pre-forward step) must not leak the capture-global runtime
        # context configured just above: the forward's own try/finally only
        # guards the window starting below. Reset here so the protected region
        # begins no later than configure_capture_runtime_context().
        _state.reset_capture_runtime_context()
        _release_capture_slot()
        raise
    try:
        # C03 live site-key minting (surgery Build 0c): when a configured
        # predicate addresses by structural site (tl.site), arm one streaming
        # minter for exactly this capture -- captures are single-threaded, so
        # the ContextVar scope is exact and unarmed surfaces keep their typed
        # capability refusal.
        from contextlib import nullcontext

        from ..intervention.site_keys import (
            armed_live_minter,
            predicate_needs_live_site_keys,
        )

        live_minter_scope = (
            armed_live_minter()
            if predicate_needs_live_site_keys(save_predicate, intervene_predicate, halt_predicate)
            else nullcontext()
        )
        with live_minter_scope:
            trace._run_and_log_inputs_through_model(
                model,
                cast(torch.Tensor | list[Any], input_args),
                input_kwargs,
                layers_to_save,
                grads_to_save,
                seed,
                reservation_resume=_reservation_token,
            )
    except BaseException as exc:
        # Crash tail (snoop D5 tail 2): synchronous flush; note rides PEP 678.
        echo_forward_failure(echo_session, exc)
        # F5: postprocess pops ``_out_writer`` at its transient-state seam, so
        # a post-seam failure (teardown, streaming tail) reaches this handler
        # on a trace WITHOUT the attribute; the unguarded read used to mask
        # the real exception with AttributeError.
        out_writer = trace.__dict__.get("_out_writer")
        if out_writer is not None:
            if not getattr(out_writer, "_closed", False):
                out_writer.abort(str(exc))
            if isinstance(exc, Exception) and not isinstance(
                exc,
                (PredicateError, SaveBudgetExceededError, ArtifactError, UserTransformError),
            ):
                raise ArtifactError("Streaming out save failed during forward pass.") from exc
        raise
    finally:
        _state.reset_capture_runtime_context()
        _release_capture_slot()
        if hasattr(trace, "_capture_container_structure"):
            delattr(trace, "_capture_container_structure")
    if isinstance(model, TLBoundMethodRoot):
        # F41 TL-authored-root disclosure (tlspec v9 entry-dark slot, C07X
        # item (iv)): the marker lands on the root op record -- the
        # output-boundary op records, the terminal records of the wrapper's
        # DAG -- coherent only beside the bound_method entry-point fact
        # (fail-closed load validation in _io/_forgery_identity_facts.py).
        for output_label in trace.output_layers:
            for output_op in trace[output_label].ops:
                output_op.tl_authored_root = True
    finish_echo(echo_session, trace)
    _warn_zero_match_capture_selectors(
        trace,
        save_selector=module_save_selector or save_predicate,
        intervene_selector=(
            module_intervene_selector or getattr(intervene_predicate, "selector", None)
        ),
        # stop_after-compiled halts carry their OWN provenance-split never-fired
        # policy at the trace entry (typed refusal / coded warning); the generic
        # halt zero-match warning would double-disclose ahead of that refusal.
        halt_selector=None if _halt_from_stop_after else halt_predicate,
        layers_to_save_request=_selective_layers_to_save_request,
    )
    record_host_tensor_source_advisory(trace)
    _record_capture_intervention_event(
        trace,
        intervene_predicate
        if intervene_predicate is not None
        else (
            lowered_intervene_spec
            if lowered_intervene_spec is not None
            else module_intervene_selector
        ),
    )
    return trace


def run_capture(request: CaptureRequest) -> ForwardResult:
    """Run one capture forward pass for a packed request.

    Seam entry: unpacks the request into ``_run_model_and_save_specified_outs``
    unchanged. The RUN lane moves the run half of that function behind this name.

    Parameters
    ----------
    request:
        Request built by ``prepare_capture``.

    Returns
    -------
    ForwardResult
        The populated trace on the completed arm.
    """
    trace = _run_model_and_save_specified_outs(
        request.model,
        request.input_args,
        request.input_kwargs,
        **dict(request.plan or {}),
    )
    return ForwardResult(trace=trace, request=request)
