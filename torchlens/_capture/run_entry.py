"""Run entry for the RUN stage: one capture forward from slot claim to ``Trace``.

The RUN stage entry ``run_capture`` and the keyword adapter the rerun engine and the
recorder still call by name.
"""

import copy
import dataclasses
import inspect
from collections.abc import Callable, Collection
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import torch
from torch import nn

from torchlens import _state

from .._capture_honesty import record_host_tensor_source_advisory
from .._data_classes.trace import (
    Trace,
)
from .._errors import (
    CaptureNotActiveError,
    InvalidArgumentError,
    UserTransformError,
)
from .._io import ArtifactError
from .._ir import ParentEdge
from .._ir.op_record import amend_graph_edge_insertion
from .._literals import (
    OutputDeviceLiteral,
    _ActivationPostfunc,
    _GradientPostfunc,
)
from .._save_budget import SaveBudgetExceededError, SaveBudgetOption
from .._utils._tensor_label import get_tensor_label
from .._utils.tensor_utils import SaveMode
from ..backends.torch._foreign_data_warning import warn_unresolved_foreign_data
from ..backends.torch.bound_root import (
    TLBoundMethodRoot,
)
from ..intervention._predicates import InterventionPredicate
from ..intervention._site_keys import armed_live_minter, predicate_needs_live_site_keys
from ..intervention._zero_fire import capture_rule_fire_ledger
from ..recording._exceptions import PredicateError
from ..recording._options import HaltPredicateFn, PredicateFn
from ._run_setup import (
    _configure_predicate_options,
    _construct_trace,
    _open_streaming_writer,
    _resolved_request,
    _stamp_identity_and_sinks,
    _stamp_save_request,
)
from .outcome import CaptureStatus
from .prepare.resolve import (
    _record_capture_intervention_event,
    _warn_zero_match_capture_selectors,
)
from .stages import CaptureRequest, ForwardArm, ForwardResult


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

    Examples
    --------
    >>> class Linked(nn.Module):
    ...     def forward(self, a):
    ...         h, y = a.relu(), a.sigmoid()
    ...         register_tensor_connection(h, y)
    ...         return y * 2
    >>> trace = tl.trace(Linked(), x)
    >>> "relu_1_1" in trace["sigmoid_1_2"].parents
    True
    >>> try:
    ...     register_tensor_connection(x, x)
    ... except tl.errors.TorchLensError as err:
    ...     print(type(err).__name__)
    CaptureNotActiveError
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
    trace._graph_manual_tensor_connections.append((parent_label, child_label))
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
                ParentEdge(parent_label_raw=parent_label, arg_position=None, _edge_use="output"),
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
    replayable: bool = False,
    capture_container_structure: bool = False,
    hooks: Any | None = None,
    intervention_spec: Any | None = None,
    normalized_hook_plan: Any | None = None,
    verbose: bool = False,
    backward_ready: bool = False,
    inference_only: bool = False,
    inference_only_implied: bool = False,
    name: str | None = None,
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
    aten_ops: bool | Literal["values"] = False,
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
            seed used is recorded on ``trace.options.seed``.
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
        out_sink: Optional callback invoked with ``(label, tensor)`` for each saved out.
        replayable: If True, capture replay-template metadata and mark the
            returned log as eligible for intervention mutators, replay, rerun, and
            intervention spec persistence.
        capture_container_structure: If True, persist input and output container
            structure without enabling intervention replay metadata.
        hooks: Optional live forward post-hook plan. Accepts the same shapes as
            ``Trace.do`` and executes during this capture when supplied.
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
            measurement. Off by default because the allocator hook taxes every traced operation.
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
        aten_ops: Record the ATen operations each logged torch call dispatched to
            (``"values"`` also keeps their output tensors).
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
    return _run_keyword_capture(locals())


def _run_keyword_capture(arguments: dict[str, Any]) -> Trace:
    """Build the capture request from the keyword adapter's arguments and run it.

    Args:
        arguments: ``locals()`` of ``_run_model_and_save_specified_outs`` taken
            before it binds any name of its own, so exactly its parameters.

    Returns:
        Fully-populated Trace.
    """

    plan = {k: v for k, v in arguments.items() if k not in _REQUEST_POSITIONALS}
    request = CaptureRequest(
        model=arguments["model"],
        input_args=arguments["input_args"],
        input_kwargs=arguments["input_kwargs"],
        plan=plan,
    )
    return run_capture(request).trace


_REQUEST_POSITIONALS = frozenset({"model", "input_args", "input_kwargs"})

_PLAN_DEFAULTS: dict[str, Any] = {
    name: parameter.default
    for name, parameter in inspect.signature(_run_model_and_save_specified_outs).parameters.items()
    if parameter.default is not inspect.Parameter.empty
}
"""Default value of every run keyword, read once from the keyword adapter's signature."""


def _with_plan_defaults(request: CaptureRequest) -> CaptureRequest:
    """Complete the request's plan with the adapter defaults for any keyword the door left out.

    The door builds the plan from its resolved state; refresh-only keywords
    (``_resolved_layer_nums_to_save``, ``_resolved_grad_layer_nums_to_save``,
    ``_refresh_projection_capture``) are supplied only by the refresh path, so a
    plain capture's plan takes the adapter's defaults for them.
    """

    plan = dict(request.plan or {})
    missing = {name: value for name, value in _PLAN_DEFAULTS.items() if name not in plan}
    if not missing:
        return request
    return dataclasses.replace(request, plan={**missing, **plan})


def _configure_runtime_context(r: SimpleNamespace) -> None:
    """Reset and configure the process-global capture runtime context.

    R54 (sol 2-thread repro): this reset and configure used to run BEFORE
    any admission check, so a concurrent capture destined for the typed
    refusal first RESET the admitted winner's live context (clearing its
    replay-template flag and intervention plan mid-capture). The caller claims
    the capture slot first and releases it exactly once on every settlement path.
    """

    _state.reset_capture_runtime_context()
    _state.configure_capture_runtime_context(
        hook_plan=r.hook_plan,
        intervention_spec=r.intervention_spec,
        capture_replay_templates=r.replayable,
        model_object_id=r.model_object_id,
        model_class_qualname=r.model_class_qualname,
        weight_fingerprint=r.weight_fingerprint,
        input_object_id=r.input_object_id,
        input_signature_hash=r.input_signature_hash,
    )


def _forward_under_minter(trace: Trace, r: SimpleNamespace, reservation_token: object) -> None:
    """Run the forward, arming the live site-key minter when a predicate needs it.

    live site-key minting (surgery Build 0c): when a configured predicate
    addresses by structural site (tl.site), arm one streaming minter for
    exactly this capture; captures are single-threaded, so the ContextVar
    scope is exact and unarmed surfaces keep their typed capability refusal.
    """

    from contextlib import nullcontext

    live_minter_scope = (
        armed_live_minter()
        if predicate_needs_live_site_keys(r.save_predicate, r.intervene_predicate, r.halt_predicate)
        else nullcontext()
    )
    with live_minter_scope:
        trace._run_and_log_inputs_through_model(
            r.model,
            cast(torch.Tensor | list[Any], r.input_args),
            r.input_kwargs,
            r.layers_to_save,
            r.grads_to_save,
            r.seed,
            reservation_resume=reservation_token,
        )


def _abort_streaming_writer(trace: Trace, exc: BaseException) -> None:
    """Abort the streaming writer of a failed forward; wrap ordinary errors as artifact errors.

    Postprocess pops ``_out_writer`` at its transient-state seam, so a
    post-seam failure (teardown, streaming tail) reaches this handler on a
    trace WITHOUT the attribute; the unguarded read used to mask the real
    exception with AttributeError.
    """

    out_writer = trace.__dict__.get("_out_writer")
    if out_writer is None:
        return
    if not getattr(out_writer, "_closed", False):
        out_writer.abort(str(exc))
    if isinstance(exc, Exception) and not isinstance(
        exc,
        (PredicateError, SaveBudgetExceededError, ArtifactError, UserTransformError),
    ):
        raise ArtifactError("Streaming out save failed during forward pass.") from exc


def _finish_run_entry(
    trace: Trace, r: SimpleNamespace, echo_session: Any, fired_rule_ids: Collection[str]
) -> None:
    """Post-forward tail: root disclosure, echo close, zero-match warnings, events.

    ``fired_rule_ids`` is the capture's fire ledger: the ids of the public-spec
    ``intervene=`` rules that fired, settled per rule by the event writer.
    """

    if isinstance(r.model, TLBoundMethodRoot):
        # TL-authored-root disclosure (tlspec v9 entry-dark slot, C07X
        # item (iv)): the marker lands on the root op record -- the
        # output-boundary op records, the terminal records of the wrapper's
        # DAG -- coherent only beside the bound_method entry-point fact
        # (fail-closed load validation in _io/_forgery_identity_facts.py).
        for output_label in trace.output_layers:
            for output_op in trace.ops.resolve_all(output_label):
                output_op._internal_set("tl_authored_root", True)
    from .._snoop._entry import finish_echo

    finish_echo(echo_session, trace)
    _warn_zero_match_capture_selectors(
        trace,
        save_selector=r.module_save_selector or r.save_predicate,
        intervene_selector=(
            r.module_intervene_selector or getattr(r.intervene_predicate, "selector", None)
        ),
        # stop_after-compiled halts carry their OWN provenance-split never-fired
        # policy at the trace entry (typed refusal / coded warning); the generic
        # halt zero-match warning would double-disclose ahead of that refusal.
        halt_selector=None if r._halt_from_stop_after else r.halt_predicate,
        layers_to_save_request=r._selective_layers_to_save_request,
    )
    record_host_tensor_source_advisory(trace)
    warn_unresolved_foreign_data(trace)
    _record_capture_intervention_event(
        trace,
        r.intervene_predicate
        if r.intervene_predicate is not None
        else (
            r.lowered_intervene_spec
            if r.lowered_intervene_spec is not None
            else r.module_intervene_selector
        ),
        fired_rule_ids=fired_rule_ids,
    )


def _forward_arm_of(trace: Trace) -> ForwardArm:
    """Read the arm a returned trace settled on: completed, or halted on a stop."""

    outcome = trace.__dict__.get("_capture_outcome")
    status = getattr(outcome, "status", None)
    return "halted" if status is CaptureStatus.STOPPED else "completed"


def _build_trace_for_run(r: SimpleNamespace, release_slot: Callable[[], None]) -> tuple[Trace, Any]:
    """Build and stamp the Trace and open the echo session inside the guarded region.

    A pre-forward setup failure (the Trace ctor, the echo sink open, or any
    later pre-forward step) must not leak the capture-global runtime context
    configured just before: the forward's own try/finally only guards the
    window starting after this. Reset here so the protected region begins no
    later than ``configure_capture_runtime_context()``.
    """

    from .._snoop._entry import open_echo_session

    try:
        trace = _construct_trace(r)
        _stamp_save_request(trace, r)
        _stamp_identity_and_sinks(trace, r)
        _configure_predicate_options(trace, r)
        _open_streaming_writer(trace, r)
        # Echo narrator: one read-only observer per capture
        # (runtime-only state). Opened INSIDE the guarded region: the echo
        # sink opens its path eagerly, and a sink-open failure (missing
        # directory, unwritable path) used to escape between the slot claim
        # and the forward's own guard, leaking the capture reservation for as
        # long as the exception object lived -- the next
        # tl.trace then refused reentrant_trace on an idle process.
        echo_session = open_echo_session(trace, r.echo_options)
    except BaseException:
        _state.reset_capture_runtime_context()
        release_slot()
        raise
    return trace, echo_session


def _run_resolved_request(r: SimpleNamespace) -> Trace:
    """The run half: slot claim, runtime context, Trace build, forward, tail.

    The slot is claimed FIRST (see ``_configure_runtime_context``); the
    inner claim in ``run_and_log_inputs_through_model`` passes on the token.
    """

    capture_slot = _state.capture_reservation()
    reservation_token = capture_slot.__enter__()
    reservation_live = [True]

    def _release_capture_slot() -> None:
        """Release the early admission claim exactly once."""

        if reservation_live[0]:
            reservation_live[0] = False
            capture_slot.__exit__(None, None, None)

    try:
        _configure_runtime_context(r)
    except BaseException:
        _release_capture_slot()
        raise
    from .._snoop._entry import echo_forward_failure

    trace, echo_session = _build_trace_for_run(r, _release_capture_slot)
    with capture_rule_fire_ledger() as fired_rule_ids:
        try:
            _forward_under_minter(trace, r, reservation_token)
        except BaseException as exc:
            # Crash tail (snoop tail 2): synchronous flush; note rides PEP 678.
            echo_forward_failure(echo_session, exc)
            _abort_streaming_writer(trace, exc)
            raise
        finally:
            _state.reset_capture_runtime_context()
            _release_capture_slot()
            if hasattr(trace, "_capture_container_structure"):
                delattr(trace, "_capture_container_structure")
    _finish_run_entry(trace, r, echo_session, frozenset(fired_rule_ids))
    return trace


def run_capture(request: CaptureRequest) -> ForwardResult:
    """Run one capture forward pass for a packed request.

    RUN stage entry. The request's ``plan`` carries the flat keyword
    arguments of ``_run_model_and_save_specified_outs``; resolution (alias
    refusal, device placement, root identity, selector lowering, hook plan)
    happens first, then the run half claims the capture slot, builds the
    Trace, runs the forward and the post-forward tail.

    Parameters
    ----------
    request:
        Request built by ``prepare_capture`` or the keyword adapter.

    Returns
    -------
    ForwardResult
        The populated trace with the arm it settled on (``completed``, or
        ``halted`` when a halt predicate returned a partial trace).
    """
    trace = _run_resolved_request(_resolved_request(_with_plan_defaults(request)))
    return ForwardResult(
        trace=trace,
        request=request,
        arm=_forward_arm_of(trace),
        raw_output=getattr(trace, "transformed_output", None),
    )
