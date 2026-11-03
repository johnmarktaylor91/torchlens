"""Output marking and settlement helpers for the FINALIZE stage.

Moved verbatim from ``torchlens._capture.trace`` by the capture seam.
"""

import time
import warnings
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from torch import nn

from torchlens.backends.registry import BackendUnsupportedError

from ..._output_slot import stopped
from ...backends.registry import (
    CaptureBackend,
    get_backend_spec,
)
from ...recording._halt import StopSignal
from ..outcome import (
    CapturePhase,
    outcome_for,
    safe_exception_str,
    set_capture_phase,
    settle_failed,
)

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
from ..._utils.display import _vprint
from .._forward_helpers import _backend_name_for_trace
from .._input_records import _get_op_nums_from_user_labels


class _RefreshRequest(NamedTuple):
    """The caller's refresh arguments, bundled so the private helpers stay narrow."""

    model: object
    input_args: Any | list[Any]
    input_kwargs: dict[Any, Any] | None
    save: str | list[Any]
    save_grads: str | list[Any] | None
    seed: int | None
    run_until_plan: Any | None


def save_new_outs(
    self: "Trace",
    model: object,
    inputs: Any | list[Any],
    input_kwargs: dict[Any, Any] | None = None,
    save: str | list[Any] = "all",
    save_grads: str | list[Any] | None = "all",
    seed: int | None = None,
    backward_ready: bool | None = None,
    _run_until_plan: Any | None = None,
) -> None:
    """Re-run the model with new inputs, saving refreshed outs.

    This is the public API for refreshing outs without rebuilding the
    computational graph.  Much faster than ``trace`` because all
    metadata (graph structure, labels, module context) was captured in the
    original exhaustive pass and is reused here.

    The refresh assumes the computational graph is identical to the original
    pass. The refresh projector validates the captured graph and raises
    ``ValueError`` when dynamic control flow changes it.

    Parameters

    ----------
        model: Model for which to save outs.
        inputs: Either a single tensor input to the model, or list of input arguments.
        input_kwargs: Dict of keyword arguments to the model.
        save: List of layers to save, using any valid lookup keys.
        save_grads: List of layers whose grads should be saved.
        seed: Which random seed to use for deterministic reproduction.
        backward_ready: Optional replay override. ``None`` inherits the existing
            model log settings; explicit values temporarily override saved
            tensor detachment for the whole replay.

    Returns

    -------
        Nothing; mutates ``self`` in place with new out values.
    """
    request = _RefreshRequest(model, inputs, input_kwargs, save, save_grads, seed, _run_until_plan)
    if backward_ready is not None:
        _refresh_under_backward_ready(self, request, backward_ready)
        return
    _refresh_and_project(self, request)


def _refresh_under_backward_ready(
    self: "Trace", request: _RefreshRequest, backward_ready: bool
) -> None:
    """Refresh under a temporary detach/backward-ready override, restoring it after."""

    (
        model,
        input_args,
        input_kwargs,
        save,
        save_grads,
        seed,
        _run_until_plan,
    ) = request
    model_detach_saved_activations = self._options_detach_saved_activations
    model_train_mode = self._options_backward_ready
    layer_detach_saved_activations = {
        layer_log_entry: layer_log_entry._detach_saved_activations for layer_log_entry in self
    }
    target_detach_saved_activations = (
        False if backward_ready else self._options_detach_saved_activations
    )
    try:
        self._options_detach_saved_activations = target_detach_saved_activations
        self._options_backward_ready = backward_ready
        for layer_log_entry in layer_detach_saved_activations:
            layer_log_entry._detach_saved_activations = target_detach_saved_activations
        save_new_outs(
            self,
            model=model,
            inputs=input_args,
            input_kwargs=input_kwargs,
            save=save,
            save_grads=save_grads,
            seed=seed,
            backward_ready=None,
            _run_until_plan=_run_until_plan,
        )
    finally:
        self._options_detach_saved_activations = model_detach_saved_activations
        self._options_backward_ready = model_train_mode
        for layer_log_entry, detach_saved_activations in layer_detach_saved_activations.items():
            layer_log_entry._detach_saved_activations = detach_saved_activations


def _refresh_and_project(self: "Trace", request: _RefreshRequest) -> None:
    """Resolve the layer selections, run the refresh forward, then project it."""

    save = request.save
    save_grads = request.save_grads
    seed = request.seed
    _run_until_plan = request.run_until_plan
    layer_nums_to_save = _get_op_nums_from_user_labels(self, save)
    grad_layer_nums_to_save = _get_op_nums_from_user_labels(self, save_grads)
    refresh_seed = self._options_seed if seed is None else seed
    resolved_layer_nums: tuple[int, ...] | None = None
    if layer_nums_to_save != "all":
        expanded_layer_nums = set(cast(list[int], layer_nums_to_save))
        for output_label in self.output_layers:
            output = self._layer_dict_all_keys[output_label]
            expanded_layer_nums.update(
                self._layer_dict_all_keys[parent].raw_index for parent in output.parents
            )
        resolved_layer_nums = tuple(sorted(expanded_layer_nums))
    refreshed = _run_refresh_forward(
        self, request, refresh_seed, resolved_layer_nums, grad_layer_nums_to_save
    )
    _project_refreshed(
        self, refreshed, layer_nums_to_save, grad_layer_nums_to_save, _run_until_plan
    )


def _refresh_replay_config(self: "Trace") -> dict[str, Any]:
    """Return the replay configuration the refresh capture inherits from ``self``."""

    return {
        "output_device": self._options_output_device,
        "activation_transform": self._options_activation_transform,
        "grad_transform": self._options_grad_transform,
        "save_raw_activations": self._options_save_raw_activations,
        "save_raw_grads": getattr(self, "save_raw_gradients", True),
        "save_mode": self._options_save_mode,
        "capture_tensor_grad_hooks": getattr(self, "capture_tensor_grad_hooks", True),
        "keep_orphans": self._options_keep_orphans,
        "mark_layer_depths": self._options_mark_layer_depths,
        "detach_saved_activations": self._options_detach_saved_activations,
        "save_args": self._options_save_args,
    }


def _run_refresh_forward(
    self: "Trace",
    request: _RefreshRequest,
    refresh_seed: int | None,
    resolved_layer_nums: tuple[int, ...] | None,
    grad_layer_nums_to_save: str | list[Any],
) -> Any:
    """Run the internal refresh capture over the new inputs and return its trace."""

    from ..._user_funcs import _run_model_and_save_specified_outs

    model, input_args, input_kwargs, _layers, save_grads, _seed, _run_until_plan = request
    save_grads_policy = self._options_save_grads
    return _run_model_and_save_specified_outs(
        model=cast(nn.Module, model),
        input_args=input_args,
        input_kwargs=input_kwargs or {},
        layers_to_save="all" if resolved_layer_nums is None else "none",
        **_refresh_replay_config(self),
        save_grads=save_grads_policy not in (None, False),
        grads_to_save=save_grads,
        seed=refresh_seed,
        num_context_lines=getattr(self, "num_context_lines", 7),
        optimizer=getattr(self, "_optimizer", None),
        save_source_context=self._options_save_source_context,
        save_rng=self._options_save_rng,
        recurrence_detection=self._options_recurrence_detection,
        verbose=self._options_verbose,
        backward_ready=self._options_backward_ready,
        inference_only=self._options_inference_only,
        # A refresh retains payloads like any capture; dropping the configured
        # budget here silently rebudgeted every refresh/run() forward at the
        # default "auto" (or left a save_budget=None session budgeted).
        save_budget=getattr(self, "save_budget", "auto"),
        # F2: a refresh re-arms the nonfinite tripwire. The historical refresh
        # forwarded only inference_only, so a raise_on_nan capture silently
        # lost its abort policy on every refreshed forward.
        raise_on_nan=bool(self._options_on_nonfinite),
        # A refresh measures CPU peaks only if the source opted in.
        measure_host_peak_memory=bool(getattr(self, "measure_host_peak_memory", False)),
        # A refresh records ATen operations only if the source opted in.
        aten_ops=bool(getattr(self, "_options_aten_ops", False)),
        output_transform=getattr(self, "_output_transform", None),
        save_transformed_output=self._options_save_transformed_output,
        retain_output_parents_for_layers_to_save=True,
        _resolved_layer_nums_to_save=resolved_layer_nums,
        _resolved_grad_layer_nums_to_save=(
            grad_layer_nums_to_save
            if grad_layer_nums_to_save == "all"
            else tuple(cast(list[int], grad_layer_nums_to_save))
        ),
        _refresh_projection_capture=True,
        # L4 2.3 live until=: the run-installed halt latch rides the EXISTING
        # halt= surface of the internal refresh capture (one forwarded argument;
        # the driver's halt arm settles the throwaway HALTED and returns the
        # partial normally into the projection flow below).
        halt_predicate=None if _run_until_plan is None else _run_until_plan.halt_predicate,
    )


def _project_refreshed(
    self: "Trace",
    refreshed: "Trace",
    layer_nums_to_save: str | list[Any],
    grad_layer_nums_to_save: str | list[Any],
    _run_until_plan: Any | None,
) -> None:
    """Project the refreshed capture onto ``self`` (prefix-scoped when a latch fired)."""

    from ..projectors import RefreshProjector

    projected_layer_nums = (
        "all" if layer_nums_to_save == "all" else tuple(cast(list[int], layer_nums_to_save))
    )
    projected_grad_layer_nums = (
        "all"
        if grad_layer_nums_to_save == "all"
        else tuple(cast(list[int], grad_layer_nums_to_save))
    )
    if _run_until_plan is not None and _run_until_plan.fired:
        # L4 2.3 truncated refresh: the latch fired and the internal capture
        # settled HALTED. The projection is PREFIX-SCOPED (all projector
        # tripwires at full strength on the executed prefix; a prefix mismatch
        # refuses exactly like a full mismatch), and BOTH post-return feedback
        # writes are suppressed with their inherited authority explicitly
        # neutralized -- never left to absence:
        #   (1) output-losslessness: the fork INHERITED the source's positive
        #       stamp via the fork builder's runnable copy pass, and the halt
        #       frontier's own proof attests the truncated frontier tensor,
        #       never the full-forward output -- so the :672-style copy is
        #       SKIPPED and the inherited proof is CLEARED to None (the shipped
        #       fail-closed state; the live reconstructor fails closed).
        #   (2) replay-arg completeness: a truncated refresh did not witness
        #       complete replay arg/version data over a partial forward, and the
        #       field initializes True at construction -- so it is SET FALSE
        #       explicitly, never merely skipped.
        RefreshProjector(
            self,
            projected_layer_nums,
            projected_grad_layer_nums,
        ).project_prefix(refreshed, _run_until_plan)
        self._output_slot = refreshed._output_slot
        self._runnable.output_losslessness = None
        self._replay_arg_version_data_complete = False
        return
    RefreshProjector(
        self,
        projected_layer_nums,
        projected_grad_layer_nums,
    ).project(refreshed)
    # r39 corr2_5: copy the FRESH refresh forward's output-losslessness proof onto the
    # projected fork. A changed input may select a different return-container KIND than the
    # original capture, so the live provider must gate its bare-tensor fast path on the fresh
    # proof (``bare_tensor_root``), not the stale capture-time one. Missing/malformed fresh
    # proof leaves the field absent -> the live reconstructor fails closed (not faithful).
    self._runnable.output_losslessness = refreshed._runnable.output_losslessness
    if self._options_save_args:
        self._replay_arg_version_data_complete = True


#: Session-time semantic-output scratch (B1-02). Written at capture entry
#: (``user_funcs.py``) and consumed ONLY by ``decode_outputs_for_trace`` on the
#: normal forward-return arm. Both pin LIVE USER OBJECTS -- an HF
#: tokenizer (``bridge/hf.py``) and a model-derived metadata key -- so a copy
#: surviving onto a settled product is both a retention leak and a privacy leak:
#: plain ``pickle``/``torch.save`` of the Trace serializes the tokenizer's
#: vocab/merges into an artifact the user believes is a graph. Declared
#: ``FieldPolicy.DROP`` on ``Trace`` and dropped on EVERY settlement path.
_SEMANTIC_OUTPUT_TRANSIENT_FIELDS: tuple[str, ...] = (
    "_output_tokenizer",
    "_semantic_output_metadata",
)


def _drop_semantic_output_transients(self: "Trace") -> None:
    """Drop the session-time semantic-output scratch from one trace.

    Two call sites: ``finalize.arms.finalize_completed`` drops right
    after the sole consumer (``decode_outputs_for_trace``) on the completed
    arm, and ``run._finish_run`` repeats the drop in RUN's outermost
    ``finally`` so the halted, failed and interrupted arms (which never
    decode) drop too. Idempotent, so the completed arm's second drop is a no-op.

    Parameters
    ----------
    self:
        Trace whose semantic-output scratch should be discarded.

    Returns
    -------
    None. Mutates ``self.__dict__``.
    """

    for attr_name in _SEMANTIC_OUTPUT_TRANSIENT_FIELDS:
        self.__dict__.pop(attr_name, None)


def _publish_streamed_bundle_at_settlement(self: "Trace") -> None:
    """Publish a staged streamed bundle with its just-settled outcome.

    WT1 A-IV item 18 (lane A08): postprocess step 18 STAGES the streamed
    bundle and this seam -- immediately after ``settle_completed`` /
    ``settle_halted`` -- performs the tmp->final publish, injecting the
    settled capture-outcome attestation into ``metadata.pkl`` (the same
    ``_capture_outcome`` key ordinary saves persist). A capture that settles
    FAILED never reaches here: the propagating exception hits the streaming
    abort handler in ``user_funcs.py`` and the temp bundle is marked PARTIAL,
    never published. No-op when nothing is staged (non-streamed captures, the
    deferred grad-streaming mode, the postprocess-free fastlog arm).
    """

    writer = self.__dict__.get("_out_writer")
    if writer is None or not getattr(writer, "staged_for_settlement", False):
        return
    settled = outcome_for(self)
    writer.publish_staged(None if settled is None else settled.to_payload())
    self._out_writer = None


def _scrub_failed_capture_transients(self: "Trace") -> None:
    """Failure-axis twin of the success arms' transient drops (R11/R32).

    The success arms pop ``_output_attribution_input_tensors`` (live USER
    INPUT tensors) and postprocess replaces the mutable ``capture_events``
    alias with the payload-free ``_capture_events`` home. A failed or
    interrupted forward reached neither, so the trace escaping on
    ``exc.partial_trace`` pickled the user's input tensors through an
    undeclared attribute and retained every activation payload and grad_fn
    handle of the failed run (GB-class on real models).

    Partial diagnostics stay intact: ``PartialTrace.from_trace`` materialized
    the raw layers during backend cleanup, strictly before this scrub, and
    sidecar release keeps the structural event facts.

    Predicate (fastlog) captures keep their event buffer untouched apart from
    the trace-side alias pop: the buffer may be OWNED by the live Recorder
    (a shared object accumulating prior passes), and the failed-pass snapshot
    (``_failed_fastlog_capture_events``) already isolated the failing pass.

    Parameters
    ----------
    self:
        Trace settled FAILED whose capture transients should be discarded.

    Returns
    -------
    None. Mutates ``self.__dict__``.
    """

    from ..._utils.tensor_utils import synchronize_pending_cpu_async_copies

    # A failed forward may leave in-flight cpu_async D2H copies whose
    # pinned host buffers escape on ``exc.partial_trace`` (R36-1): fence them
    # before anything reads the partial's payloads.
    synchronize_pending_cpu_async_copies()
    self.__dict__.pop("_output_attribution_input_tensors", None)
    # The aborted-nonfinite frontier stash is a live activation tensor; a
    # failed capture that never ran the prefix finalization must not retain
    # it past settlement (same R11/R32 class as the attribution inputs).
    self.__dict__.pop("_nonfinite_frontier_out", None)
    events = self.__dict__.pop("capture_events", None)
    if events is None:
        return
    if self._options_mode == "predicate":
        return
    if hasattr(events, "release_runtime_sidecars"):
        events.release_runtime_sidecars()
        # The trace is the sole strong owner of its (sidecar-released) event
        # stream, mirroring the postprocess success seam.
        self.__dict__["_capture_events"] = events


def _extract_and_mark_outputs(
    self: "Trace",
    outputs: Any,
    backend: CaptureBackend | None = None,
) -> tuple[list[Any], list[str]]:
    """Extract output tensors from model outputs through the active backend.

    Called AFTER the forward pass completes (outside ``active_logging``). The
    backend marks each output tensor's graph entry as ``is_output_parent=True``
    so postprocessing can identify them.

    Parameters
    ----------
    self:
        Active trace.
    outputs:
        Raw model output object returned by the captured forward pass.
    backend:
        Active capture backend. When omitted, the backend is loaded from the
        trace backend name through the registry.

    Returns
    -------
    tuple[list[Any], list[str]]
        Output tensors and output tensor addresses.
    """
    if backend is None:
        spec = get_backend_spec(str(_backend_name_for_trace(self)))
        if spec.capture_backend is None:
            raise BackendUnsupportedError(
                f"backend={spec.name!r} does not expose a shared capture Protocol adapter."
            )
        backend = spec.capture_backend()
    output_tensors, output_tensor_addresses = backend.extract_and_mark_outputs(
        self,
        outputs,
    )
    return list(output_tensors), output_tensor_addresses


def _settle_interrupted_halted_arm(
    trace: "Trace",
    capture_session: Any,
    interrupt_exc: BaseException,
    halt_exc: StopSignal,
) -> None:
    """Stamp an interrupt that escaped the halted arm, never masking it.

    Mirrors the outer ``except BaseException`` arm's guarded settlement: the
    stamp is FAILED/interrupted with the halt boundary disclosed, and an
    ordinary settlement/scrub failure attaches as a note instead of replacing
    the unwinding KeyboardInterrupt/SystemExit (B8-23 discipline).
    """

    try:
        settle_failed(
            trace,
            capture_session,
            interrupt_exc,
            interrupted=True,
            note=(
                "interrupted during halted finalization after halt at "
                f"{getattr(halt_exc, 'reason', '')!r}"
            ),
        )
        _scrub_failed_capture_transients(trace)
    except Exception as settle_exc:
        note = (
            "TorchLens settlement/scrub also failed while handling this "
            f"interrupt: {type(settle_exc).__name__}: {safe_exception_str(settle_exc)}"
        )
        add_note = getattr(interrupt_exc, "add_note", None)
        if add_note is not None:
            add_note(note)
        else:  # Python 3.10: no PEP 678 notes -- surface via warning.
            warnings.warn(note, RuntimeWarning, stacklevel=2)


def _recover_halt_frontier_output(self: "Trace", halt_exc: StopSignal) -> Any | None:
    """Recover the halt-frontier tensor from the signal or the raw layer dict."""

    frontier_output = halt_exc.frontier_output
    if frontier_output is None:
        raw_layer_dict = self._raw_graph_ws.raw_layer_dict
        for event in reversed(getattr(self.capture_events, "op_events", ())):
            entry = raw_layer_dict.get(event.label_raw)
            if entry is not None and getattr(entry, "out", None) is not None:
                frontier_output = entry.out
                break
    return frontier_output


def _extract_and_postprocess_halted(
    self: "Trace", backend: CaptureBackend, frontier_output: Any
) -> None:
    """Extract and mark the frontier outputs, then run the halted postprocess."""

    output_tensors, output_tensor_addresses = _extract_and_mark_outputs(
        self,
        frontier_output,
        backend,
    )
    # Mirror the completed paths: the transient attribution inputs are consumed
    # by output extraction and must never survive onto the finished product
    # (an unpopped copy blocked halted analysis saves via _PORTABLE_STATE_SPEC).
    self.__dict__.pop("_output_attribution_input_tensors", None)
    _vprint(self, f"Postprocessing halted graph at {self._graph_stop_frontier!r}...")
    set_capture_phase(self, CapturePhase.POSTPROCESS)
    self._postprocess(output_tensors, output_tensor_addresses)


def _finalize_halted_trace(
    self: "Trace",
    backend: CaptureBackend,
    halt_exc: StopSignal,
    model: object,
    input_tensors: list[Any],
    postprocess: bool,
) -> Any | None:
    """Finalize a predicate trace that stopped at a halt frontier.

    Parameters
    ----------
    self:
        Active trace.
    backend:
        Active capture backend.
    halt_exc:
        Halt signal raised by the predicate layer.
    model:
        Model whose session metadata should be cleaned up.
    input_tensors:
        Input tensors tagged for the current pass.
    postprocess:
        Whether to run the standard postprocess pipeline.

    Returns
    -------
    Any | None
        Halt-frontier output object when available.
    """

    # Settlement phase marker: everything from halted cleanup through frontier
    # recovery and output extraction is the FINALIZE failure class; the halted
    # postprocess below is POSTPROCESS. The two classes are never conflated.
    set_capture_phase(self, CapturePhase.FINALIZE)
    # F3a: recover the frontier BEFORE model-session cleanup. The cleanup
    # strips TorchLens metadata from model-owned tensors, so the reverse scan
    # must read the raw entries while their capture-time state is intact.
    frontier_output = _recover_halt_frontier_output(self, halt_exc)
    backend.cleanup_model_session(self, (model, input_tensors))
    if frontier_output is None:
        raise RuntimeError(
            "trace(stop_after=...) could not identify a tensor frontier for the halted partial graph."
        ) from halt_exc

    self.is_stopped = True
    self.stop_reason = halt_exc.reason
    self._graph_stop_frontier = halt_exc.reason
    self.transformed_output = None

    self._output_slot = stopped(halt_exc.reason)
    if not postprocess:
        self.__dict__.pop("_output_attribution_input_tensors", None)
        self._provenance_end_time = time.time()
        return frontier_output

    _extract_and_postprocess_halted(self, backend, frontier_output)
    return frontier_output
