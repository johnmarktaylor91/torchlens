"""Torch door for the PREPARE stage: admission, refusals and request resolution.

The door body moved here verbatim from ``torchlens._user_funcs`` by the capture
seam; the PREPARE lane split it into stage functions over a frozen ``_DoorState``
that end in ``prepare_capture`` (one ``CaptureRequest``), the cache lookup, the
chunk fan-out, the run bracket and the door epilogue. The cache-key, selector and
chunk-options stages live in the sibling ``cache``, ``resolve`` and ``chunked`` homes.
"""

import collections.abc
import os
import warnings
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from ..._capture_state_helpers import (
    _reject_opaque_wrappers,
    _unwrap_data_parallel,
    unwrap_compiled_model,
)
from ..._data_classes.trace import (
    Trace,
)
from ..._deprecations import MISSING, MissingType
from ..._errors import (
    ArgumentConflictError,
    ArgumentTypeError,
    CaptureNotActiveError,
    InvalidArgumentError,
    TorchLensWarning,
)
from ..._input_coerce import (
    _coerce_input_args,
    _reject_non_mapping_input_kwargs,
)
from ..._ir.selector_eval import looks_like_finalized_label
from ..._options_validation import _validate_output_device
from ..._robustness import check_model_and_input_variants
from ..._runnable.state_live import capture_observer_bracket
from ..._selection.selectors import BaseSelector, FuncSelector, ModuleSelector
from ..._trace_selector_helpers import (
    _resolve_save_transforms,
    _route_save_label_selection,
    _split_save_options_and_predicate,
)
from ..._training_validation import (
    TrainingModeConfigError,
    effective_inference_only,
    validate_training_compatibility,
)
from ..._utils.display import warn_parallel
from ..._utils.tensor_utils import SaveMode
from ...backends.torch import rescue as _rescue
from ...backends.torch._op_markers import refuse_unavailable_nvtx
from ...backends.torch.bound_root import (
    TLBoundMethodRoot,
    is_bound_method_of_module,
)
from ...errors.episode import EpisodeDeclarationError
from ...intervention.errors import ChunkedForwardConfigError
from ...options import (
    CaptureOptions,
    ReplayOptions,
    _SaveTransforms,
    merge_capture_options,
    merge_streaming_options,
)
from ...recording.options import PredicateFn
from ...types import _ActivationPostfunc, _GradientPostfunc
from .._episode_coupling import CouplingSession, rule_identity_of
from .._episode_join import armed_capture
from .._episode_ledger import (
    resolve_episode_declaration,
)
from .._structure_only_entry import (
    _enforce_structure_only_entry_contract,
    _StructureOnlyEntryFacts,
)
from .._weightsfree_admission import pending_admission
from ..finalize.epilogue import (
    _render_layer_visualizers,
    _restrict_layer_nums_to_saved,
    _store_capture_cache,
    attach_failed_episode_ledger,
    finish_trace,
)
from ..run_entry import run_capture
from ..stages import CaptureRequest, ForwardResult
from ..structure_only import STRUCTURE_ONLY_EPISODE_UNSUPPORTED
from .cache import (
    _load_authenticated_capture_cache,
    _resolve_cache_key,
    _store_capture_cache_entry,
)
from .chunked import (
    _append_chunk_trace_state,
    _ChunkAppend,
    _should_store_auto_coerced_raw_input,
    _typed_chunk_forward,
    _validate_chunked_forward_capture,
    iter_chunked_inputs,
    normalize_chunk_paths,
    normalize_chunk_size,
    plan_chunks,
)
from .dispatch import _reject_unsupported_torch_trace_option_values
from .recursive_options import _recursive_capture_options
from .request import _build_request, _DoorArgs, _DoorState
from .resolve import (
    _refuse_predicate_shapes,
    _resolve_selectors,
    _validate_predicate_capabilities,
)

# list-A row 30 latch: once-per-process disclosure that train-mode tracing
# advances norm running statistics. Process-global by design ("warn once");
# tests that assert the warning must save and restore this flag.
_BATCHNORM_TRAIN_STATS_WARNED = False


def _warn_once_train_mode_running_stats(model: nn.Module) -> None:
    """Disclose (once per process) that this capture mutates running stats.

    Parameters
    ----------
    model:
        Model about to run its real forward under capture.

    Returns
    -------
    None
        Warns with code ``batchnorm_train_stats_mutated`` and latches the
        process-global flag when a train-mode norm layer tracking running
        statistics is found; otherwise a no-op that leaves the flag unset
        (a later capture of a mutating model still discloses).
    """

    global _BATCHNORM_TRAIN_STATS_WARNED
    from torch.nn.modules.batchnorm import _BatchNorm
    from torch.nn.modules.instancenorm import _InstanceNorm

    mutating = [
        module
        for module in model.modules()
        if isinstance(module, (_BatchNorm, _InstanceNorm))
        and module.training
        and getattr(module, "track_running_stats", False)
        and getattr(module, "running_mean", None) is not None
    ]
    if not mutating:
        return
    _BATCHNORM_TRAIN_STATS_WARNED = True
    warnings.warn(
        TorchLensWarning(
            "tracing runs the model's REAL forward with CaptureOptions(carry_state=True) "
            f"(forward-equivalent state): {len(mutating)} norm layer(s) in train mode with track_running_stats=True updated "
            "their running statistics in place during this capture "
            "(running_mean/running_var advance under momentum, even inside "
            "torch.no_grad or inference_only=True). This disclosure fires "
            "once per process. "
            "Remedy: drop carry_state=True for an observational capture (the "
            "default restores the statistics afterwards); "
            "training-through-trace users can ignore this",
            code="batchnorm_train_stats_mutated",
        ),
        stacklevel=2,
    )


def _release_preparation_after_failed_capture(model: nn.Module) -> None:
    """Strip TorchLens preparation from a model whose capture failed.

    A FAILED capture must leave the model and TorchLens's preparation
    bookkeeping "as if never traced", whatever ended it: the session cleanup
    already put every submodule ``forward`` back, and this also evicts the
    module metadata and prepared-model registry entries so the next capture
    prepares from scratch. Every ``BaseException`` (``KeyboardInterrupt``,
    ``SystemExit``) takes this path, exactly like an ordinary ``Exception``.
    State the partial forward mutated (buffers, random state, modes, plain
    attributes) is rolled back by the capture observer bracket, before this
    release, unless
    ``CaptureOptions(carry_state=True)``; the failure warning states the boundary.

    Parameters
    ----------
    model:
        Model whose capture attempt raised. Releasing an unprepared model is
        a no-op, so chunked fan-outs that already released on an inner
        failure are safe.

    Returns
    -------
    None
        The model is released in place; a secondary release failure warns
        coded (``failed_capture_release_incomplete``) and never masks the
        capture exception.
    """

    from ...backends.torch.model_prep import release_model

    try:
        release_model(model)
    except CaptureNotActiveError as refusal:
        if refusal.fields.get("code") == "release_during_active_capture":
            # A NESTED capture failed while an outer capture still owns the
            # model's instrumentation (e.g. tl.trace called from a forward
            # hook): releasing here would strip the module metadata the live
            # capture is reading. The refusal is the guard working; the outer
            # capture's own failure/teardown path owns the release, so
            # nothing leaks by skipping.
            return
        _warn_failed_capture_release_incomplete(refusal)
    except Exception as release_exc:
        _warn_failed_capture_release_incomplete(release_exc)


def _warn_failed_capture_release_incomplete(release_exc: BaseException) -> None:
    """Disclose a secondary failure while releasing after a failed capture.

    Parameters
    ----------
    release_exc:
        The exception the release raised.

    Returns
    -------
    None
        Warns coded; never raises (the capture exception must propagate).
    """

    warnings.warn(
        TorchLensWarning(
            "TorchLens could not fully release its preparation of "
            f"the model after the failed capture "
            f"({type(release_exc).__name__}: {release_exc}); its module "
            "metadata and prepared-model bookkeeping may stay registered and "
            "plain attributes holding torch functions may stay unnormalized. "
            "Remedy: call tl.release_model(model) once the underlying "
            "condition is resolved",
            code="failed_capture_release_incomplete",
        ),
        stacklevel=4,
    )


def _model_type_remedy(model: Any) -> str:
    """Remedy for ``model_type_unsupported``; export products get their own."""

    exported_program = getattr(getattr(torch, "export", None), "ExportedProgram", None)
    if isinstance(exported_program, type) and isinstance(model, exported_program):
        return (
            "trace the source nn.Module before export: tl.trace(model, x) on the eager "
            "module you passed to torch.export.export (an ExportedProgram is a lowered "
            "graph, not the model's own forward)"
        )
    return (
        "pass a torch.nn.Module, or a bound method of an nn.Module "
        "(owner resolved via method.__self__, e.g. "
        "tl.trace(model.generate, input_ids)), or select the backend "
        "that owns the supplied model"
    )


def _trace_torch_model(
    model: nn.Module | Callable[..., Any],
    input_args: str | torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    **options: Any,
) -> Trace:
    """Run the registry-owned torch trace implementation.

    Parameters
    ----------
    model:
        PyTorch model.
    input_args:
        Positional args for ``model.forward()``.
    input_kwargs:
        Keyword args for ``model.forward()``.
    **options:
        The remaining parameters match ``trace`` exactly, excluding ``backend``;
        see ``_DoorArgs`` for the accepted names and defaults.

    Returns
    -------
    Trace
        Captured torch trace.
    """
    state = _resolve_door(model, input_args, input_kwargs, _DoorArgs(**options))
    request = _build_request(state, cache_store=_store_capture_cache_entry)
    cached = _lookup_capture_cache(request)
    if cached is not None:
        return cached
    _disclose_running_stats_mutation(request)
    if _chunk_fanout_needed(state):
        return _run_chunked_capture(state, request)
    return finish_trace(_run_request(request), request)


def prepare_capture(
    model: nn.Module | Callable[..., Any],
    input_args: Any,
    input_kwargs: dict[Any, Any] | None = None,
    **options: Any,
) -> CaptureRequest:
    """Resolve one torch door call into the request the RUN stage consumes.

    Every refusal, warning and resolution that happens before the forward pass
    runs here, in today's order; no torch state is touched, no wrapper installed
    and no ``Trace`` built. The capture-cache directory is prepared here (as
    today); the lookup itself is ``_lookup_capture_cache(request)``.

    Parameters
    ----------
    model:
        The user's model (an ``nn.Module`` or a bound method of one).
    input_args:
        Positional forward inputs.
    input_kwargs:
        Keyword forward inputs.
    **options:
        The door keywords (``_DoorArgs``).

    Returns
    -------
    CaptureRequest
        ``plan`` holds the run keywords; ``cache`` the ``key``, ``path`` and
        ``secret`` (None when not caching); ``admission`` the door facts the run
        bracket and epilogue read: ``meta_admission``, ``rescue_eligible``,
        ``episode_resolved``, ``stop_after_site``, ``stop_after_selector_shaped``,
        ``stop_after_ambient``, ``save_grads_policy`` and
        ``uses_selective_layers_to_save``.
    """
    return _build_request(
        _resolve_door(model, input_args, input_kwargs, _DoorArgs(**options)),
        cache_store=_store_capture_cache_entry,
    )


def _resolve_door(
    model: nn.Module | Callable[..., Any],
    input_args: Any,
    input_kwargs: dict[Any, Any] | None,
    args: _DoorArgs,
) -> _DoorState:
    """Run the PREPARE stages in today's order and return the resolved door state."""
    admitted = _admit_model(model, input_kwargs)
    capture_options = _merge_door_capture_options(args)
    _refuse_option_shapes(capture_options)
    state = _open_door_state(args, admitted, input_args, input_kwargs, capture_options)
    state = _admit_and_route_save(state)
    _refuse_predicate_shapes(args)
    state = _resolve_episode(state)
    state = _resolve_save_and_storage(state)
    state = _unpack_save_policies(state)
    state = _resolve_stop_after(state)
    state = _resolve_gradient_policy(state)
    _refuse_output_routing(state)
    state = _apply_structure_only_entry(state)
    state = _resolve_backward_flags(state)
    state = _validate_training_and_inference(state)
    state = _resolve_chunk_plan(state)
    state = _resolve_selectors(state)
    _validate_predicate_capabilities(state)
    return _resolve_cache_key(state)


def _admit_model(model: nn.Module | Callable[..., Any], input_kwargs: Any) -> nn.Module:
    """Admit the root: an ``nn.Module`` or a bound method of one, unwrapped for capture."""
    # DataParallel is not supported - unwrap and warn if present.
    warn_parallel()
    # Reject non-Module input first: `_reject_opaque_wrappers` calls
    # `model.named_modules()` (FSDP/ScriptModule checks), which only makes sense on a real
    # nn.Module. A non-Module input previously leaked an AttributeError from that call
    # instead of the documented "Unsupported model type" ValueError.
    # The ruled root contract (foldA MEMO s5 item 9, a direct maintainer ruling) is
    # nn.Module OR a bound method of one (owner resolved via ``__self__``):
    # a bound method wraps into the TL-authored synthetic root (owner
    # registered as a submodule; method called exactly once); anything else
    # refuses teaching that spelling at the point of failure.
    if not isinstance(model, nn.Module):
        if is_bound_method_of_module(model):
            model = TLBoundMethodRoot(model)
        else:
            raise InvalidArgumentError(
                f"Unsupported model type for capture: received {type(model).__name__}, "
                "not a torch.nn.Module or a bound method of one",
                code="model_type_unsupported",
                remedy=_model_type_remedy(model),
                argument="model",
                received_type=type(model).__name__,
            )
    _reject_opaque_wrappers(model)
    # grind-r5 b7 R55: ``tl.trace(model, q, k, v)`` lands ``k`` in
    # ``input_kwargs``; trace() refuses it before dispatch for every backend,
    # and direct callers of this torch path get the same check.
    _reject_non_mapping_input_kwargs(input_kwargs)
    model = unwrap_compiled_model(model)
    return _unwrap_data_parallel(model)


def _merge_door_capture_options(args: _DoorArgs) -> CaptureOptions:
    """Merge the grouped ``capture=`` object with the flat door keywords."""
    save_args = args.save_args
    save_rng = args.save_rng
    if args.reconstruction_ready is not MISSING and args.reconstruction_ready:
        save_args = True
        save_rng = True
    return merge_capture_options(
        capture=args.capture,
        _layers_to_save=args.layers_to_save,
        input_transform=args.transform,
        save_raw_input=args.save_raw_input,
        batch_render=args.batch_render,
        output_transform=args.output_transform,
        save_transformed_output=args.save_transformed_output,
        layer_visualizers=MISSING,
        save_visualizations=MISSING,
        keep_orphans=args.keep_orphans,
        output_device=args.output_device,
        save_args=save_args,
        save_grads=args.save_grads,
        capture_tensor_grad_hooks=args.capture_tensor_grad_hooks,
        save_source_context=args.save_source_context,
        save_rng=save_rng,
        seed=args.seed,
        source_context_lines=args.source_context_lines,
        num_context_lines=args.num_context_lines,
        optimizer=args.optimizer,
        compute_input_output_distances=args.compute_input_output_distances,
        mark_layer_depths=args.mark_layer_depths,
        detach_saved_activations=args.detach_saved_activations,
        recurrence_detection=args.recurrence_detection,
        replayable=args.replayable,
        capture_container_structure=args.capture_container_structure,
        capture_output_structure=args.capture_output_structure,
        hooks=args.hooks,
        unwrap_when_done=args.unwrap_when_done,
        verbose=args.verbose,
        backward_ready=args.backward_ready,
        inference_only=args.inference_only,
        name=args.name,
        cache=args.cache,
        cache_dir=args.cache_dir,
        jax_control_flow=args.jax_control_flow,
        jax_max_control_flow_unroll=args.jax_max_control_flow_unroll,
        module_identity_mode=args.module_identity_mode,
        payload_policy=args.payload_policy,
        structure_only=args.structure_only,
    )


def _refuse_option_shapes(capture_options: CaptureOptions) -> None:
    """Refuse option values the torch door cannot honor."""
    if isinstance(capture_options._layers_to_save, torch.Tensor):
        raise ArgumentTypeError(
            "save received a torch.Tensor -- this is almost "
            "always a model input routed to the wrong slot. Bundle model inputs "
            "as one tuple, e.g. tl.trace(model, (input_a, input_b, input_c)), "
            "and use save= for selective capture.",
            code="layers_to_save_type_invalid",
            remedy="bundle positional inputs into one tuple; use save= for selection",
        )
    _reject_unsupported_torch_trace_option_values(capture_options)


def _open_door_state(
    args: _DoorArgs,
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[Any, Any] | None,
    capture_options: CaptureOptions,
) -> _DoorState:
    """Resolve echo options and the forward inputs (transform or coercion) into a state."""
    # Echo teaching refusals fire whether or not narrate= is armed: NarrationOptions
    # routed into save=/hooks= refuses with the measured receipts, and
    # echo x cache/chunked/structure_only composition refuses typed (snoop D1).
    from ..._snoop._entry import resolve_echo_capture_options

    echo_options = resolve_echo_capture_options(
        args.narrate,
        save=args.save,
        capture_options=capture_options,
        chunked=args.chunk_size is not MISSING and args.chunk_size is not None,
    )
    raw_input = None
    input_transform = capture_options.input_transform
    if input_transform is not None:
        raw_input = input_args
        transformed_input = input_transform(input_args)
        # The caller's own ``input_kwargs`` survive the transform and win on a shared key.
        if isinstance(transformed_input, collections.abc.Mapping):
            input_args = []
            input_kwargs = {**dict(transformed_input), **(input_kwargs or {})}
        else:
            input_args = transformed_input
    else:
        original_input_args = input_args
        input_args = _coerce_input_args(model, input_args)
        if _should_store_auto_coerced_raw_input(original_input_args, input_args):
            raw_input = original_input_args
    return _DoorState(
        args=args,
        model=model,
        input_args=input_args,
        input_kwargs=input_kwargs,
        capture_options=capture_options,
        raw_input=raw_input,
        echo_options=echo_options,
    )


def _admit_and_route_save(state: _DoorState) -> _DoorState:
    """Derive the meta admission once and route ``save=`` into options and predicate."""
    # W2 (weightsfree memo D2): admit_meta derives ONCE from the resolved
    # structure-only option state at this CAPTURE entry; the returned
    # admission record (None on ordinary/real captures) is armed around the
    # capture driver below and consumed by the forward scope + settlement.
    meta_admission = check_model_and_input_variants(
        state.model,
        state.input_args,
        state.input_kwargs,
        admit_meta=bool(state.capture_options.structure_only),
    )
    routed_save, capture_options = _route_save_label_selection(
        state.args.save, state.args.capture, state.capture_options, state.model
    )
    grouped_save_options, save_predicate = _split_save_options_and_predicate(routed_save)
    return replace(
        state,
        meta_admission=meta_admission,
        capture_options=capture_options,
        grouped_save_options=grouped_save_options,
        save_predicate=save_predicate,
    )


def _refuse_episode_combinations(args: _DoorArgs, capture_options: CaptureOptions) -> None:
    """Refuse the episode declarations the marker-combination table rules out."""
    if args.chunk_size not in (MISSING, None) or args.chunk_paths not in (MISSING, None):
        raise EpisodeDeclarationError(
            "episode= cannot combine with chunked forwards (chunk_size/"
            "chunk_paths): an episode is ONE wrapped session capture, and "
            "the chunk fan-out produces several.",
            code="episode_declaration_invalid",
        )
    if args.cache is not MISSING and args.cache:
        raise EpisodeDeclarationError(
            "episode= cannot combine with cache=True in this release: the "
            "capture cache replays a stored product, and episode ledgers "
            "are settled per capture.",
            code="episode_declaration_invalid",
        )
    if getattr(capture_options, "structure_only", False):
        # S2 marker-combination table: episode x structure_only is TYPED
        # REFUSE this sprint (no value semantics to fold). The refusal
        # code is L7a's capability-table constant — ONE vocabulary for
        # the combination across the entry check and the post-capture
        # capability gate (_capture/structure_only.py owns the row).
        raise EpisodeDeclarationError(
            "structure-only episodes are out of scope this sprint: "
            "capture_kind=episode with the structure-only marker refuses "
            "typed per the ratified marker-combination table.",
            code=STRUCTURE_ONLY_EPISODE_UNSUPPORTED,
        )
    if getattr(args.episode, "step_output_kind", "tokens") != "none" and (
        capture_options._layers_to_save in (None, "none", "None", "NONE")
        or (
            isinstance(capture_options._layers_to_save, list)
            and not capture_options._layers_to_save
        )
    ):
        # Kind-conditional save rule (foldA D8, lane F40b): tokens/digest
        # evidence derives from the retained root output, so a value-free
        # save policy refuses; a declared status-only episode
        # (step_output_kind='none') derives no evidence and is admitted.
        raise EpisodeDeclarationError(
            "episode= with step_output_kind='tokens'/'digest' derives its "
            "per-step evidence column from the retained root output; "
            "save='none' retains no output payload. Remedy: keep the "
            "default save policy, include the output in the save= "
            "selection, or declare step_output_kind='none' for a "
            "status-only ledger.",
            code="episode_declaration_invalid",
        )


def _resolve_episode(state: _DoorState) -> _DoorState:
    """Resolve the ``episode=`` declaration, with its coupling session when intervening."""
    episode = state.args.episode
    if episode is None:
        return state
    _refuse_episode_combinations(state.args, state.capture_options)
    episode_resolved = resolve_episode_declaration(episode, state.model)
    if state.args.intervene is not None:
        # ATTESTED COUPLING (lane F42, the foldA D5 flip): the session
        # armed here feeds the settlement writers (fire counts,
        # intervention digest, perturbed fidelity) -- the D5 evidence bar.
        episode_resolved = replace(
            episode_resolved,
            coupling_session=CouplingSession(rule_identity=rule_identity_of(state.args.intervene)),
        )
    return replace(state, episode_resolved=episode_resolved)


def _resolve_save_and_storage(state: _DoorState) -> _DoorState:
    """Merge the save options, resolve storage routing and the chunk request."""
    args = state.args
    save_options = (
        state.grouped_save_options
        if state.grouped_save_options is not None
        else _resolve_save_transforms(
            state.capture_options,
            activation_transform=args.activation_transform,
            grad_transform=args.grad_transform,
            save_raw_activations=args.save_raw_activations,
            save_raw_grads=args.save_raw_grads,
        )
    )
    streaming_options = merge_streaming_options(streaming=args.storage)
    chunk_size_value = None if isinstance(args.chunk_size, MissingType) else args.chunk_size
    chunk_paths_value = None if isinstance(args.chunk_paths, MissingType) else args.chunk_paths
    normalized_chunk_size = normalize_chunk_size(chunk_size_value)
    if chunk_paths_value is not None and normalized_chunk_size is None:
        raise ChunkedForwardConfigError("chunk_paths requires chunk_size.")
    return replace(
        state,
        save_options=save_options,
        streaming_options=streaming_options,
        chunk_size_value=chunk_size_value,
        chunk_paths_value=chunk_paths_value,
        normalized_chunk_size=normalized_chunk_size,
    )


def _unpack_save_policies(state: _DoorState) -> _DoorState:
    """Take the save policies the door rewrites later and refuse a bad ``save_mode``."""
    capture_options = state.capture_options
    save_mode_value = "copy" if state.args.save_mode is MISSING else state.args.save_mode
    if save_mode_value not in {"copy", "reference", "view", "cpu_async"}:
        raise InvalidArgumentError(
            f"save_mode={save_mode_value!r} is not supported",
            code="save_mode_invalid",
            remedy="set save_mode to 'copy', 'reference', 'view', or 'cpu_async'",
            argument="save_mode",
        )
    refuse_unavailable_nvtx(capture_options.emit_nvtx)
    facet_recipes = None if isinstance(state.args.facets, MissingType) else state.args.facets
    return replace(
        state,
        layers_to_save=capture_options._layers_to_save,
        save_raw_input_policy=capture_options.save_raw_input,
        save_transformed_output_policy=capture_options.save_transformed_output,
        save_mode_value=save_mode_value,
        facet_recipes=facet_recipes,
        halt=state.args.halt,
    )


def _resolve_stop_after(state: _DoorState) -> _DoorState:
    """Compile the door's ``stop_after=`` site (or the ambient one) into the halt slot."""
    halt = state.halt
    stop_after = state.args.stop_after
    stop_after_site = None if isinstance(stop_after, MissingType) else stop_after
    stop_after_ambient = False
    if stop_after_site is None and halt is None:
        from ...experimental import _active_stop_after_site

        stop_after_site = _active_stop_after_site()
        stop_after_ambient = stop_after_site is not None
    stop_after_selector_shaped = False
    if stop_after_site is not None:
        if state.normalized_chunk_size is not None:
            raise ChunkedForwardConfigError(
                "chunk_size cannot be combined with stop_after: the chunk fan-out runs "
                "several captures and a single stop frontier is ambiguous across them. "
                "Remedy: drop chunk_size or drop stop_after.",
                code="stop_after_chunked_conflict",
            )
        halt, stop_after_selector_shaped = _compile_stop_after_site(stop_after_site)
    return replace(
        state,
        halt=halt,
        stop_after_site=stop_after_site,
        stop_after_ambient=stop_after_ambient,
        stop_after_selector_shaped=stop_after_selector_shaped,
    )


def _compile_stop_after_site(stop_after_site: Any) -> tuple[Any, bool]:
    """Compile one stop_after site into the halt slot; the flag is True when selector-shaped."""
    if isinstance(stop_after_site, BaseSelector):
        return stop_after_site, True
    if isinstance(stop_after_site, str):
        if looks_like_finalized_label(stop_after_site):
            raise InvalidArgumentError(
                f"stop_after={stop_after_site!r} looks like a finalized postprocess "
                "label, but stop_after runs live during capture, before finalized "
                "labels exist",
                code="stop_after_site_not_live",
                remedy=(
                    "pass a module address string (e.g. 'encoder.layer.4'), a live "
                    "selector such as tl.func('relu') or tl.module('encoder.layer.4'), "
                    "or a callable predicate"
                ),
                argument="stop_after",
            )
        # A bare string is the pluck-vocabulary spelling: halt at the first
        # emission whose module address OR function name matches, whichever
        # fires first ("encoder.layer.4" never names a func; "relu" never
        # names a module address in practice, so collisions are benign).
        return ModuleSelector(stop_after_site) | FuncSelector(stop_after_site), True
    if callable(stop_after_site):
        return stop_after_site, False
    raise ArgumentTypeError(
        f"stop_after received unsupported type {type(stop_after_site).__name__}",
        code="stop_after_type_invalid",
        remedy=("pass a module address string, a tl.* selector, or a callable predicate"),
        argument="stop_after",
        received_type=type(stop_after_site).__name__,
    )


def _resolve_gradient_policy(state: _DoorState) -> _DoorState:
    """Resolve the save_grads policy into the retention selector and gradient storage."""
    save_grads_policy = state.capture_options.save_grads
    should_save_grads = save_grads_policy not in (None, False)
    if save_grads_policy is True:
        grads_to_save_resolved: Any = "all"
    elif save_grads_policy in (None, False):
        grads_to_save_resolved = None
    elif callable(save_grads_policy):
        # Selector/callable retention predicates are HONORED, never collapsed
        # to "all": they ride the deferred-gradient path and are resolved
        # against the finalized ops post-postprocess, so only matching ops
        # get gradient hooks (listA row 10 first clause: the collapse saved
        # ALL grads and silently discarded the predicate).
        grads_to_save_resolved = save_grads_policy
    else:
        grads_to_save_resolved = cast("str | list[Any] | None", save_grads_policy)
    streaming_options = state.streaming_options
    return replace(
        state,
        save_grads_policy=save_grads_policy,
        should_save_grads=should_save_grads,
        grads_to_save_resolved=grads_to_save_resolved,
        grad_storage_path_value=streaming_options.path if should_save_grads else None,
        retain_grads_in_memory_value=streaming_options.retain_in_memory,
    )


def _refuse_output_routing(state: _DoorState) -> None:
    """Refuse an invalid output device and a disk-plus-callback output sink."""
    _validate_output_device(state.capture_options.output_device)
    streaming_options = state.streaming_options
    if streaming_options.path is not None and streaming_options.on_save is not None:
        raise ArgumentConflictError(
            "Both disk-backed output storage and an output callback were configured",
            code="output_sink_conflict",
            remedy=(
                "choose either disk storage (storage=tl.to_disk(...) / "
                "streaming.path) or streaming.on_save"
            ),
            arguments=("path", "on_save"),
        )


def _apply_structure_only_entry(state: _DoorState) -> _DoorState:
    """Apply the structure-only entry contract: no values retained, raw policies off."""
    capture_options = state.capture_options
    if not capture_options.structure_only:
        return state
    layers_to_save = _enforce_structure_only_entry_contract(
        capture_options,
        _StructureOnlyEntryFacts(
            layers_to_save=state.layers_to_save,
            save_predicate=state.save_predicate,
            halt=state.halt,
            streaming_options=state.streaming_options,
            lookback_payload_policy=state.args.lookback_payload_policy,
            raise_on_nan_value=capture_options.on_nonfinite == "raise",
            track_nonfinite_value=capture_options.on_nonfinite in ("track", "warn"),
            replayable=capture_options.replayable,
            should_save_grads=state.should_save_grads,
        ),
    )
    return replace(
        state,
        layers_to_save=layers_to_save,
        save_raw_input_policy=False,
        save_transformed_output_policy=False,
    )


def _resolve_backward_flags(state: _DoorState) -> _DoorState:
    """Resolve backward_ready from the explicit flag, save_grads and gradient streaming."""
    capture_options = state.capture_options
    should_save_grads = state.should_save_grads
    train_mode_explicit = capture_options._is_field_explicit("backward_ready")
    train_mode_value = capture_options.backward_ready
    inference_only_conflicts: list[str] = []
    if capture_options._is_field_explicit("backward_ready") and train_mode_value is True:
        inference_only_conflicts.append("backward_ready")
    if capture_options._is_field_explicit("save_grads") and should_save_grads:
        inference_only_conflicts.append("save_grads")
    if capture_options._is_field_explicit("replayable") and capture_options.replayable is True:
        inference_only_conflicts.append("replayable")
    backward_opted_in = (
        capture_options._is_field_explicit("save_grads")
        and should_save_grads
        and state.save_grads_policy is not True
    )
    grad_streaming_requested = state.grad_storage_path_value is not None
    if grad_streaming_requested:
        should_save_grads = True
    if backward_opted_in:
        if train_mode_explicit and train_mode_value is False:
            raise InvalidArgumentError(
                "save_grads requests backward capture and requires backward_ready=True, but "
                "backward_ready=False was supplied",
                code="backward_capture_conflict",
                remedy="omit backward_ready or set backward_ready=True",
                arguments=("save_grads", "backward_ready"),
            )
        train_mode_value = True
        should_save_grads = True
    if train_mode_value and state.grad_storage_path_value is not None:
        raise TrainingModeConfigError(
            "backward_ready=True is not compatible with disk-backed gradient storage. "
            "Remedy: drop the gradient storage path or drop backward_ready=True.",
            code="backward_ready_conflict",
        )
    return replace(
        state,
        train_mode_value=train_mode_value,
        should_save_grads=should_save_grads,
        inference_only_conflicts=tuple(inference_only_conflicts),
    )


def _validate_training_and_inference(state: _DoorState) -> _DoorState:
    """Validate the training-mode combination and settle the effective inference_only."""
    capture_options = state.capture_options
    validate_training_compatibility(
        backward_ready=state.train_mode_value,
        streaming=state.streaming_options,
        detach_saved_activations=capture_options.detach_saved_activations,
        inference_mode_active=torch.is_inference_mode_enabled(),
        inference_only=capture_options.inference_only,
        inference_only_conflicts=state.inference_only_conflicts,
    )
    inference_only_value = effective_inference_only(
        capture_options,
        backward_requested=bool(
            state.train_mode_value or state.should_save_grads or capture_options.replayable
        ),
    )
    return replace(state, inference_only_value=inference_only_value)


def _resolve_chunk_plan(state: _DoorState) -> _DoorState:
    """Validate the chunked-forward request and plan the chunks."""
    if state.normalized_chunk_size is None:
        return state
    _validate_chunked_forward_capture(
        input_kwargs=state.input_kwargs,
        backward_ready=state.train_mode_value,
        save_grads=state.should_save_grads,
        hooks=state.capture_options.hooks,
        intervene=state.args.intervene,
        halt=state.halt,
        streaming=state.streaming_options,
    )
    chunk_plan = plan_chunks(
        state.input_args,
        chunk_size=state.normalized_chunk_size,
        chunk_paths=state.chunk_paths_value,
    )
    return replace(state, chunk_plan=chunk_plan)


def _door_options(request: CaptureRequest) -> CaptureOptions:
    """The resolved ``CaptureOptions`` of a torch door request."""
    return cast(CaptureOptions, request.options)


def _door_facts(request: CaptureRequest) -> dict[str, Any]:
    """The door's admission and epilogue facts (``CaptureRequest.admission``)."""
    return request.admission or {}


def _lookup_capture_cache(request: CaptureRequest) -> Trace | None:
    """Return the authenticated cached trace for this request, or None on a miss."""
    cache = request.cache
    if cache is None:
        return None
    cache_path = cast(Path, cache["path"])
    if not cache_path.exists():
        return None
    cached_log = cast(
        "Trace | None", _load_authenticated_capture_cache(cache_path, cache["secret"])
    )
    if cached_log is None:
        return None
    try:
        os.utime(cache_path, None)
    except OSError:
        pass
    cached_log._provenance_cache_hit = True
    cached_log._provenance_cache_key = cache["key"]
    cached_log._provenance_cache_path = str(cache_path)
    cached_log.batch_render = _door_options(request).batch_render
    return cached_log


def _disclose_running_stats_mutation(request: CaptureRequest) -> None:
    """Warn once per process that this capture advances train-mode running statistics."""
    # list-A row 30: tracing runs the model's REAL forward, so train-mode norm
    # layers with running statistics advance them in place (momentum applies
    # even inside torch.no_grad / inference_only). Disclose once per process at
    # the point the mutation is about to happen. Cache hits return above (no
    # forward, no mutation) and structure-only captures retain no values, so
    # neither path reaches this warn. The chunked fan-out recurses back into
    # the door, but the outer call warns first and latches the flag.
    capture_options = _door_options(request)
    if (
        not _BATCHNORM_TRAIN_STATS_WARNED
        and not capture_options.structure_only
        and capture_options.carry_state
    ):
        _warn_once_train_mode_running_stats(request.model)


def _chunk_fanout_needed(state: _DoorState) -> bool:
    """Whether the request fans out into several per-chunk captures."""
    chunk_plan = state.chunk_plan
    chunk_size = state.normalized_chunk_size
    return chunk_plan is not None and chunk_size is not None and chunk_size < chunk_plan.total_size


def _recursive_door_kwargs(state: _DoorState) -> dict[str, Any]:
    """Door keywords for each per-chunk recursive capture of the chunk fan-out."""
    save_options = state.save_options
    save_predicate = state.save_predicate
    activation_transform = save_options.activation_transform
    grad_transform = save_options.grad_transform
    save_raw_activations = save_options.save_raw_activations
    save_raw_grads = save_options.save_raw_gradients
    recursive_save_options = _SaveTransforms(
        activation_transform=activation_transform,
        grad_transform=grad_transform,
        save_raw_activations=save_raw_activations,
        save_raw_grads=save_raw_grads,
    )
    recursive_save_value: _SaveTransforms | PredicateFn | BaseSelector | None = (
        save_predicate if save_predicate is not None else recursive_save_options
    )
    recursive_activation_transform: _ActivationPostfunc | None | MissingType = (
        activation_transform
        if save_predicate is not None and activation_transform is not None
        else MISSING
    )
    recursive_grad_transform: _GradientPostfunc | None | MissingType = (
        grad_transform if save_predicate is not None and grad_transform is not None else MISSING
    )
    recursive_save_raw_activations: bool | MissingType = (
        save_raw_activations
        if save_predicate is not None and save_raw_activations is not True
        else MISSING
    )
    recursive_save_raw_gradients: bool | MissingType = (
        save_raw_grads if save_predicate is not None and save_raw_grads is not True else MISSING
    )
    uses_selective = state.uses_selective_layers_to_save
    return {
        "activation_transform": recursive_activation_transform,
        "grad_transform": recursive_grad_transform,
        "save_raw_activations": recursive_save_raw_activations,
        "save_raw_grads": recursive_save_raw_gradients,
        "save_mode": cast(SaveMode, state.save_mode_value),
        "capture": _recursive_capture_options(state),
        "save": recursive_save_value,
        "intervene": None,
        "halt": state.halt,
        "lookback": state.args.lookback,
        "lookback_payload_policy": state.args.lookback_payload_policy,
        "storage": None,
        "facets": state.facet_recipes,
        "chunk_size": None,
        "chunk_paths": None,
        "retain_output_parents_for_layers_to_save": uses_selective,
        "_selective_layers_to_save_request": (
            state.requested_layers_to_save if uses_selective else None
        ),
    }


def _run_chunked_capture(state: _DoorState, request: CaptureRequest) -> Trace:
    """Run the chunk fan-out: one recursive capture per chunk, appended into one trace."""
    chunk_plan = state.chunk_plan
    normalized_chunk_size = cast(int, state.normalized_chunk_size)
    chunks = iter_chunked_inputs(state.input_args, chunk_plan)
    recursive_kwargs = _recursive_door_kwargs(state)
    with _typed_chunk_forward(0, normalized_chunk_size, chunk_plan) as initial_chunk_size:
        trace = _trace_torch_model(
            state.model,
            cast(torch.Tensor | list[Any] | tuple[Any, ...], chunks[0]),
            None,
            **recursive_kwargs,
        )
    initial_record = {
        "engine": "trace",
        "append": False,
        "chunk_size": initial_chunk_size,
        "total_batch_size": chunk_plan.total_size,
        "append_sequence_id": 0,
        "chunk_paths": normalize_chunk_paths(state.chunk_paths_value),
    }
    append = ReplayOptions(append=True)
    for chunk_index, chunk in enumerate(chunks[1:], start=1):
        with _typed_chunk_forward(chunk_index, normalized_chunk_size, chunk_plan) as rows:
            if state.save_predicate is None:
                trace.run(state.model, chunk, replay=append, input_transform=False)
                continue
            new_trace = _trace_torch_model(
                state.model,
                cast(torch.Tensor | list[Any] | tuple[Any, ...], chunk),
                None,
                **recursive_kwargs,
            )
            _append_chunk_trace_state(
                trace,
                new_trace,
                _ChunkAppend(
                    chunk_size=rows,
                    total_batch_size=chunk_plan.total_size,
                    append_sequence_id=chunk_index,
                    chunk_paths=normalize_chunk_paths(state.chunk_paths_value),
                ),
            )
    trace._append_history = [initial_record, *trace._append_history]
    return _finish_chunked_trace(state, request, trace)


def _finish_chunked_trace(state: _DoorState, request: CaptureRequest, trace: Trace) -> Trace:
    """Stamp the chunked-forward facts and run the shared door epilogue tail."""
    trace._options_chunk_size = True
    trace.last_run = dict(trace.last_run or {})
    trace.last_run["chunk_size"] = state.normalized_chunk_size
    trace.last_run["chunk_paths"] = normalize_chunk_paths(state.chunk_paths_value)
    if state.uses_selective_layers_to_save:
        _restrict_layer_nums_to_saved(trace, bool(_door_options(request).save_args))
    _render_and_unwrap(request, trace)
    _store_capture_cache(trace, request.cache)
    return trace


def _run_request(request: CaptureRequest) -> ForwardResult:
    """Run the forward pass for a prepared request inside the observer bracket."""

    model = request.model
    facts = _door_facts(request)
    episode_resolved = facts["episode_resolved"]

    def run_prepared() -> Trace:
        """Run the RUN stage for this request and return its trace."""
        return run_capture(request).trace

    # a default capture is a pure observer. Buffers, random
    # state, training modes and plain attributes are restored on every path,
    # failure included; CaptureOptions(carry_state=True) keeps forward-equivalent
    # state. Both record the forward's attribute writes. On failure the bracket
    # restores FIRST and then runs the release (on_failure), so the release's
    # normalization of held torch functions stands and is not a forward write.
    # M(oracles) item 8 (FORK-A 3/3-agreed cell): every FAILED call is PURE of
    # TorchLens instrumentation; one settlement path for every escape (a
    # KeyboardInterrupt or SystemExit is released exactly like a RuntimeError).
    # The capture slot was released as the exception unwound the inner finally,
    # so the public release door is legal there; exc.partial_trace keeps its
    # already-materialized records and stays recoverable after the release.
    with capture_observer_bracket(
        model,
        carry_state=_door_options(request).carry_state,
        on_failure=lambda: _release_preparation_after_failed_capture(model),
    ) as observer:
        try:
            with (
                armed_capture(episode_resolved, run_prepared) as capture_callable,
                pending_admission(facts["meta_admission"]),
            ):
                trace = _rescue.capture_with_rescue(
                    capture_callable, eligible=facts["rescue_eligible"], model=model
                )
                observer.trace = trace
        except BaseException as capture_exc:
            if episode_resolved is not None:
                # Best-effort: the FAILED partial product carries the episode
                # declaration; attach the derived ledger disclosure without ever
                # masking the user's exception.
                attach_failed_episode_ledger(capture_exc, episode_resolved)
            raise
    return ForwardResult(trace=trace, request=request)


def _render_and_unwrap(request: CaptureRequest, trace: Trace) -> None:
    """Render the layer visualizers, then unwrap torch when the user asked for it."""
    capture_options = _door_options(request)
    layer_visualizers_value = cast(
        "dict[Any, Callable[..., Any]] | None", capture_options.layer_visualizers
    )
    if layer_visualizers_value:
        _render_layer_visualizers(trace, layer_visualizers_value)
    if capture_options.unwrap_when_done:
        from ...backends.torch.wrappers import unwrap_torch

        unwrap_torch()
