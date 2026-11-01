"""Torch door body for the PREPARE stage: admission, refusals and request resolution.

Moved verbatim from ``torchlens._user_funcs`` by the capture seam.
"""

import collections.abc
import functools
import os
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn

from torchlens import _state
from torchlens.backends.registry import BackendUnsupportedError

from ..._autoroute._builtin_output import semantic_output_cache_key
from ..._capture_state_helpers import (
    _capture_cache_key,
    _facet_recipe_cache_key,
    _prepare_log_for_capture_cache,
    _reject_opaque_wrappers,
    _unwrap_data_parallel,
    unwrap_compiled_model,
)
from ..._data_classes._preprocessing_provenance import stamp_user_transform_provenance
from ..._data_classes.trace import (
    Trace,
)
from ..._deprecations import MISSING, MissingType
from ..._errors import (
    ArgumentConflictError,
    ArgumentTypeError,
    CaptureNotActiveError,
    InvalidArgumentError,
    KeywordConflictError,
)
from ..._input_coerce import (
    _coerce_input_args,
    _reject_non_mapping_input_kwargs,
)
from ..._literals import (
    OutputDeviceLiteral,
)
from ..._robustness import check_model_and_input_variants
from ..._trace_selector_helpers import (
    _combine_save_predicates,
    _is_selective_label_save,
    _layers_to_save_has_negative_index,
    _layers_to_save_live_subset,
    _layers_to_save_mentions_identity,
    _layers_to_save_mentions_output,
    _layers_to_save_needs_final_resolution,
    _make_layers_to_save_predicate,
    _predicate_cache_key,
    _route_save_label_selection,
    _split_save_options_and_predicate,
    _stable_cache_fragment,
)
from ..._training_validation import (
    TrainingModeConfigError,
    effective_inference_only,
    validate_training_compatibility,
)
from ..._utils.display import _vprint, warn_parallel
from ..._utils.tensor_utils import SaveMode
from ...backends.torch._op_markers import refuse_unavailable_nvtx
from ...backends.torch.bound_root import (
    TLBoundMethodRoot,
    is_bound_method_of_module,
)
from ...bridge import transformers as _hf_bridge
from ...intervention.errors import ChunkedForwardConfigError
from ...intervention.predicates import InterventionPredicate
from ...intervention.selectors import BaseSelector
from ...options import (
    CaptureOptions,
    EpisodeOptions,
    ReplayOptions,
    SaveOptions,
    StorageOptions,
    merge_capture_options,
    merge_save_options,
    merge_streaming_options,
)
from ...recording.options import HaltPredicateFn, PredicateFn
from ...types import _ActivationPostfunc, _GradientPostfunc
from .._episode_failed import attach_failed_episode_ledger
from .._episode_ledger import (
    resolve_episode_declaration,
    write_episode_ledger,
)
from .._structure_only_entry import (
    _enforce_structure_only_entry_contract,
    _StructureOnlyEntryFacts,
)
from ..finalize.epilogue import _render_layer_visualizers
from ..run_entry import _run_model_and_save_specified_outs
from ..stages import CaptureRequest
from .cache import (
    CAPTURE_CACHE_KEY_CURATED,
    CAPTURE_CACHE_KEY_NEUTRAL,
    _evict_capture_cache,
    _load_authenticated_capture_cache,
    _prepare_capture_cache_dir,
    _store_authenticated_capture_cache,
    _sweep_option_fields_into_cache_config,
)
from .chunked import (
    _append_chunk_trace_state,
    _ChunkAppend,
    _should_store_auto_coerced_raw_input,
    _validate_chunked_forward_capture,
    iter_chunked_inputs,
    normalize_chunk_paths,
    normalize_chunk_size,
    plan_chunks,
)
from .dispatch import _reject_unsupported_torch_trace_option_values

_can_resolve_hf_processor = _hf_bridge._can_resolve_hf_processor
_can_resolve_hf_tokenizer = _hf_bridge._can_resolve_hf_tokenizer
_has_attached_image_processor = _hf_bridge._has_attached_image_processor
_is_hf_image_input = _hf_bridge._is_hf_image_input
_is_hf_multimodal_input = _hf_bridge._is_hf_multimodal_input
_is_hf_text_input = _hf_bridge._is_hf_text_input


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
    from ...errors import TorchLensWarning as _TorchLensWarning

    warnings.warn(
        _TorchLensWarning(
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

    from ...errors import TorchLensWarning as _TorchLensWarning

    warnings.warn(
        _TorchLensWarning(
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


def _refuse_non_torch_episode(public_trace_kwargs: dict[str, Any], resolved_spec: Any) -> None:
    """episode= capture (capture_kind=episode) is torch-only in this release."""

    if str(resolved_spec.name) != "torch":
        episode_value = public_trace_kwargs.pop("episode", None)
        if episode_value is not None:
            raise BackendUnsupportedError(
                "episode= capture (capture_kind=episode) is torch-only in this "
                f"release; backend {str(resolved_spec.name)!r} does not support "
                "episode declarations."
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
    layers_to_save: str | list[Any] | None | MissingType = MISSING,
    transform: Callable[[Any], Any] | None | MissingType = MISSING,
    save_raw_input: str | bool | MissingType = MISSING,
    batch_render: str | MissingType = MISSING,
    output_transform: Callable[[Any], Any] | None | MissingType = MISSING,
    save_transformed_output: str | bool | MissingType = MISSING,
    keep_orphans: bool | MissingType = MISSING,
    output_device: OutputDeviceLiteral | MissingType = MISSING,
    activation_transform: _ActivationPostfunc | None | MissingType = MISSING,
    grad_transform: _GradientPostfunc | None | MissingType = MISSING,
    save_raw_activations: bool | MissingType = MISSING,
    save_raw_grads: bool | MissingType = MISSING,
    save_mode: SaveMode | MissingType = MISSING,
    capture_tensor_grad_hooks: bool | MissingType = MISSING,
    mark_layer_depths: bool | MissingType = MISSING,
    detach_saved_activations: bool | MissingType = MISSING,
    save_args: bool | MissingType = MISSING,
    save_grads: bool | str | list[Any] | PredicateFn | BaseSelector | None | MissingType = MISSING,
    save_source_context: bool | MissingType = MISSING,
    save_rng: bool | MissingType = MISSING,
    reconstruction_ready: bool | MissingType = MISSING,
    seed: int | None | MissingType = MISSING,
    num_context_lines: int | MissingType = MISSING,
    optimizer: Any | MissingType = MISSING,
    save_outs_to: str | Path | None | MissingType = MISSING,
    keep_outs_in_memory: bool | MissingType = MISSING,
    out_sink: Callable[[str, torch.Tensor], None] | None | MissingType = MISSING,
    intervention_ready: bool | MissingType = MISSING,
    capture_container_structure: bool | MissingType = MISSING,
    hooks: Any | None | MissingType = MISSING,
    unwrap_when_done: bool | MissingType = MISSING,
    verbose: bool | MissingType = MISSING,
    source_context_lines: int | MissingType = MISSING,
    compute_input_output_distances: bool | MissingType = MISSING,
    recurrence_detection: bool | MissingType = MISSING,
    capture: CaptureOptions | None = None,
    save: SaveOptions
    | PredicateFn
    | BaseSelector
    | str
    | list[Any]
    | tuple[Any, ...]
    | None = None,
    intervene: InterventionPredicate | None = None,
    halt: HaltPredicateFn | None = None,
    lookback: int = 0,
    lookback_payload_policy: str = "metadata_only",
    storage: StorageOptions | None = None,
    streaming: StorageOptions | None = None,
    backward_ready: bool | MissingType = MISSING,
    inference_only: bool | MissingType = MISSING,
    name: str | None | MissingType = MISSING,
    cache: bool | MissingType = MISSING,
    cache_dir: str | Path | None | MissingType = MISSING,
    module_filter: Callable[[Any], bool] | None | MissingType = MISSING,
    stop_after: Any | None | MissingType = MISSING,
    raise_on_nan: bool | MissingType = MISSING,
    structure_only: bool | MissingType = MISSING,
    profile: bool | MissingType = MISSING,
    jax_control_flow: Literal["reject", "unroll", "region"] | MissingType = MISSING,
    jax_max_control_flow_unroll: int | MissingType = MISSING,
    module_identity_mode: str | None | MissingType = MISSING,
    payload_policy: str | None | MissingType = MISSING,
    save_preview: bool | MissingType = MISSING,
    facets: (
        list[Callable[[Any], dict[str, Any]]]
        | tuple[Callable[[Any], dict[str, Any]], ...]
        | None
        | MissingType
    ) = MISSING,
    episode: EpisodeOptions | None = None,
    capture_output_structure: bool | MissingType = MISSING,
    chunk_size: int | None | MissingType = MISSING,
    chunk_paths: Iterable[Any] | None | MissingType = MISSING,
    narrate: Any | None = None,
    retain_output_parents_for_layers_to_save: bool = False,
    _selective_layers_to_save_request: object | None = None,
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
    **capture_options:
        The remaining parameters match ``trace`` exactly, excluding ``backend``.

    Returns
    -------
    Trace
        Captured torch trace.
    """
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
    model = _unwrap_data_parallel(model)
    if reconstruction_ready is not MISSING and reconstruction_ready:
        save_args = True
        save_rng = True

    capture_options = merge_capture_options(
        capture=capture,
        layers_to_save=layers_to_save,
        input_transform=transform,
        save_raw_input=save_raw_input,
        batch_render=batch_render,
        output_transform=output_transform,
        save_transformed_output=save_transformed_output,
        layer_visualizers=MISSING,
        save_visualizations=MISSING,
        keep_orphans=keep_orphans,
        output_device=output_device,
        save_args=save_args,
        save_grads=save_grads,
        capture_tensor_grad_hooks=capture_tensor_grad_hooks,
        save_source_context=save_source_context,
        save_rng=save_rng,
        seed=seed,
        source_context_lines=source_context_lines,
        num_context_lines=num_context_lines,
        optimizer=optimizer,
        compute_input_output_distances=compute_input_output_distances,
        mark_layer_depths=mark_layer_depths,
        detach_saved_activations=detach_saved_activations,
        recurrence_detection=recurrence_detection,
        intervention_ready=intervention_ready,
        capture_container_structure=capture_container_structure,
        capture_output_structure=capture_output_structure,
        hooks=hooks,
        unwrap_when_done=unwrap_when_done,
        verbose=verbose,
        backward_ready=backward_ready,
        inference_only=inference_only,
        name=name,
        cache=cache,
        cache_dir=cache_dir,
        module_filter=module_filter,
        stop_after=stop_after,
        jax_control_flow=jax_control_flow,
        jax_max_control_flow_unroll=jax_max_control_flow_unroll,
        module_identity_mode=module_identity_mode,
        payload_policy=payload_policy,
        save_preview=save_preview,
        raise_on_nan=raise_on_nan,
        structure_only=structure_only,
    )
    if isinstance(capture_options.layers_to_save, torch.Tensor):
        raise ArgumentTypeError(
            "capture.layers_to_save received a torch.Tensor -- this is almost "
            "always a model input routed to the wrong slot. Bundle model inputs "
            "as one tuple, e.g. tl.trace(model, (input_a, input_b, input_c)), "
            "and use save= for selective capture.",
            code="layers_to_save_type_invalid",
            remedy="bundle positional inputs into one tuple; use save= for selection",
        )
    _reject_unsupported_torch_trace_option_values(capture_options)
    # Echo teaching refusals fire whether or not narrate= is armed: NarrationOptions
    # routed into save=/hooks= refuses with the measured receipts, and
    # echo x cache/chunked/structure_only composition refuses typed (snoop D1).
    from ..._snoop._entry import resolve_echo_capture_options

    echo_options = resolve_echo_capture_options(
        narrate,
        save=save,
        capture_options=capture_options,
        chunked=chunk_size is not MISSING and chunk_size is not None,
    )
    profile_enabled = False if isinstance(profile, MissingType) else bool(profile)
    raw_input = None
    input_transform = capture_options.input_transform
    if input_transform is not None:
        raw_input = input_args
        transformed_input = input_transform(input_args)
        if isinstance(transformed_input, collections.abc.Mapping):
            input_args = []
            input_kwargs = dict(transformed_input)
        else:
            input_args = transformed_input
            input_kwargs = None
    else:
        original_input_args = input_args
        input_args = _coerce_input_args(model, input_args)
        if _should_store_auto_coerced_raw_input(original_input_args, input_args):
            raw_input = original_input_args

    # W2 (weightsfree memo D2): admit_meta derives ONCE from the resolved
    # structure-only option state at this CAPTURE entry; the returned
    # admission record (None on ordinary/real captures) is armed around the
    # capture driver below and consumed by the forward scope + settlement.
    meta_admission = check_model_and_input_variants(
        model,
        input_args,
        input_kwargs,
        admit_meta=bool(capture_options.structure_only),
    )
    routed_save, capture_options = _route_save_label_selection(
        save, capture, capture_options, model
    )
    grouped_save_options, save_predicate = _split_save_options_and_predicate(routed_save)
    if intervene is not None and not callable(intervene):
        raise ArgumentTypeError(
            f"intervene received non-callable type {type(intervene).__name__}",
            code="intervention_predicate_type_invalid",
            remedy="pass a predicate callable such as tl.when(...) or None",
            argument="intervene",
            received_type=type(intervene).__name__,
        )
    if halt is not None and not callable(halt):
        raise ArgumentTypeError(
            f"halt received non-callable type {type(halt).__name__}",
            code="halt_predicate_type_invalid",
            remedy="pass a predicate callable or None",
            argument="halt",
            received_type=type(halt).__name__,
        )
    if not isinstance(lookback, int) or not 0 <= lookback <= 1024:
        raise InvalidArgumentError(
            f"lookback={lookback!r} is not an integer in the supported range [0, 1024]",
            code="lookback_invalid",
            remedy="set lookback to an integer from 0 through 1024 inclusive",
            argument="lookback",
        )
    if lookback_payload_policy not in {
        "metadata_only",
        "detached_raw",
        "transformed",
        "grad_connected",
        "disk_spilled",
    }:
        raise InvalidArgumentError(
            f"lookback_payload_policy={lookback_payload_policy!r} is not supported",
            code="lookback_payload_policy_invalid",
            remedy=(
                "set lookback_payload_policy to 'metadata_only', 'detached_raw', "
                "'transformed', 'grad_connected', or 'disk_spilled'"
            ),
            argument="lookback_payload_policy",
        )
    episode_resolved = None
    if episode is not None:
        from ...errors.episode import EpisodeDeclarationError

        if chunk_size not in (MISSING, None) or chunk_paths not in (MISSING, None):
            raise EpisodeDeclarationError(
                "episode= cannot combine with chunked forwards (chunk_size/"
                "chunk_paths): an episode is ONE wrapped session capture, and "
                "the chunk fan-out produces several.",
                code="episode_declaration_invalid",
            )
        if cache is not MISSING and cache:
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
            from ..structure_only import STRUCTURE_ONLY_EPISODE_UNSUPPORTED

            raise EpisodeDeclarationError(
                "structure-only episodes are out of scope this sprint: "
                "capture_kind=episode with the structure-only marker refuses "
                "typed per the ratified marker-combination table.",
                code=STRUCTURE_ONLY_EPISODE_UNSUPPORTED,
            )
        if getattr(episode, "step_output_kind", "tokens") != "none" and (
            capture_options.layers_to_save in (None, "none", "None", "NONE")
            or (
                isinstance(capture_options.layers_to_save, list)
                and not capture_options.layers_to_save
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
        episode_resolved = resolve_episode_declaration(episode, model)
        if intervene is not None:
            # ATTESTED COUPLING (lane F42, the foldA D5 flip): the session
            # armed here feeds the settlement writers (fire counts,
            # intervention digest, perturbed fidelity) -- the D5 evidence bar.
            from dataclasses import replace as _dataclass_replace

            from .._episode_coupling import CouplingSession, rule_identity_of

            episode_resolved = _dataclass_replace(
                episode_resolved,
                coupling_session=CouplingSession(rule_identity=rule_identity_of(intervene)),
            )
    save_options = merge_save_options(
        save=grouped_save_options,
        activation_transform=activation_transform,
        grad_transform=grad_transform,
        save_raw_activations=save_raw_activations,
        save_raw_grads=save_raw_grads,
    )
    if storage is not None and streaming is not None:
        raise KeywordConflictError(
            "Both storage and streaming options were supplied",
            code="storage_argument_conflict",
            remedy="remove streaming and pass only storage, or remove storage",
            arguments=("storage", "streaming"),
        )
    streaming_options = merge_streaming_options(
        streaming=storage if storage is not None else streaming,
    )
    chunk_size_value = None if isinstance(chunk_size, MissingType) else chunk_size
    chunk_paths_value = None if isinstance(chunk_paths, MissingType) else chunk_paths
    normalized_chunk_size = normalize_chunk_size(chunk_size_value)
    if chunk_paths_value is not None and normalized_chunk_size is None:
        raise ChunkedForwardConfigError("chunk_paths requires chunk_size.")
    layers_to_save = capture_options.layers_to_save
    save_raw_input_policy = capture_options.save_raw_input
    batch_render_policy = capture_options.batch_render
    output_transform_value = capture_options.output_transform
    save_raw_output_policy = capture_options.save_transformed_output
    layer_visualizers_value = cast(
        "dict[Any, Callable[..., Any]] | None", capture_options.layer_visualizers
    )
    save_visualizations_value = capture_options.save_visualizations
    keep_orphans = capture_options.keep_orphans
    output_device = capture_options.output_device
    activation_transform = save_options.activation_transform
    grad_transform = save_options.grad_transform
    save_raw_activations = save_options.save_raw_activations
    save_raw_grads = save_options.save_raw_gradients
    save_mode_value = "copy" if save_mode is MISSING else save_mode
    if save_mode_value not in {"copy", "reference", "view", "cpu_async"}:
        raise InvalidArgumentError(
            f"save_mode={save_mode_value!r} is not supported",
            code="save_mode_invalid",
            remedy="set save_mode to 'copy', 'reference', 'view', or 'cpu_async'",
            argument="save_mode",
        )
    refuse_unavailable_nvtx(capture_options.emit_nvtx)
    save_args = capture_options.save_args
    capture_tensor_grad_hooks = capture_options.capture_tensor_grad_hooks
    save_source_context = capture_options.save_source_context
    save_rng = capture_options.save_rng
    seed = capture_options.seed
    source_context_lines = capture_options.source_context_lines
    optimizer = capture_options.optimizer
    compute_input_output_distances = capture_options.compute_input_output_distances
    detach_saved_activations = capture_options.detach_saved_activations
    recurrence_detection = capture_options.recurrence_detection
    intervention_ready = capture_options.intervention_ready
    capture_container_structure = capture_options.capture_container_structure
    hooks = capture_options.hooks
    unwrap_when_done = capture_options.unwrap_when_done
    verbose = capture_options.verbose
    name = capture_options.name
    cache_enabled = capture_options.cache
    facet_recipes = None if isinstance(facets, MissingType) else facets
    if capture_options.stop_after is not None and halt is not None:
        raise ArgumentConflictError(
            "Both stop_after= and halt= were configured; they compile into the same "
            "stop-directive slot and cannot combine",
            code="stop_after_halt_conflict",
            remedy=(
                "pass one stop directive: keep halt= (compose predicates with & |) "
                "or keep stop_after="
            ),
            arguments=("stop_after", "halt"),
        )
    stop_after_site = capture_options.stop_after
    stop_after_ambient = False
    if stop_after_site is None and halt is None:
        from ...experimental import _active_stop_after_site

        stop_after_site = _active_stop_after_site()
        stop_after_ambient = stop_after_site is not None
    stop_after_selector_shaped = False
    if stop_after_site is not None:
        if normalized_chunk_size is not None:
            raise ChunkedForwardConfigError(
                "chunk_size cannot be combined with stop_after: the chunk fan-out runs "
                "several captures and a single stop frontier is ambiguous across them. "
                "Remedy: drop chunk_size or drop stop_after.",
                code="stop_after_chunked_conflict",
            )
        if isinstance(stop_after_site, BaseSelector):
            halt = stop_after_site
            stop_after_selector_shaped = True
        elif isinstance(stop_after_site, str):
            from ..._ir.selector_eval import looks_like_finalized_label

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
            from ...intervention.selectors import FuncSelector, ModuleSelector

            # A bare string is the pluck-vocabulary spelling: halt at the first
            # emission whose module address OR function name matches, whichever
            # fires first ("encoder.layer.4" never names a func; "relu" never
            # names a module address in practice, so collisions are benign).
            halt = ModuleSelector(stop_after_site) | FuncSelector(stop_after_site)
            stop_after_selector_shaped = True
        elif callable(stop_after_site):
            halt = stop_after_site
        else:
            raise ArgumentTypeError(
                f"stop_after received unsupported type {type(stop_after_site).__name__}",
                code="stop_after_type_invalid",
                remedy=("pass a module address string, a tl.* selector, or a callable predicate"),
                argument="stop_after",
                received_type=type(stop_after_site).__name__,
            )
    save_grads_policy = capture_options.save_grads
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
    grad_storage_path_value = streaming_options.path if should_save_grads else None
    retain_grads_in_memory_value = streaming_options.retain_in_memory

    from ..._options_validation import _validate_output_device

    _validate_output_device(output_device)
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
    if capture_options.structure_only:
        layers_to_save = _enforce_structure_only_entry_contract(
            capture_options,
            _StructureOnlyEntryFacts(
                layers_to_save=layers_to_save,
                save_predicate=save_predicate,
                halt=halt,
                streaming_options=streaming_options,
                lookback_payload_policy=lookback_payload_policy,
                raise_on_nan_value=capture_options.raise_on_nan,
                track_nonfinite_value=capture_options.track_nonfinite,
                intervention_ready=intervention_ready,
                should_save_grads=should_save_grads,
            ),
        )
        save_raw_input_policy = False
        save_raw_output_policy = False
    train_mode_explicit = capture_options.is_field_explicit("backward_ready")
    train_mode_value = capture_options.backward_ready
    inference_only_conflicts: list[str] = []
    if capture_options.is_field_explicit("backward_ready") and train_mode_value is True:
        inference_only_conflicts.append("backward_ready")
    if capture_options.is_field_explicit("save_grads") and should_save_grads:
        inference_only_conflicts.append("save_grads")
    if capture_options.is_field_explicit("intervention_ready") and intervention_ready is True:
        inference_only_conflicts.append("intervention_ready")
    backward_opted_in = (
        capture_options.is_field_explicit("save_grads")
        and should_save_grads
        and save_grads_policy is not True
    )
    grad_streaming_requested = grad_storage_path_value is not None
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
    if train_mode_value and grad_storage_path_value is not None:
        raise TrainingModeConfigError(
            "backward_ready=True is not compatible with disk-backed gradient storage. "
            "Remedy: drop the gradient storage path or drop backward_ready=True.",
            code="backward_ready_conflict",
        )

    validate_training_compatibility(
        backward_ready=train_mode_value,
        streaming=streaming_options,
        detach_saved_activations=detach_saved_activations,
        inference_mode_active=torch.is_inference_mode_enabled(),
        inference_only=capture_options.inference_only,
        inference_only_conflicts=tuple(inference_only_conflicts),
    )
    inference_only_value = effective_inference_only(
        capture_options,
        backward_requested=bool(train_mode_value or should_save_grads or intervention_ready),
    )
    chunk_plan = None
    if normalized_chunk_size is not None:
        _validate_chunked_forward_capture(
            input_kwargs=input_kwargs,
            backward_ready=train_mode_value,
            save_grads=should_save_grads,
            hooks=hooks,
            intervene=intervene,
            halt=halt,
            streaming=streaming_options,
        )
        chunk_plan = plan_chunks(
            input_args,
            chunk_size=normalized_chunk_size,
            chunk_paths=chunk_paths_value,
        )

    if type(layers_to_save) is str:
        layers_to_save = layers_to_save.lower()
    if type(grads_to_save_resolved) is str:
        grads_to_save_resolved = grads_to_save_resolved.lower()
    requested_layers_to_save = layers_to_save
    uses_deferred_gradients = grads_to_save_resolved not in ["all", "none", None, []]
    uses_deferred_activation = _is_selective_label_save(layers_to_save) and (
        _layers_to_save_mentions_output(layers_to_save)
        or _layers_to_save_has_negative_index(layers_to_save)
        or _layers_to_save_mentions_identity(layers_to_save)
        # Integer ordinals and final-label-shaped strings are defined against
        # FINAL layer numbering, which orphan removal renumbers after capture;
        # they must resolve post-postprocess, never against raw indexes.
        or _layers_to_save_needs_final_resolution(layers_to_save)
    )
    live_layers_to_save = (
        _layers_to_save_live_subset(layers_to_save) if uses_deferred_activation else layers_to_save
    )
    needs_live_output_projection = (
        _is_selective_label_save(layers_to_save)
        and should_save_grads
        and not uses_deferred_activation
    )
    uses_selective_layers_to_save = _is_selective_label_save(layers_to_save)
    if uses_selective_layers_to_save and not uses_deferred_activation:
        layers_predicate = _make_layers_to_save_predicate(layers_to_save)
        save_predicate = _combine_save_predicates(save_predicate, layers_predicate)
        layers_to_save = "all"
    elif uses_deferred_activation and live_layers_to_save is not None:
        layers_predicate = _make_layers_to_save_predicate(live_layers_to_save)
        save_predicate = _combine_save_predicates(save_predicate, layers_predicate)
    if save_predicate is not None or intervene is not None or halt is not None:
        from ..predicates import validate_followed_by_capability

    if save_predicate is not None:
        validate_followed_by_capability(
            save_predicate,
            api_name="trace(save=...)",
            supports_retroactive=True,
        )
    if intervene is not None:
        validate_followed_by_capability(
            intervene,
            api_name="trace(intervene=...)",
            supports_retroactive=False,
        )
    if halt is not None:
        validate_followed_by_capability(
            halt,
            api_name="trace(halt=...)",
            supports_retroactive=False,
        )
    log_name = name if name is not None else _state._auto_name(model)
    cache_path: Path | None = None
    cache_key: str | None = None
    cache_secret: bytes | None = None
    if cache_enabled:
        cache_config = {
            "layers_to_save": requested_layers_to_save,
            "keep_orphans": keep_orphans,
            "output_device": output_device,
            "save_args": save_args,
            "capture_tensor_grad_hooks": capture_tensor_grad_hooks,
            "save_grads": repr(save_grads_policy),
            "save_source_context": save_source_context,
            "save_rng": save_rng,
            "source_context_lines": source_context_lines,
            "compute_input_output_distances": compute_input_output_distances,
            "detach_saved_activations": detach_saved_activations,
            "save_mode": save_mode_value,
            "recurrence_detection": recurrence_detection,
            "backward_ready": train_mode_value,
            "inference_only": inference_only_value,
            "chunk_size": normalized_chunk_size,
            "chunk_paths": normalize_chunk_paths(chunk_paths_value),
            "capture_container_structure": capture_container_structure,
            "output_transform": _stable_cache_fragment(output_transform_value),
            "semantic_output_cache_key": semantic_output_cache_key(model),
            "facets": _facet_recipe_cache_key(facet_recipes),
            "save_predicate": _predicate_cache_key(save_predicate),
            "intervene": _predicate_cache_key(intervene),
            "halt": _predicate_cache_key(halt),
            # stop_after compiles INTO the halt slot, but its never-fired policy
            # differs by provenance, so the two spellings must not share a key:
            # a completed halt= capture could otherwise satisfy a stop_after=
            # request whose cold run would have refused stop_after_never_fired.
            "stop_after": _stable_cache_fragment(stop_after_site),
            # Capability / payload-policy options that change WHAT is captured or
            # stored in the returned trace. Omitting any of these let a second
            # trace() with a different capability silently return an earlier cached
            # trace built with the wrong capability (e.g. asked intervention_ready=True,
            # got a cached False). Include them so a capability change misses the cache;
            # callables are repr-keyed (conservative -- a distinct object misses rather
            # than risking a false hit), matching how output_transform is keyed above.
            "intervention_ready": intervention_ready,
            "structure_only": capture_options.structure_only,
            "track_device_memory": capture_options.track_device_memory,
            # A cached capture without the opt-in holds no CPU peak.
            "measure_host_peak_memory": capture_options.measure_host_peak_memory,
            "log_injections": capture_options.log_injections,
            "save_raw_input": repr(save_raw_input_policy),
            "save_transformed_output": repr(save_raw_output_policy),
            "save_raw_activations": save_raw_activations,
            "save_raw_gradients": save_raw_grads,
            "activation_transform": _stable_cache_fragment(activation_transform),
            "grad_transform": _stable_cache_fragment(grad_transform),
            "seed": seed,
            "module_filter": _stable_cache_fragment(capture_options.module_filter),
            "layer_visualizers": _stable_cache_fragment(layer_visualizers_value),
            "save_visualizations": _stable_cache_fragment(save_visualizations_value),
            "optimizer": repr(optimizer),
            "hooks": _stable_cache_fragment(hooks),
            "lookback": lookback,
            "lookback_payload_policy": lookback_payload_policy,
            "jax_control_flow": capture_options.jax_control_flow,
            "jax_max_control_flow_unroll": capture_options.jax_max_control_flow_unroll,
            "module_identity_mode": capture_options.module_identity_mode,
            "payload_policy": capture_options.payload_policy,
            "save_preview": capture_options.save_preview,
            "profile": profile_enabled,
        }
        # The inversion sweep: every CaptureOptions field not hand-curated
        # above and not declared session-neutral enters the key here --
        # raise_on_nan, track_nonfinite, save_budget, distributed_witness,
        # measure_python_peak_memory, emit_nvtx, transform, name, and any
        # future field. Streaming options change what the returned trace
        # retains (and where), so they sweep in full as well.
        _sweep_option_fields_into_cache_config(
            cache_config,
            capture_options,
            prefix="capture_option",
            curated=CAPTURE_CACHE_KEY_CURATED,
            neutral=CAPTURE_CACHE_KEY_NEUTRAL,
        )
        _sweep_option_fields_into_cache_config(
            cache_config,
            streaming_options,
            prefix="streaming_option",
        )
        cache_key = _capture_cache_key(model, input_args, input_kwargs, cache_config)
        cache_root, cache_secret = _prepare_capture_cache_dir(capture_options.cache_dir)
        cache_path = cache_root / f"{cache_key}.pkl"
        if cache_path.exists():
            cached_log = cast(
                "Trace | None", _load_authenticated_capture_cache(cache_path, cache_secret)
            )
            if cached_log is not None:
                try:
                    os.utime(cache_path, None)
                except OSError:
                    pass
                cached_log.capture_cache_hit = True
                cached_log.capture_cache_key = cache_key
                cached_log.capture_cache_path = str(cache_path)
                cached_log.batch_render = batch_render_policy
                return cached_log
    # list-A row 30: tracing runs the model's REAL forward, so train-mode norm
    # layers with running statistics advance them in place (momentum applies
    # even inside torch.no_grad / inference_only). Disclose once per process at
    # the point the mutation is about to happen. Cache hits return above (no
    # forward, no mutation) and structure-only captures retain no values, so
    # neither path reaches this warn. The chunked fan-out recurses back into
    # this function, but the outer call warns first and latches the flag.
    if (
        not _BATCHNORM_TRAIN_STATS_WARNED
        and not capture_options.structure_only
        and capture_options.carry_state
    ):
        _warn_once_train_mode_running_stats(model)
    if (
        chunk_plan is not None
        and normalized_chunk_size is not None
        and normalized_chunk_size < chunk_plan.total_size
    ):
        chunks = iter_chunked_inputs(input_args, chunk_plan)
        recursive_jax_control_flow = (
            capture_options.jax_control_flow
            if capture_options.is_field_explicit("jax_control_flow")
            else MISSING
        )
        recursive_jax_max_control_flow_unroll = (
            capture_options.jax_max_control_flow_unroll
            if capture_options.is_field_explicit("jax_max_control_flow_unroll")
            else MISSING
        )
        recursive_capture_options = CaptureOptions(
            layers_to_save=layers_to_save,
            input_transform=None,
            save_raw_input=save_raw_input_policy,
            batch_render=batch_render_policy,
            output_transform=output_transform_value,
            save_transformed_output=save_raw_output_policy,
            layer_visualizers=layer_visualizers_value,
            save_visualizations=save_visualizations_value,
            keep_orphans=keep_orphans,
            output_device=output_device,
            save_args=save_args,
            save_grads=save_grads_policy,
            capture_tensor_grad_hooks=capture_tensor_grad_hooks,
            save_source_context=save_source_context,
            save_rng=save_rng,
            seed=seed,
            source_context_lines=source_context_lines,
            optimizer=optimizer,
            compute_input_output_distances=compute_input_output_distances,
            detach_saved_activations=detach_saved_activations,
            recurrence_detection=recurrence_detection,
            intervention_ready=intervention_ready,
            capture_container_structure=capture_container_structure,
            hooks=MISSING,
            unwrap_when_done=False,
            verbose=verbose,
            backward_ready=train_mode_value,
            inference_only=inference_only_value,
            name=log_name,
            cache=False,
            cache_dir=capture_options.cache_dir,
            module_filter=capture_options.module_filter,
            stop_after=MISSING,
            jax_control_flow=recursive_jax_control_flow,
            jax_max_control_flow_unroll=recursive_jax_max_control_flow_unroll,
            module_identity_mode=capture_options.module_identity_mode,
            payload_policy=capture_options.payload_policy,
            save_preview=capture_options.save_preview,
            # Session knobs the chunk path historically DROPPED (list-A row 26,
            # second clause): the recursive options object was rebuilt from a
            # hand-enumerated field list, so a chunked capture silently reset
            # save_budget (and the other session-time knobs) to defaults. The
            # recursive constructor must cover EVERY CaptureOptions field;
            # tests/test_capopts_truth_cache_key.py pins the full-coverage
            # set difference so a future field cannot silently drop here.
            emit_nvtx=capture_options.emit_nvtx,
            measure_python_peak_memory=capture_options.measure_python_peak_memory,
            distributed_witness=capture_options.distributed_witness,
            save_budget=capture_options.save_budget,
            raise_on_nan=capture_options.raise_on_nan,
            track_nonfinite=capture_options.track_nonfinite,
            track_device_memory=capture_options.track_device_memory,
            structure_only=capture_options.structure_only,
            log_injections=capture_options.log_injections,
            carry_state=capture_options.carry_state,
            measure_host_peak_memory=capture_options.measure_host_peak_memory,
        )
        recursive_save_options = SaveOptions(
            activation_transform=activation_transform,
            grad_transform=grad_transform,
            save_raw_activations=save_raw_activations,
            save_raw_grads=save_raw_grads,
        )
        recursive_save_value: SaveOptions | PredicateFn | BaseSelector | None = (
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
        trace = _trace_torch_model(
            model=model,
            input_args=cast(torch.Tensor | list[Any] | tuple[Any, ...], chunks[0]),
            input_kwargs=None,
            layers_to_save=MISSING,
            transform=MISSING,
            save_raw_input=MISSING,
            batch_render=MISSING,
            output_transform=MISSING,
            save_transformed_output=MISSING,
            keep_orphans=MISSING,
            output_device=MISSING,
            activation_transform=recursive_activation_transform,
            grad_transform=recursive_grad_transform,
            save_raw_activations=recursive_save_raw_activations,
            save_raw_grads=recursive_save_raw_gradients,
            save_mode=cast(SaveMode, save_mode_value),
            capture_tensor_grad_hooks=MISSING,
            mark_layer_depths=MISSING,
            detach_saved_activations=MISSING,
            save_args=MISSING,
            save_grads=MISSING,
            save_source_context=MISSING,
            save_rng=MISSING,
            reconstruction_ready=MISSING,
            seed=MISSING,
            num_context_lines=MISSING,
            optimizer=MISSING,
            save_outs_to=MISSING,
            keep_outs_in_memory=MISSING,
            out_sink=MISSING,
            intervention_ready=MISSING,
            capture_container_structure=MISSING,
            hooks=MISSING,
            unwrap_when_done=MISSING,
            verbose=MISSING,
            source_context_lines=MISSING,
            compute_input_output_distances=MISSING,
            recurrence_detection=MISSING,
            capture=recursive_capture_options,
            save=recursive_save_value,
            intervene=None,
            halt=halt,
            lookback=lookback,
            lookback_payload_policy=lookback_payload_policy,
            storage=None,
            streaming=None,
            backward_ready=MISSING,
            inference_only=MISSING,
            name=MISSING,
            cache=MISSING,
            cache_dir=MISSING,
            module_filter=MISSING,
            stop_after=MISSING,
            raise_on_nan=MISSING,
            profile=profile_enabled,
            jax_control_flow=MISSING,
            jax_max_control_flow_unroll=MISSING,
            module_identity_mode=MISSING,
            payload_policy=MISSING,
            save_preview=MISSING,
            facets=facet_recipes,
            capture_output_structure=MISSING,
            chunk_size=None,
            chunk_paths=None,
            retain_output_parents_for_layers_to_save=uses_selective_layers_to_save,
            _selective_layers_to_save_request=(
                requested_layers_to_save if uses_selective_layers_to_save else None
            ),
        )
        initial_chunk_size = min(normalized_chunk_size, chunk_plan.total_size)
        initial_record = {
            "engine": "trace",
            "append": False,
            "chunk_size": initial_chunk_size,
            "total_batch_size": chunk_plan.total_size,
            "append_sequence_id": 0,
            "chunk_paths": normalize_chunk_paths(chunk_paths_value),
        }
        for chunk_index, chunk in enumerate(chunks[1:], start=1):
            if save_predicate is None:
                trace.run(model, chunk, replay=ReplayOptions(append=True), input_transform=False)
                continue
            new_trace = _trace_torch_model(
                model=model,
                input_args=cast(torch.Tensor | list[Any] | tuple[Any, ...], chunk),
                input_kwargs=None,
                layers_to_save=MISSING,
                transform=MISSING,
                save_raw_input=MISSING,
                batch_render=MISSING,
                output_transform=MISSING,
                save_transformed_output=MISSING,
                keep_orphans=MISSING,
                output_device=MISSING,
                activation_transform=recursive_activation_transform,
                grad_transform=recursive_grad_transform,
                save_raw_activations=recursive_save_raw_activations,
                save_raw_grads=recursive_save_raw_gradients,
                save_mode=cast(SaveMode, save_mode_value),
                capture_tensor_grad_hooks=MISSING,
                mark_layer_depths=MISSING,
                detach_saved_activations=MISSING,
                save_args=MISSING,
                save_grads=MISSING,
                save_source_context=MISSING,
                save_rng=MISSING,
                reconstruction_ready=MISSING,
                seed=MISSING,
                num_context_lines=MISSING,
                optimizer=MISSING,
                save_outs_to=MISSING,
                keep_outs_in_memory=MISSING,
                out_sink=MISSING,
                intervention_ready=MISSING,
                capture_container_structure=MISSING,
                hooks=MISSING,
                unwrap_when_done=MISSING,
                verbose=MISSING,
                source_context_lines=MISSING,
                compute_input_output_distances=MISSING,
                recurrence_detection=MISSING,
                capture=recursive_capture_options,
                save=recursive_save_value,
                intervene=None,
                halt=halt,
                lookback=lookback,
                lookback_payload_policy=lookback_payload_policy,
                storage=None,
                streaming=None,
                backward_ready=MISSING,
                inference_only=MISSING,
                name=MISSING,
                cache=MISSING,
                cache_dir=MISSING,
                module_filter=MISSING,
                stop_after=MISSING,
                raise_on_nan=MISSING,
                profile=profile_enabled,
                jax_control_flow=MISSING,
                jax_max_control_flow_unroll=MISSING,
                module_identity_mode=MISSING,
                payload_policy=MISSING,
                save_preview=MISSING,
                facets=facet_recipes,
                capture_output_structure=MISSING,
                chunk_size=None,
                chunk_paths=None,
                retain_output_parents_for_layers_to_save=uses_selective_layers_to_save,
                _selective_layers_to_save_request=(
                    requested_layers_to_save if uses_selective_layers_to_save else None
                ),
            )
            appended_chunk_size = min(
                normalized_chunk_size,
                chunk_plan.total_size - (chunk_index * normalized_chunk_size),
            )
            _append_chunk_trace_state(
                trace,
                new_trace,
                _ChunkAppend(
                    chunk_size=appended_chunk_size,
                    total_batch_size=chunk_plan.total_size,
                    append_sequence_id=chunk_index,
                    chunk_paths=normalize_chunk_paths(chunk_paths_value),
                ),
            )
        trace.append_history = [initial_record, *trace.append_history]
        trace.chunked_forward = True
        trace.last_run = dict(trace.last_run or {})
        trace.last_run["chunk_size"] = normalized_chunk_size
        trace.last_run["chunk_paths"] = normalize_chunk_paths(chunk_paths_value)
        trace.profile_enabled = profile_enabled
        if uses_selective_layers_to_save:
            trace._layer_nums_to_save = [
                op.raw_index
                for op in trace.layer_list
                if op.has_saved_activation and op.layer_type not in {"input", "output"}
            ]
            if not save_args:
                trace._replay_arg_version_data_complete = False
        if layer_visualizers_value:
            _render_layer_visualizers(trace, layer_visualizers_value)
        if unwrap_when_done:
            from ...backends.torch.wrappers import unwrap_torch

            unwrap_torch()
        if cache_path is not None and cache_key is not None and cache_secret is not None:
            trace.capture_cache_hit = False
            trace.capture_cache_key = cache_key
            trace.capture_cache_path = str(cache_path)
            _prepare_log_for_capture_cache(trace)
            if _store_authenticated_capture_cache(trace, cache_path, cache_secret):
                _evict_capture_cache(cache_path.parent, keep=cache_path)
        return trace

    run_capture = functools.partial(
        _run_model_and_save_specified_outs,
        model=model,
        input_args=cast(torch.Tensor | list[Any] | tuple[Any, ...], input_args),
        input_kwargs=input_kwargs,
        layers_to_save=layers_to_save,
        keep_orphans=keep_orphans,
        output_device=output_device,
        activation_transform=activation_transform,
        grad_transform=grad_transform,
        save_raw_activations=save_raw_activations,
        save_raw_grads=save_raw_grads,
        save_mode=cast(SaveMode, save_mode_value),
        capture_tensor_grad_hooks=capture_tensor_grad_hooks,
        mark_layer_depths=compute_input_output_distances,
        detach_saved_activations=detach_saved_activations,
        save_args=save_args,
        save_grads=should_save_grads,
        grads_to_save=grads_to_save_resolved,
        seed=seed,
        num_context_lines=source_context_lines,
        optimizer=optimizer,
        echo_options=echo_options,
        save_source_context=save_source_context,
        save_rng=save_rng,
        recurrence_detection=recurrence_detection,
        save_outs_to=streaming_options.path,
        keep_outs_in_memory=streaming_options.retain_in_memory,
        stream_custom_attributes=streaming_options.include_custom_attributes,
        stream_buffer_values=streaming_options.include_buffer_values,
        stream_async_writes=streaming_options.async_writes,
        stream_max_pending_bytes=streaming_options.max_pending_bytes,
        grad_storage_path=grad_storage_path_value,
        retain_grads_in_memory=retain_grads_in_memory_value,
        out_sink=streaming_options.on_save,
        intervention_ready=intervention_ready,
        capture_container_structure=capture_container_structure,
        hooks=hooks,
        intervention_spec=None,
        normalized_hook_plan=None,
        verbose=verbose,
        backward_ready=train_mode_value,
        inference_only=inference_only_value,
        inference_only_implied=(
            inference_only_value and not capture_options.is_field_explicit("inference_only")
        ),
        name=log_name,
        module_filter=capture_options.module_filter,
        emit_nvtx=capture_options.emit_nvtx,
        measure_python_peak_memory=capture_options.measure_python_peak_memory,
        distributed_witness=capture_options.distributed_witness,
        save_budget=capture_options.save_budget,
        raise_on_nan=capture_options.raise_on_nan,
        track_nonfinite=capture_options.track_nonfinite,
        track_device_memory=capture_options.track_device_memory,
        structure_only=capture_options.structure_only,
        log_injections=capture_options.log_injections,
        measure_host_peak_memory=capture_options.measure_host_peak_memory,
        transform=input_transform,
        raw_input=raw_input,
        save_raw_input=save_raw_input_policy,
        batch_render=batch_render_policy,
        output_transform=output_transform_value,
        save_transformed_output=save_raw_output_policy,
        layer_visualizers=layer_visualizers_value,
        save_visualizations=save_visualizations_value,
        recipes=facet_recipes,
        save_predicate=save_predicate,
        intervene_predicate=intervene,
        halt_predicate=halt,
        _halt_from_stop_after=stop_after_site is not None,
        lookback=lookback,
        lookback_payload_policy=lookback_payload_policy,
        retain_output_parents_for_layers_to_save=(
            retain_output_parents_for_layers_to_save
            or uses_selective_layers_to_save
            or episode_resolved is not None
        ),
        episode_resolved=episode_resolved,
        _selective_layers_to_save_request=(
            _selective_layers_to_save_request
            if _selective_layers_to_save_request is not None
            else (requested_layers_to_save if uses_selective_layers_to_save else None)
        ),
        _deferred_retention_selector=(
            requested_layers_to_save
            if uses_deferred_activation
            else (["output"] if needs_live_output_projection else None)
        ),
        _deferred_gradient_selector=(
            grads_to_save_resolved
            if uses_deferred_gradients
            else ("all" if uses_deferred_activation and train_mode_value else None)
        ),
    )
    from ...backends.torch.rescue import capture_with_rescue

    # Streaming saves, sinks, and halt-predicate partials are not re-runnable;
    # they report an escape as before instead of attempting a rescue re-run.
    # EVERY user-callable channel the re-run would invoke a SECOND time is
    # refused (fail closed): side effects (counters, file writes,
    # externally-held state) would double-apply with only a session-time
    # disclosure. That covers intervention transforms, pre-attached hooks,
    # AND the in-capture transform callables -- ``activation_transform``,
    # ``grad_transform``, and ``output_transform`` run inside ``run_capture``
    # and measurably fired twice on a recovered rescue (b6-opus-R16-1 reopen).
    # Channels applied OUTSIDE the rescue boundary stay eligible: ``save=``
    # selector predicates (pure by contract), the input ``transform=``
    # (applied before the re-run boundary; fires once), and
    # ``layer_visualizers`` (rendered once on the returned trace, after the
    # driver).
    rescue_eligible = (
        streaming_options.path is None
        and streaming_options.on_save is None
        and grad_storage_path_value is None
        and halt is None
        and intervene is None
        and not hooks
        and activation_transform is None
        and grad_transform is None
        and output_transform_value is None
    )
    from ..._runnable_state_live import capture_observer_bracket
    from .._episode_join import armed_capture
    from .._weightsfree_admission import pending_admission

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
    # so the public release door is legal there; exc.partial_log keeps its
    # already-materialized records and stays recoverable after the release.
    with capture_observer_bracket(
        model,
        carry_state=capture_options.carry_state,
        on_failure=lambda: _release_preparation_after_failed_capture(model),
    ) as observer:
        try:
            with (
                armed_capture(episode_resolved, run_capture) as capture_callable,
                pending_admission(meta_admission),
            ):
                trace = capture_with_rescue(capture_callable, eligible=rescue_eligible, model=model)
                observer.trace = trace
        except BaseException as capture_exc:
            if episode_resolved is not None:
                # Best-effort: the FAILED partial product carries the episode
                # declaration; attach the derived ledger disclosure without ever
                # masking the user's exception.
                attach_failed_episode_ledger(capture_exc, episode_resolved)
            raise
    if stop_after_site is not None and not bool(getattr(trace, "halted", False)):
        # Never-fired policy split by provenance (brainpipe D-16): a
        # selector-shaped site that never fired is a typo-shaped wrong result
        # (the full forward ran; the caller asked for a frontier) and refuses
        # typed; an exploratory callable or ambient context-manager site
        # legitimately may never fire and warns with a durable ledger entry.
        site_repr = repr(stop_after_site)
        if stop_after_selector_shaped and not stop_after_ambient:
            raise InvalidArgumentError(
                f"stop_after={site_repr} never fired: the capture ran the full "
                "forward and completed without halting, so the result is not the "
                "requested frontier",
                code="stop_after_never_fired",
                remedy=(
                    "check the site against the model's module addresses "
                    "(model.named_modules()) or use tl.func(...)/tl.module(...); "
                    "an exploratory predicate that may legitimately never fire "
                    "belongs in a callable"
                ),
                argument="stop_after",
            )
        stop_annotations = getattr(trace, "annotations", None)
        if isinstance(stop_annotations, dict):
            stop_annotations.setdefault("unmatched_capture_selectors", []).append(
                {"slot": "stop_after", "selector": site_repr}
            )
        from ...errors import TorchLensWarning

        warnings.warn(
            TorchLensWarning(
                f"stop_after={site_repr} never fired; the capture ran the full "
                "forward and the outputs are the model's real outputs, not a "
                "frontier. Remedy: check the stop_after site against the executed "
                "model, or drop stop_after",
                code="stop_after_never_fired_callable",
            ),
            stacklevel=2,
        )
    module_filter_suppressed = int(trace.__dict__.pop("_tl_module_filter_suppressed", 0))
    if (
        capture_options.module_filter is not None
        and module_filter_suppressed > 0
        and int(getattr(trace, "num_saved_ops", 0) or 0) == 0
    ):
        from ...errors import TorchLensWarning as _TorchLensWarning

        warnings.warn(
            _TorchLensWarning(
                f"module_filter suppressed every selected payload "
                f"({module_filter_suppressed} ops): the returned trace has full "
                "metadata but ZERO saved activations. module_filter is a third "
                "save gate composed with save=/layers_to_save, and its argument "
                "is an op-record namespace (fields such as func_name, "
                "layer_label, modules), NEVER an nn.Module. "
                "Remedy: write the filter against op-record fields (e.g. lambda "
                "op: op.func_name == 'linear'), or drop module_filter",
                code="module_filter_zero_saved",
            ),
            stacklevel=2,
        )
    if input_transform is not None:
        stamp_user_transform_provenance(trace, input_transform)
    trace.profile_enabled = profile_enabled
    trace.save_grads = save_grads_policy
    if uses_selective_layers_to_save:
        trace._layer_nums_to_save = [
            op.raw_index
            for op in trace.layer_list
            if op.has_saved_activation and op.layer_type not in {"input", "output"}
        ]
        if not save_args:
            trace._replay_arg_version_data_complete = False

    # Print final summary.
    _vprint(
        trace,
        f"Done: {len(trace.layer_logs)} layers, "
        f"{trace.num_saved_ops} saved, "
        f"{trace.total_activation_memory}",
    )

    if layer_visualizers_value:
        _render_layer_visualizers(trace, layer_visualizers_value)

    if unwrap_when_done:
        from ...backends.torch.wrappers import unwrap_torch

        unwrap_torch()

    if cache_path is not None and cache_key is not None and cache_secret is not None:
        trace.capture_cache_hit = False
        trace.capture_cache_key = cache_key
        trace.capture_cache_path = str(cache_path)
        _prepare_log_for_capture_cache(trace)
        if _store_authenticated_capture_cache(trace, cache_path, cache_secret):
            _evict_capture_cache(cache_path.parent, keep=cache_path)

    if episode_resolved is not None:
        # Settlement-time episode ledger (S7): written ONCE, after postprocess
        # and outcome settlement, from the settled record + module-call truth.
        write_episode_ledger(trace, episode_resolved)

    # Weightsfree settlement (F33): wrap-generation stamp on every capture
    # (W1-ORD preflight input), D22 invariant net + persisted evidence
    # envelope on structure-only captures.
    from .._structure_evidence import settle_weightsfree_capture

    settle_weightsfree_capture(trace, meta_admission)

    return trace


def prepare_capture(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[Any, Any] | None = None,
    **flat: Any,
) -> CaptureRequest:
    """Pack one torch capture call into the request the RUN stage consumes.

    Seam entry: ``flat`` holds the keyword arguments of
    ``_run_model_and_save_specified_outs`` unchanged, so
    ``run_capture(prepare_capture(model, args, kwargs, **flat))`` runs exactly
    ``_run_model_and_save_specified_outs(model, args, kwargs, **flat)``. The
    PREPARE lane moves request resolution behind this name.

    Parameters
    ----------
    model:
        The user's model.
    input_args:
        Positional forward inputs.
    input_kwargs:
        Keyword forward inputs.
    **flat:
        Remaining run keyword arguments, carried in ``CaptureRequest.plan``.

    Returns
    -------
    CaptureRequest
        The packed request.
    """
    return CaptureRequest(
        model=model,
        input_args=input_args,
        input_kwargs=input_kwargs,
        plan=dict(flat),
    )
