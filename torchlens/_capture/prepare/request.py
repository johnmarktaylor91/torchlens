"""The torch door's request half: its arguments, resolution state and the packed request.

Leaf module of the PREPARE stage: at runtime it imports only the ``MISSING``
sentinel and ``CaptureRequest``, so it never joins the package import cycle.
``_DoorArgs`` is the door's keyword surface, ``_DoorState`` the state the door's
stage functions fill, and ``_build_request`` packs that state into the
``CaptureRequest`` the RUN stage consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from ..._deprecations import MISSING, MissingType
from ..stages import CaptureRequest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    import torch
    from torch import nn

    from ..._literals import OutputDeviceLiteral
    from ..._selection.selectors import BaseSelector
    from ..._utils.tensor_utils import SaveMode
    from ...intervention.predicates import InterventionPredicate
    from ...options import CaptureOptions, EpisodeOptions, StorageOptions, _SaveTransforms
    from ...recording.options import HaltPredicateFn, PredicateFn
    from ...types import _ActivationPostfunc, _GradientPostfunc


@dataclass(frozen=True, slots=True)
class _DoorArgs:
    """The torch door's keyword arguments as received (``MISSING`` means not passed).

    One field per keyword of ``_trace_torch_model`` after ``model``, ``input_args``
    and ``input_kwargs``, with the same defaults, so ``_DoorArgs(**options)`` refuses
    an unknown keyword exactly as the explicit signature did.
    """

    layers_to_save: str | list[Any] | None | MissingType = MISSING
    transform: Callable[[Any], Any] | None | MissingType = MISSING
    save_raw_input: str | bool | MissingType = MISSING
    batch_render: str | MissingType = MISSING
    output_transform: Callable[[Any], Any] | None | MissingType = MISSING
    save_transformed_output: str | bool | MissingType = MISSING
    keep_orphans: bool | MissingType = MISSING
    output_device: OutputDeviceLiteral | MissingType = MISSING
    activation_transform: _ActivationPostfunc | None | MissingType = MISSING
    grad_transform: _GradientPostfunc | None | MissingType = MISSING
    save_raw_activations: bool | MissingType = MISSING
    save_raw_grads: bool | MissingType = MISSING
    save_mode: SaveMode | MissingType = MISSING
    capture_tensor_grad_hooks: bool | MissingType = MISSING
    mark_layer_depths: bool | MissingType = MISSING
    detach_saved_activations: bool | MissingType = MISSING
    save_args: bool | MissingType = MISSING
    save_grads: bool | str | list[Any] | PredicateFn | BaseSelector | None | MissingType = MISSING
    save_source_context: bool | MissingType = MISSING
    save_rng: bool | MissingType = MISSING
    reconstruction_ready: bool | MissingType = MISSING
    seed: int | None | MissingType = MISSING
    num_context_lines: int | MissingType = MISSING
    optimizer: Any | MissingType = MISSING
    save_outs_to: str | Path | None | MissingType = MISSING
    keep_outs_in_memory: bool | MissingType = MISSING
    out_sink: Callable[[str, torch.Tensor], None] | None | MissingType = MISSING
    replayable: bool | MissingType = MISSING
    capture_container_structure: bool | MissingType = MISSING
    hooks: Any | None | MissingType = MISSING
    unwrap_when_done: bool | MissingType = MISSING
    verbose: bool | MissingType = MISSING
    source_context_lines: int | MissingType = MISSING
    compute_input_output_distances: bool | MissingType = MISSING
    recurrence_detection: bool | MissingType = MISSING
    capture: CaptureOptions | None = None
    save: (
        _SaveTransforms | PredicateFn | BaseSelector | str | list[Any] | tuple[Any, ...] | None
    ) = None
    intervene: InterventionPredicate | None = None
    halt: HaltPredicateFn | None = None
    lookback: int = 0
    lookback_payload_policy: str = "metadata_only"
    storage: StorageOptions | None = None
    backward_ready: bool | MissingType = MISSING
    inference_only: bool | MissingType = MISSING
    name: str | None | MissingType = MISSING
    cache: bool | MissingType = MISSING
    cache_dir: str | Path | None | MissingType = MISSING
    module_filter: Callable[[Any], bool] | None | MissingType = MISSING
    stop_after: Any | None | MissingType = MISSING
    structure_only: bool | MissingType = MISSING
    jax_control_flow: Literal["reject", "unroll", "region"] | MissingType = MISSING
    jax_max_control_flow_unroll: int | MissingType = MISSING
    module_identity_mode: str | None | MissingType = MISSING
    payload_policy: str | None | MissingType = MISSING
    facets: (
        list[Callable[[Any], dict[str, Any]]]
        | tuple[Callable[[Any], dict[str, Any]], ...]
        | None
        | MissingType
    ) = MISSING
    episode: EpisodeOptions | None = None
    capture_output_structure: bool | MissingType = MISSING
    chunk_size: int | None | MissingType = MISSING
    chunk_paths: Iterable[Any] | None | MissingType = MISSING
    narrate: Any | None = None
    retain_output_parents_for_layers_to_save: bool = False
    _selective_layers_to_save_request: object | None = None


@dataclass(frozen=True, slots=True)
class _DoorState:
    """Resolution state of one torch door call, filled stage by stage with ``replace``.

    Values that are plain aliases of a ``capture_options`` or ``save_options`` field are
    read from those records; only values the door computes or rewrites live here.
    """

    args: _DoorArgs
    model: nn.Module
    input_args: Any
    input_kwargs: dict[Any, Any] | None
    capture_options: CaptureOptions
    raw_input: Any = None
    echo_options: Any = None
    meta_admission: Any = None
    save_options: Any = None
    grouped_save_options: Any = None
    save_predicate: Any = None
    streaming_options: Any = None
    episode_resolved: Any = None
    facet_recipes: Any = None
    chunk_size_value: int | None = None
    chunk_paths_value: Any = None
    normalized_chunk_size: int | None = None
    chunk_plan: Any = None
    layers_to_save: Any = None
    requested_layers_to_save: Any = None
    save_raw_input_policy: Any = None
    save_transformed_output_policy: Any = None
    save_mode_value: Any = "copy"
    halt: Any = None
    stop_after_site: Any = None
    stop_after_ambient: bool = False
    stop_after_selector_shaped: bool = False
    save_grads_policy: Any = None
    should_save_grads: bool = False
    grads_to_save_resolved: Any = None
    grad_storage_path_value: Any = None
    retain_grads_in_memory_value: Any = None
    train_mode_value: Any = None
    inference_only_value: Any = None
    uses_deferred_gradients: bool = False
    uses_deferred_activation: bool = False
    needs_live_output_projection: bool = False
    uses_selective_layers_to_save: bool = False
    inference_only_conflicts: tuple[str, ...] = ()
    log_name: str | None = None
    cache_key: str | None = None
    cache_path: Path | None = None
    cache_secret: bytes | None = field(default=None, repr=False)


def _run_plan_capture_rows(state: _DoorState) -> dict[str, Any]:
    """First half of the run keywords: retention, gradients, streaming and runtime knobs."""
    opts = state.capture_options
    save_options = state.save_options
    streaming_options = state.streaming_options
    return {
        "layers_to_save": state.layers_to_save,
        "keep_orphans": opts.keep_orphans,
        "output_device": opts.output_device,
        "activation_transform": save_options.activation_transform,
        "grad_transform": save_options.grad_transform,
        "save_raw_activations": save_options.save_raw_activations,
        "save_raw_grads": save_options.save_raw_gradients,
        "save_mode": cast("SaveMode", state.save_mode_value),
        "capture_tensor_grad_hooks": opts.capture_tensor_grad_hooks,
        "mark_layer_depths": opts.compute_input_output_distances,
        "detach_saved_activations": opts.detach_saved_activations,
        "save_args": opts.save_args,
        "save_grads": state.should_save_grads,
        "grads_to_save": state.grads_to_save_resolved,
        "seed": opts.seed,
        "num_context_lines": opts.source_context_lines,
        "optimizer": opts.optimizer,
        "echo_options": state.echo_options,
        "save_source_context": opts.save_source_context,
        "save_rng": opts.save_rng,
        "recurrence_detection": opts.recurrence_detection,
        "save_outs_to": streaming_options.path,
        "keep_outs_in_memory": streaming_options.retain_in_memory,
        "stream_custom_attributes": streaming_options.include_custom_attributes,
        "stream_buffer_values": streaming_options.include_buffer_values,
        "stream_async_writes": streaming_options.async_writes,
        "stream_max_pending_bytes": streaming_options.max_pending_bytes,
        "grad_storage_path": state.grad_storage_path_value,
        "retain_grads_in_memory": state.retain_grads_in_memory_value,
        "out_sink": streaming_options.on_save,
        "replayable": opts.replayable,
        "capture_container_structure": opts.capture_container_structure,
        "hooks": opts.hooks,
        "intervention_spec": None,
        "normalized_hook_plan": None,
        "verbose": opts.verbose,
        "backward_ready": state.train_mode_value,
        "inference_only": state.inference_only_value,
        "inference_only_implied": (
            state.inference_only_value and not opts._is_field_explicit("inference_only")
        ),
        "name": state.log_name,
        "module_filter": opts.module_filter,
    }


def _run_plan_policy_rows(state: _DoorState) -> dict[str, Any]:
    """Second half of the run keywords: session knobs, transforms, predicates, selectors."""
    opts = state.capture_options
    args = state.args
    uses_selective = state.uses_selective_layers_to_save
    return {
        "emit_nvtx": opts.emit_nvtx,
        "measure_python_peak_memory": opts.measure_python_peak_memory,
        "distributed_witness": opts.distributed_witness,
        "save_budget": opts.save_budget,
        # One closed vocabulary drives both nonfinite mechanisms: "raise" is the
        # stop-and-throw tripwire, "track" and "warn" record per-op verdicts.
        "raise_on_nan": opts.on_nonfinite == "raise",
        "track_nonfinite": opts.on_nonfinite in ("track", "warn"),
        "track_device_memory": opts.track_device_memory,
        "structure_only": opts.structure_only,
        "log_injections": opts.log_injections,
        "measure_host_peak_memory": opts.measure_host_peak_memory,
        "aten_ops": opts.aten_ops,
        "transform": opts.input_transform,
        "raw_input": state.raw_input,
        "save_raw_input": state.save_raw_input_policy,
        "batch_render": opts.batch_render,
        "output_transform": opts.output_transform,
        "save_transformed_output": state.save_transformed_output_policy,
        "layer_visualizers": opts.layer_visualizers,
        "save_visualizations": opts.save_visualizations,
        "recipes": state.facet_recipes,
        "save_predicate": state.save_predicate,
        "intervene_predicate": args.intervene,
        "halt_predicate": state.halt,
        "_halt_from_stop_after": state.stop_after_site is not None,
        "lookback": args.lookback,
        "lookback_payload_policy": args.lookback_payload_policy,
        "retain_output_parents_for_layers_to_save": (
            args.retain_output_parents_for_layers_to_save
            or uses_selective
            or state.episode_resolved is not None
        ),
        "episode_resolved": state.episode_resolved,
        "_selective_layers_to_save_request": (
            args._selective_layers_to_save_request
            if args._selective_layers_to_save_request is not None
            else (state.requested_layers_to_save if uses_selective else None)
        ),
        "_deferred_retention_selector": (
            state.requested_layers_to_save
            if state.uses_deferred_activation
            else (["output"] if state.needs_live_output_projection else None)
        ),
        "_deferred_gradient_selector": (
            state.grads_to_save_resolved
            if state.uses_deferred_gradients
            else ("all" if state.uses_deferred_activation and state.train_mode_value else None)
        ),
    }


def _rescue_eligible(state: _DoorState) -> bool:
    """Whether a failed forward may be re-run under the rescue path.

    Streaming saves, sinks, and halt-predicate partials are not re-runnable;
    they report an escape as before instead of attempting a rescue re-run.
    EVERY user-callable channel the re-run would invoke a SECOND time is
    refused (fail closed): side effects (counters, file writes,
    externally-held state) would double-apply with only a session-time
    disclosure. That covers intervention transforms, pre-attached hooks,
    AND the in-capture transform callables -- ``activation_transform``,
    ``grad_transform``, and ``output_transform`` run inside ``run_capture``
    and measurably fired twice on a recovered rescue (b6-opus-R16-1 reopen).
    Channels applied OUTSIDE the rescue boundary stay eligible: ``save=``
    selector predicates (pure by contract), the input ``transform=``
    (applied before the re-run boundary; fires once), and
    ``layer_visualizers`` (rendered once on the returned trace, after the
    driver).
    """
    streaming_options = state.streaming_options
    save_options = state.save_options
    return (
        streaming_options.path is None
        and streaming_options.on_save is None
        and state.grad_storage_path_value is None
        and state.halt is None
        and state.args.intervene is None
        and not state.capture_options.hooks
        and save_options.activation_transform is None
        and save_options.grad_transform is None
        and state.capture_options.output_transform is None
    )


def _build_request(
    state: _DoorState, *, cache_store: Callable[[Any, Path, Any], None] | None = None
) -> CaptureRequest:
    """Pack the resolved door state into the request the RUN stage consumes.

    ``cache_store`` is the door's authenticated capture-cache writer; the
    epilogue calls it through ``CaptureRequest.cache["store"]``.
    """
    cache = None
    if state.cache_key is not None:
        cache = {
            "key": state.cache_key,
            "path": state.cache_path,
            "secret": state.cache_secret,
            "store": cache_store,
        }
    return CaptureRequest(
        model=state.model,
        input_args=state.input_args,
        input_kwargs=state.input_kwargs,
        backend="torch",
        options=state.capture_options,
        plan={**_run_plan_capture_rows(state), **_run_plan_policy_rows(state)},
        cache=cache,
        admission={
            "meta_admission": state.meta_admission,
            "rescue_eligible": _rescue_eligible(state),
            "episode_resolved": state.episode_resolved,
            "stop_after_site": state.stop_after_site,
            "stop_after_selector_shaped": state.stop_after_selector_shaped,
            "stop_after_ambient": state.stop_after_ambient,
            "save_grads_policy": state.save_grads_policy,
            "uses_selective_layers_to_save": state.uses_selective_layers_to_save,
        },
        postprocess=True,
    )
