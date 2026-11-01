"""Public API entry points for TorchLens.

This module contains every user-facing function:
  - ``trace``  - the main entry point (runs model, returns Trace)
  - ``validate_forward_pass`` - replay-based correctness check
  - ``show_model_graph`` - visualization convenience wrapper
  - ``draw_backward`` - backward grad_fn_handle visualization wrapper
  - ``log_model_metadata`` - metadata-only convenience wrapper
  - ``validate_batch_of_models_and_inputs`` - bulk validation harness

**Selective save strategy**:
Predicate ``save=`` and most string/substring ``layers_to_save`` requests are
resolved during the primary forward pass. TorchLens falls back to the two-pass
discovery/replay path only for selectors that require finalized labels or
gradient-specific resolution.
"""

import functools
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from torchlens._capture.finalize.epilogue import (
    _render_layer_visualizers as _render_layer_visualizers,
    _render_one_layer_visualizer as _render_one_layer_visualizer,
    _safe_visualizer_filename as _safe_visualizer_filename,
)
from torchlens._capture.prepare.cache import (
    _CAPTURE_CACHE_HEADER_BYTES as _CAPTURE_CACHE_HEADER_BYTES,
    _CAPTURE_CACHE_MAGIC as _CAPTURE_CACHE_MAGIC,
    _CAPTURE_CACHE_MAX_BYTES as _CAPTURE_CACHE_MAX_BYTES,
    _CAPTURE_CACHE_MAX_ENTRIES as _CAPTURE_CACHE_MAX_ENTRIES,
    _CAPTURE_CACHE_SECRET_BYTES as _CAPTURE_CACHE_SECRET_BYTES,
    _CAPTURE_CACHE_SECRET_FILE as _CAPTURE_CACHE_SECRET_FILE,
    _CAPTURE_CACHE_TAG_HEX_CHARS as _CAPTURE_CACHE_TAG_HEX_CHARS,
    _CAPTURE_CACHE_TAG_SUFFIX as _CAPTURE_CACHE_TAG_SUFFIX,
    _CAPTURE_CACHE_TEMP_DEBRIS_AGE_SECONDS as _CAPTURE_CACHE_TEMP_DEBRIS_AGE_SECONDS,
    CAPTURE_CACHE_KEY_CURATED as CAPTURE_CACHE_KEY_CURATED,
    CAPTURE_CACHE_KEY_NEUTRAL as CAPTURE_CACHE_KEY_NEUTRAL,
    _capture_cache_io_error as _capture_cache_io_error,
    _capture_cache_secret as _capture_cache_secret,
    _CaptureCacheEntryOverCeilingError as _CaptureCacheEntryOverCeilingError,
    _evict_capture_cache as _evict_capture_cache,
    _harden_capture_cache_dir as _harden_capture_cache_dir,
    _load_authenticated_capture_cache as _load_authenticated_capture_cache,
    _prepare_capture_cache_dir as _prepare_capture_cache_dir,
    _store_authenticated_capture_cache as _store_authenticated_capture_cache,
    _sweep_option_fields_into_cache_config as _sweep_option_fields_into_cache_config,
    _sweep_stale_capture_cache_temp_files as _sweep_stale_capture_cache_temp_files,
    _TaggingWriter as _TaggingWriter,
)
from torchlens._capture.prepare.dispatch import (
    _MLX_STATIC_LABEL_SAVE_SELECTOR_KINDS as _MLX_STATIC_LABEL_SAVE_SELECTOR_KINDS,
    _capability_option_requested as _capability_option_requested,
    _enforce_capability_option_gates as _enforce_capability_option_gates,
    _filter_trace_kwargs_for_backend as _filter_trace_kwargs_for_backend,
    _reject_unsupported_torch_trace_option_values as _reject_unsupported_torch_trace_option_values,
    _trace_mlx_model as _trace_mlx_model,
    _trace_mlx_model_from_public_kwargs as _trace_mlx_model_from_public_kwargs,
    _trace_option_explicit as _trace_option_explicit,
    _unsupported_trace_option_message as _unsupported_trace_option_message,
)
from torchlens._capture.prepare.door import (
    _BATCHNORM_TRAIN_STATS_WARNED as _BATCHNORM_TRAIN_STATS_WARNED,
    _model_type_remedy as _model_type_remedy,
    _release_preparation_after_failed_capture as _release_preparation_after_failed_capture,
    _trace_torch_model as _trace_torch_model,
    _warn_failed_capture_release_incomplete as _warn_failed_capture_release_incomplete,
    _warn_once_train_mode_running_stats as _warn_once_train_mode_running_stats,
)
from torchlens._capture.prepare.resolve import (
    _warn_zero_match_capture_selectors as _warn_zero_match_capture_selectors,
)
from torchlens._capture.run_entry import (
    _register_live_tensor_connection as _register_live_tensor_connection,
    _run_model_and_save_specified_outs as _run_model_and_save_specified_outs,
    register_tensor_connection as register_tensor_connection,
)

from . import _state
from ._autoroute._builtin_output import semantic_output_cache_key as semantic_output_cache_key
from ._capture_state_helpers import (
    _capture_cache_dir,
    _clone_state_dict_with_metadata as _clone_state_dict_with_metadata,
    _reject_opaque_wrappers as _reject_opaque_wrappers,
    _unwrap_data_parallel as _unwrap_data_parallel,
)
from ._data_classes.trace import (
    Trace,
)
from ._deprecations import MISSING, MissingType
from ._errors import (
    CaptureNotActiveError as CaptureNotActiveError,
    UserTransformError as UserTransformError,
)
from ._input_coerce import (
    _reject_extra_positional_input,
    _reject_non_mapping_input_kwargs,
    _reject_unrouted_forward_kwargs,
)
from ._robustness import check_model_and_input_variants as check_model_and_input_variants
from ._user_funcs_guards import (
    _refuse_non_torch_episode,
    _reject_auto_environment,
    _reject_compiled_plain_callable,
    _reject_unrouted_jax_options,
)
from ._utils.tensor_utils import SaveMode
from .backends.registry import (
    BackendName,
    get_backend_spec,
    resolve_backend_spec,
)
from .backends.torch.bound_root import (
    _reject_non_module_ladder_root,
)
from .bridge import transformers as _hf_bridge
from .intervention import model_door as _model_door
from .intervention._zero_fire import zero_fire_policy_scope
from .intervention.errors import ChunkedForwardConfigError
from .intervention.predicates import InterventionPredicate
from .intervention.selectors import BaseSelector
from .options import (
    CaptureOptions,
    EpisodeOptions,
    SaveOptions,
    StorageOptions,
)
from .recording.options import HaltPredicateFn, PredicateFn
from .types import _GradientPostfunc

_can_resolve_hf_processor = _hf_bridge._can_resolve_hf_processor
_can_resolve_hf_tokenizer = _hf_bridge._can_resolve_hf_tokenizer
_has_attached_image_processor = _hf_bridge._has_attached_image_processor
_is_hf_image_input = _hf_bridge._is_hf_image_input
_is_hf_multimodal_input = _hf_bridge._is_hf_multimodal_input
_is_hf_text_input = _hf_bridge._is_hf_text_input


def clear_cache(cache_dir: str | Path | None = None) -> int:
    """Delete authenticated capture-cache entries while preserving the secret.

    Parameters
    ----------
    cache_dir:
        Cache root accepted by ``trace(cache_dir=...)``. ``None`` uses the
        configured default.

    Returns
    -------
    int
        Number of payload entries removed.
    """

    cache_root = _capture_cache_dir(cache_dir) / "capture"
    if not cache_root.exists():
        return 0
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise _capture_cache_io_error(f"Refusing non-directory capture cache {cache_root}.")
    _sweep_stale_capture_cache_temp_files(cache_root)
    removed = 0
    for payload in cache_root.glob("*.pkl"):
        if payload.is_symlink():
            continue
        tag = payload.with_name(payload.name + _CAPTURE_CACHE_TAG_SUFFIX)
        payload.unlink(missing_ok=True)
        if not tag.is_symlink():
            tag.unlink(missing_ok=True)
        removed += 1
    return removed


def list_logs() -> tuple[Trace, ...]:
    """Return a snapshot of currently live ``Trace`` objects.

    Returns
    -------
    tuple[Trace, ...]
        Immutable snapshot from TorchLens' process-wide weak registry.
    """

    return _state.list_logs()


def reset_naming_counter(class_name: str | None = None) -> None:
    """Reset automatic ``Trace`` naming counters.

    Parameters
    ----------
    class_name:
        Lowercase short class name to reset, or ``None`` to reset all counters.

    Returns
    -------
    None
        The process-global counter dictionary is updated.
    """

    _state.reset_naming_counter(class_name)


def render(*args: Any, **kwargs: Any) -> Any:
    """One-call model picture over the quickstart input ladder (F17 B15).

    ONE metadata-only eval/no-grad capture (training flags, RNG, and norm
    buffers restored), rendered with the facade default ``collapse="auto"``
    (``Trace.draw`` keeps ``"none"``). Detached
    :class:`torchlens._quickstart.Drawing`; never auto-opens a viewer.
    See :func:`torchlens._quickstart.render` for the parameters.
    """

    from ._quickstart._render import render as _render_impl

    return _render_impl(*args, **kwargs)


def trace(
    model: nn.Module | Callable[..., Any],
    input_args: str | torch.Tensor | list[Any] | tuple[Any, ...] | None = None,
    input_kwargs: dict[Any, Any] | None = None,
    grad_transform: _GradientPostfunc | None | MissingType = MISSING,
    save_mode: SaveMode | MissingType = MISSING,
    reconstruction_ready: bool | MissingType = MISSING,
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
    profile: bool | MissingType = MISSING,
    facets: (
        list[Callable[[Any], dict[str, Any]]]
        | tuple[Callable[[Any], dict[str, Any]], ...]
        | None
        | MissingType
    ) = MISSING,
    *,
    grouping: str | MissingType = MISSING,
    narrate: Any | None = None,
    jax_static_argnums: int | Sequence[int] | MissingType = MISSING,
    grad_options: Any | None | MissingType = MISSING,
    episode: EpisodeOptions | None = None,
    chunk_size: int | None | MissingType = MISSING,
    chunk_paths: Iterable[Any] | None | MissingType = MISSING,
    backend: BackendName | None = None,
    input_shape: Any | None = None,
    on_zero_fire: str | MissingType = MISSING,
    **forward_kwargs: Any,
) -> Trace:
    """Run a forward pass through *model*, log every operation, and return a Trace.

    This is the primary user-facing entry point for TorchLens.  It intercepts every
    tensor-producing operation during ``model.forward()``, records metadata and
    (optionally) saves outs, then returns a ``Trace`` that provides
    dict-like access to every layer's data.

    Torch functions are automatically wrapped on the first call and stay wrapped
    afterward.  Pass ``capture=CaptureOptions(unwrap_when_done=True)`` to restore
    the original torch callables after logging completes.

    Faster paths (capture-free ``spec.bind(model)`` steering, ``tl.record``): see
    ``docs/guides/fast_paths.md``.

    **Layer selection** (``save=``, the canonical spelling):

    - ``'all'`` (default) - save outs for every layer.
    - ``'none'`` / ``None`` / ``[]`` - save no outs (metadata only).
    - ``'stats'`` - fold every out into a ``Spine``, keep no payload (``tl.stats.site_stats``).
    - A predicate selector, e.g. ``tl.func("relu")``, ``tl.in_module("encoder")``,
      or combinators such as ``tl.func("conv2d") & tl.followed_by(tl.func("relu"))``.
    - A list containing any mix of:
      1. Layer name, e.g. ``'conv2d_1_1'`` (all ops).
      2. Pass-qualified label, e.g. ``'conv2d_1_1:2'`` (second pass only).
      3. Module address, e.g. ``'features.0'`` (output of that module).
      4. Integer index (ordinal position; negative indices work).
      5. Substring filter, e.g. ``'conv2d'`` (all matching layers).

    Most string and substring layer selections are absorbed into a single-pass
    predicate save. TorchLens falls back to the two-pass discovery/replay path
    only for selectors that require finalized labels, such as negative indexes,
    identity/output labels, or gradient-specific selection.

    Parameters
    ----------
    model:
        PyTorch model, or a bound method of one (the ruled root contract,
        F41): ``tl.trace(model.generate, ids, ...)`` resolves the owner via
        ``method.__self__``, wraps it in a TL-authored synthetic root, and
        calls the method exactly once.
    input_args:
        Positional args for ``model.forward()``; a single tensor or list.
        The input ladder (memo D2): a real input is the gold rung; omit
        it and pass ``input_shape=``, or omit both to infer -- synthesized
        values, disclosed, or a teach; mixing refuses ``input_rung_conflict``.
    input_kwargs:
        Keyword args for ``model.forward()``.
    input_shape:
        Declared input shape(s) (torch backend only): one flat positive-int
        tuple, a sequence of shape tuples, or a mapping of forward keyword
        names to shapes (batch included exactly as written). Dtype/value
        recipes are fail-closed static facts (override:
        ``torchlens.types.InputSpec``); local seed-0 synthesis,
        disclosed in the trace's persistent provenance record.
    save:
        Which layers to save outs for; the canonical selection kwarg (see
        **Layer selection** above). Accepts ``'all'``, ``'none'``/``None``/``[]``,
        label/module/index/substring lists, predicate selectors
        (``tl.func``, ``tl.in_module``, ``tl.followed_by``, ``tl.when``
        combinators), or a grouped ``SaveOptions``.
    grad_transform:
        Optional function applied to each grad before saving. The raw
        grad remains in ``layer.grad`` by default, and the transform result is stored
        in ``layer.transformed_grad``.
    save_mode:
        Tensor retention mode for saved activation and gradient payloads.
        ``"copy"`` is the safe cloning default; ``"reference"`` preserves the
        captured value through in-place handling; ``"view"`` is a live alias that
        downstream in-place operations can mutate; and ``"cpu_async"`` clones to CPU.
    reconstruction_ready:
        If True, auto-enable the argument and RNG capture
        prerequisites needed by read-only reconstructed facets such as fused
        SDPA ``scores``, ``pattern``, and ``z``.
    capture:
        Grouped capture options (``CaptureOptions``). The one spelling for
        every capture knob (``layers_to_save``, ``random_seed``,
        ``intervention_ready``, ``verbose``, ``structure_only``, ...); the
        former flat kwargs are removed.
    intervene:
        Optional predicate returning an intervention decision for
        current-op live mutation.
    halt:
        Optional predicate returning ``True`` to stop after the matching
        source, operation, or module-boundary event and return the partial trace.
    lookback:
        Number of recent capture events queryable by predicate-window helpers.
    lookback_payload_policy:
        Retention policy for retroactive ``followed_by`` saves.
        ``"metadata_only"`` keeps the default metadata-only window and cannot
        retroactively save payloads. Non-default policies retain up to ``lookback``
        candidate payloads, for a memory cost of roughly ``lookback`` times the
        candidate payload size.
    storage:
        Shared storage routing option. ``storage=tl.to_disk(path)``
        streams predicate-selected saves to a disk bundle during the
        forward pass. ``None`` preserves the existing in-RAM behavior.
    streaming:
        Grouped streaming-save options (``StorageOptions``).
    chunk_size:
        If supplied, split a positional tensor input into forward
        chunks of this size along dimension 0 and append them into one
        in-memory ``Trace``. Forward-only and torch-only.
    chunk_paths:
        Optional explicit tensor leaf paths to split when multiple
        batched tensor leaves are present.
    profile:
        If True, explicitly marks the returned trace as profiled. Phase timings are
        always populated on ``trace._phase_timings``.
    facets:
        Per-trace additive facet recipes captured into the immutable
        registry snapshot for the returned trace.
    grouping:
        UNSTABLE (no deprecation shim owed). Closed-vocabulary grouping
        policy: ``"structural"`` (the default -- today's recurrence
        grouping), ``"strict_shapes"`` (reserved; refuses typed until its
        own reviewed design lands), ``"fold_sites"`` (the D1 folding axis;
        refuses typed on plain captures until an affirmative D1 ruling).
        The requested value is recorded on ``trace.grouping`` and the
        policy that actually ran on ``trace.grouping_policy``. Distinct
        from the display-only ``fold_repeats`` viz knob, which folds
        repeated module runs at RENDER time and never changes grouping.
    narrate:
        Live narration (DOCUMENTED-UNSTABLE, lane F28): ``True`` narrates
        every completed tensor-output event plus module structure lines;
        ``"modules"`` narrates structure only; a live selector scopes the
        stream; ``tl.options.NarrationOptions`` groups the full surface (stats
        rungs, sink, crash tail). Narration is a display of capture
        events -- independent of ``save=`` retention. Torch-only.
    jax_static_argnums:
        JAX-only positional argument indexes passed to
        ``jax.make_jaxpr(..., static_argnums=...)`` when
        ``backend="jax"``. Non-default values require the explicit JAX
        backend.
    grad_options:
        Backend-specific derived-gradient options for the
        leaf-level preview. Supported by explicit ``backend="jax"`` and
        ``backend="tinygrad"`` only.
    episode:
        Torch-only episode capture declaration (``EpisodeOptions``).
    backend:
        Explicit backend name. ``None`` preserves legacy auto-resolution.
    on_zero_fire:
        What a capture-time ``intervene=`` that fired at zero sites does once the fact
        is recorded (``annotations["unmatched_capture_selectors"]``, intervention audit):
        ``"warn"``, used when not passed, warns ``trace_zero_fire``; ``"error"`` raises
        it; ``"disclose"`` records only. Same knob as ``spec.bind`` and ``Trace.run``
        (spelling DOCUMENTED-UNSTABLE).
    **forward_kwargs:
        Never accepted -- unknown keywords refuse typed with
        ``trace_forward_kwargs_unrouted`` naming the ``input_kwargs=`` routing.

    Postfunc behavior:
        ``save.activation_transform`` and ``grad_transform`` both take a tensor, should return a
        tensor for portable-save and streaming compatibility, run under ``pause_logging()``, and
        raise ``UserTransformError`` with layer/function/tensor context if they fail.

        Activation transforms run during forward capture. Their result is stored alongside the
        raw out by default, and ``capture.backward_ready=True`` requires the transformed out to
        stay graph-connected and differentiable when the raw out requires grads.

        Gradient transforms run from the backward hook output, so they follow the grad tensor's
        shorter lifetime rather than forward out retention. When the raw grad itself
        requires grads under ``capture.backward_ready=True``, the same checks apply.

    Returns
    -------
    Trace
        A ``Trace`` containing layer outs (if requested) and full metadata.
    """
    with zero_fire_policy_scope(None if on_zero_fire is MISSING else on_zero_fire):
        _reject_unrouted_forward_kwargs(forward_kwargs)
        del forward_kwargs  # must never ride the ladder's locals() snapshot below
        _reject_non_mapping_input_kwargs(input_kwargs)
        model, intervene = _model_door.resolve_trace_operands(model, intervene)
        if input_shape is not None or input_args is None:
            model = _reject_non_module_ladder_root(model)
            # Quickstart input ladder (F17, memo D2/D4): declared/inferred rungs
            # resolve BEFORE any capture machinery, then re-enter trace() as gold.
            ladder_kwargs = locals().copy()
            for consumed in ("model", "input_args", "input_kwargs", "input_shape"):
                ladder_kwargs.pop(consumed)
            from ._quickstart._ladder import _trace_via_ladder

            return _trace_via_ladder(model, input_args, input_kwargs, input_shape, ladder_kwargs)
        _reject_compiled_plain_callable(model)
        _reject_extra_positional_input(grad_transform)
        public_trace_kwargs = locals().copy()
        public_trace_kwargs.pop("backend")
        public_trace_kwargs.pop("input_shape")  # ladder-consumed; always None here
        # Settled by the zero_fire_policy_scope above, never a backend option.
        public_trace_kwargs.pop("on_zero_fire")
        # grouping= (UNSTABLE, keyword-only; L1 wave 0): closed-vocabulary knob.
        # Only "structural" (today's grouping, the default) is entry-legal;
        # "strict_shapes" waits on its own reviewed design and "fold_sites" on
        # an affirmative D1 ruling (the flip PR) / the episode capture kind.
        public_trace_kwargs.pop("grouping")
        if grouping is not MISSING:
            from ._postprocess._grouping_stamp import validate_grouping_knob

            validate_grouping_knob(grouping)
        if narrate is not None and narrate is not False:
            # Entry-time echo refusals (bad spellings, finalized-label selectors)
            # fire BEFORE any capture work on every backend path.
            from ._snoop import normalize_echo

            normalize_echo(narrate)
        if chunk_paths is not MISSING and chunk_paths is not None and chunk_size in (MISSING, None):
            raise ChunkedForwardConfigError("chunk_paths requires chunk_size.")
        _reject_unrouted_jax_options(backend, jax_static_argnums, grad_options)
        explicit_backend_spec = None
        if backend is not None:
            explicit_backend_spec = get_backend_spec(str(backend))
            _filter_trace_kwargs_for_backend(public_trace_kwargs, explicit_backend_spec)
            explicit_backend_spec = resolve_backend_spec(backend, model, input_args, input_kwargs)
        if (
            backend is None
            and chunk_size in (MISSING, None)
            and (capture is None or not capture.is_field_explicit("input_transform"))
        ):
            from . import _autoroute as autoroute

            # Only the surviving public kwargs are forwarded: capture knobs travel
            # inside the grouped ``capture=`` object, so re-entrant trace() calls
            # inside detectors receive exactly what the user could have passed.
            autoroute_kwargs = {
                "input_kwargs": input_kwargs,
                "grad_transform": grad_transform,
                "save_mode": save_mode,
                "reconstruction_ready": reconstruction_ready,
                "capture": capture,
                "save": save,
                "intervene": intervene,
                "halt": halt,
                "lookback": lookback,
                "lookback_payload_policy": lookback_payload_policy,
                "storage": storage,
                "streaming": streaming,
                "profile": profile,
                "facets": facets,
                "episode": episode,
                "grouping": grouping,
                "narrate": narrate,
                "on_zero_fire": on_zero_fire,
            }
            for detector in autoroute.input.iter_by_priority():
                result = detector(model, input_args, **autoroute_kwargs)
                if result is not None:
                    return cast("Trace", result)
        _reject_auto_environment()
        resolved_spec = explicit_backend_spec or resolve_backend_spec(
            backend, model, input_args, input_kwargs
        )
        from ._snoop._entry import refuse_echo_non_torch

        # Never a silent no-op: narrate= is torch-only in wave 1 (typed refusal).
        refuse_echo_non_torch(narrate, resolved_spec)
        _filter_trace_kwargs_for_backend(public_trace_kwargs, resolved_spec)
        _enforce_capability_option_gates(public_trace_kwargs, resolved_spec)
        _refuse_non_torch_episode(public_trace_kwargs, resolved_spec)
        return cast("Trace", resolved_spec.capture_trace(**public_trace_kwargs))


def log_model_metadata(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
) -> Trace:
    """Return model metadata without saving any outs.

    Parameters
    ----------
    model:
        Model whose metadata should be captured.
    input_args:
        Positional input arguments for ``model.forward``.
    input_kwargs:
        Keyword input arguments for ``model.forward``.

    Returns
    -------
    Trace
        Metadata-only trace with input/output distance metadata enabled.
    """

    return trace(
        model,
        input_args,
        input_kwargs,
        capture=CaptureOptions(
            layers_to_save=None,
            compute_input_output_distances=True,
        ),
    )


def _public_impls_module() -> Any:
    """Return private public-command implementations.

    The implementations call ``trace`` and ``_run_model_and_save_specified_outs``
    through this module at call time; nothing is copied into theirs, so a patch
    of either name here is seen while it is in place and gone once it is undone.
    """

    from . import _user_public_impls

    _sync_public_impl_wrapper_metadata(_user_public_impls)
    return _user_public_impls


def release_model(model: nn.Module) -> None:
    """Release a traced PyTorch model from persistent TorchLens preparation.

    Parameters
    ----------
    model:
        Model whose full module tree should be restored. The operation is safe
        for never-traced and already-released models.

    Returns
    -------
    None
        The model is restored in place and may be pickled or traced again.

    Notes
    -----
    TorchLens installs persistent, toggle-gated wrappers on non-root module
    ``forward`` methods. Call ``release_model`` after the final trace when the
    complete model object must be serialized with :func:`torch.save` or
    :mod:`pickle`. Saving ``model.state_dict()`` is unaffected by preparation
    and does not require release.
    """
    model = _model_door.resolve_released_model(model)
    _public_impls_module().release_model(model)


def summary(*args: Any, **kwargs: Any) -> str:
    """Run a one-call capture and return the rendered summary string.

    ``tl.summary(model, x)`` is the first-class one-call front door: it runs a
    metadata-only capture and RETURNS the rendered summary text (it never
    auto-prints). ``trace.summary()`` reports an existing capture instead.
    """

    return cast(str, _public_impls_module().summary(*args, **kwargs))


def show_model_graph(*args: Any, **kwargs: Any) -> None:
    """Forward to the model-graph rendering implementation."""

    return cast(None, _public_impls_module().show_model_graph(*args, **kwargs))


def draw_backward(*args: Any, **kwargs: Any) -> str:
    """Forward to the backward graph rendering implementation."""

    return cast(str, _public_impls_module().draw_backward(*args, **kwargs))


def draw_combined(*args: Any, **kwargs: Any) -> str:
    """Forward to the combined graph rendering implementation."""

    return cast(str, _public_impls_module().draw_combined(*args, **kwargs))


def show_bundle_graph(*args: Any, **kwargs: Any) -> str | None:
    """Forward to the bundle graph rendering implementation."""

    return cast(str | None, _public_impls_module().show_bundle_graph(*args, **kwargs))


def validate_forward_pass(*args: Any, **kwargs: Any) -> bool:
    """Forward to the backend-dispatched forward validation implementation."""

    return cast(bool, _public_impls_module().validate_forward_pass(*args, **kwargs))


def _validate_forward_pass_torch(*args: Any, **kwargs: Any) -> bool:
    """Forward to the torch forward validation implementation."""

    return cast(bool, _public_impls_module()._validate_forward_pass_torch(*args, **kwargs))


def validate_backward_pass(*args: Any, **kwargs: Any) -> bool:
    """Forward to the backward validation implementation."""

    return cast(bool, _public_impls_module().validate_backward_pass(*args, **kwargs))


def validate_batch_of_models_and_inputs(*args: Any, **kwargs: Any) -> Any:
    """Forward to the batch validation implementation."""

    return _public_impls_module().validate_batch_of_models_and_inputs(*args, **kwargs)


_PUBLIC_IMPL_WRAPPER_NAMES = (
    "show_model_graph",
    "draw_backward",
    "draw_combined",
    "show_bundle_graph",
    "validate_forward_pass",
    "validate_backward_pass",
    "validate_batch_of_models_and_inputs",
)

_public_impl_metadata_synced = False


def _sync_public_impl_wrapper_metadata(implementations: Any = None) -> None:
    """Expose canonical signatures on lazily delegated public wrappers.

    ``torchlens._user_public_impls`` imports this module at its top, so when IT is
    the module imported first (``import torchlens._user_public_impls``) the sync
    below observes a partially initialized implementation module. Skipping the
    not-yet-defined names -- instead of raising ``AttributeError`` and breaking a
    standalone import -- keeps the cycle inert; the next
    :func:`_public_impls_module` call completes the sync, and the flag makes the
    completed sync a one-time cost.

    Parameters
    ----------
    implementations:
        Already-resolved implementation module, when the caller holds one.

    Returns
    -------
    None
        Updates wrapper metadata in place once the implementation module is fully
        loaded.
    """

    global _public_impl_metadata_synced
    if _public_impl_metadata_synced:
        return
    if implementations is None:
        from . import _user_public_impls as implementations
    pending = False
    for name in _PUBLIC_IMPL_WRAPPER_NAMES:
        implementation = getattr(implementations, name, None)
        if implementation is None:
            pending = True
            continue
        functools.update_wrapper(globals()[name], implementation)
    _public_impl_metadata_synced = not pending


_sync_public_impl_wrapper_metadata()
