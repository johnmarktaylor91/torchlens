"""Public API entry points for TorchLens.

This module contains every user-facing function:
  - ``trace``  - the main entry point (runs model, returns Trace)
  - ``validate_forward_pass`` - replay-based correctness check
  - ``show_model_graph`` - visualization convenience wrapper
  - ``log_model_metadata`` - metadata-only convenience wrapper

**Selective save strategy**:
Predicate ``save=`` and most string/substring ``save=`` label requests are
resolved during the primary forward pass. TorchLens falls back to the two-pass
discovery/replay path only for selectors that require finalized labels or
gradient-specific resolution.
"""

import functools
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from torchlens._capture.prepare.cache import (
    _CAPTURE_CACHE_TAG_SUFFIX,
    _capture_cache_io_error,
    _sweep_stale_capture_cache_temp_files,
)
from torchlens._capture.prepare.dispatch import (
    _dispatch_trace,
    _refuse_compiled_callable_root,
    _refuse_trace_entry_shapes,
    _unfold_door_capture_fields,
)
from torchlens._capture.run_entry import (
    _run_model_and_save_specified_outs as _run_model_and_save_specified_outs,
)

from . import _state
from ._capture_state_helpers import (
    _capture_cache_dir,
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
from ._input_coerce import _reject_extra_positional_input, _reject_non_mapping_input_kwargs
from ._registry._facet_snapshot import register_facet_snapshot_provider
from ._selection.selectors import BaseSelector
from ._trace_selector_helpers import _is_save_label_selection, _route_save_label_selection
from ._unrouted_keywords import reject_unrouted_forward_kwargs
from .backends.registry import (
    BackendName,
)
from .backends.torch.bound_root import (
    _reject_non_module_ladder_root,
)
from .intervention import _model_door as _model_door
from .intervention._predicates import InterventionPredicate
from .intervention._zero_fire import zero_fire_policy_scope
from .options import (
    CaptureOptions,
    EpisodeOptions,
    StorageOptions,
    merge_capture_options,
)
from .recording._options import PredicateFn

#: ``trace()`` parameters that take an options object; unrouted-keyword remedies name them.
_TRACE_OPTION_CLASSES: dict[str, type] = {
    "capture": CaptureOptions,
    "storage": StorageOptions,
    "episode": EpisodeOptions,
}


def _facet_snapshot_provider(extra_recipes: Any) -> Any:
    """Build a trace's facet snapshot, loading the facets appliance on first use.

    ``tl.trace(model, x, facets=[fn])`` accepts plain callables, so a capture can carry
    recipes before anything imported ``torchlens._semantic.facets``; this facade door
    (Rule F) registers the lazy load so the engine never imports upward itself.
    """

    from ._semantic import facets as _facets  # deferred: startup cost (facade door)

    return _facets.snapshot(extra_recipes)


register_facet_snapshot_provider(_facet_snapshot_provider, replace=False)


def clear_cache(cache_dir: str | Path | None = None) -> int:
    """Delete capture-cache entries; the per-user secret lives elsewhere and is kept.

    Parameters
    ----------
    cache_dir:
        Cache root accepted by ``trace(cache_dir=...)``. ``None`` uses the
        configured default.

    Returns
    -------
    int
        Number of payload entries removed.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     opts = tl.options.CaptureOptions(cache=True, cache_dir=tmp)
    ...     _ = tl.trace(model, x, capture=opts)
    ...     print(tl.clear_cache(tmp))
    1
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


def draw(*args: Any, **kwargs: Any) -> Any:
    """One-call model picture: resolve the input, capture metadata only, draw.

    ONE metadata-only eval/no-grad capture (training flags, RNG, and norm
    buffers restored), rendered with ``collapse="auto"`` by default
    (``Trace.draw`` keeps ``"none"``). Returns a detached
    :class:`torchlens.viz.Drawing`; never auto-opens a viewer.
    See :func:`torchlens._quickstart.draw` for the parameters.

    Examples
    --------
    >>> # requires: graphviz
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     drawing = tl.draw(model, x, path=f"{folder}/net.svg", file_format="svg")
    TorchLens graph written to ...net.svg
    >>> type(drawing).__name__, drawing.file_format, drawing.path.name
    ('Drawing', 'svg', 'net.svg')
    """

    from ._quickstart._render import draw as _draw_impl

    return _draw_impl(*args, **kwargs)


def trace(
    model: nn.Module | Callable[..., Any],
    inputs: str | torch.Tensor | list[Any] | tuple[Any, ...] | None = None,
    *extra_positional: Any,
    input_kwargs: dict[Any, Any] | None = None,
    capture: CaptureOptions | None = None,
    save: PredicateFn
    | BaseSelector
    | str
    | list[Any]
    | tuple[Any, ...]
    | None
    | MissingType = MISSING,
    intervene: InterventionPredicate | None = None,
    stop_after: Any | None = None,
    storage: StorageOptions | None = None,
    facets: (
        list[Callable[[Any], dict[str, Any]]]
        | tuple[Callable[[Any], dict[str, Any]], ...]
        | None
        | MissingType
    ) = MISSING,
    narrate: Any | None = None,
    episode: EpisodeOptions | None = None,
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
        PyTorch model, or a bound method of one: ``tl.trace(model.generate, ids, ...)``
        resolves the owner via ``method.__self__``, wraps it in a synthetic root module
        that TorchLens builds, and calls the method exactly once.
    inputs:
        Positional args for ``model.forward()``; a single tensor or list. A real input
        is best; else omit it and pass ``input_shape=``, or omit both to infer the shape
        (synthesized values, disclosed, or an error that names the fix). Passing both a
        real input and ``input_shape=`` raises ``input_rung_conflict``.
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
        combinators) or ``"stats"`` (per-op summaries, no payload). Activation and
        gradient transforms ride ``CaptureOptions``.
    capture:
        Grouped capture options (``CaptureOptions``). The one spelling for
        every capture knob (``seed``, ``replayable``, ``verbose``,
        ``structure_only``, ``save_mode``, ``lookback``, ``chunk_size``,
        ``grouping``, ...); the former flat kwargs are removed. ``grouping``
        (UNSTABLE) is the closed-vocabulary grouping policy: ``"structural"``
        (the default); ``"strict_shapes"`` and ``"fold_sites"`` raise a typed
        error (not implemented yet); the request is recorded on
        ``trace.options.grouping`` and the policy that ran on
        ``trace.options.grouping_policy``.
    intervene:
        Optional predicate returning an intervention decision for
        current-op live mutation.
    stop_after:
        One stop directive, a site or a predicate: a module address string
        (``"encoder.layer.4"``), a function name (``"relu"``), a live selector
        (``tl.func("relu")``, ``tl.module(...)``) or a callable predicate
        taking the live op context. Capture halts inclusively after the first
        match and returns a partial halted trace. A selector-shaped site that
        never fires raises (``stop_after_never_fired``); a callable that
        never fires warns (``stop_after_never_fired_callable``). Finalized
        labels raise (``stop_after_site_not_live``); an unsupported type
        raises (``stop_after_type_invalid``). Cannot combine with
        ``CaptureOptions(chunk_size=...)``. ``None`` reads the ambient
        ``tl.experimental.stop_after`` site, if one is active.
    storage:
        Shared storage routing option. ``storage=tl.to_disk(path)``
        streams predicate-selected saves to a disk bundle during the
        forward pass. ``None`` preserves the existing in-RAM behavior.
    chunk_paths:
        Optional explicit tensor leaf paths to split when multiple
        batched tensor leaves are present.
    facets:
        Per-trace additive facet recipes captured into the immutable
        registry snapshot for the returned trace.
    narrate:
        Live narration (DOCUMENTED-UNSTABLE): ``True`` narrates
        every completed tensor-output event plus module structure lines;
        ``"modules"`` narrates structure only; a live selector scopes the
        stream; ``tl.options.NarrationOptions`` groups the full surface (stats
        levels, sink, crash tail). Narration is a display of capture
        events -- independent of ``save=`` retention. Torch-only.
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
        Never accepted -- unknown keywords raise ``trace_forward_kwargs_unrouted``,
        which says to pass forward keywords through ``input_kwargs=``.

    Postfunc behavior:
        ``CaptureOptions.activation_transform`` and ``grad_transform`` both take a tensor, should return a
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

    Examples
    --------
    >>> trace = tl.trace(model, x, save="all")
    >>> trace.layer_labels
    ['input_1', 'conv2d_1_1', 'relu_1_2', 'conv2d_2_3', 'output_1']
    >>> trace["relu_1_2"].out.shape
    torch.Size([1, 4, 8, 8])
    """
    with zero_fire_policy_scope(None if on_zero_fire is MISSING else on_zero_fire):
        _reject_extra_positional_input(extra_positional)
        reject_unrouted_forward_kwargs(forward_kwargs, _TRACE_OPTION_CLASSES)
        del forward_kwargs  # must never ride the ladder's locals() snapshot below
        _reject_non_mapping_input_kwargs(input_kwargs)
        model, intervene = _model_door.resolve_trace_operands(model, intervene)
        if input_shape is not None or inputs is None:
            model = _reject_non_module_ladder_root(model)
            # Quickstart input ladder: declared/inferred rungs
            # resolve BEFORE any capture machinery, then re-enter trace() as gold.
            ladder_kwargs = locals().copy()
            for consumed in (
                "model",
                "inputs",
                "extra_positional",
                "input_kwargs",
                "input_shape",
            ):
                ladder_kwargs.pop(consumed)
            from ._quickstart._ladder import _trace_via_ladder

            return _trace_via_ladder(model, inputs, input_kwargs, input_shape, ladder_kwargs)
        _refuse_compiled_callable_root(model)
        if save is MISSING:
            save = None
        elif save is None:
            capture = merge_capture_options(capture=capture, _layers_to_save=None)
        if _is_save_label_selection(save):
            save, capture = _route_save_label_selection(
                save, capture, merge_capture_options(capture=capture), model
            )
        public_trace_kwargs = locals().copy()
        # The capture pipeline's internal keyword bundle keeps the slot under ``input_args``.
        public_trace_kwargs["input_args"] = public_trace_kwargs.pop("inputs")
        # Consumed above (refused, routed, ladder-consumed or settled); never backend options.
        for consumed in ("extra_positional", "backend", "input_shape", "on_zero_fire"):
            public_trace_kwargs.pop(consumed)
        # The capture knobs that used to be door keywords travel inside
        # ``capture=``; the backend entries still take them flat, so the door
        # unfolds them here (explicit when the caller set the field, else MISSING
        # or the door's former default) and the backend kwarg bundle keeps its
        # shape. ``halt=`` is gone: ``stop_after=`` is the one stop directive.
        flat_capture_values = _unfold_door_capture_fields(capture)
        public_trace_kwargs.update(flat_capture_values)
        public_trace_kwargs["halt"] = None
        # grouping is validated as an entry knob, never a backend option.
        grouping = public_trace_kwargs.pop("grouping")
        _refuse_trace_entry_shapes(
            grouping=grouping,
            narrate=narrate,
            chunk_size=flat_capture_values["chunk_size"],
            chunk_paths=chunk_paths,
            backend=backend,
            jax_static_argnums=flat_capture_values["jax_static_argnums"],
            grad_options=flat_capture_values["grad_options"],
        )
        return _dispatch_trace(
            public_trace_kwargs, backend=backend, grouping=grouping, on_zero_fire=on_zero_fire
        )


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
        input_kwargs=input_kwargs,
        capture=CaptureOptions(
            compute_input_output_distances=True,
        ),
        save=None,
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

    Examples
    --------
    >>> import pickle
    >>> net = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    >>> _ = tl.trace(net, torch.randn(1, 2))
    >>> tl.release_model(net)
    >>> len(pickle.dumps(net)) > 0
    True
    """
    model = _model_door.resolve_released_model(model)
    _public_impls_module().release_model(model)


def summary(*args: Any, **kwargs: Any) -> str:
    """Run a one-call capture and return the rendered summary string.

    ``tl.summary(model, x)`` is the one-call entry point: it runs a
    metadata-only capture and RETURNS the rendered summary text (it never
    auto-prints). ``trace.summary()`` reports an existing capture instead.

    Examples
    --------
    >>> text = tl.summary(model, x)
    >>> isinstance(text, str), "conv2d" in text.lower()
    (True, True)
    """

    return cast(str, _public_impls_module().summary(*args, **kwargs))


def show_model_graph(*args: Any, **kwargs: Any) -> None:
    """Forward to the model-graph rendering implementation."""

    return cast(None, _public_impls_module().show_model_graph(*args, **kwargs))


def validate_forward_pass(*args: Any, **kwargs: Any) -> bool:
    """Forward to the backend-dispatched forward validation implementation."""

    return cast(bool, _public_impls_module().validate_forward_pass(*args, **kwargs))


def _validate_forward_pass_torch(*args: Any, **kwargs: Any) -> bool:
    """Forward to the torch forward validation implementation."""

    return cast(bool, _public_impls_module()._validate_forward_pass_torch(*args, **kwargs))


def validate_backward_pass(*args: Any, **kwargs: Any) -> bool:
    """Forward to the backward validation implementation."""

    return cast(bool, _public_impls_module().validate_backward_pass(*args, **kwargs))


_PUBLIC_IMPL_WRAPPER_NAMES = (
    "show_model_graph",
    "validate_forward_pass",
    "validate_backward_pass",
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
