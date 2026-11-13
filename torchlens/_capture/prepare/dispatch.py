"""Backend capability gating and the MLX door for the PREPARE stage.

Moved verbatim from ``torchlens._user_funcs`` by the capture seam.
"""

import collections.abc
from collections.abc import Callable
from typing import Any, Final, cast

from torchlens.backends.registry import BackendUnsupportedError

from ..._data_classes.trace import (
    Trace,
)
from ..._deprecations import MISSING, MissingType
from ..._input_coerce import _materialize_image_paths
from ..._literals import (
    OutputDeviceLiteral,
    _ActivationPostfunc,
    _GradientPostfunc,
)
from ..._postprocess._grouping_stamp import validate_grouping_knob
from ..._postprocess._selective_save import (
    apply_backend_post_plan,
    apply_static_label_save_policy,
    needs_post_save_predicate,
    reject_selector_outside_kinds,
    split_backend_post_options,
)
from ..._selection.selectors import BaseSelector
from ..._trace_selector_helpers import (
    _TRACE_OPTION_FILTERED_NAMES,
    _resolve_save_transforms,
    _split_save_options_and_predicate,
)
from ..._user_funcs_guards import (
    _refuse_non_torch_episode,
    _reject_auto_environment,
    _reject_compiled_plain_callable,
    _reject_unrouted_jax_options,
    _trace_option_refusal_code,
)
from ...backends._facet_recipes import attach_facet_recipes, pop_facet_recipes
from ...backends._options import (
    MLX_EXTRA_KWARG_POLICY,
    TRACE_OPTION_CAPABILITY_GATES,
    pop_stop_directive,
    reject_extra_trace_kwargs,
)
from ...backends._trace_options import _CONTROL_FLOW
from ...backends.mlx import MLXBackend
from ...backends.mlx._retention import RetentionRequest, retention_scope
from ...backends.registry import (
    BackendSpec,
    get_backend_spec,
    resolve_backend_spec,
)
from ...intervention.errors import ChunkedForwardConfigError
from ...options import (
    CaptureOptions,
    VisualizationOptions,
    _SaveTransforms,
    merge_capture_options,
)
from ...recording._options import PredicateFn

_MLX_STATIC_LABEL_SAVE_SELECTOR_KINDS = frozenset(
    {"label", "func", "module", "contains", "in_module", "and", "or", "not"}
)


def _trace_mlx_model(
    model: object,
    input_args: object,
    input_kwargs: dict[Any, Any] | None,
    *,
    layers_to_save: str | list[Any] | None | MissingType,
    transform: Callable[[Any], Any] | None | MissingType,
    save_raw_input: str | bool | MissingType,
    batch_render: str | MissingType,
    output_transform: Callable[[Any], Any] | None | MissingType,
    save_transformed_output: str | bool | MissingType,
    layer_visualizers: dict[Any, Callable[..., Any]] | None | MissingType,
    save_visualizations: bool | MissingType,
    keep_orphans: bool | MissingType,
    output_device: OutputDeviceLiteral | MissingType,
    activation_transform: _ActivationPostfunc | None | MissingType,
    grad_transform: _GradientPostfunc | None | MissingType,
    save_raw_activations: bool | MissingType,
    save_raw_grads: bool | MissingType,
    capture_tensor_grad_hooks: bool | MissingType,
    save_args: bool | MissingType,
    save_grads: bool | str | list[Any] | PredicateFn | BaseSelector | None | MissingType,
    save_source_context: bool | MissingType,
    save_rng: bool | MissingType,
    seed: int | None | MissingType,
    num_context_lines: int | MissingType,
    compute_input_output_distances: bool | MissingType,
    recurrence_detection: bool | MissingType,
    replayable: bool | MissingType,
    hooks: Any | None | MissingType,
    capture: CaptureOptions | None,
    save: _SaveTransforms | None,
    save_predicate: PredicateFn | BaseSelector | None,
    visualization: VisualizationOptions | None,
    backward_ready: bool | MissingType,
    name: str | None | MissingType,
    module_identity_mode: str | None | MissingType,
    grad_options: Any | None | MissingType,
    verbose: bool | MissingType,
    intervene: Any | None = None,
    halt: Any | None = None,
) -> Trace:
    """Dispatch an MLX module capture through the optional MLX backend.

    Parameters
    ----------
    model, input_args, input_kwargs:
        MLX model and forward inputs.

    Returns
    -------
    Trace
        Captured technical-preview MLX trace.
    """

    activation_fn = None if activation_transform is MISSING else activation_transform
    capture_options = merge_capture_options(
        capture=capture,
        _layers_to_save=layers_to_save,
        input_transform=transform,
        save_raw_input=save_raw_input,
        batch_render=batch_render,
        output_transform=output_transform,
        save_transformed_output=save_transformed_output,
        layer_visualizers=layer_visualizers,
        save_visualizations=save_visualizations,
        keep_orphans=keep_orphans,
        output_device=output_device,
        capture_tensor_grad_hooks=capture_tensor_grad_hooks,
        save_args=save_args,
        save_grads=save_grads,
        save_source_context=save_source_context,
        save_rng=save_rng,
        seed=seed,
        source_context_lines=MISSING,
        num_context_lines=num_context_lines,
        compute_input_output_distances=compute_input_output_distances,
        mark_layer_depths=MISSING,
        detach_saved_activations=MISSING,
        recurrence_detection=recurrence_detection,
        replayable=replayable,
        hooks=hooks,
        unwrap_when_done=MISSING,
        verbose=verbose,
        backward_ready=backward_ready,
        inference_only=MISSING,
        name=name,
        cache=MISSING,
        cache_dir=MISSING,
        module_identity_mode=module_identity_mode,
    )
    save_options = (
        save
        if save is not None
        else _resolve_save_transforms(
            capture_options,
            activation_transform=activation_fn,
            grad_transform=grad_transform,
            save_raw_activations=save_raw_activations,
            save_raw_grads=save_raw_grads,
        )
    )
    if capture_options.replayable:
        raise BackendUnsupportedError(
            "MLX backend does not support replayable=True: post-capture "
            "rerun readiness requires PyTorch autograd integration not present in "
            "MLX. Live static-label trace(intervene=tl.when(...)) IS supported; "
            "omit replayable or set False."
        )
    if capture_options.hooks:
        raise BackendUnsupportedError(
            "MLX backend does not support pre-attached hooks. "
            "Omit hooks or use the PyTorch backend."
        )
    if capture_options.save_grads:
        raise BackendUnsupportedError("backward capture is not supported on the mlx backend")
    raw_input = None
    model_input_args = input_args
    model_input_kwargs = input_kwargs
    if capture_options.input_transform is not None:
        raw_input = input_args
        transformed_input = capture_options.input_transform(input_args)
        # As on the torch door: the caller's ``input_kwargs`` survive the transform.
        if isinstance(transformed_input, collections.abc.Mapping):
            model_input_args = []
            model_input_kwargs = {**dict(transformed_input), **(input_kwargs or {})}
        else:
            model_input_args = transformed_input

    backend = MLXBackend()
    retention = _mlx_retention_request(save_predicate, capture_options)
    with retention_scope(retention):
        trace = backend.capture_trace(
            model,
            model_input_args,
            model_input_kwargs,
            save=capture_options._layers_to_save,
            keep_orphans=capture_options.keep_orphans,
            output_device=capture_options.output_device,
            activation_transform=save_options.activation_transform,
            save_raw_activations=save_options.save_raw_activations,
            detach_saved_activations=capture_options.detach_saved_activations,
            save_grads=capture_options.save_grads,
            seed=capture_options.seed,
            num_context_lines=capture_options.source_context_lines,
            save_args=capture_options.save_args,
            save_source_context=capture_options.save_source_context,
            save_rng=capture_options.save_rng,
            recurrence_detection=capture_options.recurrence_detection,
            compute_input_output_distances=capture_options.compute_input_output_distances,
            verbose=capture_options.verbose,
            backward_ready=capture_options.backward_ready,
            name=capture_options.name,
            transform=None,
            raw_input=raw_input,
            save_raw_input=capture_options.save_raw_input,
            batch_render=capture_options.batch_render,
            output_transform=capture_options.output_transform,
            save_transformed_output=capture_options.save_transformed_output,
            layer_visualizers=cast(
                "dict[Any, Callable[..., Any]] | None", capture_options.layer_visualizers
            ),
            save_visualizations=capture_options.save_visualizations,
            module_identity_mode=capture_options.module_identity_mode,
            grad_options=cast("Any", None if grad_options is MISSING else grad_options),
            intervene=intervene,
            halt=halt,
        )
    if retention is None:
        apply_static_label_save_policy(trace, save_predicate, backend_name="MLX")
    return trace


def _mlx_retention_request(save_predicate: Any, capture_options: Any) -> RetentionRequest | None:
    """The capture-time retention request for an MLX capture, if any.

    Static-label save= and structure-only are decided at capture: only selected
    values stay in memory and the rest are spilled replay witnesses.
    """

    if bool(getattr(capture_options, "structure_only", False)):
        return RetentionRequest(structure_only=True)
    if isinstance(save_predicate, BaseSelector) and not needs_post_save_predicate(save_predicate):
        return RetentionRequest(selector=save_predicate)
    return None


def _trace_mlx_model_from_public_kwargs(**kwargs: Any) -> Trace:
    """Dispatch MLX capture from the public ``trace`` keyword bundle.

    Parameters
    ----------
    **kwargs:
        Public ``trace`` keyword bundle captured before torch-specific normalization.

    Returns
    -------
    Trace
        Captured MLX trace.
    """

    # Idempotent when the registry entry already resolved it; load-bearing for
    # direct/autoroute callers so the internal flat keys are honored, not dropped.
    # intervene=/halt= are DISPATCHED options on MLX (static-label live
    # interventions); facets= is snapshotted onto the finished trace below.
    # None of the three may flow through the capability-gated reject helper:
    # with interventions=True that would raise the never-dispatches
    # conformance error instead of running or refusing them properly.
    # ``capture`` itself is NOT checked here: it is a grouped
    # CaptureOptions object, almost never strictly None/MISSING in practice,
    # and MLX supports many of its fields (layers_to_save, output_device,
    # ...). Blanket-rejecting the whole object made every
    # ``capture=CaptureOptions(...)`` call fail with "does not support:
    # capture" regardless of content; the real per-field support/rejection
    # already happens downstream in ``_trace_mlx_model`` (e.g.
    # ``capture_options.replayable``, ``capture_options.hooks``).
    reject_extra_trace_kwargs(
        {
            "lookback": kwargs["lookback"],
            "lookback_payload_policy": kwargs["lookback_payload_policy"],
            "storage": kwargs["storage"],
            "inference_only": kwargs.get("inference_only", MISSING),
            "cache": kwargs.get("cache", MISSING),
            "on_nonfinite": (
                kwargs["capture"].on_nonfinite if kwargs.get("capture") is not None else "ignore"
            ),
            "payload_policy": kwargs.get("payload_policy", MISSING),
            "chunk_size": kwargs.get("chunk_size", MISSING),
            "chunk_paths": kwargs.get("chunk_paths", MISSING),
            "save_outs_to": kwargs.get("save_outs_to", MISSING),
            "keep_outs_in_memory": kwargs.get("keep_outs_in_memory", MISSING),
            "out_sink": kwargs.get("out_sink", MISSING),
            "cache_dir": kwargs.get("cache_dir", MISSING),
            "save_mode": kwargs.get("save_mode", MISSING),
            "capture_tensor_grad_hooks": kwargs.get("capture_tensor_grad_hooks", MISSING),
            "save_raw_gradients": kwargs.get("save_raw_gradients", MISSING),
            "source_context_lines": kwargs.get("source_context_lines", MISSING),
            "unwrap_when_done": kwargs.get("unwrap_when_done", MISSING),
            "reconstruction_ready": kwargs.get("reconstruction_ready", MISSING),
        },
        MLX_EXTRA_KWARG_POLICY,
        spec=get_backend_spec("mlx"),
    )
    save_options, save_predicate = _split_save_options_and_predicate(kwargs["save"])
    if save_predicate is not None:
        reject_selector_outside_kinds(
            save_predicate,
            allowed=_MLX_STATIC_LABEL_SAVE_SELECTOR_KINDS,
            backend_name="MLX",
        )
    facet_recipes = pop_facet_recipes(dict(kwargs))
    # N5 fix: every name below except the genuine top-level ``trace()``
    # parameters (model, input_args, input_kwargs, capture, save,
    # grad_transform, grad_options, intervene, halt) moved into the grouped
    # ``capture=CaptureOptions(...)`` object and no longer reaches this
    # function as a flat key -- ``kwargs["activation_transform"]`` and
    # friends raised ``KeyError`` on every call once the sprint removed the
    # flat spelling from ``trace()``'s own signature. ``.get(..., MISSING)``
    # reports "not flatly specified" so ``_trace_mlx_model``'s own
    # ``merge_capture_options``/``_resolve_save_transforms`` calls fall through to
    # the grouped object's value, exactly like direct/internal callers who
    # still pass the flat spelling.
    activation_transform = kwargs.get("activation_transform", MISSING)
    trace = _trace_mlx_model(
        kwargs["model"],
        kwargs["input_args"],
        kwargs["input_kwargs"],
        layers_to_save=kwargs.get("layers_to_save", MISSING),
        transform=kwargs.get("transform", MISSING),
        save_raw_input=kwargs.get("save_raw_input", MISSING),
        batch_render=kwargs.get("batch_render", MISSING),
        output_transform=kwargs.get("output_transform", MISSING),
        save_transformed_output=kwargs.get("save_transformed_output", MISSING),
        layer_visualizers=MISSING,
        save_visualizations=MISSING,
        keep_orphans=kwargs.get("keep_orphans", MISSING),
        output_device=kwargs.get("output_device", MISSING),
        activation_transform=activation_transform,
        grad_transform=kwargs["grad_transform"],
        save_raw_activations=kwargs.get("save_raw_activations", MISSING),
        save_raw_grads=kwargs.get("save_raw_gradients", MISSING),
        capture_tensor_grad_hooks=kwargs.get("capture_tensor_grad_hooks", MISSING),
        save_args=kwargs.get("save_arg_values", MISSING),
        save_grads=kwargs.get("save_grads", MISSING),
        save_source_context=kwargs.get("save_code_context", MISSING),
        save_rng=kwargs.get("save_rng_states", MISSING),
        seed=kwargs.get("random_seed", MISSING),
        num_context_lines=kwargs.get("num_context_lines", MISSING),
        compute_input_output_distances=kwargs.get("compute_input_output_distances", MISSING),
        recurrence_detection=kwargs.get("recurrence_detection", MISSING),
        replayable=kwargs.get("replayable", MISSING),
        hooks=kwargs.get("hooks", MISSING),
        capture=kwargs["capture"],
        save=save_options,
        save_predicate=save_predicate,
        visualization=None,
        backward_ready=kwargs.get("backward_ready", MISSING),
        name=kwargs.get("name", MISSING),
        module_identity_mode=kwargs.get("module_identity_mode", MISSING),
        grad_options=kwargs["grad_options"],
        verbose=kwargs.get("verbose", MISSING),
        intervene=kwargs["intervene"],
        halt=pop_stop_directive(dict(kwargs)),
    )
    attach_facet_recipes(trace, facet_recipes, backend_name="MLX")
    return trace


def _trace_option_explicit(option_name: str, public_trace_kwargs: dict[str, Any]) -> bool:
    """Return whether a public trace option was explicitly supplied.

    Parameters
    ----------
    option_name:
        Public trace option name.
    public_trace_kwargs:
        Mutable public trace keyword bundle.

    Returns
    -------
    bool
        ``True`` when the flat option or grouped ``CaptureOptions`` field was
        explicitly supplied by the caller.
    """

    flat_value = public_trace_kwargs.get(option_name, MISSING)
    if flat_value is not MISSING:
        return True
    capture_options = public_trace_kwargs.get("capture")
    if isinstance(capture_options, CaptureOptions) and hasattr(capture_options, option_name):
        return capture_options._is_field_explicit(option_name)
    return False


def _unsupported_trace_option_message(option_name: str, backend_name: str) -> str:
    """Return an actionable unsupported-option message.

    Parameters
    ----------
    option_name:
        Public trace option name.
    backend_name:
        Resolved backend name.

    Returns
    -------
    str
        Error message for unsupported explicit option use.
    """

    if option_name == "jax_static_argnums":
        return "jax_static_argnums is only supported with backend='jax'."
    if option_name == "grad_options":
        return (
            "grad_options is only supported with backend='jax', backend='mlx', "
            "backend='tinygrad', backend='paddle', or backend='tf'."
        )
    if option_name in {"jax_control_flow", "jax_max_control_flow_unroll"}:
        return (
            f"backend={backend_name!r} does not yet support {option_name}. "
            "JAX control-flow unrolling is declared but not implemented in this backend phase; "
            "omit the option or use backend='torch'."
        )
    if option_name == "module_identity_mode":
        return (
            f"backend={backend_name!r} does not yet support module_identity_mode selection. "
            "Module-mode selection is declared but not implemented for this backend phase; "
            "omit the option or use backend='torch'."
        )
    if option_name == "payload_policy":
        return (
            f"backend={backend_name!r} does not yet support payload_policy. "
            "Non-torch payload codec policy is declared but not implemented in this backend "
            "phase; omit the option or use backend='torch'."
        )
    return (
        f"backend={backend_name!r} does not support trace option {option_name!r}; "
        "omit the option or choose a backend that declares it."
    )


def _filter_trace_kwargs_for_backend(
    public_trace_kwargs: dict[str, Any],
    resolved_spec: BackendSpec,
) -> None:
    """Strip unsupported omitted options and reject unsupported explicit ones.

    Parameters
    ----------
    public_trace_kwargs:
        Mutable public trace keyword bundle passed to the backend entry.
    resolved_spec:
        Backend selected for this trace call.

    Returns
    -------
    None
        ``public_trace_kwargs`` is updated in place.
    """

    supported_trace_options = set(resolved_spec.capabilities.trace_options)
    backend_name = str(resolved_spec.name)
    for option_name in _TRACE_OPTION_FILTERED_NAMES:
        if option_name in supported_trace_options:
            continue
        if _trace_option_explicit(option_name, public_trace_kwargs):
            raise BackendUnsupportedError(
                _unsupported_trace_option_message(option_name, backend_name),
                code=_trace_option_refusal_code(option_name),
                backend=backend_name,
                option=option_name,
            )
        public_trace_kwargs.pop(option_name, None)


def _capability_option_requested(value: Any) -> bool:
    """Return whether a gated public option value actually requests behavior.

    ``MISSING``, ``None``, and explicit ``False`` are all "off"; identity
    checks avoid calling ``bool()`` on predicate/selector values.
    """

    return not (value is MISSING or value is None or value is False)


def _enforce_capability_option_gates(
    public_trace_kwargs: dict[str, Any],
    resolved_spec: BackendSpec,
) -> None:
    """Refuse gated public options whose owning capability flag is ``False``.

    The registered capability table is the load-bearing authority in BOTH
    directions for EVERY backend, torch included: a flag flipped to ``False``
    must refuse the corresponding public surface typed instead of silently
    running (or silently ignoring) it. Options settable through
    ``capture=CaptureOptions(...)`` are gated on their explicit fields too.

    Parameters
    ----------
    public_trace_kwargs:
        Public trace keyword bundle about to be dispatched.
    resolved_spec:
        Backend spec selected for this trace call.

    Returns
    -------
    None
        Returns when every requested gated option's flag is ``True``.
    """

    capture_value = public_trace_kwargs.get("capture")
    for option_name, flag in TRACE_OPTION_CAPABILITY_GATES.items():
        if getattr(resolved_spec.capabilities, flag):
            continue
        requested = _capability_option_requested(public_trace_kwargs.get(option_name, MISSING))
        if (
            not requested
            and isinstance(capture_value, CaptureOptions)
            and hasattr(capture_value, option_name)
            and capture_value._is_field_explicit(option_name)
        ):
            requested = _capability_option_requested(getattr(capture_value, option_name))
        if requested:
            from torchlens.backends._backend_protocol import refusal_code_for_capability

            raise BackendUnsupportedError(
                f"backend {str(resolved_spec.name)!r} declares capability "
                f"{flag}=False, so explicit {option_name} is refused instead of "
                "silently ignored or run unsupported. Use a backend whose "
                f"registered capability table declares {flag}=True (e.g. "
                "backend='torch').",
                code=refusal_code_for_capability(flag),
                backend=str(resolved_spec.name),
                option=option_name,
            )


def _reject_unsupported_torch_trace_option_values(capture_options: CaptureOptions) -> None:
    """Reject explicit torch trace-option values that torch does not implement.

    Parameters
    ----------
    capture_options:
        Normalized capture options for a torch trace call.

    Returns
    -------
    None
        Returns when all explicit values are supported or default-equivalent.
    """

    if capture_options._is_field_explicit("module_identity_mode") and (
        capture_options.module_identity_mode not in {None, "torch_module"}
    ):
        raise BackendUnsupportedError(
            "backend='torch' supports module_identity_mode=None or 'torch_module' only."
        )
    if capture_options._is_field_explicit("payload_policy") and (
        capture_options.payload_policy not in {None, "full"}
    ):
        raise BackendUnsupportedError(
            "backend='torch' supports payload_policy=None or 'full' only."
        )
    for option_name in _CONTROL_FLOW.trace_options:
        if capture_options._is_field_explicit(option_name):
            raise BackendUnsupportedError(
                f"backend='torch' does not support explicit {option_name}; "
                "JAX control-flow options are only meaningful with backend='jax'."
            )


_AUTOROUTE_FORWARDED_KWARGS = (
    "input_kwargs",
    "capture",
    "save",
    "intervene",
    "stop_after",
    "storage",
    "facets",
    "episode",
    "narrate",
    "on_zero_fire",
)

#: Capture fields the doors used to take as flat keywords, with the flat default
#: the backend entries still expect when the field was not set explicitly.
_DOOR_CAPTURE_FLAT_DEFAULTS: Final[dict[str, Any]] = {
    "grad_transform": MISSING,
    "save_mode": MISSING,
    "reconstruction_ready": MISSING,
    "lookback": 0,
    "lookback_payload_policy": "metadata_only",
    "grouping": MISSING,
    "jax_static_argnums": MISSING,
    "grad_options": MISSING,
    "chunk_size": MISSING,
}


def _unfold_door_capture_fields(capture: CaptureOptions | None) -> dict[str, Any]:
    """Return the former door keywords as flat values read from ``capture=``.

    Only a ``CaptureOptions`` is probed: another options object passed as
    ``capture=`` (for example a ``ReplayOptions``) contributes no field here and
    reaches the door's typed refusal unchanged.
    """

    unfolded: dict[str, Any] = {}
    for field_name, flat_default in _DOOR_CAPTURE_FLAT_DEFAULTS.items():
        if isinstance(capture, CaptureOptions) and capture._is_field_explicit(field_name):
            unfolded[field_name] = getattr(capture, field_name)
        else:
            unfolded[field_name] = flat_default
    return unfolded


def _refuse_compiled_callable_root(model: Any) -> None:
    """A ``torch.compile``-produced plain callable is not a capturable root."""

    _reject_compiled_plain_callable(model)


def _refuse_trace_entry_shapes(
    *,
    grouping: Any,
    narrate: Any,
    chunk_size: Any,
    chunk_paths: Any,
    backend: Any,
    jax_static_argnums: Any,
    grad_options: Any,
) -> None:
    """Entry-time refusals that fire before any backend or capture work."""

    if grouping is not MISSING:
        # grouping= (UNSTABLE, keyword-only): closed-vocabulary knob.
        # Only "structural" (today's grouping, the default) is entry-legal;
        # "strict_shapes" waits on its own reviewed design and "fold_sites" on
        # an affirmative ruling (the flip PR) / the episode capture kind.
        validate_grouping_knob(grouping)
    if narrate is not None and narrate is not False:
        # Entry-time echo refusals (bad spellings, finalized-label selectors)
        # fire BEFORE any capture work on every backend path.
        from ..._snoop import normalize_echo

        normalize_echo(narrate)
    if chunk_paths is not MISSING and chunk_paths is not None and chunk_size in (MISSING, None):
        raise ChunkedForwardConfigError("chunk_paths requires chunk_size.")
    _reject_unrouted_jax_options(backend, jax_static_argnums, grad_options)


def _autoroute_trace(model: Any, input_args: Any, forwarded: dict[str, Any]) -> Trace | None:
    """Offer the call to the autoroute input detectors; ``None`` when none claims it."""

    from ... import _autoroute as autoroute

    # Only the surviving public kwargs are forwarded: capture knobs travel
    # inside the grouped ``capture=`` object, so re-entrant trace() calls
    # inside detectors receive exactly what the user could have passed.
    autoroute_kwargs = {name: forwarded[name] for name in _AUTOROUTE_FORWARDED_KWARGS}
    # Label/None selections are already carried privately by capture.
    # Omitting the consumed save value preserves that receipt on re-entry.
    if autoroute_kwargs["save"] is None:
        autoroute_kwargs.pop("save")
    for detector in autoroute.input.iter_by_priority():
        result = detector(model, input_args, **autoroute_kwargs)
        if result is not None:
            return cast("Trace", result)
    return None


def _dispatch_trace(
    public_trace_kwargs: dict[str, Any], *, backend: Any, grouping: Any, on_zero_fire: Any
) -> Trace:
    """Route a refused-clean ``trace`` call to its backend and run it.

    ``public_trace_kwargs`` is the facade's surviving keyword set (``backend``,
    ``input_shape``, ``on_zero_fire`` and ``grouping`` already popped); the
    backend filters mutate it in place before ``capture_trace`` receives it.
    """

    model = public_trace_kwargs["model"]
    # A path to an image file is opened here, before routing, so it reaches
    # the image detectors as an image instead of the text path as a string.
    input_args = _materialize_image_paths(model, public_trace_kwargs["input_args"])
    public_trace_kwargs["input_args"] = input_args
    input_kwargs = public_trace_kwargs["input_kwargs"]
    chunk_size = public_trace_kwargs["chunk_size"]
    capture = public_trace_kwargs["capture"]
    narrate = public_trace_kwargs["narrate"]
    explicit_backend_spec = None
    if backend is not None:
        explicit_backend_spec = get_backend_spec(str(backend))
        _filter_trace_kwargs_for_backend(public_trace_kwargs, explicit_backend_spec)
        explicit_backend_spec = resolve_backend_spec(backend, model, input_args, input_kwargs)
    if (
        backend is None
        and chunk_size in (MISSING, None)
        and not (
            isinstance(capture, CaptureOptions) and capture._is_field_explicit("input_transform")
        )
    ):
        forwarded = {**public_trace_kwargs, "grouping": grouping, "on_zero_fire": on_zero_fire}
        routed = _autoroute_trace(model, input_args, forwarded)
        if routed is not None:
            return routed
    _reject_auto_environment()
    resolved_spec = explicit_backend_spec or resolve_backend_spec(
        backend, model, input_args, input_kwargs
    )
    from ..._snoop._entry import refuse_echo_non_torch

    # Never a silent no-op: narrate= is torch-only in wave 1 (typed refusal).
    refuse_echo_non_torch(narrate, resolved_spec)
    _filter_trace_kwargs_for_backend(public_trace_kwargs, resolved_spec)
    _enforce_capability_option_gates(public_trace_kwargs, resolved_spec)
    _refuse_non_torch_episode(public_trace_kwargs, resolved_spec)
    # Non-torch: value-dependent/lookback save= and save shaping run on the finished capture.
    post_plan = split_backend_post_options(public_trace_kwargs, resolved_spec)
    trace = resolved_spec.capture_trace(**public_trace_kwargs)
    return cast("Trace", apply_backend_post_plan(trace, post_plan, resolved_spec))
