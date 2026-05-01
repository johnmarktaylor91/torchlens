"""Grouped option dataclasses for public TorchLens APIs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Final, Mapping, TypeVar, cast

import torch

from ._deprecations import MISSING, MissingType, warn_deprecated_alias
from ._literals import (
    BufferVisibilityLiteral,
    OutputDeviceLiteral,
    VisDirectionLiteral,
    VisInterventionModeLiteral,
    VisModeLiteral,
    VisNodeModeLiteral,
    VisNodePlacementLiteral,
    VisRendererLiteral,
)
from .visualization.node_spec import NodeSpec

if TYPE_CHECKING:
    from .data_classes.layer_log import LayerLog
    from .data_classes.module_log import ModuleLog

T = TypeVar("T")
ActivationPostfunc = Callable[[torch.Tensor], torch.Tensor]
GradientPostfunc = Callable[[torch.Tensor], torch.Tensor]

_CAPTURE_FIELDS: Final[tuple[str, ...]] = (
    "layers_to_save",
    "keep_unsaved_layers",
    "output_device",
    "save_function_args",
    "save_gradients",
    "gradients_to_save",
    "save_source_context",
    "save_rng_states",
    "random_seed",
    "source_context_lines",
    "optimizer",
    "compute_input_output_distances",
    "detach_saved_tensors",
    "detect_recurrent_patterns",
    "intervention_ready",
    "hooks",
    "unwrap_when_done",
    "verbose",
    "train_mode",
    "name",
    "cache",
    "cache_dir",
    "module_filter_fn",
    "stop_after",
    "emit_nvtx",
    "raise_on_nan",
)
_SAVE_FIELDS: Final[tuple[str, ...]] = (
    "output_dir",
    "activation_transform",
    "gradient_postfunc",
    "save_raw_activation",
    "save_raw_gradient",
    "save_level",
    "bundle_format",
)
_VISUALIZATION_FIELDS: Final[tuple[str, ...]] = (
    "view",
    "depth",
    "output_path",
    "save_only",
    "file_format",
    "show_buffers",
    "direction",
    "graph_overrides",
    "node_style",
    "node_spec_fn",
    "collapsed_node_spec_fn",
    "collapse_fn",
    "skip_fn",
    "edge_overrides",
    "gradient_edge_overrides",
    "module_overrides",
    "layout",
    "renderer",
    "theme",
    "intervention_mode",
    "show_cone",
    "node_overlay",
    "node_label_fields",
    "show_legend",
    "font_size",
    "dpi",
    "for_paper",
    "return_graph",
)
_REPLAY_FIELDS: Final[tuple[str, ...]] = (
    "strict",
    "hooks",
    "append",
    "chunk_size",
    "is_appended",
    "device_override",
)
_INTERVENTION_FIELDS: Final[tuple[str, ...]] = (
    "engine",
    "confirm_mutation",
    "strict",
    "helper_validation",
    "auto_promote",
    "cohort_migration",
    "error_severity_threshold",
)
_STREAMING_FIELDS: Final[tuple[str, ...]] = (
    "bundle_path",
    "retain_in_memory",
    "activation_callback",
)

_CAPTURE_FLAT_TO_GROUP: Final[dict[str, str]] = {
    "layers_to_save": "layers_to_save",
    "keep_unsaved_layers": "keep_unsaved_layers",
    "output_device": "output_device",
    "save_function_args": "save_function_args",
    "save_gradients": "save_gradients",
    "gradients_to_save": "gradients_to_save",
    "save_source_context": "save_source_context",
    "save_rng_states": "save_rng_states",
    "random_seed": "random_seed",
    "source_context_lines": "source_context_lines",
    "num_context_lines": "source_context_lines",
    "optimizer": "optimizer",
    "compute_input_output_distances": "compute_input_output_distances",
    "mark_input_output_distances": "compute_input_output_distances",
    "detach_saved_tensors": "detach_saved_tensors",
    "detect_recurrent_patterns": "detect_recurrent_patterns",
    "detect_loops": "detect_recurrent_patterns",
    "intervention_ready": "intervention_ready",
    "hooks": "hooks",
    "unwrap_when_done": "unwrap_when_done",
    "verbose": "verbose",
    "train_mode": "train_mode",
    "name": "name",
    "cache": "cache",
    "cache_dir": "cache_dir",
    "module_filter_fn": "module_filter_fn",
    "stop_after": "stop_after",
    "raise_on_nan": "raise_on_nan",
}
_SAVE_FLAT_TO_GROUP: Final[dict[str, str]] = {
    "activation_postfunc": "activation_transform",
    "activation_transform": "activation_transform",
    "gradient_postfunc": "gradient_postfunc",
    "save_raw_activation": "save_raw_activation",
    "save_raw_gradient": "save_raw_gradient",
}
_VISUALIZATION_FLAT_TO_GROUP: Final[dict[str, str]] = {
    "view": "view",
    "vis_mode": "view",
    "depth": "depth",
    "vis_nesting_depth": "depth",
    "vis_outpath": "output_path",
    "vis_save_only": "save_only",
    "vis_fileformat": "file_format",
    "vis_buffer_layers": "show_buffers",
    "vis_direction": "direction",
    "vis_graph_overrides": "graph_overrides",
    "node_style": "node_style",
    "vis_node_mode": "node_style",
    "vis_edge_overrides": "edge_overrides",
    "vis_gradient_edge_overrides": "gradient_edge_overrides",
    "vis_module_overrides": "module_overrides",
    "layout": "layout",
    "vis_node_placement": "layout",
    "renderer": "renderer",
    "vis_renderer": "renderer",
    "vis_theme": "theme",
    "vis_intervention_mode": "intervention_mode",
    "vis_show_cone": "show_cone",
}
_VISUALIZATION_DEPRECATED_FLAT: Final[set[str]] = {
    "vis_mode",
    "vis_nesting_depth",
    "vis_outpath",
    "vis_save_only",
    "vis_fileformat",
    "vis_buffer_layers",
    "vis_direction",
    "vis_graph_overrides",
    "vis_node_mode",
    "vis_edge_overrides",
    "vis_gradient_edge_overrides",
    "vis_module_overrides",
    "vis_node_placement",
    "vis_renderer",
    "vis_theme",
    "vis_intervention_mode",
    "vis_show_cone",
}
_REPLAY_FLAT_TO_GROUP: Final[dict[str, str]] = {
    "strict": "strict",
    "hooks": "hooks",
    "append": "append",
}
_INTERVENTION_FLAT_TO_GROUP: Final[dict[str, str]] = {
    "engine": "engine",
    "confirm_mutation": "confirm_mutation",
    "strict": "strict",
}
_STREAMING_FLAT_TO_GROUP: Final[dict[str, str]] = {
    "bundle_path": "bundle_path",
    "save_activations_to": "bundle_path",
    "retain_in_memory": "retain_in_memory",
    "keep_activations_in_memory": "retain_in_memory",
    "activation_callback": "activation_callback",
    "activation_sink": "activation_callback",
}


class _MutateWarningSuppression:
    """Session-level toggle and context manager for mutation warnings."""

    def __init__(self) -> None:
        """Initialize the suppression flag."""

        self._suppress = False
        self._prior = False

    def __call__(self, on: bool = True) -> "_MutateWarningSuppression":
        """Set suppression state and return this context-capable object.

        Parameters
        ----------
        on:
            Whether mutate-in-place warnings should be suppressed.

        Returns
        -------
        _MutateWarningSuppression
            This suppression controller.
        """

        self._suppress = bool(on)
        return self

    def __enter__(self) -> "_MutateWarningSuppression":
        """Temporarily suppress mutate-in-place warnings.

        Returns
        -------
        _MutateWarningSuppression
            This suppression controller.
        """

        self._prior = self._suppress
        self._suppress = True
        return self

    def __exit__(self, *exc: object) -> None:
        """Restore the suppression state active before the context.

        Parameters
        ----------
        *exc:
            Exception triple supplied by the context manager protocol.
        """

        self._suppress = self._prior

    @property
    def is_suppressed(self) -> bool:
        """Whether mutate-in-place warnings are currently suppressed."""

        return self._suppress


suppress_mutate_warnings = _MutateWarningSuppression()


def _resolve_option_value(
    field_name: str,
    supplied_value: Any,
    default_value: Any,
    specified_fields: set[str],
) -> Any:
    """Resolve an option field while tracking explicit caller presence.

    Parameters
    ----------
    field_name:
        Dataclass field name being resolved.
    supplied_value:
        Value supplied by the caller, or ``MISSING``.
    default_value:
        Public default for the field.
    specified_fields:
        Mutable set populated with fields explicitly supplied by the caller.

    Returns
    -------
    Any
        Resolved field value.
    """

    if supplied_value is MISSING:
        return default_value
    specified_fields.add(field_name)
    return supplied_value


def _validate_node_style(node_style: VisNodeModeLiteral) -> None:
    """Validate a visualization node-style preset name.

    Parameters
    ----------
    node_style:
        Candidate node-style preset name.

    Raises
    ------
    ValueError
        If ``node_style`` is not a registered public preset.
    """

    if node_style not in {"default", "profiling", "vision", "attention"}:
        raise ValueError(
            "Visualization node_style/node_mode must be one of 'default', "
            "'profiling', 'vision', or 'attention'."
        )
    if node_style in {"vision", "attention"}:
        warnings.warn(
            f"node_style={node_style!r} is moving out of core; use the equivalent "
            f"recipe at examples/recipes/{node_style}.py or wait for the "
            f"torchlens.{node_style} plugin",
            DeprecationWarning,
            stacklevel=3,
        )


def _validate_intervention_mode(intervention_mode: VisInterventionModeLiteral) -> None:
    """Validate an intervention visualization mode name.

    Parameters
    ----------
    intervention_mode:
        Candidate intervention visualization mode.

    Raises
    ------
    ValueError
        If ``intervention_mode`` is not a supported mode.
    """

    if intervention_mode not in {"node_mark", "as_node"}:
        raise ValueError("vis_intervention_mode must be either 'node_mark' or 'as_node'.")


def _validate_buffer_visibility(value: BufferVisibilityLiteral | bool) -> None:
    """Validate buffer visibility options.

    Parameters
    ----------
    value:
        Buffer visibility mode. Legacy bools are still accepted: ``True`` maps
        to ``"always"`` and ``False`` maps to ``"never"``.

    Raises
    ------
    ValueError
        If ``value`` is not a supported tri-state mode.
    """

    if value is True:
        return
    if value is False:
        return
    if value in {"never", "meaningful", "always"}:
        return
    raise ValueError("Buffer visibility must be 'never', 'meaningful', 'always', or a bool.")


def _set_frozen_fields(
    instance: object, field_names: tuple[str, ...], values: Mapping[str, Any]
) -> None:
    """Set dataclass fields while constructing a frozen options object.

    Parameters
    ----------
    instance:
        Instance under construction.
    field_names:
        Ordered dataclass field names to populate.
    values:
        Resolved field values.
    """

    for field_name in field_names:
        object.__setattr__(instance, field_name, values[field_name])


def _explicit_fields(instance: object) -> frozenset[str]:
    """Return explicitly supplied option fields from an option object.

    Parameters
    ----------
    instance:
        Option object carrying ``_specified_fields``.

    Returns
    -------
    frozenset[str]
        Explicit field names.
    """

    return cast(frozenset[str], getattr(instance, "_specified_fields"))


def _field_is_explicit(instance: object, field_name: str) -> bool:
    """Return whether an option object field was explicitly supplied.

    Parameters
    ----------
    instance:
        Option object carrying ``_specified_fields``.
    field_name:
        Canonical field name to test.

    Returns
    -------
    bool
        Whether the field was explicit.
    """

    return field_name in _explicit_fields(instance)


def _merge_grouped_options(
    *,
    option: Any | None,
    option_factory: Callable[[], Any],
    fields: tuple[str, ...],
    flat_to_group: Mapping[str, str],
    flat_values: Mapping[str, Any],
    group_name: str,
    conflict_message: str | None,
    deprecated_flat_names: set[str] | None = None,
    warn_individual_kwargs: bool = True,
) -> Any:
    """Merge flat kwargs into a grouped options object.

    Parameters
    ----------
    option:
        Caller-supplied grouped options, if any.
    option_factory:
        Zero-argument constructor for defaults.
    fields:
        Canonical fields to preserve.
    flat_to_group:
        Mapping from flat kwarg names to canonical option field names.
    flat_values:
        Flat kwarg values or ``MISSING``.
    group_name:
        Public grouped option parameter name.
    conflict_message:
        Optional message for same-field grouped/flat conflicts.
    deprecated_flat_names:
        Flat names that should warn as renamed aliases. If ``None``, every flat
        name warns when supplied.
    warn_individual_kwargs:
        Whether canonical individual kwargs should warn when supplied.

    Returns
    -------
    Any
        New grouped option object with merged values.

    Raises
    ------
    ValueError
        If a field is supplied by both grouped and flat styles.
    """

    if option is None:
        values = option_factory().as_dict()
        specified_fields: frozenset[str] = frozenset()
    else:
        values = option.as_dict()
        specified_fields = _explicit_fields(option)

    for flat_name, group_field in flat_to_group.items():
        flat_value = flat_values.get(flat_name, MISSING)
        if flat_value is MISSING:
            continue
        if option is not None and group_field in specified_fields:
            if conflict_message is not None:
                raise ValueError(conflict_message)
            raise TypeError(f"Do not pass both `{flat_name}` and `{group_name}.{group_field}`.")
        should_warn = warn_individual_kwargs
        if deprecated_flat_names is not None:
            should_warn = flat_name in deprecated_flat_names
        if should_warn:
            warn_deprecated_alias(flat_name, f"{group_name}.{group_field}")
        values[group_field] = flat_value
        specified_fields = frozenset((*specified_fields, group_field))
    return option_factory().from_values(values, specified_fields)


@dataclass(frozen=True, init=False)
class CaptureOptions:
    """Grouped capture options for ``log_forward_pass``.

    Parameters
    ----------
    layers_to_save:
        Activation layer selector to capture.
    keep_unsaved_layers:
        Whether metadata-only layers remain in the returned log.
    output_device:
        Device placement for saved tensors.
    save_function_args:
        Whether non-tensor function arguments are captured.
    save_gradients:
        Whether backward gradients are captured.
    gradients_to_save:
        Gradient layer selector; defaults to the activation selector when explicitly enabled.
    save_source_context:
        Whether source-text context is captured in addition to source identity.
    save_rng_states:
        Whether operation-level RNG states are captured.
    random_seed:
        Fixed seed used for deterministic capture.
    source_context_lines:
        Number of source-context lines to store.
    optimizer:
        Optional optimizer used to annotate optimized parameters.
    compute_input_output_distances:
        Whether input/output graph distances are computed.
    detach_saved_tensors:
        Whether saved tensors are detached from autograd.
    detect_recurrent_patterns:
        Whether repeated graph patterns are detected during postprocess.
    intervention_ready:
        Whether replay-template metadata is captured for intervention APIs.
    hooks:
        Optional live hook plan applied during capture.
    unwrap_when_done:
        Whether Torch functions are unwrapped after this call.
    verbose:
        Whether progress messages are printed.
    train_mode:
        Whether capture keeps autograd-connected tensors for training workflows.
    name:
        Optional user-facing name for the returned log.
    cache:
        Whether to use the content-hash capture cache.
    cache_dir:
        Optional directory for content-hash cache entries.
    module_filter_fn:
        Optional predicate receiving a ``LayerPassLog`` after construction.
        Returning ``False`` keeps metadata but skips activation saving.
    stop_after:
        Experimental stop-early site. Only supported by ``torchlens.peek``.
    emit_nvtx:
        Placeholder toggle for future NVTX ranges; currently inert.
    raise_on_nan:
        Whether capture should stop at the first NaN or Inf tensor.

    Examples
    --------
    >>> opts = CaptureOptions(layers_to_save=["fc1"], random_seed=0)
    >>> opts.layers_to_save
    ['fc1']
    """

    layers_to_save: str | list[Any] | None = "all"
    keep_unsaved_layers: bool = True
    output_device: OutputDeviceLiteral = "same"
    save_function_args: bool = False
    save_gradients: bool = False
    gradients_to_save: str | list[Any] | None = "all"
    save_source_context: bool = False
    save_rng_states: bool = False
    random_seed: int | None = None
    source_context_lines: int = 7
    optimizer: Any = None
    compute_input_output_distances: bool = False
    detach_saved_tensors: bool = False
    detect_recurrent_patterns: bool = True
    intervention_ready: bool = False
    hooks: Any | None = None
    unwrap_when_done: bool = False
    verbose: bool = False
    train_mode: bool = False
    name: str | None = None
    cache: bool = False
    cache_dir: str | Path | None = None
    module_filter_fn: Callable[[Any], bool] | None = None
    stop_after: Any | None = None
    emit_nvtx: bool = False
    raise_on_nan: bool = False
    _specified_fields: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)

    def __init__(
        self,
        layers_to_save: str | list[Any] | None | MissingType = MISSING,
        keep_unsaved_layers: bool | MissingType = MISSING,
        output_device: OutputDeviceLiteral | MissingType = MISSING,
        save_function_args: bool | MissingType = MISSING,
        save_gradients: bool | MissingType = MISSING,
        gradients_to_save: str | list[Any] | None | MissingType = MISSING,
        save_source_context: bool | MissingType = MISSING,
        save_rng_states: bool | MissingType = MISSING,
        random_seed: int | None | MissingType = MISSING,
        source_context_lines: int | MissingType = MISSING,
        optimizer: Any | MissingType = MISSING,
        compute_input_output_distances: bool | MissingType = MISSING,
        detach_saved_tensors: bool | MissingType = MISSING,
        detect_recurrent_patterns: bool | MissingType = MISSING,
        intervention_ready: bool | MissingType = MISSING,
        hooks: Any | MissingType = MISSING,
        unwrap_when_done: bool | MissingType = MISSING,
        verbose: bool | MissingType = MISSING,
        train_mode: bool | MissingType = MISSING,
        name: str | None | MissingType = MISSING,
        cache: bool | MissingType = MISSING,
        cache_dir: str | Path | None | MissingType = MISSING,
        module_filter_fn: Callable[[Any], bool] | None | MissingType = MISSING,
        stop_after: Any | None | MissingType = MISSING,
        emit_nvtx: bool | MissingType = MISSING,
        raise_on_nan: bool | MissingType = MISSING,
        *,
        mark_input_output_distances: bool | MissingType = MISSING,
        num_context_lines: int | MissingType = MISSING,
        detect_loops: bool | MissingType = MISSING,
    ) -> None:
        """Initialize a frozen capture option bundle."""

        if mark_input_output_distances is not MISSING:
            if compute_input_output_distances is not MISSING:
                raise TypeError(
                    "kwarg mark_input_output_distances deprecated, use "
                    "compute_input_output_distances; do not pass both"
                )
            warn_deprecated_alias(
                "mark_input_output_distances", "capture.compute_input_output_distances"
            )
            compute_input_output_distances = mark_input_output_distances
        if num_context_lines is not MISSING:
            if source_context_lines is not MISSING:
                raise TypeError(
                    "kwarg num_context_lines deprecated, use source_context_lines; do not pass both"
                )
            warn_deprecated_alias("num_context_lines", "capture.source_context_lines")
            source_context_lines = num_context_lines
        if detect_loops is not MISSING:
            if detect_recurrent_patterns is not MISSING:
                raise TypeError(
                    "kwarg detect_loops deprecated, use detect_recurrent_patterns; do not pass both"
                )
            warn_deprecated_alias("detect_loops", "capture.detect_recurrent_patterns")
            detect_recurrent_patterns = detect_loops

        specified_fields: set[str] = set()
        values: dict[str, Any] = {
            "layers_to_save": _resolve_option_value(
                "layers_to_save", layers_to_save, "all", specified_fields
            ),
            "keep_unsaved_layers": _resolve_option_value(
                "keep_unsaved_layers", keep_unsaved_layers, True, specified_fields
            ),
            "output_device": _resolve_option_value(
                "output_device", output_device, "same", specified_fields
            ),
            "save_function_args": _resolve_option_value(
                "save_function_args", save_function_args, False, specified_fields
            ),
            "save_gradients": _resolve_option_value(
                "save_gradients", save_gradients, False, specified_fields
            ),
            "gradients_to_save": _resolve_option_value(
                "gradients_to_save", gradients_to_save, "all", specified_fields
            ),
            "save_source_context": _resolve_option_value(
                "save_source_context", save_source_context, False, specified_fields
            ),
            "save_rng_states": _resolve_option_value(
                "save_rng_states", save_rng_states, False, specified_fields
            ),
            "random_seed": _resolve_option_value(
                "random_seed", random_seed, None, specified_fields
            ),
            "source_context_lines": _resolve_option_value(
                "source_context_lines", source_context_lines, 7, specified_fields
            ),
            "optimizer": _resolve_option_value("optimizer", optimizer, None, specified_fields),
            "compute_input_output_distances": _resolve_option_value(
                "compute_input_output_distances",
                compute_input_output_distances,
                False,
                specified_fields,
            ),
            "detach_saved_tensors": _resolve_option_value(
                "detach_saved_tensors", detach_saved_tensors, False, specified_fields
            ),
            "detect_recurrent_patterns": _resolve_option_value(
                "detect_recurrent_patterns", detect_recurrent_patterns, True, specified_fields
            ),
            "intervention_ready": _resolve_option_value(
                "intervention_ready", intervention_ready, False, specified_fields
            ),
            "hooks": _resolve_option_value("hooks", hooks, None, specified_fields),
            "unwrap_when_done": _resolve_option_value(
                "unwrap_when_done", unwrap_when_done, False, specified_fields
            ),
            "verbose": _resolve_option_value("verbose", verbose, False, specified_fields),
            "train_mode": _resolve_option_value("train_mode", train_mode, False, specified_fields),
            "name": _resolve_option_value("name", name, None, specified_fields),
            "cache": _resolve_option_value("cache", cache, False, specified_fields),
            "cache_dir": _resolve_option_value("cache_dir", cache_dir, None, specified_fields),
            "module_filter_fn": _resolve_option_value(
                "module_filter_fn", module_filter_fn, None, specified_fields
            ),
            "stop_after": _resolve_option_value("stop_after", stop_after, None, specified_fields),
            "emit_nvtx": _resolve_option_value("emit_nvtx", emit_nvtx, False, specified_fields),
            "raise_on_nan": _resolve_option_value(
                "raise_on_nan", raise_on_nan, False, specified_fields
            ),
        }
        _set_frozen_fields(self, _CAPTURE_FIELDS, values)
        object.__setattr__(self, "_specified_fields", frozenset(specified_fields))

    def as_dict(self) -> dict[str, Any]:
        """Return the option values as a plain dictionary."""

        return {field_name: getattr(self, field_name) for field_name in _CAPTURE_FIELDS}

    def is_field_explicit(self, field_name: str) -> bool:
        """Return whether a field was explicitly supplied by the caller."""

        return _field_is_explicit(self, field_name)

    @classmethod
    def from_values(
        cls, values: Mapping[str, Any], specified_fields: frozenset[str]
    ) -> "CaptureOptions":
        """Build an instance from already-resolved field values."""

        instance = object.__new__(cls)
        _set_frozen_fields(instance, _CAPTURE_FIELDS, values)
        object.__setattr__(instance, "_specified_fields", specified_fields)
        return instance


@dataclass(frozen=True, init=False)
class SaveOptions:
    """Grouped activation-save options for ``log_forward_pass``.

    Parameters
    ----------
    output_dir:
        Future save target directory; currently inert during capture.
    activation_transform:
        Optional transform applied to each activation before storage.
    gradient_postfunc:
        Optional transform applied to each gradient before storage.
    save_raw_activation:
        Whether raw activations remain available when transformed.
    save_raw_gradient:
        Whether raw gradients remain available when transformed.
    save_level:
        Future portable-save level; currently inert during capture.
    bundle_format:
        Future save bundle format selector; currently inert during capture.

    Examples
    --------
    >>> opts = SaveOptions(activation_transform=lambda x: x.detach())
    >>> opts.save_raw_activation
    True
    """

    output_dir: str | Path | None = None
    activation_transform: ActivationPostfunc | None = None
    gradient_postfunc: GradientPostfunc | None = None
    save_raw_activation: bool = True
    save_raw_gradient: bool = True
    save_level: str | None = None
    bundle_format: str | None = None
    _specified_fields: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)

    def __init__(
        self,
        output_dir: str | Path | None | MissingType = MISSING,
        activation_transform: ActivationPostfunc | None | MissingType = MISSING,
        gradient_postfunc: GradientPostfunc | None | MissingType = MISSING,
        save_raw_activation: bool | MissingType = MISSING,
        save_raw_gradient: bool | MissingType = MISSING,
        save_level: str | None | MissingType = MISSING,
        bundle_format: str | None | MissingType = MISSING,
        *,
        activation_postfunc: ActivationPostfunc | None | MissingType = MISSING,
    ) -> None:
        """Initialize a frozen save option bundle."""

        if activation_postfunc is not MISSING:
            if activation_transform is not MISSING:
                raise TypeError(
                    "kwarg activation_postfunc deprecated, use activation_transform; "
                    "do not pass both"
                )
            warn_deprecated_alias("activation_postfunc", "save.activation_transform")
            activation_transform = activation_postfunc
        specified_fields: set[str] = set()
        values: dict[str, Any] = {
            "output_dir": _resolve_option_value("output_dir", output_dir, None, specified_fields),
            "activation_transform": _resolve_option_value(
                "activation_transform", activation_transform, None, specified_fields
            ),
            "gradient_postfunc": _resolve_option_value(
                "gradient_postfunc", gradient_postfunc, None, specified_fields
            ),
            "save_raw_activation": _resolve_option_value(
                "save_raw_activation", save_raw_activation, True, specified_fields
            ),
            "save_raw_gradient": _resolve_option_value(
                "save_raw_gradient", save_raw_gradient, True, specified_fields
            ),
            "save_level": _resolve_option_value("save_level", save_level, None, specified_fields),
            "bundle_format": _resolve_option_value(
                "bundle_format", bundle_format, None, specified_fields
            ),
        }
        _set_frozen_fields(self, _SAVE_FIELDS, values)
        object.__setattr__(self, "_specified_fields", frozenset(specified_fields))

    @property
    def activation_postfunc(self) -> ActivationPostfunc | None:
        """Deprecated alias for ``activation_transform``."""

        return self.activation_transform

    def as_dict(self) -> dict[str, Any]:
        """Return the option values as a plain dictionary."""

        return {field_name: getattr(self, field_name) for field_name in _SAVE_FIELDS}

    def is_field_explicit(self, field_name: str) -> bool:
        """Return whether a field was explicitly supplied by the caller."""

        return _field_is_explicit(self, field_name)

    @classmethod
    def from_values(
        cls, values: Mapping[str, Any], specified_fields: frozenset[str]
    ) -> "SaveOptions":
        """Build an instance from already-resolved field values."""

        instance = object.__new__(cls)
        _set_frozen_fields(instance, _SAVE_FIELDS, values)
        object.__setattr__(instance, "_specified_fields", specified_fields)
        return instance


@dataclass(frozen=True, init=False)
class VisualizationOptions:
    """Grouped visualization options for capture and graph-rendering APIs.

    Parameters
    ----------
    view:
        Graph view: ``"none"``, ``"rolled"``, or ``"unrolled"``.
    depth:
        Maximum module nesting depth shown in the graph.
    output_path:
        Output path stem for rendered graph files.
    save_only:
        Whether rendering saves without opening a viewer.
    file_format:
        Graph output file format.
    show_buffers:
        Buffer visibility policy.
    direction:
        Graph layout direction.
    graph_overrides:
        Graphviz graph-level style overrides.
    node_style:
        Built-in node label/style preset.
    node_spec_fn:
        Optional layer-node customization callback.
    collapsed_node_spec_fn:
        Optional collapsed module-node customization callback.
    collapse_fn:
        Optional module collapse predicate.
    skip_fn:
        Optional layer skip predicate.
    edge_overrides:
        Forward-edge style overrides.
    gradient_edge_overrides:
        Gradient-edge style overrides.
    module_overrides:
        Module cluster style overrides.
    layout:
        Layout engine selector. ``"elk"`` remains accepted as an internal
        backend escape hatch; public API callers should prefer ``"auto"``.
    renderer:
        Renderer backend selector.
    theme:
        Theme name.
    intervention_mode:
        Intervention overlay mode.
    show_cone:
        Whether intervention cones are highlighted.
    node_overlay:
        Built-in overlay name or external score mapping for graph nodes.
    node_label_fields:
        Optional explicit label row fields.
    show_legend:
        Whether to render the theme legend with the graph.
    font_size:
        Optional Graphviz font size.
    dpi:
        Optional Graphviz output DPI.
    for_paper:
        Convenience toggle forcing the paper theme preset.
    return_graph:
        Whether rendering returns the renderer object instead of DOT source.

    Examples
    --------
    >>> opts = VisualizationOptions(view="rolled", depth=2)
    >>> opts.layout
    'auto'
    """

    view: VisModeLiteral = "none"
    depth: int = 1000
    output_path: str = "graph.gv"
    save_only: bool = False
    file_format: str = "pdf"
    show_buffers: BufferVisibilityLiteral = "meaningful"
    direction: VisDirectionLiteral = "bottomup"
    graph_overrides: dict[str, Any] | None = None
    node_style: VisNodeModeLiteral = "default"
    node_spec_fn: Callable[["LayerLog", NodeSpec], NodeSpec | None] | None = None
    collapsed_node_spec_fn: Callable[["ModuleLog", NodeSpec], NodeSpec | None] | None = None
    collapse_fn: Callable[["ModuleLog"], bool] | None = None
    skip_fn: Callable[["LayerLog"], bool] | None = None
    edge_overrides: dict[str, Any] | None = None
    gradient_edge_overrides: dict[str, Any] | None = None
    module_overrides: dict[str, Any] | None = None
    layout: VisNodePlacementLiteral = "auto"
    renderer: VisRendererLiteral = "graphviz"
    theme: str = "torchlens"
    intervention_mode: VisInterventionModeLiteral = "node_mark"
    show_cone: bool = True
    node_overlay: str | Mapping[str, Any] | None = None
    node_label_fields: list[str] | None = None
    show_legend: bool = False
    font_size: int | None = None
    dpi: int | None = None
    for_paper: bool = False
    return_graph: bool = False
    _specified_fields: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)

    def __init__(
        self,
        view: VisModeLiteral | MissingType = MISSING,
        depth: int | MissingType = MISSING,
        output_path: str | MissingType = MISSING,
        save_only: bool | MissingType = MISSING,
        file_format: str | MissingType = MISSING,
        show_buffers: BufferVisibilityLiteral | bool | MissingType = MISSING,
        direction: VisDirectionLiteral | MissingType = MISSING,
        graph_overrides: dict[str, Any] | None | MissingType = MISSING,
        node_style: VisNodeModeLiteral | MissingType = MISSING,
        node_spec_fn: (
            Callable[["LayerLog", NodeSpec], NodeSpec | None] | None | MissingType
        ) = MISSING,
        collapsed_node_spec_fn: (
            Callable[["ModuleLog", NodeSpec], NodeSpec | None] | None | MissingType
        ) = MISSING,
        collapse_fn: Callable[["ModuleLog"], bool] | None | MissingType = MISSING,
        skip_fn: Callable[["LayerLog"], bool] | None | MissingType = MISSING,
        edge_overrides: dict[str, Any] | None | MissingType = MISSING,
        gradient_edge_overrides: dict[str, Any] | None | MissingType = MISSING,
        module_overrides: dict[str, Any] | None | MissingType = MISSING,
        layout: VisNodePlacementLiteral | MissingType = MISSING,
        renderer: VisRendererLiteral | MissingType = MISSING,
        theme: str | MissingType = MISSING,
        intervention_mode: VisInterventionModeLiteral | MissingType = MISSING,
        show_cone: bool | MissingType = MISSING,
        node_overlay: str | Mapping[str, Any] | None | MissingType = MISSING,
        node_label_fields: list[str] | None | MissingType = MISSING,
        show_legend: bool | MissingType = MISSING,
        font_size: int | None | MissingType = MISSING,
        dpi: int | None | MissingType = MISSING,
        for_paper: bool | MissingType = MISSING,
        return_graph: bool | MissingType = MISSING,
        *,
        mode: VisModeLiteral | MissingType = MISSING,
        max_module_depth: int | MissingType = MISSING,
        layout_engine: VisNodePlacementLiteral | MissingType = MISSING,
        node_mode: VisNodeModeLiteral | MissingType = MISSING,
    ) -> None:
        """Initialize a frozen visualization option bundle."""

        if mode is not MISSING:
            if view is not MISSING:
                raise TypeError("kwarg mode deprecated, use view; do not pass both")
            warn_deprecated_alias("mode", "visualization.view")
            view = mode
        if max_module_depth is not MISSING:
            if depth is not MISSING:
                raise TypeError("kwarg max_module_depth deprecated, use depth; do not pass both")
            warn_deprecated_alias("max_module_depth", "visualization.depth")
            depth = max_module_depth
        if layout_engine is not MISSING:
            if layout is not MISSING:
                raise TypeError("kwarg layout_engine deprecated, use layout; do not pass both")
            warn_deprecated_alias("layout_engine", "visualization.layout")
            layout = layout_engine
        if node_mode is not MISSING:
            if node_style is not MISSING:
                raise TypeError("kwarg node_mode deprecated, use node_style; do not pass both")
            warn_deprecated_alias("node_mode", "visualization.node_style")
            node_style = node_mode

        specified_fields: set[str] = set()
        values: dict[str, Any] = {
            "view": _resolve_option_value("view", view, "none", specified_fields),
            "depth": _resolve_option_value("depth", depth, 1000, specified_fields),
            "output_path": _resolve_option_value(
                "output_path", output_path, "graph.gv", specified_fields
            ),
            "save_only": _resolve_option_value("save_only", save_only, False, specified_fields),
            "file_format": _resolve_option_value(
                "file_format", file_format, "pdf", specified_fields
            ),
            "show_buffers": _resolve_option_value(
                "show_buffers", show_buffers, "meaningful", specified_fields
            ),
            "direction": _resolve_option_value(
                "direction", direction, "bottomup", specified_fields
            ),
            "graph_overrides": _resolve_option_value(
                "graph_overrides", graph_overrides, None, specified_fields
            ),
            "node_style": _resolve_option_value(
                "node_style", node_style, "default", specified_fields
            ),
            "node_spec_fn": _resolve_option_value(
                "node_spec_fn", node_spec_fn, None, specified_fields
            ),
            "collapsed_node_spec_fn": _resolve_option_value(
                "collapsed_node_spec_fn", collapsed_node_spec_fn, None, specified_fields
            ),
            "collapse_fn": _resolve_option_value(
                "collapse_fn", collapse_fn, None, specified_fields
            ),
            "skip_fn": _resolve_option_value("skip_fn", skip_fn, None, specified_fields),
            "edge_overrides": _resolve_option_value(
                "edge_overrides", edge_overrides, None, specified_fields
            ),
            "gradient_edge_overrides": _resolve_option_value(
                "gradient_edge_overrides", gradient_edge_overrides, None, specified_fields
            ),
            "module_overrides": _resolve_option_value(
                "module_overrides", module_overrides, None, specified_fields
            ),
            "layout": _resolve_option_value("layout", layout, "auto", specified_fields),
            "renderer": _resolve_option_value("renderer", renderer, "graphviz", specified_fields),
            "theme": _resolve_option_value("theme", theme, "torchlens", specified_fields),
            "intervention_mode": _resolve_option_value(
                "intervention_mode", intervention_mode, "node_mark", specified_fields
            ),
            "show_cone": _resolve_option_value("show_cone", show_cone, True, specified_fields),
            "node_overlay": _resolve_option_value(
                "node_overlay", node_overlay, None, specified_fields
            ),
            "node_label_fields": _resolve_option_value(
                "node_label_fields", node_label_fields, None, specified_fields
            ),
            "show_legend": _resolve_option_value(
                "show_legend", show_legend, False, specified_fields
            ),
            "font_size": _resolve_option_value("font_size", font_size, None, specified_fields),
            "dpi": _resolve_option_value("dpi", dpi, None, specified_fields),
            "for_paper": _resolve_option_value("for_paper", for_paper, False, specified_fields),
            "return_graph": _resolve_option_value(
                "return_graph", return_graph, False, specified_fields
            ),
        }
        _validate_buffer_visibility(values["show_buffers"])
        _validate_node_style(cast(VisNodeModeLiteral, values["node_style"]))
        _validate_intervention_mode(cast(VisInterventionModeLiteral, values["intervention_mode"]))
        _set_frozen_fields(self, _VISUALIZATION_FIELDS, values)
        object.__setattr__(self, "_specified_fields", frozenset(specified_fields))

    @property
    def mode(self) -> VisModeLiteral:
        """Deprecated alias for ``view``."""

        return self.view

    @property
    def max_module_depth(self) -> int:
        """Deprecated alias for ``depth``."""

        return self.depth

    @property
    def layout_engine(self) -> VisNodePlacementLiteral:
        """Deprecated alias for ``layout``."""

        return self.layout

    @property
    def node_mode(self) -> VisNodeModeLiteral:
        """Deprecated alias for ``node_style``."""

        return self.node_style

    def as_dict(self) -> dict[str, Any]:
        """Return the option values as a plain dictionary."""

        return {field_name: getattr(self, field_name) for field_name in _VISUALIZATION_FIELDS}

    def is_field_explicit(self, field_name: str) -> bool:
        """Return whether a field was explicitly supplied by the caller."""

        return _field_is_explicit(self, field_name)

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, Any],
        specified_fields: frozenset[str],
    ) -> "VisualizationOptions":
        """Build an instance from already-resolved field values."""

        instance = object.__new__(cls)
        _validate_node_style(cast(VisNodeModeLiteral, values["node_style"]))
        _validate_intervention_mode(cast(VisInterventionModeLiteral, values["intervention_mode"]))
        _validate_buffer_visibility(values["show_buffers"])
        _set_frozen_fields(instance, _VISUALIZATION_FIELDS, values)
        object.__setattr__(instance, "_specified_fields", specified_fields)
        return instance


@dataclass(frozen=True, init=False)
class ReplayOptions:
    """Grouped replay/rerun options for ``ModelLog`` propagation APIs.

    Parameters
    ----------
    strict:
        Whether divergence warnings should raise.
    hooks:
        Optional replay hook mapping.
    append:
        Whether rerun appends a compatible batch chunk.
    chunk_size:
        Future explicit append chunk size; currently inert.
    is_appended:
        Future append-state override; currently inert.
    device_override:
        Future replay device override; currently inert.

    Examples
    --------
    >>> opts = ReplayOptions(strict=True)
    >>> opts.strict
    True
    """

    strict: bool = False
    hooks: dict[Any, Any] | None = None
    append: bool = False
    chunk_size: int | None = None
    is_appended: bool | None = None
    device_override: str | torch.device | None = None
    _specified_fields: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)

    def __init__(
        self,
        strict: bool | MissingType = MISSING,
        hooks: dict[Any, Any] | None | MissingType = MISSING,
        append: bool | MissingType = MISSING,
        chunk_size: int | None | MissingType = MISSING,
        is_appended: bool | None | MissingType = MISSING,
        device_override: str | torch.device | None | MissingType = MISSING,
    ) -> None:
        """Initialize a frozen replay option bundle."""

        specified_fields: set[str] = set()
        values: dict[str, Any] = {
            "strict": _resolve_option_value("strict", strict, False, specified_fields),
            "hooks": _resolve_option_value("hooks", hooks, None, specified_fields),
            "append": _resolve_option_value("append", append, False, specified_fields),
            "chunk_size": _resolve_option_value("chunk_size", chunk_size, None, specified_fields),
            "is_appended": _resolve_option_value(
                "is_appended", is_appended, None, specified_fields
            ),
            "device_override": _resolve_option_value(
                "device_override", device_override, None, specified_fields
            ),
        }
        _set_frozen_fields(self, _REPLAY_FIELDS, values)
        object.__setattr__(self, "_specified_fields", frozenset(specified_fields))

    def as_dict(self) -> dict[str, Any]:
        """Return the option values as a plain dictionary."""

        return {field_name: getattr(self, field_name) for field_name in _REPLAY_FIELDS}

    def is_field_explicit(self, field_name: str) -> bool:
        """Return whether a field was explicitly supplied by the caller."""

        return _field_is_explicit(self, field_name)

    @classmethod
    def from_values(
        cls, values: Mapping[str, Any], specified_fields: frozenset[str]
    ) -> "ReplayOptions":
        """Build an instance from already-resolved field values."""

        instance = object.__new__(cls)
        _set_frozen_fields(instance, _REPLAY_FIELDS, values)
        object.__setattr__(instance, "_specified_fields", specified_fields)
        return instance


@dataclass(frozen=True, init=False)
class InterventionOptions:
    """Grouped intervention options for ``ModelLog.do``.

    Parameters
    ----------
    engine:
        Propagation engine: ``"auto"``, ``"replay"``, ``"rerun"``, or ``"set_only"``.
    confirm_mutation:
        Whether root-mutation warnings are suppressed for intentional mutation.
    strict:
        Whether selector and propagation checks raise instead of warning.
    helper_validation:
        Future helper-validation mode; currently inert.
    auto_promote:
        Future capture auto-promotion toggle; currently inert.
    cohort_migration:
        Future cohort migration toggle; currently inert.
    error_severity_threshold:
        Future severity threshold for intervention errors; currently inert.

    Examples
    --------
    >>> opts = InterventionOptions(engine="set_only", strict=True)
    >>> opts.engine
    'set_only'
    """

    engine: str = "auto"
    confirm_mutation: bool = False
    strict: bool = False
    helper_validation: str = "default"
    auto_promote: bool = False
    cohort_migration: bool = True
    error_severity_threshold: str = "recoverable"
    _specified_fields: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)

    def __init__(
        self,
        engine: str | MissingType = MISSING,
        confirm_mutation: bool | MissingType = MISSING,
        strict: bool | MissingType = MISSING,
        helper_validation: str | MissingType = MISSING,
        auto_promote: bool | MissingType = MISSING,
        cohort_migration: bool | MissingType = MISSING,
        error_severity_threshold: str | MissingType = MISSING,
    ) -> None:
        """Initialize a frozen intervention option bundle."""

        specified_fields: set[str] = set()
        values: dict[str, Any] = {
            "engine": _resolve_option_value("engine", engine, "auto", specified_fields),
            "confirm_mutation": _resolve_option_value(
                "confirm_mutation", confirm_mutation, False, specified_fields
            ),
            "strict": _resolve_option_value("strict", strict, False, specified_fields),
            "helper_validation": _resolve_option_value(
                "helper_validation", helper_validation, "default", specified_fields
            ),
            "auto_promote": _resolve_option_value(
                "auto_promote", auto_promote, False, specified_fields
            ),
            "cohort_migration": _resolve_option_value(
                "cohort_migration", cohort_migration, True, specified_fields
            ),
            "error_severity_threshold": _resolve_option_value(
                "error_severity_threshold",
                error_severity_threshold,
                "recoverable",
                specified_fields,
            ),
        }
        _set_frozen_fields(self, _INTERVENTION_FIELDS, values)
        object.__setattr__(self, "_specified_fields", frozenset(specified_fields))

    def as_dict(self) -> dict[str, Any]:
        """Return the option values as a plain dictionary."""

        return {field_name: getattr(self, field_name) for field_name in _INTERVENTION_FIELDS}

    def is_field_explicit(self, field_name: str) -> bool:
        """Return whether a field was explicitly supplied by the caller."""

        return _field_is_explicit(self, field_name)

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, Any],
        specified_fields: frozenset[str],
    ) -> "InterventionOptions":
        """Build an instance from already-resolved field values."""

        instance = object.__new__(cls)
        _set_frozen_fields(instance, _INTERVENTION_FIELDS, values)
        object.__setattr__(instance, "_specified_fields", specified_fields)
        return instance


@dataclass(frozen=True, init=False)
class StreamingOptions:
    """Grouped streaming-save options for ``log_forward_pass``.

    Parameters
    ----------
    bundle_path:
        Portable bundle directory for streamed activation saves.
    retain_in_memory:
        Whether streamed activations remain in memory.
    activation_callback:
        Callback invoked with ``(label, tensor)`` for each saved activation.

    Examples
    --------
    >>> opts = StreamingOptions(bundle_path="run.tlspec", retain_in_memory=False)
    >>> opts.retain_in_memory
    False
    """

    bundle_path: str | Path | None = None
    retain_in_memory: bool = True
    activation_callback: Callable[[str, torch.Tensor], None] | None = None
    _specified_fields: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)

    def __init__(
        self,
        bundle_path: str | Path | None | MissingType = MISSING,
        retain_in_memory: bool | MissingType = MISSING,
        activation_callback: Callable[[str, torch.Tensor], None] | None | MissingType = MISSING,
        *,
        save_activations_to: str | Path | None | MissingType = MISSING,
        keep_activations_in_memory: bool | MissingType = MISSING,
        activation_sink: Callable[[str, torch.Tensor], None] | None | MissingType = MISSING,
    ) -> None:
        """Initialize a frozen streaming option bundle."""

        if save_activations_to is not MISSING:
            if bundle_path is not MISSING:
                raise TypeError(
                    "kwarg save_activations_to deprecated, use bundle_path; do not pass both"
                )
            warn_deprecated_alias("save_activations_to", "streaming.bundle_path")
            bundle_path = save_activations_to
        if keep_activations_in_memory is not MISSING:
            if retain_in_memory is not MISSING:
                raise TypeError(
                    "kwarg keep_activations_in_memory deprecated, use retain_in_memory; "
                    "do not pass both"
                )
            warn_deprecated_alias("keep_activations_in_memory", "streaming.retain_in_memory")
            retain_in_memory = keep_activations_in_memory
        if activation_sink is not MISSING:
            if activation_callback is not MISSING:
                raise TypeError(
                    "kwarg activation_sink deprecated, use activation_callback; do not pass both"
                )
            warn_deprecated_alias("activation_sink", "streaming.activation_callback")
            activation_callback = activation_sink

        specified_fields: set[str] = set()
        values: dict[str, Any] = {
            "bundle_path": _resolve_option_value(
                "bundle_path", bundle_path, None, specified_fields
            ),
            "retain_in_memory": _resolve_option_value(
                "retain_in_memory", retain_in_memory, True, specified_fields
            ),
            "activation_callback": _resolve_option_value(
                "activation_callback", activation_callback, None, specified_fields
            ),
        }
        _set_frozen_fields(self, _STREAMING_FIELDS, values)
        object.__setattr__(self, "_specified_fields", frozenset(specified_fields))

    def as_dict(self) -> dict[str, Any]:
        """Return the option values as a plain dictionary."""

        return {field_name: getattr(self, field_name) for field_name in _STREAMING_FIELDS}

    def is_field_explicit(self, field_name: str) -> bool:
        """Return whether a field was explicitly supplied by the caller."""

        return _field_is_explicit(self, field_name)

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, Any],
        specified_fields: frozenset[str],
    ) -> "StreamingOptions":
        """Build an instance from already-resolved field values."""

        instance = object.__new__(cls)
        _set_frozen_fields(instance, _STREAMING_FIELDS, values)
        object.__setattr__(instance, "_specified_fields", specified_fields)
        return instance


def merge_capture_options(
    *,
    capture: CaptureOptions | None,
    **flat_values: Any,
) -> CaptureOptions:
    """Merge individual capture kwargs into a grouped options object."""

    for old_name, new_name in (
        ("num_context_lines", "source_context_lines"),
        ("mark_input_output_distances", "compute_input_output_distances"),
        ("detect_loops", "detect_recurrent_patterns"),
    ):
        if (
            flat_values.get(old_name, MISSING) is not MISSING
            and flat_values.get(new_name, MISSING) is not MISSING
        ):
            raise TypeError(f"kwarg {old_name} deprecated, use {new_name}; do not pass both")

    return cast(
        CaptureOptions,
        _merge_grouped_options(
            option=capture,
            option_factory=CaptureOptions,
            fields=_CAPTURE_FIELDS,
            flat_to_group=_CAPTURE_FLAT_TO_GROUP,
            flat_values=flat_values,
            group_name="capture",
            conflict_message=(
                "conflicting capture options: pass either CaptureOptions or individual kwargs, not both"
            ),
        ),
    )


def merge_save_options(*, save: SaveOptions | None, **flat_values: Any) -> SaveOptions:
    """Merge individual save kwargs into a grouped options object."""

    return cast(
        SaveOptions,
        _merge_grouped_options(
            option=save,
            option_factory=SaveOptions,
            fields=_SAVE_FIELDS,
            flat_to_group=_SAVE_FLAT_TO_GROUP,
            flat_values=flat_values,
            group_name="save",
            conflict_message=(
                "conflicting save options: pass either SaveOptions or individual kwargs, not both"
            ),
        ),
    )


def merge_visualization_options(
    *,
    function_default_mode: VisModeLiteral,
    visualization: VisualizationOptions | None,
    view: VisModeLiteral | MissingType = MISSING,
    depth: int | MissingType = MISSING,
    layout: VisNodePlacementLiteral | MissingType = MISSING,
    node_style: VisNodeModeLiteral | MissingType = MISSING,
    renderer: VisRendererLiteral | MissingType = MISSING,
    vis_mode: VisModeLiteral | MissingType = MISSING,
    vis_nesting_depth: int | MissingType = MISSING,
    vis_outpath: str | MissingType = MISSING,
    vis_save_only: bool | MissingType = MISSING,
    vis_fileformat: str | MissingType = MISSING,
    vis_buffer_layers: BufferVisibilityLiteral | bool | MissingType = MISSING,
    vis_direction: VisDirectionLiteral | MissingType = MISSING,
    vis_graph_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_node_mode: VisNodeModeLiteral | MissingType = MISSING,
    vis_edge_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_gradient_edge_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_module_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_node_placement: VisNodePlacementLiteral | MissingType = MISSING,
    vis_renderer: VisRendererLiteral | MissingType = MISSING,
    vis_theme: str | MissingType = MISSING,
    vis_intervention_mode: VisInterventionModeLiteral | MissingType = MISSING,
    vis_show_cone: bool | MissingType = MISSING,
) -> VisualizationOptions:
    """Merge flat visualization kwargs into a grouped options object."""

    if visualization is None:
        values = VisualizationOptions().as_dict()
        values["view"] = function_default_mode
        specified_fields: frozenset[str] = frozenset()
    else:
        values = visualization.as_dict()
        specified_fields = _explicit_fields(visualization)

    flat_values: dict[str, Any] = {
        "view": view,
        "depth": depth,
        "layout": layout,
        "node_style": node_style,
        "renderer": renderer,
        "vis_mode": vis_mode,
        "vis_nesting_depth": vis_nesting_depth,
        "vis_outpath": vis_outpath,
        "vis_save_only": vis_save_only,
        "vis_fileformat": vis_fileformat,
        "vis_buffer_layers": vis_buffer_layers,
        "vis_direction": vis_direction,
        "vis_graph_overrides": vis_graph_overrides,
        "vis_node_mode": vis_node_mode,
        "vis_edge_overrides": vis_edge_overrides,
        "vis_gradient_edge_overrides": vis_gradient_edge_overrides,
        "vis_module_overrides": vis_module_overrides,
        "vis_node_placement": vis_node_placement,
        "vis_renderer": vis_renderer,
        "vis_theme": vis_theme,
        "vis_intervention_mode": vis_intervention_mode,
        "vis_show_cone": vis_show_cone,
    }
    for flat_name, group_name in _VISUALIZATION_FLAT_TO_GROUP.items():
        flat_value = flat_values[flat_name]
        if flat_value is MISSING:
            continue
        if visualization is not None and group_name in specified_fields:
            raise TypeError(f"Do not pass both `{flat_name}` and `visualization.{group_name}`.")
        if flat_name in _VISUALIZATION_DEPRECATED_FLAT:
            warn_deprecated_alias(flat_name, f"visualization.{group_name}")
        values[group_name] = flat_value
        specified_fields = frozenset((*specified_fields, group_name))
    return VisualizationOptions.from_values(values, specified_fields)


def merge_replay_options(*, replay: ReplayOptions | None, **flat_values: Any) -> ReplayOptions:
    """Merge individual replay kwargs into a grouped options object."""

    return cast(
        ReplayOptions,
        _merge_grouped_options(
            option=replay,
            option_factory=ReplayOptions,
            fields=_REPLAY_FIELDS,
            flat_to_group=_REPLAY_FLAT_TO_GROUP,
            flat_values=flat_values,
            group_name="replay",
            conflict_message=(
                "conflicting replay options: pass either ReplayOptions or individual kwargs, not both"
            ),
        ),
    )


def merge_intervention_options(
    *,
    intervention: InterventionOptions | None,
    **flat_values: Any,
) -> InterventionOptions:
    """Merge individual intervention kwargs into a grouped options object."""

    return cast(
        InterventionOptions,
        _merge_grouped_options(
            option=intervention,
            option_factory=InterventionOptions,
            fields=_INTERVENTION_FIELDS,
            flat_to_group=_INTERVENTION_FLAT_TO_GROUP,
            flat_values=flat_values,
            group_name="intervention",
            conflict_message=(
                "conflicting intervention options: pass either InterventionOptions or individual "
                "kwargs, not both"
            ),
        ),
    )


def merge_streaming_options(
    *,
    streaming: StreamingOptions | None,
    **flat_values: Any,
) -> StreamingOptions:
    """Merge individual streaming kwargs into a grouped options object."""

    return cast(
        StreamingOptions,
        _merge_grouped_options(
            option=streaming,
            option_factory=StreamingOptions,
            fields=_STREAMING_FIELDS,
            flat_to_group=_STREAMING_FLAT_TO_GROUP,
            flat_values=flat_values,
            group_name="streaming",
            conflict_message=(
                "conflicting streaming options: pass either StreamingOptions or individual kwargs, "
                "not both"
            ),
        ),
    )


def visualization_to_render_kwargs(visualization: VisualizationOptions) -> dict[str, Any]:
    """Translate grouped visualization options into ``ModelLog.render_graph`` kwargs.

    Parameters
    ----------
    visualization:
        Resolved grouped visualization options.

    Returns
    -------
    dict[str, Any]
        Keyword arguments expected by ``ModelLog.render_graph``.
    """

    kwargs: dict[str, Any] = {
        "vis_mode": visualization.view,
        "vis_nesting_depth": visualization.depth,
        "vis_outpath": visualization.output_path,
        "vis_graph_overrides": visualization.graph_overrides,
        "node_mode": visualization.node_style,
        "node_spec_fn": visualization.node_spec_fn,
        "collapsed_node_spec_fn": visualization.collapsed_node_spec_fn,
        "collapse_fn": visualization.collapse_fn,
        "skip_fn": visualization.skip_fn,
        "vis_edge_overrides": visualization.edge_overrides,
        "vis_gradient_edge_overrides": visualization.gradient_edge_overrides,
        "vis_module_overrides": visualization.module_overrides,
        "vis_save_only": visualization.save_only,
        "vis_fileformat": visualization.file_format,
        "show_buffer_layers": visualization.show_buffers,
        "direction": visualization.direction,
        "vis_node_placement": visualization.layout,
        "vis_renderer": visualization.renderer,
        "vis_theme": visualization.theme,
        "vis_intervention_mode": visualization.intervention_mode,
        "vis_show_cone": visualization.show_cone,
    }
    phase7_kwargs = {
        "node_overlay": visualization.node_overlay,
        "node_label_fields": visualization.node_label_fields,
        "show_legend": visualization.show_legend,
        "font_size": visualization.font_size,
        "dpi": visualization.dpi,
        "for_paper": visualization.for_paper,
        "return_graph": visualization.return_graph,
    }
    for field_name, value in phase7_kwargs.items():
        if visualization.is_field_explicit(field_name) or (
            value is not None and value is not False
        ):
            kwargs[field_name] = value
    return kwargs


__all__ = [
    "CaptureOptions",
    "InterventionOptions",
    "ReplayOptions",
    "SaveOptions",
    "StreamingOptions",
    "VisualizationOptions",
    "merge_capture_options",
    "merge_intervention_options",
    "merge_replay_options",
    "merge_save_options",
    "merge_streaming_options",
    "merge_visualization_options",
    "suppress_mutate_warnings",
    "visualization_to_render_kwargs",
]
