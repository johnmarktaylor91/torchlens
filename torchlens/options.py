"""Grouped option dataclasses for public TorchLens APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Final, Mapping, cast

import torch

from ._deprecations import MISSING, MissingType, warn_deprecated_alias
from ._literals import (
    BufferVisibilityLiteral,
    VisDirectionLiteral,
    VisModeLiteral,
    VisNodeModeLiteral,
    VisNodePlacementLiteral,
    VisRendererLiteral,
)
from .visualization.node_spec import NodeSpec

if TYPE_CHECKING:
    from .data_classes.layer_log import LayerLog
    from .data_classes.module_log import ModuleLog

_VISUALIZATION_FIELDS: Final[tuple[str, ...]] = (
    "mode",
    "max_module_depth",
    "output_path",
    "save_only",
    "file_format",
    "show_buffers",
    "direction",
    "graph_overrides",
    "node_mode",
    "node_spec_fn",
    "collapsed_node_spec_fn",
    "collapse_fn",
    "skip_fn",
    "edge_overrides",
    "gradient_edge_overrides",
    "module_overrides",
    "layout_engine",
    "renderer",
    "theme",
)
_STREAMING_FIELDS: Final[tuple[str, ...]] = (
    "bundle_path",
    "retain_in_memory",
    "activation_callback",
)
_VISUALIZATION_FLAT_TO_GROUP: Final[dict[str, str]] = {
    "vis_mode": "mode",
    "vis_nesting_depth": "max_module_depth",
    "vis_outpath": "output_path",
    "vis_save_only": "save_only",
    "vis_fileformat": "file_format",
    "vis_buffer_layers": "show_buffers",
    "vis_direction": "direction",
    "vis_graph_overrides": "graph_overrides",
    "vis_node_mode": "node_mode",
    "vis_edge_overrides": "edge_overrides",
    "vis_gradient_edge_overrides": "gradient_edge_overrides",
    "vis_module_overrides": "module_overrides",
    "vis_node_placement": "layout_engine",
    "vis_renderer": "renderer",
    "vis_theme": "theme",
}
_STREAMING_FLAT_TO_GROUP: Final[dict[str, str]] = {
    "save_activations_to": "bundle_path",
    "keep_activations_in_memory": "retain_in_memory",
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


def _validate_node_mode(node_mode: VisNodeModeLiteral) -> None:
    """Validate a visualization node-mode preset name.

    Parameters
    ----------
    node_mode:
        Candidate node-mode preset name.

    Raises
    ------
    ValueError
        If ``node_mode`` is not a registered public preset.
    """

    if node_mode not in {"default", "profiling", "vision", "attention"}:
        raise ValueError(
            "Visualization node_mode must be one of 'default', 'profiling', "
            "'vision', or 'attention'."
        )


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


@dataclass(frozen=True, init=False)
class VisualizationOptions:
    """Grouped visualization options for ``log_forward_pass`` and ``show_model_graph``."""

    mode: VisModeLiteral = "none"
    max_module_depth: int = 1000
    output_path: str = "graph.gv"
    save_only: bool = False
    file_format: str = "pdf"
    show_buffers: BufferVisibilityLiteral = "meaningful"
    direction: VisDirectionLiteral = "bottomup"
    graph_overrides: dict[str, Any] | None = None
    node_mode: VisNodeModeLiteral = "default"
    node_spec_fn: Callable[["LayerLog", NodeSpec], NodeSpec | None] | None = None
    collapsed_node_spec_fn: Callable[["ModuleLog", NodeSpec], NodeSpec | None] | None = None
    collapse_fn: Callable[["ModuleLog"], bool] | None = None
    skip_fn: Callable[["LayerLog"], bool] | None = None
    edge_overrides: dict[str, Any] | None = None
    gradient_edge_overrides: dict[str, Any] | None = None
    module_overrides: dict[str, Any] | None = None
    layout_engine: VisNodePlacementLiteral = "auto"
    renderer: VisRendererLiteral = "graphviz"
    theme: str = "torchlens"
    _specified_fields: frozenset[str] = field(
        default_factory=frozenset,
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        mode: VisModeLiteral | MissingType = MISSING,
        max_module_depth: int | MissingType = MISSING,
        output_path: str | MissingType = MISSING,
        save_only: bool | MissingType = MISSING,
        file_format: str | MissingType = MISSING,
        show_buffers: BufferVisibilityLiteral | bool | MissingType = MISSING,
        direction: VisDirectionLiteral | MissingType = MISSING,
        graph_overrides: dict[str, Any] | None | MissingType = MISSING,
        node_mode: VisNodeModeLiteral | MissingType = MISSING,
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
        layout_engine: VisNodePlacementLiteral | MissingType = MISSING,
        renderer: VisRendererLiteral | MissingType = MISSING,
        theme: str | MissingType = MISSING,
    ) -> None:
        """Initialize a frozen visualization option bundle.

        Parameters
        ----------
        mode, max_module_depth, output_path, save_only, file_format, show_buffers, direction,
        graph_overrides, node_mode, node_spec_fn, collapsed_node_spec_fn, collapse_fn, skip_fn,
        edge_overrides, gradient_edge_overrides, module_overrides, layout_engine,
        renderer, theme:
            Visualization option values. Explicitly supplied fields are tracked so
            deprecated flat kwargs can detect same-field conflicts later.
            ``show_buffers`` accepts ``"never"``, ``"meaningful"``, or
            ``"always"``. Legacy bools are deprecated but supported:
            ``True`` maps to ``"always"`` and ``False`` maps to ``"never"``.
        """

        specified_fields: set[str] = set()
        values: dict[str, Any] = {
            "mode": _resolve_option_value("mode", mode, "none", specified_fields),
            "max_module_depth": _resolve_option_value(
                "max_module_depth",
                max_module_depth,
                1000,
                specified_fields,
            ),
            "output_path": _resolve_option_value(
                "output_path", output_path, "graph.gv", specified_fields
            ),
            "save_only": _resolve_option_value("save_only", save_only, False, specified_fields),
            "file_format": _resolve_option_value(
                "file_format", file_format, "pdf", specified_fields
            ),
            "show_buffers": _resolve_option_value(
                "show_buffers",
                show_buffers,
                "meaningful",
                specified_fields,
            ),
            "direction": _resolve_option_value(
                "direction",
                direction,
                "bottomup",
                specified_fields,
            ),
            "graph_overrides": _resolve_option_value(
                "graph_overrides",
                graph_overrides,
                None,
                specified_fields,
            ),
            "node_mode": _resolve_option_value(
                "node_mode",
                node_mode,
                "default",
                specified_fields,
            ),
            "node_spec_fn": _resolve_option_value(
                "node_spec_fn",
                node_spec_fn,
                None,
                specified_fields,
            ),
            "collapsed_node_spec_fn": _resolve_option_value(
                "collapsed_node_spec_fn",
                collapsed_node_spec_fn,
                None,
                specified_fields,
            ),
            "collapse_fn": _resolve_option_value(
                "collapse_fn",
                collapse_fn,
                None,
                specified_fields,
            ),
            "skip_fn": _resolve_option_value(
                "skip_fn",
                skip_fn,
                None,
                specified_fields,
            ),
            "edge_overrides": _resolve_option_value(
                "edge_overrides",
                edge_overrides,
                None,
                specified_fields,
            ),
            "gradient_edge_overrides": _resolve_option_value(
                "gradient_edge_overrides",
                gradient_edge_overrides,
                None,
                specified_fields,
            ),
            "module_overrides": _resolve_option_value(
                "module_overrides",
                module_overrides,
                None,
                specified_fields,
            ),
            "layout_engine": _resolve_option_value(
                "layout_engine",
                layout_engine,
                "auto",
                specified_fields,
            ),
            "renderer": _resolve_option_value("renderer", renderer, "graphviz", specified_fields),
            "theme": _resolve_option_value("theme", theme, "torchlens", specified_fields),
        }
        _validate_buffer_visibility(values["show_buffers"])
        for field_name in _VISUALIZATION_FIELDS:
            object.__setattr__(self, field_name, values[field_name])
        _validate_node_mode(cast(VisNodeModeLiteral, values["node_mode"]))
        object.__setattr__(self, "_specified_fields", frozenset(specified_fields))

    def as_dict(self) -> dict[str, Any]:
        """Return the option values as a plain dictionary."""

        return {field_name: getattr(self, field_name) for field_name in _VISUALIZATION_FIELDS}

    def is_field_explicit(self, field_name: str) -> bool:
        """Return whether a field was explicitly supplied by the caller."""

        return field_name in self._specified_fields

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, Any],
        specified_fields: frozenset[str],
    ) -> "VisualizationOptions":
        """Build an instance from already-resolved field values."""

        instance = object.__new__(cls)
        _validate_node_mode(cast(VisNodeModeLiteral, values["node_mode"]))
        _validate_buffer_visibility(values["show_buffers"])
        for field_name in _VISUALIZATION_FIELDS:
            object.__setattr__(instance, field_name, values[field_name])
        object.__setattr__(instance, "_specified_fields", specified_fields)
        return cast("VisualizationOptions", instance)


@dataclass(frozen=True, init=False)
class StreamingOptions:
    """Grouped streaming-save options for ``log_forward_pass``."""

    bundle_path: str | Path | None = None
    retain_in_memory: bool = True
    activation_callback: Callable[[str, torch.Tensor], None] | None = None
    _specified_fields: frozenset[str] = field(
        default_factory=frozenset,
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        bundle_path: str | Path | None | MissingType = MISSING,
        retain_in_memory: bool | MissingType = MISSING,
        activation_callback: Callable[[str, torch.Tensor], None] | None | MissingType = MISSING,
    ) -> None:
        """Initialize a frozen streaming option bundle.

        Parameters
        ----------
        bundle_path, retain_in_memory, activation_callback:
            Streaming option values. Explicitly supplied fields are tracked so
            deprecated flat kwargs can detect same-field conflicts later.
        """

        specified_fields: set[str] = set()
        values: dict[str, Any] = {
            "bundle_path": _resolve_option_value(
                "bundle_path", bundle_path, None, specified_fields
            ),
            "retain_in_memory": _resolve_option_value(
                "retain_in_memory",
                retain_in_memory,
                True,
                specified_fields,
            ),
            "activation_callback": _resolve_option_value(
                "activation_callback",
                activation_callback,
                None,
                specified_fields,
            ),
        }
        for field_name in _STREAMING_FIELDS:
            object.__setattr__(self, field_name, values[field_name])
        object.__setattr__(self, "_specified_fields", frozenset(specified_fields))

    def as_dict(self) -> dict[str, Any]:
        """Return the option values as a plain dictionary."""

        return {field_name: getattr(self, field_name) for field_name in _STREAMING_FIELDS}

    def is_field_explicit(self, field_name: str) -> bool:
        """Return whether a field was explicitly supplied by the caller."""

        return field_name in self._specified_fields

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, Any],
        specified_fields: frozenset[str],
    ) -> "StreamingOptions":
        """Build an instance from already-resolved field values."""

        instance = object.__new__(cls)
        for field_name in _STREAMING_FIELDS:
            object.__setattr__(instance, field_name, values[field_name])
        object.__setattr__(instance, "_specified_fields", specified_fields)
        return cast("StreamingOptions", instance)


def merge_visualization_options(
    *,
    function_default_mode: VisModeLiteral,
    visualization: VisualizationOptions | None,
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
) -> VisualizationOptions:
    """Merge deprecated flat visualization kwargs into a grouped options object.

    Parameters
    ----------
    function_default_mode:
        Per-function default visualization mode when no explicit grouped object
        was supplied.
    visualization:
        Caller-supplied grouped visualization options, if any.
    vis_mode, vis_nesting_depth, vis_outpath, vis_save_only, vis_fileformat,
    vis_buffer_layers, vis_direction, vis_graph_overrides, vis_node_mode, vis_edge_overrides,
    vis_gradient_edge_overrides,
    vis_module_overrides, vis_node_placement, vis_renderer, vis_theme:
        Deprecated flat visualization kwargs. Presence is tracked with
        ``MISSING`` so explicit default-valued calls still count as supplied.
        ``vis_buffer_layers`` accepts ``"never"``, ``"meaningful"``, or
        ``"always"``. Legacy bools are deprecated but supported: ``True`` maps
        to ``"always"`` and ``False`` maps to ``"never"``.

    Returns
    -------
    VisualizationOptions
        Resolved visualization options for the current call.

    Raises
    ------
    TypeError
        If the same visualization field was supplied via both grouped and flat
        forms.
    """

    if visualization is None:
        values = VisualizationOptions().as_dict()
        values["mode"] = function_default_mode
        specified_fields: frozenset[str] = frozenset()
    else:
        values = visualization.as_dict()
        specified_fields = visualization._specified_fields

    flat_values: dict[str, Any] = {
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
    }
    for flat_name, group_name in _VISUALIZATION_FLAT_TO_GROUP.items():
        flat_value = flat_values[flat_name]
        if flat_value is MISSING:
            continue
        if visualization is not None and group_name in specified_fields:
            raise TypeError(f"Do not pass both `{flat_name}` and `visualization.{group_name}`.")
        warn_deprecated_alias(flat_name, f"visualization.{group_name}")
        values[group_name] = flat_value
        specified_fields = frozenset((*specified_fields, group_name))
    return VisualizationOptions.from_values(values, specified_fields)


def merge_streaming_options(
    *,
    streaming: StreamingOptions | None,
    save_activations_to: str | Path | None | MissingType = MISSING,
    keep_activations_in_memory: bool | MissingType = MISSING,
    activation_sink: Callable[[str, torch.Tensor], None] | None | MissingType = MISSING,
) -> StreamingOptions:
    """Merge deprecated flat streaming kwargs into a grouped options object.

    Parameters
    ----------
    streaming:
        Caller-supplied grouped streaming options, if any.
    save_activations_to, keep_activations_in_memory, activation_sink:
        Deprecated flat streaming kwargs. Presence is tracked with ``MISSING``
        so explicit default-valued calls still count as supplied.

    Returns
    -------
    StreamingOptions
        Resolved streaming options for the current call.

    Raises
    ------
    TypeError
        If the same streaming field was supplied via both grouped and flat
        forms.
    """

    if streaming is None:
        values = StreamingOptions().as_dict()
        specified_fields: frozenset[str] = frozenset()
    else:
        values = streaming.as_dict()
        specified_fields = streaming._specified_fields

    flat_values: dict[str, Any] = {
        "save_activations_to": save_activations_to,
        "keep_activations_in_memory": keep_activations_in_memory,
        "activation_sink": activation_sink,
    }
    for flat_name, group_name in _STREAMING_FLAT_TO_GROUP.items():
        flat_value = flat_values[flat_name]
        if flat_value is MISSING:
            continue
        if streaming is not None and group_name in specified_fields:
            raise TypeError(f"Do not pass both `{flat_name}` and `streaming.{group_name}`.")
        warn_deprecated_alias(flat_name, f"streaming.{group_name}")
        values[group_name] = flat_value
    return StreamingOptions.from_values(values, specified_fields)


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

    return {
        "vis_mode": visualization.mode,
        "vis_nesting_depth": visualization.max_module_depth,
        "vis_outpath": visualization.output_path,
        "vis_graph_overrides": visualization.graph_overrides,
        "node_mode": visualization.node_mode,
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
        "vis_node_placement": visualization.layout_engine,
        "vis_renderer": visualization.renderer,
        "vis_theme": visualization.theme,
    }
