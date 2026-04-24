"""Grouped option dataclasses for public TorchLens APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Mapping, cast

import torch

from ._deprecations import MISSING, MissingType, warn_deprecated_alias
from ._literals import (
    VisDirectionLiteral,
    VisModeLiteral,
    VisNodePlacementLiteral,
    VisRendererLiteral,
)

_VISUALIZATION_FIELDS: Final[tuple[str, ...]] = (
    "mode",
    "max_module_depth",
    "output_path",
    "save_only",
    "file_format",
    "show_buffers",
    "direction",
    "graph_overrides",
    "node_overrides",
    "nested_node_overrides",
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
    "vis_node_overrides": "node_overrides",
    "vis_nested_node_overrides": "nested_node_overrides",
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


@dataclass(frozen=True, init=False)
class VisualizationOptions:
    """Grouped visualization options for ``log_forward_pass`` and ``show_model_graph``."""

    mode: VisModeLiteral = "none"
    max_module_depth: int = 1000
    output_path: str = "graph.gv"
    save_only: bool = False
    file_format: str = "pdf"
    show_buffers: bool = False
    direction: VisDirectionLiteral = "bottomup"
    graph_overrides: dict[str, Any] | None = None
    node_overrides: dict[str, Any] | None = None
    nested_node_overrides: dict[str, Any] | None = None
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
        show_buffers: bool | MissingType = MISSING,
        direction: VisDirectionLiteral | MissingType = MISSING,
        graph_overrides: dict[str, Any] | None | MissingType = MISSING,
        node_overrides: dict[str, Any] | None | MissingType = MISSING,
        nested_node_overrides: dict[str, Any] | None | MissingType = MISSING,
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
        graph_overrides, node_overrides, nested_node_overrides, edge_overrides,
        gradient_edge_overrides, module_overrides, layout_engine, renderer, theme:
            Visualization option values. Explicitly supplied fields are tracked so
            deprecated flat kwargs can detect same-field conflicts later.
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
                False,
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
            "node_overrides": _resolve_option_value(
                "node_overrides",
                node_overrides,
                None,
                specified_fields,
            ),
            "nested_node_overrides": _resolve_option_value(
                "nested_node_overrides",
                nested_node_overrides,
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
        for field_name in _VISUALIZATION_FIELDS:
            object.__setattr__(self, field_name, values[field_name])
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
    vis_buffer_layers: bool | MissingType = MISSING,
    vis_direction: VisDirectionLiteral | MissingType = MISSING,
    vis_graph_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_node_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_nested_node_overrides: dict[str, Any] | None | MissingType = MISSING,
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
    vis_buffer_layers, vis_direction, vis_graph_overrides, vis_node_overrides,
    vis_nested_node_overrides, vis_edge_overrides, vis_gradient_edge_overrides,
    vis_module_overrides, vis_node_placement, vis_renderer, vis_theme:
        Deprecated flat visualization kwargs. Presence is tracked with
        ``MISSING`` so explicit default-valued calls still count as supplied.

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
        "vis_node_overrides": vis_node_overrides,
        "vis_nested_node_overrides": vis_nested_node_overrides,
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
        "vis_node_overrides": visualization.node_overrides,
        "vis_nested_node_overrides": visualization.nested_node_overrides,
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
