"""Private implementations for public user-facing utility commands."""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import nn
from tqdm import tqdm

from . import user_funcs as _user_funcs
from ._deprecations import MISSING, MissingType, warn_deprecated_alias
from ._input_coerce import _coerce_input_args
from ._literals import (
    BufferVisibilityLiteral,
    CollapseLiteral,
    VisDirectionLiteral,
    VisInterventionModeLiteral,
    VisModeLiteral,
    VisNodeModeLiteral,
    VisNodePlacementLiteral,
    VisRendererLiteral,
    FoldRunsLiteral,
)
from ._capture_state_helpers import (
    _clone_state_dict_with_metadata,
    _model_for_ground_truth_validation,
    _move_tensors_to_device,
    _reject_opaque_wrappers,
    _unwrap_data_parallel,
    unwrap_compiled_model,
)
from .backends import BackendName, resolve_backend_spec
from .data_classes.trace import Trace
from .options import (
    VisualizationOptions,
    merge_visualization_options,
    visualization_to_render_kwargs,
)
from .utils.arg_handling import normalize_input_args, safe_copy_args, safe_copy_kwargs
from .utils.display import warn_parallel
from .utils.introspection import get_vars_of_type_from_obj
from .utils.rng import set_random_seed
from .visualization.code_panel import CodePanelOption
from ._robustness import check_model_and_input_variants

if TYPE_CHECKING:
    import pandas as pd

    from .data_classes.module import Module

trace = _user_funcs.trace
_run_model_and_save_specified_outs = _user_funcs._run_model_and_save_specified_outs


def log_model_metadata(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
) -> Trace:
    """Return model metadata without saving any outs.

    Equivalent to ``trace(model, input_args, input_kwargs, layers_to_save=None,
    compute_input_output_distances=True)``.

    Parameters
    ----------
    model:
        PyTorch model to inspect.
    input_args:
        Positional args for ``model.forward()``.
    input_kwargs:
        Keyword args for ``model.forward()``.

    Returns
    -------
    Trace
        Trace with full metadata but no saved outs.
    """
    model = unwrap_compiled_model(model)
    model_trace = trace(
        model,
        input_args,
        input_kwargs,
        layers_to_save=None,
        compute_input_output_distances=True,
    )
    return model_trace


def get_model_metadata(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
) -> Trace:
    """Deprecated alias for :func:`log_model_metadata`."""

    warn_deprecated_alias("get_model_metadata", "log_model_metadata")
    return log_model_metadata(model, input_args, input_kwargs)


def summary(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    **summary_kwargs: Any,
) -> str:
    """Run a metadata-only forward pass and return a rendered summary string.

    Parameters
    ----------
    model:
        PyTorch model to inspect.
    input_args:
        Positional args for ``model.forward()``.
    input_kwargs:
        Keyword args for ``model.forward()``.
    **summary_kwargs:
        Forwarded to ``Trace.summary``.

    Returns
    -------
    str
        Rendered summary text.
    """
    _reject_opaque_wrappers(model)
    model = unwrap_compiled_model(model)
    model = _unwrap_data_parallel(model)
    if input_kwargs is None:
        input_kwargs = {}
    input_args = _coerce_input_args(model, input_args)
    check_model_and_input_variants(model, input_args, input_kwargs)

    trace = _run_model_and_save_specified_outs(
        model=model,
        input_args=input_args,
        input_kwargs=input_kwargs,
        layers_to_save=None,
        recurrence_detection=True,
    )
    try:
        return trace.summary(**summary_kwargs)
    finally:
        trace.cleanup()


def show_model_graph(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    view: VisModeLiteral | MissingType = MISSING,
    depth: int | MissingType = MISSING,
    renderer: VisRendererLiteral | MissingType = MISSING,
    layout: VisNodePlacementLiteral | MissingType = MISSING,
    node_style: VisNodeModeLiteral | MissingType = MISSING,
    vis_mode: VisModeLiteral | MissingType = MISSING,
    vis_call_depth: int | MissingType = MISSING,
    vis_outpath: str | MissingType = MISSING,
    vis_graph_overrides: dict[str, Any] | None | MissingType = MISSING,
    module: "Module | str | None" = None,
    vis_edge_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_grad_edge_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_module_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_save_only: bool | MissingType = MISSING,
    vis_fileformat: str | MissingType = MISSING,
    vis_buffers: BufferVisibilityLiteral | bool | MissingType = MISSING,
    vis_direction: VisDirectionLiteral | MissingType = MISSING,
    vis_node_placement: VisNodePlacementLiteral | MissingType = MISSING,
    vis_renderer: VisRendererLiteral | MissingType = MISSING,
    vis_theme: str | MissingType = MISSING,
    vis_intervention_mode: VisInterventionModeLiteral | MissingType = MISSING,
    vis_show_cone: bool | MissingType = MISSING,
    vis_node_mode: VisNodeModeLiteral | MissingType = MISSING,
    collapse: CollapseLiteral | MissingType = MISSING,
    fold_runs: FoldRunsLiteral | MissingType = MISSING,
    order_siblings: bool | MissingType = MISSING,
    code_panel: CodePanelOption = False,
    random_seed: int | None = None,
    verbose: bool = False,
    recurrence_detection: bool | MissingType = MISSING,
    visualization: VisualizationOptions | None = None,
) -> None:
    """Convenience wrapper: visualize the computational graph without saving outs.

    Runs an exhaustive forward pass (no outs saved) to discover the graph
    structure, renders the visualization, then cleans up the Trace.  For more
    control, use ``trace`` with ``vis_mode`` set and access the Trace
    directly.

    Parameters
    ----------
    model:
        PyTorch model.
    input_args:
        Positional args for ``model.forward()``.
    input_kwargs:
        Keyword args for ``model.forward()``.
    vis_mode:
        Deprecated alias for ``visualization.mode``.
    vis_call_depth:
        Deprecated alias for ``visualization.max_module_depth``.
    vis_outpath:
        Deprecated alias for ``visualization.container_path``.
    vis_graph_overrides:
        Deprecated alias for ``visualization.graph_overrides``.
    module:
        Optional module focus. Pass a Module or module address string to render
        only layers that ran inside that module.
    vis_edge_overrides:
        Deprecated alias for ``visualization.edge_overrides``.
    vis_grad_edge_overrides:
        Deprecated alias for ``visualization.grad_edge_overrides``.
    vis_module_overrides:
        Deprecated alias for ``visualization.module_overrides``.
    vis_save_only:
        Deprecated alias for ``visualization.save_only``.
    vis_fileformat:
        Deprecated alias for ``visualization.file_format``.
    vis_buffers:
        Deprecated alias for ``visualization.show_buffers``. Accepts
        ``"never"``, ``"meaningful"``, or ``"always"``. Legacy bools are
        deprecated but supported: ``True`` maps to ``"always"`` and ``False``
        maps to ``"never"``.
    vis_direction:
        Deprecated alias for ``visualization.direction``.
    vis_node_placement:
        Deprecated alias for ``visualization.layout_engine``. Accepts
        ``"auto"``, ``"dot"``, or ``"rank"``.
    vis_renderer:
        Deprecated alias for ``visualization.renderer``. The ``"dagua"``
        renderer is experimental and requires ``from torchlens.experimental
        import dagua`` before use.
    vis_theme:
        Deprecated alias for ``visualization.theme``.
    vis_intervention_mode:
        Intervention overlay mode. ``"node_mark"`` marks intervention sites
        and optionally their cones. ``"as_node"`` inserts a small hook node
        after each intervention site.
    vis_show_cone:
        Whether ``"node_mark"`` mode also marks downstream cone-of-effect
        members.
    order_siblings:
        Whether Graphviz ``dot`` renders should verify and apply
        execution-order placement for true parallel siblings.
    code_panel:
        Optional source-code panel mode. ``True`` is equivalent to
        ``"forward"``. Built-in modes use source captured at log time; callable
        modes receive the live model object and are only available while that
        object is still alive.
    vis_node_mode:
        Deprecated alias for ``visualization.node_mode``.
    collapse:
        Smart module-collapse mode: ``"none"``, ``"auto"``, ``"max"``, or a
        float in ``[0.0, 1.0]``. Float levels follow the public monotone
        schedule.
    fold_runs:
        Run-fold policy. ``None`` preserves the default policy. ``True`` folds
        every eligible repeated run. ``False`` disables run folding.
    random_seed:
        Fixed RNG seed for stochastic models.
    recurrence_detection:
        If True, run full isomorphic subgraph expansion. Set this to False when
        the forward pass has more than about 1M operations and postprocessing
        speed matters.
    visualization:
        Grouped visualization options. When omitted, ``show_model_graph``
        defaults to ``VisualizationOptions(mode="unrolled")``.

    Returns
    -------
    None
        The graph is rendered for side effects.
    """
    _reject_opaque_wrappers(model)
    model = unwrap_compiled_model(model)
    model = _unwrap_data_parallel(model)
    if not input_kwargs:
        input_kwargs = {}
    input_args = _coerce_input_args(model, input_args)
    check_model_and_input_variants(model, input_args, input_kwargs)

    if recurrence_detection is MISSING:
        recurrence_detection = True
    recurrence_detection_enabled = bool(recurrence_detection)
    visualization_options = merge_visualization_options(
        function_default_mode="unrolled",
        visualization=visualization,
        view=view,
        depth=depth,
        renderer=renderer,
        layout=layout,
        node_style=node_style,
        vis_mode=vis_mode,
        vis_call_depth=vis_call_depth,
        vis_outpath=vis_outpath,
        vis_save_only=vis_save_only,
        vis_fileformat=vis_fileformat,
        vis_buffers=vis_buffers,
        vis_direction=vis_direction,
        vis_graph_overrides=vis_graph_overrides,
        vis_node_mode=vis_node_mode,
        collapse=collapse,
        fold_runs=fold_runs,
        vis_edge_overrides=vis_edge_overrides,
        vis_grad_edge_overrides=vis_grad_edge_overrides,
        vis_module_overrides=vis_module_overrides,
        vis_node_placement=vis_node_placement,
        vis_renderer=vis_renderer,
        vis_theme=vis_theme,
        vis_intervention_mode=vis_intervention_mode,
        vis_show_cone=vis_show_cone,
        order_siblings=order_siblings,
    )

    if visualization_options.mode not in ["none", "rolled", "unrolled"]:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    trace = _run_model_and_save_specified_outs(
        model=model,
        input_args=input_args,
        input_kwargs=input_kwargs,
        layers_to_save=None,
        activation_transform=None,
        mark_layer_depths=False,
        detach_saved_activations=False,
        save_grads=False,
        random_seed=random_seed,
        recurrence_detection=recurrence_detection_enabled,
        verbose=verbose,
    )
    # Render in a try/finally so temporary TorchLens metadata on the model is
    # always cleaned up, even if Graphviz rendering raises.
    try:
        render_kwargs = visualization_to_render_kwargs(visualization_options)
        if module is not None:
            from .data_classes.module import Module

            render_kwargs["module"] = module.address if isinstance(module, Module) else module
        if code_panel is not False:
            render_kwargs["code_panel"] = code_panel
        trace.draw(**render_kwargs)
    finally:
        trace.cleanup()


def draw_backward(
    trace: Trace,
    vis_outpath: str | MissingType = MISSING,
    vis_save_only: bool | MissingType = MISSING,
    vis_fileformat: str | MissingType = MISSING,
    vis_direction: VisDirectionLiteral | MissingType = MISSING,
    vis_graph_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_edge_overrides: dict[str, Any] | None | MissingType = MISSING,
    node_spec_fn: Callable[[Any, Any], Any] | None = None,
    collapsed_node_spec_fn: Callable[[Any, Any], Any] | None = None,
    node_style: VisNodeModeLiteral | MissingType = MISSING,
    vis_node_mode: VisNodeModeLiteral | MissingType = MISSING,
    code_panel: CodePanelOption = False,
    vis_mode: VisModeLiteral = "rolled",
    bwd: int | Iterable[int] | None = None,
    visualization: VisualizationOptions | None = None,
) -> str:
    """Render an existing Trace's captured backward grad_fn_handle graph.

    Parameters
    ----------
    trace:
        Trace with backward metadata captured by ``trace.log_backward(loss)``
        or ``trace.recording_backward()``.
    vis_outpath:
        Output path for the rendered graph.
    vis_save_only:
        If True, save without opening a viewer.
    vis_fileformat:
        Output format.
    vis_direction:
        Layout direction. Defaults to ``"topdown"`` for backward graphs.
    vis_graph_overrides:
        Graphviz graph-level overrides.
    vis_edge_overrides:
        Graphviz edge-level overrides.
    node_spec_fn:
        Optional callback receiving ``(grad_fn_handle, default_spec)``.
    collapsed_node_spec_fn:
        Accepted for forward-visualization API symmetry. Not applied because
        backward graphs do not render collapsed module nodes.
    vis_node_mode:
        Accepted for forward-visualization API symmetry. Not applied to grad_fn_handle
        nodes.
    code_panel:
        Optional source-code panel mode.
    vis_mode:
        ``"rolled"`` renders one node per GradFn; ``"unrolled"`` renders one
        node per GradFnCall grouped by backward pass.
    bwd:
        Optional one-based backward pass number or numbers to render.
    visualization:
        Grouped visualization options. Only output path, save behavior, file
        format, direction, graph overrides, and edge overrides are used.

    Returns
    -------
    str
        Graphviz DOT source.
    """

    if visualization is None:
        container_path = "backward_modelgraph"
        save_only = False
        file_format = "pdf"
        direction: VisDirectionLiteral = "topdown"
        graph_overrides = None
        edge_overrides = None
        node_mode: VisNodeModeLiteral = "default"
    else:
        container_path = visualization.container_path
        save_only = visualization.save_only
        file_format = visualization.file_format
        direction = visualization.direction
        graph_overrides = visualization.graph_overrides
        edge_overrides = visualization.edge_overrides
        node_mode = visualization.node_style

    if vis_outpath is not MISSING:
        container_path = cast(str, vis_outpath)
    if vis_save_only is not MISSING:
        save_only = cast(bool, vis_save_only)
    if vis_fileformat is not MISSING:
        file_format = cast(str, vis_fileformat)
    if vis_direction is not MISSING:
        direction = cast(VisDirectionLiteral, vis_direction)
    if vis_graph_overrides is not MISSING:
        graph_overrides = cast(dict[str, Any] | None, vis_graph_overrides)
    if vis_edge_overrides is not MISSING:
        edge_overrides = cast(dict[str, Any] | None, vis_edge_overrides)
    if vis_node_mode is not MISSING:
        warn_deprecated_alias("vis_node_mode", "node_style")
        node_mode = cast(VisNodeModeLiteral, vis_node_mode)
    if node_style is not MISSING:
        node_mode = cast(VisNodeModeLiteral, node_style)

    return trace.draw_backward(
        vis_outpath=container_path,
        vis_graph_overrides=graph_overrides,
        node_spec_fn=node_spec_fn,
        collapsed_node_spec_fn=collapsed_node_spec_fn,
        vis_node_mode=node_mode,
        vis_edge_overrides=edge_overrides,
        vis_save_only=save_only,
        vis_fileformat=file_format,
        vis_direction=direction,
        code_panel=code_panel,
        vis_mode=vis_mode,
        bwd=bwd,
    )


def draw_combined(
    trace: Trace,
    vis_outpath: str | MissingType = MISSING,
    vis_save_only: bool | MissingType = MISSING,
    vis_fileformat: str | MissingType = MISSING,
    vis_direction: VisDirectionLiteral | MissingType = MISSING,
    vis_graph_overrides: dict[str, Any] | None | MissingType = MISSING,
    vis_edge_overrides: dict[str, Any] | None | MissingType = MISSING,
    node_spec_fn: Callable[[Any, Any], Any] | None = None,
    backward_node_spec_fn: Callable[[Any, Any], Any] | None = None,
    vis_mode: VisModeLiteral = "unrolled",
    intervening_cluster: Literal["upstream", "outside", "downstream", "own"] = "upstream",
    show_buffer_layers: BufferVisibilityLiteral | bool = "meaningful",
    bwd: int | Iterable[int] | None = None,
    visualization: VisualizationOptions | None = None,
) -> str:
    """Render an existing Trace's forward ops and backward grad_fns together.

    Parameters
    ----------
    trace:
        Trace with backward metadata captured by ``trace.log_backward(loss)``
        or ``trace.recording_backward()``.
    vis_outpath:
        Output path for the rendered graph.
    vis_save_only:
        If True, save without opening a viewer.
    vis_fileformat:
        Output format.
    vis_direction:
        Layout direction. Defaults to ``"leftright"`` for combined graphs.
    vis_graph_overrides:
        Graphviz graph-level overrides.
    vis_edge_overrides:
        Graphviz forward-edge overrides.
    node_spec_fn:
        Optional callback receiving ``(layer_log, default_spec)``.
    backward_node_spec_fn:
        Optional callback receiving ``(grad_fn_handle, default_spec)``.
    vis_mode:
        Combined rendering currently supports only ``"unrolled"``.
    intervening_cluster:
        Placement mode for grad_fns without a corresponding forward op.
    show_buffer_layers:
        Buffer visibility mode for the forward side.
    bwd:
        Optional one-based backward pass number or numbers to render.
    visualization:
        Grouped visualization options. Only output path, save behavior, file
        format, direction, graph overrides, and edge overrides are used.

    Returns
    -------
    str
        Graphviz DOT source.
    """

    if visualization is None:
        container_path = "combined_modelgraph"
        save_only = False
        file_format = "pdf"
        direction: VisDirectionLiteral = "leftright"
        graph_overrides = None
        edge_overrides = None
    else:
        container_path = visualization.container_path
        save_only = visualization.save_only
        file_format = visualization.file_format
        direction = visualization.direction
        graph_overrides = visualization.graph_overrides
        edge_overrides = visualization.edge_overrides

    if vis_outpath is not MISSING:
        container_path = cast(str, vis_outpath)
    if vis_save_only is not MISSING:
        save_only = cast(bool, vis_save_only)
    if vis_fileformat is not MISSING:
        file_format = cast(str, vis_fileformat)
    if vis_direction is not MISSING:
        direction = cast(VisDirectionLiteral, vis_direction)
    if vis_graph_overrides is not MISSING:
        graph_overrides = cast(dict[str, Any] | None, vis_graph_overrides)
    if vis_edge_overrides is not MISSING:
        edge_overrides = cast(dict[str, Any] | None, vis_edge_overrides)

    return trace.draw_combined(
        vis_outpath=container_path,
        vis_graph_overrides=graph_overrides,
        node_spec_fn=node_spec_fn,
        backward_node_spec_fn=backward_node_spec_fn,
        vis_edge_overrides=edge_overrides,
        vis_save_only=save_only,
        vis_fileformat=file_format,
        vis_direction=direction,
        vis_mode=vis_mode,
        intervening_cluster=intervening_cluster,
        show_buffer_layers=show_buffer_layers,
        bwd=bwd,
    )


def _bundle_node_display_label(graph_node_label: str, node: Any, vis_mode: str) -> str:
    """Return a compact Graphviz label for a bundle supergraph node.

    Parameters
    ----------
    graph_node_label:
        Canonical supergraph node name.
    node:
        Supergraph node-like object.
    vis_mode:
        Bundle visualization mode.

    Returns
    -------
    str
        Display label.
    """

    traces = ",".join(getattr(node, "traces", []))
    mode_suffix = " rolled" if vis_mode == "rolled" else ""
    op_type = getattr(node, "op_type", "") or "op"
    return f"{graph_node_label}\n{op_type}{mode_suffix}\n[{traces}]"


def _bundle_module_groups(bundle: Any) -> dict[str, list[str]]:
    """Return bundle supergraph nodes grouped by representative module path.

    Parameters
    ----------
    bundle:
        Bundle with a ``supergraph`` accessor.

    Returns
    -------
    dict[str, list[str]]
        Module path to canonical node names.
    """

    groups: dict[str, list[str]] = {}
    for graph_node_label in bundle.supergraph.topological_order:
        node = bundle.supergraph.nodes[graph_node_label]
        module_path = getattr(node, "module_path", None)
        if module_path:
            groups.setdefault(str(module_path), []).append(graph_node_label)
    return groups


def _add_bundle_forward_nodes(
    dot: Any,
    bundle: Any,
    vis_mode: str,
    node_styles: dict[str, Any] | None,
) -> None:
    """Add forward supergraph nodes to a Graphviz digraph.

    Parameters
    ----------
    dot:
        Graphviz digraph.
    bundle:
        Bundle to render.
    vis_mode:
        Bundle visualization mode.
    node_styles:
        Optional per-node style overrides.

    Returns
    -------
    None
        ``dot`` is mutated in place.
    """

    from .visualization._render_utils import (
        compute_module_penwidth,
        make_module_cluster_attrs,
        merge_node_style,
    )

    base_style = {
        "shape": "box",
        "style": "filled,rounded",
        "fillcolor": "#F7F7F7",
        "color": "#333333",
    }
    module_groups = _bundle_module_groups(bundle)
    grouped_nodes = {node for nodes in module_groups.values() for node in nodes}
    for module_path, node_names in module_groups.items():
        first_node = bundle.supergraph.nodes[node_names[0]]
        cluster_name = "cluster_bundle_" + "".join(
            char if char.isalnum() else "_" for char in module_path
        )
        with dot.subgraph(name=cluster_name) as subgraph:
            subgraph.attr(
                **make_module_cluster_attrs(
                    title=module_path,
                    module_type=getattr(first_node, "module_type", None),
                    line_style="solid",
                    penwidth=compute_module_penwidth(0, 1),
                )
            )
            for graph_node_label in node_names:
                node = bundle.supergraph.nodes[graph_node_label]
                attrs = merge_node_style(base_style, node_styles, graph_node_label, node)
                subgraph.node(
                    f"fwd_{graph_node_label}",
                    label=_bundle_node_display_label(graph_node_label, node, vis_mode),
                    **attrs,
                )
    for graph_node_label in bundle.supergraph.topological_order:
        if graph_node_label in grouped_nodes:
            continue
        node = bundle.supergraph.nodes[graph_node_label]
        attrs = merge_node_style(base_style, node_styles, graph_node_label, node)
        dot.node(
            f"fwd_{graph_node_label}",
            label=_bundle_node_display_label(graph_node_label, node, vis_mode),
            **attrs,
        )


def _add_bundle_forward_edges(
    dot: Any,
    bundle: Any,
    edge_styles: dict[tuple[str, str], Any] | None,
) -> None:
    """Add forward supergraph edges to a Graphviz digraph.

    Parameters
    ----------
    dot:
        Graphviz digraph.
    bundle:
        Bundle to render.
    edge_styles:
        Optional per-edge style overrides.

    Returns
    -------
    None
        ``dot`` is mutated in place.
    """

    from .visualization._render_utils import merge_edge_style

    base_style = {"color": "#555555", "fontcolor": "#555555"}
    for edge_key, traces in bundle.supergraph.edges.items():
        attrs = merge_edge_style(base_style, edge_styles, edge_key, {"traces": traces})
        dot.edge(
            f"fwd_{edge_key[0]}", f"fwd_{edge_key[1]}", label=",".join(sorted(traces)), **attrs
        )


def _add_bundle_backward_graph(dot: Any, bundle: Any) -> None:
    """Add per-member backward graph clusters to a Graphviz digraph.

    Parameters
    ----------
    dot:
        Graphviz digraph.
    bundle:
        Bundle to render.

    Returns
    -------
    None
        ``dot`` is mutated in place.
    """

    for member_name, member in bundle.members.items():
        with dot.subgraph(name=f"cluster_backward_{member_name}") as subgraph:
            subgraph.attr(label=f"{member_name} backward", color="#7A3E9D")
            grad_fns = list(getattr(member, "grad_fns", []))
            if not grad_fns:
                subgraph.node(
                    f"bwd_{member_name}_empty",
                    label="no backward graph",
                    shape="box",
                    style="dashed",
                    color="#7A3E9D",
                )
                continue
            visible_ids = {grad_fn_handle.grad_fn_object_id for grad_fn_handle in grad_fns}
            for grad_fn_handle in grad_fns:
                subgraph.node(
                    f"bwd_{member_name}_{grad_fn_handle.grad_fn_object_id}",
                    label=str(
                        getattr(
                            grad_fn_handle,
                            "label",
                            getattr(grad_fn_handle, "name", "grad_fn_handle"),
                        )
                    ),
                    shape="box",
                    style="filled,rounded",
                    fillcolor="#F4E8FA",
                    color="#7A3E9D",
                )
            for grad_fn_handle in grad_fns:
                for next_id in getattr(grad_fn_handle, "next_grad_fn_ids", []):
                    if next_id in visible_ids:
                        subgraph.edge(
                            f"bwd_{member_name}_{grad_fn_handle.grad_fn_object_id}",
                            f"bwd_{member_name}_{next_id}",
                            color="#7A3E9D",
                        )


def show_bundle_graph(
    bundle: Any,
    vis_outpath: str = "bundle_modelgraph",
    vis_mode: VisModeLiteral = "unrolled",
    direction: str = "forward",
    vis_direction: VisDirectionLiteral = "bottomup",
    vis_graph_overrides: dict[str, Any] | None = None,
    vis_node_overrides: dict[str, Any] | None = None,
    vis_edge_overrides: dict[tuple[str, str], Any] | None = None,
    vis_save_only: bool = False,
    vis_fileformat: str = "pdf",
) -> str | None:
    """Render a multi-trace bundle graph.

    Parameters
    ----------
    bundle:
        ``torchlens.Bundle`` instance.
    vis_outpath:
        Output path for Graphviz rendering.
    vis_mode:
        ``"rolled"``, ``"unrolled"``, or ``"none"``.
    direction:
        Graph content direction: ``"forward"``, ``"backward"``, ``"both"``, or
        ``"overlay"``.
    vis_direction:
        Graphviz layout direction.
    vis_graph_overrides:
        Graph-level Graphviz overrides.
    vis_node_overrides:
        Per-node Graphviz style overrides.
    vis_edge_overrides:
        Per-edge Graphviz style overrides keyed by ``(source, target)``.
    vis_save_only:
        If True, save without opening a viewer.
    vis_fileformat:
        Output file format.

    Returns
    -------
    str | None
        DOT source, or ``None`` when ``vis_mode='none'``.
    """

    if vis_mode == "none":
        return None
    if vis_mode not in {"rolled", "unrolled"}:
        raise ValueError("vis_mode must be 'rolled', 'unrolled', or 'none'.")
    if direction not in {"forward", "backward", "both", "overlay"}:
        raise ValueError("direction must be 'forward', 'backward', 'both', or 'overlay'.")

    import graphviz

    from .visualization._render_utils import (
        direction_to_rankdir,
        render_dot_to_file,
        strip_known_extension,
    )

    dot = graphviz.Digraph(
        name="TorchLens_Bundle",
        comment="TorchLens bundle graph",
        format=vis_fileformat,
    )
    graph_attrs = {
        "rankdir": direction_to_rankdir(vis_direction),
        "label": f"TorchLens bundle graph ({vis_mode}, {direction})",
        "labelloc": "t",
        "labeljust": "left",
        "compound": "true",
    }
    graph_attrs.update({key: str(value) for key, value in (vis_graph_overrides or {}).items()})
    dot.graph_attr.update(graph_attrs)

    if direction in {"forward", "both", "overlay"}:
        _add_bundle_forward_nodes(dot, bundle, vis_mode, vis_node_overrides)
        _add_bundle_forward_edges(dot, bundle, vis_edge_overrides)
    if direction in {"backward", "both", "overlay"}:
        _add_bundle_backward_graph(dot, bundle)
    return render_dot_to_file(
        dot,
        strip_known_extension(vis_outpath),
        vis_fileformat,
        vis_save_only,
    )


def validate_forward_pass(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    random_seed: int | None = None,
    verbose: bool = False,
    validate_metadata: bool = True,
    *,
    backend: BackendName | None = None,
) -> bool:
    """Validate that saved outs faithfully reproduce the model's output.

    Parameters
    ----------
    model:
        Model or callable to validate.
    input_args:
        Input for which to validate the saved outs.
    input_kwargs:
        Keyword arguments for model forward pass.
    random_seed:
        Fixed RNG seed for reproducibility.
    verbose:
        If True, print detailed error messages on validation failure.
    validate_metadata:
        If True, also run metadata invariant checks.
    backend:
        Explicit backend name. ``None`` preserves legacy auto-resolution.

    Returns
    -------
    bool
        True if all validation checks pass, False otherwise.
    """

    spec = resolve_backend_spec(backend, model, input_args, input_kwargs)
    return spec.validate_entry(
        model,
        input_args,
        input_kwargs=input_kwargs,
        random_seed=random_seed,
        verbose=verbose,
        validate_metadata=validate_metadata,
    )


def _validate_forward_pass_torch(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    random_seed: int | None = None,
    verbose: bool = False,
    validate_metadata: bool = True,
    *,
    num_threads: int | None = None,
    _trace_observer: Callable[[Trace], None] | None = None,
) -> bool:
    """Validate that saved outs faithfully reproduce the model's output.

    **How it works:**

    1. Run model.forward() *without* TorchLens to get ground-truth output tensors.
    2. Run ``trace`` with ``save_arg_values=True`` and ``layers_to_save='all'``
       to capture every out and its creating function's arguments.
    3. Call ``Trace.validate_forward_pass`` which replays the forward pass
       layer-by-layer from saved outs, checking that the output matches
       ground truth.  It also injects random outs and verifies the output
       changes (proving the saved outs are actually used, not just ignored).
    4. If ``validate_metadata=True``, run comprehensive invariant checks on all
       metadata cross-references (graph edges, module containment, labels, etc.).

    **Why save_arg_values=True is required:**  The validation replay re-executes
    each function using its saved non-tensor arguments (e.g., stride, padding for
    conv2d).  Without them, replay cannot reconstruct the correct computation.

    Parameters
    ----------
    model:
        PyTorch model.
    input_args:
        Input for which to validate the saved outs.
    input_kwargs:
        Keyword arguments for model forward pass.
    random_seed:
        Fixed RNG seed for reproducibility (auto-generated if None).
    verbose:
        If True, print detailed error messages on validation failure.
    validate_metadata:
        If True (default), also run metadata invariant checks.
    num_threads:
        Optional intra-op thread count for the validation forwards. ``None``
        preserves the process default; an integer pins for this harness call and
        restores the previous thread count afterward.
    _trace_observer:
        Optional private callback invoked with the completed validation trace
        after replay validation and before cleanup.

    Returns
    -------
    bool
        True if all validation checks pass, False otherwise.
    """
    warn_parallel()
    _reject_opaque_wrappers(model)
    model = unwrap_compiled_model(model)
    model = _unwrap_data_parallel(model)
    input_args = _coerce_input_args(model, input_args)
    check_model_and_input_variants(model, input_args, input_kwargs)
    # Fix a random seed so both the ground-truth run and the logged run see
    # identical randomness (critical for models with dropout, etc.).
    if random_seed is None:
        random_seed = random.randint(1, 4294967294)
    set_random_seed(random_seed)
    input_args = normalize_input_args(input_args, model)
    if not input_kwargs:
        input_kwargs = {}
    # Deep-copy inputs so the ground-truth forward pass doesn't mutate the
    # originals (some models modify inputs in-place).
    input_args_copy = safe_copy_args(input_args)
    input_kwargs_copy = safe_copy_kwargs(input_kwargs)

    model_device = next((p.device for p in model.parameters()), None)
    if model_device is not None:
        input_args_copy = _move_tensors_to_device(input_args_copy, model_device)
        input_kwargs_copy = _move_tensors_to_device(input_kwargs_copy, model_device)

    # Step 1: Get ground-truth outputs by running the model *outside* TorchLens.
    # Save state_dict first because requires_grad forcing during logging can
    # alter parameter metadata; we restore it afterward.
    state_dict = _clone_state_dict_with_metadata(model)
    trace: Trace | None = None
    outs_are_valid = False
    # Determinism stabilizer for the capture + replay + perturbation region.
    #
    # The replay-validation drift between an op's value captured INLINE in the
    # full forward and the value recomputed by ISOLATED per-op replay is, for
    # most ops, float-reduction-ORDER non-determinism in parallel CPU kernels
    # (multi-threaded conv/matmul, atomic-add scatter) -- NOT randomness (RNG is
    # already seeded above). Forcing deterministic algorithms makes the
    # ground-truth forward, the TorchLens capture, and the per-op replay use the
    # same reduction order, which removes the nondeterministic in-place-scatter
    # PERTURBATION flake (the GNN/molecular "regression" class: a wrong value
    # injected into a scatter destination sometimes produced an output
    # indistinguishable from the original under a thread race, spuriously failing
    # the sensitivity check). warn_only=True so no op raises if it lacks a
    # deterministic impl (CPU scatter_add IS deterministic in torch 2.8, so this
    # is not even exercised there, but it keeps the stabilizer safe on any op).
    #
    # In addition to deterministic algorithms, callers may pass ``num_threads``
    # to PIN an intra-op thread count (save/restore, scoped to this harness only)
    # for the duration of the validation forwards. Multi-threaded float
    # reduction-ORDER is non-deterministic ACROSS RUNS even with deterministic
    # algorithms enabled: two CLEAN forwards of the SAME nn.Module can disagree
    # by ~3e-7 at the output. That straddles the strict phase-0 ground-truth bar
    # (GROUND_TRUTH_OUTPUT_RTOL=1e-6), making the spectral-GCN family (MSTGCN /
    # TGT-MSTGCN Chebyshev sparse aggregation) FLAKY, and the same thread
    # non-determinism in MoE masked-gate routing makes the perturbation
    # sensitivity check FLAKY (minimax / nllb-moe). Pinning one thread makes both
    # bit-exact (abs diff -> exactly 0.0), so the strict bar becomes DETERMINISTIC
    # -- this does NOT loosen any tolerance, it removes the inter-run thread
    # non-determinism the bar was never meant to police.
    #
    # The menagerie validator normally runs this harness at the worker process
    # default thread count for throughput, then retries exactly the failed forward
    # validation once with ``num_threads=1``. A genuine capture/replay bug still
    # fails the single-thread retry; the known reduction-order flakes are rescued
    # by a strict bit-exact rerun instead of by loosening any tolerance.
    #
    # LOAD-BEARING: when a deterministic retry is needed, pinning at PROCESS start
    # does NOT reliably fix the validation path (the capture forward and/or model
    # internals re-parallelize); the pin must wrap the forwards INSIDE the
    # harness, which is what ``num_threads=1`` does.
    #
    # Deterministic algorithms remain on as well: the optional thread pin removes
    # inter-run reduction-order drift, while deterministic algorithms remove
    # intra-run scatter non-determinism. (The single-thread pin alone covers most
    # of it, but keeping deterministic algorithms is strictly safer and free
    # here.) The thread pin does NOT make deep CPU conv inline-vs-isolated replay
    # bit-exact (that residual is oneDNN kernel selection, not threading) -- that
    # is what the band-C reduction-depth tolerance in validation/core.py covers;
    # the changes are complementary.
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    prior_num_threads = torch.get_num_threads()
    torch.use_deterministic_algorithms(True, warn_only=True)
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    try:
        ground_truth_model, plain_attr_snapshot = _model_for_ground_truth_validation(model)
        from .backends.torch.ops import _walk_output_tensors_with_paths

        ground_truth_output = ground_truth_model(*input_args_copy, **input_kwargs_copy)
        ground_truth_output_all = [
            (tensor, tuple(path))
            for tensor, path, _container_spec in _walk_output_tensors_with_paths(
                ground_truth_output
            )
        ]
        if not ground_truth_output_all:
            ground_truth_output_all = get_vars_of_type_from_obj(
                ground_truth_output,
                torch.Tensor,
                search_depth=5,
                return_addresses=True,
                allow_repeats=True,
            )
        # Deduplicate by structural address to match how capture/trace.py extracts
        # outputs (same tensor returned in multiple positions is counted once).
        addresses_used = []
        ground_truth_output_tensors = []
        for entry in ground_truth_output_all:
            if entry[1] in addresses_used:
                continue
            # Clone/detach the ground-truth output BEFORE restoring state_dict below.
            # When the model returns a registered buffer directly (e.g. `return self.h`
            # after `self.h = ...`), the output tensor IS the live buffer object;
            # `model.load_state_dict` writes buffers in-place, which would clobber this
            # saved ground-truth reference back to its initial value and produce a
            # validation FALSE-NEGATIVE. (Inputs are already deep-copied above; outputs
            # were not.) Snapshotting the value here corrects the ground truth fed to the
            # tripwire — it does NOT weaken any check.
            ground_truth_output_tensors.append(entry[0].detach().clone())
            addresses_used.append(entry[1])
        model.load_state_dict(state_dict)
        if plain_attr_snapshot is not None:
            plain_attr_snapshot.restore_changed_attrs()

        # Step 2: Run the model *through* TorchLens, saving all outs.
        # save_arg_values=True is essential - the replay needs each function's
        # non-tensor arguments to re-execute the computation from saved outs.
        trace = _run_model_and_save_specified_outs(
            model=model,
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save="all",
            activation_transform=None,
            mark_layer_depths=False,
            detach_saved_activations=False,
            save_grads=False,
            save_arg_values=True,
            random_seed=random_seed,
            save_rng_states=True,
        )
        # Step 3: Validate by replaying the forward pass from saved outs.
        validation_result = trace.validate_forward_pass(
            ground_truth_output_tensors, verbose, validate_metadata=validate_metadata
        )
        if isinstance(validation_result, bool):
            outs_are_valid = validation_result
        else:
            outs_are_valid = bool(getattr(validation_result, "passed", False))
        if _trace_observer is not None:
            _trace_observer(trace)
    finally:
        torch.use_deterministic_algorithms(prior_deterministic, warn_only=prior_warn_only)
        if num_threads is not None:
            torch.set_num_threads(prior_num_threads)
        model.load_state_dict(state_dict)
        if "plain_attr_snapshot" in locals() and plain_attr_snapshot is not None:
            plain_attr_snapshot.restore_changed_attrs()
        if trace is not None:
            trace.cleanup()
    return outs_are_valid


def validate_backward_pass(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    loss_fn: Callable[[Any], torch.Tensor] | None = None,
    *,
    perturb_saved_grads: bool = False,
    validate_metadata: bool = True,
    random_seed: int | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    validate_layer_grads: bool = False,
    layer_grad_atol: float | None = None,
    layer_grad_rtol: float | None = None,
) -> bool:
    """Validate first-class backward capture against stock autograd.

    Parameters
    ----------
    model:
        PyTorch model.
    input_args:
        Positional args for ``model.forward()``.
    input_kwargs:
        Keyword args for ``model.forward()``.
    loss_fn:
        Optional callable mapping model outputs to a scalar loss. Defaults to
        summing all returned tensors.
    perturb_saved_grads:
        If True, perturb a saved grad and require validation to fail.
    validate_metadata:
        If True, run metadata invariant checks on the captured backward trace.
    random_seed:
        Fixed RNG seed for stock and candidate passes.
    atol:
        Absolute allclose tolerance.
    rtol:
        Relative allclose tolerance.
    validate_layer_grads:
        If True, also validate per-module-output gradients.
    layer_grad_atol:
        Optional layer-gradient absolute tolerance.
    layer_grad_rtol:
        Optional layer-gradient relative tolerance.

    Returns
    -------
    bool
        True if backward capture matches stock autograd.
    """
    from .validation.backward import validate_backward_pass as _impl

    return _impl(
        model,
        input_args,
        input_kwargs=input_kwargs,
        loss_fn=loss_fn,
        perturb_saved_grads=perturb_saved_grads,
        validate_metadata=validate_metadata,
        random_seed=random_seed,
        atol=atol,
        rtol=rtol,
        validate_layer_grads=validate_layer_grads,
        layer_grad_atol=layer_grad_atol,
        layer_grad_rtol=layer_grad_rtol,
    )


def validate_saved_outs(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    random_seed: int | None = None,
    verbose: bool = False,
    validate_metadata: bool = True,
) -> bool:
    """Deprecated alias for :func:`validate_forward_pass`."""

    warn_deprecated_alias("validate_saved_outs", "validate_forward_pass")
    return validate_forward_pass(
        model,
        input_args,
        input_kwargs,
        random_seed=random_seed,
        verbose=verbose,
        validate_metadata=validate_metadata,
    )


def validate_batch_of_models_and_inputs(
    models_and_inputs_dict: dict[str, dict[str, Any]],
    out_path: str,
    redo_model_if_already_run: bool = True,
) -> "pd.DataFrame":
    """Batch-validate multiple models, writing incremental results to a CSV.

    For each model/input pair, calls ``validate_forward_pass`` and appends the
    result to a running CSV at *out_path*.  If the CSV already exists, previously
    validated models can be skipped (controlled by *redo_model_if_already_run*).

    Parameters

    ----------
        models_and_inputs_dict: Mapping of model_class_name to a dict with keys:
            - ``model_category`` (str): grouping label (e.g. 'torchvision').
            - ``model_loading_func`` (callable): zero-arg function returning an nn.Module.
            - ``model_sample_inputs`` (dict[str, input]): named sample inputs.
        out_path: File path for the results CSV (created if absent, appended otherwise).
        redo_model_if_already_run: Re-validate models already present in the CSV.

    Returns

    -------
        DataFrame with columns: model_category, model_class_name, input_name, validation_success.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
        ) from e

    if os.path.exists(out_path):
        current_csv = pd.read_csv(out_path)
    else:
        current_csv = pd.DataFrame.from_dict(
            {
                "model_category": [],
                "model_class_name": [],
                "input_name": [],
                "validation_success": [],
            }
        )
    models_already_run = current_csv["model_class_name"].unique()
    for model_class_name, model_info in tqdm(
        models_and_inputs_dict.items(), desc="Validating models"
    ):
        print(f"Validating model {model_class_name}")
        if model_class_name in models_already_run and not redo_model_if_already_run:
            continue
        model_category = model_info["model_category"]
        model_loading_func = model_info["model_loading_func"]
        model = model_loading_func()
        model_sample_inputs = model_info["model_sample_inputs"]
        for input_name, input_data in model_sample_inputs.items():
            validation_success = validate_forward_pass(model, input_data)
            current_csv = pd.concat(
                [
                    current_csv,
                    pd.DataFrame(
                        [
                            {
                                "model_category": model_category,
                                "model_class_name": model_class_name,
                                "input_name": input_name,
                                "validation_success": validation_success,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        current_csv.to_csv(out_path, index=False)
        del model
    return current_csv
