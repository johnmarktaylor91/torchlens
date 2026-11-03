"""Bodies of the public graph verbs (``show_model_graph``, ``draw_backward``,
``draw_combined``, ``show_bundle_graph``).

Loaded by :mod:`torchlens._user_public_impls` on first use of one of these
verbs, so the renderers behind them stay off the path of the other verbs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import nn

from . import _user_funcs as _user_funcs
from ._capture_state_helpers import (
    _reject_opaque_wrappers,
    _unwrap_data_parallel,
    unwrap_compiled_model,
)
from ._data_classes.trace import Trace
from ._deprecations import MISSING, MissingType
from ._input_coerce import _coerce_input_args
from ._literals import (
    BufferVisibilityLiteral,
    CollapseLiteral,
    FoldRepeatsLiteral,
    VisDirectionLiteral,
    VisModeLiteral,
    VisNodeModeLiteral,
    VisNodePlacementLiteral,
    VisRendererLiteral,
)
from ._robustness import check_model_and_input_variants
from ._visualization._bundle_graph import BundlePictureOptions, render_bundle_graph
from ._visualization._render_utils import direction_to_rankdir
from ._visualization.code_panel import CodePanelOption
from .options import (
    VisualizationOptions,
    merge_visualization_options,
    visualization_to_render_kwargs,
)

# FACADE by role, like _user_public_impls: the body of public verbs, whose
# upward calls into the renderers are the facade's own (Rule F).
__tl_layer__ = "FACADE"

if TYPE_CHECKING:
    from ._data_classes.module import Module


def show_model_graph(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    view: VisModeLiteral | MissingType = MISSING,
    depth: int | MissingType = MISSING,
    renderer: VisRendererLiteral | MissingType = MISSING,
    layout: VisNodePlacementLiteral | MissingType = MISSING,
    node_style: VisNodeModeLiteral | MissingType = MISSING,
    module: Module | str | None = None,
    collapse: CollapseLiteral | MissingType = MISSING,
    fold_repeats: FoldRepeatsLiteral | MissingType = MISSING,
    order_siblings: bool | MissingType = MISSING,
    code_panel: CodePanelOption = False,
    seed: int | None = None,
    verbose: bool = False,
    recurrence_detection: bool | MissingType = MISSING,
    visualization: VisualizationOptions | None = None,
) -> None:
    """Convenience wrapper: visualize the computational graph without saving outs.

    Runs an exhaustive forward pass (no outs saved) to discover the graph
    structure, renders the visualization, then cleans up the Trace.  For more
    control, use ``trace`` and call ``Trace.draw`` on the result directly.

    Parameters
    ----------
    model:
        PyTorch model.
    input_args:
        Positional args for ``model.forward()``.
    input_kwargs:
        Keyword args for ``model.forward()``.
    module:
        Optional module focus. Pass a Module or module address string to render
        only layers that ran inside that module.
    order_siblings:
        Whether Graphviz ``dot`` renders should verify and apply
        execution-order placement for true parallel siblings.
    code_panel:
        Optional source-code panel mode. ``True`` is equivalent to
        ``"forward"``. Built-in modes use source captured at log time; callable
        modes receive the live model object and are only available while that
        object is still alive.
    collapse:
        Smart module-collapse mode: ``"none"``, ``"auto"``, ``"max"``, or a
        float in ``[0.0, 1.0]``. Float levels follow the public monotone
        schedule.
    fold_repeats:
        Repeat-fold policy. ``None`` preserves the default policy. ``True`` folds
        every eligible repeated run. ``False`` disables run folding.
    seed:
        Fixed RNG seed for stochastic models. The capture reseeds the global
        RNG engines for its forward and restores their prior states afterwards;
        see ``CaptureOptions.seed``.
    recurrence_detection:
        If True, run full isomorphic subgraph expansion. Set this to False when
        the forward pass has more than about 1M operations and postprocessing
        speed matters.
    visualization:
        Grouped visualization options. When omitted, ``show_model_graph``
        defaults to ``VisualizationOptions(view="unrolled")``.

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
        collapse=collapse,
        fold_repeats=fold_repeats,
        order_siblings=order_siblings,
    )

    if visualization_options.view not in ["none", "rolled", "unrolled"]:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    trace = _user_funcs._run_model_and_save_specified_outs(
        model=model,
        input_args=input_args,
        input_kwargs=input_kwargs,
        layers_to_save=None,
        activation_transform=None,
        mark_layer_depths=False,
        detach_saved_activations=False,
        save_grads=False,
        seed=seed,
        recurrence_detection=recurrence_detection_enabled,
        verbose=verbose,
    )
    # Render in a try/finally so temporary TorchLens metadata on the model is
    # always cleaned up, even if Graphviz rendering raises.
    try:
        render_kwargs = visualization_to_render_kwargs(visualization_options)
        if module is not None:
            from ._data_classes.module import Module

            render_kwargs["module"] = module.address if isinstance(module, Module) else module
        if code_panel is not False:
            render_kwargs["code_panel"] = code_panel
        trace.draw(**render_kwargs)
    finally:
        trace.cleanup()


def draw_backward(
    trace: Trace,
    node_spec_fn: Callable[[Any, Any], Any] | None = None,
    collapsed_node_spec_fn: Callable[[Any, Any], Any] | None = None,
    node_style: VisNodeModeLiteral | MissingType = MISSING,
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
        or ``trace.capture_backward()``.
    node_spec_fn:
        Optional callback receiving ``(grad_fn_handle, default_spec)``.
    collapsed_node_spec_fn:
        Accepted for forward-visualization API symmetry. Not applied because
        backward graphs do not render collapsed module nodes.
    node_style:
        Node-style preset applied to grad_fn nodes.
    code_panel:
        Optional source-code panel mode.
    vis_mode:
        ``"rolled"`` renders one node per GradFn; ``"unrolled"`` renders one
        node per GradFnCall grouped by backward pass.
    bwd:
        Optional one-based backward pass number or numbers to render.
    visualization:
        Grouped visualization options. Only output path, save behavior, file
        format, direction, skin (``theme``), graph overrides, and edge
        overrides are used.

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
        theme = "torchlens"
    else:
        container_path = visualization.container_path
        save_only = visualization.save_only
        file_format = visualization.file_format
        direction = visualization.direction
        graph_overrides = visualization.graph_overrides
        edge_overrides = visualization.edge_overrides
        node_mode = visualization.node_style
        theme = visualization.theme

    if node_style is not MISSING:
        node_mode = cast(VisNodeModeLiteral, node_style)

    return trace.draw_backward(
        path=container_path,
        vis_graph_overrides=graph_overrides,
        node_spec_fn=node_spec_fn,
        collapsed_node_spec_fn=collapsed_node_spec_fn,
        vis_node_mode=node_mode,
        vis_edge_overrides=edge_overrides,
        vis_save_only=save_only,
        file_format=file_format,
        vis_direction=direction,
        code_panel=code_panel,
        vis_mode=vis_mode,
        bwd=bwd,
        theme=theme,
    )


def draw_combined(
    trace: Trace,
    node_spec_fn: Callable[[Any, Any], Any] | None = None,
    backward_node_spec_fn: Callable[[Any, Any], Any] | None = None,
    vis_mode: VisModeLiteral = "unrolled",
    intervening_cluster: Literal["upstream", "outside", "downstream", "own"] = "upstream",
    show_buffers: BufferVisibilityLiteral = "meaningful",
    bwd: int | Iterable[int] | None = None,
    visualization: VisualizationOptions | None = None,
) -> str:
    """Render an existing Trace's forward ops and backward grad_fns together.

    Parameters
    ----------
    trace:
        Trace with backward metadata captured by ``trace.log_backward(loss)``
        or ``trace.capture_backward()``.
    node_spec_fn:
        Optional callback receiving ``(layer_log, default_spec)``.
    backward_node_spec_fn:
        Optional callback receiving ``(grad_fn_handle, default_spec)``.
    vis_mode:
        Combined rendering currently supports only ``"unrolled"``.
    intervening_cluster:
        Placement mode for grad_fns without a corresponding forward op.
    show_buffers:
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

    return trace.draw_combined(
        path=container_path,
        vis_graph_overrides=graph_overrides,
        node_spec_fn=node_spec_fn,
        backward_node_spec_fn=backward_node_spec_fn,
        vis_edge_overrides=edge_overrides,
        vis_save_only=save_only,
        file_format=file_format,
        vis_direction=direction,
        vis_mode=vis_mode,
        intervening_cluster=intervening_cluster,
        show_buffers=show_buffers,
        bwd=bwd,
    )


def show_bundle_graph(
    bundle: Any,
    path: str = "bundle_modelgraph",
    vis_mode: VisModeLiteral = "unrolled",
    signal: str = "forward",
    vis_direction: VisDirectionLiteral = "bottomup",
    vis_graph_overrides: dict[str, Any] | None = None,
    vis_node_overrides: dict[str, Any] | None = None,
    vis_edge_overrides: dict[tuple[str, str], Any] | None = None,
    vis_save_only: bool = False,
    file_format: str = "pdf",
    nodes: Literal["ops", "members"] = "ops",
    theme: str = "torchlens",
    show_legend: bool | None = None,
) -> str | None:
    """Render a multi-trace bundle graph.

    Parameters
    ----------
    bundle:
        ``torchlens.Bundle`` instance.
    path:
        Output path for Graphviz rendering.
    vis_mode:
        ``"rolled"``, ``"unrolled"``, or ``"none"``.
    signal:
        Which pass's graph to draw: ``"forward"``, ``"backward"``, ``"both"``, or
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
    file_format:
        Output file format.
    nodes:
        What the picture's nodes are (spelling documented-unstable).
        ``"ops"`` (default): the bundle's union op graph. ``"members"``: one
        node per member and one edge per relation row, from the earlier
        member to the later one, labelled with the kind, the row's
        ``label`` and, on ``fed_by``, its basis and coverage; solid edges
        carry token spans TorchLens checked, every other row is dashed.
        Node overrides key by member name, edge overrides by
        ``(earlier, later)``; ``signal`` must stay ``"forward"``.
    theme:
        Skin preset, the names ``Trace.draw`` takes: ``"torchlens"`` (default),
        ``"paper"``, ``"dark"``, ``"colorblind"`` or ``"high_contrast"``.
        Op nodes take ``draw``'s shapes and fills; a node or arrow every
        member runs carries no member list.
    show_legend:
        ``True`` draws the key under the header, as ``draw`` does.

    Returns
    -------
    str | None
        DOT source, or ``None`` when ``vis_mode='none'``.
    """

    if nodes not in {"ops", "members"}:
        raise ValueError("nodes must be 'ops' or 'members'.")
    if nodes == "members" and signal != "forward":
        raise ValueError("nodes='members' draws the relation graph; leave signal='forward'.")
    if vis_mode == "none":
        return None
    if vis_mode not in {"rolled", "unrolled"}:
        raise ValueError("vis_mode must be 'rolled', 'unrolled', or 'none'.")
    if signal not in {"forward", "backward", "both", "overlay"}:
        raise ValueError("signal must be 'forward', 'backward', 'both', or 'overlay'.")

    options = BundlePictureOptions(
        outpath=path,
        vis_mode=vis_mode,
        direction=signal,
        rankdir=direction_to_rankdir(vis_direction),
        graph_overrides=vis_graph_overrides,
        node_overrides=vis_node_overrides,
        edge_overrides=vis_edge_overrides,
        save_only=vis_save_only,
        fileformat=file_format,
        nodes=nodes,
        theme_name=theme,
        show_legend=show_legend is True,
    )
    return render_bundle_graph(bundle, options)
