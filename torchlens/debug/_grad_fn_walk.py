"""Post-hoc ``grad_fn`` walker (lane F37; approved diagnostic, R grad_fn ruling).

``make_dot(y)``-class capability: draw the autograd graph of a tensor you
ALREADY HAVE -- computed in an earlier notebook cell, inside someone else's
training loop -- with no re-execution and no capture context. The walk reads
``tensor.grad_fn`` / ``next_functions`` only; it is explicitly STRUCTURE-ONLY
(no values, no timing, no verification) and every render carries that legend.

Spellings are DOCUMENTED-UNSTABLE pending the naming session (the memo's
``sketch_grad_fn`` name was an explicit placeholder). The namespace is
``tl.debug`` by ruling (diagnostic one-off, like ``bisect_nan``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..errors import CaptureError

if TYPE_CHECKING:
    import graphviz

__all__ = [
    "GradFnNode",
    "GradFnSketch",
    "GradFnWalkError",
    "sketch_grad_fn",
    "walk_grad_fn",
]

# Defensive ceiling, disclosed on truncation (legend line + ``truncated``
# flag). Real losses on real models stay far below it; a runaway generated
# graph must not stall a diagnostic one-off.
GRAD_FN_WALK_MAX_NODES = 20_000


class GradFnWalkError(CaptureError):
    """A tensor without a walkable autograd graph was passed to the walker."""


@dataclass(frozen=True)
class GradFnNode:
    """One node of a structure-only ``grad_fn`` sketch.

    Parameters
    ----------
    node_id:
        Stable id within this sketch (the ``id()`` of the grad_fn / tensor).
    kind:
        ``"op"`` (a grad_fn), ``"leaf"`` (``AccumulateGrad`` variable), or
        ``"root"`` (a walked output tensor).
    name:
        grad_fn class name (``AddmmBackward0``), ``"AccumulateGrad"``, or the
        root tensor's marker.
    param_name:
        Resolved parameter name for leaves when a ``params`` mapping / model
        was supplied; ``None`` otherwise.
    shape:
        Variable/tensor shape for leaves and roots; ``None`` for op nodes.
    dtype:
        Tensor dtype string for leaves and roots; ``None`` for op nodes.
    requires_grad:
        Leaf variable ``requires_grad`` (trainable vs frozen); ``None``
        elsewhere.
    """

    node_id: int
    kind: str
    name: str
    param_name: str | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    requires_grad: bool | None = None


@dataclass(frozen=True)
class GradFnSketch:
    """Structure-only autograd-graph sketch of already-computed tensors.

    Parameters
    ----------
    nodes:
        Every walked node.
    edges:
        ``(producer_node_id, consumer_node_id)`` pairs in FORWARD dataflow
        direction (``next_functions`` reversed for readability).
    truncated:
        Whether the defensive node ceiling stopped the walk early.
    """

    nodes: tuple[GradFnNode, ...]
    edges: tuple[tuple[int, int], ...]
    truncated: bool

    @property
    def n_nodes(self) -> int:
        """Number of walked nodes."""

        return len(self.nodes)


def _normalize_outputs(outputs: Any) -> list[torch.Tensor]:
    """Return the walkable tensor list, refusing typed when nothing walks."""

    if isinstance(outputs, torch.Tensor):
        tensors = [outputs]
    elif isinstance(outputs, (list, tuple)):
        tensors = [t for t in outputs if isinstance(t, torch.Tensor)]
    else:
        tensors = []
    walkable = [t for t in tensors if t.grad_fn is not None]
    if not walkable:
        raise GradFnWalkError(
            "No walkable autograd graph: the value has no grad_fn. The walker "
            "sketches tensors that were COMPUTED with autograd enabled -- a "
            "detached tensor, a leaf, or a value produced under torch.no_grad() "
            "carries no graph. Recompute the tensor with requires_grad inputs "
            "(outside no_grad/inference_mode) and pass that result. "
            "Remedy: pass a tensor whose .grad_fn is not None.",
            code="grad_fn_walk_no_graph",
        )
    return walkable


def _param_names_by_id(
    params: Mapping[str, torch.Tensor] | None,
    model: Any | None,
) -> dict[int, str]:
    """Build the ``id(variable) -> name`` map for leaf naming."""

    named: dict[int, str] = {}
    if model is not None and hasattr(model, "named_parameters"):
        for name, param in model.named_parameters():
            named[id(param)] = name
    if params is not None:
        for name, value in params.items():
            if isinstance(value, torch.Tensor):
                named[id(value)] = str(name)
    return named


def walk_grad_fn(
    outputs: Any,
    *,
    params: Mapping[str, torch.Tensor] | None = None,
    model: Any | None = None,
    max_nodes: int = GRAD_FN_WALK_MAX_NODES,
) -> GradFnSketch:
    # Cognitive complexity ~16 accepted by design: the root seeding and the
    # bounded DFS share visited/edge state; the walk reads as one unit and
    # the truncation rule must sit inside the loop it bounds.
    """Walk ``outputs``' autograd graph into a structure-only sketch.

    Parameters
    ----------
    outputs:
        A tensor with a ``grad_fn`` (a real loss tensor, a logits tensor, an
        ``autograd.grad(..., create_graph=True)`` result) or a sequence of
        them.
    params:
        Optional ``name -> tensor`` mapping (the torchviz convention,
        ``dict(model.named_parameters())``) used to name ``AccumulateGrad``
        leaves.
    model:
        Optional module whose ``named_parameters()`` provide leaf names
        (composes with ``params``; explicit ``params`` wins on collisions).
    max_nodes:
        Defensive walk ceiling; hitting it sets ``truncated`` and is
        disclosed on every render.

    Returns
    -------
    GradFnSketch
        Frozen structure-only record (nodes, forward-direction edges,
        truncation flag). No values, timing, or verification.
    """

    walkable = _normalize_outputs(outputs)
    names_by_id = _param_names_by_id(params, model)

    nodes: dict[int, GradFnNode] = {}
    edges: list[tuple[int, int]] = []
    truncated = False

    stack: list[Any] = []
    for tensor in walkable:
        root_id = id(tensor)
        nodes[root_id] = GradFnNode(
            node_id=root_id,
            kind="root",
            name=f"output {tuple(tensor.shape)}",
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
        )
        fn = tensor.grad_fn
        fn_id = id(fn)
        edges.append((fn_id, root_id))
        if fn_id not in nodes:
            stack.append(fn)
            nodes[fn_id] = _op_node(fn, names_by_id)

    while stack:
        fn = stack.pop()
        for parent, _input_index in getattr(fn, "next_functions", ()) or ():
            if parent is None:
                continue
            parent_id = id(parent)
            edges.append((parent_id, id(fn)))
            if parent_id in nodes:
                continue
            if len(nodes) >= max_nodes:
                truncated = True
                nodes[parent_id] = _op_node(parent, names_by_id)
                continue
            nodes[parent_id] = _op_node(parent, names_by_id)
            stack.append(parent)

    return GradFnSketch(
        nodes=tuple(nodes.values()),
        edges=tuple(dict.fromkeys(edges)),
        truncated=truncated,
    )


def _op_node(fn: Any, names_by_id: dict[int, str]) -> GradFnNode:
    """Build the node record for one grad_fn (op or AccumulateGrad leaf)."""

    name = type(fn).__name__
    variable = getattr(fn, "variable", None)
    if isinstance(variable, torch.Tensor):
        return GradFnNode(
            node_id=id(fn),
            kind="leaf",
            name=name,
            param_name=names_by_id.get(id(variable)),
            shape=tuple(variable.shape),
            dtype=str(variable.dtype),
            requires_grad=bool(variable.requires_grad),
        )
    return GradFnNode(node_id=id(fn), kind="op", name=name)


_STRUCTURE_ONLY_LEGEND = (
    "structure-only sketch of grad_fn graph: no values, timing, or verification"
)


def _sketch_digraph(sketch: GradFnSketch) -> graphviz.Digraph:
    """Render a sketch into a themed ``graphviz.Digraph`` (TL palette + legend)."""

    import graphviz

    from ..visualization._render_common import (
        BACKWARD_NODE_BORDER_COLOR,
        BACKWARD_NODE_COLOR,
        FROZEN_PARAMS_BG_COLOR,
        OUTPUT_COLOR,
        TRAINABLE_PARAMS_BG_COLOR,
    )

    legend = _STRUCTURE_ONLY_LEGEND
    if sketch.truncated:
        legend += f"\\nWALK TRUNCATED at {sketch.n_nodes} nodes -- graph is incomplete"
    dot = graphviz.Digraph(
        graph_attr={
            "label": legend,
            "labelloc": "b",
            "fontsize": "10",
            "rankdir": "TB",
        }
    )
    for node in sketch.nodes:
        node_key = str(node.node_id)
        if node.kind == "root":
            dot.node(
                node_key,
                label=f"{node.name}\\n{node.dtype}",
                shape="box",
                style="filled,rounded",
                fillcolor=OUTPUT_COLOR,
            )
        elif node.kind == "leaf":
            title = node.param_name or node.name
            shape_line = f"\\n{node.shape}" if node.shape is not None else ""
            fill = TRAINABLE_PARAMS_BG_COLOR if node.requires_grad else FROZEN_PARAMS_BG_COLOR
            dot.node(
                node_key,
                label=f"{title}{shape_line}",
                shape="box",
                style="filled",
                fillcolor=fill,
            )
        else:
            dot.node(
                node_key,
                label=node.name,
                shape="box",
                style="filled,rounded",
                fillcolor=BACKWARD_NODE_COLOR,
                color=BACKWARD_NODE_BORDER_COLOR,
            )
    for src, dst in sketch.edges:
        dot.edge(str(src), str(dst))
    return dot


def sketch_grad_fn(  # noqa: PLR0913 - torchviz-parity public door: each keyword mirrors one documented torchviz/graphviz knob (path/format/params/model/ceiling/view); bundling them would break the migration parity the function exists for
    outputs: Any,
    path: str | None = None,
    *,
    file_format: str = "png",
    params: Mapping[str, torch.Tensor] | None = None,
    model: Any | None = None,
    max_nodes: int = GRAD_FN_WALK_MAX_NODES,
    view: bool = False,
) -> str:
    """Draw the autograd graph of an already-computed tensor (torchviz parity).

    Builds :func:`walk_grad_fn`'s structure-only sketch and renders it through
    the bounded TorchLens graphviz runner (a wedged ``dot`` cannot hang the
    caller; renders are time-bounded and atomically published). The render
    always carries the structure-only legend, plus a truncation disclosure
    when the defensive walk ceiling fired.

    Parameters
    ----------
    outputs:
        Tensor(s) with a ``grad_fn``; see :func:`walk_grad_fn`.
    path:
        Output path WITHOUT extension (torchviz/graphviz convention:
        ``path.<file_format>`` is written). ``None`` skips file rendering and
        returns DOT source only.
    file_format:
        Graphviz output format (``png``, ``svg``, ``pdf``).
    params:
        Optional ``name -> tensor`` leaf-naming mapping.
    model:
        Optional module providing leaf names via ``named_parameters()``.
    max_nodes:
        Defensive walk ceiling (disclosed on truncation).
    view:
        Open the rendered file in the managed viewer (never a raw Popen).

    Returns
    -------
    str
        The DOT source of the sketch.
    """

    sketch = walk_grad_fn(outputs, params=params, model=model, max_nodes=max_nodes)
    dot = _sketch_digraph(sketch)
    if path is None:
        return str(dot.source)
    from ..visualization._render_utils import render_dot_to_file

    return render_dot_to_file(dot, path, file_format, save_only=not view)
