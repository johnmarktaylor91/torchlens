"""The input and output stubs of a ``module=`` focused draw.

A focused draw shows one module's ops. Each tensor that crosses the module's
edge gets a stub: an input stub stands for the outside op that made a tensor
the module reads (``from <op>``, its shape and size, and that op's module),
an output stub for the outside op that takes the module's output (``to
<op>`` and its module). Stubs sit outside the focused module's frame.

Spellings here are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ._focus_scope import module_at

if TYPE_CHECKING:
    from .._data_classes.module import Module
    from .._data_classes.trace import Trace
    from ._render_common import GraphNode

__all__ = [
    "BoundaryNode",
    "_boundary_address",
    "_boundary_lines",
    "_boundary_module_path",
    "_get_or_create_boundary_node",
]


@dataclass
class BoundaryNode:
    """Synthetic node representing a focused module boundary.

    Parameters
    ----------
    layer_label:
        DOT-safe synthetic label.
    display_label:
        Human-readable label shown in the node.
    boundary_kind:
        ``"input"`` for external upstreams, ``"output"`` for external sinks.
    children:
        Outgoing rendered labels.
    parents:
        Incoming rendered labels.
    modules:
        Module path used for cluster placement.
    lines:
        Label rows of the drawn stub: the outside op the tensor comes from or
        goes to, the tensor's shape and size (inputs), and that op's module.
        Empty rows fall back to ``display_label``.
    """

    layer_label: str
    display_label: str
    boundary_kind: str
    children: list[str]
    parents: list[str]
    modules: list[str]
    lines: tuple[str, ...] = ()
    is_buffer: bool = False
    has_input_ancestor: bool = True
    is_final_output: bool = False
    is_atomic_module: bool = False
    output_of_modules: list[str] = field(default_factory=list)
    is_input: bool = False
    is_output: bool = False
    is_terminal_bool: bool = False
    has_params: bool = False
    num_param_tensors: int = 0
    _param_logs: list[Any] = field(default_factory=list)
    param_shapes: list[tuple[Any, ...]] = field(default_factory=list)
    num_calls: int = 1
    call_index: int = 1
    type_index: int = 1
    step_index: int = 1
    shape: tuple[Any, ...] = ()
    activation_memory: str = "0 B"
    io_role: str = ""
    layer_type: str = "input"

    def __post_init__(self) -> None:
        """Fill mutable defaults and role flags."""

        self.is_input = self.boundary_kind == "input"
        self.is_output = self.boundary_kind == "output"
        self.layer_type = self.boundary_kind
        self.io_role = self.boundary_kind


def _get_or_create_boundary_node(
    boundary_nodes: dict[str, BoundaryNode],
    external_node: GraphNode,
    target_module: Module,
    *,
    trace: Trace,
    vis_mode: str,
    boundary_kind: str,
    inner_node: GraphNode,
    tensor_row: str | None = None,
) -> BoundaryNode:
    """Create or return a boundary node for one external layer next to ``inner_node``.

    One stub stands for one outside op: an input stub for the op whose output
    the focused module reads, an output stub for the op that reads the
    module's output. The stub is placed outside the focused module's frame,
    in the frame of the deepest called module that encloses the focus (the
    top level once a focused draw flattens its ancestors). ``tensor_row`` is
    the shape and size of the tensor an input stub brings.
    """

    external_label = external_node.layer_label.replace(":", "pass")
    boundary_label = f"__module_focus_{boundary_kind}_{external_label}"
    boundary = boundary_nodes.get(boundary_label)
    if boundary is not None:
        return boundary

    inner = [inner_node.layer_label]
    boundary = BoundaryNode(
        layer_label=boundary_label,
        display_label=boundary_kind,
        boundary_kind=boundary_kind,
        children=inner if boundary_kind == "input" else [],
        parents=inner if boundary_kind == "output" else [],
        modules=_boundary_module_path(target_module, vis_mode, inner_node.modules),
        lines=_boundary_lines(trace, external_node, vis_mode, boundary_kind, tensor_row),
    )
    boundary_nodes[boundary_label] = boundary
    return boundary


def _boundary_module_path(target_module: Module, vis_mode: str, stack: Any) -> list[str]:
    """Boundary cluster path: the inner op's called-module stack above the focus.

    The stack carries only modules that ran (no uncalled ModuleList), so the
    path is the part of it before the focused module itself: the stub sits
    outside the focused frame, next to the op it stands for.
    """

    addresses = [str(call).split(":", 1)[0] for call in stack]
    if target_module.address not in addresses:
        return []
    path = [str(call) for call in stack][: addresses.index(target_module.address)]
    return [call.split(":", 1)[0] for call in path] if vis_mode == "rolled" else path


def _boundary_lines(
    trace: Trace,
    external_node: GraphNode,
    vis_mode: str,
    boundary_kind: str,
    tensor_row: str | None,
) -> tuple[str, ...]:
    """Return the label rows of a focus stub standing for ``external_node``.

    ``from <op>`` (an input stub) or ``to <op>`` (an output stub) names the
    outside op by its label (the name its own node carries); an input stub adds
    ``tensor_row``, the shape and size of the tensor it brings, which no drawn
    node shows;
    the last row is that op's module (``@input.x`` or ``@output.logits`` for
    the model's own input or output).
    """

    title = external_node.layer_label
    rows = [f"{'from' if boundary_kind == 'input' else 'to'} {title}"]
    if boundary_kind == "input" and tensor_row:
        rows.append(tensor_row)
    address = _boundary_address(trace, external_node, vis_mode)
    if address is not None:
        rows.append(f"@{address}")
    return tuple(rows)


def _boundary_address(trace: Trace, external_node: GraphNode, vis_mode: str) -> str | None:
    """Return the module address an outside op's own node would show, or ``None``.

    A model input or output shows its role (``input.x``, ``output.logits``);
    an op inside a module shows its innermost module, pass-qualified in the
    unrolled view when that module ran more than once; a top-level op none.
    """

    if external_node.is_input or external_node.is_output:
        role = getattr(external_node, "io_role", None)
        return str(role) if role else None
    modules = [str(call) for call in getattr(external_node, "modules", ()) or ()]
    if not modules:
        return None
    module_pass = modules[-1]
    address = module_pass.split(":", 1)[0]
    if vis_mode == "rolled" or ":" not in module_pass:
        return address
    module = module_at(trace, address)
    return address if module is None or module.num_calls == 1 else module_pass
