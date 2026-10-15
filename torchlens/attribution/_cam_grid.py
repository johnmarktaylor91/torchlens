"""Token-grid derivation for native Grad-CAM on vision transformers.

A transformer layer emits a token sequence ``N, T, C`` rather than a feature
map ``N, C, H, W``. Grad-CAM needs the spatial grid behind those tokens and the
count of leading non-grid tokens (a class token, a distillation token,
register tokens). This module reads both from a metadata-only TorchLens trace
of the same forward call, from the shapes and edges of the graph alone; it
never consults a table of model names. When the graph does not prove one grid
it refuses, and the caller may pass a ``reshape`` callable instead.

Two graph patterns establish a grid:

* grid flatten: a rank-4 map with feature-map lineage (it descends from a
  rank-4 input, or from a token sequence viewed as a grid, through rank-4
  ops) has two spatial axes merged into one token axis, as a ViT patch
  embedding does (``flatten(2)`` then ``transpose``) or as Swin patch merging
  does (a channels-last ``view``). Leading tokens are then proven by a token
  concatenation whose last piece descends from those grid tokens.
* grid view: a token sequence ``N, T, C`` viewed as ``N, H, W, C`` with
  ``H * W == T``, as a Swin stage does before its windowed attention.

Every valid candidate on the layer's ancestry must agree on one grid and one
prefix count, or the call refuses.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from torch import Tensor
from torch.nn import Module

from torchlens.attribution._core import _temporarily_eval
from torchlens.attribution._result import AttributionError

TokenReshape = Callable[[Tensor], Tensor]

_DOCS_POINTER = "See docs/reference/attribution.md, section grad_cam."


@dataclass(frozen=True)
class _TokenGrid:
    """A proven token grid for one layer output.

    Attributes
    ----------
    height, width
        Spatial grid behind the patch tokens.
    prefix
        Count of leading non-grid tokens dropped before the reshape.
    evidence
        Labels of the trace ops that establish the grid.
    """

    height: int
    width: int
    prefix: int
    evidence: tuple[str, ...]


class _TraceGraph:
    """Label-indexed view of a metadata-only trace's op graph."""

    def __init__(self, trace: Any) -> None:
        """Index every op of ``trace`` by its label.

        Parameters
        ----------
        trace
            Metadata-only TorchLens trace of the attributed forward call.
        """

        self.trace = trace
        self.ops: dict[str, Any] = {}
        self.order: dict[str, int] = {}
        for index, op in enumerate(trace.ops):
            label = str(op.layer_label)
            self.ops[label] = op
            self.order[label] = index
        self._add_boundary_nodes()
        self._lineage: dict[str, bool] = {}
        input_shapes = [self.shape(label) for label in self.input_labels()]
        batches = {shape[0] for shape in input_shapes if len(shape) >= 1}
        self.batch = batches.pop() if len(batches) == 1 else None

    def _add_boundary_nodes(self) -> None:
        """Index the input, buffer and output nodes that ``trace.ops`` does not list.

        Model inputs and buffers precede every op and model outputs follow
        them, so their order is set outside the op range.
        """

        last = len(self.order)
        for op in list(self.ops.values()):
            for ref, order in [(ref, -1) for ref in (op.parents or ())] + [
                (ref, last) for ref in (op.children or ())
            ]:
                label = str(ref)
                if label not in self.ops:
                    self.ops[label] = self.trace[ref]
                    self.order[label] = order

    def label_of(self, ref: Any) -> str:
        """Return the op label for a parent or child reference."""

        label = str(ref)
        if label in self.ops:
            return label
        return str(self.trace[ref].layer_label)

    def parents(self, label: str) -> list[str]:
        """Return the parent labels of ``label`` in argument order."""

        return [self.label_of(ref) for ref in (self.ops[label].parents or ())]

    def children(self, label: str) -> list[str]:
        """Return the child labels of ``label``."""

        return [self.label_of(ref) for ref in (self.ops[label].children or ())]

    def shape(self, label: str) -> tuple[int, ...]:
        """Return the output shape of ``label`` (empty when unrecorded)."""

        shape = self.ops[label].shape
        return tuple(int(value) for value in shape) if shape is not None else ()

    def func(self, label: str) -> str:
        """Return the function name of ``label``."""

        return str(self.ops[label].func_name or "")

    def input_labels(self) -> list[str]:
        """Return the labels of the model-input ops."""

        return [
            label
            for label, op in self.ops.items()
            if op.is_input and not getattr(op, "is_buffer", False)
        ]

    def output_labels(self) -> list[str]:
        """Return the labels of the model-output ops."""

        return [label for label, op in self.ops.items() if op.is_output]

    def ancestors(self, label: str) -> set[str]:
        """Return every op that ``label`` depends on, ``label`` included."""

        seen = {label}
        queue = deque([label])
        while queue:
            for parent in self.parents(queue.popleft()):
                if parent not in seen:
                    seen.add(parent)
                    queue.append(parent)
        return seen

    def reaches_output_avoiding(self, blocked: str) -> bool:
        """Return whether any input reaches any output without passing ``blocked``."""

        outputs = set(self.output_labels())
        seen = {label for label in self.input_labels() if label != blocked}
        queue = deque(seen)
        while queue:
            current = queue.popleft()
            if current in outputs:
                return True
            for child in self.children(current):
                if child != blocked and child not in seen:
                    seen.add(child)
                    queue.append(child)
        return False

    def has_map_lineage(self, label: str) -> bool:
        """Return whether ``label`` is a rank-4 map descended from a spatial grid."""

        if label not in self._lineage:
            self._lineage[label] = False  # cycle guard; the graph is acyclic
            self._lineage[label] = self._compute_map_lineage(label)
        return self._lineage[label]

    def _compute_map_lineage(self, label: str) -> bool:
        """Compute :meth:`has_map_lineage` without the memo."""

        shape = self.shape(label)
        if len(shape) != 4 or shape[0] != self.batch:
            return False
        if self.ops[label].is_input or _grid_view(self, label) is not None:
            return True
        return any(self.has_map_lineage(parent) for parent in self.parents(label))


def _grid_view(graph: _TraceGraph, label: str) -> tuple[int, int] | None:
    """Return ``(H, W)`` when ``label`` views a token sequence ``N, T, C`` as ``N, H, W, C``."""

    parents = graph.parents(label)
    shape = graph.shape(label)
    if len(parents) != 1 or len(shape) != 4:
        return None
    source = graph.shape(parents[0])
    if len(source) != 3 or source[0] != graph.batch or shape[0] != graph.batch:
        return None
    height, width = shape[1], shape[2]
    if height > 1 and width > 1 and height * width == source[1] and shape[3] == source[2]:
        return height, width
    return None


def _grid_flatten(graph: _TraceGraph, label: str) -> tuple[int, int, str] | None:
    """Return ``(H, W, token_label)`` when ``label`` flattens a map's grid into tokens."""

    parents = graph.parents(label)
    shape = graph.shape(label)
    if len(parents) != 1 or len(shape) != 3 or not graph.has_map_lineage(parents[0]):
        return None
    source = graph.shape(parents[0])
    # Channels-last merge (Swin patch merging): N, H, W, C -> N, H*W, C.
    if shape == (source[0], source[1] * source[2], source[3]):
        return source[1], source[2], label
    # Channels-first merge (patch embedding): N, C, H, W -> N, C, H*W, then the
    # token axis moves to position 1 in a child op.
    if shape == (source[0], source[1], source[2] * source[3]):
        tokens = (source[0], source[2] * source[3], source[1])
        for child in graph.children(label):
            if graph.shape(child) == tokens:
                return source[2], source[3], child
    return None


def _prefix_from_cat(
    graph: _TraceGraph,
    region: set[str],
    token_label: str,
    tokens: int,
    patches: int,
) -> int | None:
    """Return the leading-token count proven by a token concatenation, if any.

    A valid concatenation lies on the layer's ancestry, emits ``tokens`` tokens
    along axis 1 from rank-3 pieces, and its LAST piece holds ``patches``
    tokens that descend from the grid tokens, so every leading token sits
    before the grid.
    """

    for label in region:
        shape = graph.shape(label)
        if graph.func(label) != "cat" or len(shape) != 3 or shape[1] != tokens:
            continue
        pieces = [graph.shape(parent) for parent in graph.parents(label)]
        if len(pieces) < 2 or any(len(piece) != 3 for piece in pieces):
            continue
        if sum(piece[1] for piece in pieces) != tokens or pieces[-1][1] != patches:
            continue
        if token_label in graph.ancestors(graph.parents(label)[-1]):
            return tokens - patches
    return None


def _grid_candidates(graph: _TraceGraph, layer_label: str) -> set[tuple[int, int, int, str]]:
    """Return every ``(H, W, prefix, evidence)`` the layer's ancestry proves."""

    shape = graph.shape(layer_label)
    tokens = shape[1]
    region = graph.ancestors(layer_label)
    found: set[tuple[int, int, int, str]] = set()
    for label in region:
        view = _grid_view(graph, label)
        if view is not None and view[0] * view[1] == tokens:
            found.add((view[0], view[1], 0, label))
        flatten = _grid_flatten(graph, label)
        if flatten is None:
            continue
        height, width, token_label = flatten
        patches = height * width
        if patches == tokens:
            found.add((height, width, 0, label))
        elif patches < tokens:
            prefix = _prefix_from_cat(graph, region, token_label, tokens, patches)
            if prefix is not None:
                found.add((height, width, prefix, label))
    return found


def _metadata_trace(model: Module, inputs: Any, input_kwargs: Any) -> Any:
    """Capture a metadata-only trace of the attributed forward call."""

    import torchlens as tl

    options = tl.options.CaptureOptions(layers_to_save=None)
    args = inputs if inputs is not None else ()
    with _temporarily_eval(model):
        return tl.trace(model, args, input_kwargs, capture=options)


def _module_outputs(graph: _TraceGraph) -> dict[str, tuple[int, str]]:
    """Map each module address to ``(call count, label of its last output op)``.

    A module call can mark more than one op as its output (an op and a view
    of it, say); the last of them in execution order is the value the call
    returned. Calls are counted from ``output_of_module_calls`` when the trace
    records them, else from the number of output ops.
    """

    labels: dict[str, list[str]] = {}
    calls: dict[str, set[str]] = {}
    for label, op in graph.ops.items():
        for address in tuple(op.output_of_modules or ()):
            labels.setdefault(str(address), []).append(label)
        for call in tuple(getattr(op, "output_of_module_calls", None) or ()):
            address, _, index = str(call).rpartition(":")
            calls.setdefault(address or str(call), set()).add(index)
    return {
        address: (
            len(calls.get(address, ())) or len(found),
            max(found, key=graph.order.__getitem__),
        )
        for address, found in labels.items()
    }


def _module_output_label(graph: _TraceGraph, layer: str) -> str | None:
    """Return the op that is ``layer``'s output in its single call, or ``None``."""

    count, label = _module_outputs(graph).get(layer, (0, ""))
    return label if count == 1 else None


def _guess_grid(tokens: int) -> tuple[int, int, int]:
    """Return a square ``(prefix, H, W)`` guess for the remedy spelling."""

    for prefix in range(0, min(tokens, 9)):
        side = math.isqrt(tokens - prefix)
        if side > 1 and side * side == tokens - prefix:
            return prefix, side, side
    return 0, tokens, 1


def _target_spelling(target: Any) -> str:
    """Return the ``target=`` spelling for a remedy line."""

    return f"target={target!r}" if isinstance(target, int) else "target=target"


def _unprovable(
    layer: str,
    shape: tuple[int, ...],
    target: Any,
    why: str,
    *,
    code: str,
) -> AttributionError:
    """Build the coded refusal for a token grid the trace does not prove."""

    tokens = shape[1] if len(shape) == 3 else 0
    prefix, height, width = _guess_grid(tokens) if tokens else (0, 1, 1)
    reshape = (
        f"lambda a: a[:, {prefix}:].transpose(1, 2)"
        f".reshape(a.shape[0], a.shape[2], {height}, {width})"
    )
    return AttributionError(
        f"grad_cam cannot place layer {layer!r} output of shape {shape} on a spatial grid: "
        f"{why}. Remedy: pass the grid yourself, e.g. tl.attribution.grad_cam(model, inputs, "
        f"{_target_spelling(target)}, layer={layer!r}, reshape={reshape}) "
        f"(drop {prefix} leading tokens and lay the rest on a {height}x{width} grid; change "
        f"those numbers if your grid differs). {_DOCS_POINTER}",
        code=code,
    )


def _derive_token_grid(
    graph: _TraceGraph,
    layer: str,
    activation: Tensor,
    target: Any,
) -> _TokenGrid:
    """Return the token grid the trace proves for ``layer``'s output.

    Raises
    ------
    AttributionError
        ``grad_cam_token_grid_unprovable`` when the trace does not prove one
        grid and one prefix count for the layer output.
    """

    shape = tuple(int(value) for value in activation.shape)
    label = _module_output_label(graph, layer)
    if label is None or graph.shape(label) != shape:
        raise _unprovable(
            layer,
            shape,
            target,
            "the trace has no single op for this output",
            code="grad_cam_token_grid_unprovable",
        )
    candidates = _grid_candidates(graph, label)
    grids = {(height, width, prefix) for height, width, prefix, _ in candidates}
    if not grids:
        raise _unprovable(
            layer,
            shape,
            target,
            "no patch grid on its ancestry has this token count",
            code="grad_cam_token_grid_unprovable",
        )
    if len(grids) > 1:
        options = ", ".join(f"{h}x{w} after {p} leading tokens" for h, w, p in sorted(grids))
        raise _unprovable(
            layer,
            shape,
            target,
            f"its ancestry proves several grids ({options})",
            code="grad_cam_token_grid_unprovable",
        )
    height, width, prefix = grids.pop()
    evidence = tuple(sorted({item[3] for item in candidates}, key=graph.order.__getitem__))
    return _TokenGrid(height=height, width=width, prefix=prefix, evidence=evidence)


def _grid_to_map(tensor: Tensor, grid: _TokenGrid) -> Tensor:
    """Drop leading tokens and lay ``N, T, C`` tokens out as ``N, C, H, W``."""

    patches = tensor[:, grid.prefix :, :]
    return patches.transpose(1, 2).reshape(
        tensor.shape[0], tensor.shape[2], grid.height, grid.width
    )


def _apply_reshape(reshape: TokenReshape, tensor: Tensor, layer: str) -> Tensor:
    """Apply a caller reshape and validate that it returns an ``N, C, H, W`` map."""

    if not callable(reshape):
        raise AttributionError(
            f"grad_cam reshape must be a callable; got {type(reshape).__name__}. Remedy: pass "
            "reshape=lambda a: a[:, 1:].transpose(1, 2).reshape(a.shape[0], a.shape[2], H, W) "
            f"with your grid's H and W. {_DOCS_POINTER}",
            code="grad_cam_reshape_invalid",
        )
    mapped = reshape(tensor)
    if not isinstance(mapped, Tensor) or mapped.ndim != 4 or mapped.shape[0] != tensor.shape[0]:
        got = tuple(mapped.shape) if isinstance(mapped, Tensor) else type(mapped).__name__
        raise AttributionError(
            f"grad_cam reshape for layer {layer!r} must return an N,C,H,W tensor with batch "
            f"{tensor.shape[0]}; got {got}. Remedy: return a 4D tensor such as "
            "a[:, 1:].transpose(1, 2).reshape(a.shape[0], a.shape[2], H, W). "
            f"{_DOCS_POINTER}",
            code="grad_cam_reshape_invalid",
        )
    return mapped


def _cam_feature_maps(
    model: Module,
    raw_inputs: tuple[Any, Any],
    layer: str,
    pair: tuple[Tensor, Tensor],
    target: Any,
    reshape: TokenReshape | None,
    graph: _TraceGraph | None,
) -> tuple[Tensor, Tensor, dict[str, Any]] | None:
    """Return the activation and gradient as ``N, C, H, W`` maps plus grid metadata.

    Parameters
    ----------
    model
        Attributed model (traced only when a token grid must be derived).
    raw_inputs
        The caller's ``(inputs, input_kwargs)``, used for the metadata trace.
    layer
        Layer name used for the trace lookup and error reporting.
    pair
        Captured activation and its target gradient.
    target
        Caller target, echoed in remedy spellings.
    reshape
        Optional caller callable mapping the activation to ``N, C, H, W``.
    graph
        Trace graph already captured for this call, if any.

    Returns
    -------
    tuple[Tensor, Tensor, dict[str, Any]] | None
        Mapped activation, mapped gradient and the ``grid_route``,
        ``prefix_tokens`` and ``grid_evidence`` metadata, or ``None`` when the
        activation is not a token sequence and no reshape was given (the
        caller then requires a 4D feature map).
    """

    activation, gradient = pair
    if reshape is not None:
        mapped = _apply_reshape(reshape, activation, layer)
        cells = int(mapped.shape[2]) * int(mapped.shape[3])
        dropped = int(activation.shape[1]) - cells if activation.ndim == 3 else 0
        extra = {"grid_route": "reshape", "prefix_tokens": dropped, "grid_evidence": ()}
        return mapped, _apply_reshape(reshape, gradient, layer), extra
    if activation.ndim != 3:
        return None
    if graph is None:
        graph = _TraceGraph(_metadata_trace(model, *raw_inputs))
    grid = _derive_token_grid(graph, layer, activation, target)
    extra = {"grid_route": "derived", "prefix_tokens": grid.prefix, "grid_evidence": grid.evidence}
    return _grid_to_map(activation, grid), _grid_to_map(gradient, grid), extra


def _labels(values: Iterable[str]) -> str:
    """Return a comma-separated quoted label list."""

    return ", ".join(repr(value) for value in values)


__all__: list[str] = []
