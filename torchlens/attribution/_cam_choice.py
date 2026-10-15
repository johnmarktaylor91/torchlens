"""Automatic Grad-CAM layer choice from a metadata-only trace.

When ``grad_cam`` is called without ``layer``, the layer is chosen from the
trace of the same forward call by one rule:

1. Candidates are named modules that fire once and whose single output op is
   a spatial map (rank 4, more than one grid cell, descended from a spatial
   input) or a token sequence whose grid the trace proves.
2. A candidate must lie on every path from the model inputs to the model
   outputs (it dominates the output), so the map carries everything the
   target sees rather than one branch of it.
3. Among those, the deepest one (latest in execution order) whose grid
   carries a nonzero target gradient is chosen; a later layer that the
   target reads only through its prefix tokens is skipped, since its
   Grad-CAM would be identically zero. Several modules returning the same
   tensor resolve to the outermost of them.

When candidates exist but none dominates the output, the choice is
ambiguous and the call refuses with the explicit spellings to pick from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from torchlens.attribution._cam_grid import (
    _DOCS_POINTER,
    _derive_token_grid,
    _grid_to_map,
    _labels,
    _target_spelling,
    _TokenGrid,
    _TraceGraph,
)
from torchlens.attribution._core import (
    _call_model,
    _make_input_leaves,
    _PreparedInputs,
    _scalarize_output,
    _temporarily_eval,
)
from torchlens.attribution._result import AttributionError


@dataclass(frozen=True)
class _CamCandidate:
    """One module output eligible for Grad-CAM.

    Attributes
    ----------
    layer
        Module name from ``model.named_modules()``.
    label
        Trace label of the op that is the module's output.
    grid
        Proven token grid, or ``None`` for a rank-4 spatial map.
    dominates
        Whether every input-to-output path passes through the output op.
    """

    layer: str
    label: str
    grid: _TokenGrid | None
    dominates: bool


@dataclass(frozen=True)
class _LayerChoice:
    """The chosen layer and the recorded reason."""

    layer: str
    reason: str


def _single_call_outputs(graph: _TraceGraph, model: Module) -> dict[str, str]:
    """Map each once-firing named module to the label of its single output op."""

    outputs: dict[str, list[str]] = {}
    for label, op in graph.ops.items():
        for address in tuple(op.output_of_modules or ()):
            outputs.setdefault(str(address), []).append(label)
    named = {name for name, _ in model.named_modules() if name}
    return {
        name: labels[0] for name, labels in outputs.items() if name in named and len(labels) == 1
    }


def _is_spatial_map(graph: _TraceGraph, label: str) -> bool:
    """Return whether ``label`` is a rank-4 map with more than one grid cell."""

    shape = graph.shape(label)
    return graph.has_map_lineage(label) and shape[2] * shape[3] > 1


def _candidate_for(
    graph: _TraceGraph,
    layer: str,
    label: str,
    target: Any,
) -> _CamCandidate | None:
    """Return the candidate for one module output, or ``None`` when ineligible."""

    shape = graph.shape(label)
    grid: _TokenGrid | None = None
    if len(shape) == 4:
        if not _is_spatial_map(graph, label):
            return None
    elif len(shape) == 3 and shape[0] == graph.batch:
        try:
            grid = _derive_token_grid(graph, layer, torch.empty(shape, device="meta"), target)
        except AttributionError:
            return None
    else:
        return None
    dominates = not graph.reaches_output_avoiding(label)
    return _CamCandidate(layer=layer, label=label, grid=grid, dominates=dominates)


def _outermost_per_op(candidates: list[_CamCandidate]) -> list[_CamCandidate]:
    """Keep one candidate per output op: the outermost module (shortest name)."""

    best: dict[str, _CamCandidate] = {}
    for candidate in candidates:
        kept = best.get(candidate.label)
        if kept is None or len(candidate.layer) < len(kept.layer):
            best[candidate.label] = candidate
    return list(best.values())


def _grid_gradients(
    model: Module,
    inputs: _PreparedInputs,
    target: Any,
    candidates: list[_CamCandidate],
) -> dict[str, bool]:
    """Return, per candidate layer, whether the target gradient on its grid is nonzero."""

    named = dict(model.named_modules())
    captured: dict[str, Tensor] = {}
    handles: list[RemovableHandle] = []

    def _hook_for(name: str) -> Any:
        def _hook(_module: Module, _args: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, Tensor):
                captured[name] = output

        return _hook

    for candidate in candidates:
        handles.append(named[candidate.layer].register_forward_hook(_hook_for(candidate.layer)))
    try:
        with _temporarily_eval(model):
            leaves = _make_input_leaves(inputs)
            scalar = _scalarize_output(_call_model(model, inputs, leaves), target)
            names = [name for name, value in captured.items() if value.requires_grad]
            grads = torch.autograd.grad(
                scalar, [captured[name] for name in names], allow_unused=True
            )
    finally:
        for handle in handles:
            handle.remove()
    by_layer = {candidate.layer: candidate for candidate in candidates}
    nonzero: dict[str, bool] = {}
    for name, grad in zip(names, grads, strict=True):
        grid = by_layer[name].grid
        if grad is not None and grid is not None:
            grad = _grid_to_map(grad, grid)
        nonzero[name] = grad is not None and bool((grad != 0).any())
    return nonzero


def _reason(candidate: _CamCandidate, graph: _TraceGraph, skipped: list[str]) -> str:
    """Return the recorded reason for a chosen layer."""

    if candidate.grid is None:
        shape = graph.shape(candidate.label)
        kind = f"spatial map {shape[2]}x{shape[3]}"
    else:
        grid = candidate.grid
        kind = f"token grid {grid.height}x{grid.width} after {grid.prefix} leading tokens"
    reason = (
        f"deepest module output that is a {kind}, lies on every input-to-output path, "
        "and carries a nonzero target gradient on its grid"
    )
    if skipped:
        reason += f"; deeper candidates with a zero grid gradient: {_labels(skipped)}"
    return reason


def _refuse_no_choice(target: Any, why: str, options: list[str], code: str) -> AttributionError:
    """Build the coded refusal for an automatic choice that cannot be made."""

    if options:
        remedy = (
            f"Remedy: name the layer, e.g. tl.attribution.grad_cam(model, inputs, "
            f"{_target_spelling(target)}, layer={options[0]!r}); other candidates: "
            f"{_labels(options[1:6]) or 'none'}."
        )
    else:
        remedy = (
            "Remedy: name a layer whose output is N,C,H,W, e.g. tl.attribution.grad_cam("
            f"model, inputs, {_target_spelling(target)}, layer='<module name>'), or pass "
            "reshape= with it."
        )
    return AttributionError(
        f"grad_cam could not choose a layer automatically: {why}. {remedy} {_DOCS_POINTER}",
        code=code,
    )


def _choose_cam_layer(
    model: Module,
    inputs: _PreparedInputs,
    target: Any,
    graph: _TraceGraph,
) -> _LayerChoice:
    """Choose the Grad-CAM layer for ``model`` from its trace.

    Raises
    ------
    AttributionError
        ``grad_cam_layer_choice_ambiguous`` when candidates exist but none lies
        on every input-to-output path; ``grad_cam_layer_unchosen`` when no
        module output is a spatial map or a provable token grid that the
        target depends on.
    """

    found = [
        candidate
        for layer, label in _single_call_outputs(graph, model).items()
        if (candidate := _candidate_for(graph, layer, label, target)) is not None
    ]
    deepest_first = sorted(
        _outermost_per_op(found), key=lambda item: graph.order[item.label], reverse=True
    )
    dominators = [candidate for candidate in deepest_first if candidate.dominates]
    if not dominators:
        options = [candidate.layer for candidate in deepest_first]
        why = (
            "several module outputs are spatial maps or token grids, but none lies on every "
            "path from the inputs to the outputs"
            if options
            else "no module output is a spatial map or a provable token grid"
        )
        code = "grad_cam_layer_choice_ambiguous" if options else "grad_cam_layer_unchosen"
        raise _refuse_no_choice(target, why, options, code)
    nonzero = _grid_gradients(model, inputs, target, dominators)
    skipped: list[str] = []
    for candidate in dominators:
        if nonzero.get(candidate.layer, False):
            return _LayerChoice(candidate.layer, _reason(candidate, graph, skipped))
        skipped.append(candidate.layer)
    raise _refuse_no_choice(
        target,
        "the target gradient is zero on the grid of every candidate layer",
        [candidate.layer for candidate in dominators],
        "grad_cam_layer_unchosen",
    )


__all__: list[str] = []
