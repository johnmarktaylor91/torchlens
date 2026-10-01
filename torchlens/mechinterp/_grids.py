"""The generic patching-grid driver + thin presets (mikit section 7).

Rides the live-hook rerun route (the shipped ``torchlens.semantic.patching``
primitives, receipts included): every cell earns a positive fire ledger --
a hook that did not fire is an EXCEPTION, never a number -- plus donor
difference accounting, transactional model/RNG/hook cleanup, and metric
provenance. Cost honesty: the cell count and a measured-baseline time
estimate are printed BEFORE the run, and a hard ``budget=`` refuses rather
than running for a day. ``engine="replay"`` refuses typed until FIX-A lands
(exact path patching then becomes a one-line addition, not a redesign).
Degeneracy WARNS (mikit D21): a genuinely flat effect is a finding;
refusals are reserved for receipt failures.

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import torch

from ..errors._base import TorchLensWarning
from ..intervention.selectors import facet as facet_selector
from ..semantic.patching import (
    _baseline_traces,
    _CounterfactualStateGuard,
    _facet_tensor,
    _metric_scalar,
    _modules_with_facet,
    _run_patch,
    _teardown,
    _warn_if_campaign_all_identical,
)
from ._errors import refuse
from ._records import _pandas_frame

__all__ = ["PatchGrid", "grid", "patch_heads_grid", "patch_residual_grid"]

#: Head-axis index per facet layout (captured layouts, per the recipe docs).
_HEAD_AXIS = {"z": 1, "pattern": 1, "scores": 1, "result": 2, "q": 2, "k": 2, "v": 2}

_AXES = ("layer", "position", "head")


@dataclass(frozen=True)
class PatchGrid:
    """A labelled patch grid: values + coordinates + receipts.

    ``values`` is shaped by ``axes`` (e.g. ``[layer, position]``);
    ``coordinates`` maps each axis to its labels (module addresses carry
    site-key-bearing provenance through the trace itself). ``receipts``
    records fires, identical-value fires, baselines, and the disclosed cost.
    """

    axes: tuple[str, ...]
    coordinates: dict[str, tuple[Any, ...]]
    values: torch.Tensor
    receipts: dict[str, Any] = field(default_factory=dict)

    def to_pandas(self) -> Any:
        """Return the long-form DataFrame (one row per cell)."""

        rows = []
        flat = self.values.reshape(-1)
        shapes = [len(self.coordinates[axis]) for axis in self.axes]
        for index in range(flat.numel()):
            remainder = index
            coordinate = {}
            for axis, size in zip(reversed(self.axes), reversed(shapes), strict=True):
                coordinate[axis] = self.coordinates[axis][remainder % size]
                remainder //= size
            rows.append({**coordinate, "metric": float(flat[index])})
        return _pandas_frame(rows)


def _estimate_cost(n_cells: int, baseline_seconds: float) -> str:
    """Return the pre-run cost disclosure line."""

    total = n_cells * baseline_seconds
    return f"patching grid: {n_cells} cells x ~{baseline_seconds:.2f}s/rerun ~= {total:.0f}s total"


def grid(  # noqa: PLR0913 -- the memo-normative driver signature (section 7)
    model: Any,
    clean_input: Any,
    corrupted_input: Any,
    metric: Any,
    *,
    sites: str = "resid_pre",
    axes: tuple[str, ...] = ("layer", "position"),
    positions: Any = None,
    heads: Any = None,
    budget: int | None = None,
    engine: str = "rerun",
    trace_kwargs: Any = None,
) -> PatchGrid:
    """Run a labelled activation-patching grid (mikit section 7).

    Patches CLEAN values into CORRUPTED reruns cell by cell (the standard
    denoising direction) and scores each patched run with ``metric``.

    Parameters
    ----------
    model:
        Model to trace and rerun (state transactionally restored).
    clean_input / corrupted_input:
        The two prompts; length mismatch refuses at the facet shapes.
    metric:
        Callable ``Trace -> scalar tensor``.
    sites:
        Facet to patch (``resid_pre`` / ``attn_out`` / ``z`` / ...).
    axes:
        Grid axes from {layer, position, head}; ``layer`` is always present.
    positions:
        Position indices for the position axis (default: all).
    heads:
        Head indices for the head axis (default: all).
    budget:
        Hard cell-count ceiling; exceeding it REFUSES before any rerun.
    engine:
        ``"rerun"`` (live-hook route). ``"replay"`` refuses typed until the
        known HF replay crash (FIX-A) lands.
    trace_kwargs:
        Extra kwargs forwarded to ``tl.trace``.
    """

    _validate_grid_request(engine, axes, sites)

    guard = _CounterfactualStateGuard(model)
    guard.open()
    clean_log = corrupted_log = None
    try:
        started = time.perf_counter()
        clean_log, corrupted_log = _baseline_traces(
            model,
            clean_input,
            corrupted_input,
            trace_kwargs=dict(trace_kwargs or {}),
            guard=guard,
        )
        baseline_seconds = (time.perf_counter() - started) / 2.0
        modules = _modules_with_facet(clean_log, sites)
        if not modules:
            refuse(
                code="mi_payload_missing",
                message=f"No module exposes facet {sites!r} on the clean baseline.",
                remedy="check tl.facets.facet_coverage(trace) for the vocabulary",
                analysis="patching.grid",
                missing_sites=[],
            )
        first = _facet_tensor(clean_log.modules[modules[0]].facets[sites]).detach()
        corrupt_first = _facet_tensor(corrupted_log.modules[modules[0]].facets[sites]).detach()
        if tuple(first.shape) != tuple(corrupt_first.shape):
            refuse(
                code="mi_grid_length_mismatch",
                message=f"Clean and corrupted {sites!r} shapes differ "
                f"({tuple(first.shape)} vs {tuple(corrupt_first.shape)}); grids never "
                "broadcast across prompts.",
                remedy="pad or re-tokenize so both prompts have equal length",
            )

        position_axis = 2 if sites in ("z", "pattern", "scores") else 1
        position_list = (
            list(range(first.shape[position_axis]))
            if positions is None and "position" in axes
            else list(positions or ())
        )
        head_list = (
            list(range(first.shape[_HEAD_AXIS[sites]]))
            if heads is None and "head" in axes
            else list(heads or ())
        )
        shape = [len(modules)]
        coordinates: dict[str, tuple[Any, ...]] = {"layer": tuple(modules)}
        if "position" in axes:
            shape.append(len(position_list))
            coordinates["position"] = tuple(position_list)
        if "head" in axes:
            shape.append(len(head_list))
            coordinates["head"] = tuple(head_list)
        n_cells = _disclose_cost_and_enforce_budget(shape, baseline_seconds, budget)

        values = torch.empty(shape, dtype=torch.float32)
        campaign_ledger: dict[str, int] = {}
        cells = _cell_indices(axes, len(modules), position_list, head_list)
        for cell in cells:
            address = modules[cell["layer"]]
            clean_value = _facet_tensor(clean_log.modules[address].facets[sites]).detach().clone()
            hook = _cell_hook(
                clean_value,
                position=cell.get("position"),
                position_axis=position_axis,
                head=cell.get("head"),
                head_axis=_HEAD_AXIS.get(sites),
            )
            patched_log = _run_patch(
                model,
                corrupted_input,
                corrupted_log,
                facet_selector(sites).in_module(address),
                hook,
                name=f"grid_{sites}_{cell}",
                guard=guard,
                facet_name=sites,
                address=address,
                campaign_ledger=campaign_ledger,
            )
            try:
                index = tuple(
                    cell[axis] if axis == "layer" else list(coordinates[axis]).index(cell[axis])
                    for axis in axes
                )
                values[index] = _metric_scalar(metric(patched_log), like=values)
            finally:
                patched_log.cleanup()
        _warn_if_campaign_all_identical(campaign_ledger, f"grid over facet {sites!r}")
        if values.numel() > 1 and bool((values == values.reshape(-1)[0]).all()):
            warnings.warn(
                TorchLensWarning(
                    "Every grid cell produced an identical metric. A genuinely flat "
                    "effect is a finding, so this is a disclosure, not a refusal "
                    "(mikit D21). Remedy: check the fire receipts and the metric before "
                    "publishing a null result",
                    code="mi_grid_all_identical",
                ),
                stacklevel=2,
            )
        return PatchGrid(
            axes=tuple(axes),
            coordinates=coordinates,
            values=values,
            receipts={
                "engine": "rerun",
                "fires": campaign_ledger.get("fires", 0),
                "identical_fires": campaign_ledger.get("identical", 0),
                "n_cells": n_cells,
                "baseline_seconds_per_rerun": baseline_seconds,
                "metric_provenance": getattr(metric, "__name__", repr(metric)),
            },
        )
    finally:
        _teardown(guard, clean_log, corrupted_log)


def _disclose_cost_and_enforce_budget(
    shape: list[int], baseline_seconds: float, budget: int | None
) -> int:
    """Print the pre-run cost line; refuse over-budget grids BEFORE any rerun."""

    n_cells = 1
    for extent in shape:
        n_cells *= extent
    print(_estimate_cost(n_cells, baseline_seconds))
    if budget is not None and n_cells > budget:
        refuse(
            code="mi_grid_budget_exceeded",
            message=f"The grid has {n_cells} cells, over the budget of {budget}.",
            remedy="narrow layers/positions/heads, or raise budget= explicitly",
            n_cells=n_cells,
            budget=budget,
        )
    return n_cells


def _validate_grid_request(engine: str, axes: tuple[str, ...], sites: str) -> None:
    """Refuse invalid engines/axes BEFORE any model work (typed, teaching)."""

    if engine == "replay":
        refuse(
            code="mi_grid_engine_unsupported",
            message='engine="replay" is reserved: exact path patching waits on the known '
            "HF replay crash (FIX-A); the refusal is the whole placeholder.",
            remedy='use engine="rerun" (the live-hook route, receipted)',
        )
    if engine != "rerun":
        refuse(
            code="mi_grid_engine_unsupported",
            message=f"Unknown engine {engine!r}.",
            remedy='use engine="rerun"',
        )
    unknown_axes = [axis for axis in axes if axis not in _AXES]
    if unknown_axes or "layer" not in axes:
        refuse(
            code="mi_grid_axes_invalid",
            message=f"axes={axes!r} must be drawn from {_AXES} and include 'layer'.",
            remedy="pass axes such as ('layer',), ('layer', 'position'), ('layer', 'head')",
            axes=list(axes),
        )
    if "head" in axes and sites not in _HEAD_AXIS:
        refuse(
            code="mi_grid_axes_invalid",
            message=f"The head axis needs a head-carrying facet; {sites!r} has no known "
            "head layout.",
            remedy=f"pick sites among {sorted(_HEAD_AXIS)} for head grids",
            sites=sites,
        )


def _cell_indices(
    axes: tuple[str, ...], n_layers: int, positions: list[int], heads: list[int]
) -> list[dict[str, int]]:
    """Enumerate grid cells as {axis: index-or-value} dicts."""

    cells: list[dict[str, int]] = [{"layer": layer} for layer in range(n_layers)]
    if "position" in axes:
        cells = [{**cell, "position": position} for cell in cells for position in positions]
    if "head" in axes:
        cells = [{**cell, "head": head} for cell in cells for head in heads]
    return cells


def _cell_hook(
    clean_value: torch.Tensor,
    *,
    position: int | None,
    position_axis: int,
    head: int | None,
    head_axis: int | None,
) -> Any:
    """Build the per-cell replacement hook (clean into corrupted)."""

    def _hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
        """Replace the cell's slice of ``out`` with the clean activation."""

        del hook
        patched = out.clone(memory_format=torch.preserve_format)
        target = patched
        source = clean_value
        if head is not None and head_axis is not None:
            target = target.select(head_axis, head)
            source = source.select(head_axis, head)
            axis_after_head = position_axis - (1 if head_axis < position_axis else 0)
        else:
            axis_after_head = position_axis
        if position is not None:
            target = target.select(axis_after_head, position)
            source = source.select(axis_after_head, position)
        target.copy_(source)
        return patched

    return _hook


def patch_residual_grid(
    model: Any, clean_input: Any, corrupted_input: Any, metric: Any, **kwargs: Any
) -> PatchGrid:
    """Preset: the classic [layer, position] residual-stream grid."""

    return grid(
        model,
        clean_input,
        corrupted_input,
        metric,
        sites="resid_pre",
        axes=("layer", "position"),
        **kwargs,
    )


def patch_heads_grid(
    model: Any, clean_input: Any, corrupted_input: Any, metric: Any, **kwargs: Any
) -> PatchGrid:
    """Preset: the [layer, head] z-patching grid."""

    return grid(
        model,
        clean_input,
        corrupted_input,
        metric,
        sites="z",
        axes=("layer", "head"),
        **kwargs,
    )
