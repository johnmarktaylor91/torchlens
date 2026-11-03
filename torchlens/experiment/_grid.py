"""``sweep_grid``: named axes over the candidate engine, a labeled N-dimensional result.

The flat ``sweep_sites`` face takes a list of candidates; this face takes a
mapping of named axes (``axes={"layer": [...], "position": {...}}``), forms
one candidate per cell of their product, runs the SAME engine (preflight
once, serial execution, extract before cleanup, retention under policy) and
returns a :class:`SweepGrid`: a float64 tensor of metric values shaped by
the axes, NaN where a cell was refused or failed, beside the Bundle that is
the experiment's record. One axis may hold edits instead of sites (the
edit then varies per cell); a hole in the grid (a cell resolving to no site,
or to several without a ``multi_site`` declaration) is a refused cell and
every other cell runs, exactly as the flat sweep's refused rows (ruled
1 Nov 2026). Axis-shape mistakes refuse the whole grid before any forward.

Every spelling DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .._utils._display_bounded import short_dataclass_repr
from ..errors.episode import BundleExperimentError
from ._effects import EffectsView
from ._engine import (
    _Candidate,
    _declared_multi_ids,
    _multi_site_reason,
    _refuse_duplicates,
    _resolve_candidate,
    _run_plan,
    _SweepRun,
    _validate_sweep_entry,
)

if TYPE_CHECKING:
    from torch import nn

    from .._data_classes.trace import Trace

__all__ = ["GridCell", "SweepGrid", "sweep_grid"]

_UNRESOLVED = "grid_cell_unresolved"
_MULTI_SITE = "grid_cell_multi_site"
_SITE_KIND = "site"
_EDIT_KIND = "edit"


@dataclass(frozen=True)
class GridCell:
    """One cell of a :class:`SweepGrid`: its coordinates and effect-row facts.

    Attributes
    ----------
    candidate_id:
        The comma-joined value names (``"L3,t7"``), also the member name of
        a retained cell.
    coords:
        Axis name to value name.
    status:
        ``completed`` (retained), ``released``, ``refused`` or ``failed``.
    value:
        The metric value, None for a refused or failed cell.
    error:
        The refusal (``grid_cell_unresolved: ...``, ``grid_cell_multi_site:
        ...``) or the failure text; None otherwise.
    member_name:
        The retained member's name, None when no member is held.

    Examples
    --------
    >>> base = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    >>> grid = tl.experiment.sweep_grid(
    ...     base, axes={"site": {"a": "conv2d_1_1", "b": "relu_1_2"}}, edit=tl.zero_ablate(),
    ...     metric=lambda t: float(t[t.output_layers[0]].out.sum()), retain="none",
    ...     model=model, inputs=x)
    >>> cell = grid.cell(site="b")
    >>> isinstance(cell, GridCell), cell.candidate_id, cell.coords, cell.status
    (True, 'b', {'site': 'b'}, 'released')
    """

    candidate_id: str
    coords: Mapping[str, str]
    status: str
    value: float | None
    error: str | None
    member_name: str | None


@dataclass(frozen=True)
class SweepGrid:
    """The labeled result of :func:`sweep_grid`.

    ``values`` is plain data: no plotting helper, no threshold, no ranking
    beyond what ``effects`` already offers.

    Examples
    --------
    >>> base = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    >>> grid = tl.experiment.sweep_grid(
    ...     base, axes={"site": ["conv2d_1_1", "relu_1_2"]}, edit=tl.scale(0.5),
    ...     metric=lambda t: float(t[t.output_layers[0]].out.sum()), retain="none",
    ...     model=model, inputs=x)
    >>> isinstance(grid, SweepGrid), len(grid.cells), grid.refused
    (True, 2, ())
    """

    axes: dict[str, tuple[str, ...]]
    values: torch.Tensor
    baseline_value: float
    cells: tuple[GridCell, ...]
    edit_axis: str | None
    bundle: Any
    effects: EffectsView
    history_id: str

    @property
    def deltas(self) -> torch.Tensor:
        """``values - baseline_value`` (NaN stays NaN)."""

        return self.values - self.baseline_value

    @property
    def failed(self) -> tuple[str, ...]:
        """Candidate ids of the cells whose forward or metric raised."""

        return tuple(cell.candidate_id for cell in self.cells if cell.status == "failed")

    @property
    def refused(self) -> tuple[str, ...]:
        """Candidate ids of the cells refused at preflight (never run)."""

        return tuple(cell.candidate_id for cell in self.cells if cell.status == "refused")

    def index(self, **coords: str) -> tuple[int, ...]:
        """The integer index tuple into ``values`` at these coordinates."""

        _require_full_coords(self.axes, coords)
        return tuple(self.axes[axis].index(coords[axis]) for axis in self.axes)

    def candidate_id(self, **coords: str) -> str:
        """The cell id at these coordinates (the comma-joined value names)."""

        _require_full_coords(self.axes, coords)
        return ",".join(coords[axis] for axis in self.axes)

    def coords(self, candidate_id: str) -> dict[str, str]:
        """The coordinates of a cell id (the inverse of :meth:`candidate_id`)."""

        for cell in self.cells:
            if cell.candidate_id == candidate_id:
                return dict(cell.coords)
        raise BundleExperimentError(
            f"no grid cell is named {candidate_id!r}; cell ids are the comma-joined "
            f"value names over axes {list(self.axes)}.",
            code="grid_cell_unknown",
            candidate_id=candidate_id,
        )

    def cell(self, **coords: str) -> GridCell:
        """The :class:`GridCell` at these coordinates (every axis required)."""

        wanted = self.candidate_id(**coords)
        return next(cell for cell in self.cells if cell.candidate_id == wanted)

    def to_pandas(self) -> Any:
        """Long format, one row per cell (needs the ``tabular`` extra)."""

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "SweepGrid.to_pandas() requires pandas. Install with "
                "`pip install torchlens[tabular]`."
            ) from exc
        rows = []
        for cell in self.cells:
            row: dict[str, Any] = dict(cell.coords)
            row["candidate_id"] = cell.candidate_id
            row["status"] = cell.status
            row["value"] = cell.value
            row["delta"] = None if cell.value is None else cell.value - self.baseline_value
            row["member_name"] = cell.member_name
            rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        shape = ", ".join(f"{axis}={len(names)}" for axis, names in self.axes.items())
        measured = sum(1 for cell in self.cells if cell.value is not None)
        failed = len(self.failed)
        refused = len(self.refused)
        extra = f", refused={refused}" if refused else ""
        retain = str(self.effects.table.retain_policy or "").strip("'\"")
        return (
            f"SweepGrid({shape}, measured={measured}, failed={failed}{extra}, "
            f"retain={retain}, lane={self.effects.table.lane})"
        )


def _require_full_coords(axes: Mapping[str, Sequence[str]], coords: Mapping[str, str]) -> None:
    """Refuse coordinates that miss an axis, name an unknown one, or an unknown value."""

    if set(coords) != set(axes) or any(coords[axis] not in axes[axis] for axis in axes):
        raise BundleExperimentError(
            f"grid coordinates need every axis with a known value name; got {dict(coords)!r} "
            f"for axes { ({axis: list(names) for axis, names in axes.items()})!r}.",
            code="grid_coords_invalid",
        )


# ---- declaring axes ----------------------------------------------------------


def _axis_values(axes: Any) -> dict[str, list[tuple[str, Any]]]:
    """Normalize ``axes`` to axis name -> ordered (value name, value) pairs."""

    if not isinstance(axes, Mapping) or not axes:
        _refuse_axes(
            "sweep_grid axes= must be a non-empty mapping of axis name -> values "
            f"(a list, or a mapping of value name -> value); got {type(axes).__name__}."
        )
    out: dict[str, list[tuple[str, Any]]] = {}
    for axis, values in axes.items():
        if not isinstance(axis, str) or not axis:
            _refuse_axes(f"sweep_grid axis names must be non-empty strings, got {axis!r}.")
        if isinstance(values, Mapping):
            pairs = [(name, value) for name, value in values.items()]
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            pairs = [(str(index), value) for index, value in enumerate(values)]
        else:
            _refuse_axes(
                f"sweep_grid axis {axis!r} needs a list of values or a mapping of value "
                f"name -> value, got {type(values).__name__}.",
                axis=axis,
            )
        _validate_value_names(axis, [name for name, _ in pairs])
        out[axis] = pairs
    return out


def _validate_value_names(axis: str, names: list[Any]) -> None:
    """Value names: non-empty strings without commas, never ``baseline``, unique."""

    if not names:
        _refuse_axes(
            f"sweep_grid axis {axis!r} has no values; every axis needs at least one.", axis=axis
        )
    for name in names:
        if not isinstance(name, str) or not name or "," in name or name == "baseline":
            _refuse_axes(
                f"sweep_grid axis {axis!r} value name {name!r} is invalid: names are "
                "non-empty strings without commas, and 'baseline' is reserved.",
                axis=axis,
                value_name=repr(name),
            )
    if len(set(names)) != len(names):
        _refuse_axes(
            f"sweep_grid axis {axis!r} repeats a value name; names are unique within an axis.",
            axis=axis,
        )


def _refuse_axes(message: str, **fields: Any) -> None:
    """The one ``grid_axes_invalid`` raise: an axes= shape the grid cannot read."""

    raise BundleExperimentError(message, code="grid_axes_invalid", **fields)


def _value_kind(value: Any) -> str | None:
    """``site`` for a selector / Selection / label, ``edit`` for an Edit, else None."""

    from .._selection.selection import Selection, Selector
    from .._selection.selector_base import BaseSelector
    from ..intervention.types import Edit

    if isinstance(value, Edit):
        return _EDIT_KIND
    if isinstance(value, (str, BaseSelector, Selector, Selection)):
        return _SITE_KIND
    if callable(getattr(value, "__selection__", None)):
        return _SITE_KIND
    return None


def _axis_kinds(axes: dict[str, list[tuple[str, Any]]]) -> dict[str, str]:
    """One kind per axis (``site`` or ``edit``); a mixed or unknown axis refuses."""

    kinds: dict[str, str] = {}
    for axis, pairs in axes.items():
        seen = {_value_kind(value) for _, value in pairs}
        if None in seen or len(seen) != 1:
            raise BundleExperimentError(
                f"sweep_grid axis {axis!r} must hold sites only or edits only "
                f"(selectors, Selections and labels are sites; helper edits such as "
                "tl.zero_ablate() are edits; wrap a hook callable with the helper "
                f"constructors); found kinds {sorted(str(kind) for kind in seen)}.",
                code="grid_axis_mixed",
                axis=axis,
            )
        (kind,) = seen
        kinds[axis] = str(kind)
    return kinds


def _edit_axis(kinds: dict[str, str], edit: Any) -> str | None:
    """The single edit axis, checked against ``edit=`` (exactly one source of edits)."""

    site_axes = [axis for axis, kind in kinds.items() if kind == _SITE_KIND]
    edit_axes = [axis for axis, kind in kinds.items() if kind == _EDIT_KIND]
    if not site_axes:
        raise BundleExperimentError(
            "sweep_grid needs at least one site axis (an axis of selectors, Selections or "
            "labels), even one with a single value.",
            code="grid_site_axis_missing",
        )
    if len(edit_axes) > 1:
        raise BundleExperimentError(
            f"sweep_grid allows one edit axis; got {edit_axes}. Keep one axis of edits.",
            code="grid_edit_axis_multiple",
            edit_axes=tuple(edit_axes),
        )
    if edit_axes and edit is not None:
        raise BundleExperimentError(
            f"sweep_grid got an edit axis {edit_axes[0]!r} and edit=; drop one.",
            code="grid_edit_conflict",
            edit_axis=edit_axes[0],
        )
    if not edit_axes and edit is None:
        raise BundleExperimentError(
            "sweep_grid needs an edit: pass edit= or give one axis edit values.",
            code="grid_edit_missing",
        )
    return edit_axes[0] if edit_axes else None


# ---- forming cells -----------------------------------------------------------


def _lift_site(value: Any) -> Any:
    """A label string becomes ``tl.label``; every other site value passes through."""

    from .._selection.selectors import label

    return label(value) if isinstance(value, str) else value


def _combine_sites(cell_id: str, parts: list[tuple[str, Any]]) -> Any:
    """Form one cell's site from its site-axis values (one value passes through)."""

    from .._selection.selectors import FacetSelector, ModuleSelector

    if len(parts) == 1:
        return parts[0][1]
    facets = [(name, value) for name, value in parts if isinstance(value, FacetSelector)]
    if facets:
        (facet_name, facet), *more = facets
        others = [(name, value) for name, value in parts if not isinstance(value, FacetSelector)]
        scoped = facet.module_address is None
        modules = [value for _, value in others if isinstance(value, ModuleSelector)]
        if not more and scoped and len(others) == 1 and len(modules) == 1:
            return facet.in_module(modules[0].address)
        partner = (more or others)[0]
        raise BundleExperimentError(
            f"grid cell {cell_id!r} cannot combine facet value {facet_name!r} with "
            f"{partner[0]!r}: a facet selector combines only with exactly one module "
            "value (becoming facet.in_module(address)).",
            code="grid_cell_uncombinable",
            cell=cell_id,
            values=(facet_name, partner[0]),
        )
    combined = _lift_site(parts[0][1])
    for name, value in parts[1:]:
        try:
            combined = combined & _lift_site(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise BundleExperimentError(
                f"grid cell {cell_id!r} cannot combine values {parts[0][0]!r} and {name!r} "
                f"with &: {exc}",
                code="grid_cell_uncombinable",
                cell=cell_id,
                values=(parts[0][0], name),
            ) from exc
    return combined


@dataclass(frozen=True)
class _Plan:
    """The preflighted grid: resolved cells, refused cells, and the recorded shape."""

    __repr__ = short_dataclass_repr

    names: dict[str, tuple[str, ...]]
    edit_axis: str | None
    resolved: list[_Candidate]
    refused: list[tuple[_Candidate, str]]
    edit_repr: str

    @property
    def cell_count(self) -> int:
        """The number of cells in the full grid, refused cells included."""

        return math.prod(len(values) for values in self.names.values())


def _form_cells(
    baseline: Trace,
    axes: dict[str, list[tuple[str, Any]]],
    edit_axis: str | None,
    edit: Any,
    declared_multi: set[str],
) -> tuple[list[_Candidate], list[tuple[_Candidate, str]]]:
    """Form and resolve every cell; holes become refused cells (ruled F6).

    ``_edit_axis`` has already checked the axis kinds, so every axis other
    than ``edit_axis`` is a site axis.
    """

    from .._selection.errors import SiteResolutionError

    resolved: list[_Candidate] = []
    refused: list[tuple[_Candidate, str]] = []
    for combo in itertools.product(*[axes[axis] for axis in axes]):
        coords = dict(zip(axes, combo, strict=True))
        cell_id = ",".join(name for name, _ in combo)
        sites = [(name, value) for axis, (name, value) in coords.items() if axis != edit_axis]
        cell_edit = coords[edit_axis][1] if edit_axis is not None else edit
        where = _combine_sites(cell_id, sites)
        try:
            candidate = _resolve_candidate(baseline, cell_id, where)
        except SiteResolutionError as exc:
            refused.append(
                (_hole(cell_id, where, cell_edit), _unresolved_text(cell_id, sites, exc))
            )
            continue
        except BundleExperimentError as exc:
            if exc.fields.get("code") != "site_sweep_candidate_unresolved":
                raise
            refused.append(
                (_hole(cell_id, where, cell_edit), _unresolved_text(cell_id, sites, exc))
            )
            continue
        candidate = _with_edit(candidate, cell_edit)
        reason = _multi_site_reason(candidate, declared_multi)
        if reason is not None:
            count = candidate.resolved_site_count
            shown = "unprovable" if count is None else str(count)
            refused.append(
                (
                    candidate,
                    f"{_MULTI_SITE}: cell {cell_id!r} resolved_site_count={shown} without a "
                    "multi_site declaration; pass multi_site=True or add the cell id to it.",
                )
            )
            continue
        resolved.append(candidate)
    return resolved, refused


def _hole(cell_id: str, where: Any, edit: Any) -> _Candidate:
    """A refused cell's placeholder candidate (never executed)."""

    return _Candidate(
        candidate_id=cell_id,
        where=where,
        execution_where=where,
        resolved_site_count=0,
        resolution_digest="",
        edit=edit,
    )


def _unresolved_text(cell_id: str, sites: list[tuple[str, Any]], exc: BaseException) -> str:
    """The refusal text for a cell whose sites resolve to nothing on the baseline."""

    values = ", ".join(f"{name}={value!r}" for name, value in sites)
    return (
        f"{_UNRESOLVED}: cell {cell_id!r} ({values}) resolves to no site on the baseline; "
        f"check the labels and selectors against baseline.ops. ({exc})"
    )


def _with_edit(candidate: _Candidate, edit: Any) -> _Candidate:
    """A copy of ``candidate`` carrying this cell's edit."""

    from dataclasses import replace

    return replace(candidate, edit=edit)


def _check_positions(baseline: Trace, resolved: list[_Candidate]) -> None:
    """Refuse integer token positions a cell's site cannot hold (recorded shapes)."""

    from ..intervention.types import Edit

    for candidate in resolved:
        edit = candidate.edit
        if not isinstance(edit, Edit) or edit.name != "at_positions":
            continue
        kwargs = dict(edit.kwargs)
        positions = kwargs.get("positions")
        axis = int(kwargs.get("axis", 1))
        # ``tl.positions`` canonicalizes ints and int lists to a tuple of ints;
        # slices, named positions and per-row tensors are checked at run time.
        if not isinstance(positions, tuple):
            continue
        ints = [int(position) for position in positions]
        for label in candidate.labels:
            shape = getattr(baseline.ops[label], "shape", None)
            if not isinstance(shape, (tuple, list)) or len(shape) <= axis:
                continue
            extent = int(shape[axis])
            bad = [p for p in ints if p >= extent or p < -extent]
            if bad:
                raise BundleExperimentError(
                    f"grid cell {candidate.candidate_id!r} edits position(s) {bad} on axis "
                    f"{axis}, but site {label!r} has extent {extent} there (recorded shape "
                    f"{tuple(shape)}). Shrink the position axis or pick a wider site.",
                    code="grid_position_out_of_range",
                    cell=candidate.candidate_id,
                    positions=tuple(bad),
                    extent=extent,
                )


def _preflight_grid(
    baseline: Trace,
    axes: Any,
    *,
    edit: Any,
    multi_site: Any,
    replicates: bool,
    max_cells: int | None,
) -> _Plan:
    """Every grid refusal that happens before a forward, in order."""

    values = _axis_values(axes)
    kinds = _axis_kinds(values)
    edit_axis = _edit_axis(kinds, edit)
    names = {axis: tuple(name for name, _ in pairs) for axis, pairs in values.items()}
    cell_count = math.prod(len(pairs) for pairs in values.values())
    if max_cells is not None and cell_count > int(max_cells):
        raise BundleExperimentError(
            f"sweep_grid would run {cell_count} cells, above max_cells={max_cells}; "
            "lower an axis or raise max_cells.",
            code="grid_too_large",
            cells=cell_count,
            max_cells=int(max_cells),
        )
    cell_ids = [",".join(combo) for combo in itertools.product(*names.values())]
    declared_multi = _declared_multi_ids([(cell_id, None) for cell_id in cell_ids], multi_site)
    resolved, refused = _form_cells(baseline, values, edit_axis, edit, declared_multi)
    _check_positions(baseline, resolved)
    _refuse_duplicates(resolved, replicates)
    edit_repr = f"axis:{edit_axis}" if edit_axis is not None else repr(edit)
    return _Plan(
        names=names, edit_axis=edit_axis, resolved=resolved, refused=refused, edit_repr=edit_repr
    )


# ---- the face ---------------------------------------------------------------


def sweep_grid(
    baseline: Trace,
    *,
    axes: Mapping[str, Any],
    metric: Callable[[Any], Any],
    retain: Any,
    edit: Any = None,
    engine: str = "live_hook",
    model: nn.Module | None = None,
    inputs: Any = None,
    multi_site: bool | Collection[str] = False,
    replicates: bool = False,
    retain_ceiling_bytes: int | None = None,
    max_cells: int | None = None,
) -> SweepGrid:
    """Sweep the product of named axes; one member per cell, baseline first.

    ``axes`` maps axis name to values (a list named ``"0"``, ``"1"``, ... or
    a mapping of value name to value), in declaration order; each axis holds
    sites (selectors, Selections, labels) or edits, with at most one edit
    axis, which replaces ``edit=``. A cell's site is its site values
    combined: one value as is, a module-free facet plus exactly one module
    value as ``facet.in_module(address)``, otherwise the ``&`` intersection.
    The other keywords mean what they mean for ``sweep_sites`` (``multi_site``
    takes cell ids); ``max_cells`` guards the product of the axis lengths. A
    cell resolving to no site, or to several undeclared, is a refused cell
    (NaN) and every other cell runs; an axis-shape mistake refuses the whole
    grid before any forward. Returns a :class:`SweepGrid`.

    Examples
    --------
    >>> base = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    >>> metric = lambda t: float(t[t.output_layers[0]].out.sum())
    >>> grid = tl.experiment.sweep_grid(
    ...     base, axes={"site": {"a": "conv2d_1_1", "b": "relu_1_2"},
    ...                 "scale": [tl.scale(0.0), tl.scale(2.0)]},
    ...     metric=metric, retain="none", model=model, inputs=x)
    >>> grid.values.shape, grid.axes
    (torch.Size([2, 2]), {'site': ('a', 'b'), 'scale': ('0', '1')})
    """

    _validate_sweep_entry(engine=engine, metric=metric, retain=retain, model=model, x=inputs)
    plan = _preflight_grid(
        baseline,
        axes,
        edit=edit,
        multi_site=multi_site,
        replicates=replicates,
        max_cells=max_cells,
    )
    run = _SweepRun(
        attempted=plan.cell_count,
        metric=metric,
        retain=retain,
        engine=engine,
        model=model,
        x=inputs,
        retain_ceiling_bytes=retain_ceiling_bytes,
        edit_repr=plan.edit_repr,
        extra_params={
            "axes": {axis: list(names) for axis, names in plan.names.items()},
            "edit_axis": plan.edit_axis,
            "cells": plan.cell_count,
        },
    )
    bundle = _run_plan(baseline, plan.resolved, plan.refused, run)
    return grid_from_effects(bundle.effects(bundle.history[-1].operation_id))


def grid_from_effects(view: EffectsView) -> SweepGrid:
    """Rebuild a :class:`SweepGrid` from an effect table and its recorded axes."""

    params = view._operation_params or {}
    axes_param = params.get("axes")
    if not isinstance(axes_param, Mapping) or not axes_param:
        raise BundleExperimentError(
            f"operation {view.table.operation_id!r} recorded no grid axes; run sweep_grid, "
            "or read the effect table directly through bundle.effects().",
            code="grid_axes_unavailable",
            operation_id=view.table.operation_id,
        )
    axes = {str(axis): tuple(str(name) for name in names) for axis, names in axes_param.items()}
    rows = {row.candidate_id: row for row in view.rows if row.candidate_id != "__baseline__"}
    expected = [",".join(combo) for combo in itertools.product(*axes.values())]
    baseline_value = view.baseline_value
    if set(rows) != set(expected) or baseline_value is None:
        raise BundleExperimentError(
            f"operation {view.table.operation_id!r}: the effect table's rows and the recorded "
            f"axes do not match one-to-one ({len(rows)} rows, {len(expected)} cells); the "
            "artifact was not written by sweep_grid for these axes.",
            code="grid_result_shape_mismatch",
            operation_id=view.table.operation_id,
            rows=len(rows),
            cells=len(expected),
        )
    values = torch.full([len(names) for names in axes.values()], math.nan, dtype=torch.float64)
    cells: list[GridCell] = []
    for index, combo in enumerate(itertools.product(*axes.values())):
        row = rows[expected[index]]
        if row.value is not None:
            values.view(-1)[index] = float(row.value)
        cells.append(
            GridCell(
                candidate_id=row.candidate_id,
                coords=dict(zip(axes, combo, strict=True)),
                status=row.status,
                value=row.value,
                error=row.error,
                member_name=row.member_name,
            )
        )
    edit_axis = params.get("edit_axis")
    return SweepGrid(
        axes=axes,
        values=values,
        baseline_value=float(baseline_value),
        cells=tuple(cells),
        edit_axis=None if edit_axis is None else str(edit_axis),
        bundle=view._bundle,
        effects=view,
        history_id=view.table.operation_id,
    )
