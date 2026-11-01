"""Cell generator for the stress matrix.

A cell is one (model, input, dtype, device, options) combination. Axes are
plain lists of frozen row dataclasses defined under ``axes/``; this module
turns them into cells with stable ids, selects the pre-merge subset, and
reads ``subset.json`` back for the test module.

Run ``python -m tests.stress.generator --write-subset`` after adding a row so
``subset.json`` covers the new value; ``--list`` prints every generated cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

Inputs = tuple[tuple[Any, ...], dict[str, Any]]
"""Positional arguments and keyword arguments for one forward call."""

SUBSET_PATH = Path(__file__).with_name("subset.json")
SUBSET_SIZE = 200
FULL_MATRIX_ENV = "TORCHLENS_STRESS_FULL"
"""Set to ``1`` to run every generated cell instead of the pre-merge subset."""

_ROW_ID = re.compile(r"^[a-z0-9_]+$")
_SEPARATOR = "-"


def _check_row_id(row_id: str) -> str:
    """Return ``row_id`` when it is lowercase ``[a-z0-9_]`` (so cell ids split cleanly)."""

    if not _ROW_ID.match(row_id):
        raise ValueError(f"axis row id {row_id!r} must match {_ROW_ID.pattern}")
    return row_id


@dataclass(frozen=True)
class ModelSpec:
    """One model row.

    Attributes
    ----------
    id:
        Row id, lowercase ``[a-z0-9_]``.
    build:
        Zero-argument constructor; the generator seeds torch before calling it.
    example_input:
        ``(model) -> (args, kwargs)``: the default input row for the built model.
    reason:
        One line on why this row exists (what it stresses).
    dtypes:
        Dtype row ids this model may be cast to. ``"f32"`` is the identity row.
    cost:
        Relative cost rank, 1 (cheapest) to 5; drives subset selection.
    memory_budget_mib:
        Peak-memory budget for the whole cell (capture, validation, save, load).
    tags:
        Free-form coverage tags recorded with the row.
    """

    id: str
    build: Callable[[], nn.Module]
    example_input: Callable[[nn.Module], Inputs]
    reason: str
    dtypes: tuple[str, ...] = ("f32",)
    cost: int = 1
    memory_budget_mib: int = 512
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        _check_row_id(self.id)


@dataclass(frozen=True)
class InputSpec:
    """One input-shape row.

    Attributes
    ----------
    id:
        Row id.
    make:
        ``(model) -> (args, kwargs)``; ``None`` uses the model's own example input.
    reason:
        One line on what this input shape stresses.
    models:
        Model ids this row fits; ``None`` fits every model (the default row).
    """

    id: str
    make: Callable[[nn.Module], Inputs] | None
    reason: str
    models: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _check_row_id(self.id)

    def fits(self, model: ModelSpec) -> bool:
        """Return whether this input row applies to ``model``."""

        return self.models is None or model.id in self.models


@dataclass(frozen=True)
class DtypeSpec:
    """One dtype row: a cast applied to the built model and its inputs.

    Attributes
    ----------
    id:
        Row id (``f32`` is the identity row every model carries).
    cast:
        ``(model, inputs) -> (model, inputs)``, applied before the eager run.
    reason:
        One line on what the dtype stresses.
    """

    id: str
    cast: Callable[[nn.Module, Inputs], tuple[nn.Module, Inputs]]
    reason: str

    def __post_init__(self) -> None:
        _check_row_id(self.id)


@dataclass(frozen=True)
class DeviceSpec:
    """One device row."""

    id: str
    device: str
    reason: str

    def __post_init__(self) -> None:
        _check_row_id(self.id)

    def available(self) -> bool:
        """Return whether the device can run in this process."""

        if self.device == "cpu":
            return True
        if self.device.startswith("cuda"):
            return torch.cuda.is_available()
        if self.device == "mps":
            backend = getattr(torch.backends, "mps", None)
            return bool(backend is not None and backend.is_available())
        return False


@dataclass(frozen=True)
class OptionsSpec:
    """One options row: extra ``tl.trace`` keywords and ``CaptureOptions`` fields.

    Attributes
    ----------
    id:
        Row id.
    reason:
        One line on what the options stress.
    trace_kwargs:
        Extra keyword arguments passed to ``tl.trace`` (never ``capture=``).
    capture_kwargs:
        Extra ``CaptureOptions`` fields (``random_seed`` is set by the runner).
    """

    id: str
    reason: str
    trace_kwargs: Mapping[str, Any] = field(default_factory=dict)
    capture_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_row_id(self.id)


@dataclass(frozen=True)
class Axes:
    """The five axis lists the generator crosses."""

    models: tuple[ModelSpec, ...]
    inputs: tuple[InputSpec, ...]
    dtypes: tuple[DtypeSpec, ...]
    devices: tuple[DeviceSpec, ...]
    options: tuple[OptionsSpec, ...]

    def values(self) -> tuple[tuple[str, str], ...]:
        """Return every ``(axis, row id)`` pair, in axis order."""

        return (
            *(("model", row.id) for row in self.models),
            *(("input", row.id) for row in self.inputs),
            *(("dtype", row.id) for row in self.dtypes),
            *(("device", row.id) for row in self.devices),
            *(("options", row.id) for row in self.options),
        )


@dataclass(frozen=True)
class Cell:
    """One generated cell."""

    model: ModelSpec
    input: InputSpec
    dtype: DtypeSpec
    device: DeviceSpec
    options: OptionsSpec

    @property
    def id(self) -> str:
        """Stable id: the five row ids joined by ``-`` in axis order."""

        return _SEPARATOR.join(
            (self.model.id, self.input.id, self.dtype.id, self.device.id, self.options.id)
        )

    @property
    def cost(self) -> int:
        """Relative cost rank of the cell (the model's rank)."""

        return self.model.cost

    def axis_values(self) -> tuple[tuple[str, str], ...]:
        """Return the ``(axis, row id)`` pairs this cell covers."""

        return (
            ("model", self.model.id),
            ("input", self.input.id),
            ("dtype", self.dtype.id),
            ("device", self.device.id),
            ("options", self.options.id),
        )


def generate(axes: Axes) -> tuple[Cell, ...]:
    """Cross the axes into cells, keeping only compatible combinations.

    An input row joins a model only when it declares that model (or every
    model); a dtype row joins a model only when the model lists it. Device and
    options rows join every model.

    Parameters
    ----------
    axes:
        The axis lists.

    Returns
    -------
    tuple[Cell, ...]
        Cells in axis order, ids unique.
    """

    dtype_rows = {row.id: row for row in axes.dtypes}
    cells = tuple(_cross(axes, dtype_rows))
    ids = [cell.id for cell in cells]
    duplicates = sorted({cell_id for cell_id in ids if ids.count(cell_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate cell ids: {duplicates}")
    return cells


def _cross(axes: Axes, dtype_rows: Mapping[str, DtypeSpec]) -> Iterator[Cell]:
    """Yield every compatible cell in axis order."""

    for model in axes.models:
        for dtype_id in model.dtypes:
            if dtype_id not in dtype_rows:
                raise ValueError(f"model {model.id!r} names unknown dtype row {dtype_id!r}")
        for input_row in axes.inputs:
            if not input_row.fits(model):
                continue
            for dtype_id in model.dtypes:
                for device in axes.devices:
                    for options in axes.options:
                        yield Cell(model, input_row, dtype_rows[dtype_id], device, options)


def select_subset(cells: tuple[Cell, ...], size: int = SUBSET_SIZE) -> tuple[str, ...]:
    """Choose the pre-merge subset: cover every axis value once, then the cheapest.

    Parameters
    ----------
    cells:
        Every generated cell.
    size:
        Target subset size; the whole matrix when it is smaller.

    Returns
    -------
    tuple[str, ...]
        Selected cell ids in matrix order.
    """

    by_cost = sorted(cells, key=lambda cell: (cell.cost, cell.id))
    chosen: dict[str, Cell] = {}
    covered: set[tuple[str, str]] = set()
    for cell in by_cost:
        if len(chosen) >= size:
            break
        new_values = set(cell.axis_values()) - covered
        if new_values:
            chosen[cell.id] = cell
            covered |= new_values
    for cell in by_cost:
        if len(chosen) >= size:
            break
        chosen.setdefault(cell.id, cell)
    order = {cell.id: index for index, cell in enumerate(cells)}
    return tuple(sorted(chosen, key=order.__getitem__))


def matrix_fingerprint(cells: tuple[Cell, ...]) -> str:
    """Return a short digest of the generated cell ids (changes when a row changes).

    The ``sha256-`` prefix keeps the value from reading as a bare hex secret to
    the repository's secret scanner.
    """

    digest = hashlib.sha256("\n".join(cell.id for cell in cells).encode("utf-8"))
    return f"sha256-{digest.hexdigest()[:12]}"


def subset_document(axes: Axes, size: int = SUBSET_SIZE) -> dict[str, Any]:
    """Build the ``subset.json`` document for ``axes``."""

    cells = generate(axes)
    return {
        "schema": "torchlens-stress-subset-v1",
        "size": size,
        "matrix_cells": len(cells),
        "matrix_fingerprint": matrix_fingerprint(cells),
        "cells": list(select_subset(cells, size)),
    }


def write_subset(axes: Axes, path: Path = SUBSET_PATH, size: int = SUBSET_SIZE) -> dict[str, Any]:
    """Write ``subset.json`` for ``axes`` and return the document."""

    document = subset_document(axes, size)
    path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return document


def load_subset(path: Path = SUBSET_PATH) -> dict[str, Any]:
    """Read ``subset.json``."""

    return json.loads(path.read_text(encoding="utf-8"))


def full_matrix_requested() -> bool:
    """Return whether the environment asks for the full matrix."""

    return os.environ.get(FULL_MATRIX_ENV, "") == "1"


def selected_cells(axes: Axes, *, full: bool | None = None) -> tuple[Cell, ...]:
    """Return the cells the test module parametrizes over.

    Parameters
    ----------
    axes:
        The axis lists.
    full:
        ``True`` for every cell, ``False`` for the subset, ``None`` to read the
        ``TORCHLENS_STRESS_FULL`` environment switch.

    Returns
    -------
    tuple[Cell, ...]
        Cells in matrix order.

    Raises
    ------
    ValueError
        When ``subset.json`` names a cell the axes no longer generate, so a
        row rename cannot silently shrink the pre-merge run.
    """

    cells = generate(axes)
    if full is None:
        full = full_matrix_requested()
    if full:
        return cells
    wanted = set(load_subset()["cells"])
    known = {cell.id for cell in cells}
    stale = sorted(wanted - known)
    if stale:
        raise ValueError(
            f"subset.json names cells the axes do not generate: {stale[:5]}; "
            "run `python -m tests.stress.generator --write-subset`"
        )
    return tuple(cell for cell in cells if cell.id in wanted)


def _main(argv: list[str] | None = None) -> int:
    """Command-line entry: write or list the subset."""

    from .axes import default_axes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-subset", action="store_true", help="rewrite subset.json")
    parser.add_argument("--size", type=int, default=SUBSET_SIZE)
    parser.add_argument("--list", action="store_true", help="print every generated cell id")
    args = parser.parse_args(argv)
    axes = default_axes()
    cells = generate(axes)
    if args.list:
        for cell in cells:
            print(cell.id)
    if args.write_subset:
        document = write_subset(axes, size=args.size)
        print(f"wrote {len(document['cells'])} of {len(cells)} cells to {SUBSET_PATH}")
    if not (args.list or args.write_subset):
        print(f"{len(cells)} cells; fingerprint {matrix_fingerprint(cells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
