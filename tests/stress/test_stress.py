"""One parametrized test per oracle over the stress matrix cells.

Every cell runs once, through the module-scoped ``cell_result`` fixture, and
each oracle test reads that result. Cells carry an ``xdist_group`` mark so
``-n N --dist loadgroup`` keeps a cell's six oracle tests on one worker.

The pre-merge run takes the cells in ``subset.json``; ``TORCHLENS_STRESS_FULL=1``
takes every generated cell. A ``fail`` is a finding against the cell id; a
``not_applicable`` verdict is reported as a skip with its reason.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterator

import pytest

from .axes import default_axes
from .generator import (
    SUBSET_SIZE,
    Cell,
    generate,
    load_subset,
    matrix_fingerprint,
    select_subset,
    selected_cells,
)
from .oracles import (
    CellResult,
    Verdict,
    invariants_pass,
    peak_memory_within_budget,
    replay_equals_eager,
    run_cell,
    save_load_equals_live,
    torch_clean_after_cell,
    typed_refusal_names_condition,
)

AXES = default_axes()
CELLS = selected_cells(AXES)


def _cell_param(cell: Cell) -> object:
    return pytest.param(cell, id=cell.id, marks=pytest.mark.xdist_group(cell.id))


@pytest.fixture(scope="module", params=[_cell_param(cell) for cell in CELLS])
def cell_result(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[CellResult]:
    """Run one cell once; the six oracle tests for that cell share the result."""

    cell: Cell = request.param
    if not cell.device.available():
        pytest.skip(f"device {cell.device.device!r} is not available in this process")
    result = run_cell(cell, tmp_path_factory.mktemp("stress"))
    try:
        yield result
    finally:
        del result
        gc.collect()


def _judge(oracle: Callable[[CellResult], Verdict], result: CellResult) -> None:
    """Turn an oracle verdict into a pass, a failure naming the cell, or a skip."""

    verdict = oracle(result)
    if verdict.status == "fail":
        pytest.fail(f"{result.cell_id}: {verdict.detail}", pytrace=False)
    if verdict.status == "not_applicable":
        pytest.skip(f"{result.cell_id}: {verdict.detail}")


@pytest.mark.slow
def test_replay_equals_eager(cell_result: CellResult) -> None:
    """Forward replay of every captured op reproduces eager."""

    _judge(replay_equals_eager, cell_result)


@pytest.mark.slow
def test_invariants_pass(cell_result: CellResult) -> None:
    """Default validation's metadata invariants hold."""

    _judge(invariants_pass, cell_result)


@pytest.mark.slow
def test_save_load_equals_live(cell_result: CellResult) -> None:
    """A portable save and load returns the live ops and outputs."""

    _judge(save_load_equals_live, cell_result)


@pytest.mark.slow
def test_typed_refusal_names_condition(cell_result: CellResult) -> None:
    """Anything TorchLens refused is a typed error with a stable code."""

    _judge(typed_refusal_names_condition, cell_result)


@pytest.mark.slow
def test_torch_clean_after_cell(cell_result: CellResult) -> None:
    """Torch and TorchLens process state are clean after the cell."""

    _judge(torch_clean_after_cell, cell_result)


@pytest.mark.slow
def test_peak_memory_within_budget(cell_result: CellResult) -> None:
    """The cell's peak memory stays under its row's budget."""

    _judge(peak_memory_within_budget, cell_result)


def test_matrix_rows_are_well_formed() -> None:
    """Every row carries a reason and every generated cell id is unique."""

    for axis, rows in (
        ("model", AXES.models),
        ("input", AXES.inputs),
        ("dtype", AXES.dtypes),
        ("device", AXES.devices),
        ("options", AXES.options),
    ):
        for row in rows:
            assert row.reason.strip(), f"{axis} row {row.id!r} has no reason"
    cells = generate(AXES)
    assert cells, "the matrix generated no cells"
    assert len({cell.id for cell in cells}) == len(cells)


def test_subset_matches_the_axes() -> None:
    """``subset.json`` was selected from the current axes and covers every axis value."""

    cells = generate(AXES)
    document = load_subset()
    assert document["matrix_fingerprint"] == matrix_fingerprint(cells), (
        "the axes changed since subset.json was written; "
        "run `python -m tests.stress.generator --write-subset`"
    )
    assert document["cells"] == list(select_subset(cells, document["size"]))
    assert len(document["cells"]) <= SUBSET_SIZE
    reachable = {value for cell in cells for value in cell.axis_values()}
    by_id = {cell.id: cell for cell in cells}
    covered = {value for cell_id in document["cells"] for value in by_id[cell_id].axis_values()}
    assert covered == reachable, f"subset misses axis values: {sorted(reachable - covered)}"
