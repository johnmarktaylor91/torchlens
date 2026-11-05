"""One test per stress matrix cell, judging the cell with all six oracles.

Every cell runs once, through the ``cell_result`` fixture, and its one test,
``test_cell[<cell id>]``, asks every oracle in ``ORACLES`` for a verdict and
never stops at the first failing one. Cells carry an ``xdist_group`` mark so
``-n N --dist loadgroup`` schedules them as whole units.

The scheduled run takes the cells in ``subset.json``; ``TORCHLENS_STRESS_FULL=1``
takes every generated cell. A ``fail`` from any oracle is a finding against the
cell id: the test fails with one message naming each failing oracle and its
reason. An oracle that raises counts as failing, with the exception named. A
cell whose every oracle is ``not_applicable`` is a skip with the reasons; a
``not_applicable`` beside other passing verdicts is recorded as a junit
property (``not_applicable:<oracle>``) and the cell passes.
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
from .oracles import ORACLES, CellResult, Verdict, run_cell

AXES = default_axes()
CELLS = selected_cells(AXES)


def _cell_param(cell: Cell) -> object:
    return pytest.param(cell, id=cell.id, marks=pytest.mark.xdist_group(cell.id))


@pytest.fixture(scope="module", params=[_cell_param(cell) for cell in CELLS])
def cell_result(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[CellResult]:
    """Run one cell once; its test judges the result with every oracle."""

    cell: Cell = request.param
    if not cell.device.available():
        pytest.skip(f"device {cell.device.device!r} is not available in this process")
    result = run_cell(cell, tmp_path_factory.mktemp("stress"))
    try:
        yield result
    finally:
        del result
        gc.collect()


def _verdicts(result: CellResult) -> list[tuple[str, Verdict]]:
    """Ask every oracle for its verdict; an oracle that raises is a failing verdict."""

    verdicts = []
    for name, oracle in ORACLES:
        try:
            verdict = oracle(result)
        except Exception as exc:  # judged as a failure below, never swallowed
            verdict = Verdict.failed(f"the oracle raised {type(exc).__name__}: {exc}")
        verdicts.append((name, verdict))
    return verdicts


@pytest.mark.slow
def test_cell(cell_result: CellResult, record_property: Callable[[str, object], None]) -> None:
    """Every oracle holds on the cell; a failure names each failing oracle.

    The six oracles: forward replay of every captured op reproduces eager
    (``replay_equals_eager``); default validation's metadata invariants hold
    (``invariants_pass``); a portable save and load returns the live ops and
    outputs (``save_load_equals_live``); anything TorchLens refused is a typed
    error with a stable code (``typed_refusal_names_condition``); torch and
    TorchLens process state are clean after the cell
    (``torch_clean_after_cell``); the cell's peak memory stays under its row's
    budget (``peak_memory_within_budget``).
    """

    verdicts = _verdicts(cell_result)
    failures = [f"{name}: {v.detail}" for name, v in verdicts if v.status == "fail"]
    if failures:
        pytest.fail(
            f"{cell_result.cell_id}: {len(failures)} oracle(s) failed\n" + "\n".join(failures),
            pytrace=False,
        )
    skipped = [(name, v.detail) for name, v in verdicts if v.status == "not_applicable"]
    if len(skipped) == len(verdicts):
        pytest.skip(f"{cell_result.cell_id}: " + "; ".join(f"{n}: {d}" for n, d in skipped))
    for name, detail in skipped:
        record_property(f"not_applicable:{name}", detail)


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
