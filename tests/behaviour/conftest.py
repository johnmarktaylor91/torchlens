"""Fixtures local to the behaviour net.

One cell's work (trace, validate, save, load, compare, feature checks) runs
once and its outcomes serve every oracle test of that cell. Each cell's tests
carry an ``xdist_group`` mark so ``--dist loadgroup`` keeps them on one
worker; without it each worker recomputes the cells it draws.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest

from tests.behaviour.harness import CellResult, cell_id, run_cell

_RESULTS: OrderedDict[str, CellResult] = OrderedDict()
_KEEP = 64


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:80]


@pytest.fixture
def cell_result(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> CellResult:
    """Return the cached outcomes of the cell the test is parametrized with."""

    case, features = request.param
    key = cell_id(case, features)
    cached = _RESULTS.get(key)
    if cached is None:
        tmp: Path = tmp_path_factory.mktemp(_safe(key))
        cached = run_cell(case, features, tmp)
        _RESULTS[key] = cached
        while len(_RESULTS) > _KEEP:
            _RESULTS.popitem(last=False)
    return cached
