"""Fixtures local to the behaviour net.

One cell's work (trace, validate, save, load, compare, feature checks) runs
once and its outcomes serve every oracle test of that cell. Under xdist the
cell's tests are grouped (``xdist_group``) so ``--dist loadgroup`` keeps them
on one worker; without it each worker recomputes the cells it draws.
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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Group one cell's oracle tests under one xdist group."""

    if not config.pluginmanager.hasplugin("xdist"):
        return
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None or "cell_result" not in callspec.params:
            continue
        case, features = callspec.params["cell_result"]
        item.add_marker(pytest.mark.xdist_group(name=cell_id(case, features)))
