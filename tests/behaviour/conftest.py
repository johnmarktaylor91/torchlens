"""Fixtures local to the behaviour net.

One cell's work (trace, validate, save, load, compare, feature checks) runs
once and its outcomes serve every oracle test of that cell. Each cell's tests
carry an ``xdist_group`` mark so ``--dist loadgroup`` keeps them on one
worker; without it each worker recomputes the cells it draws.

Each cell's temp directory is removed as soon as the cell has run, pass or
fail (:func:`tests.behaviour.harness.run_cell_and_clean`): pytest would
otherwise keep every cell's saved bundle, streamed payloads and rendered
graphs until the session ends, and a slow-universe run fills a worker's disk.
Set ``TORCHLENS_BEHAVIOUR_KEEP_TMP=1`` to keep the directory of a cell with a
failing oracle (or one that crashed) for inspection; pytest prints the
basetemp path, and passing cells are removed either way.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import pytest

from tests.behaviour.harness import CellResult, cell_id, run_cell_and_clean

_RESULTS: OrderedDict[str, CellResult] = OrderedDict()
_KEEP = 64
KEEP_TMP_ENV = "TORCHLENS_BEHAVIOUR_KEEP_TMP"


def _keep_failed_tmp() -> bool:
    return os.environ.get(KEEP_TMP_ENV, "").strip() not in {"", "0"}


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
        cached = run_cell_and_clean(case, features, tmp, keep_failed=_keep_failed_tmp())
        _RESULTS[key] = cached
        while len(_RESULTS) > _KEEP:
            _RESULTS.popitem(last=False)
    return cached
