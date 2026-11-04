"""Pin the save-time counters of one ``trace.save`` (the baseline the next format writer is measured against).

``benchmarks/save_path_counters.py`` counts the work one save of the ``tinynet`` capture does:
canonical-pickler passes and ``save`` calls, verifying re-reads, ``fsync`` calls, Python-level
opens, files and bytes. The pinned values live beside the ``aux_save`` row of
``benchmarks/perf_baselines/linux-cpu-provisional.json`` under
``metadata.save_path_counters``. Counters gate by exact equality; the recorded wall clock is
information only and is never read here.

Counters that depend only on the save path's structure gate everywhere. The pickler call
count and the byte counts also depend on the torch build (module docstrings and attributes
are part of the metadata), so they gate only when torch and the Python minor version match
the pinned environment.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

from benchmarks import save_path_counters as spc

_BASELINE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "perf_baselines"
    / "linux-cpu-provisional.json"
)
_ROW_LABEL = "tl.save fresh tlspec path"

# Structure of the save path: the same on every torch build.
STRUCTURAL_COUNTERS = (
    "metadata_pickle_dumps",
    "verify_reloads",
    "fsync_calls",
    "read_opens",
    "write_opens",
    "files_written",
)
# Depend on the torch build as well as the save path.
BUILD_KEYED_COUNTERS = ("pickler_save_calls", "bytes_written")


def _pinned() -> dict[str, Any]:
    """Return the pinned ``save_path_counters`` record beside the ``aux_save`` row."""

    document = json.loads(_BASELINE.read_text(encoding="utf-8"))
    rows = [
        row
        for row in document["rows"]
        if row.get("operation") == "aux_save" and row.get("label") == _ROW_LABEL
    ]
    assert len(rows) == 1, f"expected one {_ROW_LABEL!r} aux_save row, found {len(rows)}"
    return rows[0]["metadata"]["save_path_counters"]


@pytest.fixture(scope="module")
def measured(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two counted saves of the same capture (they must agree exactly)."""

    workdir = tmp_path_factory.mktemp("save_counters")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = spc.build_capture()
        return spc.measure(trace, workdir / "a"), spc.measure(trace, workdir / "b")


def test_pinned_record_names_every_gated_counter() -> None:
    """The pinned record covers exactly the counters the script gates."""

    pinned = _pinned()
    assert set(pinned["counters"]) == set(spc.GATED_COUNTERS) | {"bytes_by_part"}
    assert set(STRUCTURAL_COUNTERS) | set(BUILD_KEYED_COUNTERS) == set(spc.GATED_COUNTERS)
    assert pinned["schema"] == "torchlens.save_path_counters.v1"


def test_save_counters_repeat_exactly(measured: tuple[dict[str, Any], dict[str, Any]]) -> None:
    """Two saves of one capture do identical work (the counters are deterministic)."""

    first, second = measured
    assert first == second


def test_structural_save_counters_hold_exactly(
    measured: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """One pickler pass, one verifying re-read, and the pinned fsync, open and file counts."""

    pinned = _pinned()["counters"]
    first, _ = measured
    assert {key: first[key] for key in STRUCTURAL_COUNTERS} == {
        key: pinned[key] for key in STRUCTURAL_COUNTERS
    }
    assert first["metadata_pickle_dumps"] == 1
    assert first["verify_reloads"] == 1


def test_build_keyed_save_counters_hold_exactly(
    measured: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """Pickler calls and bytes per part hold exactly on the pinned torch and Python."""

    pinned = _pinned()
    python_minor = ".".join(sys.version.split()[0].split(".")[:2])
    pinned_minor = ".".join(pinned["python"].split(".")[:2])
    if (torch.__version__, python_minor) != (pinned["torch"], pinned_minor):
        pytest.skip(
            f"pinned on torch {pinned['torch']} / Python {pinned_minor}; "
            f"running torch {torch.__version__} / Python {python_minor}"
        )
    first, _ = measured
    expected = pinned["counters"]
    assert {key: first[key] for key in BUILD_KEYED_COUNTERS} == {
        key: expected[key] for key in BUILD_KEYED_COUNTERS
    }
    assert first["bytes_by_part"] == expected["bytes_by_part"]


def test_save_counter_census_is_red_capable(tmp_path: Path) -> None:
    """The counters see a write: one extra fsync and one extra open are counted."""

    import os

    with spc.counting(tmp_path) as counts:
        target = tmp_path / "probe.bin"
        with open(target, "wb") as handle:
            handle.write(b"x")
            handle.flush()
            os.fsync(handle.fileno())
    assert counts["fsync_calls"] == 1
    assert counts["write_opens"] == 1
