"""The hooks pass recorder leaves the capture and replay doors as it found them.

The recorder in ``tests/behaviour/_hook_passes.py`` wraps
``torchlens._capture.run_entry._run_resolved_request`` and
``torchlens.intervention._replay._run_replay``. A wrapper left installed after
its cell adds a test-helper frame to every later capture on the same xdist
worker, so a later ``tl.trace`` resolves its call site into the helper
(``tests/test_site_key.py::test_witness_on_real_capture_is_operation_frame``
went red that way when it ran after a hooks cell).
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import torch

import torchlens as tl
from tests.behaviour import _hook_passes
from tests.behaviour.cases import case_by_name
from tests.behaviour.features import feature
from tests.behaviour.harness import run_cell
from torchlens._capture import run_entry
from torchlens.intervention import _replay as replay


def _doors() -> tuple[object, object]:
    """Return the functions now bound at the capture door and the replay door."""

    return run_entry._run_resolved_request, replay._run_replay


def _first_operation_call_site(log: tl.Trace) -> str:
    """Return the call-site file of the first non-I/O op in ``log``."""

    for label in log.op_labels:
        op = log.ops[label]
        if op.type not in ("input", "output", "buffer"):
            return op.source.frames[0].file
    raise AssertionError("the trace recorded no operation")


def test_hooks_cell_leaves_the_capture_doors_unwrapped(tmp_path: Path) -> None:
    """After a hooks cell the doors are the originals and a later trace's call site is this file."""

    before = _doors()
    run_cell(case_by_name("toy:simple_ff"), (feature("hooks"),), tmp_path / "cell")
    assert _doors() == before
    log = tl.trace(torch.nn.Linear(4, 2), torch.randn(1, 4))
    assert _first_operation_call_site(log) == __file__


def test_recording_restores_the_doors_when_its_body_raises() -> None:
    """``recording()`` restores both doors when its body raises."""

    before = _doors()
    with pytest.raises(RuntimeError, match="cell body failed"):
        with _hook_passes.recording(lambda out, *, ctx: out, []):
            assert _doors() != before
            raise RuntimeError("cell body failed")
    assert _doors() == before


def test_stop_recording_restores_the_doors() -> None:
    """``stop_recording`` restores both doors."""

    before = _doors()
    recorder = _hook_passes.start_recording(lambda out, *, ctx: out, [])
    assert _doors() != before
    _hook_passes.stop_recording(recorder)
    assert _doors() == before


def test_patch_tripwire_names_and_unwraps_a_left_wrapper() -> None:
    """The tripwire names a wrapper left on a door and finds the original under it."""

    from _leak_guards import _unwrapped, leaked_internal_patches

    original = run_entry._run_resolved_request

    @functools.wraps(original)
    def left_behind(r: object) -> object:
        """Delegate unchanged: a stand-in for a wrapper a helper forgot to remove."""

        return original(r)

    run_entry._run_resolved_request = left_behind
    try:
        door = (("torchlens._capture.run_entry", "_run_resolved_request"),)
        assert leaked_internal_patches(door) == [(run_entry, "_run_resolved_request", left_behind)]
        assert (run_entry, "_run_resolved_request", left_behind) in leaked_internal_patches()
        assert _unwrapped(left_behind) is original
    finally:
        run_entry._run_resolved_request = original
    assert not leaked_internal_patches()
