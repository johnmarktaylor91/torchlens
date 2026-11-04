"""The hooks pass recorder leaves the capture and replay doors as it found them.

The recorder in ``tests/behaviour/_hook_passes.py`` wraps
``torchlens._capture.run_entry._run_resolved_request`` and
``torchlens.intervention.replay._run_replay``. A wrapper left installed after
its cell adds a test-helper frame to every later capture on the same xdist
worker, so a later ``tl.trace`` resolves its call site into the helper
(``tests/test_site_key.py::test_witness_on_real_capture_is_operation_frame``
went red that way when it ran after a hooks cell).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import torchlens as tl
from tests.behaviour import _hook_passes
from tests.behaviour.cases import case_by_name
from tests.behaviour.features import feature
from tests.behaviour.harness import run_cell
from torchlens._capture import run_entry
from torchlens.intervention import replay


def _doors() -> tuple[object, object]:
    return run_entry._run_resolved_request, replay._run_replay


def _first_operation_call_site(log: tl.Trace) -> str:
    for label in log.op_labels:
        op = log.ops[label]
        if op.type not in ("input", "output", "buffer"):
            return op.code_context[0].file
    raise AssertionError("the trace recorded no operation")


def test_hooks_cell_leaves_the_capture_doors_unwrapped(tmp_path: Path) -> None:
    before = _doors()
    run_cell(case_by_name("toy:simple_ff"), (feature("hooks"),), tmp_path / "cell")
    assert _doors() == before
    log = tl.trace(torch.nn.Linear(4, 2), torch.randn(1, 4))
    assert _first_operation_call_site(log) == __file__


def test_recording_restores_the_doors_when_its_body_raises() -> None:
    before = _doors()
    with pytest.raises(RuntimeError, match="cell body failed"):
        with _hook_passes.recording(lambda out, *, ctx: out, []):
            assert _doors() != before
            raise RuntimeError("cell body failed")
    assert _doors() == before


def test_stop_recording_restores_the_doors() -> None:
    before = _doors()
    recorder = _hook_passes.start_recording(lambda out, *, ctx: out, [])
    assert _doors() != before
    _hook_passes.stop_recording(recorder)
    assert _doors() == before
