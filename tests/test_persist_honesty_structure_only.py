"""Persistence honesty: structure-only saves never write a poison artifact.

WT1 A-IV item 22 + M(weightsfree) W3 (lane A08):

* Item 22 -- ``level="executable_with_callables"`` on a structure-only trace
  used to WRITE an artifact whose load ALWAYS refused (forced saved-args
  payloads under the marker fail the M-C2 coherence gate). The level now
  refuses typed at save entry: an executable artifact is unconstructible from
  a capture that retains no values.
* W3 -- buffer registration wrote buffer ``out`` payloads under the
  structure-only marker (BatchNorm running stats: training-derived state a
  weights-free artifact must not carry), so EVERY structure-only save of a
  buffer-holding model was save-then-cannot-load. Buffer payloads now strip
  to declared geometry at save; a save-side coherence belt refuses (never
  writes) anything that would still fail M-C2; the load-side M-C2/M-C3 gates
  stay strict (pinned by tests/test_structure_only_capabilities.py and the
  forgery-validation suite, unchanged by this lane).
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.options import CaptureOptions


@pytest.fixture(scope="module")
def buffer_structure_trace():
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU()).eval()
    x = torch.randn(2, 4)
    log = tl.trace(model, x, capture=CaptureOptions(structure_only=True))
    yield log
    log.cleanup()


@pytest.fixture(scope="module")
def plain_structure_trace():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    x = torch.randn(2, 4)
    log = tl.trace(model, x, capture=CaptureOptions(structure_only=True))
    yield log
    log.cleanup()


def test_structure_only_executable_level_refuses_at_save(plain_structure_trace, tmp_path):
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.save(plain_structure_trace, tmp_path / "bundle", level="executable_with_callables")
    assert excinfo.value.fields["code"] == "save_payload_level_conflict"
    assert not (tmp_path / "bundle").exists()


def test_structure_only_buffer_model_round_trips(buffer_structure_trace, tmp_path):
    """The W3 headline: BN-holding structure-only saves load again (were poison)."""

    tl.save(buffer_structure_trace, tmp_path / "bundle")
    loaded = tl.load(tmp_path / "bundle")
    assert loaded.structure_only is True
    # Declared geometry survives the payload strip.
    buffer_ops = [op for layer in loaded.layer_list for op in layer.ops if op.is_buffer]
    assert buffer_ops, "expected buffer graph nodes on the loaded trace"
    for op in buffer_ops:
        assert op.out is None
        assert op.shape is not None


def test_structure_only_buffer_model_audit_round_trips(buffer_structure_trace, tmp_path):
    tl.save(buffer_structure_trace, tmp_path / "bundle", level="audit")
    loaded = tl.load(tmp_path / "bundle")
    assert loaded.structure_only is True


def test_structure_only_save_ships_no_buffer_initial_values(buffer_structure_trace, tmp_path):
    """Pre-forward buffer VALUES are training-derived state; weights-free saves drop
    the whole channel and the manifest disclosure says so."""

    tl.save(buffer_structure_trace, tmp_path / "bundle")
    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
    disclosure = manifest.get("buffer_values_disclosure", {})
    assert disclosure.get("included") is False


def test_structure_only_save_writes_no_value_blobs(buffer_structure_trace, tmp_path):
    tl.save(buffer_structure_trace, tmp_path / "bundle")
    assert list((tmp_path / "bundle" / "blobs").iterdir()) == []


def test_live_structure_trace_untouched_by_save(buffer_structure_trace, tmp_path):
    """The strip is save-side only: the LIVE trace keeps its session state."""

    before = {
        op.label: op.out is not None
        for layer in buffer_structure_trace.layer_list
        for op in layer.ops
        if op.is_buffer
    }
    tl.save(buffer_structure_trace, tmp_path / "bundle")
    after = {
        op.label: op.out is not None
        for layer in buffer_structure_trace.layer_list
        for op in layer.ops
        if op.is_buffer
    }
    assert before == after
