"""A multi-pass Recorder keeps each pass's records exactly once.

Each pass of a ``Recorder`` seals its own run core, and the Recording's records
are projected from those cores. A selector ``save=`` used to store the shared,
cumulative record list in every core, so k passes projected 1 + 2 + ... + k
records. These tests pin one record per selected site per pass, identical to
the plain-callable ``save=`` spelling, and a disk bundle that loads the same.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import Recorder
from torchlens.fastlog.recover import load
from torchlens.options import StorageOptions


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 3)).eval()


def _inputs() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return [torch.randn(1, 3, generator=generator) for _ in range(3)]


def _record_passes(save, **recorder_kwargs) -> tl.Recording:
    model = _model()
    with Recorder(model, save=save, **recorder_kwargs) as recorder:
        for x in _inputs():
            recorder((x,))
    return recorder.recording


def _is_relu(ctx) -> bool:
    return ctx.kind == "op" and ctx.func_name == "relu"


def test_selector_save_keeps_one_record_per_pass() -> None:
    recording = _record_passes(tl.func("relu"))

    assert [record.ctx.pass_index for record in recording.records] == [1, 2, 3]
    assert recording.n_records == 3
    assert recording.by_pass == {1: [0], 2: [1], 3: [2]}


def test_selector_save_matches_plain_callable_record_for_record() -> None:
    by_selector = _record_passes(tl.module("1"))
    by_callable = _record_passes(_is_relu)

    assert len(by_selector.records) == len(by_callable.records) == 3
    for left, right in zip(by_selector.records, by_callable.records, strict=True):
        assert left.ctx.pass_index == right.ctx.pass_index
        assert left.ctx.label == right.ctx.label
        assert torch.equal(left.ram_payload, right.ram_payload)
    assert by_selector.by_pass == by_callable.by_pass
    assert by_selector.by_label == by_callable.by_label


def test_selector_save_disk_bundle_loads_one_record_per_pass(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    recording = _record_passes(
        tl.func("relu"), streaming=StorageOptions(path=bundle, retain_in_memory=True)
    )
    loaded = load(bundle)

    assert [record.ctx.pass_index for record in loaded.records] == [1, 2, 3]
    assert [record.ctx.pass_index for record in recording.records] == [1, 2, 3]
