"""R10-15: sparse-runnable ``tensor_slots`` slot_id uniqueness was enforced
nowhere, while every sibling namespace (call ids, witness families, registry
ids, boundary positions) dup-checks. A duplicated slot_id parsed clean, then
last-wins slot indexes disagreed with list-iterating consumers and the binder
dropped one value at collapse. The descriptor parser now refuses at parse.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import runnable_load
from torchlens.options import CaptureOptions


class _SlotModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


def _runnable_descriptor(tmp_path: Path) -> dict:
    trace = tl.trace(
        _SlotModel(),
        torch.randn(2, 4),
        capture=CaptureOptions(intervention_ready=True, cache=False),
    )
    path = tmp_path / "sparse.tlspec"
    trace.save(path, level="runnable")
    import json

    with (path / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest["run"]


def test_honest_descriptor_parses(tmp_path: Path) -> None:
    run = _runnable_descriptor(tmp_path)
    descriptor = runnable_load.parse_sparse_run_descriptor(run)
    ids = [slot.slot_id for slot in descriptor.tensor_slots]
    assert len(ids) == len(set(ids))


def test_duplicate_slot_id_refuses_at_parse(tmp_path: Path) -> None:
    run = _runnable_descriptor(tmp_path)
    assert run["tensor_slots"], "test model produced no tensor slots"
    forged = copy.deepcopy(run)
    # Duplicate the first slot verbatim -> two entries share one slot_id.
    forged["tensor_slots"].append(copy.deepcopy(forged["tensor_slots"][0]))
    with pytest.raises(runnable_load.ContextFieldInvalidError) as excinfo:
        runnable_load.parse_sparse_run_descriptor(forged)
    assert "duplicate tensor-slot id" in str(excinfo.value)
