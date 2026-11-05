"""Forward validation on models whose ``state_dict`` holds non-tensor entries.

``state_dict`` may carry values that are not tensors: a module's
``get_extra_state`` payload, or a library ``state_dict`` override that adds a
metadata dict under its own key and pops it again in ``load_state_dict``.
Validation snapshots the state before its runs and restores it afterwards; the
snapshot must hold such entries as deep copies and hand them back through the
restore, never treat them as tensors.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens._validation_state_snapshot import (
    restore_snapshot_if_changed,
    snapshot_state_for_restore,
    state_matches_snapshot,
)


class _ExtraStateModel(nn.Module):
    """Linear layer whose extra state is a dict the forward updates."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(5, 3)
        self.info: dict[str, Any] = {"calls": 0, "tags": ["a"]}

    def get_extra_state(self) -> dict[str, Any]:
        return self.info

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.info = state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.info["calls"] += 1
        self.info["tags"].append("seen")
        return torch.relu(self.lin(x))


class _MetadataEntryModel(nn.Module):
    """A ``state_dict`` override that adds a metadata dict under its own key."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 2)
        self.init_kwargs = {"in_features": 4, "out_features": 2}

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        state = super().state_dict(*args, **kwargs)
        state["_metadata"] = dict(self.init_kwargs)
        return state

    def load_state_dict(self, state_dict: Any, strict: bool = True, assign: bool = False) -> Any:
        state_dict = OrderedDict(state_dict)
        state_dict.pop("_metadata", None)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).sigmoid()


def test_snapshot_deep_copies_extra_state_and_restore_returns_it() -> None:
    """The snapshot keeps a copy of the extra state; the restore hands back a fresh one."""

    torch.manual_seed(0)
    model = _ExtraStateModel()
    snapshot = snapshot_state_for_restore(model)
    assert snapshot.extras["_extra_state"] == {"calls": 0, "tags": ["a"]}
    assert snapshot.extras["_extra_state"] is not model.info
    assert not state_matches_snapshot(model, snapshot)

    model(torch.randn(2, 5))
    with torch.no_grad():
        model.lin.weight.add_(1.0)
    weight_before = snapshot.tensors["lin.weight"].clone()
    restore_snapshot_if_changed(model, snapshot, lambda m, sd: m.load_state_dict(sd))
    assert model.info == {"calls": 0, "tags": ["a"]}
    assert model.info is not snapshot.extras["_extra_state"]
    assert torch.equal(model.lin.weight, weight_before)

    # A second restore from the same snapshot still sees the original value.
    model(torch.randn(2, 5))
    restore_snapshot_if_changed(model, snapshot, lambda m, sd: m.load_state_dict(sd))
    assert model.info == {"calls": 0, "tags": ["a"]}


def test_forward_validation_passes_and_restores_extra_state() -> None:
    """Validation of a model with dict extra state passes and leaves it as it was."""

    torch.manual_seed(1)
    model = _ExtraStateModel()
    assert tl.validate(model, torch.randn(3, 5), scope="forward")
    assert model.info == {"calls": 0, "tags": ["a"]}


def test_forward_validation_passes_with_a_metadata_dict_entry() -> None:
    """A ``state_dict`` override that adds a dict entry validates and round-trips."""

    torch.manual_seed(2)
    model = _MetadataEntryModel()
    weight = model.lin.weight.detach().clone()
    assert tl.validate(model, torch.randn(3, 4), scope="forward")
    assert torch.equal(model.lin.weight, weight)
    assert model.state_dict()["_metadata"] == {"in_features": 4, "out_features": 2}
