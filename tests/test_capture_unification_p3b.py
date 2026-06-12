"""Phase 3b capture-unification alias, compatibility, and orphan tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import TorchLensIOError
from torchlens.fastlog import RecordContext


class ViewThenMutate(nn.Module):
    """Model with a view alias that mutates an unsaved parent."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a view-then-in-place mutation before a saved child."""

        parent = x + 1
        view = parent.view_as(parent)
        view.add_(2)
        return parent * 3


class OutKwargMutate(nn.Module):
    """Model using a non-first mutated input through an ``out=`` kwarg."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write into an unsaved parent via ``out=`` before a saved child."""

        parent = x + 1
        torch.add(x, 2, out=parent)
        return parent * 3


class MultiOutputAlias(nn.Module):
    """Model where a multi-output view aliases an unsaved parent."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mutate a split view before saving a later child."""

        parent = x + 1
        first, _second = parent.split(2, dim=1)
        first.add_(2)
        return parent * 3


class TinyLinear(nn.Module):
    """Small differentiable model for compatibility matrix checks."""

    def __init__(self) -> None:
        """Initialize the linear layer."""

        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a simple linear forward pass."""

        return torch.relu(self.linear(x))


class SavedOrphan(nn.Module):
    """Model producing a saved unused factory tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create an orphan tensor and return an unrelated output."""

        _unused = torch.randn(x.shape)
        return x + 1


def _save_only_mul(ctx: RecordContext) -> bool:
    """Select only the downstream multiplication op."""

    return ctx.func_name in {"__mul__", "mul"}


@pytest.mark.parametrize(
    "model_factory",
    (ViewThenMutate, OutKwargMutate, MultiOutputAlias),
)
def test_alias_contract_snapshots_unsaved_mutated_parent(
    model_factory: Callable[[], nn.Module],
) -> None:
    """Selective save forces parent snapshots for aliased unsaved parents."""

    model = model_factory()
    x = torch.ones(2, 4)
    full = tl.trace(model, x.clone(), layers_to_save="all", save_arg_values=True)
    selective = tl.trace(model, x.clone(), save=_save_only_mul, save_arg_values=True)
    saved_children = [
        op
        for op in selective.layer_list
        if op.has_saved_activation and op.func_name in {"__mul__", "mul"}
    ]
    assert saved_children
    unsaved_snapshot_parents = [
        op
        for op in selective.layer_list
        if not op.has_saved_activation and bool(op.out_versions_by_child)
    ]
    assert unsaved_snapshot_parents
    assert any(op.out_versions_by_child for op in full.layer_list)

    expected = [model_factory()(x.clone()).detach().clone()]
    try:
        assert selective.validate_forward_pass(expected, validate_metadata=False)
    except ValueError as exc:
        assert "Cannot validate saved layer" in str(exc) or "was not saved" in str(exc)


def test_layers_to_save_rejects_disk_streaming_with_save_guidance(tmp_path: Path) -> None:
    """Selective two-pass disk streaming rejects with predicate save guidance."""

    with pytest.raises(TorchLensIOError, match="predicate save=.*selective streaming"):
        tl.trace(TinyLinear(), torch.randn(2, 4), layers_to_save=["linear"], save_outs_to=tmp_path)


def test_layers_to_save_supports_backward_ready_and_gradients_two_pass() -> None:
    """Selective two-pass supports backward-ready and gradient selection."""

    log = tl.trace(
        TinyLinear(),
        torch.randn(2, 4),
        layers_to_save=["linear"],
        save_grads=["linear"],
        backward_ready=True,
    )
    saved = [op for op in log.layer_list if op.has_saved_activation and op.layer_type == "linear"]
    assert saved
    assert saved[0].out.requires_grad


def test_layers_to_save_rejects_intervention_ready_and_hooks() -> None:
    """Selective two-pass rejects intervention-ready and hook capture."""

    x = torch.randn(2, 4)
    with pytest.raises(ValueError, match="intervention_ready=True.*selective two-pass"):
        tl.trace(TinyLinear(), x, layers_to_save=["linear"], intervention_ready=True)
    with pytest.raises(ValueError, match="hooks/intervention capture.*selective two-pass"):
        tl.trace(TinyLinear(), x, layers_to_save=["linear"], hooks=tl.tap(tl.func("relu")))


def test_orphan_records_expose_saved_payload_when_pruned_or_retained() -> None:
    """Saved orphan payloads are exposed regardless of graph pruning mode."""

    x = torch.ones(2, 2)
    pruned = tl.trace(SavedOrphan(), x, save=tl.func("randn"), random_seed=1)
    assert pruned.orphan_records
    assert pruned.orphan_records[0]["raw_label"].startswith("randn")
    assert isinstance(pruned.orphan_records[0]["payload_ref"], torch.Tensor)
    assert not any(label.startswith("randn") for label in pruned.op_labels)

    retained = tl.trace(
        SavedOrphan(),
        x,
        save=tl.func("randn"),
        random_seed=1,
        keep_orphans=True,
    )
    assert retained.orphan_records
    assert retained.orphans
