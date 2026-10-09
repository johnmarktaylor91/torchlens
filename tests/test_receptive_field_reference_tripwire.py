"""The reference-mode mutation tripwire still fires in ``scope="receptive_field"``.

``tl.validate(scope="receptive_field")`` captures with ``save_mode="reference"``
and runs the metadata invariants, which read every saved reference and raise
``MutatedReferenceError`` when its version counter moved after it was saved.
A BatchNorm buffer version node stores the write journal's private copy, so
its stamp must describe that copy; these tests pin that the fix for the eval
BatchNorm false positive left the tripwire armed on every real mutation path:
a saved activation, a registered buffer the forward writes in place, and the
version node's own stored copy.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes import op as op_module
from torchlens.options import CaptureOptions

_X = torch.tensor([[1.0, -2.0, 0.5, 0.25], [0.5, 1.5, -1.0, 2.0], [-0.5, 0.0, 1.0, -1.5]])


class _MutatesSavedActivation(nn.Module):
    """Writes into a hidden activation after its consumer has read it."""

    def __init__(self) -> None:
        """Build the layers."""

        super().__init__()
        self.inp = nn.Linear(4, 5)
        self.head = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stack, then scale ``hidden`` in place."""

        hidden = self.inp(x)
        out = self.head(torch.relu(hidden))
        hidden.mul_(2.0)
        return out


class _WritesItsBuffer(nn.Module):
    """Reads a registered buffer, then writes it in place."""

    def __init__(self) -> None:
        """Build the layer and the buffer."""

        super().__init__()
        self.inp = nn.Linear(4, 5)
        self.register_buffer("shift", torch.linspace(-0.5, 0.5, 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the buffer, then advance it in place."""

        out = self.inp(x) + self.shift
        self.shift.add_(1.0)
        return out


class _BatchNormModel(nn.Module):
    """Linear, BatchNorm1d with non-trivial running statistics, linear."""

    def __init__(self) -> None:
        """Build the layers and seed the running statistics."""

        super().__init__()
        self.inp = nn.Linear(4, 5)
        self.bn = nn.BatchNorm1d(5)
        self.head = nn.Linear(5, 2)
        with torch.no_grad():
            self.bn.running_mean.copy_(torch.linspace(-0.5, 0.5, 5))
            self.bn.running_var.copy_(torch.linspace(0.5, 1.5, 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stack."""

        return self.head(torch.relu(self.bn(self.inp(x))))


def _validate_rf(model: nn.Module) -> object:
    """Run receptive-field validation with warnings silenced."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tl.validate(model, _X, scope="receptive_field")


def test_mutated_saved_activation_raises_in_receptive_field_scope() -> None:
    """An in-place write to a saved activation fails receptive-field validation."""

    torch.manual_seed(3)
    with pytest.raises(tl.errors.MutatedReferenceError, match="mutated after capture"):
        _validate_rf(_MutatesSavedActivation().eval())


def test_buffer_written_in_place_raises_in_receptive_field_scope() -> None:
    """A forward that writes a buffer it read fails receptive-field validation."""

    torch.manual_seed(3)
    with pytest.raises(tl.errors.MutatedReferenceError, match="mutated after capture"):
        _validate_rf(_WritesItsBuffer().eval())


def test_train_mode_batchnorm_is_refused_in_receptive_field_scope() -> None:
    """Train-mode BatchNorm is refused, by design, not passed.

    A train-mode BatchNorm updates ``running_mean``, ``running_var`` and
    ``num_batches_tracked`` in place during the forward. The scope's
    reference-mode capture saved those buffers by reference before the
    update, so the saved values no longer are what the op read: the tripwire
    must refuse rather than validate against post-update statistics.
    """

    torch.manual_seed(3)
    with pytest.raises(tl.errors.MutatedReferenceError, match="mutated after capture"):
        _validate_rf(_BatchNormModel().train())


def test_buffer_version_node_copy_is_still_guarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The version node's stored copy reads cleanly, then raises once mutated."""

    monkeypatch.setattr(op_module, "_WARNED_REFERENCE_SAVE_MODE", True)
    torch.manual_seed(3)
    trace = tl.trace(
        _BatchNormModel().eval(),
        _X,
        save_mode="reference",
        capture=CaptureOptions(layers_to_save="all"),
    )
    version_nodes = [op for op in trace.layer_list if op._slot("buffer_write_kind") == "fused"]
    assert version_nodes, "eval BatchNorm logged no buffer version node"
    for node in version_nodes:
        assert isinstance(node.out, torch.Tensor)

    version_nodes[0]._slot("out").add_(1.0)
    with pytest.raises(tl.errors.MutatedReferenceError, match="mutated after capture"):
        _ = version_nodes[0].out
