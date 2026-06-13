"""Phase 6 capture-unification storage-axis tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._training_validation import TrainingModeConfigError
from torchlens.fastlog import CaptureSpec, PredicateError, RecordContext


class EncoderBlock(nn.Module):
    """Tiny model with a module-scoped selective save site."""

    def __init__(self) -> None:
        """Initialize deterministic linear weights."""

        super().__init__()
        self.encoder = nn.Linear(4, 4)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(4))
            self.encoder.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one encoder layer followed by an unsaved relu."""

        hidden = self.encoder(x)
        return torch.relu(hidden)


def _keep_linear_grad(ctx: RecordContext) -> CaptureSpec | bool:
    """Request graph-connected capture at linear ops."""

    if ctx.func_name == "linear":
        return CaptureSpec(keep_grad=True)
    return False


def _saved_linear(log: tl.Trace) -> tl.Layer:
    """Return the saved linear layer from a trace."""

    return next(layer for layer in log.layer_list if layer.layer_type == "linear")


def test_trace_predicate_save_to_disk_loads_layer_out(tmp_path: Path) -> None:
    """Predicate-selected trace saves stream to disk and load through ``Layer.out``."""

    model = EncoderBlock()
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    expected = _saved_linear(tl.trace(model, x, save=tl.in_module("encoder"))).out

    log = tl.trace(
        model,
        x,
        save=tl.in_module("encoder"),
        storage=tl.to_disk(tmp_path / "streamed.tlspec"),
    )
    saved = _saved_linear(log)

    assert saved.ops[0].out_ref is not None
    assert saved.ops[0]._slot("out") is None
    assert torch.equal(saved.out, expected)
    assert torch.equal(saved.ops[0].out, expected)


def test_trace_predicate_disk_only_rejects_keep_grad(tmp_path: Path) -> None:
    """Disk-only predicate storage rejects graph-connected payload requests."""

    with pytest.raises(PredicateError, match="keep_grad=True.*disk-only"):
        tl.trace(
            EncoderBlock(),
            torch.ones(1, 4),
            save=_keep_linear_grad,
            storage=tl.to_disk(tmp_path / "keep_grad.tlspec"),
        )


def test_trace_backward_ready_rejects_disk_only_storage(tmp_path: Path) -> None:
    """``backward_ready=True`` rejects disk-only predicate storage."""

    with pytest.raises(TrainingModeConfigError, match="backward_ready=True.*disk"):
        tl.trace(
            EncoderBlock(),
            torch.ones(1, 4),
            save=tl.func("linear"),
            storage=tl.to_disk(tmp_path / "backward_ready.tlspec"),
            backward_ready=True,
        )


def test_trace_predicate_storage_none_remains_ram_backed() -> None:
    """Default predicate save storage keeps the existing RAM-backed behavior."""

    log = tl.trace(EncoderBlock(), torch.ones(1, 4), save=tl.func("linear"), storage=None)
    saved = _saved_linear(log)

    assert saved.ops[0].out_ref is None
    assert isinstance(saved.out, torch.Tensor)


def test_trace_disk_streamed_save_load_round_trip(tmp_path: Path) -> None:
    """A disk-streamed trace can be saved and loaded with the same payload."""

    x = torch.ones(1, 4)
    log = tl.trace(
        EncoderBlock(),
        x,
        save=tl.func("linear"),
        storage=tl.to_disk(tmp_path / "streamed.tlspec"),
    )
    expected = _saved_linear(log).out
    bundle_path = tmp_path / "round_trip.tlspec"

    tl.save(log, bundle_path)
    loaded = tl.load(bundle_path)

    assert isinstance(loaded, tl.Trace)
    assert torch.equal(_saved_linear(loaded).out, expected)
