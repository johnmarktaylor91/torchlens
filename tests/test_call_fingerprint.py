"""Capture-time ordered call fingerprint vs its native-forward twin.

Every torch capture stores ``Trace._raw_call_fingerprint``: an ordered rolling
hash over the unpaused owner-thread torch calls and non-root module entries of
the captured forward. ``torchlens._call_fingerprint.fingerprint_native_forward``
reproduces the same stream around a plain forward, so a guarded fast re-run can
prove the same op structure ran (e.g. on a different-length input).
"""

from __future__ import annotations

import threading
import zlib
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._call_fingerprint import (
    fingerprint_native_forward,
    fingerprinting,
    install_module_token_hooks,
)
from torchlens.options import CaptureOptions


class _Block(nn.Module):
    """Linear + ReLU block."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc = nn.Linear(width, width)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block."""
        return self.act(self.fc(x))


class _NestedMLP(nn.Module):
    """Two nested blocks and a head."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(_Block(4), _Block(4))
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the blocks and the head."""
        return self.head(self.blocks(x))


class _SharedTwice(nn.Module):
    """Calls one submodule twice."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``fc`` twice with a nonlinearity between."""
        return self.fc(torch.tanh(self.fc(x)))


class _InplaceFunctional(nn.Module):
    """Mixes in-place and functional ops."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run in-place relu/add, a cat, and indexing."""
        y = self.fc(x)
        y.relu_()
        y.add_(1.0)
        z = torch.cat([y, x], dim=-1)
        return z[:, :3] * 2


class _TwoLayer(nn.Module):
    """``fc1 -> relu -> fc2``."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the two layers."""
        return self.fc2(torch.relu(self.fc1(x)))


class _ShapeBranch(nn.Module):
    """Takes a different arm depending on the input width."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on ``x.shape[1] > 4``."""
        if x.shape[1] > 4:
            return torch.relu(x) + 1
        return torch.tanh(x) * 2


class _SequenceMLP(nn.Module):
    """Per-token MLP with layer norm over ``(1, L, d)`` inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(6)
        self.up = nn.Linear(6, 12)
        self.down = nn.Linear(12, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a residual pre-norm MLP."""
        return x + self.down(torch.nn.functional.gelu(self.up(self.norm(x))))


def _seeded(model: nn.Module) -> nn.Module:
    """Return ``model`` in eval mode (construction already seeded by the caller)."""
    return model.eval()


class _BoolGateIdentity(nn.Module):
    """A scalar-bool gate (TorchLens reads its value) plus an Identity module."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.readout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on a scalar bool, then pass through Identity."""
        y = self.fc(x)
        if (y.abs() >= 0).all():
            y = y * 2
        return self.readout(y)


def _native(model: nn.Module, x: torch.Tensor) -> tuple[int, int]:
    """Return the native-forward fingerprint of ``model(x)``."""
    value, _ = fingerprint_native_forward(model, (x,), {})
    return value


@pytest.mark.parametrize(
    "model_cls", [_NestedMLP, _SharedTwice, _InplaceFunctional, _BoolGateIdentity]
)
def test_capture_fingerprint_equals_native_forward(model_cls: type[nn.Module]) -> None:
    """Capture and a plain forward fold the identical ordered token stream."""
    torch.manual_seed(0)
    model = _seeded(model_cls())
    x = torch.randn(2, 4)
    trace = tl.trace(model, x)
    captured = trace._raw_call_fingerprint  # noqa: SLF001
    assert captured is not None and captured[0] > 0
    assert captured == _native(model, x)


def test_shared_module_tokens_count_every_call() -> None:
    """A module called twice contributes two entry tokens (order-sensitive)."""
    torch.manual_seed(0)
    model = _seeded(_SharedTwice())
    x = torch.randn(2, 4)
    tl.trace(model, x)
    with fingerprinting() as fp:
        handles = install_module_token_hooks(model)
        try:
            model(x)
        finally:
            for handle in handles:
                handle.remove()
    with fingerprinting() as ops_only:
        model(x)
    assert fp.count - ops_only.count == 2


def test_module_intervention_and_sparse_save_keep_the_fingerprint() -> None:
    """Hook arithmetic and save selection never enter the structural fingerprint."""
    torch.manual_seed(0)
    model = _seeded(_TwoLayer())
    x = torch.randn(2, 4)
    plain = tl.trace(model, x)._raw_call_fingerprint  # noqa: SLF001
    direction = torch.linspace(-1.0, 1.0, 8)
    steered = tl.trace(
        model,
        x,
        intervene=tl.when(tl.module("fc1"), tl.steer(direction, magnitude=2, feature_axis=-1)),
    )
    sparse = tl.trace(model, x, save=tl.module("fc1"))
    assert plain == _native(model, x)
    assert steered._raw_call_fingerprint == plain  # noqa: SLF001
    assert sparse._raw_call_fingerprint == plain  # noqa: SLF001


def test_shape_dependent_branch_changes_the_fingerprint() -> None:
    """Two control-flow arms produce different fingerprints in both engines."""
    model = _ShapeBranch()
    wide, narrow = torch.randn(1, 6), torch.randn(1, 3)
    wide_fp = tl.trace(model, wide)._raw_call_fingerprint  # noqa: SLF001
    narrow_fp = tl.trace(model, narrow)._raw_call_fingerprint  # noqa: SLF001
    assert wide_fp != narrow_fp
    assert _native(model, wide) == wide_fp
    assert _native(model, narrow) == narrow_fp


def test_different_length_input_with_same_structure_matches() -> None:
    """A longer sequence through the same structure reproduces the capture's value."""
    torch.manual_seed(0)
    model = _seeded(_SequenceMLP())
    short, long = torch.randn(1, 5, 6), torch.randn(1, 9, 6)
    captured = tl.trace(model, short)._raw_call_fingerprint  # noqa: SLF001
    assert _native(model, long) == captured
    assert tl.trace(model, long)._raw_call_fingerprint == captured  # noqa: SLF001


def test_fingerprint_is_runtime_only(tmp_path: Path) -> None:
    """A loaded trace never claims a fingerprint (FieldPolicy.DROP)."""
    torch.manual_seed(0)
    model = _seeded(_TwoLayer())
    trace = tl.trace(model, torch.randn(2, 4))
    assert trace._raw_call_fingerprint is not None  # noqa: SLF001
    path = tmp_path / "fp.tlspec"
    trace.save(path)
    loaded = tl.load(path)
    assert getattr(loaded, "_raw_call_fingerprint", None) is None


def test_legacy_rerun_refreshes_the_fingerprint() -> None:
    """``trace.run`` stores the rerun's own fingerprint on the trace."""
    torch.manual_seed(0)
    model = _seeded(_TwoLayer())
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=CaptureOptions(intervention_ready=True))
    original = trace._raw_call_fingerprint  # noqa: SLF001
    trace._raw_call_fingerprint = (0, 0)  # noqa: SLF001
    new_x = torch.randn(2, 4)
    trace.run(model, new_x)
    assert trace._raw_call_fingerprint == original == _native(model, new_x)  # noqa: SLF001


def test_paused_and_foreign_thread_calls_are_excluded() -> None:
    """Only unpaused calls on the fingerprint's owner thread count."""
    tl.trace(_TwoLayer(), torch.randn(1, 4))  # make sure torch is wrapped
    x = torch.randn(3)
    with fingerprinting() as fp:
        torch.relu(x)
        with _state.pause_logging():
            torch.tanh(x)
            torch.sigmoid(x)
        worker = threading.Thread(target=lambda: torch.exp(x))
        worker.start()
        worker.join()
        torch.relu(x)
    with fingerprinting() as reference:
        torch.relu(x)
        torch.relu(x)
    assert fp.value == reference.value
    assert _state._pause_depth == 0  # noqa: SLF001
    assert _state._call_fingerprint is None  # noqa: SLF001


def test_rolling_hash_is_ordered_and_deterministic() -> None:
    """Token order changes the digest; tokens are stable CRC32 values."""
    first = _state.CallFingerprint(threading.get_ident())
    second = _state.CallFingerprint(threading.get_ident())
    a, b = _state.call_token("relu"), _state.module_token("fc1")
    first.add(a)
    first.add(b)
    second.add(b)
    second.add(a)
    assert first.count == second.count == 2
    assert first.value != second.value
    assert a == zlib.crc32(b"call:relu") and b == zlib.crc32(b"module:fc1")
    assert a != _state.module_token("relu")
