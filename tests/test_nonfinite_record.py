"""Queryable per-op nonfinite record: ``Trace.nonfinite_ops`` and friends.

Covers the two evidence bases (post-hoc saved-payload scan vs capture-time
``track_nonfinite`` recording), the coverage disclosure that keeps a scoped
clean answer honest, the deferred-flag drain, the zero-cost-when-off
contract, and the structure-only conflict refusal. ``raise_on_nan`` is a
stop-and-throw and stays untouched; nothing here weakens it.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes._nonfinite import (
    _CAPTURE_STORE_ATTR,
    drain_pending_nonfinite,
)
from torchlens._errors import StructureOnlyOptionConflictError


class _NanModel(nn.Module):
    """Deterministic model whose truediv output holds Inf/NaN."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce a non-finite output via division by zero.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Non-finite output tensor.
        """

        y = torch.relu(self.fc(x))
        y = torch.tanh(y)
        return y / 0.0


class _CleanModel(nn.Module):
    """Deterministic model with every activation finite."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply finite-only ops.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Finite output tensor.
        """

        return torch.tanh(torch.relu(x)) + 1.0


def test_drain_settles_pending_flags_and_quarantines_failures() -> None:
    """Deferred flags read in one batch; unreadable flags disclose as unchecked."""

    log = tl.trace(_CleanModel(), torch.randn(2, 4))
    store = {
        "events": {},
        "unchecked": [],
        "pending": [
            ("finite_op_raw", torch.tensor(True)),
            ("nonfinite_op_raw", torch.tensor(False)),
        ],
    }
    log.__dict__[_CAPTURE_STORE_ATTR] = store
    drain_pending_nonfinite(log)
    assert store["pending"] == []
    assert store["events"] == {"finite_op_raw": False, "nonfinite_op_raw": True}

    class _Unreadable:
        def item(self) -> bool:
            raise RuntimeError("device read failed")

    store["pending"] = [("broken_raw", _Unreadable())]
    drain_pending_nonfinite(log)
    assert store["pending"] == []
    assert store["unchecked"] == ["broken_raw"]


def test_structure_only_refuses_track_nonfinite() -> None:
    """A value-dependent per-op check cannot combine with structure-only capture."""

    with pytest.raises(StructureOnlyOptionConflictError) as excinfo:
        tl.trace(
            _CleanModel(),
            torch.randn(2, 4),
            capture=tl.options.CaptureOptions(structure_only=True, on_nonfinite="track"),
        )
    assert excinfo.value.fields["code"] == "structure_only_option_conflict"


class _BufferedNanModel(nn.Module):
    """BatchNorm-bearing model whose output holds NaN (buffer-source coverage)."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize then divide by zero.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Non-finite output tensor.
        """

        return self.bn(x) / 0.0


def test_raise_on_nan_survives_buffer_sources() -> None:
    """raise_on_nan on a buffered model completes clean / aborts on NaN.

    Regression: the ``tensor.numel()`` read ran OUTSIDE ``pause_logging``, and
    ``numel`` is a wrapped call -- on a just-committed BUFFER source tensor it
    re-entered buffer source logging and recursed without bound, so
    ``raise_on_nan=True`` crashed with RecursionError on ANY BatchNorm-bearing
    model (resnet18 included). Buffered models had simply never been exercised
    with the tripwire armed.
    """

    class _BufferedClean(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn = nn.BatchNorm1d(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.bn(x))

    log = tl.trace(
        _BufferedClean().eval(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(on_nonfinite="raise"),
    )
    assert str(log.outcome.status).endswith("COMPLETE")
    with pytest.raises(Exception) as excinfo:
        tl.trace(
            _BufferedNanModel().eval(),
            torch.randn(2, 4),
            capture=tl.options.CaptureOptions(on_nonfinite="raise"),
        )
    assert "finite" in str(excinfo.value).lower() or "nan" in str(excinfo.value).lower()


def test_loaded_trace_restores_default_and_serves_saved_basis(tmp_path) -> None:
    """track_nonfinite is session-time (DROP): load restores the default, and the
    loaded record derives from the archived saved payloads."""

    log = tl.trace(
        _NanModel(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(on_nonfinite="track"),
    )
    assert log.nonfinite_coverage.basis == "capture"
    path = tmp_path / "nonfinite.tlspec"
    log.save(path)
    loaded = tl.load(path)
    assert loaded.track_nonfinite is False
    assert loaded.nonfinite_coverage.basis == "saved_payloads"
    assert any(label.startswith("truediv") for label in loaded.nonfinite_ops)
