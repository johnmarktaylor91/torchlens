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


def test_lazy_basis_finds_nonfinite_ops_on_default_capture() -> None:
    """The default capture serves the record from saved payloads, free at capture."""

    log = tl.trace(_NanModel(), torch.randn(2, 4))
    labels = log.nonfinite_ops
    assert any(label.startswith("truediv") for label in labels)
    coverage = log.nonfinite_coverage
    assert coverage.basis == "saved_payloads"
    assert coverage.nonfinite == len(labels) > 0
    assert coverage.unexamined == 0
    # Every reported label is a valid trace key.
    for label in labels:
        assert log[label] is not None


def test_lazy_basis_clean_model_returns_empty() -> None:
    """A finite forward yields an empty record with full coverage."""

    log = tl.trace(_CleanModel(), torch.randn(2, 4))
    assert log.nonfinite_ops == ()
    coverage = log.nonfinite_coverage
    assert coverage.nonfinite == 0
    assert coverage.checked > 0


def test_track_nonfinite_records_capture_basis() -> None:
    """track_nonfinite=True serves capture-time verdicts."""

    log = tl.trace(
        _NanModel(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(track_nonfinite=True),
    )
    labels = log.nonfinite_ops
    assert any(label.startswith("truediv") for label in labels)
    coverage = log.nonfinite_coverage
    assert coverage.basis == "capture"
    assert coverage.nonfinite >= 1
    assert coverage.checked >= 4


def test_track_nonfinite_covers_unsaved_ops_on_selective_save() -> None:
    """The capture-time record sees ops the saved-payload scan cannot."""

    model = _NanModel()
    x = torch.randn(2, 4)
    lazy = tl.trace(model, x, save=tl.func("relu"))
    # The saved-payload basis is honestly blind here: the non-finite op
    # retained no payload, and coverage says so.
    assert lazy.nonfinite_ops == ()
    lazy_coverage = lazy.nonfinite_coverage
    assert lazy_coverage.basis == "saved_payloads"
    assert lazy_coverage.unexamined > 0
    tracked = tl.trace(
        model,
        x,
        save=tl.func("relu"),
        capture=tl.options.CaptureOptions(track_nonfinite=True),
    )
    assert any(label.startswith("truediv") for label in tracked.nonfinite_ops)
    assert tracked.nonfinite_coverage.basis == "capture"


def test_track_nonfinite_off_means_no_store_and_no_per_op_work() -> None:
    """The default path never allocates the capture store (zero hot-path cost)."""

    log = tl.trace(_CleanModel(), torch.randn(2, 4))
    assert log.track_nonfinite is False
    assert _CAPTURE_STORE_ATTR not in log.__dict__


def test_multipass_nonfinite_op_is_pass_qualified() -> None:
    """A hit inside a recurrence-grouped loop names the exact pass."""

    class _LoopLog(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = x
            for _ in range(2):
                # First log stays finite (input is > 1); the second log sees
                # negative values and produces NaN on pass 2.
                y = torch.log(y - 0.5)
            return y

    log = tl.trace(
        _LoopLog(),
        torch.full((2, 4), 2.0),
        capture=tl.options.CaptureOptions(track_nonfinite=True),
    )
    hits = [label for label in log.nonfinite_ops if "log" in label]
    assert hits, log.nonfinite_ops
    assert all(":" in label for label in log.nonfinite_ops)
    assert any(label.endswith(":2") for label in hits)


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
            capture=tl.options.CaptureOptions(structure_only=True, track_nonfinite=True),
        )
    assert excinfo.value.fields["code"] == "structure_only_option_conflict"


def test_raise_on_nan_stop_and_throw_unchanged_by_tracking() -> None:
    """raise_on_nan still aborts at the first non-finite op with tracking on."""

    with pytest.raises(Exception) as excinfo:
        tl.trace(
            _NanModel(),
            torch.randn(2, 4),
            capture=tl.options.CaptureOptions(raise_on_nan=True, track_nonfinite=True),
        )
    assert "nan" in str(excinfo.value).lower() or "finite" in str(excinfo.value).lower()


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
        capture=tl.options.CaptureOptions(raise_on_nan=True),
    )
    assert str(log.outcome.status).endswith("COMPLETE")
    with pytest.raises(Exception) as excinfo:
        tl.trace(
            _BufferedNanModel().eval(),
            torch.randn(2, 4),
            capture=tl.options.CaptureOptions(raise_on_nan=True),
        )
    assert "finite" in str(excinfo.value).lower() or "nan" in str(excinfo.value).lower()


def test_track_nonfinite_survives_buffer_sources() -> None:
    """The recorder handles buffer source commits without recursion."""

    log = tl.trace(
        _BufferedNanModel().eval(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(track_nonfinite=True),
    )
    assert any(label.startswith("truediv") for label in log.nonfinite_ops)


def test_loaded_trace_restores_default_and_serves_saved_basis(tmp_path) -> None:
    """track_nonfinite is session-time (DROP): load restores the default, and the
    loaded record derives from the archived saved payloads."""

    log = tl.trace(
        _NanModel(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(track_nonfinite=True),
    )
    assert log.nonfinite_coverage.basis == "capture"
    path = tmp_path / "nonfinite.tlspec"
    tl.save(log, path)
    loaded = tl.load(path)
    assert loaded.track_nonfinite is False
    assert loaded.nonfinite_coverage.basis == "saved_payloads"
    assert any(label.startswith("truediv") for label in loaded.nonfinite_ops)
