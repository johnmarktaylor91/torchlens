"""A06 capture-options truth: a FAILED capture leaves the model clean.

M(oracles) item 8, the FORK-A cell all three labs agreed on: every FAILED
call is PURE of TorchLens instrumentation under either state-contract branch.
Historically a failed capture kept the per-module instance ``forward``
wrappers installed (the unconditional instance-forward installation -- the
one root cause behind three impure surfaces), so the model could no longer be
pickled and the failure warning's claim of restoration was false. The
capture entry now releases the preparation on the failure path; the
success-path lifecycle (persistent wrappers until ``tl.release_model``) is
deliberately unchanged.
"""

from __future__ import annotations

import pickle

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._save_budget import SaveBudgetExceededError
from torchlens.options import CaptureOptions


class SmallNet(nn.Module):
    """fc -> relu."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


class MidForwardBoom(nn.Module):
    """Raises from user code after one real op."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        raise RuntimeError("user forward failure")
        return y  # pragma: no cover


def _fail_capture(model: nn.Module) -> BaseException:
    with pytest.raises(SaveBudgetExceededError) as excinfo:
        tl.trace(model, torch.randn(8, 4), capture=CaptureOptions(save_budget=1))
    return excinfo.value


def test_failed_capture_removes_instance_forwards_and_pickles() -> None:
    """After a failed capture the model is as if never traced."""

    model = SmallNet()
    _fail_capture(model)
    assert "forward" not in model.fc.__dict__, (
        "the failed capture left TorchLens' instance forward wrapper installed"
    )
    pickle.dumps(model)


def test_failed_capture_then_retrace_keeps_module_attribution() -> None:
    """The release evicts prep bookkeeping so the next capture re-prepares.

    Stripping wrappers while leaving the prepared-model registry entry would
    make the next capture skip decoration and silently lose module
    containment; this pins the full-release semantics.
    """

    model = SmallNet()
    _fail_capture(model)
    log = tl.trace(model, torch.randn(2, 4))
    assert log["linear_1_1"].modules, (
        "module containment lost after a failed-capture release + re-trace"
    )
    tl.release_model(model)
    pickle.dumps(model)


def test_user_forward_failure_also_releases() -> None:
    """The purity contract covers user-code failures, not just TL refusals."""

    model = MidForwardBoom()
    with pytest.raises(RuntimeError, match="user forward failure"):
        tl.trace(model, torch.randn(2, 4))
    assert "forward" not in model.fc.__dict__
    pickle.dumps(model)


def test_partial_recovery_survives_the_release() -> None:
    """exc.partial_log stays recoverable after the model is released."""

    model = MidForwardBoom()
    with pytest.raises(RuntimeError) as excinfo:
        tl.trace(model, torch.randn(2, 4))
    partial = tl.partial.from_failed_capture(excinfo.value)
    assert partial is not None
    func_names = [getattr(op, "func_name", "") for op in partial.raw_layers]
    assert any(name == "linear" for name in func_names), func_names


def test_secondary_release_failure_warns_coded_and_never_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A release failure surfaces as failed_capture_release_incomplete."""

    import warnings

    from torchlens.backends.torch import model_prep
    from torchlens.errors import TorchLensWarning

    def _boom_release(model: nn.Module) -> None:
        raise RuntimeError("simulated release failure")

    monkeypatch.setattr(model_prep, "release_model", _boom_release)
    model = SmallNet()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(SaveBudgetExceededError):
            tl.trace(model, torch.randn(8, 4), capture=CaptureOptions(save_budget=1))
    codes = [
        w.message.fields.get("code")
        for w in caught
        if isinstance(w.message, TorchLensWarning) and hasattr(w.message, "fields")
    ]
    assert "failed_capture_release_incomplete" in codes, codes


def test_successful_capture_lifecycle_is_unchanged() -> None:
    """Success keeps the persistent wrappers until tl.release_model."""

    model = SmallNet()
    tl.trace(model, torch.randn(2, 4))
    assert "forward" in model.fc.__dict__, (
        "the failure-path release must not leak onto the success path; the "
        "persistent-preparation lifecycle is deliberate (re-capture speed) "
        "and its dissolution is List-B feature work, not this fix"
    )
    tl.release_model(model)
    assert "forward" not in model.fc.__dict__
    pickle.dumps(model)
