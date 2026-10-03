"""``tl.validate(scope="forward")`` settles ``False`` on a metadata-invariant failure (W051-HONESTY L9).

Before: stale-parent models returned ``False`` while the worker-thread escape
model RAISED ``MetadataInvariantError`` through the same door, so callers
branching on the documented bool missed the second failure class. The
invariant is not weakened: it still fires, still records its identity on the
side channel, and the lower-level ``validate_forward_pass`` door still raises.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import TorchLensWarning
from torchlens.user_funcs import validate_forward_pass
from torchlens.validation import invariants as _invariants
from torchlens.validation.diagnostics import CHECK_METADATA_INVARIANT, last_validation_failure
from torchlens.validation.invariants import MetadataInvariantError


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


def _arm_forced_invariant(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forced(trace):  # noqa: ANN001 -- mirrors check_metadata_invariants(trace)
        raise MetadataInvariantError("module_hierarchy", "forced by the L9 regression test")

    monkeypatch.setattr(_invariants, "check_metadata_invariants", _forced)


def test_validate_forward_returns_false_on_metadata_invariant_failure(monkeypatch) -> None:
    _arm_forced_invariant(monkeypatch)
    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(2, 4)
    with pytest.warns(TorchLensWarning, match="tl.validate FAILED"):
        verdict = tl.validate(model, x, scope="forward")
    assert verdict is False
    failure = last_validation_failure()
    assert failure is not None
    assert failure.check == CHECK_METADATA_INVARIANT
    assert "forced by the L9 regression test" in failure.message


def test_validate_forward_without_metadata_is_unaffected(monkeypatch) -> None:
    _arm_forced_invariant(monkeypatch)
    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(2, 4)
    assert tl.validate(model, x, scope="forward", validate_metadata=False) is True


def test_lower_level_validate_forward_pass_still_raises(monkeypatch) -> None:
    """The tripwire is not weakened: the direct door keeps raising."""

    _arm_forced_invariant(monkeypatch)
    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(2, 4)
    with pytest.raises(MetadataInvariantError):
        validate_forward_pass(model, x)
