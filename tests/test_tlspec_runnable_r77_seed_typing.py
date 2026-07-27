"""Typed run-door refusal for ``run(seed=<non-int>)`` (r77 nit).

Round-76 (free-roam) found ``seed`` typed ``int | None`` but unchecked: junk reached
``torch.manual_seed`` raw and escaped as torch's ``RuntimeError: manual_seed expected a
long`` instead of the typed precondition lane. Verified transactional pre-fix (global
torch RNG bit-identical after the failed call) -- pure boundary typing. The run door now
raises ``RunPreconditionError`` for any non-int, non-None ``seed`` before any provider
work; valid ``seed=0`` / ``seed=None`` runs are untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import RunPreconditionError
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


class _Small(nn.Module):
    """Minimal runnable carrier."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Affine map."""

        return self.linear(x)


def _saved(tmp_path: Path, x: torch.Tensor) -> Path:
    """Capture and save one runnable artifact."""

    path = tmp_path / "seed.tlspec"
    trace = tl.trace(_Small().eval(), x, capture=_CAPTURE)
    trace.save(path, level="runnable", include_weights=True)
    return path


@pytest.mark.smoke
@pytest.mark.parametrize(
    "bad_seed",
    ["x", 1.5, (), object()],
    ids=["str", "float", "tuple", "object"],
)
def test_r77_non_int_seed_refuses_typed_and_transactional(tmp_path: Path, bad_seed: object) -> None:
    """RED-now-fixed: junk ``seed`` raises ``RunPreconditionError``, not a raw torch error.

    The global torch RNG state must be bit-identical after the refused call, and the
    same loaded trace must still run and verify afterwards (transactional door).
    """

    torch.manual_seed(0)
    x = torch.randn(2, 4)
    path = _saved(tmp_path, x)
    loaded = tl.load(path)

    rng_before = torch.get_rng_state()
    with pytest.raises(RunPreconditionError):
        loaded.run(inputs=x.clone(), seed=bad_seed)  # type: ignore[arg-type]
    assert torch.equal(torch.get_rng_state(), rng_before)

    result = loaded.run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.smoke
@pytest.mark.parametrize("good_seed", [0, None], ids=["zero", "none"])
def test_r77_valid_seed_still_verifies(tmp_path: Path, good_seed: int | None) -> None:
    """Zero collateral: ``seed=0`` and ``seed=None`` keep their verified runs."""

    torch.manual_seed(0)
    x = torch.randn(2, 4)
    path = _saved(tmp_path, x)

    result = tl.load(path).run(inputs=x.clone(), seed=good_seed)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
