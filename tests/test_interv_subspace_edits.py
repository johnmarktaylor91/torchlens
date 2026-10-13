"""k-dimensional subspace ablation modes and basis interchange (R2-25, N-IV-6).

Every mode is checked against an INDEPENDENT projector ``B^T pinv(B B^T) B``
(pseudo-inverse, not the helper's QR path), applied by hand to the same site
value and pushed through the rest of the model. The concept-erasure row checks
the LEACE guarantee computed in the test (no new dependency): erasing the span
of the class-mean differences leaves every class with the same mean.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.intervention import subspace_ablate, subspace_interchange
from torchlens.intervention.errors import HookValueError

_D = 8


def _model() -> nn.Sequential:
    """Two linear layers; the first one's output is the edited site."""

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(5, _D), nn.Linear(_D, 3)).eval()


def _oracle_projector(basis: torch.Tensor) -> torch.Tensor:
    """Orthogonal projector onto the row span of ``basis`` via the pseudo-inverse."""

    rows = basis.to(torch.float64)
    return rows.T @ torch.linalg.pinv(rows @ rows.T) @ rows


def _edited_output(model: nn.Sequential, x: torch.Tensor, edit: Any) -> torch.Tensor:
    """Run the edit at the first layer's output through TorchLens."""

    trace = tl.trace(model, x, intervene=tl.when(tl.module("0"), edit))
    return trace[trace.output_layers[0]].out.detach()


def _tail(model: nn.Sequential, hidden: torch.Tensor) -> torch.Tensor:
    """Apply the model after the edited site to a hand-edited hidden value."""

    with torch.no_grad():
        return model[1](hidden.to(torch.float32))


_X = torch.randn(6, 5, generator=torch.Generator().manual_seed(1))
_BASIS = torch.randn(3, _D, generator=torch.Generator().manual_seed(2))


def test_zero_mode_matches_independent_projection() -> None:
    """``mode="zero"`` is ``x - P x`` for a non-orthonormal 3-dim basis."""

    model = _model()
    hidden = model[0](_X).detach().to(torch.float64)
    projector = _oracle_projector(_BASIS)
    want = _tail(model, hidden - hidden @ projector)
    got = _edited_output(model, _X, subspace_ablate(_BASIS))
    torch.testing.assert_close(got, want, rtol=0.0, atol=1e-5)
    with torch.no_grad():
        assert not torch.allclose(got, model(_X), atol=1e-3)


def test_mean_mode_matches_independent_projection() -> None:
    """``mode="mean"`` replaces the in-span component with the mean's."""

    model = _model()
    hidden = model[0](_X).detach().to(torch.float64)
    mean = torch.randn(_D, generator=torch.Generator().manual_seed(3))
    projector = _oracle_projector(_BASIS)
    want = _tail(model, hidden - hidden @ projector + mean.to(torch.float64) @ projector)
    got = _edited_output(model, _X, subspace_ablate(_BASIS, mode="mean", mean=mean))
    torch.testing.assert_close(got, want, rtol=0.0, atol=1e-5)


def test_resample_mode_matches_independent_projection() -> None:
    """``mode="resample"`` takes the in-span component from a seeded batch permutation."""

    model = _model()
    hidden = model[0](_X).detach().to(torch.float64)
    order = torch.randperm(_X.shape[0], generator=torch.Generator().manual_seed(11))
    projector = _oracle_projector(_BASIS)
    want = _tail(model, hidden - hidden @ projector + hidden[order] @ projector)
    got = _edited_output(model, _X, subspace_ablate(_BASIS, mode="resample", seed=11))
    torch.testing.assert_close(got, want, rtol=0.0, atol=1e-5)
    again = _edited_output(model, _X, subspace_ablate(_BASIS, mode="resample", seed=11))
    assert torch.equal(got, again)


def test_interchange_matches_independent_projection() -> None:
    """The k-dim interchange writes the source's in-span component over the target's."""

    model = _model()
    source_x = torch.randn(6, 5, generator=torch.Generator().manual_seed(4))
    source = tl.trace(model, source_x)
    hidden = model[0](_X).detach().to(torch.float64)
    donor = model[0](source_x).detach().to(torch.float64)
    projector = _oracle_projector(_BASIS)
    want = _tail(model, hidden - hidden @ projector + donor @ projector)
    got = _edited_output(model, _X, subspace_interchange(_BASIS, source))
    torch.testing.assert_close(got, want, rtol=0.0, atol=1e-5)


def test_concept_erasure_equalizes_class_means() -> None:
    """Erasing the class-mean-difference span leaves no linear mean signal (LEACE property)."""

    generator = torch.Generator().manual_seed(5)
    labels = torch.arange(30) % 3
    offsets = torch.randn(3, _D, generator=generator) * 4.0
    features = torch.randn(30, _D, generator=generator) + offsets[labels]
    means = torch.stack([features[labels == c].mean(0) for c in range(3)])
    basis = means[1:] - means[0]
    model = nn.Sequential(nn.Linear(_D, _D), nn.Linear(_D, _D)).eval()
    with torch.no_grad():
        for layer in model:
            layer.weight.copy_(torch.eye(_D))
            layer.bias.zero_()
    trace = tl.trace(model, features, intervene=tl.when(tl.module("0"), subspace_ablate(basis)))
    erased = trace[trace.output_layers[0]].out.detach()
    erased_means = torch.stack([erased[labels == c].mean(0) for c in range(3)])
    before_gap = float((means - means.mean(0)).abs().max())
    after_gap = float((erased_means - erased_means.mean(0)).abs().max())
    assert before_gap > 1.0
    assert after_gap < 1e-5
    projector = _oracle_projector(basis)
    torch.testing.assert_close(
        erased.to(torch.float64),
        features.to(torch.float64) - features.to(torch.float64) @ projector,
        rtol=0.0,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"basis": torch.stack([_BASIS[0], 2 * _BASIS[0]])}, "subspace_basis_invalid"),
        ({"basis": torch.zeros(0, _D)}, "subspace_basis_invalid"),
        ({"basis": _BASIS, "mode": "flip"}, "subspace_mode_invalid"),
        ({"basis": _BASIS, "mode": "mean"}, "subspace_mode_invalid"),
        ({"basis": _BASIS, "mode": "resample"}, "subspace_mode_invalid"),
        ({"basis": _BASIS, "seed": 3}, "subspace_mode_invalid"),
    ],
)
def test_invalid_arguments_refuse_at_construction(kwargs: dict[str, Any], code: str) -> None:
    """Bad bases and inconsistent mode arguments refuse typed when the edit is built."""

    with pytest.raises(InvalidArgumentError) as info:
        subspace_ablate(**kwargs)
    assert info.value.fields["code"] == code


def test_basis_dimension_mismatch_refuses_at_fire_time() -> None:
    """A basis over the wrong width refuses typed instead of broadcasting."""

    with pytest.raises(HookValueError) as info:
        _edited_output(_model(), _X, subspace_ablate(torch.randn(2, _D + 1)))
    assert info.value.fields["code"] == "subspace_basis_dim_mismatch"
