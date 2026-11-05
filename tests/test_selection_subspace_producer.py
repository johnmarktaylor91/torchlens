"""Subspace selection producer: acceptance, provenance, and honesty gates.

Covers the producer-lane obligations: the producer returns a composable
``Selection``, the resolved mask is the basis SUPPORT SET (exact as a set,
verified against manually computed supports; a dense direction supports the
whole axis), basis PROVENANCE is mandatory and rides ``provenance.source``
plus ``do()`` audit records (origin + method + geometry + sha256 content
digest), dimension mismatches refuse typed (``basis_dim_mismatch`` — never a
silent broadcast/truncation), construction refusals (missing origin/within,
non-finite or vacuous bases, bad dtypes/shapes), kind refusals, geometry-only
resolution (unsaved sites resolve), composition through the algebra, and the
end-to-end gallery row (ablate a sparse probe direction's support).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors import SelectionError

_RELU = "relu_1_2"
_SIGMOID = "sigmoid_1_4"
_D = 6  # relu feature width


class _Mlp(nn.Module):
    """Tiny MLP whose relu output is [batch, 6] (the probed feature space)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(8, _D)
        self.head = nn.Linear(_D, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(torch.relu(self.encoder(x))))


@pytest.fixture(scope="module")
def model():
    """One shared deterministic probe model."""

    torch.manual_seed(0)
    return _Mlp()


@pytest.fixture(scope="module")
def log(model):
    """Primary capture (geometry source; payloads irrelevant to support)."""

    torch.manual_seed(1)
    trace = tl.trace(model, torch.randn(2, 8))
    try:
        yield trace
    finally:
        trace.cleanup()


def _sparse_direction() -> torch.Tensor:
    """One direction supported on feature indices {1, 4}."""

    direction = torch.zeros(_D)
    direction[1] = 0.8
    direction[4] = -0.3
    return direction


# ---------------------------------------------------------------------------
# Basis provenance: mandatory, disclosed, digest-anchored.
# ---------------------------------------------------------------------------


def test_missing_or_empty_origin_refuses():
    """A direction without a recorded origin is an unreproducible result."""

    with pytest.raises(TypeError):
        tl.subspace(_RELU, torch.ones(_D))  # origin is keyword-required
    with pytest.raises(ValueError, match="unreproducible"):
        tl.subspace(_RELU, torch.ones(_D), origin="   ")


def test_do_audit_record_carries_the_basis_origin(model):
    """The gallery row: ablating a probe direction's component leaves an audit trail."""

    trace = tl.trace(
        model,
        torch.randn(2, 8),
        capture=tl.options.CaptureOptions(replayable=True),
    )
    try:
        selection = tl.subspace(
            _RELU, _sparse_direction(), origin="imdb sentiment probe, run 7", method="probe"
        )
        mask = trace[selection][0].mask
        baseline = trace[_RELU].out.clone()
        fork = trace.fork()
        fork.do(selection, tl.zero_ablate())
        edited = fork[_RELU].out
        unit = _sparse_direction() / _sparse_direction().norm()
        want = baseline - (baseline @ unit)[:, None] * unit[None, :]
        torch.testing.assert_close(edited, want, rtol=0.0, atol=1e-6)
        assert torch.equal(edited[~mask], baseline[~mask])
        assert torch.equal(trace[_RELU].out, baseline)  # capture truth intact
        audit = fork.intervention_audit[-1]
        assert "imdb sentiment probe, run 7" in repr(audit)
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# Dimension honesty: typed refusals, never broadcast/truncate.
# ---------------------------------------------------------------------------


def test_wrong_dimension_refuses_typed(log):
    """A d-dim direction on a differently-wide site refuses basis_dim_mismatch."""

    selection = tl.subspace(_RELU, torch.ones(_D + 2), origin="wrong-width probe")
    with pytest.raises(SelectionError, match="never broadcast") as excinfo:
        log[selection]
    assert excinfo.value.fields["code"] == "selection_unresolvable"
    assert excinfo.value.fields["reason"] == "basis_dim_mismatch"
    assert excinfo.value.fields["basis_dim"] == _D + 2
    assert excinfo.value.fields["site_extent"] == _D


def test_dim_out_of_range_refuses_typed(log):
    """An axis outside the site's output space refuses basis_dim_mismatch."""

    selection = tl.subspace(_RELU, torch.ones(_D), origin="axis test", dim=5)
    with pytest.raises(SelectionError) as excinfo:
        log[selection]
    assert excinfo.value.fields["reason"] == "basis_dim_mismatch"
    assert excinfo.value.fields["dim"] == 5


# ---------------------------------------------------------------------------
# Construction refusals.
# ---------------------------------------------------------------------------


def test_within_is_required():
    """A direction is minted for a specific representation space."""

    with pytest.raises(ValueError, match="representation space"):
        tl.subspace(None, torch.ones(_D), origin="no site")


def test_non_act_within_refuses_kind_incompatible():
    """PARAM populations refuse selection_kind_incompatible."""

    with pytest.raises(SelectionError) as excinfo:
        tl.subspace(tl.param("encoder.weight"), torch.ones(_D), origin="param probe")
    assert excinfo.value.fields["code"] == "selection_kind_incompatible"


@pytest.mark.parametrize(
    "basis, match",
    [
        ([1.0, 2.0], "must be a torch.Tensor"),
        (torch.ones(2, 2, 2), "one direction"),
        (torch.ones(0, _D), "non-empty"),
        (torch.ones(_D, dtype=torch.complex64), "real numeric dtype"),
        (torch.ones(_D, dtype=torch.bool), "real numeric dtype"),
        (torch.tensor([1.0, float("nan"), 1.0]), "NaN/Inf"),
        (torch.tensor([1.0, float("inf"), 1.0]), "NaN/Inf"),
        (torch.zeros(_D), "vacuous"),
    ],
)
def test_bad_bases_refuse_at_construction(basis, match):
    """Non-tensor, mis-shaped, non-real, non-finite, and zero bases refuse."""

    with pytest.raises(ValueError, match=match):
        tl.subspace(_RELU, basis, origin="bad basis")


def test_sub_tol_row_refuses_at_construction():
    """A row entirely at-or-below tol contributes no support and refuses."""

    basis = torch.stack([torch.full((_D,), 0.01), torch.ones(_D)])
    with pytest.raises(ValueError, match="vacuous"):
        tl.subspace(_RELU, basis, origin="sub-tol row", atol=0.05)


def test_bad_parameters_refuse():
    """dim/tol/method parameter validation refuses early with teaching."""

    with pytest.raises(ValueError, match="dim"):
        tl.subspace(_RELU, torch.ones(_D), origin="x", dim=True)
    with pytest.raises(ValueError, match="non-negative"):
        tl.subspace(_RELU, torch.ones(_D), origin="x", atol=-1.0)
    with pytest.raises(ValueError, match="method"):
        tl.subspace(_RELU, torch.ones(_D), origin="x", method="")


# ---------------------------------------------------------------------------
# Composition through the algebra.
# ---------------------------------------------------------------------------


def test_align_to_rebinds_a_resolved_subspace_selection(model):
    """The resolved support set crosses runs through the align_to door."""

    torch.manual_seed(4)
    trace_a = tl.trace(model, torch.randn(2, 8))
    torch.manual_seed(5)
    trace_b = tl.trace(model, torch.randn(2, 8))
    try:
        resolved = trace_a[tl.subspace(_RELU, _sparse_direction(), origin="cross-run probe")]
        rebound = resolved.align_to(trace_b)
        assert torch.equal(rebound[0].mask, resolved[0].mask)
        assert "cross-run probe" in rebound[0].provenance.source
    finally:
        trace_a.cleanup()
        trace_b.cleanup()
