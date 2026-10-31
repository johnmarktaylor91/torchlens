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

import hashlib

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._selection import SelectionError
from torchlens._selection_subspace import BasisProvenance

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
# Support masks are exact as sets.
# ---------------------------------------------------------------------------


def test_sparse_direction_selects_exactly_its_support(log):
    """The mask is the support set, expanded across the batch axis."""

    selection = tl.subspace(_RELU, _sparse_direction(), origin="hand-specified test vector")
    resolved = selection.resolve(log)
    assert len(resolved) == 1
    entry = resolved[0]
    assert entry.shape == (2, _D)
    expected = torch.zeros(2, _D, dtype=torch.bool)
    expected[:, [1, 4]] = True
    assert torch.equal(entry.mask, expected)
    assert entry.provenance.relation == "exact"
    assert entry.num_selected == 4


def test_multi_vector_basis_unions_row_supports(log):
    """A [k, d] basis supports the union of its rows' supports."""

    basis = torch.zeros(2, _D)
    basis[0, 0] = 1.0
    basis[1, 5] = -2.0
    resolved = tl.subspace(_RELU, basis, origin="two test directions").resolve(log)
    expected = torch.zeros(2, _D, dtype=torch.bool)
    expected[:, [0, 5]] = True
    assert torch.equal(resolved[0].mask, expected)


def test_dense_direction_supports_the_whole_axis(log):
    """A dense direction resolves to the whole site — the documented set claim."""

    resolved = tl.subspace(_RELU, torch.ones(_D), origin="dense test direction").resolve(log)
    assert resolved[0].num_selected == 2 * _D
    assert bool(resolved[0].mask.all())


def test_tol_is_a_strict_support_threshold(log):
    """Weights with |w| <= tol contribute no support; |w| > tol do."""

    direction = torch.tensor([0.25, 0.5, -0.25, -0.5, 0.25, 0.5])
    resolved = tl.subspace(_RELU, direction, origin="tol test", atol=0.25).resolve(log)
    expected = torch.zeros(2, _D, dtype=torch.bool)
    expected[:, [1, 3, 5]] = True
    assert torch.equal(resolved[0].mask, expected)


def test_resolution_is_geometry_only_unsaved_sites_resolve(model):
    """Support reads no payloads: a site excluded from save= still resolves."""

    trace = tl.trace(model, torch.randn(2, 8), save=tl.func("sigmoid"))
    try:
        assert not trace[_RELU].has_saved_activation
        resolved = tl.subspace(_RELU, _sparse_direction(), origin="unsaved-site test").resolve(
            trace
        )
        assert resolved[0].num_selected == 4
    finally:
        trace.cleanup()


def test_explicit_dim_binds_a_non_trailing_axis():
    """dim= names the bound axis: a channel direction on a conv output."""

    class _Cnn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 4, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.conv(x))

    torch.manual_seed(2)
    trace = tl.trace(_Cnn(), torch.randn(1, 1, 8, 8))
    try:
        channel_direction = torch.tensor([0.0, 1.0, 0.0, 1.0])
        resolved = tl.subspace(
            "relu_1_2", channel_direction, origin="channel direction test", dim=1
        ).resolve(trace)
        mask = resolved[0].mask
        assert mask[:, [1, 3]].all()
        assert not mask[:, [0, 2]].any()
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# Basis provenance: mandatory, disclosed, digest-anchored.
# ---------------------------------------------------------------------------


def test_provenance_source_carries_origin_method_geometry_and_digest(log):
    """The resolved source string is the full reproducibility disclosure."""

    basis = _sparse_direction()
    resolved = tl.subspace(
        _RELU, basis, origin="linear probe, run 42", method="probe", atol=0.1
    ).resolve(log)
    source = resolved[0].provenance.source
    assert "origin='linear probe, run 42'" in source
    assert "method='probe'" in source
    assert f"k=1, d={_D}, dim=-1, tol=0.1" in source
    canonical = basis.to(torch.float64).contiguous()
    digest = hashlib.sha256()
    digest.update(repr((1, _D)).encode())
    digest.update(canonical.unsqueeze(0).numpy().tobytes())
    assert f"basis_sha256={digest.hexdigest()}" in source


def test_missing_or_empty_origin_refuses():
    """A direction without a recorded origin is an unreproducible result."""

    with pytest.raises(TypeError):
        tl.subspace(_RELU, torch.ones(_D))  # origin is keyword-required
    with pytest.raises(ValueError, match="unreproducible"):
        tl.subspace(_RELU, torch.ones(_D), origin="   ")


def test_provenance_record_fields_and_repr(log):
    """The frozen BasisProvenance record is the programmatic face."""

    selection = tl.subspace(_RELU, _sparse_direction(), origin="repr test", method="manual")
    record = selection._node.provenance
    assert isinstance(record, BasisProvenance)
    assert (record.num_vectors, record.space_dim, record.dim, record.tol) == (1, _D, -1, 0.0)
    assert record.origin == "repr test"
    assert len(record.sha256) == 64
    assert "origin='repr test'" in repr(selection)
    assert record.sha256 in repr(selection)


def test_digest_identifies_basis_content(log):
    """Different bases mint different digests; identical bases the same one."""

    a = tl.subspace(_RELU, _sparse_direction(), origin="a")._node.provenance.sha256
    b = tl.subspace(_RELU, _sparse_direction(), origin="b")._node.provenance.sha256
    c = tl.subspace(_RELU, torch.ones(_D), origin="c")._node.provenance.sha256
    assert a == b
    assert a != c


def test_do_audit_record_carries_the_basis_origin(model):
    """The gallery row: ablating a probe direction's component leaves an audit trail."""

    trace = tl.trace(
        model,
        torch.randn(2, 8),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    try:
        selection = tl.subspace(
            _RELU, _sparse_direction(), origin="imdb sentiment probe, run 7", method="probe"
        )
        mask = selection.resolve(trace)[0].mask
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
        selection.resolve(log)
    assert excinfo.value.fields["code"] == "selection_unresolvable"
    assert excinfo.value.fields["reason"] == "basis_dim_mismatch"
    assert excinfo.value.fields["basis_dim"] == _D + 2
    assert excinfo.value.fields["site_extent"] == _D


def test_population_with_one_mismatched_site_refuses(log):
    """Structures never silently intersect: one bad site fails the whole resolve."""

    within = tl.units(_RELU, torch.ones(2, _D, dtype=torch.bool)) | tl.units(
        _SIGMOID, torch.ones(2, 3, dtype=torch.bool)
    )
    selection = tl.subspace(within, torch.ones(_D), origin="mixed-population probe")
    with pytest.raises(SelectionError) as excinfo:
        selection.resolve(log)
    assert excinfo.value.fields["reason"] == "basis_dim_mismatch"


def test_dim_out_of_range_refuses_typed(log):
    """An axis outside the site's output space refuses basis_dim_mismatch."""

    selection = tl.subspace(_RELU, torch.ones(_D), origin="axis test", dim=5)
    with pytest.raises(SelectionError) as excinfo:
        selection.resolve(log)
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


def test_basis_is_canonicalized_defensively(log):
    """Mutating the caller's basis after construction cannot alter the selection."""

    basis = _sparse_direction()
    selection = tl.subspace(_RELU, basis, origin="mutation test")
    basis[0] = 99.0
    resolved = selection.resolve(log)
    assert resolved[0].num_selected == 4  # still {1, 4} x batch 2


# ---------------------------------------------------------------------------
# Composition through the algebra.
# ---------------------------------------------------------------------------


def test_composes_with_value_producers(log):
    """subspace & top_k intersects masks; relations join through the table."""

    support = tl.subspace(_RELU, _sparse_direction(), origin="compose test")
    combined = (support & tl.top_k(_RELU, _D * 2)).resolve(log)
    top = tl.top_k(_RELU, _D * 2).resolve(log)[0].mask
    expected = torch.zeros(2, _D, dtype=torch.bool)
    expected[:, [1, 4]] = True
    assert torch.equal(combined[0].mask, expected & top)
    assert combined[0].provenance.relation == "exact"


def test_complement_is_the_off_support_set(log):
    """~subspace selects the off-support elements of the touched site."""

    resolved = (~tl.subspace(_RELU, _sparse_direction(), origin="complement test")).resolve(log)
    expected = torch.ones(2, _D, dtype=torch.bool)
    expected[:, [1, 4]] = False
    assert torch.equal(resolved[0].mask, expected)


def test_bare_layer_label_on_multipass_layer_supports_every_pass():
    """A bare recurrent-layer label is the all-passes spelling; each pass is checked."""

    class _Recurrent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cell = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for _ in range(3):
                x = torch.tanh(self.cell(x))
            return x

    torch.manual_seed(3)
    trace = tl.trace(_Recurrent(), torch.randn(1, 4))
    try:
        direction = torch.tensor([1.0, 0.0, 0.0, 1.0])
        resolved = tl.subspace("tanh_1_2", direction, origin="recurrent probe").resolve(trace)
        assert len(resolved) == 3  # one entry per pass
        assert {entry.key for entry in resolved} == {
            ("tanh_1_2", 1),
            ("tanh_1_2", 2),
            ("tanh_1_2", 3),
        }
        for entry in resolved:
            expected = torch.zeros(1, 4, dtype=torch.bool)
            expected[:, [0, 3]] = True
            assert torch.equal(entry.mask, expected)
        # The dim check runs per pass: a wrong-width basis refuses on the same spelling.
        with pytest.raises(SelectionError) as excinfo:
            tl.subspace("tanh_1_2", torch.ones(5), origin="wrong width").resolve(trace)
        assert excinfo.value.fields["reason"] == "basis_dim_mismatch"
    finally:
        trace.cleanup()


def test_align_to_rebinds_a_resolved_subspace_selection(model):
    """The resolved support set crosses runs through the align_to door."""

    torch.manual_seed(4)
    trace_a = tl.trace(model, torch.randn(2, 8))
    torch.manual_seed(5)
    trace_b = tl.trace(model, torch.randn(2, 8))
    try:
        resolved = tl.subspace(_RELU, _sparse_direction(), origin="cross-run probe").resolve(
            trace_a
        )
        rebound = resolved.align_to(trace_b)
        assert torch.equal(rebound[0].mask, resolved[0].mask)
        assert "cross-run probe" in rebound[0].provenance.source
    finally:
        trace_a.cleanup()
        trace_b.cleanup()


def test_within_selection_restricts_the_population(log):
    """An element-masked within intersects with the support (JOIN composition)."""

    row_mask = torch.zeros(2, _D, dtype=torch.bool)
    row_mask[0] = True  # batch row 0 only
    resolved = tl.subspace(
        tl.units(_RELU, row_mask), _sparse_direction(), origin="restricted population"
    ).resolve(log)
    expected = torch.zeros(2, _D, dtype=torch.bool)
    expected[0, [1, 4]] = True
    assert torch.equal(resolved[0].mask, expected)
