"""Helpers + fitted-projection contract tests (memo B7/P6/B8; lane F19).

Covers ``srp_dims_for`` (the honestly-labeled dense-JL heuristic),
``srp_fidelity_probe`` (the T-C12 provenance-bearing report), the P6
``tl.stats.PCA`` additive upgrade (mean/n_samples/n_features/digest,
deterministic sign canonicalization, the digest-checked persisted sidecar),
``project``'s digest-addressed staging, and ``pca_apply`` with the O11
centering asymmetry pinned (Euclidean RDMs translation-invariant; corr-RDMs
moved).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torchlens.stats import PCA, FittedArtifactError, FittedPCA, load_fitted, save_fitted
from torchlens.transforms import (
    TensorSpec,
    TransformContractError,
    chain,
    pca_apply,
    pipeline_from_record,
    pipeline_record,
    project,
    srp,
    srp_dims_for,
    srp_fidelity_probe,
)
from torchlens.transforms._helpers import _condensed_corr_rdm, _condensed_l2_rdm, spearman
from torchlens.transforms._projection import _BASIS_STORE, _reset_projection_store

pytestmark = pytest.mark.smoke


@pytest.fixture(autouse=True)
def _fresh_store() -> None:
    """Isolate the digest-addressed basis store per test."""

    _reset_projection_store()
    yield
    _reset_projection_store()


def _fit(
    rows: int = 60, width: int = 20, k: int = 5, seed: int = 3
) -> tuple[FittedPCA, torch.Tensor]:
    """Fit a small PCA and return (fitted payload, the fitting data)."""

    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, width, generator=generator)
    estimator = PCA(k)
    estimator.update(x[: rows // 2])
    estimator.update(x[rows // 2 :])
    return estimator.fitted(fit_scope="test fit"), x


# --- srp_dims_for -------------------------------------------------------------


def test_srp_dims_for_matches_the_dense_jl_formula() -> None:
    """k = ceil(4 ln n / (eps^2/2 - eps^3/3)); the sklearn-practice value."""

    sizing = srp_dims_for(128, eps=0.2)
    assert sizing.n_components == 1120  # ceil(4*ln(128) / (0.02 - 0.008/3))
    assert int(sizing) == 1120
    assert sizing.formula == "dense_jl_l2_heuristic"
    assert "corr" in sizing.note  # it SAYS it is a poor corr-RDM guide
    record = sizing.to_record()
    assert record["eps"] == 0.2 and record["n_samples"] == 128
    # The sizing feeds srp() directly.
    assert srp(int(sizing)).params_dict()["n_components"] == 1120


def test_srp_dims_for_validates_inputs() -> None:
    """Row count and eps domains refuse typed."""

    for kwargs in ({"n_samples": 1}, {"n_samples": 10, "eps": 0.0}, {"n_samples": 10, "eps": 1.0}):
        with pytest.raises(TransformContractError) as excinfo:
            srp_dims_for(**kwargs)  # type: ignore[arg-type]
        assert excinfo.value.fields["code"] == "transform_params_invalid"


# --- spearman + probe ----------------------------------------------------------


def test_spearman_matches_hand_computed_ranks_with_ties() -> None:
    """Tie-averaged rank correlation against a hand-checked case."""

    a = torch.tensor([1.0, 2.0, 2.0, 3.0])
    b = torch.tensor([10.0, 20.0, 30.0, 40.0])
    # ranks(a) = [1, 2.5, 2.5, 4]; ranks(b) = [1, 2, 3, 4]
    # pearson: 4.5 / (sqrt(4.5) * sqrt(5)) = sqrt(0.9) = 0.948683298...
    assert spearman(a, b) == pytest.approx(0.9486832980, abs=1e-9)
    assert spearman(b, b) == pytest.approx(1.0)
    assert spearman(b, -b) == pytest.approx(-1.0)


def test_probe_reports_scoped_rows_and_is_deterministic() -> None:
    """The probe report carries T-C12 scope fields; same seed, same numbers."""

    generator = torch.Generator().manual_seed(0)
    pilot = torch.randn(12, 300, generator=generator)
    first = srp_fidelity_probe(pilot, n_components=(16, 64), seed=7)
    second = srp_fidelity_probe(pilot, n_components=(16, 64), seed=7)
    assert first.to_record() == second.to_record()
    assert first.corpus == "user_pilot_batch"
    assert first.distance_claim == "empirical_only"
    assert first.n_rows == 12 and first.input_extent == 300
    assert [row["n_components"] for row in first.rows] == [16, 64]
    for row in first.rows:
        assert -1.0 <= row["l2_rdm_spearman"] <= 1.0
        assert -1.0 <= row["corr_rdm_spearman"] <= 1.0
    text = str(first)
    assert "scoped to THIS batch" in text and "empirical_only" in text
    assert first.canonical_json().startswith('{"algorithm_version"')


def test_probe_refusals_are_typed() -> None:
    """Too-few rows, non-float input, empty widths, constant rows refuse."""

    with pytest.raises(TransformContractError) as excinfo:
        srp_fidelity_probe(torch.randn(3, 10), n_components=(4,))
    assert excinfo.value.fields["code"] == "transform_params_invalid"
    with pytest.raises(TransformContractError):
        srp_fidelity_probe(torch.randint(0, 5, (8, 10)), n_components=(4,))
    with pytest.raises(TransformContractError):
        srp_fidelity_probe(torch.randn(8, 10), n_components=())
    constant = torch.ones(6, 12)
    with pytest.raises(TransformContractError) as excinfo:
        srp_fidelity_probe(constant, n_components=(4,))
    assert "zero variance" in str(excinfo.value)


# --- P6: the tl.stats fitted upgrade ---------------------------------------------


def test_pca_result_carries_the_fitted_payload_facts() -> None:
    """result() gains mean/n_samples/n_features/digest; legacy keys intact."""

    fitted, x = _fit()
    estimator = PCA(5)
    estimator.update(x)
    result = estimator.result()
    assert set(result) == {
        "components",
        "explained_variance",
        "mean",
        "n_samples",
        "n_features",
        "digest",
    }
    assert result["n_samples"] == 60 and result["n_features"] == 20
    assert torch.allclose(result["mean"], x.double().mean(dim=0), atol=1e-9)
    assert result["digest"].startswith("sha256:")
    # The SAME estimator's result() and fitted() agree byte-for-byte (a
    # different batch split legitimately differs in accumulation ulps).
    same = estimator.fitted(fit_scope="same estimator")
    assert result["digest"] == same.digest
    assert torch.equal(result["components"], same.components)


def test_pca_sign_canonicalization_is_deterministic() -> None:
    """Each component's largest-magnitude entry is positive; fits reproduce."""

    fitted, x = _fit()
    for row in fitted.components:
        assert row[row.abs().argmax()] > 0
    again, _ = _fit()
    assert torch.equal(fitted.components, again.components)
    assert fitted.digest == again.digest


def test_pca_fitted_refuses_before_data() -> None:
    """fitted() before two rows refuses pca_fitted_unavailable."""

    with pytest.raises(FittedArtifactError) as excinfo:
        PCA(2).fitted()
    assert excinfo.value.fields["code"] == "pca_fitted_unavailable"


def test_fitted_sidecar_round_trips_and_digest_checks(tmp_path: Path) -> None:
    """save/load round trip; tampered arrays refuse pca_fitted_digest_mismatch."""

    fitted, _ = _fit()
    path = save_fitted(fitted, tmp_path / "pca.safetensors")
    loaded = load_fitted(path)
    assert loaded.digest == fitted.digest
    assert torch.equal(loaded.components, fitted.components)
    assert torch.equal(loaded.mean, fitted.mean)
    assert loaded.fit_scope == "test fit"
    assert loaded.n_samples == 60 and loaded.n_features == 20
    raw = bytearray(path.read_bytes())
    raw[-3] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(FittedArtifactError) as excinfo:
        load_fitted(path)
    assert excinfo.value.fields["code"] == "pca_fitted_digest_mismatch"


def test_fitted_sidecar_malformed_refusals(tmp_path: Path) -> None:
    """Non-safetensors and wrong-schema files refuse pca_fitted_record_invalid."""

    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a safetensors file")
    with pytest.raises(FittedArtifactError) as excinfo:
        load_fitted(junk)
    assert excinfo.value.fields["code"] == "pca_fitted_record_invalid"
    from safetensors.torch import save_file

    wrong = tmp_path / "wrong.safetensors"
    save_file({"something": torch.zeros(2)}, str(wrong), metadata={"schema": "other"})
    with pytest.raises(FittedArtifactError) as excinfo:
        load_fitted(wrong)
    assert excinfo.value.fields["code"] == "pca_fitted_record_invalid"
    with pytest.raises(FittedArtifactError) as excinfo:
        save_fitted("nope", tmp_path / "x.safetensors")  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "pca_fitted_record_invalid"


# --- B8: project + pca_apply -------------------------------------------------------


def test_pca_apply_matches_the_fp64_reference() -> None:
    """(x - mean) @ components.T in fp64, through the chain door."""

    fitted, x = _fit()
    step = pca_apply(fitted)
    out = step.apply(x[:8], None)
    reference = (x[:8].double() - fitted.mean) @ fitted.components.T
    assert torch.allclose(out.double(), reference, atol=1e-5)
    plan = step.plan(TensorSpec.of(x))
    assert plan.output.shape == (None, 5)
    params = step.params_dict()
    assert params["centered"] is True
    assert params["fit_scope"] == "test fit"
    assert params["source"] == fitted.digest


def test_o11_centering_scoped_euclidean_intact_corr_moved() -> None:
    """O11: uncentered apply leaves L2 RDMs exact; corr-RDMs move."""

    generator = torch.Generator().manual_seed(9)
    offset = 10.0 * torch.randn(16, generator=generator)  # large anisotropic mean
    x = torch.randn(40, 16, generator=generator) + offset
    estimator = PCA(8)
    estimator.update(x)
    fitted = estimator.fitted(fit_scope="off-center fixture")
    batch = x[:12].double()  # fp64 so the L2 invariance is exact
    centered = pca_apply(fitted).apply(batch, None).double()
    uncentered = project(fitted.components).apply(batch, None).double()
    l2_centered = _condensed_l2_rdm(centered)
    l2_uncentered = _condensed_l2_rdm(uncentered)
    assert float((l2_centered - l2_uncentered).abs().max()) < 1e-6
    corr_centered = _condensed_corr_rdm(centered, context="test")
    corr_uncentered = _condensed_corr_rdm(uncentered, context="test")
    # The dominant uncentered mean direction collapses row correlations:
    # the corr-RDM moves materially while the Euclidean RDM was untouched.
    assert spearman(corr_centered, corr_uncentered) < 0.9
    assert float((corr_centered - corr_uncentered).abs().max()) > 0.05


def test_project_extent_checks_are_exact() -> None:
    """Wrong feature width refuses at plan AND apply."""

    fitted, _ = _fit()
    step = pca_apply(fitted)
    with pytest.raises(TransformContractError) as excinfo:
        step.plan(TensorSpec(shape=(None, 21), dtype="torch.float32"))
    assert excinfo.value.fields["code"] == "transform_plan_invalid"
    with pytest.raises(TransformContractError) as excinfo:
        step.apply(torch.randn(4, 21), None)
    assert excinfo.value.fields["code"] == "transform_plan_invalid"


def test_project_params_and_payload_validation() -> None:
    """Factory refusals: bad basis/center; mutated fitted payloads refuse."""

    with pytest.raises(TransformContractError):
        project(torch.zeros(0, 3))
    with pytest.raises(TransformContractError):
        project("basis")  # type: ignore[arg-type]
    with pytest.raises(TransformContractError):
        project(torch.full((2, 3), float("nan")))
    with pytest.raises(TransformContractError):
        project(torch.randn(2, 3), center=torch.randn(4))
    with pytest.raises(TransformContractError) as excinfo:
        pca_apply("not fitted")  # type: ignore[arg-type]
    assert "REFUSED" in str(excinfo.value)  # fit-during-write teaching
    fitted, _ = _fit()
    tampered = FittedPCA(
        components=fitted.components + 1.0,
        mean=fitted.mean,
        explained_variance=fitted.explained_variance,
        n_samples=fitted.n_samples,
        n_features=fitted.n_features,
        fit_scope=fitted.fit_scope,
        digest=fitted.digest,
    )
    with pytest.raises(TransformContractError) as excinfo:
        pca_apply(tampered)
    assert "hash" in str(excinfo.value)


def test_rehydrated_project_refuses_until_restaged() -> None:
    """Digest addressing: records carry the ADDRESS; arrays restage explicitly."""

    fitted, x = _fit()
    step = pca_apply(fitted)
    record = pipeline_record(chain(step))
    assert record is not None
    _reset_projection_store()
    rebuilt = pipeline_from_record(record)
    with pytest.raises(TransformContractError) as excinfo:
        rebuilt.apply(x[:4], None)
    err = excinfo.value
    assert err.fields["code"] == "transform_basis_unavailable"
    assert "load_fitted" in str(err)
    pca_apply(fitted)  # restage
    out = rebuilt.apply(x[:4], None)
    reference = (x[:4].double() - fitted.mean) @ fitted.components.T
    assert torch.allclose(out.double(), reference, atol=1e-5)


def test_basis_store_is_digest_addressed_and_bounded() -> None:
    """Equal arrays share one staged entry; the store stays bounded."""

    basis = torch.randn(3, 7, generator=torch.Generator().manual_seed(1))
    a = project(basis)
    b = project(basis.clone())
    assert a.params_dict()["basis_digest"] == b.params_dict()["basis_digest"]
    assert len(_BASIS_STORE) == 1
    for i in range(20):
        project(torch.randn(2, 4, generator=torch.Generator().manual_seed(i)))
    assert len(_BASIS_STORE) <= 16


def test_pca_consumes_srp_reduced_features_composition_row() -> None:
    """Composition row 10.4: tl.stats.PCA consuming transformed features."""

    generator = torch.Generator().manual_seed(4)
    activations = torch.randn(30, 8, 8, generator=generator)
    reducer = srp(24, seed=1)
    reduced = reducer.apply(activations, None)
    estimator = PCA(4)
    estimator.update(reduced)
    fitted = estimator.fitted(fit_scope="fit on srp-reduced pilot")
    pipeline = chain(reducer, pca_apply(fitted))
    out = pipeline.apply(activations, None)
    assert out.shape == (30, 4)
    reference = (reduced.double() - fitted.mean) @ fitted.components.T
    assert torch.allclose(out.double(), reference, atol=1e-5)
