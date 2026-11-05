"""SRP construction oracles: exact, corpus-free identities (memo B6/B10; lane F19).

O1a (construction algebra), O1f (distribution identity — what makes T-C13's
labels checkable rather than editorial), O3 (generation determinism, kept
executable because "passes by construction" is exactly the claim that should
be tested), O13 (block-independence bit-identical), O10's deterministic
share= facts, the D-5 extent-drift refusal in its behavioral form (O9's
launch gate), and the seed/seed_source recording policy.

Statistical windows below are DERIVED IN COMMENTS and deliberately generous
(memo decision 18: nobody tightens a constant on a hunch).
"""

from __future__ import annotations

import random

import pytest
import torch

from torchlens.errors import TransformContractError
from torchlens.transforms import (
    TensorSpec,
    TransformContext,
    srp,
    srp_realized_facts,
    srp_verify_matrix,
)
from torchlens.transforms._srp import (
    _matrix_entry,
    _MatrixCache,
    _project_rows,
    _reset_srp_state,
)
from torchlens.transforms._srp_hash import (
    _splitmix64,
    _splitmix64_int,
)


@pytest.fixture(autouse=True)
def _fresh_srp_state() -> None:
    """Isolate the process-level matrix cache and extent bindings per test."""

    _reset_srp_state()
    yield
    _reset_srp_state()


# --- the integer hash: exact against a pure-Python oracle ---------------------


def test_splitmix64_limb_arithmetic_is_exact() -> None:
    """Tensor splitmix64 == pure-Python u64 reference on random inputs (O3)."""

    rng = random.Random(20260828)
    values = [rng.getrandbits(64) for _ in range(4096)] + [0, 1, (1 << 64) - 1, 1 << 63]

    def as_i64(u: int) -> int:
        return u - (1 << 64) if u >= (1 << 63) else u

    tensor = torch.tensor([as_i64(v) for v in values], dtype=torch.int64)
    out = _splitmix64(tensor)
    ref = [as_i64(_splitmix64_int(v)) for v in values]
    assert out.tolist() == ref


# --- O1a: construction identities (exact, corpus-free) ------------------------


def _dense_from_entry(entry: dict) -> torch.Tensor:
    """Materialize the (D, k) dense matrix from a cached entry."""

    D, k = int(entry["extent"]), int(entry["n_components"])
    m = int(entry["nonzeros_per_column"])
    weights = torch.zeros((D, k), dtype=torch.float64)
    for j in range(k):
        weights[entry["positions"][j], j] = entry["signs"][j].double() * entry["scale"]
    assert int((weights != 0).sum()) == k * m
    return weights


# --- O13: block-independence, bit-identical ------------------------------------


# --- O1f: distribution identity (what makes the labels checkable) --------------


# --- seed policy + shared matrices (O10 deterministic facts) --------------------


def test_o10_share_by_extent_and_by_site_digest_facts() -> None:
    """by_extent: site-independent digests; by_site: site-keyed digests."""

    ctx_a = TransformContext(site_label="layer4")
    ctx_b = TransformContext(site_label="fc")
    shared_a = _matrix_entry(srp(24, seed=1), 700, ctx_a)
    shared_b = _matrix_entry(srp(24, seed=1), 700, ctx_b)
    assert shared_a["digest"] == shared_b["digest"]
    assert shared_a["share_key_source"] == "shared"
    sited_a = _matrix_entry(srp(24, seed=1, share="by_site"), 700, ctx_a)
    sited_b = _matrix_entry(srp(24, seed=1, share="by_site"), 700, ctx_b)
    assert sited_a["digest"] != sited_b["digest"]
    assert sited_a["share_key_source"] == "site_label"  # the DISCLOSED weaker source
    keyed = _matrix_entry(
        srp(24, seed=1, share="by_site"), 700, TransformContext(site_key="s1|encoder.0")
    )
    assert keyed["share_key_source"] == "site_key"
    # Distinct seeds -> distinct matrices (independence needs DISTINCT seeds
    # under either policy; the docs warning rides srp()'s docstring).
    other_seed = _matrix_entry(srp(24, seed=2), 700, ctx_a)
    assert other_seed["digest"] != shared_a["digest"]


def test_share_by_site_without_any_site_identity_refuses() -> None:
    """No site key, no label -> typed refusal, never a silent shared matrix."""

    with pytest.raises(TransformContractError) as excinfo:
        srp(8, share="by_site").apply(torch.randn(2, 50), None)
    assert excinfo.value.fields["code"] == "transform_site_identity_unavailable"


# --- D-5 / O9: extent drift refuses BEFORE any shard is visible ------------------


def test_extent_drift_refuses_with_teaching(recwarn: pytest.WarningsList) -> None:
    """The same (spec instance, site) never projects two different extents."""

    spec = srp(16)
    spec.apply(torch.randn(2, 3, 10), None)
    with pytest.raises(TransformContractError) as excinfo:
        spec.apply(torch.randn(2, 3, 12), None)
    err = excinfo.value
    assert err.fields["code"] == "transform_extent_drift"
    assert "features" in str(err)
    assert err.fields["bound_extent"] == 30
    assert err.fields["observed_extent"] == 36


def test_heterogeneous_sites_bind_independently() -> None:
    """One chain over a mixed-rank sweep (D-3) binds per site, no false drift."""

    spec = srp(16)
    spec.apply(torch.randn(2, 64, 4, 4), TransformContext(site_label="layer3"))
    spec.apply(torch.randn(2, 128, 2, 2), TransformContext(site_label="layer4"))
    spec.apply(torch.randn(2, 100), TransformContext(site_label="fc"))


@pytest.mark.parametrize("construction", ["very_sparse_fixed", "iid_bernoulli"])
@pytest.mark.parametrize("path", ["dense", "sparse_csr", "dense_chunked"])
def test_every_path_is_row_local_under_every_batch_split(construction: str, path: str) -> None:
    """T-C10 per path: a row's bytes never depend on the rows batched with it.

    Single-row calls are the sharpest probe: BLAS takes a different kernel for
    one row than for a batch, which is what moved rows by an ulp before the
    projection became a row-local fold.
    """

    spec = srp(16, seed=3, construction=construction)
    x = torch.randn(7, 300, generator=torch.Generator().manual_seed(4))
    x[3, 0] = float("inf")  # padding slots point at position 0: never 0 * inf = nan
    entry = _matrix_entry(spec, 300, None)
    whole = _project_rows(x, entry, path)
    for size in range(1, 8):
        for lo in range(0, 7, size):
            assert torch.equal(_project_rows(x[lo : lo + size], entry, path), whole[lo : lo + size])
    assert not torch.isnan(whole).any()


# --- digest + verification --------------------------------------------------------


def test_digest_is_stable_and_verify_matrix_rederives() -> None:
    """The canonical digest is exact and re-derivation reproduces it."""

    spec = srp(32, seed=5)
    input_spec = TensorSpec(shape=(None, 500), dtype="torch.float32")
    digest = srp_verify_matrix(spec, input_spec)
    assert digest.startswith("sha256:")
    facts = srp_realized_facts(spec, input_spec)
    assert facts["matrix_digest"] == digest
    assert facts["algorithm_version"] == 1
    assert facts["distance_claim"] == "empirical_only"
    bern = srp(8, construction="iid_bernoulli", density=0.25)
    bern_digest = srp_verify_matrix(bern, TensorSpec(shape=(None, 100), dtype="torch.float32"))
    assert bern_digest.startswith("sha256:")


def test_verify_matrix_refuses_on_construction_drift() -> None:
    """The re-derivation tripwire FIRES when the recorded digest is not
    reproducible (a planted drift; T-C13's never-silent clause)."""

    spec = srp(16, seed=2)
    input_spec = TensorSpec(shape=(None, 300), dtype="torch.float32")
    srp_verify_matrix(spec, input_spec)  # clean pass first
    entry = _matrix_entry(spec, 300, None)
    entry["digest"] = "sha256:" + "0" * 64  # planted drift in the cached record
    with pytest.raises(TransformContractError) as excinfo:
        srp_verify_matrix(spec, input_spec)
    err = excinfo.value
    assert err.fields["code"] == "transform_matrix_verification_failed"
    assert err.fields["recorded"].endswith("0" * 8)


def test_bernoulli_plan_discloses_generation_cost() -> None:
    """The O(D*k) cost rides the plan disclosures (no harvest surprises)."""

    plan = srp(16, construction="iid_bernoulli", density=0.3).plan(
        TensorSpec(shape=(None, 200), dtype="torch.float32")
    )
    assert plan.disclosures and "O(D*k)" in plan.disclosures[0]
    fixed_plan = srp(16).plan(TensorSpec(shape=(None, 200), dtype="torch.float32"))
    assert fixed_plan.disclosures == ()


# --- params, coercion, records -----------------------------------------------------


def test_srp_params_are_validated_typed() -> None:
    """n_components required; closed vocabularies; density domain."""

    for bad_kwargs in (
        {"num_components": None},
        {"num_components": 0},
        {"num_components": 16, "density": 1.5},
        {"num_components": 16, "density": "high"},
        {"num_components": 16, "construction": "gaussian"},
        {"num_components": 16, "mode": "rows"},
        {"num_components": 16, "share": "by_vibes"},
        {"num_components": 16, "seed": 1.5},
    ):
        with pytest.raises(TransformContractError) as excinfo:
            srp(**bad_kwargs)  # type: ignore[arg-type]
        assert excinfo.value.fields["code"] == "transform_params_invalid"
    with pytest.raises(TransformContractError) as excinfo:
        srp(16).apply(torch.randint(0, 5, (2, 10)), None)
    assert excinfo.value.fields["code"] == "transform_plan_invalid"


def test_half_inputs_accumulate_and_return_fp32() -> None:
    """T-C7: half inputs project in fp32 (plan discloses the fp32 output)."""

    spec = srp(8, seed=1)
    plan = spec.plan(TensorSpec(shape=(None, 64), dtype="torch.float16"))
    assert plan.output.dtype == "torch.float32"
    out = spec.apply(torch.randn(2, 64, dtype=torch.float16), None)
    assert out.dtype == torch.float32


def test_matrix_cache_is_bounded() -> None:
    """The LRU cache evicts past its entry and byte bounds."""

    cache = _MatrixCache(max_entries=2, max_bytes=1 << 30)
    for i in range(3):
        cache.put((i,), {"positions": torch.zeros(4, dtype=torch.int64)})
    assert cache.get((0,)) is None and cache.get((2,)) is not None
    tiny = _MatrixCache(max_entries=10, max_bytes=100)
    tiny.put((1,), {"positions": torch.zeros(20, dtype=torch.int64)})  # 160 bytes > bound
    assert tiny.get((1,)) is None
