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

import math
import random

import pytest
import torch

from torchlens.transforms import (
    TensorSpec,
    TransformContext,
    TransformContractError,
    chain,
    coerce_transform,
    pipeline_from_record,
    pipeline_record,
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
    generate_bernoulli_columns,
    generate_fixed_columns,
)

pytestmark = pytest.mark.smoke


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


def test_o1a_construction_identities() -> None:
    """Exact per-column counts, value multiset, column norms, nnz, signs.

    The value-multiset assertion is the one that FAILS on the retracted
    draw-and-coalesce construction (colliding entries summed to 0 or
    +/- 2*scale).
    """

    spec = srp(48, seed=3, density=0.05)
    entry = _matrix_entry(spec, 1200, None)
    D, k, m = 1200, 48, round(0.05 * 1200)
    assert entry["nonzeros_per_column"] == m
    assert entry["realized_nnz"] == k * m
    scale = entry["scale"]
    assert scale == pytest.approx(1.0 / math.sqrt((m / D) * k))
    positions, signs = entry["positions"], entry["signs"]
    # Exactly m DISTINCT nonzeros per output column, ascending order.
    for j in range(k):
        column = positions[j].tolist()
        assert len(set(column)) == m
        assert column == sorted(column)
        assert all(0 <= p < D for p in column)
    # Value multiset is exactly {+scale, -scale}: signs are strictly +/-1.
    assert set(signs.reshape(-1).tolist()) <= {-1, 1}
    dense = _dense_from_entry(entry)
    values = dense[dense != 0]
    assert torch.equal(values.abs(), torch.full_like(values, scale))
    # Realized column-norm^2 == m * scale^2 to float rounding.
    norms_sq = (dense**2).sum(dim=0)
    expected = m * scale * scale
    assert float((norms_sq - expected).abs().max()) < 1e-9
    # Sign balance within 4 standard deviations: sum of nnz iid +/-1 signs
    # has sd sqrt(nnz); the memo's measured ratios (1.7/2.9/2.0/0.4 sd) sit
    # comfortably inside, and nobody tightens this constant on a hunch.
    nnz = k * m
    assert abs(int(signs.to(torch.int64).sum())) <= 4 * math.sqrt(nnz)


def test_o1a_sparse_path_matches_dense_matmul() -> None:
    """Sparse and chunked paths reproduce the materialized dense matmul.

    dense_chunked is BIT-IDENTICAL in generation and exact here; the sparse
    kernel's accumulation order differs, so its parity is tolerance-based
    with the realized kernel recorded (memo section 6, verification).
    """

    spec = srp(32, seed=5)
    x = torch.randn(4, 500, generator=torch.Generator().manual_seed(1))
    entry = _matrix_entry(spec, 500, None)
    dense = _project_rows(x, entry, "dense")
    chunked = _project_rows(x, entry, "dense_chunked")
    sparse = _project_rows(x, entry, "sparse_csr")
    assert torch.allclose(dense, chunked, atol=0.0, rtol=0.0) or torch.equal(dense, chunked)
    assert torch.allclose(dense, sparse, atol=1e-5, rtol=1e-5)


# --- O13: block-independence, bit-identical ------------------------------------


def test_o13_generation_is_block_independent() -> None:
    """One block vs many chunks: bit-identical positions and signs."""

    seed, D, k, m = 99, 2048, 64, 45
    whole_pos, whole_sig = generate_fixed_columns(seed, D, range(k), m)
    for offset, width in ((0, 16), (16, 16), (32, 32)):
        part_pos, part_sig = generate_fixed_columns(seed, D, range(offset, offset + width), m)
        assert torch.equal(whole_pos[offset : offset + width], part_pos)
        assert torch.equal(whole_sig[offset : offset + width], part_sig)
    cid, pos, sig = generate_bernoulli_columns(seed, 512, range(32), 0.1)
    cid2, pos2, sig2 = generate_bernoulli_columns(seed, 512, range(16, 32), 0.1)
    keep = cid >= 16
    assert torch.equal(cid[keep], cid2)
    assert torch.equal(pos[keep], pos2)
    assert torch.equal(sig[keep], sig2)


def test_o3_generation_determinism_on_available_devices() -> None:
    """Same inputs -> bit-identical matrices, per available device.

    On a CPU-only box this pins CPU determinism; the CUDA/MPS legs run in
    the machine-gated GPU suite (tests/test_transforms_lib_gpu_gate.py).
    """

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():  # pragma: no cover - GPU boxes only
        devices.append(torch.device("cuda"))
    reference_pos, reference_sig = generate_fixed_columns(7, 1024, range(32), 30)
    for device in devices:
        pos, sig = generate_fixed_columns(7, 1024, range(32), 30, device=device)
        assert torch.equal(pos.cpu(), reference_pos)
        assert torch.equal(sig.cpu(), reference_sig)


# --- O1f: distribution identity (what makes the labels checkable) --------------


def test_o1f_distribution_identity() -> None:
    """vsf counts are exactly m (variance 0); iid counts match the Binomial.

    Windows (derived, generous): per-column count ~ Bin(D=2000, p=0.1) has
    mean 200, var 180 (sd 13.4). Over k=200 columns the sample-mean se is
    13.4/sqrt(200) ~ 0.95 -> window +/- 5 se ~ 4.8. The sample-variance se
    is roughly var * sqrt(2/(k-1)) ~ 18 -> window +/- 5 se = 90.
    """

    fixed = _matrix_entry(srp(64, seed=11, density=0.05), 1500, None)
    counts = torch.tensor(
        [len(set(fixed["positions"][j].tolist())) for j in range(64)], dtype=torch.float64
    )
    assert float(counts.var()) == 0.0
    assert int(counts[0]) == fixed["nonzeros_per_column"]

    D, k, p = 2000, 200, 0.1
    cid, _pos, _sig = generate_bernoulli_columns(31, D, range(k), p)
    per_column = torch.bincount(cid, minlength=k).double()
    mean, var = float(per_column.mean()), float(per_column.var())
    assert abs(mean - D * p) < 4.8
    assert abs(var - D * p * (1 - p)) < 90.0


# --- seed policy + shared matrices (O10 deterministic facts) --------------------


def test_seed_default_and_source_are_recorded() -> None:
    """seed=0 + seed_source recorded; explicit 0 records 'explicit' (T-C4)."""

    default = srp(16)
    assert default.seed == 0 and default.seed_source == "library_default"
    explicit = srp(16, seed=0)
    assert explicit.seed == 0 and explicit.seed_source == "explicit"
    record = default.canonical_record()
    assert record["seed"] == 0 and record["seed_source"] == "library_default"
    # The two spellings produce the SAME matrix (the recorded source is an
    # audit fact, not a numeric input) but DIFFERENT canonical chains.
    assert default.canonical_json() != explicit.canonical_json()
    same_a = _matrix_entry(default, 300, None)
    same_b = _matrix_entry(explicit, 300, None)
    assert same_a["digest"] == same_b["digest"]


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


def test_fresh_spec_instances_bind_fresh_extents() -> None:
    """A new run (new spec object) legitimately binds a new extent."""

    srp(16).apply(torch.randn(2, 10), None)
    srp(16).apply(torch.randn(2, 20), None)


def test_batch_composition_independence_of_srp_rows() -> None:
    """T-C10 for SRP itself: same extent, different batch splits, same rows."""

    spec = srp(32, seed=9)
    x = torch.randn(6, 40, generator=torch.Generator().manual_seed(2))
    whole = spec.apply(x, None)
    parts = torch.cat([spec.apply(x[:2], None), spec.apply(x[2:], None)])
    assert torch.equal(whole, parts)


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
        {"n_components": None},
        {"n_components": 0},
        {"n_components": 16, "density": 1.5},
        {"n_components": 16, "density": "high"},
        {"n_components": 16, "construction": "gaussian"},
        {"n_components": 16, "mode": "rows"},
        {"n_components": 16, "share": "by_vibes"},
        {"n_components": 16, "seed": 1.5},
    ):
        with pytest.raises(TransformContractError) as excinfo:
            srp(**bad_kwargs)  # type: ignore[arg-type]
        assert excinfo.value.fields["code"] == "transform_params_invalid"
    with pytest.raises(TransformContractError) as excinfo:
        srp(16).apply(torch.randint(0, 5, (2, 10)), None)
    assert excinfo.value.fields["code"] == "transform_plan_invalid"


def test_srp_rides_chains_and_pipeline_records() -> None:
    """Chain coercion preserves seed facts; records rehydrate the same chain."""

    pipeline = chain(srp(16, seed=4), "unit_norm")
    record = pipeline_record(pipeline)
    assert record is not None
    step = record["steps"][0]
    assert step["name"] == "srp" and step["seed"] == 4 and step["seed_source"] == "explicit"
    rebuilt = pipeline_from_record(record)
    assert rebuilt.canonical_chain() == pipeline.canonical_chain()
    coerced = coerce_transform([srp(16, seed=4), "unit_norm"])
    assert coerced is not None
    assert coerced.canonical_chain() == pipeline.canonical_chain()


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
