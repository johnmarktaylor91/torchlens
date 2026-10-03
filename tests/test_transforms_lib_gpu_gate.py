"""GPU/MPS gate for the transforms library (memo B12/10.3; C-XFORM row).

GPU and MPS behavior is UNMEASURED by the design panel (no CUDA/MPS on the
authoring box) and the memo publishes NO on-device number; these tests ARE
the required gate that must run green (on the cluster, D02's C-XFORM row)
before any on-device claim or default ships. Every test SKIPS typed where
the hardware is absent — the claims stay off, the CPU merges stay ungated
by hardware (ruling T3).

What the gate proves (memo 10.3): cross-device generation determinism
(integer hashing is REASONED bit-exact, measured on CPU only — the review's 16.4
rebuttal stands until this runs), on-device chain execution with NO hidden
CPU transfer, canonical digest parity, dense_chunked-vs-CSR output parity
with the realized kernel recorded, and an MPS tested path (the planner
never silently falls back to CPU).
"""

from __future__ import annotations

import pytest
import torch

from torchlens.transforms import (
    BTD,
    TensorSpec,
    TransformContext,
    chain,
    pool_tokens,
    srp,
    srp_realized_facts,
    unit_norm,
)
from torchlens.transforms._srp import _matrix_entry, _project_rows, _reset_srp_state
from torchlens.transforms._srp_hash import (
    MatrixHeader,
    digest_fixed_matrix,
    generate_fixed_columns,
)

pytestmark = pytest.mark.real_model

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
_MPS = pytest.mark.skipif(
    not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available(),
    reason="MPS device required",
)


@pytest.fixture(autouse=True)
def _fresh_state() -> None:
    """Isolate SRP process state per test."""

    _reset_srp_state()
    yield
    _reset_srp_state()


def _device_generation_matches_cpu(device: torch.device) -> None:
    """Generation on ``device`` is bit-identical to CPU, digest included."""

    cpu_pos, cpu_sig = generate_fixed_columns(11, 4096, range(64), 60)
    dev_pos, dev_sig = generate_fixed_columns(11, 4096, range(64), 60, device=device)
    assert torch.equal(dev_pos.cpu(), cpu_pos)
    assert torch.equal(dev_sig.cpu(), cpu_sig)
    scale = 0.25
    header = MatrixHeader(
        construction="very_sparse_fixed",
        extent=4096,
        n_components=64,
        nonzeros_per_column=60,
        scale=scale,
    )
    cpu_digest = digest_fixed_matrix(cpu_pos, cpu_sig, header)
    dev_digest = digest_fixed_matrix(dev_pos, dev_sig, header)
    assert dev_digest == cpu_digest


def _chain_runs_on_device(device: torch.device) -> None:
    """A pooling+SRP+norm chain executes ON the device (no hidden CPU hop)."""

    generator = torch.Generator().manual_seed(1)
    hidden = torch.randn(4, 12, 256, generator=generator).to(device)
    mask = torch.ones(4, 12, dtype=torch.bool, device=device)
    pipeline = chain(pool_tokens("mean"), srp(64, seed=2, mode="features"), unit_norm())
    out = pipeline.apply(hidden, TransformContext(roles=BTD, mask=mask))
    assert out.device.type == device.type
    cpu_out = pipeline.apply(hidden.cpu(), TransformContext(roles=BTD, mask=mask.cpu()))
    # Output parity across devices is tolerance-based with the realized
    # kernel recorded (the digest promise is exact; floats are not).
    assert torch.allclose(out.cpu(), cpu_out, atol=1e-4, rtol=1e-4)
    facts = srp_realized_facts(
        srp(64, seed=2, mode="features"),
        TensorSpec(shape=(None, 256), dtype="torch.float32"),
    )
    assert facts["matrix_digest"].startswith("sha256:")


def _paths_agree_on_device(device: torch.device) -> None:
    """dense / sparse_csr (CUDA) / dense_chunked agree on-device."""

    spec = srp(32, seed=5)
    entry = _matrix_entry(spec, 2000, None)
    rows = torch.randn(8, 2000, generator=torch.Generator().manual_seed(2)).to(device)
    dense = _project_rows(rows, entry, "dense")
    chunked = _project_rows(rows, entry, "dense_chunked")
    assert dense.device.type == device.type
    assert torch.allclose(dense, chunked, atol=1e-4, rtol=1e-4)
    if device.type == "cuda":
        sparse = _project_rows(rows, entry, "sparse_csr")
        assert torch.allclose(dense, sparse, atol=1e-4, rtol=1e-4)


@_CUDA
def test_cuda_generation_is_bit_identical_to_cpu() -> None:
    """O3's on-device leg: the reasoned claim, finally measured (16.4)."""

    _device_generation_matches_cpu(torch.device("cuda"))


@_CUDA
def test_cuda_chain_executes_on_device() -> None:
    """Chain execution stays on CUDA; outputs match CPU within tolerance."""

    _chain_runs_on_device(torch.device("cuda"))


@_CUDA
def test_cuda_multiply_paths_agree() -> None:
    """dense / sparse_csr / dense_chunked parity on CUDA."""

    _paths_agree_on_device(torch.device("cuda"))


@_MPS
def test_mps_generation_is_bit_identical_to_cpu() -> None:
    """O3's MPS leg (integer ops only; the claim is gate-verified, never assumed)."""

    _device_generation_matches_cpu(torch.device("mps"))


@_MPS
def test_mps_chain_executes_on_device_without_sparse() -> None:
    """MPS gets the TESTED dense/dense_chunked path (planner never picks CSR)."""

    from torchlens.transforms._srp import _multiply_path

    assert _multiply_path(200_000, 4096, torch.device("mps"), None) == "dense_chunked"
    _chain_runs_on_device(torch.device("mps"))
    _paths_agree_on_device(torch.device("mps"))
