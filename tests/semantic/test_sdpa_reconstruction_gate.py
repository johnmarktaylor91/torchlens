"""The SDPA facet-reconstruction gate must refuse corrupted reconstructions.

``_allclose_sdpa`` decides whether a reconstructed facet is served as real
or refused as ``MissingFacet``. Its former hand-picked absolute floors
(atol 2e-2 fp16/bf16, 1e-5 fp32) blessed ALL-ZERO and SIGN-FLIPPED
reconstructions of any payload living below the floor -- exactly where
post-softmax attention values live (b4-opus round-2 F13-1, probe-proven on
this tip). The gate now derives its pair from the payload dtype's replay
error model.
"""

from __future__ import annotations

import pytest
import torch

from torchlens.semantic.reconstruction import _allclose_sdpa
from torchlens.utils._torch_compat import get_cpu_half_kernels_support

pytestmark = pytest.mark.smoke

_SKIP_FLOAT16_NEXTAFTER = pytest.mark.skipif(
    not get_cpu_half_kernels_support(),
    reason='torch 2.1-2.2\'s CPU nextafter_cpu kernel does not cover float16 ("nextafter_cpu" '
    "not implemented for 'Half')",
)


@pytest.mark.parametrize(
    ("dtype", "magnitude"),
    [
        (torch.bfloat16, 5e-3),
        (torch.float16, 5e-4),
        (torch.float32, 5e-6),
    ],
)
def test_all_zero_reconstruction_is_refused(dtype: torch.dtype, magnitude: float) -> None:
    """The exact probe: an all-zero reconstruction must NOT match.

    Red-capable: every one of these cases returned ``match=True`` under the
    former absolute floors.
    """

    target = torch.full((16,), magnitude, dtype=dtype)
    assert not _allclose_sdpa(torch.zeros_like(target), target)


@pytest.mark.parametrize(
    ("dtype", "magnitude"),
    [
        (torch.bfloat16, 5e-3),
        (torch.float16, 5e-4),
        (torch.float32, 5e-6),
    ],
)
def test_sign_flipped_reconstruction_is_refused(dtype: torch.dtype, magnitude: float) -> None:
    """The probe's second half: a sign-flipped reconstruction must NOT match."""

    target = torch.full((16,), magnitude, dtype=dtype)
    assert not _allclose_sdpa(-target, target)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        pytest.param(torch.float16, marks=_SKIP_FLOAT16_NEXTAFTER),
        torch.float32,
        torch.float64,
    ],
)
def test_storage_rounding_agreement_still_matches(dtype: torch.dtype) -> None:
    """A legitimate one-ULP storage-rounding difference stays a match."""

    target = torch.full((16,), 0.73, dtype=dtype)
    one_ulp = torch.nextafter(target, torch.ones_like(target))
    assert _allclose_sdpa(one_ulp, target)
    assert _allclose_sdpa(target.clone(), target)


def test_real_sdpa_reconstruction_round_trip_matches() -> None:
    """An honest unfused recomputation of SDPA passes the gate end-to-end."""

    torch.manual_seed(0)
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    fused = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
    pattern = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    z = pattern @ v
    assert _allclose_sdpa(z, fused)
