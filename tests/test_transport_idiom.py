"""R35: device/layout-aware host transport replaces detach().cpu().contiguous().

Converged disputed-r2 acceptance battery (b5 #2): (a) an already-contiguous CPU
tensor still no-ops (zero-copy alias of the same storage), (b) sol's CPU
strided-view counterexample still yields a CONTIGUOUS result — the naive
``.to("cpu", memory_format=contiguous_format)`` one-liner returns an aliased
noncontiguous tensor there, which is exactly why the helper branches on device —
and (c) the channels_last cross-device path takes one host copy, not two
(CUDA-gated here; the real-GPU D2H attestation also rides the standing GPU
probe list).
"""

from __future__ import annotations

import pytest
import torch

from torchlens._transport import to_cpu_contiguous

pytestmark = pytest.mark.smoke


def _assert_equivalent_to_old_idiom(source: torch.Tensor) -> torch.Tensor:
    """Assert helper output matches the historical idiom exactly; return it."""

    result = to_cpu_contiguous(source)
    reference = source.detach().cpu().contiguous()
    assert result.device.type == "cpu"
    assert result.is_contiguous()
    assert result.dtype == reference.dtype
    assert result.shape == reference.shape
    assert torch.equal(result, reference)
    assert not result.requires_grad
    return result


def test_contiguous_cpu_tensor_no_ops() -> None:
    """(a) Already-contiguous CPU input: zero-copy, same storage as the source."""

    source = torch.randn(64, 64)
    result = _assert_equivalent_to_old_idiom(source)
    assert result.data_ptr() == source.data_ptr()


def test_cpu_strided_view_yields_contiguous_copy() -> None:
    """(b) The review's counterexample: a CPU noncontiguous view must come back CONTIGUOUS.

    The disproved one-liner returns the ALIASED noncontiguous view here (0 bytes
    allocated); the shipped helper must materialize a standard-contiguous copy
    exactly like the historical idiom.
    """

    base = torch.randn(64, 8192)
    view = base[:, ::2]
    assert not view.is_contiguous()
    result = _assert_equivalent_to_old_idiom(view)
    assert result.data_ptr() != base.data_ptr()


def test_cpu_transposed_dense_yields_contiguous_copy() -> None:
    """(b) Dense-permuted CPU layout (transposed) contiguizes in one copy."""

    source = torch.randn(128, 256).t()
    assert not source.is_contiguous()
    _assert_equivalent_to_old_idiom(source)


def test_cpu_channels_last_yields_contiguous_copy() -> None:
    """(b) channels_last CPU input comes back standard-contiguous."""

    source = torch.randn(2, 8, 16, 16).to(memory_format=torch.channels_last)
    assert not source.is_contiguous()
    _assert_equivalent_to_old_idiom(source)


def test_requires_grad_source_is_detached() -> None:
    """Transport output never carries autograd tape, matching the old idiom."""

    source = torch.randn(8, 8, requires_grad=True)
    result = to_cpu_contiguous(source)
    assert not result.requires_grad
    assert result.grad_fn is None


def test_uint8_byte_view_works_on_all_layouts() -> None:
    """Digest call sites reshape(-1).view(torch.uint8); helper output must allow it."""

    for source in (
        torch.randn(16, 16),
        torch.randn(16, 32)[:, ::2],
        torch.randn(2, 4, 8, 8).to(memory_format=torch.channels_last),
    ):
        flat = to_cpu_contiguous(source).reshape(-1)
        flat.view(torch.uint8)  # raises on non-contiguous input


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cross_device_channels_last_single_host_copy() -> None:
    """(c) CUDA channels_last transport allocates ONE host buffer, not two."""

    source = torch.randn(2, 8, 32, 32, device="cuda").to(memory_format=torch.channels_last)
    result = _assert_equivalent_to_old_idiom(source)
    assert result.is_contiguous()
    # Single-copy claim: the fused .to() lands directly in standard-contiguous
    # host storage, so no second CPU materialization is observable — the result
    # must not alias any intermediate preserve-format host tensor.
    preserved = source.detach().to("cpu")
    assert preserved.data_ptr() != result.data_ptr()
