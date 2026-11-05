"""Replay of an out-of-place ``as_strided`` reads the receiver's whole saved storage.

``Tensor.as_strided`` builds its view from the receiver's storage, not its elements.
xformers ``stack_or_none`` (AMPLIFY-350M's SwiGLU, Menagerie m38) views both halves
of a packed weight from a slice of its first half; a compact clone of that slice has
no second half, so replay raised. Capture saves the receiver with its whole storage
and replay rebuilds it there, keeping the grad flag a parameter slice carries.
"""

from __future__ import annotations

import torch
from _validation_capture import _quiet_validate
from torch import nn

from torchlens.validation import last_validation_failure


class _PackedSlice(nn.Module):
    """View two halves of a packed weight through ``as_strided`` on one half."""

    def __init__(self, second_half: bool) -> None:
        super().__init__()
        self.w12 = nn.Parameter(torch.randn(8, 4))
        self.second_half = second_half

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.w12[4:] if self.second_half else self.w12[:4]
        # Offset 0 counts from the storage start, so both receivers view all of w12.
        stacked = half.as_strided((2, 4, 4), (16, 4, 1), 0)
        return torch.einsum("bi,koi->bko", x, stacked)


def test_as_strided_on_a_packed_weight_slice_validates() -> None:
    """The first half's view reaches past its own elements into the second half."""

    torch.manual_seed(0)
    assert _quiet_validate(_PackedSlice(second_half=False), torch.randn(3, 4)) is True


def test_as_strided_offset_before_the_receiver_validates() -> None:
    """An explicit offset before the receiver's own start reads the bytes ahead of it."""

    torch.manual_seed(0)
    assert _quiet_validate(_PackedSlice(second_half=True), torch.randn(3, 4)) is True


def test_compact_receiver_replay_still_fails(monkeypatch) -> None:
    """Planted bug: a compact receiver clone makes replay fail, so the check still bites."""

    from torchlens.validation import _replay_storage_context as ctx

    def _compact(copy: torch.Tensor, requires_grad: bool, leaf: bool) -> torch.Tensor:
        del leaf
        return copy.detach().clone().requires_grad_(requires_grad)

    monkeypatch.setattr(ctx, "_grad_view_keeping_storage", _compact)
    torch.manual_seed(0)
    assert _quiet_validate(_PackedSlice(second_half=False), torch.randn(3, 4)) is False
    failure = last_validation_failure()
    assert failure is not None and failure.func_name == "as_strided", failure
