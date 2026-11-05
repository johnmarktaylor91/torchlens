"""In-place writes through one channel slice of a batch must not relabel its siblings.

BackgroundMattingV2's ``MattingRefine`` (Menagerie m3600, corpus cluster
c-cddc1def) splits one decoder output into channel slices and writes some of
them in place: ``pha = x[:, 0:1].clamp_(0, 1)``, ``fgr = x[:, 1:4]``,
``hid = x[:, 5:].relu_()``, then reads ``torch.cat([pha, fgr])``. With a batch
of two, each channel slice is two element runs interleaved with its siblings'
runs, so the byte-interval overlap test called them overlapping and every
in-place write relabeled the untouched sibling slices onto the mutation op.
Argument logging then failed at the ``cat`` (and at ``fgr.add_(src)`` after the
refiner's ``out[:, :1].clamp_()``). The element-exact run-lattice test keeps the
mutation edge only on views that share elements.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.backends.torch import _view_writeback
from torchlens.backends.torch._storage_overlap import _strided_views_share_storage_elements
from torchlens.validation import last_validation_failure, validate_forward_pass


class _MattingSlices(nn.Module):
    """The MattingRefine slice-and-mutate pattern at toy size."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Conv2d(3, 9, 3, padding=1)
        self.head = nn.Conv2d(8, 4, 1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Split, mutate and recombine channel slices like MattingRefine.

        Parameters
        ----------
        src:
            ``(B, 3, H, W)`` image batch with ``B > 1``.

        Returns
        -------
        torch.Tensor
            Recombined foreground and alpha.
        """

        x = self.decoder(src)
        pha = x[:, 0:1].clamp_(0.0, 1.0)
        fgr = x[:, 1:4]
        err = x[:, 4:5].clamp_(0.0, 1.0)
        hid = x[:, 5:].relu_()
        mixed = self.head(torch.cat([hid, pha, fgr], dim=1)) * err
        out = torch.cat([pha, fgr], dim=1) + mixed
        pha = out[:, :1]
        fgr = out[:, 1:]
        pha = pha.clamp_(0.0, 1.0)
        fgr = fgr.add_(src).clamp_(0.0, 1.0)
        return torch.cat([pha, fgr], dim=1)


def _ops_by_func(trace: object, func_name: str) -> list:
    """Return the ops of ``trace`` whose ``func_name`` matches, in capture order."""

    return [op for op in trace.ops if op.func_name == func_name]


def _validate(model: nn.Module, x: torch.Tensor) -> bool:
    """Run public forward validation with provenance warnings silenced."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(validate_forward_pass(model, x))


def test_batched_channel_slices_are_element_disjoint() -> None:
    """Sibling channel slices of a batch share no elements; real overlaps still do."""

    x = torch.zeros(2, 9, 4, 5)
    assert _strided_views_share_storage_elements(x[:, 0:1], x[:, 1:4]) is False
    assert _strided_views_share_storage_elements(x[:, 5:], x[:, 1:4]) is False
    assert _strided_views_share_storage_elements(x[:, 4:5], x[:, 5:]) is False
    assert _strided_views_share_storage_elements(x[:, 0:2], x[:, 1:4]) is True
    assert _strided_views_share_storage_elements(x[:, 5:], x) is True
    assert _strided_views_share_storage_elements(x[1:, 0:1], x[:1]) is False
    assert _strided_views_share_storage_elements(x[1:, 0:1], x[:, 1:, 1:2, 3:]) is False
    assert _strided_views_share_storage_elements(x[:, 0:1, 2], x[:, :, 2:3, 1]) is True
    permuted = x.permute(0, 2, 3, 1)
    assert _strided_views_share_storage_elements(permuted[..., 0:1], x[:, 1:4]) is False
    assert _strided_views_share_storage_elements(permuted[..., 2:3], x[:, 1:4]) is True


def test_self_overlapping_layout_stays_conservative() -> None:
    """``unfold`` windows overlap themselves: no exact answer, so keep the edge."""

    base = torch.zeros(12)
    windows = base.unfold(0, 4, 2)
    assert _strided_views_share_storage_elements(windows, base[5:6]) is True


def test_matting_slice_writes_keep_sibling_parents() -> None:
    """The recombining ``cat`` reads the clamped alpha and the untouched foreground."""

    torch.manual_seed(0)
    model = _MattingSlices().eval()
    src = torch.rand(2, 3, 6, 8)
    trace = tl.trace(model, src)
    first_clamp = _ops_by_func(trace, "clamp_")[0]
    fgr_slice = _ops_by_func(trace, "__getitem__")[1]
    relu = _ops_by_func(trace, "relu_")[0]
    second_cat = _ops_by_func(trace, "cat")[1]

    assert set(second_cat.parents) == {first_clamp.layer_label, fgr_slice.layer_label}
    assert relu.layer_label not in second_cat.parents
    add_inplace = _ops_by_func(trace, "add_")[0]
    late_fgr_slice = _ops_by_func(trace, "__getitem__")[5]
    assert late_fgr_slice.layer_label in add_inplace.parents
    assert _validate(_MattingSlices().eval(), src), last_validation_failure()


def test_planted_conservative_overlap_fails_argument_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planted bug: re-inventing sibling overlap must trip argument logging."""

    torch.manual_seed(0)
    src = torch.rand(2, 3, 6, 8)
    monkeypatch.setattr(
        _view_writeback, "_strided_views_share_storage_elements", lambda *_args: True
    )
    assert not _validate(_MattingSlices().eval(), src)
    failure = last_validation_failure()
    assert failure is not None and failure.check_name == "argument_logging", failure
