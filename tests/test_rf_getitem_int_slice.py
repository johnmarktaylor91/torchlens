"""R20-2/R20-3: rank-changing getitem receptive fields are exact and honest.

Disputed-r2 b6 seed battery. The rank-changing getitem path (int+slice mixes)
DISCARDED the computed slice affine edges and served unshifted boxes under
``exact=True`` (pad0: H [1,4) vs gradient truth [4,7)), while the int-selected
batch axis over-covered as ``full [0,2)`` with the exact claim intact and
``check()`` blind to it (the adjoint cross-check probed only windowed axes).
The fix composes the slice affine through both engines and both query walks,
narrows scalar-selected axes to their recorded index, folds non-windowed
conservatism into box exactness, and extends the reverse-membership tripwire
to concrete-bounded non-windowed axes. Every box below is pinned against an
independent autograd oracle; all of these were RED before the fix.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl


def _gradient_truth(
    model: nn.Module, x: torch.Tensor, unit: tuple[int, ...]
) -> dict[int, tuple[int, int]]:
    """Per-input-axis support hull [min, stop) of d out[unit] / d x, via autograd."""

    probe = x.clone().detach().requires_grad_(True)
    out = model(probe)
    scalar = out[(0, 0, *unit)]
    scalar.backward()
    assert probe.grad is not None
    support = torch.nonzero(probe.grad != 0, as_tuple=False)
    assert support.numel() > 0, "oracle gradient is empty; probe misconstructed"
    hull: dict[int, tuple[int, int]] = {}
    for axis in range(probe.dim()):
        coords = support[:, axis]
        hull[axis] = (int(coords.min()), int(coords.max()) + 1)
    return hull


def _armed_trace(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    return tl.trace(
        model,
        x.requires_grad_(True),
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )


class _IntSliceConv(nn.Module):
    """conv3x3(x[0, :, 3:, :].unsqueeze(0)) — the exact filed construction."""

    def __init__(self, padding: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x[0, :, 3:, :].unsqueeze(0))


@pytest.mark.parametrize("padding", [0, 1])
def test_int_plus_slice_box_matches_gradient_truth(padding: int) -> None:
    """The served exact box composes the slice offset and narrows the batch."""

    torch.manual_seed(0)
    model = _IntSliceConv(padding).eval()
    x = torch.randn(2, 1, 8, 8)
    unit = (1, 2)
    truth = _gradient_truth(model, x, unit)

    trace = _armed_trace(model, x)
    op = next(o for o in trace.layer_list if "conv" in o.label)
    box = op.receptive_field.at(unit)
    assert box.exact is True
    by_axis = {axis.input_axis: axis for axis in box.axes}
    # Batch: int-selected axis narrows to the recorded singleton (R20-3).
    assert (by_axis[0].index_start, by_axis[0].index_stop) == truth[0] == (0, 1)
    # H: slice's +3 affine offset composes through the rank change (R20-2).
    assert (by_axis[2].index_start, by_axis[2].index_stop) == truth[2]
    # W: untouched axis stays correct.
    assert (by_axis[3].index_start, by_axis[3].index_stop) == truth[3]


@pytest.mark.smoke_cells("test_int_plus_slice_check_and_verify_pass[1]")
@pytest.mark.parametrize("padding", [0, 1])
def test_int_plus_slice_check_and_verify_pass(padding: int) -> None:
    """The armed tripwire agrees with the now-honest geometry."""

    torch.manual_seed(0)
    model = _IntSliceConv(padding).eval()
    x = torch.randn(2, 1, 8, 8)
    trace = _armed_trace(model, x)
    op = next(o for o in trace.layer_list if "conv" in o.label)
    rf = op.receptive_field
    unit = rf.center_unit(batch_index=0)
    result = rf.check(unit)
    assert result.status.name == "PASS", result.message
    # check() consumed the captured autograd graph; verify() needs a fresh one.
    fresh = _armed_trace(model, x)
    verdict = tl.receptive_field.verify(fresh, units="center")
    # Before the fix this verdict was FAIL (real geometry violations). One
    # honest INDETERMINATE probe remains and predates the fix: the getitem op
    # probed as a source returns no VJP (a value-dead autograd path), which
    # verify reports as unprovable rather than passed. No probe may FAIL.
    assert verdict.verdict.name in {"PASS", "INDETERMINATE"}, verdict
    assert all(item.status.name != "FAIL" for item in verdict.containment), verdict.containment


def test_strided_int_plus_slice_box_matches_gradient_truth() -> None:
    """A step-2 slice through the rank change keeps the exact lattice affine."""

    class _Strided(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x[0, :, 1::2, :].unsqueeze(0))

    torch.manual_seed(0)
    model = _Strided().eval()
    x = torch.randn(2, 1, 12, 8)
    unit = (1, 2)
    truth = _gradient_truth(model, x, unit)
    trace = _armed_trace(model, x)
    op = next(o for o in trace.layer_list if "conv" in o.label)
    box = op.receptive_field.at(unit)
    by_axis = {axis.input_axis: axis for axis in box.axes}
    assert (by_axis[0].index_start, by_axis[0].index_stop) == (0, 1)
    assert (by_axis[2].index_start, by_axis[2].index_stop) == truth[2]
    assert (by_axis[3].index_start, by_axis[3].index_stop) == truth[3]


def test_strided_slice_discloses_sparse_support() -> None:
    """A step-2 slice axis must disclose ``sparse_possible`` (r3 b6-fable R20-1).

    The slice-affine composition claimed ``exact=True, sparse_possible=False``
    for step != 1 while the true support has holes (gradient truth keeps only
    every second row). RED before the fix: H axis reported dense support.
    """

    class _Strided(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x[0, :, ::2, :].unsqueeze(0))

    torch.manual_seed(0)
    model = _Strided().eval()
    x = torch.randn(2, 1, 12, 8)

    # Independent oracle: the H support of any output unit has holes.
    probe = x.clone().detach().requires_grad_(True)
    out = model(probe)
    out[(0, 0, 1, 2)].backward()
    assert probe.grad is not None
    h_rows = torch.nonzero(probe.grad[0, 0].abs().sum(dim=1) != 0).flatten()
    h_rows_list = [int(r) for r in h_rows]
    span = range(h_rows_list[0], h_rows_list[-1] + 1)
    assert set(h_rows_list) != set(span), "oracle support unexpectedly dense"

    trace = _armed_trace(model, x)
    op = next(o for o in trace.layer_list if "conv" in o.label)
    axes = {axis.input_axis: axis for axis in op.receptive_field.axes}
    # The strided H axis discloses holes; the step-1 W axis stays dense.
    assert axes[2].sparse_possible is True
    assert axes[3].sparse_possible is False


def test_int_only_batch_axis_narrows_and_stays_honest() -> None:
    """int-only selection: batch box is the singleton, not the full extent."""

    class _IntOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x[1].unsqueeze(0))

    torch.manual_seed(0)
    model = _IntOnly().eval()
    x = torch.randn(3, 1, 8, 8)
    unit = (1, 2)
    truth = _gradient_truth(model, x, unit)
    assert truth[0] == (1, 2)

    trace = _armed_trace(model, x)
    op = next(o for o in trace.layer_list if "conv" in o.label)
    rf = op.receptive_field
    box = rf.at(unit)
    by_axis = {axis.input_axis: axis for axis in box.axes}
    assert (by_axis[0].index_start, by_axis[0].index_stop) == (1, 2)
    assert (by_axis[2].index_start, by_axis[2].index_stop) == truth[2]
    result = rf.check(rf.center_unit(batch_index=0))
    assert result.status.name == "PASS", result.message


def test_slice_only_same_rank_path_unchanged() -> None:
    """Regression guard: the same-rank slice branch stays exact."""

    class _SliceOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x[:, :, 3:, :])

    torch.manual_seed(0)
    model = _SliceOnly().eval()
    x = torch.randn(2, 1, 8, 8)
    unit = (1, 2)
    truth = _gradient_truth(model, x, unit)
    trace = _armed_trace(model, x)
    op = next(o for o in trace.layer_list if "conv" in o.label)
    box = op.receptive_field.at(unit)
    by_axis = {axis.input_axis: axis for axis in box.axes}
    assert (by_axis[2].index_start, by_axis[2].index_stop) == truth[2] == (4, 7)


@pytest.mark.smoke
def test_projective_field_respects_selected_index() -> None:
    """The reverse direction prunes source units off the selected index."""

    torch.manual_seed(0)
    model = _IntSliceConv(0).eval()
    x = torch.randn(2, 1, 8, 8)
    trace = _armed_trace(model, x)
    input_op = trace.input_ops[0]
    conv = next(o for o in trace.layer_list if "conv" in o.label)

    # A source H coordinate ON the kept slice reaches the conv grid.
    on_path = input_op.projective_field.at((4, 3), target=conv)
    assert not on_path.empty
    # A source H coordinate BEFORE the slice start (rows 0-2 are dropped by
    # x[..., 3:, :]) has no influence at all.
    off_path = input_op.projective_field.at((1, 3), target=conv)
    assert off_path.empty
