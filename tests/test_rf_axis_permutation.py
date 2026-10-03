"""Axis-permutation receptive/projective geometry battery (R20-1, fixwave-2).

Any transpose/permute/movedim between the queried operation and the model
input makes the descriptor's input-axis order differ from the unit's
output-axis order. Zipping windowed unit coordinates in descriptor order
silently TRANSPOSED exact boxes (equal extents) or refused valid units as
out-of-bounds (unequal extents). Every claim here is pinned against a
brute-force perturbation oracle that never consults the TorchLens geometry
engine; the seed test is the b6 dedup repro (`/tmp/b6dedup/r20c.py`), which
failed verbatim before the ordering fix.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.receptive_field import ReceptiveFieldValidationStatus

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Independent brute-force oracles (no TorchLens geometry involved)
# ---------------------------------------------------------------------------


def _forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x).detach().clone()


def true_receptive_hulls(
    model: nn.Module, x: torch.Tensor, out_index: tuple[int, ...], axes: tuple[int, ...]
) -> dict[int, tuple[int, int]]:
    """Half-open per-axis hull of input elements influencing ``out[out_index]``."""

    base = _forward(model, x)
    hits: list[tuple[int, ...]] = []
    shape = tuple(x.shape)
    for flat in range(x.numel()):
        index = []
        remaining = flat
        for extent in reversed(shape):
            index.append(remaining % extent)
            remaining //= extent
        index_tuple = tuple(reversed(index))
        for delta in (1000.0, -1000.0, 0.5):
            perturbed = x.detach().clone()
            perturbed[index_tuple] += delta
            if not torch.allclose(_forward(model, perturbed)[out_index], base[out_index]):
                hits.append(index_tuple)
                break
    assert hits, "brute-force receptive support must be non-empty"
    return {
        axis: (min(hit[axis] for hit in hits), max(hit[axis] for hit in hits) + 1) for axis in axes
    }


def true_projective_hulls(
    model: nn.Module, x: torch.Tensor, source_index: tuple[int, ...], axes: tuple[int, ...]
) -> dict[int, tuple[int, int]]:
    """Half-open per-axis hull of outputs influenced by ``x[source_index]``."""

    base = _forward(model, x)
    hits: set[tuple[int, ...]] = set()
    for delta in (1000.0, -1000.0, 0.5):
        perturbed = x.detach().clone()
        perturbed[source_index] += delta
        out = _forward(model, perturbed)
        for row in (out != base).nonzero(as_tuple=False).tolist():
            hits.add(tuple(int(value) for value in row))
    assert hits, "brute-force projective support must be non-empty"
    return {
        axis: (min(hit[axis] for hit in hits), max(hit[axis] for hit in hits) + 1) for axis in axes
    }


def capture(model: nn.Module, x: torch.Tensor) -> object:
    return tl.trace(
        model,
        x.detach().clone().requires_grad_(True),
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )


def last_conv(trace: object) -> object:
    matches = [op for op in trace.layer_list if op.label.startswith("conv2d")]
    assert matches, "no conv2d op captured"
    return matches[-1]


def windowed_bounds(box: object) -> dict[int, tuple[int, int]]:
    return {
        axis.input_axis: (axis.clipped_start, axis.clipped_stop)
        for axis in box.axes
        if axis.kind == "windowed"
    }


class _PermutedConvPair(nn.Module):
    """conv3x3 -> axis permutation -> conv, the R20-1 corruption shape."""

    def __init__(
        self,
        permute: str,
        second_kernel: tuple[int, int] = (3, 3),
        second_padding: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 1, 3, padding=1)
        self.c2 = nn.Conv2d(1, 1, second_kernel, padding=second_padding)
        self.permute = permute

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.c1(x)
        if self.permute == "transpose":
            hidden = hidden.transpose(2, 3)
        elif self.permute == "permute":
            hidden = hidden.permute(0, 1, 3, 2)
        elif self.permute == "movedim":
            hidden = hidden.movedim(3, 2)
        else:  # pragma: no cover - guard against typo'd parametrization
            raise AssertionError(self.permute)
        return self.c2(hidden)


@pytest.mark.parametrize("permute", ["transpose", "permute", "movedim"])
def test_axis_permutation_receptive_box_matches_bruteforce(permute: str) -> None:
    """Seed test (b6 dedup repro): exact boxes must not be transposed."""

    model = _PermutedConvPair(permute).eval()
    x = torch.randn(1, 1, 12, 12)
    truth = true_receptive_hulls(model, x, (0, 0, 2, 3), axes=(2, 3))

    trace = capture(model, x)
    op = last_conv(trace)
    box = op.receptive_field.at((2, 3))
    assert box.exact
    assert windowed_bounds(box) == truth

    checked = op.receptive_field.check((0, 0, 2, 3))
    assert checked.status is ReceptiveFieldValidationStatus.PASS
    assert checked.n_violations == 0


def test_transpose_unequal_extents_accepts_valid_units() -> None:
    """Unequal extents: the corrupted pairing refused in-range units as OOB."""

    model = _PermutedConvPair("transpose").eval()
    x = torch.randn(1, 1, 10, 14)
    trace = capture(model, x)
    op = last_conv(trace)
    assert tuple(op.shape) == (1, 1, 14, 10)

    # Coordinate 13 is valid on output axis 2 (extent 14) and was rejected
    # against the wrongly-paired extent 10 before the ordering fix.
    truth = true_receptive_hulls(model, x, (0, 0, 13, 3), axes=(2, 3))
    box = op.receptive_field.at((13, 3))
    assert box.exact
    assert windowed_bounds(box) == truth

    checked = op.receptive_field.check((0, 0, 13, 3))
    assert checked.status is ReceptiveFieldValidationStatus.PASS
    assert checked.n_violations == 0


def test_transpose_asymmetric_kernel_projective_matches_bruteforce() -> None:
    """Projective direction through a transpose, distinguishable H/W geometry."""

    model = _PermutedConvPair("transpose", second_kernel=(5, 1), second_padding=(2, 0)).eval()
    x = torch.randn(1, 1, 12, 12)
    truth = true_projective_hulls(model, x, (0, 0, 4, 7), axes=(2, 3))

    trace = capture(model, x)
    op = last_conv(trace)
    source = next(item for item in trace.layer_list if item.is_input)
    box = source.projective_field.at((4, 7), target=op)
    assert box.exact
    assert windowed_bounds(box) == truth


def test_transpose_center_unit_and_center_selector_agree() -> None:
    """center_unit and at('center') address the operation's own output grid."""

    model = _PermutedConvPair("transpose").eval()
    x = torch.randn(1, 1, 10, 14)
    trace = capture(model, x)
    op = last_conv(trace)
    assert tuple(op.shape) == (1, 1, 14, 10)

    complete = op.receptive_field.center_unit(batch_index=0)
    assert complete == (0, 0, 7, 5)
    centered = op.receptive_field.at("center")
    explicit = op.receptive_field.at((7, 5))
    assert windowed_bounds(centered) == windowed_bounds(explicit)

    checked = op.receptive_field.check(complete)
    assert checked.status is ReceptiveFieldValidationStatus.PASS
    assert checked.n_violations == 0
