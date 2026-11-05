"""Transposed convolution mixes every channel: per-unit receptive-field walks must say so.

The descriptor engine marks a ``groups=1`` transposed convolution's channel axis
as whole-extent, but the per-unit reverse and forward walks used to carry the
channel index straight through. A wider output then indexed past the input's
channels (the box constructor raised ``clipped bounds must not be inverted``
inside validation's metadata invariants, so ``tl.validate`` crashed on DCGAN
generators), and a narrower one returned a wrong single-channel box.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _engine, _query
from torchlens.receptive_field._engine_forward import solve_projective
from torchlens.receptive_field._forward_query import box_for_source_unit
from torchlens.receptive_field._validation import check_geometric_metadata_invariants


def _dcgan_generator() -> nn.Module:
    """Return a tiny DCGAN-style generator: a 1x1 latent widened by transposed convs.

    The first layer has more than twice as many output channels as input
    channels, so the centre output channel lies past the input's last channel.
    """

    return nn.Sequential(
        nn.ConvTranspose2d(2, 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.ConvTranspose2d(8, 3, 4, 2, 1, bias=False),
        nn.Tanh(),
    ).eval()


def _channel_bounds(box: object, axis: int = 1) -> tuple[object, ...]:
    """Return the index and clipped bounds of one box axis."""

    item = box.axes[axis]  # type: ignore[attr-defined]
    return item.index_start, item.index_stop, item.clipped_start, item.clipped_stop


def test_dcgan_generator_validates() -> None:
    """Validation of a channel-widening transposed-conv generator completes and passes."""

    torch.manual_seed(0)
    model = _dcgan_generator()
    assert tl.validate(model, torch.randn(1, 2, 1, 1), seed=0) is True


def test_geometric_invariants_hold_for_channel_widening_transposed_conv() -> None:
    """The always-on metadata invariants run both walks without raising."""

    torch.manual_seed(0)
    trace = tl.trace(_dcgan_generator(), torch.randn(1, 2, 1, 1))
    assert check_geometric_metadata_invariants(trace) is True


@pytest.mark.parametrize(
    ("in_channels", "out_channels", "stride", "extent"),
    [(2, 6, 1, 1), (6, 2, 2, 3)],
)
def test_receptive_box_spans_every_input_channel(
    in_channels: int, out_channels: int, stride: int, extent: int
) -> None:
    """Every output channel's box covers all input channels, exactly."""

    model = nn.ConvTranspose2d(in_channels, out_channels, 3, stride, 1, bias=False)
    trace = tl.trace(model, torch.randn(1, in_channels, extent, extent))
    op = trace.output_ops[0]
    solution = _engine.solve(trace)
    for channel in range(out_channels):
        box = _query.box_for_unit(solution, op, (0, 0), complete_unit=(0, channel, 0, 0))
        assert _channel_bounds(box) == (0, in_channels, 0, in_channels)
        if stride == 1:
            assert box.exact


@pytest.mark.parametrize(("in_channels", "out_channels"), [(2, 6), (6, 2)])
def test_projective_box_spans_every_output_channel(in_channels: int, out_channels: int) -> None:
    """Every input channel's projective box covers all output channels."""

    model = nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1, bias=False)
    trace = tl.trace(model, torch.randn(1, in_channels, 3, 3))
    source = trace.input_ops[0]
    target = trace.output_ops[0]
    solution = solve_projective(trace, [target])
    for channel in range(in_channels):
        box = box_for_source_unit(
            solution, source, (1, 1), target=target, complete_unit=(0, channel, 1, 1)
        )
        assert _channel_bounds(box) == (0, out_channels, 0, out_channels)


def test_interpolation_keeps_channels_pointwise() -> None:
    """A channel-preserving window (upsampling) still narrows to the queried channel."""

    model = nn.Upsample(scale_factor=2, mode="nearest")
    trace = tl.trace(model, torch.randn(1, 4, 3, 3))
    op = trace.output_ops[0]
    box = _query.box_for_unit(_engine.solve(trace), op, (0, 0), complete_unit=(0, 2, 0, 0))
    assert _channel_bounds(box) == (2, 3, 2, 3)
