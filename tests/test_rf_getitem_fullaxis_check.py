"""T7 (grind-p3): getitem mixed-key variants stay oracle-exact, and check()
probes exact claims on non-windowed axes.

HIGH variant battery: the rank-changing getitem fix (slice affines plus
recorded selection indices composed through both engines and both query
walks) is pinned against an independent autograd oracle for the mixed-key
variants the original seed battery (``test_rf_getitem_int_slice.py``) left
uncovered: an int at a non-batch axis, ``None`` inside the key, a negative
int, and an int with two non-trivial slices. All of these served silently
wrong exact boxes before that fix (red on its parent commit).

MED tripwire strengthening: the adjoint direction-coherence oracle
enumerated corners over WINDOWED axes only, so an exact box that
over-approximated on a full (or pointwise) axis was never probed — gradient
containment is structurally unable to see over-coverage, so ``check()``
blessed the lie. The tamper test simulates exactly that regression class (a
receptive-walk bug re-widening a narrowed full axis under the exact claim)
and requires ``check()`` to FAIL. RED before the ``_validation.py``
strengthening: ``check()`` PASSed the tampered box.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

import torchlens as tl


def _gradient_truth(
    model: nn.Module, x: torch.Tensor, out_unit: tuple[int, ...]
) -> dict[int, tuple[int, int]]:
    """Per-input-axis support hull [min, stop) of d out[out_unit] / d x."""

    probe = x.clone().detach().requires_grad_(True)
    out = model(probe)
    out[out_unit].backward()
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


class _GetitemConv(nn.Module):
    """conv3x3 over a configurable getitem of the input."""

    def __init__(self, index_fn, in_channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 3)
        self._index_fn = index_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self._index_fn(x))


def _pinned_box_case(
    index_fn,
    x_shape: tuple[int, ...],
    unit: tuple[int, int],
    pinned_axes: tuple[int, ...],
) -> None:
    """Serve the box, pin it against the autograd oracle, and check() it."""

    torch.manual_seed(0)
    model = _GetitemConv(index_fn).eval()
    x = torch.randn(*x_shape)
    out_unit = (0, 0, *unit)
    truth = _gradient_truth(model, x, out_unit)

    trace = _armed_trace(model, x)
    op = next(o for o in trace.layer_list if "conv" in o.label)
    rf = op.receptive_field
    box = rf.at(unit)
    assert box.exact is True
    by_axis = {axis.input_axis: axis for axis in box.axes}
    for input_axis in pinned_axes:
        axis = by_axis[input_axis]
        assert (axis.index_start, axis.index_stop) == truth[input_axis], (
            f"axis {input_axis}: box {(axis.index_start, axis.index_stop)} "
            f"!= gradient truth {truth[input_axis]}"
        )
    result = rf.check(out_unit)
    assert result.status.name == "PASS", result.message


def test_int_at_channel_plus_slice_box_matches_gradient_truth() -> None:
    """int on the CHANNEL axis with a later slice: x[:, 0, 2:, :].unsqueeze(1)."""

    _pinned_box_case(
        lambda x: x[:, 0, 2:, :].unsqueeze(1),
        (2, 3, 8, 8),
        (1, 2),
        pinned_axes=(1, 2, 3),
    )


def test_none_inside_key_plus_slice_box_matches_gradient_truth() -> None:
    """``None`` INSIDE the key alongside int and slice: x[:, 0, None, 3:, :]."""

    _pinned_box_case(
        lambda x: x[:, 0, None, 3:, :],
        (2, 3, 8, 8),
        (1, 2),
        pinned_axes=(1, 2, 3),
    )


def test_negative_int_plus_slice_box_matches_gradient_truth() -> None:
    """Negative int selection normalizes: x[-1, :, 3:, :].unsqueeze(0)."""

    _pinned_box_case(
        lambda x: x[-1, :, 3:, :].unsqueeze(0),
        (2, 1, 8, 8),
        (1, 2),
        pinned_axes=(0, 2, 3),
    )


def test_int_plus_two_slices_box_matches_gradient_truth() -> None:
    """int with TWO non-trivial slices: x[1, :, 2:7, 1:].unsqueeze(0)."""

    _pinned_box_case(
        lambda x: x[1, :, 2:7, 1:].unsqueeze(0),
        (2, 1, 8, 8),
        (2, 3),
        pinned_axes=(0, 2, 3),
    )


def test_widened_full_axis_exact_claim_fails_check(monkeypatch) -> None:
    """check() must FAIL an exact box re-widened on a narrowed full axis.

    Simulates the asymmetric walk-regression class: the receptive query walk
    serves the full parent extent on the int-selected batch axis under the
    ``exact=True`` claim (the pre-fix getitem behavior), while the projective
    walk still prunes off-index sources correctly. Gradient containment
    cannot see the over-coverage (support stays inside the widened bounds),
    so only the non-windowed-axis adjoint probe can catch it. RED before the
    strengthening: the corner enumeration probed windowed axes only and
    check() PASSed this tampered box.
    """

    torch.manual_seed(0)
    model = _GetitemConv(lambda x: x[0, :, 3:, :].unsqueeze(0)).eval()
    x = torch.randn(2, 1, 8, 8)
    trace = _armed_trace(model, x)
    op = next(o for o in trace.layer_list if "conv" in o.label)

    from torchlens.receptive_field import _validation as validation_module

    real_box_for_unit = validation_module.box_for_unit

    def widened_box_for_unit(*args, **kwargs):
        result = real_box_for_unit(*args, **kwargs)
        axes = []
        tampered = False
        for axis, extent in zip(result.axes, result.input_shape, strict=True):
            if (
                axis.kind == "full"
                and axis.clipped_start is not None
                and axis.clipped_stop is not None
                and axis.clipped_stop - axis.clipped_start < extent
            ):
                axis = replace(
                    axis,
                    index_start=0,
                    index_stop=extent,
                    clipped_start=0,
                    clipped_stop=extent,
                )
                tampered = True
            axes.append(axis)
        return replace(result, axes=tuple(axes)) if tampered else result

    monkeypatch.setattr(validation_module, "box_for_unit", widened_box_for_unit)
    result = op.receptive_field.check((0, 0, 1, 2))
    assert result.status.name == "FAIL", (
        "an exact-claimed box widened on a narrowed full axis must fail the "
        f"adjoint probe; got {result.status.name}: {result.message}"
    )
    assert any("opposite-direction" in violation.reason for violation in result.violations)


def test_honest_full_axis_claims_still_pass_check() -> None:
    """No false positives: honest narrowed-full and pointwise axes PASS.

    The strengthened probe seeds only CLAIMED members (hull endpoints and
    the pointwise same-index coordinate); for an honest exact box each of
    those is a true influencer, so the reverse engine must confirm it.
    """

    torch.manual_seed(0)
    model = _GetitemConv(lambda x: x[0, :, 3:, :].unsqueeze(0)).eval()
    x = torch.randn(2, 1, 8, 8)
    trace = _armed_trace(model, x)
    op = next(o for o in trace.layer_list if "conv" in o.label)
    result = op.receptive_field.check((0, 0, 1, 2))
    assert result.status.name == "PASS", result.message
