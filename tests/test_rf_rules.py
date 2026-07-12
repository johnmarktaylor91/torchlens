"""Golden receptive-field tests for the built-in T5 rule pack."""

from __future__ import annotations

import importlib
from collections.abc import Iterator

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _engine, _query, _rules
from torchlens.receptive_field._types import ReceptiveFieldStatus


_PACK: dict[str, object] | None = None


@pytest.fixture(autouse=True)
def built_in_rule_pack() -> Iterator[None]:
    """Install the pack only for this test, preserving other registry tests' isolation."""

    global _PACK
    original = dict(_rules._RF_RULES)
    original_epoch = _rules._RF_RULES_EPOCH
    _rules._RF_RULES.clear()
    if _PACK is None:
        importlib.import_module("torchlens.receptive_field.rules")
        _PACK = dict(_rules._RF_RULES)
    else:
        _rules._RF_RULES.update(_PACK)
    try:
        yield
    finally:
        _rules._RF_RULES.clear()
        _rules._RF_RULES.update(original)
        _rules._RF_RULES_EPOCH = original_epoch


def _op(trace: object, name: str) -> object:
    """Return the first captured operation with the requested raw function name."""

    return next(item for item in trace.layer_list if item.func_name == name)  # type: ignore[union-attr]


def _descriptor(op: object) -> object:
    """Return the sole input descriptor for a one-input captured operation."""

    return next(iter(_engine.lookup(op.source_trace, op).values()))  # type: ignore[union-attr]


def test_conv_and_pool_golden_sizes_are_exact() -> None:
    """Compose hand-derived convolution and kernel-stride-default pool geometry."""

    model = nn.Sequential(nn.Conv2d(1, 2, 3, stride=2, padding=1), nn.MaxPool2d(2))
    trace = tl.trace(model, torch.randn(1, 1, 11, 11))
    descriptor = _descriptor(_op(trace, "max_pool2d"))

    assert descriptor.status is ReceptiveFieldStatus.EXACT
    assert descriptor.size == (5, 5)
    assert descriptor.jump == (4, 4)
    box = _query.box_for_unit(_engine.solve(trace), _op(trace, "max_pool2d"), (1, 1))
    assert box.status is ReceptiveFieldStatus.EXACT
    assert [(axis.clipped_start, axis.clipped_stop) for axis in box.axes[-2:]] == [(3, 8), (3, 8)]


@pytest.mark.parametrize(
    "stride,status", [(1, ReceptiveFieldStatus.EXACT), (2, ReceptiveFieldStatus.UPPER_BOUND)]
)
def test_transposed_convolution_descriptor_and_unit_golden(
    stride: int, status: ReceptiveFieldStatus
) -> None:
    """Keep stride-one exact and certify stride-two feasible residue sets per unit."""

    trace = tl.trace(nn.ConvTranspose1d(1, 1, 3, stride=stride, padding=1), torch.randn(1, 1, 5))
    op = _op(trace, "conv_transpose1d")
    assert _descriptor(op).status is status
    box = _query.box_for_unit(_engine.solve(trace), op, (2,))
    assert box.status is ReceptiveFieldStatus.EXACT
    assert box.axes[-1].clipped_start == (1 if stride == 1 else 1)
    assert box.axes[-1].clipped_stop == (4 if stride == 1 else 2)


@pytest.mark.parametrize(
    "training,status,batch_coupled",
    [(False, ReceptiveFieldStatus.EXACT, False), (True, ReceptiveFieldStatus.WHOLE_INPUT, True)],
)
def test_batch_norm_mode_matrix(
    training: bool, status: ReceptiveFieldStatus, batch_coupled: bool
) -> None:
    """Use the captured functional statistics switch, never a transparent training rule."""

    model = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2)).train(training)
    trace = tl.trace(model, torch.randn(3, 2, 4, 4))
    descriptor = _descriptor(_op(trace, "batch_norm"))

    assert descriptor.status is status
    assert descriptor.batch_coupled is batch_coupled


def test_group_and_current_stat_instance_norm_globalize_normalized_axes() -> None:
    """Represent statistics-dependent normalization as exact whole normalized extents."""

    class Norms(nn.Module):
        """Small current-stat normalization fixture."""

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Apply instance then group normalization using input statistics."""

            value = functional.instance_norm(value, use_input_stats=True)
            return functional.group_norm(value, 2)

    trace = tl.trace(Norms(), torch.randn(2, 4, 3, 3))
    for name in ("instance_norm", "group_norm"):
        descriptor = _descriptor(_op(trace, name))
        assert descriptor.status is ReceptiveFieldStatus.WHOLE_INPUT
        assert all(axis.kind == "full" for axis in descriptor.axes[1:])


def test_attention_mask_changes_exact_whole_domain_to_upper_bound() -> None:
    """Keep unmasked structural attention exact and masked attention explicitly bounded."""

    query = torch.randn(1, 1, 3, 2)
    unmasked = tl.trace(
        lambda value: functional.scaled_dot_product_attention(value, value, value), query
    )
    masked = tl.trace(
        lambda value: functional.scaled_dot_product_attention(value, value, value, is_causal=True),
        query,
    )

    assert (
        _descriptor(_op(unmasked, "scaled_dot_product_attention")).status
        is ReceptiveFieldStatus.WHOLE_INPUT
    )
    assert (
        _descriptor(_op(masked, "scaled_dot_product_attention")).status
        is ReceptiveFieldStatus.UPPER_BOUND
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"size": (5,), "mode": "linear", "align_corners": True}, (1, 3)),
        ({"scale_factor": 1.5, "mode": "linear", "align_corners": False}, (1, 3)),
    ],
)
def test_interpolation_branch_goldens_are_unit_exact(
    kwargs: dict[str, object], expected: tuple[int, int]
) -> None:
    """Exercise size/scale source branches and exact unit tap selection."""

    trace = tl.trace(lambda value: functional.interpolate(value, **kwargs), torch.randn(1, 1, 3))
    op = _op(trace, "interpolate")
    assert _descriptor(op).status is ReceptiveFieldStatus.UPPER_BOUND
    box = _query.box_for_unit(_engine.solve(trace), op, (3,))
    assert box.status is ReceptiveFieldStatus.EXACT
    assert (box.axes[-1].clipped_start, box.axes[-1].clipped_stop) == expected


def test_roll_is_fail_closed_and_never_pointwise() -> None:
    """Reject the tempting but unsound identity rule for circular shifts."""

    trace = tl.trace(lambda value: torch.roll(value, shifts=1, dims=-1), torch.randn(1, 1, 5))
    descriptor = _descriptor(_op(trace, "roll"))

    assert descriptor.status is ReceptiveFieldStatus.UPPER_BOUND
    assert descriptor.axes[-1].kind == "full"
