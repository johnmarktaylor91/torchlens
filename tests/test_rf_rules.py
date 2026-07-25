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
from torchlens.receptive_field._engine_forward import solve_projective
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
        module = importlib.import_module("torchlens.receptive_field.rules")
        if not _rules._RF_RULES:
            for name in module.__all__:
                importlib.reload(getattr(module, name))
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
    "training,track_running_stats,status,batch_coupled",
    [
        (False, True, ReceptiveFieldStatus.EXACT, False),
        (True, True, ReceptiveFieldStatus.WHOLE_INPUT, True),
        (True, False, ReceptiveFieldStatus.WHOLE_INPUT, True),
    ],
)
def test_batch_norm_mode_matrix(
    training: bool,
    track_running_stats: bool,
    status: ReceptiveFieldStatus,
    batch_coupled: bool,
) -> None:
    """Use the captured functional statistics switch, never a transparent training rule."""

    model = nn.BatchNorm2d(
        2,
        affine=False,
        track_running_stats=track_running_stats,
    ).train(training)
    trace = tl.trace(model, torch.randn(3, 2, 4, 4))
    descriptor = _descriptor(_op(trace, "batch_norm"))

    assert descriptor.status is status
    assert descriptor.batch_coupled is batch_coupled


def test_training_batch_norm_projective_maps_channel_vector_parents() -> None:
    """Map affine BatchNorm vectors onto the channel axis without tainting the solve."""

    trace = tl.trace(nn.BatchNorm2d(2).train(), torch.randn(3, 2, 4, 4))
    batch_norm = _op(trace, "batch_norm")
    solution = solve_projective(trace, trace.output_ops)

    for parent_reference in batch_norm.parents:
        parent = trace.layer_dict_all_keys[parent_reference]
        state = solution.states[(parent.label, trace.output_ops[0].io_role)]
        descriptor = solution.descriptors[(parent.label, trace.output_ops[0].io_role)]
        assert state.axes is not None
        assert descriptor.layout.axis_kinds == ("full", "pointwise", "full", "full")


def test_group_and_current_stat_instance_norm_globalize_normalized_axes() -> None:
    """Keep InstanceNorm exact while bounding GroupNorm to its group envelope."""

    class Norms(nn.Module):
        """Small current-stat normalization fixture."""

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Apply instance then group normalization using input statistics."""

            value = functional.instance_norm(value, use_input_stats=True)
            return functional.group_norm(value, 2)

    trace = tl.trace(Norms(), torch.randn(2, 4, 3, 3))
    instance = _descriptor(_op(trace, "instance_norm"))
    assert instance.status is ReceptiveFieldStatus.WHOLE_INPUT
    assert all(axis.kind == "full" for axis in instance.axes[2:])

    group = _descriptor(_op(trace, "group_norm"))
    assert group.status is ReceptiveFieldStatus.UPPER_BOUND
    assert all(axis.kind == "full" for axis in group.axes[1:])
    assert group.axes[1].exact is False


def test_attention_mask_changes_exact_whole_domain_to_upper_bound() -> None:
    """Bound attention on sequence and feature axes, preserving batch and head locality."""

    class Attention(nn.Module):
        """Small scaled-dot-product attention fixture."""

        def __init__(self, *, causal: bool) -> None:
            """Store whether the attention call is causal."""

            super().__init__()
            self.causal = causal

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Apply self-attention with the configured mask behavior."""

            return functional.scaled_dot_product_attention(
                value, value, value, is_causal=self.causal
            )

    query = torch.randn(2, 2, 3, 2)
    unmasked = tl.trace(Attention(causal=False), query)
    masked = tl.trace(Attention(causal=True), query)

    for trace in (unmasked, masked):
        descriptor = _descriptor(_op(trace, "scaled_dot_product_attention"))
        assert descriptor.status is ReceptiveFieldStatus.UPPER_BOUND
        assert tuple(axis.kind for axis in descriptor.axes) == (
            "pointwise",
            "pointwise",
            "full",
            "full",
        )
        assert descriptor.batch_coupled is False


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"size": (5,), "mode": "linear", "align_corners": True}, (1, 3)),
        ({"scale_factor": 1.5, "mode": "linear", "align_corners": False}, (1, 3)),
        ({"size": (2,), "mode": "linear", "align_corners": True}, (9, 10)),
    ],
)
def test_interpolation_branch_goldens_are_unit_exact(
    kwargs: dict[str, object], expected: tuple[int, int]
) -> None:
    """Exercise size/scale source branches and exact unit tap selection."""

    class Interpolation(nn.Module):
        """Small interpolation fixture with captured keyword arguments."""

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Interpolate the input with the parametrized branch arguments."""

            return functional.interpolate(value, **kwargs)

    input_extent = 10 if kwargs.get("size") == (2,) else 3
    trace = tl.trace(Interpolation(), torch.randn(1, 1, input_extent))
    op = _op(trace, "interpolate")
    descriptor = _descriptor(op)
    assert descriptor.status is ReceptiveFieldStatus.UPPER_BOUND
    unit = (1,) if kwargs.get("size") == (2,) else (3,)
    box = _query.box_for_unit(_engine.solve(trace), op, unit)
    assert box.status is ReceptiveFieldStatus.EXACT
    assert (box.axes[-1].clipped_start, box.axes[-1].clipped_stop) == expected
    if kwargs.get("size") == (2,):
        assert descriptor.axes[-1].jump == 9
        assert descriptor.axes[-1].center0 == 0


def test_roll_is_fail_closed_and_never_pointwise() -> None:
    """Reject the tempting but unsound identity rule for circular shifts."""

    class Roll(nn.Module):
        """Small circular-shift fixture."""

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Shift the final input axis circularly."""

            return torch.roll(value, shifts=1, dims=-1)

    trace = tl.trace(Roll(), torch.randn(1, 1, 5))
    descriptor = _descriptor(_op(trace, "roll"))

    assert descriptor.status is ReceptiveFieldStatus.UPPER_BOUND
    assert descriptor.axes[-1].kind == "full"


def test_pointwise_activations_pass_receptive_field_through_exactly() -> None:
    """relu and shape-preserving activations are pointwise: the RF composes through them EXACTLY.

    Regression: the elementwise rule used to mark activations whole-input, which silently degraded
    the receptive field for every CNN with an activation between convolutions.
    """

    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1),
    )
    trace = tl.trace(model, torch.randn(1, 3, 16, 16))
    deep_conv = [op for op in trace.layer_list if "conv" in str(op.func_name).lower()][-1]
    descriptor = deep_conv.receptive_field._descriptor()

    assert descriptor.status is ReceptiveFieldStatus.EXACT
    assert descriptor.layout.axis_kinds == ("pointwise", "full", "windowed", "windowed")
    assert descriptor.size == (5, 5)  # two 3x3 convs; the ReLU passes the field through unchanged
