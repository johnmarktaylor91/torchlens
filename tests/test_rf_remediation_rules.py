"""Regression goldens for exact structural and pointwise receptive-field rules."""

from __future__ import annotations

from collections.abc import Iterator
import importlib

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

import torchlens as tl
from torchlens.capture.arg_positions import _normalize_func_name
from torchlens.receptive_field import _rules
from torchlens.receptive_field._engine_forward import solve_projective
from torchlens.receptive_field._rules import ReceptiveFieldRuleContext, _RuleResult
from torchlens.receptive_field._types import (
    ReceptiveFieldStatus,
    ReceptiveFieldValidationStatus,
)

resnet = pytest.importorskip("torchvision.models.resnet")
BasicBlock = resnet.BasicBlock


_PACK: dict[str, object] | None = None


@pytest.fixture(autouse=True)
def built_in_rule_pack() -> Iterator[None]:
    """Install the built-in RF rules while preserving registry isolation."""

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


def _op(trace: object, name: str, *, last: bool = True) -> object:
    """Return a captured operation by normalized function name.

    Parameters
    ----------
    trace:
        Captured TorchLens trace.
    name:
        Normalized operation name.
    last:
        Whether to return the last rather than first match.

    Returns
    -------
    object
        Matching captured operation.
    """

    matches = [
        item
        for item in trace.layer_list  # type: ignore[union-attr]
        if item.func_name is not None and _normalize_func_name(item.func_name) == name
    ]
    assert matches
    return matches[-1] if last else matches[0]


def _trace(model: nn.Module, inputs: torch.Tensor) -> object:
    """Capture a backward-ready trace for geometric cross-validation.

    Parameters
    ----------
    model:
        Module to capture.
    inputs:
        Differentiable model input.

    Returns
    -------
    object
        Captured TorchLens trace.
    """

    capture = tl.options.CaptureOptions(backward_ready=True)
    return tl.trace(model, inputs, capture=capture, save_mode="reference")


def _fill_convolutions(model: nn.Module) -> None:
    """Fill every convolution weight with positive values for support probes.

    Parameters
    ----------
    model:
        Module whose convolution weights should be initialized.
    """

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.fill_(1.0)
                if module.bias is not None:
                    module.bias.zero_()


def test_real_resnet_basic_block_deep_field_is_exact() -> None:
    """Keep a real ResNet BasicBlock exact across its residual in-place add."""

    block = BasicBlock(4, 4).eval()
    _fill_convolutions(block)
    trace = _trace(block, torch.ones(1, 4, 15, 15, requires_grad=True))
    deep = _op(trace, "relu")
    descriptor = deep.receptive_field._descriptor()  # type: ignore[union-attr]

    assert descriptor.status is ReceptiveFieldStatus.EXACT
    assert descriptor.size == (5, 5)
    assert descriptor.batch_coupled is False
    result = deep.receptive_field.check((0, 0, 7, 7))  # type: ignore[union-attr]
    assert result.status is ReceptiveFieldValidationStatus.PASS


class _TwoBranchMerge(nn.Module):
    """Two convolution branches combined by add or channel concatenation."""

    def __init__(self, *, concatenate: bool) -> None:
        """Create two differently windowed branches.

        Parameters
        ----------
        concatenate:
            Whether to concatenate instead of adding branch outputs.
        """

        super().__init__()
        self.concatenate = concatenate
        self.small = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.large = nn.Conv2d(1, 1, 5, padding=2, bias=False)
        _fill_convolutions(self)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Merge the two convolution branches.

        Parameters
        ----------
        value:
            Input image batch.

        Returns
        -------
        torch.Tensor
            Added or channel-concatenated branch output.
        """

        small = self.small(value)
        large = self.large(value)
        return torch.cat((small, large), dim=1) if self.concatenate else small + large


@pytest.mark.parametrize(("concatenate", "name"), [(False, "add"), (True, "cat")])
def test_branch_merge_unions_convolution_fields(concatenate: bool, name: str) -> None:
    """Union 3x3 and 5x5 branch support for add and channel concatenation."""

    trace = _trace(
        _TwoBranchMerge(concatenate=concatenate),
        torch.ones(1, 1, 11, 11, requires_grad=True),
    )
    merge = _op(trace, name)
    descriptor = merge.receptive_field._descriptor()  # type: ignore[union-attr]

    expected_status = (
        ReceptiveFieldStatus.UPPER_BOUND if concatenate else ReceptiveFieldStatus.EXACT
    )
    assert descriptor.status is expected_status
    assert descriptor.size == (5, 5)
    assert descriptor.batch_coupled is False
    channel = 1 if concatenate else 0
    result = merge.receptive_field.check((0, channel, 5, 5))  # type: ignore[union-attr]
    assert result.status is ReceptiveFieldValidationStatus.PASS


def test_channel_cat_routes_each_output_channel_to_its_exact_branch_field() -> None:
    """Use the selected channel segment's 3x3 or 5x5 receptive field."""

    trace = _trace(
        _TwoBranchMerge(concatenate=True),
        torch.ones(1, 1, 9, 9, requires_grad=True),
    )
    target = _op(trace, "cat")
    results = tl.receptive_field.cross_validate(
        trace,
        ops=[target],
        units=[(0, 0, 4, 4), (0, 1, 4, 4)],
    )

    assert len(results) == 2
    expected_spatial_bounds = (((3, 6), (3, 6)), ((2, 7), (2, 7)))
    for result, expected in zip(results, expected_spatial_bounds, strict=True):
        box = next(iter(result.geometric.values()))
        actual = tuple((axis.clipped_start, axis.clipped_stop) for axis in box.axes[-2:])
        assert box.status is ReceptiveFieldStatus.EXACT
        assert actual == expected
        assert result.status is ReceptiveFieldValidationStatus.PASS

    source = trace.input_ops[0]  # type: ignore[union-attr]
    projective = tl.receptive_field.cross_validate(
        trace,
        ops=[source],
        units=(0, 0, 4, 4),
        direction="projective",
        target=target,
    )[0]
    assert projective.status is ReceptiveFieldValidationStatus.PASS


class _SpatialConcatenation(nn.Module):
    """Two positive convolution branches concatenated along the height axis."""

    def __init__(self, *, uneven: bool) -> None:
        """Create equal or uneven concatenation segments.

        Parameters
        ----------
        uneven:
            Whether to crop the first branch to three rows.
        """

        super().__init__()
        self.uneven = uneven
        self.first = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.second = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        _fill_convolutions(self)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Concatenate the optionally cropped first branch before the second branch."""

        first = self.first(value)
        second = self.second(value)
        if self.uneven:
            first = first[:, :, :3]
        return torch.cat((first, second), dim=2)


@pytest.mark.parametrize(
    ("uneven", "unit", "expected_height"),
    [
        (True, (0, 0, 5, 4), (1, 4)),
        (False, (0, 0, 12, 4), (3, 6)),
    ],
)
def test_spatial_cat_routes_second_segment_exactly(
    uneven: bool,
    unit: tuple[int, int, int, int],
    expected_height: tuple[int, int],
) -> None:
    """Apply segment offsets for uneven and even concatenation without crashes."""

    trace = _trace(
        _SpatialConcatenation(uneven=uneven),
        torch.ones(1, 1, 8, 8, requires_grad=True),
    )
    target = _op(trace, "cat")
    box = target.receptive_field.at(unit[-2:])  # type: ignore[union-attr]

    assert target.receptive_field.status is ReceptiveFieldStatus.UPPER_BOUND  # type: ignore[union-attr]
    assert (box.axes[2].clipped_start, box.axes[2].clipped_stop) == expected_height
    assert box.exact
    result = target.receptive_field.check(unit)  # type: ignore[union-attr]
    assert result.status is ReceptiveFieldValidationStatus.PASS


def test_spatial_cat_projective_offsets_route_into_target_segments() -> None:
    """Apply the shared concatenation offsets in the projective direction."""

    trace = _trace(
        _SpatialConcatenation(uneven=True),
        torch.ones(1, 1, 8, 8, requires_grad=True),
    )
    source = trace.input_ops[0]  # type: ignore[union-attr]
    target = _op(trace, "cat")
    box = source.projective_field.at((4, 4), target=target)

    assert (box.axes[2].clipped_start, box.axes[2].clipped_stop) == (6, 9)
    result = source.projective_field.check((0, 0, 4, 4), target=target)
    assert result.status is ReceptiveFieldValidationStatus.PASS


class _EmptyTensorConcatenation(nn.Module):
    """Prepend PyTorch's rank-one empty-tensor concatenation sentinel."""

    def __init__(self) -> None:
        """Create a windowed branch whose output is concatenated."""

        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        _fill_convolutions(self)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Concatenate a rank-one empty tensor without changing the input extent."""

        value = self.conv(value)
        empty = torch.empty((0,), dtype=value.dtype, device=value.device)
        return torch.cat((empty, value), dim=2)


def test_cat_rank_one_empty_tensor_contributes_zero_extent() -> None:
    """Treat PyTorch's rank-one empty concat sentinel as a zero-width segment."""

    trace = _trace(
        _EmptyTensorConcatenation(),
        torch.ones(1, 1, 8, 8, requires_grad=True),
    )
    target = _op(trace, "cat")
    box = target.receptive_field.at((4, 4))  # type: ignore[union-attr]

    assert tuple(target.shape) == (1, 1, 8, 8)  # type: ignore[union-attr]
    assert (box.axes[2].clipped_start, box.axes[2].clipped_stop) == (3, 6)
    result = target.receptive_field.check((0, 0, 4, 4))  # type: ignore[union-attr]
    assert result.status is ReceptiveFieldValidationStatus.PASS


class _StructuralChain(nn.Module):
    """Convolution followed by exact permutation and singleton reshape operations."""

    def __init__(self) -> None:
        """Create the structural regression model."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1, bias=False)
        _fill_convolutions(self)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply inverse axis transforms and singleton-only reshapes.

        Parameters
        ----------
        value:
            Input image batch.

        Returns
        -------
        torch.Tensor
            Structurally transformed convolution result.
        """

        value = self.conv(value)
        value = value.transpose(1, 2).permute(0, 2, 1, 3)
        value = value.unsqueeze(2).reshape(1, 2, 1, 9, 9)
        value = value.view(1, 2, 1, 9, 9).squeeze(2)
        return value.flatten(2, 2)


def test_structural_axis_ops_preserve_field_exactly() -> None:
    """Preserve convolution geometry through transpose, permute, reshape, view, and flatten."""

    trace = _trace(_StructuralChain(), torch.ones(1, 1, 9, 9, requires_grad=True))
    output = _op(trace, "flatten")
    descriptor = output.receptive_field._descriptor()  # type: ignore[union-attr]

    assert descriptor.status is ReceptiveFieldStatus.EXACT
    assert descriptor.size == (3, 3)
    result = output.receptive_field.check((0, 0, 4, 4))  # type: ignore[union-attr]
    assert result.status is ReceptiveFieldValidationStatus.PASS


class _MiddleReduction(nn.Module):
    """Reduction over a non-leading, non-trailing tensor axis."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Average over the middle channel-like axis.

        Parameters
        ----------
        value:
            Four-dimensional input tensor.

        Returns
        -------
        torch.Tensor
            Tensor with axis two removed.
        """

        return value.mean(dim=2)


def test_middle_axis_reduction_uses_surviving_axis_map() -> None:
    """Keep x.mean(dim=2) sound instead of trailing-aligning its surviving axes."""

    trace = _trace(
        _MiddleReduction(),
        torch.ones(2, 6, 5, 4, requires_grad=True),
    )
    reduction = _op(trace, "mean")
    descriptor = reduction.receptive_field._descriptor()  # type: ignore[union-attr]

    assert descriptor.axes is not None
    assert tuple(axis.output_axis for axis in descriptor.axes) == (0, 1, None, 2)
    result = reduction.receptive_field.check((0, 3, 2))  # type: ignore[union-attr]
    assert result.status is ReceptiveFieldValidationStatus.PASS


def test_positional_reduction_dimension_uses_surviving_axis_map() -> None:
    """Read positional reduction dimensions when builtin signatures omit ``dim``."""

    class PositionalReduction(nn.Module):
        """Reduction passing the dimension as an unnamed positional argument."""

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Sum over the trailing axis."""

            return value.sum(-1)

    trace = _trace(
        PositionalReduction(),
        torch.ones(2, 3, 4, requires_grad=True),
    )
    reduction = _op(trace, "sum")
    descriptor = reduction.receptive_field._descriptor()  # type: ignore[union-attr]

    assert descriptor.axes is not None
    assert tuple(axis.output_axis for axis in descriptor.axes) == (0, 1, None)


def test_rank_changing_full_rule_does_not_require_passthrough_map() -> None:
    """Globalize every contracted axis without inventing a rank-change map."""

    class RankChangingEinsum(nn.Module):
        """Einsum reducing two matrix axes into one batch axis."""

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Take a batched matrix trace."""

            return torch.einsum("bii->b", value)

    trace = _trace(
        RankChangingEinsum(),
        torch.ones(2, 3, 3, requires_grad=True),
    )
    einsum = _op(trace, "einsum")
    descriptor = einsum.receptive_field._descriptor()  # type: ignore[union-attr]

    assert descriptor.axes is not None
    assert all(axis.kind == "full" for axis in descriptor.axes)
    projective = solve_projective(trace, trace.output_ops)  # type: ignore[union-attr]
    assert all(state.axes is not None for state in projective.states.values())


class _PointwiseChain(nn.Module):
    """Convolution followed by formerly unsupported pointwise activations."""

    def __init__(self) -> None:
        """Create the pointwise regression model."""

        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        _fill_convolutions(self)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply relu6, hardswish, and clamp pointwise.

        Parameters
        ----------
        value:
            Input image batch.

        Returns
        -------
        torch.Tensor
            Pointwise-transformed convolution result.
        """

        value = functional.relu6(self.conv(value))
        value = functional.hardswish(value)
        return torch.clamp(value, min=0.1, max=10.0)


def test_missing_activations_pass_through_exactly() -> None:
    """Keep relu6, hardswish, and clamp transparent to receptive-field geometry."""

    trace = _trace(_PointwiseChain(), torch.ones(1, 1, 9, 9, requires_grad=True))
    output = _op(trace, "clamp")
    descriptor = output.receptive_field._descriptor()  # type: ignore[union-attr]

    assert descriptor.status is ReceptiveFieldStatus.EXACT
    assert descriptor.size == (3, 3)
    result = output.receptive_field.check((0, 0, 4, 4))  # type: ignore[union-attr]
    assert result.status is ReceptiveFieldValidationStatus.PASS


def test_constant_padding_is_an_exact_shift() -> None:
    """Map constant padding by its exact left-offset shift."""

    class Padding(nn.Module):
        """Small asymmetric constant-padding fixture."""

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Pad height and width asymmetrically.

            Parameters
            ----------
            value:
                Input image batch.

            Returns
            -------
            torch.Tensor
                Padded tensor.
            """

            return functional.pad(value, (1, 2, 2, 1), mode="constant")

    trace = _trace(Padding(), torch.ones(1, 1, 5, 5, requires_grad=True))
    padding = _op(trace, "pad")
    descriptor = padding.receptive_field._descriptor()  # type: ignore[union-attr]

    assert descriptor.status is ReceptiveFieldStatus.EXACT
    assert descriptor.center0 == (-2, -1)
    result = padding.receptive_field.check((0, 0, 4, 3))  # type: ignore[union-attr]
    assert result.status is ReceptiveFieldValidationStatus.PASS


def test_deliberately_wrong_window_fails_tripwire() -> None:
    """Prove cross-validation rejects an undersized convolution rule."""

    @_rules.register_rf_rule("conv2d", replace=True)
    def undersized_convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Return a deliberately false 1x1 rule for a real 3x3 convolution.

        Parameters
        ----------
        context:
            Captured convolution rule context.

        Returns
        -------
        _RuleResult
            Deliberately undersized window rule.
        """

        return context.window(kernel=(1, 1))

    model = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    _fill_convolutions(model)
    trace = _trace(model, torch.ones(1, 1, 5, 5, requires_grad=True))
    convolution = _op(trace, "conv2d")
    result = convolution.receptive_field.check((0, 0, 2, 2))  # type: ignore[union-attr]

    assert result.status is ReceptiveFieldValidationStatus.FAIL
    assert result.n_violations == 8
