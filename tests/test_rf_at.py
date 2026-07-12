"""Brute-force phase oracles for per-unit receptive-field queries."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from fractions import Fraction
from functools import partial

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _engine, _query, _rules
from torchlens.receptive_field._rules import ReceptiveFieldRuleContext, _RuleResult
from torchlens.receptive_field._types import ReceptiveFieldStatus


@pytest.fixture(autouse=True)
def isolated_rule_registry() -> Iterator[None]:
    """Restore the process-global RF registry after every phase oracle."""

    saved_rules = dict(_rules._RF_RULES)
    saved_epoch = _rules._RF_RULES_EPOCH
    _rules._RF_RULES.clear()
    _rules._RF_RULES_EPOCH += 1
    yield
    _rules._RF_RULES.clear()
    _rules._RF_RULES.update(saved_rules)
    _rules._RF_RULES_EPOCH = saved_epoch


def _tuple(value: object, rank: int) -> tuple[int, ...]:
    """Normalize captured scalar/tuple convolution parameters.

    Parameters
    ----------
    value:
        Captured parameter.
    rank:
        Spatial rank.

    Returns
    -------
    tuple[int, ...]
        One integer per spatial axis.
    """

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(int(item) for item in value)
    else:
        result = (int(value),) * rank
    assert len(result) == rank
    return result


def _register_query_rules() -> None:
    """Register the focused conv, pool, and residue-exact convT test pack."""

    @_rules.register_rf_rule("conv1d", "conv2d")
    def convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Emit an exact standard-convolution window."""

        raw_kernel = context.cfg("kernel_size")
        rank = len(context.out_shape) - 2
        kernel = _tuple(raw_kernel, rank)
        return context.window(
            kernel=kernel,
            stride=context.cfg("stride", (1,) * rank),
            padding=context.cfg("padding", (0,) * rank),
            dilation=context.cfg("dilation", (1,) * rank),
        )

    @_rules.register_rf_rule("max_pool1d", "max_pool2d")
    def pooling(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Emit an exact max-pool window with PyTorch's stride default."""

        rank = len(context.out_shape) - 2
        kernel = _tuple(context.cfg("kernel_size"), rank)
        return context.window(
            kernel=kernel,
            stride=context.cfg("stride", kernel),
            padding=context.cfg("padding", (0,) * rank),
            dilation=context.cfg("dilation", (1,) * rank),
        )

    @_rules.register_rf_rule("relu")
    def relu(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Emit an exact pointwise identity."""

        return context.passthrough()

    @_rules.register_rf_rule("conv_transpose1d")
    def transposed_convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Emit the rational descriptor envelope and residue-exact unit callback."""

        rank = len(context.out_shape) - 2
        kernel = _tuple(context.cfg("kernel_size"), rank)
        stride = _tuple(context.cfg("stride", (1,) * rank), rank)
        padding = _tuple(context.cfg("padding", (0,) * rank), rank)
        dilation = _tuple(context.cfg("dilation", (1,) * rank), rank)
        input_extent = tuple(int(value) for value in context.in_shapes[0][-rank:])
        edges = tuple(
            (
                (Fraction(1, stride_value), Fraction(pad - dil * (size - 1), stride_value)),
                (Fraction(1, stride_value), Fraction(pad, stride_value)),
            )
            for size, stride_value, pad, dil in zip(kernel, stride, padding, dilation, strict=True)
        )
        callback = partial(
            _query.map_transposed_convolution_index_set,
            kernel=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            input_extent=input_extent,
        )
        return context.window_edges(
            edges,
            exact=all(value == 1 for value in stride),
            map_index_set=callback,
        )


def _last_op(trace: object, func_name: str) -> object:
    """Return the final captured operation with a normalized function name."""

    matches = [op for op in trace.layer_list if op.func_name == func_name]  # type: ignore[attr-defined]
    assert matches
    return matches[-1]


def _box(trace: object, target: object, unit: tuple[int, ...], *, clip: bool = True) -> object:
    """Call the T4 core for one captured target and unit."""

    return _query.box_for_unit(_engine.solve(trace), target, unit, clip=clip)  # type: ignore[arg-type]


def _boolean_support_1d(model: nn.Module, input_extent: int) -> tuple[frozenset[int], ...]:
    """Build real-op Boolean connectivity by perturbing every input basis index."""

    basis = torch.eye(input_extent).reshape(input_extent, 1, input_extent)
    with torch.no_grad():
        output = model(basis)[:, 0, :]
    return tuple(
        frozenset(torch.nonzero(output[:, output_index], as_tuple=False).flatten().tolist())
        for output_index in range(output.shape[-1])
    )


def _assert_boxes_equal_boolean_support(
    trace: object, target: object, support: tuple[frozenset[int], ...]
) -> None:
    """Require every exact per-unit hull and empty flag to match real connectivity."""

    for output_index, expected in enumerate(support):
        box = _box(trace, target, (output_index,))
        spatial = box.axes[-1]
        assert box.exact
        assert box.status is ReceptiveFieldStatus.EXACT
        assert box.empty is (not expected)
        if expected:
            assert (spatial.index_start, spatial.index_stop) == (
                min(expected),
                max(expected) + 1,
            )
        else:
            assert (spatial.index_start, spatial.index_stop) == (None, None)


def test_progression_budget_collapse_is_explicitly_inexact() -> None:
    """Collapse more than sixteen residue runs to a visibly inexact dense hull."""

    values = [0]
    for index in range(40):
        values.append(values[-1] + (1 if index % 2 else 2))
    index_set = _query._IndexSet.from_values(values)

    assert not index_set.exact
    assert len(index_set.progressions) == 1
    assert (index_set.minimum, index_set.maximum) == (values[0], values[-1])


@pytest.mark.parametrize("kernel", [1, 2, 3])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("output_padding", [0, 1])
def test_transposed_convolution_every_unit_matches_boolean_connectivity(
    kernel: int, stride: int, dilation: int, output_padding: int
) -> None:
    """Prove every convT exact claim across small kernel/stride/dilation phases."""

    if output_padding >= max(stride, dilation):
        pytest.skip("PyTorch requires output_padding below stride or dilation")
    _register_query_rules()
    model = nn.ConvTranspose1d(
        1,
        1,
        kernel,
        stride=stride,
        padding=kernel // 2,
        output_padding=output_padding,
        dilation=dilation,
        bias=False,
    )
    nn.init.ones_(model.weight)
    support = _boolean_support_1d(model, 5)
    trace = tl.trace(model, torch.ones(1, 1, 5))
    _assert_boxes_equal_boolean_support(trace, _last_op(trace, "conv_transpose1d"), support)


@pytest.mark.parametrize(
    ("kernel", "stride", "dilation", "padding", "output_index"),
    [
        (2, 2, 2, 0, 1),
        (1, 3, 1, 0, 1),
        (1, 4, 1, 0, 2),
    ],
)
def test_transposed_convolution_empty_phase_goldens(
    kernel: int, stride: int, dilation: int, padding: int, output_index: int
) -> None:
    """Lock unreachable parity, stride-over-kernel, and gcd-insufficient empties."""

    _register_query_rules()
    model = nn.ConvTranspose1d(
        1,
        1,
        kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False,
    )
    nn.init.ones_(model.weight)
    support = _boolean_support_1d(model, 4)
    assert not support[output_index]
    trace = tl.trace(model, torch.ones(1, 1, 4))
    box = _box(trace, _last_op(trace, "conv_transpose1d"), (output_index,))
    assert box.exact and box.empty
    assert box.axes[-1].index_start is None


@pytest.mark.parametrize(
    "layers",
    [
        ((2, 2, 0, 1, 0), (3, 2, 1, 1, 1)),
        ((3, 2, 1, 1, 0), (2, 3, 0, 1, 0)),
        ((2, 2, 0, 2, 0), (3, 2, 1, 1, 1), (2, 2, 0, 1, 0)),
    ],
)
def test_stacked_transposed_convolutions_preserve_exact_residue_sets(
    layers: tuple[tuple[int, int, int, int, int], ...],
) -> None:
    """Match two- and three-layer decoder hulls across multiple phase offsets."""

    _register_query_rules()
    modules: list[nn.Module] = []
    for kernel, stride, padding, dilation, output_padding in layers:
        layer = nn.ConvTranspose1d(
            1,
            1,
            kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
            bias=False,
        )
        nn.init.ones_(layer.weight)
        modules.append(layer)
    model = nn.Sequential(*modules)
    support = _boolean_support_1d(model, 4)
    trace = tl.trace(model, torch.ones(1, 1, 4))
    _assert_boxes_equal_boolean_support(trace, _last_op(trace, "conv_transpose1d"), support)


class _ConvPool2d(nn.Module):
    """Small 2-D stack with a hand-computable per-unit support range."""

    def __init__(self) -> None:
        """Create positive-weight convolution and max-pool layers."""

        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, stride=2)
        nn.init.ones_(self.conv.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply convolution then pooling."""

        return self.pool(self.conv(inputs))


def test_conv_pool_2d_box_matches_hand_range_and_boolean_oracle() -> None:
    """Match a hand-derived 2-D pixel range and every real-op Boolean connection."""

    _register_query_rules()
    model = _ConvPool2d()
    trace = tl.trace(model, torch.ones(1, 1, 12, 12))
    target = _last_op(trace, "max_pool2d")
    box = _box(trace, target, (1, 1))
    assert box.exact
    assert [(axis.index_start, axis.index_stop) for axis in box.axes[-2:]] == [(3, 8), (3, 8)]

    basis = torch.eye(12 * 12).reshape(12 * 12, 1, 12, 12)
    with torch.no_grad():
        output = model(basis)[:, 0, 1, 1]
    indices = torch.nonzero(output, as_tuple=False).flatten()
    rows = indices // 12
    columns = indices % 12
    assert (int(rows.min()), int(rows.max()) + 1) == (3, 8)
    assert (int(columns.min()), int(columns.max()) + 1) == (3, 8)


def test_pointwise_axes_are_honest_and_clip_is_optional() -> None:
    """Keep pointwise coordinates unset and expose pre-clip border geometry."""

    _register_query_rules()
    model = nn.Conv1d(1, 1, 3, padding=1, bias=False)
    nn.init.ones_(model.weight)
    support = _boolean_support_1d(model, 5)
    trace = tl.trace(model, torch.ones(2, 1, 5))
    target = _last_op(trace, "conv1d")
    clipped = _box(trace, target, (0,))
    unclipped = _box(trace, target, (0,), clip=False)

    assert support[0] == frozenset({0, 1})
    assert clipped.exact and unclipped.exact
    assert clipped.axes[0].kind == "pointwise"
    assert clipped.axes[0].index_start is None
    assert clipped.axes[1].kind == "full"
    assert (clipped.axes[-1].index_start, clipped.axes[-1].index_stop) == (-1, 2)
    assert (clipped.axes[-1].clipped_start, clipped.axes[-1].clipped_stop) == (0, 2)
    assert clipped.clipped
    assert (unclipped.axes[-1].clipped_start, unclipped.axes[-1].clipped_stop) == (-1, 2)
    assert not unclipped.clipped


@pytest.mark.parametrize(
    ("mode", "align_corners", "scale_factor", "recompute"),
    [
        ("linear", False, None, None),
        ("linear", True, None, None),
        ("nearest", None, None, None),
        ("nearest-exact", None, None, None),
        ("linear", False, 1.5, False),
        ("linear", False, 1.5, True),
    ],
)
def test_interpolation_callback_exact_sets_match_real_operator(
    mode: str,
    align_corners: bool | None,
    scale_factor: float | None,
    recompute: bool | None,
) -> None:
    """Back every interpolation exact set with all-unit Boolean connectivity."""

    input_extent = 5
    basis = torch.eye(input_extent).reshape(input_extent, 1, input_extent)
    kwargs: dict[str, object] = {"mode": mode}
    if mode in {"linear", "bilinear", "trilinear", "bicubic"}:
        kwargs["align_corners"] = align_corners
    if scale_factor is None:
        kwargs["size"] = 7
    else:
        kwargs["scale_factor"] = scale_factor
        kwargs["recompute_scale_factor"] = recompute
    output = functional.interpolate(basis, **kwargs)
    output_extent = int(output.shape[-1])
    for output_index in range(output_extent):
        expected = frozenset(
            torch.nonzero(output[:, 0, output_index], as_tuple=False).flatten().tolist()
        )
        mapped, exact = _query.map_interpolation_index_set(
            0,
            _query._IndexSet.singleton(output_index),
            mode=mode,
            input_extent=input_extent,
            output_extent=output_extent,
            align_corners=align_corners,
            scale_factor=scale_factor,
            recompute_scale_factor=recompute,
        )
        assert exact
        assert frozenset(mapped.values()) == expected


def test_bicubic_callback_exact_sets_match_real_operator() -> None:
    """Prove bicubic zero-weight and border tap handling on both spatial axes."""

    extent = 5
    basis = torch.eye(extent * extent).reshape(extent * extent, 1, extent, extent)
    output = functional.interpolate(basis, size=(7, 7), mode="bicubic", align_corners=False)
    for row in range(7):
        for column in range(7):
            support = torch.nonzero(output[:, 0, row, column], as_tuple=False).flatten()
            expected_rows = frozenset((support // extent).tolist())
            expected_columns = frozenset((support % extent).tolist())
            for axis, index, expected in ((0, row, expected_rows), (1, column, expected_columns)):
                mapped, exact = _query.map_interpolation_index_set(
                    axis,
                    _query._IndexSet.singleton(index),
                    mode="bicubic",
                    input_extent=(extent, extent),
                    output_extent=(7, 7),
                    align_corners=False,
                )
                assert exact
                assert frozenset(mapped.values()) == expected


def test_adaptive_pool_callback_exact_sets_match_real_operator() -> None:
    """Back adaptive-bin exactness with every real average-pool output unit."""

    input_extent = 7
    output_extent = 4

    @_rules.register_rf_rule("adaptive_avg_pool1d")
    def adaptive_pool(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Emit an envelope descriptor with an exact interval callback."""

        def map_interval(axis: int, interval: tuple[int, int]) -> tuple[_query._IndexSet, bool]:
            """Map every output in one inclusive interval to its adaptive bin."""

            return _query.map_adaptive_pool_index_set(
                axis,
                _query._IndexSet.interval(*interval),
                input_extent=input_extent,
                output_extent=output_extent,
            )

        edges = (
            (
                (Fraction(input_extent, output_extent), -1),
                (Fraction(input_extent, output_extent), 1),
            ),
        )
        return context.window_edges(edges, exact=False, map_interval=map_interval)

    model = nn.AdaptiveAvgPool1d(output_extent)
    basis = torch.eye(input_extent).reshape(input_extent, 1, input_extent)
    output = model(basis)
    trace = tl.trace(model, torch.ones(1, 1, input_extent))
    target = _last_op(trace, "adaptive_avg_pool1d")
    for output_index in range(output_extent):
        expected = frozenset(
            torch.nonzero(output[:, 0, output_index], as_tuple=False).flatten().tolist()
        )
        mapped, exact = _query.map_adaptive_pool_index_set(
            0,
            _query._IndexSet.singleton(output_index),
            input_extent=input_extent,
            output_extent=output_extent,
        )
        assert exact
        assert frozenset(mapped.values()) == expected
        box = _box(trace, target, (output_index,))
        assert box.exact
        assert (box.axes[-1].index_start, box.axes[-1].index_stop) == (
            min(expected),
            max(expected) + 1,
        )
