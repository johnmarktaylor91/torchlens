"""End-to-end integration gates for receptive-field analysis."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
import importlib
import subprocess
import sys
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _rules
from torchlens.receptive_field._rules import ReceptiveFieldRuleContext, _RuleResult


_BUILTIN_RULE_MODULES = (
    "attention",
    "conv_pool",
    "elementwise",
    "interpolation",
    "linear",
    "norms",
    "sequence",
    "transforms",
)
_BUILTIN_RULE_PACK: dict[str, object] | None = None


@pytest.fixture(autouse=True)
def built_in_rule_pack() -> Iterator[None]:
    """Install the built-in rules while preserving process-global registry state."""

    global _BUILTIN_RULE_PACK
    original = dict(_rules._RF_RULES)
    original_epoch = _rules._RF_RULES_EPOCH
    _rules._RF_RULES.clear()
    if _BUILTIN_RULE_PACK is None:
        importlib.import_module("torchlens.receptive_field.rules")
        if not _rules._RF_RULES:
            for module_name in _BUILTIN_RULE_MODULES:
                module = importlib.import_module(f"torchlens.receptive_field.rules.{module_name}")
                importlib.reload(module)
        _BUILTIN_RULE_PACK = dict(_rules._RF_RULES)
    else:
        _rules._RF_RULES.update(_BUILTIN_RULE_PACK)
    try:
        yield
    finally:
        _rules._RF_RULES.clear()
        _rules._RF_RULES.update(original)
        _rules._RF_RULES_EPOCH = original_epoch


def _register_exact_residual_rules() -> None:
    """Use exact pointwise rules for the residual operations exercised here."""

    @tl.receptive_field.register_rf_rule("relu", "iadd", replace=True)
    def residual_passthrough(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Preserve geometry through ReLU and torchvision's in-place residual add."""

        return context.passthrough()

    @tl.receptive_field.register_rf_rule("add", replace=True)
    def out_of_place_add(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Merge the branches of the hand-built residual fixtures exactly."""

        return context.passthrough()


def _window_step(
    size: int, jump: int, *, kernel: int, stride: int = 1, dilation: int = 1
) -> tuple[int, int]:
    """Apply the hand-derived receptive-field recurrence for one windowed operation."""

    return size + (kernel - 1) * dilation * jump, jump * stride


def _resnet18_merge_goldens() -> tuple[tuple[int, int], ...]:
    """Derive ResNet-18 merge ``(size, jump)`` values from its branch graph."""

    size, jump = _window_step(1, 1, kernel=7, stride=2)
    size, jump = _window_step(size, jump, kernel=3, stride=2)
    goldens: list[tuple[int, int]] = []
    for stage, blocks in enumerate((2, 2, 2, 2)):
        for block in range(blocks):
            stride = 2 if stage > 0 and block == 0 else 1
            main_size, main_jump = _window_step(size, jump, kernel=3, stride=stride)
            main_size, main_jump = _window_step(main_size, main_jump, kernel=3)
            skip_size, skip_jump = _window_step(size, jump, kernel=1, stride=stride)
            assert main_jump == skip_jump
            size, jump = max(main_size, skip_size), main_jump
            goldens.append((size, jump))
    return tuple(goldens)


class _StrideTwoBasicBlock(nn.Module):
    """Small projection BasicBlock whose branches have different RF sizes."""

    def __init__(self) -> None:
        """Create a two-convolution main branch and stride-two projection."""

        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.projection = nn.Conv2d(3, 4, 1, stride=2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the activated union of the main and projection branches."""

        main = torch.relu(self.conv1(inputs))
        main = self.conv2(main)
        return torch.relu(main + self.projection(inputs))


class _ResidualOracle(nn.Module):
    """Positive linear residual graph for the independent support oracle."""

    def __init__(self) -> None:
        """Create spatial and pointwise branches."""

        super().__init__()
        self.spatial = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.pointwise = nn.Conv2d(1, 1, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the sum of independently connected branches."""

        return self.spatial(inputs) + self.pointwise(inputs)


class _TwoInputRF(nn.Module):
    """Two-input graph with a distinct receptive field for each input."""

    def __init__(self) -> None:
        """Create spatial and pointwise input branches."""

        super().__init__()
        self.left = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.right = nn.Conv2d(1, 1, 1, bias=False)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Merge the two model-input branches."""

        return self.left(left) + self.right(right)


def _last_op(trace: Any, func_name: str) -> Any:
    """Return the last operation with an exact captured function name."""

    matches = [op for op in trace.layer_list if op.func_name == func_name]
    assert matches
    return matches[-1]


def _fill_positive(model: nn.Module) -> None:
    """Set every model parameter to one for exact non-cancelling support."""

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0)


def _gradient_spatial_hull(
    target: Any, input_op: Any, unit: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Compute an independent autograd-support hull for one spatial output unit."""

    output_index = (0, 0, *unit)
    gradient = torch.autograd.grad(
        target.out[output_index], input_op.out, retain_graph=True, allow_unused=False
    )[0]
    support = gradient.detach().abs().sum(dim=(0, 1)).ne(0)
    coordinates = torch.nonzero(support, as_tuple=False)
    assert coordinates.numel()
    return tuple(
        (int(coordinates[:, axis].min()), int(coordinates[:, axis].max()) + 1) for axis in range(2)
    )  # type: ignore[return-value]


def test_lazy_namespace_and_public_surface_work_end_to_end() -> None:
    """Resolve the lazy namespace, public names, entity view, and trace table."""

    command = (
        "import sys; import torchlens as tl; "
        "assert 'torchlens.receptive_field' not in sys.modules; "
        "assert tl.receptive_field.__name__ == 'torchlens.receptive_field'; "
        "assert 'torchlens.receptive_field' in sys.modules; "
        "assert 'receptive_field' not in tl.__all__"
    )
    subprocess.run([sys.executable, "-c", command], check=True)

    public_names = (
        "register_rf_rule",
        "cross_validate",
        "node_spec",
        "ReceptiveField",
        "ReceptiveFieldBox",
        "GradientReceptiveField",
        "ReceptiveFieldView",
        "ReceptiveFieldValidation",
        "ReceptiveFieldProfile",
    )
    assert all(hasattr(tl.receptive_field, name) for name in public_names)
    trace = tl.trace(nn.Conv2d(1, 1, 3), torch.randn(1, 1, 5, 5))
    target = _last_op(trace, "conv2d")
    assert target.receptive_field.size == (3, 3)
    assert not trace.receptive_fields().to_pandas().empty
    if hasattr(target, "projective_field"):
        assert target.projective_field is not None


def test_stride_two_basic_block_uses_branch_union_not_flat_call_order() -> None:
    """Prove the skip merge keeps the branch-derived RF and sampling jump."""

    _register_exact_residual_rules()
    trace = tl.trace(_StrideTwoBasicBlock().eval(), torch.randn(1, 3, 17, 17))
    merge = _last_op(trace, "__add__")
    descriptor = next(iter(merge.receptive_field.per_input.values()))

    main_size, main_jump = _window_step(1, 1, kernel=3, stride=2)
    main_size, main_jump = _window_step(main_size, main_jump, kernel=3)
    skip_size, skip_jump = _window_step(1, 1, kernel=1, stride=2)
    assert (main_size, main_jump) == (7, 2)
    assert (skip_size, skip_jump) == (1, 2)
    assert descriptor.size == (max(main_size, skip_size),) * 2
    assert descriptor.jump == (Fraction(main_jump),) * 2

    # A flat call-order accumulator applies the projection after the main branch,
    # incorrectly doubling the merged sampling jump from 2 to 4.
    _, flat_call_order_jump = _window_step(main_size, main_jump, kernel=1, stride=2)
    assert flat_call_order_jump == 4
    assert descriptor.jump != (Fraction(flat_call_order_jump),) * 2


@pytest.mark.heavy
def test_torchvision_resnet18_merge_geometry_matches_graph_derivation() -> None:
    """Lock all ResNet-18 skip merges to independently derived graph values."""

    torchvision_models = pytest.importorskip("torchvision.models")
    _register_exact_residual_rules()
    model = torchvision_models.resnet18(weights=None).eval()
    trace = tl.trace(
        model,
        torch.randn(1, 3, 64, 64, requires_grad=True),
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    merges = [op for op in trace.layer_list if op.func_name == "__iadd__"]
    expected = _resnet18_merge_goldens()
    observed = []
    for merge in merges:
        descriptor = next(iter(merge.receptive_field.per_input.values()))
        observed.append((descriptor.size[0], int(descriptor.jump[0])))

    assert expected == (
        (27, 4),
        (43, 4),
        (67, 8),
        (99, 8),
        (147, 16),
        (211, 16),
        (307, 32),
        (435, 32),
    )
    assert tuple(observed) == expected
    assert all(earlier[0] < later[0] for earlier, later in zip(observed, observed[1:]))

    validations = tl.receptive_field.cross_validate(
        trace,
        ops=(merges[0], merges[2], merges[-1]),
        units="center",
    )
    assert validations
    assert all(result.passed for result in validations)


@pytest.mark.parametrize(
    "model",
    [
        nn.Conv2d(1, 1, 3, padding=1, bias=False),
        nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=False),
            nn.Conv2d(1, 1, 3, stride=2, padding=1, bias=False),
        ),
        _ResidualOracle(),
    ],
)
def test_geometric_hulls_equal_independent_gradient_support(model: nn.Module) -> None:
    """Match every small-model per-op spatial hull to raw autograd support."""

    _register_exact_residual_rules()
    _fill_positive(model)
    inputs = torch.ones(1, 1, 5, 5, requires_grad=True)
    capture = tl.options.CaptureOptions(backward_ready=True)
    trace = tl.trace(model.eval(), inputs, capture=capture, save_mode="reference")
    input_op = next(op for op in trace.layer_list if op.is_input)
    targets = [op for op in trace.layer_list if op.func_name in {"conv2d", "__add__"}]
    assert targets

    for target in targets:
        height, width = target.shape[-2:]
        for row in range(height):
            for column in range(width):
                box = target.receptive_field.at((row, column), input=input_op)
                geometric = tuple(
                    (int(axis.clipped_start), int(axis.clipped_stop)) for axis in box.axes[-2:]
                )
                oracle = _gradient_spatial_hull(target, input_op, (row, column))
                assert geometric == oracle


def test_multi_input_view_keeps_each_io_role_and_rejects_bare_access() -> None:
    """Expose one descriptor per model input and reject ambiguous convenience fields."""

    _register_exact_residual_rules()
    model = _TwoInputRF().eval()
    left = torch.randn(1, 1, 7, 7)
    right = torch.randn(1, 1, 7, 7)
    trace = tl.trace(model, (left, right))
    merge = _last_op(trace, "__add__")
    view = merge.receptive_field

    assert set(view.per_input) == {"input.left", "input.right"}
    assert view.per_input["input.left"].size == (3, 3)
    assert view.per_input["input.right"].size == (1, 1)
    with pytest.raises(tl.receptive_field.AmbiguousInputError):
        _ = view.status
    frame = trace.receptive_fields().to_pandas()
    merge_rows = frame[frame["output_op"] == merge.label]
    assert set(merge_rows["input_role"]) == {"input.left", "input.right"}


@pytest.mark.backend_tinygrad
def test_non_torch_geometry_works_but_gradient_probe_is_gated() -> None:
    """Keep geometry backend-neutral while rejecting non-torch gradient probes."""

    tinygrad = pytest.importorskip("tinygrad")

    @tl.receptive_field.register_rf_rule("add", replace=True)
    def tinygrad_add(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Treat tinygrad elementwise addition as an exact pointwise operation."""

        return context.passthrough()

    def add_one(value: Any) -> Any:
        """Return a tinygrad tensor after pointwise scalar addition."""

        return value + 1.0

    trace = tl.trace(add_one, tinygrad.Tensor([1.0, 2.0, 3.0]), backend="tinygrad")
    target = _last_op(trace, "add")
    descriptor = next(iter(target.receptive_field.per_input.values()))

    assert descriptor.status is tl.receptive_field.ReceptiveFieldStatus.EXACT
    assert descriptor.layout.axis_kinds == ("pointwise",)
    with pytest.raises(tl.receptive_field.BackendUnsupportedError, match="tinygrad"):
        target.receptive_field.gradient((0,))
