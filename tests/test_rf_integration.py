"""End-to-end integration gates for receptive-field analysis."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from support.rf_isolation import preserved_rf_registry
from torch import nn

import torchlens as tl
from torchlens.receptive_field._rules import ReceptiveFieldRuleContext, _RuleResult


@pytest.fixture(autouse=True)
def built_in_rule_pack() -> Iterator[None]:
    """Install the built-in rules while preserving process-global registry state."""

    with preserved_rf_registry(install_builtin=True):
        yield


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


def _last_op(trace: Any, func_name: str, *, field: str = "func_name") -> Any:
    """Return the last operation with an exact captured function or layer-type name.

    ``func_name`` is the backend's raw captured call name, which is NOT
    always the canonical op name: tinygrad ops are reconstructed from the UOp
    DAG and carry the tinygrad API name observed for them (else the lowercase
    UOp op name) as ``func_name``, while the semantic op name (``"add"``,
    ``"mul"``, ...) is on ``type``. Pass ``field="type"`` for those.
    """

    matches = [op for op in trace._layer_list if getattr(op, field) == func_name]
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
        "GradReceptiveField",
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


@pytest.mark.heavy
def test_torchvision_resnet18_merge_geometry_matches_graph_derivation() -> None:
    """Lock all ResNet-18 skip merges to independently derived graph values."""

    torchvision_models = pytest.importorskip("torchvision.models")
    _register_exact_residual_rules()
    model = torchvision_models.resnet18(weights=None).eval()
    trace = tl.trace(
        model,
        torch.randn(1, 3, 64, 64, requires_grad=True),
        capture=tl.options.CaptureOptions(backward_ready=True, save_mode="reference"),
    )
    merges = [op for op in trace._layer_list if op.func_name == "__iadd__"]
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
        _ = view.size
    assert set(view.status) == {"input.left", "input.right"}
    frame = trace.receptive_fields().to_pandas()
    merge_rows = frame[frame["output_op"] == merge.label]
    assert set(merge_rows["input_role"]) == {"input.left", "input.right"}


@pytest.mark.rare
@pytest.mark.backend_tinygrad
def test_non_torch_geometry_and_gradient_probe_both_work() -> None:
    """Keep geometry backend-neutral and probe gradients by cone replay off PyTorch."""

    tinygrad = pytest.importorskip("tinygrad")

    @tl.receptive_field.register_rf_rule("add", replace=True)
    def tinygrad_add(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Treat tinygrad elementwise addition as an exact pointwise operation."""

        return context.passthrough()

    def add_one(value: Any) -> Any:
        """Return a tinygrad tensor after pointwise scalar addition."""

        return value + 1.0

    trace = tl.trace(add_one, tinygrad.Tensor([1.0, 2.0, 3.0]), backend="tinygrad")
    target = _last_op(trace, "add", field="type")
    descriptor = next(iter(target.receptive_field.per_input.values()))

    assert descriptor.status is tl.receptive_field.ReceptiveFieldStatus.EXACT
    assert descriptor.axis_roles.axis_kinds == ("pointwise",)
    (probe,) = target.receptive_field.probe_grad((0,)).values()
    assert probe.signed_grad.tolist() == [1.0, 0.0, 0.0]
