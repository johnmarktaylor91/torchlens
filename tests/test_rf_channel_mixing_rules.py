"""Goldens for module-boundary identity, dense and layer-norm channel-mixing RF rules.

ConvNeXt-style blocks run their MLP on channels-last tensors (``permute`` then
``layer_norm`` and ``linear`` over the last axis) and wrap blocks in identity
modules. These tests pin that the geometric field passes through each of those
ops exactly, and compare every EXACT claim against an independent float64
autodiff support oracle that does not go through TorchLens.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as functional
from support.rf_isolation import preserved_rf_registry
from torch import nn
from torch.autograd import forward_ad

import torchlens as tl
from torchlens._capture.arg_positions import _normalize_func_name
from torchlens.receptive_field import _engine
from torchlens.receptive_field._rules import ReceptiveFieldRuleContext, rules
from torchlens.receptive_field._types import (
    ReceptiveFieldStatus,
    ReceptiveFieldValidationStatus,
)


@pytest.fixture(autouse=True)
def built_in_rule_pack() -> Iterator[None]:
    """Install the built-in RF rules while preserving registry isolation."""

    with preserved_rf_registry(install_builtin=True):
        yield


def _ops(trace: Any, name: str) -> list[Any]:
    """Return the captured operations with one function name, refusing an empty match."""

    key = _normalize_func_name(name)
    found = [
        op
        for op in trace._layer_list
        if op.func_name is not None and _normalize_func_name(op.func_name) == key
    ]
    assert found, f"trace has no {name!r} operation"
    return found


def _box_mask(box: Any) -> torch.Tensor:
    """Materialize a geometric box as a boolean mask over its far endpoint."""

    mask = torch.zeros(tuple(box.input_shape), dtype=torch.bool)
    mask[box.slices()] = True
    return mask


def _receptive_support(model: nn.Module, x: torch.Tensor, unit: tuple[int, int]) -> torch.Tensor:
    """Input support of every output channel at one output position (float64 reverse AD)."""

    x = x.detach().double().requires_grad_(True)
    out = model.double()(x)
    weights = torch.zeros_like(out)
    generator = torch.Generator().manual_seed(3)
    weights[0, :, unit[0], unit[1]] = torch.randn(out.shape[1], generator=generator).double()
    (grad,) = torch.autograd.grad((out * weights).sum(), x)
    model.float()
    return grad != 0


def _projective_support(model: nn.Module, x: torch.Tensor, unit: tuple[int, int]) -> torch.Tensor:
    """Output support of every input channel at one input position (float64 forward AD)."""

    tangent = torch.zeros_like(x, dtype=torch.float64)
    generator = torch.Generator().manual_seed(5)
    tangent[0, :, unit[0], unit[1]] = torch.randn(x.shape[1], generator=generator).double()
    with forward_ad.dual_level():
        out = model.double()(forward_ad.make_dual(x.detach().double(), tangent))
        support = forward_ad.unpack_dual(out).tangent != 0
    model.float()
    return support


class _ChannelsLastMixer(nn.Module):
    """Conv, then a channels-last norm and MLP, an identity, and a conv."""

    def __init__(self, channels: int = 4, hidden: int = 6) -> None:
        """Create the block's layers."""

        super().__init__()
        self.conv_in = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.LayerNorm(channels)
        self.up = nn.Linear(channels, hidden)
        self.down = nn.Linear(hidden, channels)
        self.skip = nn.Identity()
        self.conv_out = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        y = self.conv_in(x).permute(0, 2, 3, 1)
        y = self.down(functional.gelu(self.up(self.norm(y)))).permute(0, 3, 1, 2)
        return self.conv_out(self.skip(y))


class _ReturnsInput(nn.Module):
    """A module that returns its input tensor unchanged."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input."""

        return x


def test_identity_ops_pass_the_field_through_exactly() -> None:
    """nn.Identity and a module returning its input keep a conv stack EXACT."""

    model = nn.Sequential(
        nn.Conv2d(2, 3, 3, padding=1), nn.Identity(), _ReturnsInput(), nn.Conv2d(3, 3, 3, padding=1)
    ).eval()
    trace = tl.trace(model, torch.randn(1, 2, 9, 9))
    identities = _ops(trace, "identity")
    assert len(identities) == 2
    for op in identities:
        result, _ = _engine._rule_result(op)
        assert result.kind == "passthrough"
    last = _ops(trace, "conv2d")[-1]
    descriptor = last.receptive_field._descriptor()
    assert descriptor.status is ReceptiveFieldStatus.EXACT
    assert descriptor.size == (5, 5)


def test_identity_rule_refuses_anything_but_a_one_input_pass_through() -> None:
    """A parentless or reshaping op named identity is refused as unknown, never guessed."""

    rule = rules()["identity"]
    creation = SimpleNamespace(
        parents=[], input_shapes=[], shape=(3, 3), func_name="identity", func_config={}
    )
    assert rule(ReceptiveFieldRuleContext(creation)).kind == "unknown"  # type: ignore[arg-type]
    reshaping = SimpleNamespace(
        parents=["a"], input_shapes=[(2, 3)], shape=(3, 2), func_name="identity", func_config={}
    )
    assert rule(ReceptiveFieldRuleContext(reshaping)).kind == "unknown"  # type: ignore[arg-type]


def test_channels_last_norm_and_mlp_rules_are_exact() -> None:
    """Parameter-weight linear and a 4-wide layer_norm are exact over the feature axis."""

    trace = tl.trace(_ChannelsLastMixer().eval(), torch.randn(1, 4, 8, 8))
    for name in ("linear", "layer_norm"):
        for op in _ops(trace, name):
            result, _ = _engine._rule_result(op)
            assert result.kind == "full"
            assert result.values["axes"] == (-1,) or result.values["axes"] == (3,)
            assert result.values["exact"] is True


@pytest.mark.parametrize("unit", [(0, 0), (3, 4), (7, 7)])
def test_channels_last_block_receptive_box_equals_autodiff_support(unit: tuple[int, int]) -> None:
    """The EXACT receptive box equals the float64 reverse-AD support, cell for cell."""

    torch.manual_seed(0)
    model = _ChannelsLastMixer().eval()
    x = torch.randn(1, 4, 8, 8)
    trace = tl.trace(model, x)
    target = _ops(trace, "conv2d")[-1]
    box = target.receptive_field.at(unit)
    assert box.status is ReceptiveFieldStatus.EXACT
    assert torch.equal(_box_mask(box), _receptive_support(model, x, unit))


@pytest.mark.parametrize("unit", [(0, 0), (2, 5), (7, 7)])
def test_channels_last_block_projective_box_equals_autodiff_support(
    unit: tuple[int, int],
) -> None:
    """The EXACT projective box equals the float64 forward-AD support, cell for cell."""

    torch.manual_seed(0)
    model = _ChannelsLastMixer().eval()
    x = torch.randn(1, 4, 8, 8)
    trace = tl.trace(model, x)
    source = trace["input_1"]
    target = _ops(trace, "conv2d")[-1]
    box = source.projective_field.at(unit, target=target)
    assert box.status is ReceptiveFieldStatus.EXACT
    assert torch.equal(_box_mask(box), _projective_support(model, x, unit))


def test_channels_last_block_passes_the_gradient_tripwire() -> None:
    """The zero-tolerance containment check passes on the exact channels-last block."""

    model = _ChannelsLastMixer().eval()
    capture = tl.options.CaptureOptions(backward_ready=True, save_mode="reference")
    trace = tl.trace(model, torch.randn(1, 4, 8, 8, requires_grad=True), capture=capture)
    target = _ops(trace, "conv2d")[-1]
    result = target.receptive_field.check((0, 1, 4, 4))
    assert result.status is ReceptiveFieldValidationStatus.PASS


class _TracedWeightLinear(nn.Module):
    """F.linear whose weight is a model input rather than a parameter."""

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply the traced weight."""

        return functional.linear(x, weight)


def test_traced_linear_weight_is_bounded_by_its_whole_extent() -> None:
    """y[n, o] reads weight row o, so a traced weight never maps axis 0 onto the batch axis."""

    trace = tl.trace(_TracedWeightLinear(), (torch.randn(3, 5), torch.randn(4, 5)))
    op = _ops(trace, "linear")[-1]
    result, _ = _engine._rule_result(op)
    assert result.values["exact"] is False
    descriptors = _engine.lookup(trace, op)
    weight = next(d for d in descriptors.values() if d.input_shape == (4, 5))
    assert weight.status is ReceptiveFieldStatus.UPPER_BOUND
    assert weight.axes is not None
    assert all(axis.kind == "full" for axis in weight.axes)
    data = next(d for d in descriptors.values() if d.input_shape == (3, 5))
    assert data.axes is not None
    assert [axis.kind for axis in data.axes] == ["pointwise", "full"]


@pytest.mark.parametrize(
    ("width", "eps", "exact"),
    [(1, 1e-5, False), (2, 1e-5, True), (2, 0.0, False), (3, 0.0, True)],
)
def test_layer_norm_exactness_follows_the_slice_width(width: int, eps: float, exact: bool) -> None:
    """A one-wide slice reads nothing; a two-wide slice needs eps; three or more are exact."""

    trace = tl.trace(nn.LayerNorm(width, eps=eps), torch.randn(2, 5, width))
    op = _ops(trace, "layer_norm")[-1]
    result, _ = _engine._rule_result(op)
    assert result.values["exact"] is exact
    if width == 2 and eps == 0.0:
        # The analytic derivative is identically zero here (the output is the
        # sign of x0 - x1), but autograd's rounding can leave 1e-16 residue, so
        # there is no float oracle for this cell; the rule's refusal is the claim.
        return
    x = torch.randn(2, 5, width, dtype=torch.float64, requires_grad=True)
    out = functional.layer_norm(x, (width,), eps=eps)
    (grad,) = torch.autograd.grad(out[0, 0, 0], x)
    reads_whole_slice = bool((grad[0, 0] != 0).all())
    assert reads_whole_slice is exact


def test_convnext_block_is_exact_and_matches_autodiff() -> None:
    """A torchvision ConvNeXt block (depthwise 7x7, channels-last MLP, layer scale) is EXACT."""

    convnext = pytest.importorskip("torchvision.models.convnext")
    torch.manual_seed(0)
    block = convnext.CNBlock(8, layer_scale=1.0, stochastic_depth_prob=0.0).eval()
    x = torch.randn(1, 8, 12, 12)
    trace = tl.trace(block, x)
    target = [op for op in trace._layer_list if not op.is_output][-1]
    box = target.receptive_field.at((5, 6))
    assert box.status is ReceptiveFieldStatus.EXACT
    assert torch.equal(_box_mask(box), _receptive_support(block, x, (5, 6)))


def _box_extent(box: Any) -> tuple[int, tuple[int, int], tuple[int, int]]:
    """Channel count and row and column ranges of a 4-d box mask."""

    nonzero = _box_mask(box).nonzero()
    return (
        len(set(nonzero[:, 1].tolist())),
        (int(nonzero[:, 2].min()), int(nonzero[:, 2].max())),
        (int(nonzero[:, 3].min()), int(nonzero[:, 3].max())),
    )


@pytest.mark.heavy
def test_convnext_tiny_stem_to_stage_fields_are_exact() -> None:
    """ConvNeXt-tiny answers stem-to-stage questions geometrically and exactly.

    Expected boxes were checked cell for cell against float64 forward- and
    reverse-mode autodiff supports on the same model (torchvision and timm agree).
    """

    models = pytest.importorskip("torchvision.models")
    torch.manual_seed(0)
    model = models.convnext_tiny(weights=None).eval()
    trace = tl.trace(model, torch.rand(1, 3, 224, 224))
    stem, stage1, stage2 = trace.ops["features.0"], trace.ops["features.3"], trace.ops["features.5"]
    for op in (stem, stage1, stage2):
        assert op.receptive_field.status is ReceptiveFieldStatus.EXACT
    expected = {
        ((28, 28), "stage1"): (192, (0, 27), (0, 27)),
        ((10, 40), "stage1"): (192, (0, 18), (6, 27)),
        ((55, 55), "stage1"): (192, (14, 27), (14, 27)),
        ((28, 28), "stage2"): (384, (0, 13), (0, 13)),
    }
    targets = {"stage1": stage1, "stage2": stage2}
    for (unit, name), extent in expected.items():
        box = stem.projective_field.at(unit, target=targets[name])
        assert box.status is ReceptiveFieldStatus.EXACT
        assert _box_extent(box) == extent
    back = stage1.receptive_field.at((0, 27), source=stem)
    assert back.status is ReceptiveFieldStatus.EXACT
    assert _box_extent(back) == (96, (0, 28), (27, 55))
