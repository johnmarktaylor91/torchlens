"""Oracle tests for the three r22 attribution wrong-math fixes.

- W2A3-01: repeated references to one input tensor must run the user's actual
  function (identity topology preserved) and report the shared tensor's FULL
  gradient at every occurrence slot.
- W2A3-02: a module fired N times must contribute through EVERY firing; the
  reported layer attribution is the honest per-firing total, never
  last-write-wins.
- W2A3-03: Grad-CAM must upsample onto the grid of the input that actually
  feeds the target layer, and refuse ambiguous or absent spatial feeders.

Each oracle has an executed true-value ground truth; reverting the fixes in
``torchlens/attribution/_core.py`` / ``_layer.py`` turns these tests red.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import torchlens.attribution as attribution


class IdentityBranchModel(nn.Module):
    """Model whose function depends on input object identity.

    ``model(t, t)`` computes ``(t * 3).sum()``; distinct objects compute
    ``(a + 100 b).sum()``. The recorded branch witnesses which function the
    attribution forward actually ran.
    """

    def __init__(self) -> None:
        """Initialize the branch recorder."""

        super().__init__()
        self.taken_branch: str | None = None

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """Compute the identity-sensitive scalar output."""

        if a is b:
            self.taken_branch = "aliased"
            return (a * 3.0).sum().reshape(1)
        self.taken_branch = "split"
        return (a * 1.0 + b * 100.0).sum().reshape(1)


class NestedIdentityBranchModel(nn.Module):
    """Identity-sensitive model receiving both tensors inside one container."""

    def forward(self, pair: list[Tensor]) -> Tensor:
        """Compute the identity-sensitive scalar output from a list input."""

        a, b = pair
        if a is b:
            return (a * 3.0).sum().reshape(1)
        return (a * 1.0 + b * 100.0).sum().reshape(1)


class SharedReluModel(nn.Module):
    """Model that fires one shared ReLU ``n_calls`` times on the same input."""

    def __init__(self, n_calls: int) -> None:
        """Initialize the shared module and call count."""

        super().__init__()
        self.shared = nn.ReLU()
        self.n_calls = n_calls

    def forward(self, x: Tensor) -> Tensor:
        """Sum ``n_calls`` independent shared-ReLU firings of ``x``."""

        total = torch.zeros_like(x)
        for _ in range(self.n_calls):
            total = total + self.shared(x)
        return total.sum().reshape(1)


class SharedLinearModel(nn.Module):
    """Graph-cutting shared Linear fired twice: ``f(x) + f(2x)`` with ``f(u) = 2u``."""

    def __init__(self) -> None:
        """Initialize the deterministic shared linear layer."""

        super().__init__()
        self.f = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.f.weight.fill_(2.0)

    def forward(self, x: Tensor) -> Tensor:
        """Compute ``f(x) + f(2x) = 6x``."""

        return self.f(x) + self.f(2.0 * x)


class SharedIdentitySameTensorModel(nn.Module):
    """Shared ``nn.Identity`` fired twice on the SAME tensor object."""

    def __init__(self) -> None:
        """Initialize the shared identity module."""

        super().__init__()
        self.ident = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Fire the identity twice on one tensor and consume both results."""

        shared_value = x * 2.0
        a = self.ident(shared_value)
        b = self.ident(shared_value)
        return (a + b).sum().reshape(1)


class ConvOnSecondInputModel(nn.Module):
    """Conv target layer consuming ONLY the second input; the first is a decoy."""

    def __init__(self) -> None:
        """Initialize the target convolution."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)

    def forward(self, decoy: Tensor, real: Tensor) -> Tensor:
        """Run the convolution on ``real`` only."""

        del decoy
        return self.conv(real).mean(dim=(2, 3))


class ConvOnFirstInputModel(nn.Module):
    """Conv target layer consuming ONLY the first input; the second is a decoy."""

    def __init__(self) -> None:
        """Initialize the target convolution."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)

    def forward(self, real: Tensor, decoy: Tensor) -> Tensor:
        """Run the convolution on ``real`` only."""

        del decoy
        return self.conv(real).mean(dim=(2, 3))


class HeterogeneousDoubleFeedModel(nn.Module):
    """Conv target layer fed by TWO spatial inputs with different grids."""

    def __init__(self) -> None:
        """Initialize the target convolution."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """Feed a resized ``a`` plus ``b`` into the convolution."""

        resized = F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(resized + b).mean(dim=(2, 3))


class SameGridDoubleFeedModel(nn.Module):
    """Conv target layer fed by two spatial inputs sharing one grid."""

    def __init__(self) -> None:
        """Initialize the target convolution."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """Feed the sum of the two same-grid inputs into the convolution."""

        return self.conv(a + b).mean(dim=(2, 3))


class ParameterFedConvModel(nn.Module):
    """Conv target layer fed only by a parameter; the spatial input is a decoy."""

    def __init__(self) -> None:
        """Initialize the parameter seed and target convolution."""

        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, 5, 4))
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)

    def forward(self, decoy: Tensor) -> Tensor:
        """Run the convolution on the parameter only."""

        del decoy
        return self.conv(self.seed).mean(dim=(2, 3))


class SharedConvModel(nn.Module):
    """Conv target layer fired twice, for the grad_cam multi-fire refusal."""

    def __init__(self) -> None:
        """Initialize the shared convolution."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Fire the shared convolution twice on ``x``."""

        return (self.conv(x) + self.conv(x)).mean(dim=(2, 3))


# ---------------------------------------------------------------------------
# W2A3-01: repeated-reference inputs
# ---------------------------------------------------------------------------


def test_repeated_reference_runs_the_users_function_and_reports_full_gradient() -> None:
    """``model(t, t)`` attribution takes the aliased branch; each slot reports 3."""

    model = IdentityBranchModel()
    t = torch.ones(3)
    seen_outputs: list[Tensor] = []

    def target(output: Tensor) -> Tensor:
        seen_outputs.append(output.detach().clone())
        return output.sum()

    result = attribution.saliency(model, (t, t), target=target)

    assert model.taken_branch == "aliased"
    with torch.no_grad():
        torch.testing.assert_close(seen_outputs[0], model(t, t))
    slot_a, slot_b = result.values
    torch.testing.assert_close(slot_a, torch.full((3,), 3.0))
    torch.testing.assert_close(slot_b, torch.full((3,), 3.0))


def test_repeated_reference_clone_control_selects_the_split_branch() -> None:
    """A ``clone()`` occurrence is a DIFFERENT object and must take the 1/100 branch."""

    model = IdentityBranchModel()
    t = torch.ones(3)

    result = attribution.saliency(model, (t, t.clone()), target=0)

    assert model.taken_branch == "split"
    slot_a, slot_b = result.values
    torch.testing.assert_close(slot_a, torch.full((3,), 1.0))
    torch.testing.assert_close(slot_b, torch.full((3,), 100.0))


def test_repeated_reference_input_x_grad_reports_full_gradient_per_slot() -> None:
    """input_x_grad on the aliased branch reports full-gradient-times-input per slot."""

    model = IdentityBranchModel()
    t = torch.ones(3)

    result = attribution.input_x_grad(model, (t, t), target=0)

    slot_a, slot_b = result.values
    torch.testing.assert_close(slot_a, torch.full((3,), 3.0))
    torch.testing.assert_close(slot_b, torch.full((3,), 3.0))


def test_repeated_reference_integrated_gradients_stays_on_the_aliased_path() -> None:
    """IG path points preserve identity; the piecewise-linear IG value is 3 per slot."""

    model = IdentityBranchModel()
    t = torch.ones(3)

    result = attribution.integrated_gradients(model, (t, t), target=0, n_steps=8)

    slot_a, slot_b = result.values
    torch.testing.assert_close(slot_a, torch.full((3,), 3.0))
    torch.testing.assert_close(slot_b, torch.full((3,), 3.0))


def test_repeated_reference_smoothgrad_shares_one_noise_draw_per_object() -> None:
    """SmoothGrad noises the shared tensor ONCE; the aliased gradient stays 3."""

    model = IdentityBranchModel()
    t = torch.ones(3)

    result = attribution.smoothgrad(model, (t, t), target=0, n_samples=4, seed=7)

    slot_a, slot_b = result.values
    torch.testing.assert_close(slot_a, torch.full((3,), 3.0))
    torch.testing.assert_close(slot_b, torch.full((3,), 3.0))


def test_repeated_reference_across_positional_and_keyword_sites() -> None:
    """One tensor passed positionally AND as a kwarg preserves identity topology."""

    model = IdentityBranchModel()
    t = torch.ones(3)

    result = attribution.saliency(model, (t,), {"b": t}, target=0)

    assert model.taken_branch == "aliased"
    assert isinstance(result.values, dict)
    torch.testing.assert_close(result.values["inputs"][0], torch.full((3,), 3.0))
    torch.testing.assert_close(result.values["input_kwargs"]["b"], torch.full((3,), 3.0))


def test_repeated_reference_inside_nested_container() -> None:
    """Repeated references inside one list container preserve identity topology."""

    model = NestedIdentityBranchModel()
    t = torch.ones(3)

    result = attribution.saliency(model, ([t, t],), target=0)

    torch.testing.assert_close(result.values[0][0], torch.full((3,), 3.0))
    torch.testing.assert_close(result.values[0][1], torch.full((3,), 3.0))


def test_repeated_reference_conflicting_baselines_raise_typed_error() -> None:
    """Two different baselines for one shared tensor are contradictory and refuse."""

    model = IdentityBranchModel()
    t = torch.ones(3)

    with pytest.raises(
        attribution.AttributionError,
        match="repeated references to the same input tensor must match",
    ):
        attribution.integrated_gradients(
            model,
            (t, t),
            target=0,
            baseline=(torch.zeros(3), torch.ones(3)),
        )


# ---------------------------------------------------------------------------
# W2A3-02: reused-module accumulation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_calls", [1, 2, 3])
def test_reused_relu_activation_x_grad_sums_over_every_firing(n_calls: int) -> None:
    """N ReLU firings of value 4 with unit gradients total ``4 * N``; N=3 gives 12."""

    model = SharedReluModel(n_calls)
    x = torch.tensor([[4.0]])

    result = attribution.layer_attribution(
        model, x, target=0, layer="shared", method="activation_x_grad"
    )

    torch.testing.assert_close(result.values.sum(), torch.tensor(4.0 * n_calls))


@pytest.mark.parametrize("n_calls", [1, 3])
def test_reused_relu_grad_method_sums_absolute_gradients_per_firing(n_calls: int) -> None:
    """The ``grad`` method totals per-firing absolute gradients: ``1 * N``."""

    model = SharedReluModel(n_calls)
    x = torch.tensor([[4.0]])

    result = attribution.layer_attribution(model, x, target=0, layer="shared", method="grad")

    torch.testing.assert_close(result.values.sum(), torch.tensor(1.0 * n_calls))


def test_reused_layer_integrated_gradients_satisfies_completeness() -> None:
    """LIG through a graph-cutting layer fired twice sums to ``F(x) - F(0) = 6``."""

    model = SharedLinearModel()
    x = torch.tensor([[1.0]])

    result = attribution.layer_integrated_gradients(model, x, target=0, layer="f", n_steps=16)

    with torch.no_grad():
        target_delta = model(x)[..., 0].sum() - model(torch.zeros_like(x))[..., 0].sum()
    torch.testing.assert_close(result.values.sum(), target_delta)
    torch.testing.assert_close(result.values.sum(), torch.tensor(6.0))


def test_reused_layer_conductance_satisfies_completeness() -> None:
    """Conductance through a graph-cutting layer fired twice sums to 6."""

    model = SharedLinearModel()
    x = torch.tensor([[1.0]])

    result = attribution.layer_conductance(model, x, target=0, layer="f", n_steps=16)

    torch.testing.assert_close(result.values.sum(), torch.tensor(6.0))


def test_identity_module_returning_same_object_is_not_double_counted() -> None:
    """Two Identity firings on ONE tensor object are one autograd node: total 4, not 8."""

    model = SharedIdentitySameTensorModel()
    x = torch.tensor([[1.0]])

    result = attribution.layer_attribution(
        model, x, target=0, layer="ident", method="activation_x_grad"
    )

    torch.testing.assert_close(result.values.sum(), torch.tensor(4.0))


def test_grad_cam_rejects_multi_fire_layer_with_typed_error() -> None:
    """Grad-CAM has no defined semantics for a reused layer and must refuse typed."""

    model = SharedConvModel()
    x = torch.randn(1, 1, 5, 4)

    with pytest.raises(attribution.AttributionError, match="fired 2 times"):
        attribution.grad_cam(model, x, target=0, layer="conv")


# ---------------------------------------------------------------------------
# W2A3-03: multi-input Grad-CAM grid
# ---------------------------------------------------------------------------


def test_grad_cam_upsamples_to_the_feeding_input_grid_second_position() -> None:
    """The CAM lands on the grid of the input that feeds the layer, not the first leaf."""

    model = ConvOnSecondInputModel()
    decoy = torch.randn(1, 1, 9, 7)
    real = torch.randn(1, 1, 5, 4)

    result = attribution.grad_cam(model, (decoy, real), target=0, layer="conv")

    assert tuple(result.values.shape) == (1, 1, 5, 4)


def test_grad_cam_upsamples_to_the_feeding_input_grid_first_position() -> None:
    """Reversed argument order still resolves to the feeding input's grid."""

    model = ConvOnFirstInputModel()
    real = torch.randn(1, 1, 5, 4)
    decoy = torch.randn(1, 1, 9, 7)

    result = attribution.grad_cam(model, (real, decoy), target=0, layer="conv")

    assert tuple(result.values.shape) == (1, 1, 5, 4)


def test_grad_cam_same_grid_double_feed_is_unambiguous() -> None:
    """Two feeding inputs sharing one grid upsample to that grid without error."""

    model = SameGridDoubleFeedModel()
    a = torch.randn(1, 1, 5, 4)
    b = torch.randn(1, 1, 5, 4)

    result = attribution.grad_cam(model, (a, b), target=0, layer="conv")

    assert tuple(result.values.shape) == (1, 1, 5, 4)


def test_grad_cam_heterogeneous_double_feed_raises_typed_ambiguity() -> None:
    """Two feeding inputs with different grids make the upsample target ambiguous."""

    model = HeterogeneousDoubleFeedModel()
    a = torch.randn(1, 1, 9, 7)
    b = torch.randn(1, 1, 5, 4)

    with pytest.raises(attribution.AttributionError, match="different\\s+grids"):
        attribution.grad_cam(model, (a, b), target=0, layer="conv")


def test_grad_cam_refuses_when_no_spatial_input_feeds_the_layer() -> None:
    """A spatial decoy that never reaches the layer is not a coordinate system."""

    model = ParameterFedConvModel()
    decoy = torch.randn(1, 1, 9, 7)

    with pytest.raises(attribution.AttributionError, match="no spatial"):
        attribution.grad_cam(model, decoy, target=0, layer="conv")


def test_grad_cam_without_any_spatial_input_keeps_original_error() -> None:
    """No spatial attributed leaf at all keeps the original error message."""

    model = ParameterFedConvModel()
    decoy = torch.randn(1, 3)

    with pytest.raises(
        attribution.AttributionError,
        match="requires an input tensor with spatial dimensions",
    ):
        attribution.grad_cam(model, decoy, target=0, layer="conv")


def test_autograd_leaf_variable_ids_survives_a_long_chain() -> None:
    """A multi-hop autograd walk must not lose the leaf to a recycled id().

    ``grad_fn.next_functions`` mints a FRESH Python wrapper around the
    underlying autograd node on every access. ``_autograd_leaf_variable_ids``
    walks that graph with a "seen" set; if the set only remembers each
    node's bare ``id()`` instead of the node object itself, nothing keeps a
    just-processed wrapper alive once it is popped off the stack, and
    CPython is free to recycle its exact address for the NEXT node minted
    during the same walk. A long enough chain (observed on torch 2.7.1 for
    as few as 3 hops: Conv2d -> ReLU -> AvgPool2d) then has a later node's
    address alias an earlier, already-"seen" one, which makes the walk
    falsely stop before ever reaching the real input leaf's
    ``AccumulateGrad`` node -- silently losing dependency-proven feeders.
    """

    from torchlens.attribution._layer import _autograd_leaf_variable_ids

    leaf = torch.randn(1, 2, 4, 4).requires_grad_(True)
    value = leaf
    # Chain enough distinct autograd nodes that an id()-only "seen" set has
    # room to alias a freed wrapper's address against a later one.
    for _ in range(8):
        value = torch.relu(value)
        value = value * 1.0
    activation = value

    found = _autograd_leaf_variable_ids((activation,))

    assert id(leaf) in found
