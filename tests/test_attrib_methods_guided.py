"""F06 B6: guided backprop + deconvolution (attrib memo D11-D13).

Strict exact-ReLU coverage at OP level (module, functional, in-place, method,
reused firings), rule semantics against hand-rolled expectations, the
functional-in-place toy regression (the D13 unit row), zero-match teaching
refusal, forward-fidelity tripwire, state purity on success AND forced
failure, and NT composition.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import torchlens.attribution as attribution
from torchlens.attribution import AttributionError

pytestmark = pytest.mark.smoke


class _MixedReluNet(nn.Module):
    """Every exact-ReLU spelling: module, functional, method, in-place.

    The four coverage classes the op-level claim names (memo D11): a module
    ``nn.ReLU`` (dispatching ``F.relu``), a plain ``F.relu``, a method
    ``Tensor.relu()``, and an in-place ``F.relu(..., inplace=True)`` whose
    return value IS used (the covered in-place convention).
    """

    def __init__(self) -> None:
        """Build fixed weights."""

        super().__init__()
        self.module_relu = nn.ReLU()
        self.lin1 = nn.Linear(3, 3, dtype=torch.float64)
        self.lin2 = nn.Linear(3, 3, dtype=torch.float64)
        with torch.no_grad():
            self.lin1.weight.copy_(torch.eye(3, dtype=torch.float64))
            self.lin1.bias.copy_(torch.tensor([0.5, -0.5, 0.0], dtype=torch.float64))
            self.lin2.weight.copy_(-torch.eye(3, dtype=torch.float64))
            self.lin2.bias.copy_(torch.tensor([0.2, 0.2, 0.2], dtype=torch.float64))

    def forward(self, x: Tensor) -> Tensor:
        """Chain all four ReLU spellings."""

        x = self.module_relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = (x - 0.1).relu()
        x = F.relu(x - 0.05, inplace=True)
        return x


class _SingleRelu(nn.Module):
    """One linear layer into one ReLU, for analytic rule checks."""

    def __init__(self, weight: Tensor) -> None:
        """Store the fixed weight."""

        super().__init__()
        self.lin = nn.Linear(3, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.lin.weight.copy_(weight)

    def forward(self, x: Tensor) -> Tensor:
        """Linear, ReLU, then a NEGATIVE head weight to flip gradient signs."""

        return -torch.relu(self.lin(x))


class _FunctionalInPlaceToy(nn.Module):
    """The D13 toy: module ReLUs in a block plus one functional in-place ReLU.

    Mirrors torchvision DenseNet's structure -- every module ReLU lives under
    ``features`` while ``forward`` itself calls ``F.relu(..., inplace=True)``
    that module hooks cannot see. The ``mix`` linear between the block and
    the functional ReLU matters: without linear mixing, the adjacent guided
    module ReLU would re-clamp the sign difference and hide the divergence
    (in DenseNet the convolutions play this role).
    """

    def __init__(self) -> None:
        """Build the block, the mixing layer, and the head with fixed weights."""

        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(3, 3, dtype=torch.float64),
            nn.ReLU(inplace=True),
            nn.Linear(3, 3, dtype=torch.float64),
            nn.ReLU(),
        )
        self.mix = nn.Linear(3, 3, dtype=torch.float64)
        self.classifier = nn.Linear(3, 2, dtype=torch.float64)
        with torch.no_grad():
            self.features[0].weight.copy_(torch.eye(3, dtype=torch.float64))
            self.features[0].bias.zero_()
            self.features[2].weight.copy_(torch.eye(3, dtype=torch.float64))
            self.features[2].bias.copy_(torch.full((3,), 0.2, dtype=torch.float64))
            self.mix.weight.copy_(
                torch.tensor(
                    [[0.6, 0.3, -0.2], [0.5, -0.4, 0.3], [-0.1, 0.7, 0.4]],
                    dtype=torch.float64,
                )
            )
            self.mix.bias.zero_()
            self.classifier.weight.copy_(
                torch.tensor([[1.0, -1.0, 1.0], [0.5, 0.5, -0.5]], dtype=torch.float64)
            )
            self.classifier.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """Features, mixing, the functional in-place ReLU, then the head."""

        out = self.features(x)
        out = self.mix(out)
        out = F.relu(out - 0.15, inplace=True)
        return self.classifier(out)


def test_guided_rule_matches_hand_rolled_math() -> None:
    """Guided ReLU == clamp(upstream, 0) * forward mask, per site."""

    weight = torch.tensor([[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]], dtype=torch.float64)
    model = _SingleRelu(weight)
    x = torch.tensor([[0.8, 0.1, 0.2]], dtype=torch.float64)
    result = attribution.guided_backprop(model, x, target=0)
    # Head is -relu(Wx); target 0 selects row 0. Upstream gradient into the
    # ReLU is -e0 (all nonpositive), so guided clamps it to zero everywhere.
    torch.testing.assert_close(result.values, torch.zeros_like(x), rtol=0, atol=0)
    # Plain gradient is NOT zero (the ReLU is active on row 0), proving the
    # rule modified the backward.
    plain = attribution.input_x_grad(model, x, target=0)
    assert not torch.equal(plain.values, torch.zeros_like(x))
    assert result.extra["absolute"] is False
    assert result.extra["site_census"]["rewritten_firings"] == 1


def test_deconvolution_drops_the_forward_mask() -> None:
    """Deconv passes positive upstream gradient even where the input was negative."""

    weight = torch.eye(3, dtype=torch.float64)[:2]
    model = _SingleRelu(weight)
    # forward: -relu(x[:2]); callable target = -output[0,0] so the upstream
    # gradient into the ReLU is +e0 (positive).
    x = torch.tensor([[-0.5, 0.4, 0.2]], dtype=torch.float64)

    def score(output: Tensor) -> Tensor:
        """Negate so the upstream gradient into the ReLU is positive."""

        return -output[0, 0]

    guided = attribution.guided_backprop(model, x, target=score)
    deconv = attribution.deconvolution(model, x, target=score)
    # Input to the ReLU at slot 0 is negative: guided masks it to zero,
    # deconvolution passes the positive upstream gradient through.
    assert guided.values[0, 0].item() == 0.0
    assert deconv.values[0, 0].item() == pytest.approx(1.0)


def test_op_level_coverage_counts_all_four_spellings() -> None:
    """Module, functional, method, and in-place firings are all rewritten."""

    model = _MixedReluNet()
    x = torch.tensor([[0.6, -0.3, 0.2]], dtype=torch.float64)
    result = attribution.guided_backprop(model, x, target=0)
    census = result.extra["site_census"]
    assert census["total_relu_firings"] == 4
    assert census["module_dispatched_firings"] == 1
    assert census["functional_or_method_firings"] == 3
    assert census["rewritten_firings"] == 4


def test_functional_inplace_toy_module_vs_all_sites_differ() -> None:
    """The D13 unit row: restricted coverage misses the functional in-place ReLU."""

    model = _FunctionalInPlaceToy()
    x = torch.tensor([[0.5, 0.4, 0.3]], dtype=torch.float64)
    all_sites = attribution.guided_backprop(model, x, target=0, sites="all")
    module_sites = attribution.guided_backprop(model, x, target=0, sites="module")
    assert all_sites.extra["site_census"]["rewritten_firings"] == 3
    assert module_sites.extra["site_census"]["rewritten_firings"] == 2
    assert module_sites.extra["site_census"]["functional_or_method_firings"] == 1
    # The one functional in-place ReLU keeps NATIVE backward under
    # sites='module', so the two coverage classes genuinely diverge, and
    # neither result is trivially zero.
    assert all_sites.values.abs().sum().item() > 0
    assert not torch.equal(all_sites.values, module_sites.values)
    torch.testing.assert_close(
        all_sites.values,
        torch.tensor([[0.5, 1.0, 0.2]], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        module_sites.values,
        torch.tensor([[0.0, 1.4, 0.0]], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_zero_match_refusal_names_observed_activations() -> None:
    """A GELU-only model refuses typed, naming what it saw (teaching refusal)."""

    class _GeluNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(3, 2, dtype=torch.float64)

        def forward(self, x: Tensor) -> Tensor:
            return F.gelu(self.lin(x))

    with pytest.raises(AttributionError) as excinfo:
        attribution.guided_backprop(_GeluNet(), torch.zeros(1, 3, dtype=torch.float64), target=0)
    assert excinfo.value.fields["code"] == "guided_no_relu_sites"
    assert "gelu" in str(excinfo.value)


def test_forward_fidelity_tripwire_on_discarded_inplace_return() -> None:
    """A forward that discards the in-place ReLU return value refuses typed."""

    class _DiscardingNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(3, 2, dtype=torch.float64)

        def forward(self, x: Tensor) -> Tensor:
            y = x * 1.0
            torch.relu_(y)  # Return value discarded: outside the coverage claim.
            return self.head(y)

    x = torch.tensor([[-0.5, 0.4, -0.1]], dtype=torch.float64)
    with pytest.raises(AttributionError) as excinfo:
        attribution.guided_backprop(_DiscardingNet(), x, target=0)
    assert excinfo.value.fields["code"] == "guided_forward_fidelity_violated"


def test_state_purity_on_success_and_forced_failure() -> None:
    """Modes, hooks, params, and grads are restored on success AND failure."""

    model = _FunctionalInPlaceToy()
    model.train()
    x = torch.tensor([[0.2, 0.1, -0.3]], dtype=torch.float64)
    params_before = {name: param.detach().clone() for name, param in model.named_parameters()}

    attribution.guided_backprop(model, x, target=0)
    assert model.training is True
    for module in model.modules():
        assert not module._forward_hooks, "forward hooks must be removed"
        assert not module._forward_pre_hooks, "pre-hooks must be removed"
    for name, param in model.named_parameters():
        torch.testing.assert_close(param.detach(), params_before[name], rtol=0, atol=0)
        assert param.grad is None

    def exploding(_output: Tensor) -> Tensor:
        """Force a mid-attribution failure after the forward."""

        raise RuntimeError("forced failure")

    with pytest.raises(RuntimeError, match="forced failure"):
        attribution.guided_backprop(model, x, target=exploding)
    assert model.training is True
    for module in model.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks


def test_noise_tunnel_over_guided_methods_works() -> None:
    """D12: NT over the named guided methods composes."""

    model = _FunctionalInPlaceToy()
    x = torch.tensor([[0.5, 0.2, -0.4]], dtype=torch.float64)
    result = attribution.noise_tunnel(
        x,
        method=attribution.guided_backprop,
        model=model,
        target=0,
        n_samples=3,
        stdevs=0.05,
        seed=2,
    )
    assert result.method == "noise_tunnel"
    assert result.extra["child_method"] == "guided_backprop"
    assert result.values.shape == x.shape


def test_no_guided_rule_spelling_exists_on_path_methods() -> None:
    """D12: path methods expose NO kwarg that mounts a modified backward rule.

    The refusal is the ABSENCE of the knob: modified backward rules invalidate
    path/completeness semantics, so no spelling may combine them.
    """

    import inspect

    for method in (
        attribution.integrated_gradients,
        attribution.layer_integrated_gradients,
        attribution.layer_conductance,
        attribution.gradient_shap,
    ):
        parameters = inspect.signature(method).parameters
        assert "rule" not in parameters
        assert "backward_rule" not in parameters
        assert "guided" not in parameters


def test_reused_relu_module_counts_every_firing() -> None:
    """A reused nn.ReLU module contributes one rewritten firing per call."""

    class _ReusedRelu(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.act = nn.ReLU()
            self.lin = nn.Linear(3, 3, dtype=torch.float64)

        def forward(self, x: Tensor) -> Tensor:
            return self.act(self.lin(self.act(x)))

    result = attribution.guided_backprop(
        _ReusedRelu(), torch.tensor([[0.3, -0.2, 0.5]], dtype=torch.float64), target=0
    )
    census = result.extra["site_census"]
    assert census["total_relu_firings"] == 2
    assert census["module_dispatched_firings"] == 2
    assert census["rewritten_firings"] == 2
