"""F06 real-model rows: the DenseNet-121 divergence identity (attrib memo D13).

The kit's marketing sentence as a blocking test with no download (weights are
cached in this checkout's torch hub): restricted to module-dispatched ReLUs
our guided backprop matches the module-hook coverage class (captum parity is
pinned separately in the captum oracle ledger); over ALL executed ReLU ops it
differs, and the difference is exactly the one functional
``F.relu(features, inplace=True)`` in torchvision's own ``DenseNet.forward``.
The honest limit rides with the row: the magnitude is one model / one target /
one input / CPU; the IDENTITY is the claim.
"""

from __future__ import annotations

import pytest
import torch

import torchlens.attribution as attribution

torchvision_models = pytest.importorskip("torchvision.models")

pytestmark = [pytest.mark.heavy, pytest.mark.real_model]


@pytest.fixture(scope="module")
def densenet121() -> torch.nn.Module:
    """Load the cached DenseNet-121 checkpoint (no download in CI)."""

    weights = torchvision_models.DenseNet121_Weights.IMAGENET1K_V1
    model = torchvision_models.densenet121(weights=weights)
    model.eval()
    return model


@pytest.fixture(scope="module")
def image() -> torch.Tensor:
    """One deterministic normalized input image."""

    generator = torch.Generator().manual_seed(1234)
    return torch.randn(1, 3, 224, 224, generator=generator)


def test_densenet121_relu_site_census(densenet121: torch.nn.Module, image: torch.Tensor) -> None:
    """DenseNet-121 executes 121 ReLU ops: 120 module-dispatched + 1 functional.

    The single functional firing is ``F.relu(features, inplace=True)`` in
    ``DenseNet.forward`` -- the site module hooks cannot see (memo D13 (iii)).
    """

    result = attribution.guided_backprop(densenet121, image, target=207, sites="all")
    census = result.extra["site_census"]
    assert census["total_relu_firings"] == 121
    assert census["module_dispatched_firings"] == 120
    assert census["functional_or_method_firings"] == 1
    assert census["rewritten_firings"] == 121


def test_densenet121_module_vs_all_sites_diverge(
    densenet121: torch.nn.Module, image: torch.Tensor
) -> None:
    """Op-level coverage genuinely differs from module-hook coverage (D13 (ii)).

    ``sites="module"`` reproduces the module-hook coverage class (captum
    parity is the oracle-ledger row); ``sites="all"`` additionally rewrites
    the one functional ReLU, and the attributions differ.
    """

    all_sites = attribution.guided_backprop(densenet121, image, target=207, sites="all")
    module_sites = attribution.guided_backprop(densenet121, image, target=207, sites="module")
    assert module_sites.extra["site_census"]["rewritten_firings"] == 120
    assert not torch.equal(all_sites.values, module_sites.values)
    difference = (all_sites.values - module_sites.values).abs().max().item()
    assert difference > 0.0
