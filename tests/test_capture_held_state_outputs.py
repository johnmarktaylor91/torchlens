"""A module-held plain tensor returned untouched from ``forward`` is attributed.

Model preparation stamps tensors a module holds outside its registries as
buffer sources, so reading one roots at a buffer node. Returned directly with
no traced op touching it (an SSD-style anchor table, a prior distribution
built in ``__init__``, a helper object's statistics), the tensor used to fail
capture with ``OutputAttributionError`` while the plain forward ran fine; only
registered buffers were attributed.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.distributions import Normal

import torchlens as tl


class _AnchorTable(nn.Module):
    """Detector head that returns its precomputed anchors beside its predictions."""

    def __init__(self) -> None:
        """Build a 1x1 head and a plain anchor tensor."""

        super().__init__()
        self.head = nn.Conv2d(3, 4, 1)
        self.anchors = torch.rand(6, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predictions and the untouched anchors."""

        return self.head(x), self.anchors


class _WithPrior(nn.Module):
    """VAE-style encoder returning its posterior and its fixed prior."""

    def __init__(self) -> None:
        """Build the encoder and a standard-normal prior."""

        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.prior = Normal(torch.zeros(4), torch.ones(4))

    def forward(self, x: torch.Tensor) -> tuple[Normal, Normal]:
        """Return the posterior and the prior object."""

        h = self.encoder(x)
        return Normal(h, h.exp()), self.prior


class _Stats:
    """Plain helper object holding a statistic tensor."""

    def __init__(self) -> None:
        """Hold one tensor attribute."""

        self.mean = torch.arange(4.0)


class _WithStats(nn.Module):
    """Module returning a plain helper object it holds."""

    def __init__(self) -> None:
        """Build a layer and the helper."""

        super().__init__()
        self.layer = nn.Linear(4, 4)
        self.stats = _Stats()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, _Stats]:
        """Return the layer output and the helper object."""

        return self.layer(x), self.stats


@pytest.mark.parametrize(
    ("factory", "shape"),
    [(_AnchorTable, (1, 3, 4, 4)), (_WithPrior, (2, 4)), (_WithStats, (2, 4))],
    ids=["anchor_table", "prior_distribution", "helper_object"],
)
def test_held_tensor_output_traces_and_validates(factory: type[nn.Module], shape: tuple) -> None:
    """Capture attributes every output and validation passes."""

    torch.manual_seed(0)
    model = factory().eval()
    x = torch.randn(*shape)
    trace = tl.trace(model, x)
    assert trace.buffer_layers
    assert tl.validate(model, x, seed=0) is True


def test_anchor_output_keeps_the_held_value() -> None:
    """The held output's recorded value is the module's tensor, not a copy that drifted."""

    torch.manual_seed(0)
    model = _AnchorTable().eval()
    trace = tl.trace(model, torch.randn(1, 3, 4, 4))
    assert len(trace.output_layers) == 2
    assert torch.equal(trace[trace.output_layers[1]].out, model.anchors)
    assert model.anchors.grad_fn is None
