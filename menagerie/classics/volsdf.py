"""VolSDF ImplicitNetwork: Volume Rendering of Neural Implicit Surfaces.

Yariv et al., NeurIPS 2021.
Paper: https://arxiv.org/abs/2106.12052
Source: https://github.com/lioryariv/volsdf

VolSDF models a scene as a signed-distance field (SDF) decoded by an MLP and
volume-rendered through a density derived from the SDF.  The architecture-defining
component is the ``ImplicitNetwork``: an 8-layer MLP (width 256) with a single
skip connection re-injecting the (positionally-encoded) input at layer 4, a
geometric weight initialization, and a Softplus nonlinearity.  It maps a 3D point
to a 1D signed distance plus a 256-d feature vector consumed by the rendering
(radiance) network.

This is a faithful random-init reimplementation of the published ImplicitNetwork
(``model/network.py``):
  - d_in=3, positional encoding with multires=6 (-> 3 + 3*2*6 = 39 input dims)
  - dims = [256] * 8, skip_in = [4]
  - geometric initialization (bias=0.6, sphere init) reproduced exactly
  - Softplus(beta=100) activations, output = [sdf(1), feature(256)]
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """NeRF-style positional encoding: ``[p, sin(2^k p), cos(2^k p)]``."""

    def __init__(self, d_in: int = 3, multires: int = 6) -> None:
        super().__init__()
        self.d_in = d_in
        self.num_freqs = multires
        self.freq_bands = 2.0 ** torch.arange(0, multires).float()
        self.d_out = d_in * (1 + 2 * multires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, dim=-1)


class ImplicitNetwork(nn.Module):
    """VolSDF SDF MLP with geometric initialization and a single skip-in."""

    def __init__(
        self,
        feature_vector_size: int = 256,
        d_in: int = 3,
        d_out: int = 1,
        dims: List[int] = None,
        geometric_init: bool = True,
        bias: float = 0.6,
        skip_in: List[int] = None,
        multires: int = 6,
        sphere_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if dims is None:
            dims = [256] * 8
        if skip_in is None:
            skip_in = [4]

        self.embed = PositionalEncoding(d_in, multires) if multires > 0 else None
        d_in_eff = self.embed.d_out if self.embed is not None else d_in

        dims = [d_in_eff] + dims + [d_out + feature_vector_size]
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.softplus = nn.Softplus(beta=100)

        layers = []
        for layer in range(self.num_layers - 1):
            out_dim = dims[layer + 1]
            if layer + 1 in skip_in:
                out_dim = out_dim - dims[0]
            lin = nn.Linear(dims[layer], out_dim)

            if geometric_init:
                if layer == self.num_layers - 2:
                    # Final layer: initialize to an approximate sphere SDF.
                    nn.init.normal_(
                        lin.weight,
                        mean=math.sqrt(math.pi) / math.sqrt(dims[layer]),
                        std=0.0001,
                    )
                    nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and layer == 0:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.constant_(lin.weight[:, 3:], 0.0)
                    nn.init.normal_(lin.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(out_dim))
                elif multires > 0 and layer in skip_in:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, math.sqrt(2) / math.sqrt(out_dim))
                    nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, math.sqrt(2) / math.sqrt(out_dim))

            layers.append(lin)
        self.linears = nn.ModuleList(layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if self.embed is not None:
            x = self.embed(points)
        else:
            x = points
        inp = x

        for layer, lin in enumerate(self.linears):
            if layer in self.skip_in:
                x = torch.cat([x, inp], dim=-1) / math.sqrt(2)
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.softplus(x)
        return x  # (N, 1 + feature_vector_size): [sdf, feature]


def build() -> nn.Module:
    """Build the VolSDF ImplicitNetwork (8x256 SDF MLP, skip-in=[4], multires=6)."""
    return ImplicitNetwork(
        feature_vector_size=256,
        d_in=3,
        d_out=1,
        dims=[256] * 8,
        geometric_init=True,
        bias=0.6,
        skip_in=[4],
        multires=6,
    )


def example_input() -> torch.Tensor:
    """Example batch of 3D query points ``(2048, 3)`` for the SDF network."""
    return torch.randn(2048, 3)


MENAGERIE_ENTRIES = [
    (
        "VolSDF ImplicitNetwork (neural SDF, geometric init)",
        "build",
        "example_input",
        "2021",
        "DC",
    ),
]
