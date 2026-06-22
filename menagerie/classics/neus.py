"""NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction.

Wang, Liu, Liu, Theobalt, Komura & Wang, NeurIPS 2021.
Paper: https://arxiv.org/abs/2106.10689
Source: https://github.com/Totoro97/NeuS

NeuS represents a scene as a signed-distance field (SDF) and renders it via a novel
volume-rendering formulation that is unbiased and occlusion-aware.  Two networks:

  1. **SDFNetwork**: an 8-layer MLP (width 256) with:
     - Positional encoding of the input 3D point (multires=6 by default)
     - Skip connection re-injecting the encoded input at layer 4 (IDR/DeepSDF style)
     - Geometric weight initialization (bias on the last layer to initialise a sphere)
     - Output: (sdf_scalar, feature_vector_256)

  2. **RenderingNetwork**: a 4-layer MLP that maps
     (point_xyz, normal=nabla_SDF, view_dir, feature_vector) -> RGB colour.
     Applies a sigmoid at the end for colour in [0,1].

  The S-density conversion: alpha_i = max((-d/dt) sigmoid(-sdf/s), 0) * prod(1 - alpha_j)
  is the key rendering novelty (NOT reproduced in the network forward, which just outputs
  the raw SDF and feature vector).  This reimplementation faithfully captures the
  SDFNetwork and RenderingNetwork topology (the networks are the architecture;
  the rendering formula is the training objective mechanism).

Simplifications:
  - Grid of positional-encoding harmonics: multires=6 (paper default).
  - SDFNetwork: 8 layers x 256 width (paper default).
  - RenderingNetwork: 4 layers x 256 width (paper default).
  - Geometric init: last linear bias = 0.6 (paper's sphere-init).
  - Batch of 64 sample points for the trace.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding (same as NeRF / IDR / NeuS)
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with ``multires`` frequency octaves.

    Maps (N, d_in) -> (N, d_in + 2 * d_in * multires).
    """

    def __init__(self, d_in: int = 3, multires: int = 6) -> None:
        super().__init__()
        self.d_in = d_in
        self.multires = multires
        self.d_out = d_in + 2 * d_in * multires

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for i in range(self.multires):
            freq = 2.0**i * math.pi
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)


# ---------------------------------------------------------------------------
# SDFNetwork
# ---------------------------------------------------------------------------


class SDFNetwork(nn.Module):
    """NeuS SDFNetwork: 8-layer MLP with skip-connection at layer 4.

    Distinctive features:
      - Input: positional-encoded 3D point (PE with multires=6 -> 39 dims).
      - Architecture: 8 linear layers of width ``d_hidden``.
      - Skip connection: input PE re-injected at layer ``skip_in[0]`` (4).
      - Geometric initialisation: last layer bias = +0.6 (sphere init).
      - Softplus(beta=100) activation throughout.
      - Output: SDF scalar (1 dim) + feature vector (d_hidden dims).

    This faithfully reproduces the IDR/NeuS-style 8-layer SDF MLP.
    """

    def __init__(
        self,
        d_in: int = 3,
        d_hidden: int = 256,
        n_layers: int = 8,
        skip_in: List[int] = None,
        multires: int = 6,
        geometric_init: bool = True,
        bias: float = 0.6,
    ) -> None:
        super().__init__()
        if skip_in is None:
            skip_in = [4]

        self.skip_in = skip_in
        self.d_hidden = d_hidden
        self.pe = PositionalEncoding(d_in, multires)
        d_enc = self.pe.d_out

        # Build linear layers
        layers: List[nn.Module] = []
        for i in range(n_layers):
            if i == 0:
                in_dim = d_enc
            elif i in skip_in:
                # Skip: concat with the PE input
                in_dim = d_hidden + d_enc
            else:
                in_dim = d_hidden
            # Last layer outputs (1 + d_hidden): sdf + feature
            out_dim = 1 + d_hidden if i == n_layers - 1 else d_hidden
            layers.append(nn.Linear(in_dim, out_dim))

        self.linears = nn.ModuleList(layers)
        self.activation = nn.Softplus(beta=100)

        if geometric_init:
            self._geometric_init(n_layers, d_enc, bias)

    def _geometric_init(self, n_layers: int, d_enc: int, bias: float) -> None:
        """Initialise weights so the SDF approximates a unit sphere at init."""
        for idx, linear in enumerate(self.linears):
            if idx == len(self.linears) - 1:
                # Last layer: bias -> bias (positive -> interior is negative sdf)
                nn.init.normal_(
                    linear.weight, 0.0, math.sqrt(2) / math.sqrt(linear.weight.shape[0])
                )
                nn.init.constant_(linear.bias, -bias)
            elif idx == 0:
                nn.init.constant_(linear.bias, 0.0)
                nn.init.normal_(
                    linear.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(linear.weight.shape[0])
                )
                nn.init.constant_(linear.weight[:, 3:], 0.0)
            elif idx in self.skip_in:
                nn.init.constant_(linear.bias, 0.0)
                nn.init.normal_(
                    linear.weight, 0.0, math.sqrt(2) / math.sqrt(linear.weight.shape[0])
                )
                # zero out the skip weights for the PE part (first d_enc columns beyond d_hidden)
                if linear.weight.shape[1] > self.d_hidden:
                    nn.init.constant_(linear.weight[:, -(d_enc - 3) :], 0.0)
            else:
                nn.init.constant_(linear.bias, 0.0)
                nn.init.normal_(
                    linear.weight, 0.0, math.sqrt(2) / math.sqrt(linear.weight.shape[0])
                )

    def forward(self, xyz: torch.Tensor) -> tuple:
        """Forward pass.

        Args:
            xyz: (N, 3) 3D sample points.

        Returns:
            (sdf, feature): each shape (N, 1) and (N, d_hidden).
        """
        pe = self.pe(xyz)  # (N, d_enc)
        x = pe

        for idx, linear in enumerate(self.linears):
            if idx in self.skip_in:
                x = torch.cat([x, pe], dim=-1)
            x = linear(x)
            if idx < len(self.linears) - 1:
                x = self.activation(x)

        # x: (N, 1 + d_hidden)
        sdf = x[:, :1]  # (N, 1)
        feature = x[:, 1:]  # (N, d_hidden)
        return sdf, feature


# ---------------------------------------------------------------------------
# RenderingNetwork
# ---------------------------------------------------------------------------


class RenderingNetwork(nn.Module):
    """NeuS RenderingNetwork: appearance MLP.

    Input: cat(points_xyz, normals, view_dirs, feature_vector)
    Architecture: 4 linear layers of width ``d_hidden``, output = 3 (RGB).

    The normal at a point is the gradient of the SDF (computed externally during
    training via autograd).  In this forward we accept it as an explicit input.
    """

    def __init__(
        self,
        d_feature: int = 256,
        d_in_xyz: int = 3,
        d_in_normal: int = 3,
        d_in_view: int = 3,
        d_hidden: int = 256,
        n_layers: int = 4,
        multires_view: int = 4,
    ) -> None:
        super().__init__()
        self.pe_view = PositionalEncoding(d_in_view, multires_view)
        d_view_enc = self.pe_view.d_out

        d_in_total = d_in_xyz + d_in_normal + d_view_enc + d_feature
        layers: List[nn.Module] = []
        for i in range(n_layers):
            in_dim = d_in_total if i == 0 else d_hidden
            out_dim = 3 if i == n_layers - 1 else d_hidden
            layers.append(nn.Linear(in_dim, out_dim))
        self.linears = nn.ModuleList(layers)
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        xyz: torch.Tensor,
        normals: torch.Tensor,
        view_dirs: torch.Tensor,
        feature_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            xyz:            (N, 3) 3D point coordinates.
            normals:        (N, 3) SDF gradient / surface normals (unit vectors).
            view_dirs:      (N, 3) unit viewing directions.
            feature_vector: (N, d_feature) feature from SDFNetwork.

        Returns:
            (N, 3) RGB colours in [0, 1].
        """
        view_enc = self.pe_view(view_dirs)  # (N, d_view_enc)
        x = torch.cat([xyz, normals, view_enc, feature_vector], dim=-1)

        for idx, linear in enumerate(self.linears):
            x = linear(x)
            if idx < len(self.linears) - 1:
                x = self.activation(x)

        return torch.sigmoid(x)  # RGB in [0, 1]


# ---------------------------------------------------------------------------
# Combined NeuS forward (SDF net + rendering net)
# ---------------------------------------------------------------------------


class NeuS(nn.Module):
    """Combined NeuS model (SDFNetwork + RenderingNetwork).

    In training, normals are computed as nabla_SDF via autograd.  Here we provide
    a compact combined forward that accepts pre-computed normals (or random normals
    for the menagerie trace) to keep the graph tractable.
    """

    def __init__(
        self,
        d_hidden_sdf: int = 256,
        n_layers_sdf: int = 8,
        skip_in: List[int] = None,
        multires_pts: int = 6,
        d_hidden_render: int = 256,
        n_layers_render: int = 4,
        multires_view: int = 4,
    ) -> None:
        super().__init__()
        if skip_in is None:
            skip_in = [4]

        self.sdf_net = SDFNetwork(
            d_in=3,
            d_hidden=d_hidden_sdf,
            n_layers=n_layers_sdf,
            skip_in=skip_in,
            multires=multires_pts,
            geometric_init=True,
            bias=0.6,
        )
        self.render_net = RenderingNetwork(
            d_feature=d_hidden_sdf,
            d_in_xyz=3,
            d_in_normal=3,
            d_in_view=3,
            d_hidden=d_hidden_render,
            n_layers=n_layers_render,
            multires_view=multires_view,
        )

    def forward(
        self,
        xyz: torch.Tensor,
        view_dirs: torch.Tensor,
        normals: torch.Tensor,
    ) -> dict:
        """Forward pass.

        Args:
            xyz:       (N, 3) sample points.
            view_dirs: (N, 3) unit viewing directions.
            normals:   (N, 3) surface normals (gradient of SDF at xyz).

        Returns:
            dict with 'sdf' (N,1), 'rgb' (N,3).
        """
        sdf, feature = self.sdf_net(xyz)
        rgb = self.render_net(xyz, normals, view_dirs, feature)
        return {"sdf": sdf, "rgb": rgb}


# ---------------------------------------------------------------------------
# Build functions and example inputs
# ---------------------------------------------------------------------------


def build_neus_sdf() -> nn.Module:
    """Build NeuS SDFNetwork (8x256 MLP, skip at layer 4, geometric init)."""
    return SDFNetwork(
        d_in=3,
        d_hidden=256,
        n_layers=8,
        skip_in=[4],
        multires=6,
        geometric_init=True,
        bias=0.6,
    )


def build_neus_render() -> nn.Module:
    """Build NeuS RenderingNetwork (4x256 MLP, view+normal+feature -> RGB)."""
    return RenderingNetwork(
        d_feature=256,
        d_in_xyz=3,
        d_in_normal=3,
        d_in_view=3,
        d_hidden=256,
        n_layers=4,
        multires_view=4,
    )


def build_neus() -> nn.Module:
    """Build combined NeuS model (SDFNetwork + RenderingNetwork)."""
    return NeuS(
        d_hidden_sdf=256,
        n_layers_sdf=8,
        skip_in=[4],
        multires_pts=6,
        d_hidden_render=256,
        n_layers_render=4,
        multires_view=4,
    )


def example_input_sdf() -> torch.Tensor:
    """Example batch of 3D points (64, 3) for NeuS SDFNetwork."""
    return torch.randn(64, 3)


def example_input_render() -> list:
    """Example (xyz, normals, view_dirs, feature) for NeuS RenderingNetwork."""
    N = 64
    xyz = torch.randn(N, 3)
    normals = F.normalize(torch.randn(N, 3), dim=-1)
    view_dirs = F.normalize(torch.randn(N, 3), dim=-1)
    feature = torch.randn(N, 256)
    return [xyz, normals, view_dirs, feature]


def example_input_neus() -> list:
    """Example (xyz, view_dirs, normals) for combined NeuS model."""
    N = 64
    xyz = torch.randn(N, 3)
    view_dirs = F.normalize(torch.randn(N, 3), dim=-1)
    normals = F.normalize(torch.randn(N, 3), dim=-1)
    return [xyz, view_dirs, normals]


MENAGERIE_ENTRIES = [
    (
        "NeuS SDFNetwork (8-layer SDF MLP, skip@4, geometric init)",
        "build_neus_sdf",
        "example_input_sdf",
        "2021",
        "DC",
    ),
    (
        "NeuS RenderingNetwork (appearance MLP: pts+normals+views+feat -> RGB)",
        "build_neus_render",
        "example_input_render",
        "2021",
        "DC",
    ),
    (
        "NeuS (combined SDFNetwork + RenderingNetwork, unbiased volume rendering)",
        "build_neus",
        "example_input_neus",
        "2021",
        "DC",
    ),
]
