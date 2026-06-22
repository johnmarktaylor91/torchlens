"""DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation.

Park, Florence, Straub, Newcombe & Lovegrove (Facebook/MIT), CVPR 2019.
Paper: https://arxiv.org/abs/1901.05103
Source: https://github.com/facebookresearch/DeepSDF

DeepSDF represents 3D shapes as a continuous SDF (signed distance function).
The model is an *auto-decoder*: there is no explicit encoder network.  Instead
each training shape is assigned a learnable latent code z; at inference, the
latent code is optimised directly by minimising the reconstruction loss.

The decoder is a deep MLP with:
  - Input: concatenation of latent code z (dim=128) and a 3D query point p (3 dims)
  - 8 fully-connected layers, each of width 512, with ReLU activations.
  - Skip connection at layer 4: the input (z, p) is concatenated back to the
    mid-layer activations (as in the original Figure 2 architecture).
  - Output: a single scalar (the SDF value at the query point).
  - Weight normalisation on all linear layers (as in the original implementation).

This is a faithful compact random-init reimplementation of the DeepSDF decoder.
Depth reduced from 8 to 5 hidden layers for a compact trace; hidden dim 512->256.
The defining characteristics -- latent concat at input, mid-layer skip concat,
weight-norm linear layers, scalar SDF output -- are all preserved.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSDFDecoder(nn.Module):
    """DeepSDF auto-decoder MLP with mid-layer input skip connection.

    Args:
        latent_size: dimensionality of the shape latent code z.
        hidden_size: width of each hidden layer.
        num_layers: total number of hidden layers (skip inserted at half-depth).
        dropout_prob: dropout probability (0 = disabled).
    """

    def __init__(
        self,
        latent_size: int = 128,
        hidden_size: int = 256,
        num_layers: int = 5,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.skip_at = num_layers // 2  # skip connection at half depth

        # Input dimension: latent + 3D point
        in_dim = latent_size + 3
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.utils.weight_norm(nn.Linear(in_dim, hidden_size)))
            elif i == self.skip_at:
                # Skip: concatenate original input to current activations
                layers.append(nn.utils.weight_norm(nn.Linear(hidden_size + in_dim, hidden_size)))
            else:
                layers.append(nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size)))
            if dropout_prob > 0.0:
                layers.append(nn.Dropout(dropout_prob))
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.th = nn.Tanh()

    def forward(self, z: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_size) shape latent codes.
        xyz: (B, M, 3) query 3D points (M points per shape).
        Returns: (B, M, 1) SDF values.
        """
        B, M, _ = xyz.shape
        # Expand z to match M points: (B, M, latent_size)
        z_exp = z.unsqueeze(1).expand(B, M, self.latent_size)
        x = torch.cat([z_exp, xyz], dim=-1)  # (B, M, latent+3)
        inp = x  # save for skip

        # Flatten batch x points for linear layers
        x = x.reshape(B * M, -1)
        inp_flat = inp.reshape(B * M, -1)

        layer_idx = 0
        for i in range(self.num_layers):
            if i == self.skip_at:
                x = torch.cat([x, inp_flat], dim=-1)
            layer = self.layers[layer_idx]
            x = F.relu(layer(x), inplace=True)
            layer_idx += 1
            # Skip the dropout module in ModuleList if present
            if layer_idx < len(self.layers) and isinstance(self.layers[layer_idx], nn.Dropout):
                x = self.layers[layer_idx](x)
                layer_idx += 1

        out = self.th(self.output_layer(x))  # (B*M, 1)
        return out.reshape(B, M, 1)


def build() -> nn.Module:
    """Build DeepSDF decoder (latent=128, hidden=256, 5 layers, mid-skip)."""
    return DeepSDFDecoder(latent_size=128, hidden_size=256, num_layers=5)


def example_input() -> list:
    """Example input: latent code (2, 128) and 16 query points (2, 16, 3)."""
    z = torch.randn(2, 128)
    xyz = torch.randn(2, 16, 3)
    return [z, xyz]


MENAGERIE_ENTRIES = [
    (
        "DeepSDF (auto-decoder, latent+xyz MLP with mid-layer skip, SDF output)",
        "build",
        "example_input",
        "2019",
        "DC",
    ),
]
