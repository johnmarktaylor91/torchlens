# SOURCE: vendored from giovannicatalani/MARIO @ main
# Files: src/models.py (LatentToModulation, GaussianEncoding, MultiScaleModulatedFourierFeatures),
#        src/conditioning.py (shift_modulation)
# https://github.com/giovannicatalani/MARIO
#
# MARIO (Multiscale Aerodynamic Resolution Invariant Operator): a conditional Neural Field
# architecture for surrogate modeling of aerodynamic fields (pressure, velocity, stress...)
# around objects of varying geometry, from "Towards scalable surrogate models based on
# Neural Fields for large scale aerodynamic simulations" (arXiv:2505.14704). 3rd place at the
# ML4CFD Challenge at NeurIPS 2024. `MultiScaleModulatedFourierFeatures` is the flagship
# architecture: multiple `GaussianEncoding` Fourier-feature embeddings at different
# frequency scales are each fed through a shared stack of coordinate-MLP `Linear` layers,
# whose per-layer activations are FiLM-style shift-modulated (`conditioning.shift_modulation`)
# by features produced by a small hypernetwork (`LatentToModulation`) conditioned on a latent
# code z (e.g. a learned geometry-encoding SDF latent). The per-scale outputs are concatenated
# and combined by a final linear layer; an optional `scalar_head` regresses global scalar
# quantities (e.g. lift/drag) from the modulation features.
#
# Import-fix only (per rung-2 rules, architecture code is untouched): the original
# `src/models.py` does `try: from src.conditioning import shift_modulation / except:
# from conditioning import shift_modulation` (a repo-relative import with a fallback for
# running the file standalone inside src/). Inlined directly here since both files are
# vendored together in this one module. No other code was changed.

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- src/conditioning.py, shift_modulation (verbatim) --------------------------------------


def shift_modulation(position, features, layers, activation, with_batch=True):
    """Applies film conditioning (add only) on the network.
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """
    feature_shape = features.shape[0]  # features.shape[:-1]
    feature_dim = features.shape[-1]
    num_hidden = len(layers)
    # Maybe add assertion here... but if it errors, your feature_dim size is wrong

    if with_batch:
        features = features.reshape(feature_shape, 1, num_hidden, feature_dim // num_hidden)
    else:
        features = features.reshape(feature_shape, num_hidden, feature_dim // num_hidden)

    h = position

    for i, layer in enumerate(layers):
        res = layer(h)
        # Maybe also add another assertion here
        h = res * features[..., i, :] + features[..., i, :] + res
        h = activation(h)
    return h


# --- src/models.py (verbatim) ---------------------------------------------------------------


class LatentToModulation(nn.Module):
    """Maps a latent vector to a set of modulations.
    Args:
        latent_dim (int):
        num_modulations (int):
        dim_hidden (int):
        num_layers (int):
    """

    def __init__(self, latent_dim, num_modulations, dim_hidden, num_layers, activation=nn.SiLU):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modulations = num_modulations
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.activation = activation

        if num_layers == 1:
            self.net = nn.Linear(latent_dim, num_modulations)
        else:
            layers = [nn.Linear(latent_dim, dim_hidden), self.activation()]
            if num_layers > 2:
                for i in range(num_layers - 2):
                    layers += [nn.Linear(dim_hidden, dim_hidden), self.activation()]
            layers += [nn.Linear(dim_hidden, num_modulations)]
            self.net = nn.Sequential(*layers)

    def forward(self, latent):
        return self.net(latent)


class GaussianEncoding(nn.Module):
    def __init__(self, embedding_size, scale, dims=2, gaussian=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale

        if gaussian:
            bvals = torch.randn(embedding_size // 2, dims) * scale
        else:
            bvals = 2.0 ** torch.linspace(0, scale, embedding_size // 2) - 1

            if dims == 1:
                bvals = bvals[:, None]

            elif dims == 2:
                bvals = torch.stack([bvals, torch.zeros_like(bvals)], dim=-1)
                bvals = torch.cat([bvals, torch.roll(bvals, 1, -1)], dim=0)

            else:
                tmp = (dims - 1) * (torch.zeros_like(bvals),)
                bvals = torch.stack([bvals, *tmp], dim=-1)

                tmp = [torch.roll(bvals, i, -1) for i in range(1, dims)]
                bvals = torch.cat([bvals, *tmp], dim=0)

        avals = torch.ones((bvals.shape[0]))
        self.avals = nn.Parameter(avals, requires_grad=False)
        self.bvals = nn.Parameter(bvals, requires_grad=False)

    def forward(self, tensor) -> torch.Tensor:
        """
        Apply positional encoding to the input.
        """

        return torch.cat(
            [
                self.avals * torch.sin((2.0 * np.pi * tensor) @ self.bvals.T),
                self.avals * torch.cos((2.0 * np.pi * tensor) @ self.bvals.T),
            ],
            dim=-1,
        )


class MultiScaleModulatedFourierFeatures(nn.Module):
    def __init__(
        self,
        input_dim=5,
        output_dim=1,
        num_frequencies=8,
        latent_dim=4,
        width=256,
        depth=3,
        depth_hnn=3,
        include_input=True,
        scales=[1, 5],
        conditioning_type="shift_modulation",
        num_heads=4,
        scalar_hidden_dim=128,
        scalar_out_dim=6,
    ):
        super().__init__()

        self.include_input = include_input
        self.scales = scales
        self.conditioning_type = conditioning_type

        self.embeddings = nn.ModuleList(
            [
                GaussianEncoding(embedding_size=num_frequencies * 2, scale=scale, dims=input_dim)
                for scale in scales
            ]
        )
        embed_dim = num_frequencies * 2
        embed_dim += input_dim if include_input else 0
        self.in_channels = [embed_dim] + [width] * (depth - 1)

        self.out_channels = [width] * (depth - 1) + [width]
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_channels[k], self.out_channels[k]) for k in range(depth)]
        )
        self.final_linear = nn.Linear(len(self.scales) * width, output_dim)
        self.depth = depth
        self.hidden_dim = width
        self.depth_hnn = depth_hnn
        self.num_modulations = self.hidden_dim * (self.depth - 1)

        self.cond_to_modulation = LatentToModulation(
            self.latent_dim, self.num_modulations, dim_hidden=256, num_layers=self.depth_hnn
        )
        # Choose conditioning type
        if self.conditioning_type == "shift_modulation":
            self.conditioning = shift_modulation
        else:
            raise ValueError(
                "Invalid conditioning_type. Choose 'shift_modulation' or 'cross_attention'."
            )

        if scalar_out_dim is not None and scalar_out_dim > 0:
            self.scalar_head = nn.Sequential(
                nn.Linear(width * (depth - 1), scalar_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(scalar_hidden_dim, scalar_out_dim),
            )

    def modulated_forward(self, x, z):
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])

        features = self.cond_to_modulation(z)
        positions = [embedding(x) for embedding in self.embeddings]

        if self.include_input:
            positions = [torch.cat([pos, x], axis=-1) for pos in positions]

        if self.conditioning_type == "shift_modulation":
            pre_outs = [
                self.conditioning(pos, features, self.layers[:-1], torch.relu) for pos in positions
            ]
        outs = [self.layers[-1](pre_out) for pre_out in pre_outs]

        # Concatenate the outputs from each scale
        concatenated_out = torch.cat(outs, axis=-1)

        # A final linear layer to combine multi-scale outputs
        final_out = self.final_linear(concatenated_out)

        return final_out.view(*x_shape, final_out.shape[-1])

    def predict_scalars(self, z):
        """
        cond: (batch_size, latent_dim) graph‐level condition vectors
        returns: (batch_size, scalar_out_dim)
        """
        # run hyper‐network on each cond vector
        mod_feats = self.cond_to_modulation(z)  # (batch_size, M)
        return self.scalar_head(mod_feats)


# --- staging entry points --------------------------------------------------------------------


class _MarioForward(nn.Module):
    """Thin torchlens-friendly forward() wrapper around the real
    MultiScaleModulatedFourierFeatures.modulated_forward(x, z) two-tensor call, matching the
    repo's own `models.py` `__main__` self-test invocation. No architecture code is touched;
    this only adapts the call convention (forward(x, z) -> modulated_forward(x, z)) so a plain
    trace(model, (x, z)) call works."""

    def __init__(self, mario):
        super().__init__()
        self.mario = mario

    def forward(self, x, z):
        return self.mario.modulated_forward(x, z)


def build_mario():
    """Tiny random-init MultiScaleModulatedFourierFeatures, matching the repo's own
    models.py __main__ self-test configuration (input_dim=5 spatial+flow coordinates,
    shift_modulation conditioning, two Fourier-feature scales)."""
    torch.manual_seed(0)
    mario = MultiScaleModulatedFourierFeatures(
        input_dim=5,
        output_dim=1,
        num_frequencies=8,
        latent_dim=4,
        width=32,
        depth=3,
        depth_hnn=2,
        include_input=True,
        scales=[1, 5],
        conditioning_type="shift_modulation",
        scalar_out_dim=6,
    )
    return _MarioForward(mario)


def example_input_mario():
    """Real two-tensor input matching models.py's own __main__ self-test: (positions, z)
    where positions are (batch*n_samples, input_dim) coordinates and z is the per-sample
    latent conditioning vector."""
    torch.manual_seed(0)
    batch_size = 2
    n_samples = 10
    input_dim = 5
    latent_dim = 4

    positions = torch.randn(batch_size * n_samples, input_dim)
    latent_features = torch.randn(batch_size * n_samples, latent_dim)

    return (positions, latent_features)


MENAGERIE_ENTRIES = [
    (
        "MARIO",
        "build_mario",
        "example_input_mario",
        "2025",
        "SOURCE_AVAILABLE",
    ),
]
