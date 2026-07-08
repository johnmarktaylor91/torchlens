# FAITHFUL PORT of NREL/INNfoil @ master (original framework: TensorFlow 2 / tf.keras)
#   INNfoil.py :: class INNfoil(tf.keras.Model) -- dense_layer / forward_pass / inverse_pass
# INNfoil (Yildiz et al., "Bidirectional Surrogate Modeling of Airfoil Aerodynamic and
# Structural Performance Using Invertible Neural Networks", AIAA Journal 2022;
# NREL/INNfoil). An invertible coupling-flow network (RealNVP-style affine coupling
# blocks) mapping between airfoil design variables x and aerodynamic/structural outputs
# y/c/f/z in both directions. The real code is a tf.keras.Model using only Dense +
# LeakyReLU sub-layers, tf.gather-based fixed permutations, and elementwise
# exp/multiply/add coupling updates -- no TF-specific ops beyond those, but it is
# tf.keras (not torch), so it cannot run in the base torch env. Ported faithfully: every
# coupling-block layer of the original `forward_pass` (dense_layer's Dense->LeakyReLU(0.2)
# ->Dense->LeakyReLU(0.2) sub-network, the fixed per-layer node permutation via gather, and
# the exact affine-coupling update order x0 = x0*exp(s1(x1))+t1(x1); x1 = x1*exp(s0(x0))+
# t0(x0)) is reproduced. Only the traced computational graph (`forward_pass`) is ported;
# the MMD-loss / h5-weight-IO / training-loop methods are not part of the network and are
# omitted.
from typing import List

import torch
import torch.nn as nn


class _DenseCouplingBlock(nn.Module):
    """Port of INNfoil.dense_layer: Dense(expand_dim*dim)->LeakyReLU(0.2)->
    Dense(dim)->LeakyReLU(0.2), applied to one half of the coupling split."""

    def __init__(self, in_dim: int, dim: int, expand_dim: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, expand_dim * dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(expand_dim * dim, dim),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class INNfoil(nn.Module):
    """Port of INNfoil.INNfoil.forward_pass: an n_layers-deep RealNVP-style affine
    coupling flow over a fixed-width feature vector (concat of design + performance
    variables). Each layer: permute node order (fixed per-layer permutation), split into
    two halves, apply two chained affine-coupling updates conditioned on s0/t0/s1/t1
    sub-networks, and re-concatenate."""

    def __init__(self, total_dim: int, n_layers: int = 15, expand_dim: int = 3) -> None:
        super().__init__()
        assert total_dim % 2 == 0, "INNfoil coupling flow requires an even feature width"
        self.total_dim = total_dim
        self.n_layers = n_layers
        half_dim = total_dim // 2

        self.s0 = nn.ModuleList(
            [_DenseCouplingBlock(half_dim, half_dim, expand_dim) for _ in range(n_layers)]
        )
        self.t0 = nn.ModuleList(
            [_DenseCouplingBlock(half_dim, half_dim, expand_dim) for _ in range(n_layers)]
        )
        self.s1 = nn.ModuleList(
            [_DenseCouplingBlock(half_dim, half_dim, expand_dim) for _ in range(n_layers)]
        )
        self.t1 = nn.ModuleList(
            [_DenseCouplingBlock(half_dim, half_dim, expand_dim) for _ in range(n_layers)]
        )

        # Fixed per-layer node permutations. The real code defaults to the identity
        # permutation (tf.range) when none is supplied; reproduced here as a registered
        # buffer so it participates in the traced forward exactly like tf.gather does.
        permute_layers = torch.stack([torch.arange(total_dim) for _ in range(n_layers)])
        self.register_buffer("permute_layers", permute_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] % 2 == 0
        c2 = x.shape[1] // 2
        for i in range(self.n_layers):
            x = torch.gather(x, 1, self.permute_layers[i, :].unsqueeze(0).expand(x.shape[0], -1))
            x0, x1 = x[..., :c2], x[..., c2:]

            x0 = x0 * torch.exp(self.s1[i](x1)) + self.t1[i](x1)
            x1 = x1 * torch.exp(self.s0[i](x0)) + self.t0[i](x0)

            x = torch.cat([x0, x1], dim=-1)
        return x


# --- staging build/example helpers (not part of the ported source) --------------------

MENAGERIE_ZOO = "ported-pytorch"

# Feature-width layout from the real repo's data (x: airfoil design vars, y: aero
# outputs, c/f/z: structural/latent outputs) -- total width kept even per the coupling
# flow's split requirement, shrunk for a fast menagerie trace.
_TOTAL_DIM = 16
_N_LAYERS = 4


def build_innfoil() -> nn.Module:
    torch.manual_seed(0)
    model = INNfoil(total_dim=_TOTAL_DIM, n_layers=_N_LAYERS)
    model.eval()
    return model


def example_input_innfoil() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(4, _TOTAL_DIM)


MENAGERIE_ENTRIES: List[tuple] = [
    (
        "INNfoil (Invertible NN for Airfoil)",
        build_innfoil,
        example_input_innfoil,
        2022,
        MENAGERIE_ZOO,
    ),
]
