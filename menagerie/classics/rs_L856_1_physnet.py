# FAITHFUL PORT of https://github.com/MMunibas/PhysNet @ master (original framework: TensorFlow 1.x)
#
# PhysNet: A Neural Network for Predicting Energies, Forces, Dipole Moments and
# Partial Charges (Unke & Meuwly, JCTC 2019). The official repo is TF1.x
# (tf.variable_scope / tf.placeholder_with_default / tf.segment_sum) and cannot
# run in a modern base-torch env, so this is a faithful architectural
# transcription of the real repo's modules into self-contained torch:
#   - neural_network/layers/RBFLayer.py         -> RBFLayer (exponential-Bernstein
#                                                   -style RBF: learnable centers/
#                                                   widths applied to exp(-D), with
#                                                   the repo's polynomial cutoff_fn)
#   - neural_network/layers/DenseLayer.py       -> DenseLayer (plain Linear + act)
#   - neural_network/layers/ResidualLayer.py    -> ResidualLayer (pre-activation
#                                                   residual MLP block)
#   - neural_network/layers/InteractionLayer.py -> InteractionLayer (message =
#                                                   dense_i(x) + scatter-sum of
#                                                   rbf-gated dense_j(x) over
#                                                   neighbors, then residual stack
#                                                   and gated update x <- u*x + dense(m))
#   - neural_network/layers/InteractionBlock.py -> InteractionBlock (InteractionLayer
#                                                   + a stack of atomic ResidualLayers)
#   - neural_network/layers/OutputBlock.py      -> OutputBlock (residual stack +
#                                                   final Dense to [energy, charge])
#   - neural_network/activation_fn.py           -> shifted_softplus
#   - neural_network/NeuralNetwork.py           -> PhysNet.atomic_properties /
#                                                   energy_from_scaled_atomic_properties
#                                                   (electrostatic term; the D3
#                                                   dispersion term and the periodic
#                                                   short/long-range distance split
#                                                   are dropped here -- this trace
#                                                   uses a single non-periodic edge
#                                                   list, matching the repo's
#                                                   `use_dispersion=False` /
#                                                   `sr_idx_i is None` code paths, so
#                                                   no architecture is invented, only
#                                                   optional branches are omitted)
#
# MENAGERIE_ZOO = "ported-pytorch"

from __future__ import annotations

import numpy as np
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


def shifted_softplus(x: torch.Tensor) -> torch.Tensor:
    # neural_network/activation_fn.py: shifted_softplus
    return nn.functional.softplus(x) - float(np.log(2.0))


def _softplus_inverse(x: np.ndarray) -> np.ndarray:
    # neural_network/NeuralNetwork.py: softplus_inverse
    return x + np.log(-np.expm1(-x))


# ------------------------------------------------------------------
# neural_network/layers/DenseLayer.py  (verbatim structure; weight init
# simplified to nn.Linear defaults instead of the repo's semi-orthogonal
# glorot initializer, since only the traced *architecture* -- not the
# training-time initialization statistics -- is in scope)
# ------------------------------------------------------------------
class DenseLayer(nn.Module):
    def __init__(self, n_in, n_out, activation_fn=None, use_bias=True):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out, bias=use_bias)
        self.activation_fn = activation_fn

    def forward(self, x):
        y = self.linear(x)
        if self.activation_fn is not None:
            y = self.activation_fn(y)
        return y


# ------------------------------------------------------------------
# neural_network/layers/ResidualLayer.py  (verbatim: pre-activation
# dense -> dense residual add)
# ------------------------------------------------------------------
class ResidualLayer(nn.Module):
    def __init__(self, n_in, n_out, activation_fn=None):
        super().__init__()
        self.activation_fn = activation_fn
        self.dense = DenseLayer(n_in, n_out, activation_fn=activation_fn)
        self.residual = DenseLayer(n_out, n_out, activation_fn=None)

    def forward(self, x):
        y = self.activation_fn(x) if self.activation_fn is not None else x
        return x + self.residual(self.dense(y))


# ------------------------------------------------------------------
# neural_network/layers/RBFLayer.py  (verbatim: cosine polynomial cutoff
# times a Gaussian-in-exp(-D) radial basis with learnable centers/widths)
# ------------------------------------------------------------------
class RBFLayer(nn.Module):
    def __init__(self, K: int, cutoff: float):
        super().__init__()
        self.K = K
        self.cutoff = cutoff
        centers = _softplus_inverse(np.linspace(1.0, np.exp(-cutoff), K))
        self.centers = nn.Parameter(torch.as_tensor(centers, dtype=torch.float32))
        widths = np.full(K, _softplus_inverse((0.5 / ((1.0 - np.exp(-cutoff)) / K)) ** 2))
        self.widths = nn.Parameter(torch.as_tensor(widths, dtype=torch.float32))

    def cutoff_fn(self, D):
        x = D / self.cutoff
        x3 = x**3
        x4 = x3 * x
        x5 = x4 * x
        return torch.where(x < 1, 1 - 6 * x5 + 15 * x4 - 10 * x3, torch.zeros_like(x))

    def forward(self, D):
        D = D.unsqueeze(-1)
        centers = nn.functional.softplus(self.centers)
        widths = nn.functional.softplus(self.widths)
        rbf = self.cutoff_fn(D) * torch.exp(-widths * (torch.exp(-D) - centers) ** 2)
        return rbf


# ------------------------------------------------------------------
# neural_network/layers/InteractionLayer.py  (verbatim: k2f RBF gate,
# dense_i/dense_j message split, scatter-sum over neighbors idx_j -> idx_i,
# residual stack, gated update x <- u*x + dense(m))
# ------------------------------------------------------------------
class InteractionLayer(nn.Module):
    def __init__(self, K, F, num_residual, activation_fn=None):
        super().__init__()
        self.activation_fn = activation_fn
        self.k2f = DenseLayer(K, F, use_bias=False)
        self.dense_i = DenseLayer(F, F, activation_fn=activation_fn)
        self.dense_j = DenseLayer(F, F, activation_fn=activation_fn)
        self.residual_layer = nn.ModuleList(
            [ResidualLayer(F, F, activation_fn=activation_fn) for _ in range(num_residual)]
        )
        self.dense = DenseLayer(F, F)
        self.u = nn.Parameter(torch.ones(F))

    def forward(self, x, rbf, idx_i, idx_j):
        xa = self.activation_fn(x) if self.activation_fn is not None else x
        g = self.k2f(rbf)
        xi = self.dense_i(xa)
        messages = g * self.dense_j(xa).index_select(0, idx_j)
        xj = torch.zeros_like(xi).index_add_(0, idx_i, messages)
        m = xi + xj
        for layer in self.residual_layer:
            m = layer(m)
        if self.activation_fn is not None:
            m = self.activation_fn(m)
        return self.u * x + self.dense(m)


# ------------------------------------------------------------------
# neural_network/layers/InteractionBlock.py  (verbatim: one InteractionLayer
# followed by a stack of atomic ResidualLayers)
# ------------------------------------------------------------------
class InteractionBlock(nn.Module):
    def __init__(self, K, F, num_residual_atomic, num_residual_interaction, activation_fn=None):
        super().__init__()
        self.interaction = InteractionLayer(
            K, F, num_residual_interaction, activation_fn=activation_fn
        )
        self.residual_layer = nn.ModuleList(
            [ResidualLayer(F, F, activation_fn=activation_fn) for _ in range(num_residual_atomic)]
        )

    def forward(self, x, rbf, idx_i, idx_j):
        x = self.interaction(x, rbf, idx_i, idx_j)
        for layer in self.residual_layer:
            x = layer(x)
        return x


# ------------------------------------------------------------------
# neural_network/layers/OutputBlock.py  (verbatim: residual stack then a
# final Dense to 2 outputs = [energy, charge] per atom)
# ------------------------------------------------------------------
class OutputBlock(nn.Module):
    def __init__(self, F, num_residual, activation_fn=None):
        super().__init__()
        self.activation_fn = activation_fn
        self.residual_layer = nn.ModuleList(
            [ResidualLayer(F, F, activation_fn=activation_fn) for _ in range(num_residual)]
        )
        self.dense = DenseLayer(F, 2, use_bias=False)

    def forward(self, x):
        for layer in self.residual_layer:
            x = layer(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return self.dense(x)


# ------------------------------------------------------------------
# neural_network/NeuralNetwork.py  (ported: PhysNet top-level module.
# atomic_properties() + energy_from_scaled_atomic_properties() combined
# into one forward(); electrostatic term kept (kehalf/Coulomb switch),
# D3 dispersion term dropped -- an optional additive contribution in the
# real repo (use_dispersion=False path), not part of the core message-
# passing architecture)
# ------------------------------------------------------------------
class PhysNet(nn.Module):
    def __init__(
        self,
        F=64,
        K=32,
        sr_cut=10.0,
        num_blocks=3,
        num_residual_atomic=2,
        num_residual_interaction=2,
        num_residual_output=1,
        kehalf=7.199822675975274,
        max_z=95,
    ):
        super().__init__()
        self.F = F
        self.K = K
        self.sr_cut = sr_cut
        self.num_blocks = num_blocks
        self.kehalf = kehalf

        self.embeddings = nn.Parameter(torch.empty(max_z, F).uniform_(-np.sqrt(3), np.sqrt(3)))
        self.rbf_layer = RBFLayer(K, sr_cut)

        self.Eshift = nn.Parameter(torch.zeros(max_z))
        self.Escale = nn.Parameter(torch.ones(max_z))
        self.Qshift = nn.Parameter(torch.zeros(max_z))
        self.Qscale = nn.Parameter(torch.ones(max_z))

        self.interaction_block = nn.ModuleList(
            [
                InteractionBlock(
                    K,
                    F,
                    num_residual_atomic,
                    num_residual_interaction,
                    activation_fn=shifted_softplus,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_block = nn.ModuleList(
            [
                OutputBlock(F, num_residual_output, activation_fn=shifted_softplus)
                for _ in range(num_blocks)
            ]
        )

    def _switch(self, Dij):
        cut = self.sr_cut / 2
        x = Dij / cut
        x3 = x * x * x
        x4 = x3 * x
        x5 = x4 * x
        return torch.where(Dij < cut, 6 * x5 - 15 * x4 + 10 * x3, torch.ones_like(Dij))

    def electrostatic_energy_per_atom(self, Dij, Qa, idx_i, idx_j):
        Qi = Qa.index_select(0, idx_i)
        Qj = Qa.index_select(0, idx_j)
        DijS = torch.sqrt(Dij * Dij + 1.0)
        switch = self._switch(Dij)
        cswitch = 1.0 - switch
        Eele_ordinary = 1.0 / Dij
        Eele_shielded = 1.0 / DijS
        return self.kehalf * Qi * Qj * (cswitch * Eele_shielded + switch * Eele_ordinary)

    def forward(self, Z, R, idx_i, idx_j, batch_seg=None):
        if batch_seg is None:
            batch_seg = torch.zeros_like(Z)

        Ri = R.index_select(0, idx_i)
        Rj = R.index_select(0, idx_j)
        Dij = torch.sqrt(nn.functional.relu(((Ri - Rj) ** 2).sum(-1)))

        rbf = self.rbf_layer(Dij)
        x = self.embeddings.index_select(0, Z)

        Ea = torch.zeros(Z.shape[0], dtype=x.dtype, device=x.device)
        Qa = torch.zeros(Z.shape[0], dtype=x.dtype, device=x.device)
        for i in range(self.num_blocks):
            x = self.interaction_block[i](x, rbf, idx_i, idx_j)
            out = self.output_block[i](x)
            Ea = Ea + out[:, 0]
            Qa = Qa + out[:, 1]

        Ea = self.Escale.index_select(0, Z) * Ea + self.Eshift.index_select(0, Z) + 0.0 * R.sum(-1)
        Qa = self.Qscale.index_select(0, Z) * Qa + self.Qshift.index_select(0, Z)

        # scaled_charges: rescale so total charge per batch matches Q_tot=0
        n_batches = int(batch_seg.max().item()) + 1
        Na_per_batch = torch.zeros(n_batches, dtype=x.dtype, device=x.device).index_add_(
            0, batch_seg, torch.ones_like(Qa)
        )
        Qsum_per_batch = torch.zeros(n_batches, dtype=x.dtype, device=x.device).index_add_(
            0, batch_seg, Qa
        )
        Qa = Qa + (-Qsum_per_batch / Na_per_batch).index_select(0, batch_seg)

        Ea = Ea + self.electrostatic_energy_per_atom(Dij, Qa, idx_i, idx_j).new_zeros(
            Ea.shape
        ).index_add_(0, idx_i, self.electrostatic_energy_per_atom(Dij, Qa, idx_i, idx_j))

        energy_per_batch = torch.zeros(n_batches, dtype=x.dtype, device=x.device).index_add_(
            0, batch_seg, Ea
        )
        return energy_per_batch.squeeze(), Qa


# ------------------------------------------------------------------
# Menagerie staging entrypoints
# ------------------------------------------------------------------
def build_physnet():
    torch.manual_seed(0)
    return PhysNet(
        F=16,
        K=8,
        sr_cut=10.0,
        num_blocks=2,
        num_residual_atomic=1,
        num_residual_interaction=1,
        max_z=10,
    )


def example_input_physnet():
    torch.manual_seed(0)
    # 6 atoms across 2 molecules (batch_seg), fully-connected edge list per molecule.
    Z = torch.tensor([1, 1, 8, 1, 1, 8], dtype=torch.long)
    R = torch.rand(6, 3) * 3.0
    batch_seg = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    # fully connect atoms within each molecule (3 atoms -> 6 directed edges each)
    idx_i, idx_j = [], []
    for mol_atoms in ([0, 1, 2], [3, 4, 5]):
        for a in mol_atoms:
            for b in mol_atoms:
                if a != b:
                    idx_i.append(a)
                    idx_j.append(b)
    idx_i = torch.tensor(idx_i, dtype=torch.long)
    idx_j = torch.tensor(idx_j, dtype=torch.long)
    return (Z, R, idx_i, idx_j, batch_seg)


MENAGERIE_ENTRIES = [
    ("physnet", "build_physnet", "example_input_physnet", 2019, MENAGERIE_ZOO),
]
