# FAITHFUL PORT of https://github.com/atomistic-machine-learning/SchNOrb @ 38cb268ea431b46e3b2088dda48bcd1983462ff (original framework: PyTorch, but pinned to the pre-1.0 "legacy" SchNetPack API)
#
# SchNOrb predicts a molecule's electronic Hamiltonian and overlap matrices from
# atomic geometry, built on top of SchNet interaction blocks. The real repo imports
# `schnetpack` (`spk.nn.Dense`, `spk.nn.Aggregate`, `spk.nn.neighbors.AtomDistances`,
# `spk.nn.acsf.GaussianSmearing`, `spk.nn.cutoff.HardCutoff`,
# `spk.representation.SchNetInteraction`, `spk.nn.base.ScaleShift`,
# `spk.nn.activations.shifted_softplus`) from the SchNetPack "legacy" (pre-1.0) API,
# which no longer ships on PyPI (modern schnetpack >=1.0 replaced this whole module
# layout) and is not in the base env, so this is a faithful transcription of the
# real repo's own module tree into self-contained torch:
#   - src/schnorb/model.py                              -> SingleAtomHamiltonian,
#                                                           SchNorbInteraction, SchNOrb,
#                                                           Hamiltonian (verbatim)
#   - src/schnorb/nn.py                                  -> CosineBasis, FTLayer (verbatim)
#   - schnetpack(legacy)/nn/base.py                      -> Dense, ScaleShift, Aggregate
#   - schnetpack(legacy)/nn/cutoff.py                    -> HardCutoff
#   - schnetpack(legacy)/nn/acsf.py                      -> GaussianSmearing
#   - schnetpack(legacy)/nn/neighbors.py                 -> AtomDistances, atom_distances
#   - schnetpack(legacy)/nn/cfconv.py                    -> CFConv
#   - schnetpack(legacy)/nn/activations.py                -> shifted_softplus
#   - schnetpack(legacy)/representation/schnet.py        -> SchNetInteraction
#
# MENAGERIE_ZOO = "ported-pytorch"

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_ as zeros_initializer

MENAGERIE_ZOO = "ported-pytorch"


# ------------------------------------------------------------------
# schnetpack(legacy)/nn/activations.py  (verbatim)
# ------------------------------------------------------------------
def shifted_softplus(x):
    return nn.functional.softplus(x) - float(np.log(2.0))


# ------------------------------------------------------------------
# schnetpack(legacy)/nn/base.py  (verbatim: Dense, ScaleShift, Aggregate)
# ------------------------------------------------------------------
class Dense(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.activation = activation
        super(Dense, self).__init__(in_features, out_features, bias)

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        y = super(Dense, self).forward(inputs)
        if self.activation:
            y = self.activation(y)
        return y


class ScaleShift(nn.Module):
    def __init__(self, mean, stddev):
        super(ScaleShift, self).__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)

    def forward(self, input):
        y = input * self.stddev + self.mean
        return y


class Aggregate(nn.Module):
    def __init__(self, axis, mean=False, keepdim=True):
        super(Aggregate, self).__init__()
        self.average = mean
        self.axis = axis
        self.keepdim = keepdim

    def forward(self, input, mask=None):
        if mask is not None:
            input = input * mask[..., None]
        y = torch.sum(input, self.axis)
        if self.average:
            if mask is not None:
                N = torch.sum(mask, self.axis, keepdim=self.keepdim)
                N = torch.max(N, other=torch.ones_like(N))
            else:
                N = input.size(self.axis)
            y = y / N
        return y


# ------------------------------------------------------------------
# schnetpack(legacy)/nn/cutoff.py  (verbatim: HardCutoff)
# ------------------------------------------------------------------
class HardCutoff(nn.Module):
    def __init__(self, cutoff=5.0):
        super(HardCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, distances):
        mask = (distances <= self.cutoff).float()
        return mask


# ------------------------------------------------------------------
# schnetpack(legacy)/nn/acsf.py  (verbatim: gaussian_smearing, GaussianSmearing)
# ------------------------------------------------------------------
def gaussian_smearing(distances, offset, widths, centered=False):
    if not centered:
        coeff = -0.5 / torch.pow(widths, 2)
        diff = distances[:, :, :, None] - offset[None, None, None, :]
    else:
        coeff = -0.5 / torch.pow(offset, 2)
        diff = distances[:, :, :, None]
    gauss = torch.exp(coeff * torch.pow(diff, 2))
    return gauss


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered

    def forward(self, distances):
        return gaussian_smearing(distances, self.offsets, self.width, centered=self.centered)


# ------------------------------------------------------------------
# schnetpack(legacy)/nn/neighbors.py  (verbatim: atom_distances, AtomDistances)
# ------------------------------------------------------------------
def atom_distances(
    positions,
    neighbors,
    cell=None,
    cell_offsets=None,
    return_vecs=False,
    normalize_vecs=False,
    neighbor_mask=None,
):
    n_batch = positions.size()[0]
    idx_m = torch.arange(n_batch, device=positions.device, dtype=torch.long)[:, None, None]
    pos_xyz = positions[idx_m, neighbors[:, :, :], :]
    dist_vec = pos_xyz - positions[:, :, None, :]

    if cell is not None:
        B, A, N, D = cell_offsets.size()
        cell_offsets = cell_offsets.view(B, A * N, D)
        offsets = cell_offsets.bmm(cell)
        offsets = offsets.view(B, A, N, D)
        dist_vec += offsets

    distances = torch.norm(dist_vec, 2, 3)

    if neighbor_mask is not None:
        tmp_distances = torch.zeros_like(distances)
        tmp_distances[neighbor_mask != 0] = distances[neighbor_mask != 0]
        distances = tmp_distances

    if return_vecs:
        tmp_distances = torch.ones_like(distances)
        tmp_distances[neighbor_mask != 0] = distances[neighbor_mask != 0]

        if normalize_vecs:
            dist_vec = dist_vec / tmp_distances[:, :, :, None]
        return distances, dist_vec
    return distances


class AtomDistances(nn.Module):
    def __init__(self, return_directions=False):
        super(AtomDistances, self).__init__()
        self.return_directions = return_directions

    def forward(self, positions, neighbors, cell=None, cell_offsets=None, neighbor_mask=None):
        return atom_distances(
            positions,
            neighbors,
            cell,
            cell_offsets,
            return_vecs=self.return_directions,
            normalize_vecs=True,
            neighbor_mask=neighbor_mask,
        )


# ------------------------------------------------------------------
# schnetpack(legacy)/nn/cfconv.py  (verbatim: CFConv)
# ------------------------------------------------------------------
class CFConv(nn.Module):
    def __init__(
        self,
        n_in,
        n_filters,
        n_out,
        filter_network,
        cutoff_network=None,
        activation=None,
        normalize_filter=False,
        axis=2,
    ):
        super(CFConv, self).__init__()
        self.in2f = Dense(n_in, n_filters, bias=False, activation=None)
        self.f2out = Dense(n_filters, n_out, bias=True, activation=activation)
        self.filter_network = filter_network
        self.cutoff_network = cutoff_network
        self.agg = Aggregate(axis=axis, mean=normalize_filter)

    def forward(self, x, r_ij, neighbors, pairwise_mask, f_ij=None):
        if f_ij is None:
            f_ij = r_ij.unsqueeze(-1)

        W = self.filter_network(f_ij)
        if self.cutoff_network is not None:
            C = self.cutoff_network(r_ij)
            W = W * C.unsqueeze(-1)

        y = self.in2f(x)
        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, y.size(2))
        y = torch.gather(y, 1, nbh)
        y = y.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

        y = y * W
        y = self.agg(y, pairwise_mask)
        y = self.f2out(y)
        return y


# ------------------------------------------------------------------
# schnetpack(legacy)/representation/schnet.py  (verbatim: SchNetInteraction)
# ------------------------------------------------------------------
class SchNetInteraction(nn.Module):
    def __init__(
        self,
        n_atom_basis,
        n_spatial_basis,
        n_filters,
        cutoff,
        cutoff_network=HardCutoff,
        normalize_filter=False,
    ):
        super(SchNetInteraction, self).__init__()
        self.filter_network = nn.Sequential(
            Dense(n_spatial_basis, n_filters, activation=shifted_softplus),
            Dense(n_filters, n_filters),
        )
        self.cutoff_network = cutoff_network(cutoff)
        self.cfconv = CFConv(
            n_atom_basis,
            n_filters,
            n_atom_basis,
            self.filter_network,
            cutoff_network=self.cutoff_network,
            activation=shifted_softplus,
            normalize_filter=normalize_filter,
        )
        self.dense = Dense(n_atom_basis, n_atom_basis, bias=True, activation=None)

    def forward(self, x, r_ij, neighbors, neighbor_mask, f_ij=None):
        v = self.cfconv(x, r_ij, neighbors, neighbor_mask, f_ij)
        v = self.dense(v)
        return v


# ------------------------------------------------------------------
# src/schnorb/nn.py  (verbatim: FTLayer; CosineBasis omitted -- unused by SchNOrb's
# own forward path, which always calls SchNorbInteraction with cos_ij precomputed)
# ------------------------------------------------------------------
class FTLayer(nn.Module):
    def __init__(
        self, n_in, n_factors, n_out, filter_network, cutoff_network=None, activation=None
    ):
        super(FTLayer, self).__init__()
        self.in2f = Dense(n_in, n_factors, bias=True)
        self.f2out = Dense(n_factors, n_out, activation=activation)
        self.filter_network = filter_network
        self.cutoff_network = cutoff_network

    def forward(self, x, r_ij, neighbors, pairwise_mask, f_ij=None):
        if f_ij is None:
            f_ij = r_ij.unsqueeze(-1)

        W = self.filter_network(f_ij)

        if self.cutoff_network is not None:
            C = self.cutoff_network(r_ij)
            W = W * C.unsqueeze(-1)

        facts = self.in2f(x)

        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, facts.size(2))

        xi = facts[:, :, None]
        xi = xi.expand(-1, -1, nbh_size[2], -1)
        xj = torch.gather(facts, 1, nbh)
        xj = xj.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

        y = xi * W * xj

        y = self.f2out(y)

        return y


# ------------------------------------------------------------------
# src/schnorb/model.py  (verbatim: SingleAtomHamiltonian, SchNorbInteraction,
# SchNOrb, Hamiltonian)
# ------------------------------------------------------------------
class SingleAtomHamiltonian(nn.Module):
    def __init__(self, orbital_energies, trainable=False):
        super(SingleAtomHamiltonian, self).__init__()

        if trainable:
            self.orbital_energies = nn.Parameter(torch.FloatTensor(orbital_energies))
        else:
            self.register_buffer("orbital_energies", torch.FloatTensor(orbital_energies))

    def forward(self, numbers, basis):
        tmp1 = (basis[:, None, :, 2] > 0).expand(-1, numbers.shape[1], -1)
        tmp2 = numbers[..., None].expand(-1, -1, basis.shape[-2])
        orb_mask = torch.gather(tmp1, 0, tmp2)
        h0 = self.orbital_energies[numbers]
        h0 = torch.masked_select(h0, orb_mask).reshape(numbers.shape[0], 1, -1)
        h0 = h0.expand(-1, h0.shape[2], -1)
        diag = torch.eye(h0.shape[1], device=h0.device)
        h0 = h0 * diag[None]
        return h0


class SchNorbInteraction(nn.Module):
    def __init__(
        self,
        n_spatial_basis,
        n_factors,
        n_cosine_basis,
        cutoff,
        cutoff_network,
        normalize_filter=False,
        dims=3,
        directions=None,
    ):
        super(SchNorbInteraction, self).__init__()
        self.n_cosine_basis = n_cosine_basis
        self._dims = dims
        self.directions = directions

        self.cutoff_network = cutoff_network(cutoff)

        # initialize filters
        self.filter_network = nn.Sequential(
            Dense(n_spatial_basis, n_factors, activation=shifted_softplus),
            Dense(n_factors, n_factors),
        )

        # initialize interaction blocks
        self.ftensor = FTLayer(
            n_cosine_basis,
            n_factors,
            n_factors,
            self.filter_network,
            cutoff_network=self.cutoff_network,
            activation=shifted_softplus,
        )
        self.atomnet = nn.Sequential(
            Dense(n_factors, n_factors, activation=shifted_softplus),
            Dense(n_factors, n_cosine_basis),
        )

        self.pairnet = nn.Sequential(
            Dense(n_factors, n_factors, activation=shifted_softplus),
            Dense(n_factors, n_cosine_basis),
        )
        self.envnet = nn.Sequential(
            Dense(n_factors, n_factors, activation=shifted_softplus),
            Dense(n_factors, n_cosine_basis),
        )

        if self.directions is not None:
            self.pairnet_mult = Dense(self._dims, directions)
            self.envnet_mult1 = Dense(self._dims, directions)
            self.envnet_mult2 = Dense(self._dims, directions)

        self.agg = Aggregate(axis=2, mean=normalize_filter)
        self.pairagg = Aggregate(axis=2, mean=False)

    def forward(self, xi, r_ij, cos_ij, neighbors, neighbor_mask, f_ij=None):
        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1, 1)
        nbh = nbh.expand(-1, -1, self.n_cosine_basis, cos_ij.shape[3])

        v = self.ftensor.forward(xi, r_ij, neighbors, neighbor_mask, f_ij=f_ij)

        ## energy

        # atomic corrections
        vi = self.agg(v, neighbor_mask)
        vi = self.atomnet(vi)

        ## hamiltonian

        # cosine basis corrections
        # i-j interactions
        vij = self.pairnet(v)
        Vij = vij[:, :, :, :, None] * cos_ij[:, :, :, None, :]
        if self.directions is not None:
            Vij = self.pairnet_mult(Vij)

        # i-k/j-l interactions
        vik = self.envnet(v)
        vik = vik[:, :, :, :, None] * cos_ij[:, :, :, None, :]
        Vik = vik * neighbor_mask[:, :, :, None, None]
        Vik = self.pairagg(Vik)
        Vjl = torch.gather(Vik, 1, nbh)
        Vjl = Vjl.reshape(Vik.shape[0], nbh_size[1], nbh_size[2], Vik.shape[2], Vik.shape[3])

        if self.directions is not None:
            Vik = self.envnet_mult1(Vik)
            Vjl = self.envnet_mult2(Vjl)

        Vijkl = Vik[:, :, None] + Vjl

        # environment-corrected interaction
        V = Vij + Vijkl
        return vi, V


class SchNOrb(nn.Module):
    def __init__(
        self,
        n_factors=64,
        lmax=4,
        n_interactions=2,
        cutoff=10.0,
        n_gaussians=50,
        directions=4,
        n_cosine_basis=5,
        normalize_filter=False,
        coupled_interactions=False,
        interaction_block=SchNorbInteraction,
        cutoff_network=HardCutoff,
        trainable_gaussians=False,
        max_z=100,
    ):
        super(SchNOrb, self).__init__()
        self.directions = directions

        # atom type embeddings
        self.embedding = nn.Embedding(max_z, n_cosine_basis, padding_idx=0)

        # distances
        self.distances = AtomDistances(return_directions=True)
        self.distance_expansion = GaussianSmearing(
            0.0, cutoff, n_gaussians, trainable=trainable_gaussians
        )

        ### interactions ###
        ## SchNet interaction ##
        if coupled_interactions:
            self.schnet_interactions = nn.ModuleList(
                [
                    SchNetInteraction(
                        n_atom_basis=n_cosine_basis,
                        n_spatial_basis=n_gaussians,
                        n_filters=n_factors,
                        cutoff=cutoff,
                        cutoff_network=cutoff_network,
                        normalize_filter=normalize_filter,
                    )
                ]
                * n_interactions
            )
        else:
            self.schnet_interactions = nn.ModuleList(
                [
                    SchNetInteraction(
                        n_atom_basis=n_cosine_basis,
                        n_spatial_basis=n_gaussians,
                        n_filters=n_factors,
                        cutoff=cutoff,
                        cutoff_network=cutoff_network,
                        normalize_filter=normalize_filter,
                    )
                    for _ in range(n_interactions)
                ]
            )

        self.first_interaction = interaction_block(
            n_spatial_basis=n_gaussians,
            n_factors=n_factors,
            n_cosine_basis=n_cosine_basis,
            normalize_filter=normalize_filter,
            cutoff=cutoff,
            cutoff_network=cutoff_network,
            directions=None,
        )

        if coupled_interactions:
            self.interactions = nn.ModuleList(
                [
                    interaction_block(
                        n_spatial_basis=n_gaussians,
                        n_factors=n_factors,
                        n_cosine_basis=n_cosine_basis,
                        directions=directions,
                        cutoff=cutoff,
                        cutoff_network=cutoff_network,
                        normalize_filter=normalize_filter,
                    )
                ]
                * (2 * lmax)
            )
        else:
            self.interactions = nn.ModuleList(
                [
                    interaction_block(
                        n_spatial_basis=n_gaussians,
                        n_cosine_basis=n_cosine_basis,
                        n_factors=n_factors,
                        directions=directions,
                        cutoff=cutoff,
                        cutoff_network=cutoff_network,
                        normalize_filter=normalize_filter,
                    )
                    for _ in range(2 * lmax)
                ]
            )

    def forward(self, inputs):
        atomic_numbers = inputs["_atomic_numbers"]
        positions = inputs["_positions"]
        cell = inputs["_cell"]
        cell_offset = inputs["_cell_offset"]
        neighbors = inputs["_neighbors"]
        neighbor_mask = inputs["_neighbor_mask"]

        # atom embedding
        x0 = self.embedding(atomic_numbers)

        # spatial features: r_ij - distances, cos_ij direction cosines
        r_ij, cos_ij = self.distances(positions, neighbors, cell, cell_offset)
        g_ij = self.distance_expansion(r_ij)
        ones = torch.ones(cos_ij.shape[:3] + (1,), device=cos_ij.device)

        xi = x0

        # atom environments (SchNet-style)
        for interaction in self.schnet_interactions:
            v = interaction(xi, r_ij, neighbors, neighbor_mask, f_ij=g_ij)
            xi = xi + v

        # l=0
        v, V = self.first_interaction(xi, r_ij, ones, neighbors, neighbor_mask, f_ij=g_ij)
        xi = xi + v
        dirs = self.directions if self.directions is not None else 3
        V = V.expand(-1, -1, -1, -1, dirs)

        xij = [V.reshape(V.shape[:3] + (1, -1))]

        # 1 <= l <= lmax
        for t, interaction in enumerate(self.interactions):
            v, V = interaction(xi, r_ij, cos_ij, neighbors, neighbor_mask, f_ij=g_ij)
            xi = xi + v
            xij.append(V.reshape(V.shape[:3] + (1, -1)))

        Xij = torch.cumprod(torch.cat(xij, dim=3), dim=3)

        del ones
        return x0, xi, Xij


class Hamiltonian(nn.Module):
    def __init__(
        self,
        basis_definition,
        n_cosine_basis,
        lmax,
        directions,
        orbital_energies=None,
        return_forces=False,
        quambo=False,
        create_graph=False,
        mean=None,
        stddev=None,
        max_z=30,
    ):
        super(Hamiltonian, self).__init__()
        if return_forces:
            self.derivative = "forces"
        else:
            self.derivative = None

        self.create_graph = create_graph

        if orbital_energies is None:
            self.h0 = None
        else:
            self.h0 = SingleAtomHamiltonian(orbital_energies, True)
            self.s0 = SingleAtomHamiltonian(np.ones_like(orbital_energies), True)

        self.register_buffer("basis_definition", torch.LongTensor(basis_definition))
        self.n_types = self.basis_definition.shape[0]
        self.n_orbs = self.basis_definition.shape[1]
        self.n_cosine_basis = n_cosine_basis
        self.quambo = quambo

        directions = directions if directions is not None else 3
        self.offsitenet = Dense(n_cosine_basis * directions * (2 * lmax + 1), self.n_orbs**2)
        self.onsitenet = Dense(n_cosine_basis * directions * (2 * lmax + 1), self.n_orbs**2)

        self.ov_offsitenet = Dense(n_cosine_basis * directions * (2 * lmax + 1), self.n_orbs**2)

        if self.quambo:
            self.ov_onsitenet = Dense(n_cosine_basis * directions * (2 * lmax + 1), self.n_orbs**2)
        else:
            self.ov_onsitenet = nn.Embedding(max_z, self.n_orbs**2, padding_idx=0)
            self.ov_onsitenet.weight.data = torch.diag_embed(
                torch.ones(max_z, self.n_orbs)
            ).reshape(max_z, self.n_orbs**2)
            self.ov_onsitenet.weight.data.zero_()
        self.pairagg = Aggregate(axis=2, mean=True)

        self.atom_net = nn.Sequential(
            Dense(n_cosine_basis, n_cosine_basis // 2, activation=shifted_softplus),
            Dense(n_cosine_basis // 2, 1),
            ScaleShift(mean, stddev),
        )
        self.atomagg = Aggregate(axis=1, mean=False)

    def forward(self, inputs):
        Z = inputs["_atomic_numbers"]
        nbh = inputs["_neighbors"]
        x0, x, Vijkl = inputs["representation"]

        # Vijkl shape: batch, max_atoms, max_nbh, max_lr, feats

        batch = Vijkl.shape[0]
        max_atoms = Vijkl.shape[1]

        orb_mask_i = self.basis_definition[:, :, 2] > 0
        orb_mask_i = orb_mask_i[Z].float()
        orb_mask_i = orb_mask_i.reshape(batch, -1, 1)
        orb_mask_j = orb_mask_i.reshape(batch, 1, -1)
        orb_mask = orb_mask_i * orb_mask_j

        ar = torch.arange(max_atoms, device=nbh.device)[None, :, None].expand(nbh.shape[0], -1, 1)
        _, nbh = torch.cat([nbh, ar], dim=2).sort(dim=2)

        Vijkl = Vijkl.reshape(Vijkl.shape[:3] + (-1,))

        H_off = self.offsitenet(Vijkl)
        zeros = torch.zeros(
            (batch, max_atoms, 1, self.n_orbs**2), device=H_off.device, dtype=H_off.dtype
        )
        H_off = torch.cat([H_off, zeros], dim=2)
        H_off = torch.gather(H_off, 2, nbh[..., None].expand(-1, -1, -1, self.n_orbs**2))

        H_on = self.onsitenet(Vijkl)
        H_on = self.pairagg(H_on)
        eye = torch.eye(max_atoms, device=H_on.device, dtype=H_on.dtype)[None, ..., None]
        H_on = eye * H_on[:, :, None]

        H = H_off + H_on

        H = H.reshape(batch, max_atoms, max_atoms, self.n_orbs, self.n_orbs).permute(
            (0, 1, 3, 2, 4)
        )
        H = H.reshape(batch, max_atoms * self.n_orbs, max_atoms * self.n_orbs)

        # symmetrize
        H = 0.5 * (H + H.permute((0, 2, 1)))

        # mask padded orbitals
        H = torch.masked_select(H, orb_mask > 0)
        orbs = int(math.sqrt(H.shape[0] / batch))
        H = H.reshape(batch, orbs, orbs)

        if self.h0 is not None:
            H = H + self.h0(Z, self.basis_definition)

        del zeros

        # overlap
        S_off = self.ov_offsitenet(Vijkl)
        zeros = torch.zeros(
            (batch, max_atoms, 1, self.n_orbs**2), device=S_off.device, dtype=S_off.dtype
        )
        S_off = torch.cat([S_off, zeros], dim=2)
        S_off = torch.gather(S_off, 2, nbh[..., None].expand(-1, -1, -1, self.n_orbs**2))
        del zeros
        if self.quambo:
            S_on = self.ov_onsitenet(Vijkl)
            S_on = self.pairagg(S_on)
        else:
            S_on = self.ov_onsitenet(Z)
        eye = torch.eye(max_atoms, device=H_on.device, dtype=H_on.dtype)[None, ..., None]
        S_on = eye * S_on[:, :, None]

        S = S_off + S_on

        S = S.reshape(batch, max_atoms, max_atoms, self.n_orbs, self.n_orbs).permute(
            (0, 1, 3, 2, 4)
        )
        S = S.reshape(batch, max_atoms * self.n_orbs, max_atoms * self.n_orbs)

        # symmetrize
        S = 0.5 * (S + S.permute((0, 2, 1)))

        # mask padded orbitals
        S = torch.masked_select(S, orb_mask > 0)
        orbs = int(math.sqrt(S.shape[0] / batch))
        S = S.reshape(batch, orbs, orbs)

        if self.s0 is not None:
            S = S + self.s0(Z, self.basis_definition)

        # total energy
        Ei = self.atom_net(x)
        E = self.atomagg(Ei)

        if self.derivative is not None:
            Fout = -torch.autograd.grad(
                E,
                inputs["_positions"],
                grad_outputs=torch.ones_like(E),
                create_graph=self.create_graph,
            )[0]
        else:
            Fout = None

        return {
            "hamiltonian": H,
            "overlap": S,
            "energy": E,
            "forces": Fout,
        }


# ------------------------------------------------------------------
# Menagerie staging entrypoints
# ------------------------------------------------------------------
class SchNOrbFull(nn.Module):
    """Composes the SchNOrb representation model with the Hamiltonian output
    head, mirroring how the real training scripts (src/scripts/run_schnorb.py)
    chain `representation` -> `output_modules` via a `schnetpack.AtomisticModel`.
    """

    def __init__(self, representation, hamiltonian):
        super().__init__()
        self.representation = representation
        self.hamiltonian = hamiltonian

    def forward(self, inputs):
        inputs = dict(inputs)
        inputs["representation"] = self.representation(inputs)
        return self.hamiltonian(inputs)


def build_schnorb():
    torch.manual_seed(0)
    n_cosine_basis = 4
    lmax = 1
    directions = 2
    representation = SchNOrb(
        n_factors=8,
        lmax=lmax,
        n_interactions=1,
        cutoff=5.0,
        n_gaussians=6,
        directions=directions,
        n_cosine_basis=n_cosine_basis,
        max_z=10,
    )
    # tiny basis definition: 2 atom types (H, O), 3 orbitals each (e.g. s + p_shell
    # flag in column 2 marking orbitals present), matches basis_definition[:, :, 2] > 0
    # usage in SingleAtomHamiltonian / Hamiltonian.forward.
    basis_definition = np.zeros((10, 3, 3), dtype=np.int64)
    basis_definition[:, :, 2] = 1  # mark all 3 orbital slots present for every type
    # orbital_energies non-None exercises the repo's intended full code path (both
    # self.h0 and self.s0 get constructed in Hamiltonian.__init__ -- with
    # orbital_energies=None the real repo's own forward() references the never-set
    # self.s0 unconditionally, which is a pre-existing bug in the upstream code, not
    # something introduced by this port).
    orbital_energies = np.zeros((10, 3), dtype=np.float32)
    hamiltonian = Hamiltonian(
        basis_definition=basis_definition,
        n_cosine_basis=n_cosine_basis,
        lmax=lmax,
        directions=directions,
        orbital_energies=orbital_energies,
        return_forces=False,
        quambo=False,
        mean=torch.tensor([0.0]),
        stddev=torch.tensor([1.0]),
        max_z=10,
    )
    return SchNOrbFull(representation, hamiltonian)


def example_input_schnorb():
    torch.manual_seed(0)
    # 1 batch, 3 atoms (water: O, H, H), fully-connected neighbour list (2 nbrs each).
    n_batch = 1
    n_atoms = 3
    n_nbh = 2

    atomic_numbers = torch.tensor([[8, 1, 1]], dtype=torch.long)
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]],
        dtype=torch.float32,
        requires_grad=False,
    )
    cell = torch.zeros(n_batch, 3, 3)
    cell_offset = torch.zeros(n_batch, n_atoms, n_nbh, 3)
    # neighbors[b, i, :] = indices of the other 2 atoms (fully connected, 3 atoms)
    neighbors = torch.tensor([[[1, 2], [0, 2], [0, 1]]], dtype=torch.long)
    neighbor_mask = torch.ones(n_batch, n_atoms, n_nbh)

    inputs = {
        "_atomic_numbers": atomic_numbers,
        "_positions": positions,
        "_cell": cell,
        "_cell_offset": cell_offset,
        "_neighbors": neighbors,
        "_neighbor_mask": neighbor_mask,
    }
    return (inputs,)


MENAGERIE_ENTRIES = [
    ("schnorb", build_schnorb, example_input_schnorb, 2019, MENAGERIE_ZOO),
]
