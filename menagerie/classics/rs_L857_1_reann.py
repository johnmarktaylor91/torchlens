# SOURCE: vendored from https://github.com/bjiangch/REANN @ 6fbbfc28c63c84cae004efe2a7ac1c72a343fd6a (main)
#
# REANN (Recursively Embedded Atom Neural Network) interatomic potential.
# Distinct from EANN (zhangylch/EANN, already vendored as L851_eann.py): REANN adds
# a recursive orbital-coefficient message-passing loop (`ocmod`) and a learned
# `hyper` projection tensor that EANN's simpler embedded-atom density lacks
# (diffed against EANN's density.py to confirm genuine divergence, not a copy).
# Files combined (each class copied verbatim from the real repo, imports/paths
# fixed minimally so the module is self-contained):
#   - reann/src/density.py   -> GetDensity (recursive embedded-atom density descriptor
#                                with orbital-coefficient message-passing `ocmod` loop)
#   - reann/src/activate.py  -> Relu_like, Tanh_like (learnable activations)
#   - reann/src/MODEL.py     -> ResBlock, NNMod (per-element atomic-energy MLP)
#   - reann/src/Property_energy.py -> Property (forward composition: density -> NNMod
#                                     -> summed atomic energy), lightly adapted to
#                                     take an explicit constructor arg list instead of
#                                     the repo's `nnmodlist` positional convention.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import opt_einsum as oe
import torch
from torch import nn
from torch.nn import BatchNorm1d, Dropout, GELU, LayerNorm, Linear, Sequential, Softplus, SiLU, Tanh
from torch.nn.init import constant_, xavier_uniform_, zeros_

MENAGERIE_ZOO = "vendored-pytorch"


# ------------------------------------------------------------------
# reann/src/activate.py  (verbatim)
# ------------------------------------------------------------------
class Relu_like(nn.Module):
    def __init__(self, neuron1, neuron):
        super(Relu_like, self).__init__()
        self.alpha = nn.parameter.Parameter(torch.ones(1, neuron))
        self.beta = nn.parameter.Parameter(torch.ones(1, neuron) / float(neuron1))
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.alpha * self.silu(x * self.beta)


class Tanh_like(nn.Module):
    def __init__(self, neuron1, neuron):
        super(Tanh_like, self).__init__()
        self.alpha = nn.parameter.Parameter(
            torch.ones(1, neuron) / torch.sqrt(torch.tensor([float(neuron1)]))
        )
        self.beta = nn.parameter.Parameter(torch.ones(1, neuron) / float(neuron1))

    def forward(self, x):
        return self.alpha * x / torch.sqrt(1.0 + torch.square(x * self.beta))


# ------------------------------------------------------------------
# reann/src/MODEL.py  (verbatim)
# ------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, nl, dropout_p, actfun, table_norm=True):
        super(ResBlock, self).__init__()
        # activation function used for the nn module
        nhid = len(nl) - 1
        sumdrop = np.sum(dropout_p)
        modules = []
        for i in range(1, nhid):
            modules.append(actfun(nl[i - 1], nl[i]))
            if table_norm:
                modules.append(LayerNorm(nl[i]))
            if sumdrop >= 0.0001:
                modules.append(Dropout(p=dropout_p[i - 1]))
            linear = Linear(nl[i], nl[i + 1])
            if i == nhid - 1:
                zeros_(linear.weight)
            else:
                xavier_uniform_(linear.weight)
            zeros_(linear.bias)
            modules.append(linear)
        self.resblock = Sequential(*modules)

    def forward(self, x):
        return self.resblock(x) + x


class NNMod(torch.nn.Module):
    def __init__(
        self,
        maxnumtype,
        outputneuron,
        atomtype,
        nblock,
        nl,
        dropout_p,
        actfun,
        initpot=0.0,
        table_norm=True,
    ):
        """
        maxnumtype: is the maximal element
        nl: is the neural network structure;
        outputneuron: the number of output neuron of neural network
        atomtype: elements in all systems
        """
        super(NNMod, self).__init__()
        self.register_buffer("initpot", torch.Tensor([initpot]))
        # create the structure of the nn
        self.outputneuron = outputneuron
        elemental_nets = OrderedDict()
        sumdrop = np.sum(dropout_p)  # noqa: F841 (verbatim from repo; unused there too)
        with torch.no_grad():
            nl.append(nl[1])
            nhid = len(nl) - 1
            for ele in atomtype:
                modules = []
                linear = Linear(nl[0], nl[1])
                xavier_uniform_(linear.weight)
                modules.append(linear)
                for iblock in range(nblock):
                    modules.append(*[ResBlock(nl, dropout_p, actfun, table_norm=table_norm)])
                modules.append(actfun(nl[nhid - 1], nl[nhid]))
                linear = Linear(nl[nhid], self.outputneuron)
                zeros_(linear.weight)
                if abs(initpot) > 1e-6:
                    zeros_(linear.bias)
                modules.append(linear)
                elemental_nets[ele] = Sequential(*modules)
        self.elemental_nets = nn.ModuleDict(elemental_nets)

    def forward(self, density, species):
        # elements: dtype: LongTensor store the index of elements of each center atom
        output = torch.zeros(
            (density.shape[0], self.outputneuron), dtype=density.dtype, device=density.device
        )
        for itype, (_, m) in enumerate(self.elemental_nets.items()):
            mask = species == itype
            ele_index = torch.nonzero(mask).view(-1)
            if ele_index.shape[0] > 0:
                ele_den = density[ele_index].contiguous()
                output[ele_index] = m(ele_den)
        return output


# ------------------------------------------------------------------
# reann/src/density.py  (verbatim)
# ------------------------------------------------------------------
class GetDensity(torch.nn.Module):
    def __init__(self, rs, inta, cutoff, neigh_atoms, nipsin, norbit, ocmod_list):
        super(GetDensity, self).__init__()
        """
        rs: tensor[ntype,nwave] float
        inta: tensor[ntype,nwave] float
        nipsin: np.array/list   int
        cutoff: float
        """
        self.rs = nn.parameter.Parameter(rs)
        self.inta = nn.parameter.Parameter(inta)
        self.register_buffer("cutoff", torch.Tensor([cutoff]))
        self.nipsin = nipsin
        npara = [1]
        index_para = torch.tensor([0], dtype=torch.long)
        for i in range(1, nipsin):
            npara.append(np.power(3, i))
            index_para = torch.cat((index_para, torch.ones((npara[i]), dtype=torch.long) * i))
        self.register_buffer("index_para", index_para)

        self.params = nn.parameter.Parameter(torch.ones_like(self.rs) / float(neigh_atoms))
        self.hyper = nn.parameter.Parameter(
            torch.nn.init.xavier_uniform_(torch.rand(self.rs.shape[1], norbit))
            .unsqueeze(0)
            .repeat(nipsin, 1, 1)
        )
        ocmod = OrderedDict()
        for i, m in enumerate(ocmod_list):
            f_oc = "memssage_" + str(i)
            ocmod[f_oc] = m
        self.ocmod = torch.nn.ModuleDict(ocmod)

    def gaussian(self, distances, species_):
        # Tensor: rs[nwave],inta[nwave]
        # Tensor: distances[neighbour*numatom*nbatch,1]
        # return: radial[neighbour*numatom*nbatch,nwave]
        rs = self.rs.index_select(0, species_)
        inta = self.inta.index_select(0, species_)
        radial = torch.exp(inta * torch.square(distances[:, None] - rs))
        return radial

    def cutoff_cosine(self, distances):
        # assuming all elements in distances are smaller than cutoff
        # return cutoff_cosine[neighbour*numatom*nbatch]
        return torch.square(0.5 * torch.cos(distances * (np.pi / self.cutoff)) + 0.5)

    def angular(self, dist_vec, f_cut):
        # Tensor: dist_vec[neighbour*numatom*nbatch,3]
        # return: angular[neighbour*numatom*nbatch,npara[0]+npara[1]+...+npara[ipsin]]
        totneighbour = dist_vec.shape[0]
        dist_vec = dist_vec.permute(1, 0).contiguous()
        angular = [f_cut.view(1, -1)]
        for ipsin in range(1, int(self.nipsin)):
            angular.append(
                torch.einsum("ji,ki -> jki", angular[-1], dist_vec).reshape(-1, totneighbour)
            )
        return torch.vstack(angular)

    def forward(self, cart, numatoms, species, atom_index, shifts):
        """
        # input cart: coordinates (nbatch*numatom,3)
        # input shifts: coordinates shift values (unit cell)
        # input numatoms: number of atoms for each configuration
        # atom_index: neighbour list indice
        # species: indice for element of each atom
        """
        tmp_index = torch.arange(numatoms.shape[0], device=cart.device) * cart.shape[1]
        self_mol_index = tmp_index.view(-1, 1).expand(-1, atom_index.shape[2]).reshape(1, -1)
        cart_ = cart.flatten(0, 1)
        totnatom = cart_.shape[0]
        padding_mask = torch.nonzero((shifts.view(-1, 3) > -1e10).all(1)).view(-1)
        # get the index for the distance less than cutoff (the dimension is reduntant)
        atom_index12 = (atom_index.view(2, -1) + self_mol_index)[:, padding_mask]
        selected_cart = cart_.index_select(0, atom_index12.view(-1)).view(2, -1, 3)
        shift_values = shifts.view(-1, 3).index_select(0, padding_mask)
        dist_vec = selected_cart[0] - selected_cart[1] + shift_values
        distances = torch.linalg.norm(dist_vec, dim=-1)
        species_ = species.index_select(0, atom_index12[1])
        orbital = oe.contract(
            "ji,ik -> ijk",
            self.angular(dist_vec, self.cutoff_cosine(distances)),
            self.gaussian(distances, species_),
            backend="torch",
        )
        orb_coeff = torch.empty((totnatom, self.rs.shape[1]), dtype=cart.dtype, device=cart.device)
        mask = (species > -0.5).view(-1)
        orb_coeff.masked_scatter_(
            mask.view(-1, 1), self.params.index_select(0, species[torch.nonzero(mask).view(-1)])
        )
        hyper = self.hyper.index_select(0, self.index_para)
        density = self.obtain_orb_coeff(totnatom, orbital, atom_index12, orb_coeff, hyper)
        for ioc_loop, (_, m) in enumerate(self.ocmod.items()):
            orb_coeff = orb_coeff + m(density, species)
            density = self.obtain_orb_coeff(totnatom, orbital, atom_index12, orb_coeff, hyper)
        return density

    def obtain_orb_coeff(self, totnatom: int, orbital, atom_index12, orb_coeff, hyper):
        expandpara = orb_coeff.index_select(0, atom_index12[1])
        worbital = oe.contract("ijk,ik->ijk", orbital, expandpara, backend="torch")
        sum_worbital = torch.zeros(
            (totnatom, orbital.shape[1], self.rs.shape[1]),
            dtype=orbital.dtype,
            device=orbital.device,
        )
        sum_worbital = torch.index_add(sum_worbital, 0, atom_index12[0], worbital)
        hyper_worbital = oe.contract("ijk,jkm -> ijm", sum_worbital, hyper, backend="torch")
        return torch.sum(torch.square(hyper_worbital), dim=1)


# ------------------------------------------------------------------
# reann/src/Property_energy.py  (adapted: takes an explicit nnmod list
# instead of unpacking a module-level global `nnmodlist`; forward logic
# is otherwise unchanged)
# ------------------------------------------------------------------
class Property(torch.nn.Module):
    def __init__(self, density, nnmodlist):
        super(Property, self).__init__()
        self.density = density
        self.nnmod = nnmodlist[0]
        if len(nnmodlist) > 1:
            self.nnmod1 = nnmodlist[1]
            self.nnmod2 = nnmodlist[2]

    def forward(self, cart, numatoms, species, atom_index, shifts):
        species = species.view(-1)
        density = self.density(cart, numatoms, species, atom_index, shifts)
        output = self.nnmod(density, species).view(numatoms.shape[0], -1)
        varene = torch.sum(output, dim=1)
        return varene


# ------------------------------------------------------------------
# Menagerie staging entrypoints
# ------------------------------------------------------------------
def _build_reann_property(seed=0):
    torch.manual_seed(seed)
    # small water-like system: 2 element types (H, O), tiny nwave/hidden sizes.
    ntype = 2
    nwave = 4
    cutoff = 5.0
    nipsin = 2  # order of angular momentum expansion (+1 applied like PES.py does)
    nipsin_full = nipsin + 1
    norbit = int(nwave * (nwave + 1) / 2 * nipsin_full)

    rs = torch.stack([torch.linspace(0, cutoff, nwave) for _ in range(ntype)], dim=0)
    inta = torch.ones((ntype, nwave))

    atomtype = ["H", "O"]

    oc_nl = [norbit, 16, 16]
    oc_dropout_p = np.array([0.0, 0.0])
    ocmod_list = [
        NNMod(ntype, nwave, atomtype, 1, list(oc_nl), oc_dropout_p, Relu_like, table_norm=True)
    ]

    density = GetDensity(
        rs, inta, cutoff, neigh_atoms=8, nipsin=nipsin_full, norbit=norbit, ocmod_list=ocmod_list
    )

    nl = [norbit, 16, 16]
    dropout_p = np.array([0.0, 0.0])
    nnmod = NNMod(ntype, 1, atomtype, 1, list(nl), dropout_p, Relu_like, table_norm=True)

    return Property(density, [nnmod])


def build_reann():
    return _build_reann_property()


def example_input_reann():
    torch.manual_seed(0)
    # 1 batch molecule (water, 3 atoms: O, H, H), each atom sees the other 2 as
    # neighbours (neigh_atoms padding size 8, only first 2 slots used per atom).
    nbatch = 1
    numatom = 3
    neigh_atoms = 8

    cart = torch.rand(nbatch, numatom, 3) * 2.0
    numatoms = torch.tensor([numatom], dtype=torch.long)
    # species index into atomtype list ["H", "O"]: O=1, H=0, H=0
    species = torch.tensor([[1, 0, 0]], dtype=torch.long)

    atom_index = torch.zeros((2, nbatch, numatom * neigh_atoms), dtype=torch.long)
    shifts = -1e11 * torch.ones((nbatch, numatom * neigh_atoms, 3))

    # build a simple fully-connected neighbour list per atom (each atom's other
    # 2 neighbours occupy the first 2 neigh slots; remaining slots stay padded).
    for center in range(numatom):
        slot = 0
        for other in range(numatom):
            if other == center:
                continue
            flat = center * neigh_atoms + slot
            atom_index[0, 0, flat] = center
            atom_index[1, 0, flat] = other
            shifts[0, flat, :] = 0.0
            slot += 1

    return (cart, numatoms, species, atom_index, shifts)


MENAGERIE_ENTRIES = [
    ("reann", build_reann, example_input_reann, 2022, MENAGERIE_ZOO),
]
