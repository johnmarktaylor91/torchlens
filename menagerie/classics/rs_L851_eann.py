# SOURCE: vendored from https://github.com/zhangylch/EANN @ 6e8e195c08be8ec350b2eb0de9edf16ae3d79c87 (main)
#
# EANN (Embedded Atom Neural Network) interatomic potential.
# Files combined (each class copied verbatim from the real repo, imports/paths
# fixed minimally so the module is self-contained):
#   - program/inference/density.py        -> GetDensity  (embedded-atom density descriptor)
#   - program/inference/get_neigh.py      -> Neigh_List  (periodic-boundary neighbor list)
#   - program/src/activate.py             -> Relu_like, Tanh_like (learnable activations)
#   - program/src/MODEL.py                -> ResBlock, NNMod (per-element atomic-energy MLP)
#   - program/pes/PES.py                  -> PES (forward composition of the above),
#                                             lightly adapted to take explicit constructor
#                                             args instead of reading `para/input_nn` /
#                                             `para/input_density` text files via exec().
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import BatchNorm1d, Dropout, GELU, LayerNorm, Linear, Sequential, Softplus, SiLU, Tanh
from torch.nn.init import constant_, xavier_uniform_, zeros_

MENAGERIE_ZOO = "vendored-pytorch"


# ------------------------------------------------------------------
# program/src/activate.py  (verbatim)
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
# program/inference/get_neigh.py  (verbatim)
# ------------------------------------------------------------------
class Neigh_List(torch.nn.Module):
    def __init__(self, cutoff: float, nlinked: int):
        # nlinked used for the periodic boundary condition in cell linked list
        super(Neigh_List, self).__init__()
        self.cutoff = cutoff
        self.cell_list = self.cutoff / nlinked
        r1 = torch.arange(-nlinked, nlinked + 1)
        self.linked = torch.cartesian_prod(r1, r1, r1).view(1, -1, 3)

    def forward(self, period_table, coordinates, cell, mass):
        """Compute pairs of atoms that are neighbors

        Arguments:
            pbc (:class:`torch.double`): periodic boundary condition for each dimension
            coordinates (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three vectors
                defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            cutoff (float): the cutoff inside which atoms are considered pairs
            shifts (:class:`torch.Tensor`): tensor of shape (?, 3) storing shifts
        """
        numatom = coordinates.shape[0]
        inv_cell = torch.inverse(cell)
        inv_coor = torch.einsum("ij,jk -> ik", coordinates, inv_cell)
        deviation_coor = torch.round(inv_coor - inv_coor[0])
        inv_coor = inv_coor - deviation_coor
        coordinates[:, :] = torch.einsum("ij,jk -> ik", inv_coor, cell)
        totmass = torch.sum(mass)
        com = torch.einsum("i,ij->j", mass, coordinates) / totmass
        coordinates[:, :] = coordinates - com[None, :]
        num_repeats = torch.ceil(torch.min(self.cutoff / torch.abs(cell), dim=0)[0]).to(torch.int)
        # the number of periodic image in each direction
        num_repeats = period_table * num_repeats
        num_repeats_up = (num_repeats + 1).detach()
        num_repeats_down = (-num_repeats).detach()
        r1 = torch.arange(num_repeats_down[0], num_repeats_up[0], device=coordinates.device)
        r2 = torch.arange(num_repeats_down[1], num_repeats_up[1], device=coordinates.device)
        r3 = torch.arange(num_repeats_down[2], num_repeats_up[2], device=coordinates.device)
        shifts = torch.cartesian_prod(r1, r2, r3).to(coordinates.dtype)
        shifts = torch.einsum("ij,jk ->ik", shifts, cell)
        num_shifts = shifts.shape[0]
        all_shifts = torch.arange(num_shifts, device=coordinates.device)
        all_atoms = torch.arange(numatom, device=coordinates.device)
        prod = torch.cartesian_prod(all_shifts, all_atoms).t().contiguous()
        # used the modified cell_linked algorithm determine the neighbour atoms
        # cut the box with periodic image expand to the ori cell+cutoff in each direction
        # deviation for prevent the min point on the border
        mincoor = torch.min(coordinates, 0)[0] - self.cutoff - 1e-6
        coordinates = coordinates - mincoor
        maxcoor = torch.max(coordinates, 0)[0] + self.cutoff
        image = (coordinates[None, :, :] + shifts[:, None, :]).view(-1, 3)
        # get the index in the range (ori_cell-rc,ori_cell+rs) in  each direction
        mask = torch.nonzero(((image < maxcoor) * (image > 0)).all(1)).view(-1)
        image_mask = image.index_select(0, mask)
        # save the index(shifts, atoms) for each atoms in the modified cell
        prod = prod.index_select(1, mask)
        ori_image_index = torch.floor(coordinates / self.cell_list)
        # the central atoms with its linked cell index
        cell_linked = self.linked.expand(numatom, -1, 3).to(coordinates.device)
        neigh_cell = ori_image_index[:, None, :] + cell_linked
        # all the index for each atoms in the modified cell
        image_index = torch.floor(image_mask / self.cell_list)
        max_cell_index = torch.ceil(maxcoor / self.cell_list)
        neigh_cell_index = (
            neigh_cell[:, :, 2] * max_cell_index[1] * max_cell_index[0]
            + neigh_cell[:, :, 1] * max_cell_index[0]
            + neigh_cell[:, :, 0]
        )
        nimage_index = (
            image_index[:, 2] * max_cell_index[1] * max_cell_index[0]
            + image_index[:, 1] * max_cell_index[0]
            + image_index[:, 0]
        )
        mask_neigh = torch.nonzero(neigh_cell_index[:, None, :] == nimage_index[None, :, None])
        atom_index = mask_neigh[:, 0:2]
        atom_index = atom_index.t().contiguous()
        # step 5, compute distances, and find all pairs within cutoff
        selected_coordinate1 = coordinates.index_select(0, atom_index[0])
        selected_coordinate2 = image_mask.index_select(0, atom_index[1])
        distances = (selected_coordinate1 - selected_coordinate2).norm(2, -1)
        pair_index = torch.nonzero((distances < self.cutoff) * (distances > 0.001)).reshape(-1)
        neigh_index = atom_index[:, pair_index]
        tmp = prod.index_select(1, neigh_index[1])
        neigh_list = torch.vstack((neigh_index[0], tmp[1]))
        shifts = shifts.index_select(0, tmp[0])
        return neigh_list, shifts


# ------------------------------------------------------------------
# program/inference/density.py  (verbatim; the batch-free "inference" variant
# used by program/pes/PES.py, uses torch.einsum directly with no opt_einsum
# dependency)
# ------------------------------------------------------------------
class GetDensity(torch.nn.Module):
    def __init__(self, rs, inta, cutoff, nipsin, norbit):
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
        self.register_buffer("nipsin", torch.tensor([nipsin]))
        npara = [1]
        index_para = torch.tensor([0], dtype=torch.long)
        for i in range(1, nipsin):
            npara.append(int(3**i))
            index_para = torch.cat((index_para, torch.ones((npara[i]), dtype=torch.long) * i))

        self.register_buffer("index_para", index_para)
        # index_para: Type: longTensor,index_para was used to expand the dim of params
        # in nn with para(l)
        # will have the form index_para[0,|1,1,1|,2,2,2,2,2,2,2,2,2|...npara[l]..\...]
        self.params = nn.parameter.Parameter(torch.ones_like(self.rs))

    def gaussian(self, distances, species_):
        # Tensor: rs[nwave],inta[nwave]
        # Tensor: distances[neighbour*numatom*nbatch,1]
        # return: radial[neighbour*numatom*nbatch,nwave]
        distances = distances.view(-1, 1)
        radial = torch.empty(
            (distances.shape[0], self.rs.shape[1]), dtype=distances.dtype, device=distances.device
        )
        for itype in range(self.rs.shape[0]):
            mask = species_ == itype
            ele_index = torch.nonzero(mask).view(-1)
            if ele_index.shape[0] > 0:
                part_radial = torch.exp(
                    -self.inta[itype : itype + 1]
                    * torch.square(
                        distances.index_select(0, ele_index) - self.rs[itype : itype + 1]
                    )
                )
                radial.masked_scatter_(mask.view(-1, 1), part_radial)
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
        orbital = f_cut.view(1, -1)
        angular = torch.empty(
            self.index_para.shape[0], totneighbour, dtype=dist_vec.dtype, device=dist_vec.device
        )
        angular[0] = f_cut
        num = 1
        for ipsin in range(1, int(self.nipsin[0])):
            orbital = torch.einsum("ji,ki -> jki", orbital, dist_vec).reshape(-1, totneighbour)
            angular[num : num + orbital.shape[0]] = orbital
            num += orbital.shape[0]
        return angular

    def forward(self, cart, neigh_list, shifts, species):
        """
        # input cart: coordinates (numatom,3)
        # input shifts: coordinates shift values (unit cell)
        # neigh_list: neighbour list indice
        # species: indice for element of each atom
        """
        numatom = cart.shape[0]
        neigh_species = species.index_select(0, neigh_list[1])
        selected_cart = cart.index_select(0, neigh_list.view(-1)).view(2, -1, 3)
        dist_vec = selected_cart[0] - selected_cart[1] - shifts
        distances = torch.linalg.norm(dist_vec, dim=-1)
        orbital = torch.einsum(
            "ji,ik -> ijk",
            self.angular(dist_vec, self.cutoff_cosine(distances)),
            self.gaussian(distances, neigh_species),
        )
        orb_coeff = self.params.index_select(0, species)
        density = self.obtain_orb_coeff(0, numatom, orbital, neigh_list, orb_coeff)
        return density

    def obtain_orb_coeff(self, iteration: int, numatom: int, orbital, neigh_list, orb_coeff):
        expandpara = orb_coeff.index_select(0, neigh_list[1])
        worbital = torch.einsum("ijk,ik ->ijk", orbital, expandpara)
        sum_worbital = torch.zeros(
            (numatom, orbital.shape[1], self.rs.shape[1]),
            dtype=orb_coeff.dtype,
            device=orb_coeff.device,
        )
        sum_worbital = torch.index_add(sum_worbital, 0, neigh_list[0], worbital)
        density = torch.zeros(
            numatom, self.nipsin[0], self.rs.shape[1], dtype=orbital.dtype, device=orbital.device
        )
        density = torch.index_add(density, 1, self.index_para, torch.square(sum_worbital)).view(
            numatom, -1
        )
        return density


# ------------------------------------------------------------------
# program/src/MODEL.py  (verbatim)
# ------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, nl, dropout_p, actfun, table_norm=True):
        super(ResBlock, self).__init__()
        # activation function used for the nn module
        nhid = len(nl) - 1
        sumdrop = np.sum(dropout_p)
        modules = []
        for i in range(1, nhid):
            if table_norm:
                modules.append(LayerNorm(nl[i]))
            modules.append(actfun(nl[i - 1], nl[i]))
            if sumdrop >= 0.0001:
                modules.append(Dropout(p=dropout_p[i - 1]))
            linear = Linear(nl[i], nl[i + 1])
            if i == nhid - 1:
                zeros_(linear.weight)
            else:
                xavier_uniform_(linear.weight)
            modules.append(linear)
        self.resblock = Sequential(*modules)

    def forward(self, x):
        return self.resblock(x) + x


# ==================for get the atomic energy=======================================
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
        sumdrop = np.sum(dropout_p)
        with torch.no_grad():
            if nblock == 1:
                nl.append(outputneuron)
                nhid = len(nl) - 2
                for ele in atomtype:
                    modules = []
                    for i in range(nhid):
                        linear = Linear(nl[i], nl[i + 1])
                        xavier_uniform_(linear.weight)
                        modules.append(linear)
                        if table_norm:
                            modules.append(LayerNorm(nl[i + 1]))
                        modules.append(actfun(nl[i], nl[i + 1]))
                        if sumdrop >= 0.0001:
                            modules.append(Dropout(p=dropout_p[i]))
                    linear = Linear(nl[nhid], nl[nhid + 1])
                    zeros_(linear.weight)
                    if abs(initpot) > 1e-6:
                        zeros_(linear.bias)
                    modules.append(linear)
                    elemental_nets[ele] = Sequential(*modules)
            else:
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
                ele_den = density[ele_index]
                output[ele_index] = m(ele_den)
        return output


# ------------------------------------------------------------------
# program/pes/PES.py  (adapted: the real repo reads `nblock`/`nl`/`activate`/
# `cutoff`/`nwave`/`nipsin`/`atomtype` from `para/input_nn` + `para/input_density`
# text files via `exec()` at construction time -- here they are ordinary
# constructor arguments carrying the same defaults the repo's example config
# uses, so the module is runnable without any filesystem config. The forward
# pass (Neigh_List -> GetDensity -> NNMod, energy via autograd forces) is
# unchanged from the real PES.forward.)
# ------------------------------------------------------------------
class PES(torch.nn.Module):
    def __init__(
        self,
        atomtype=("H", "O"),
        cutoff=6.0,
        nwave=12,
        nipsin=2,
        nblock=1,
        nl=(64, 32),
        dropout_p=(0.0, 0.0, 0.0),
        activate="Relu_like",
        table_norm=False,
        nlinked=1,
    ):
        super(PES, self).__init__()
        actfun = Relu_like if activate == "Relu_like" else Tanh_like

        nl = list(nl)
        dropout_p = np.array(dropout_p)
        maxnumtype = len(atomtype)

        inta = torch.ones((maxnumtype, nwave))
        rs = torch.stack([torch.linspace(0, cutoff, nwave) for _ in range(maxnumtype)], dim=0)

        density_nipsin = nipsin + 1
        norbit = int(nwave * density_nipsin)
        nl = [int(norbit)] + nl

        outputneuron = 1
        self.cutoff = cutoff
        self.density = GetDensity(rs, inta, cutoff, density_nipsin, norbit)
        self.nnmod = NNMod(
            maxnumtype,
            outputneuron,
            list(atomtype),
            nblock,
            list(nl),
            dropout_p,
            actfun,
            table_norm=table_norm,
        )
        self.neigh_list = Neigh_List(cutoff, nlinked)

    def forward(self, period_table, cart, cell, species, mass):
        cart = cart.detach().clone()
        neigh_list, shifts = self.neigh_list(period_table, cart, cell, mass)
        cart.requires_grad_(True)
        density = self.density(cart, neigh_list, shifts, species)
        output = self.nnmod(density, species) + self.nnmod.initpot
        varene = torch.sum(output)
        (grad,) = torch.autograd.grad([varene], [cart], create_graph=True)
        return varene, -grad


# ------------------------------------------------------------------
# Menagerie staging entrypoints
# ------------------------------------------------------------------
def build_eann():
    torch.manual_seed(0)
    return PES(atomtype=("H", "O"), cutoff=6.0, nwave=8, nipsin=2, nblock=1, nl=(32, 16))


def example_input_eann():
    torch.manual_seed(0)
    # A small periodic water-like box: 4 atoms (2 H, 2 O), cubic 10A cell.
    period_table = torch.tensor([1, 1, 1], dtype=torch.long)
    cart = torch.rand(4, 3) * 10.0
    cell = torch.eye(3) * 10.0
    species = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # 0=H, 1=O (matches atomtype order)
    mass = torch.tensor([1.008, 1.008, 15.999, 15.999])
    return (period_table, cart, cell, species, mass)


MENAGERIE_ENTRIES = [
    ("eann", build_eann, example_input_eann, 2019, MENAGERIE_ZOO),
]
