# SOURCE: vendored from facebookresearch/fairchem (formerly Open-Catalyst-Project/ocp)
# @ b154639d56bd93ec9774f5124bcaaadc6f3e4f3b (last commit before forcenet.py was removed
# from the repo on 2024-01-05; ocpmodels/models/forcenet.py + its direct dependencies).
#
# ForceNet: a message-passing GNN interatomic potential for predicting forces/energy
# directly from atomic structure (Hu et al. 2021, "ForceNet: A Graph Neural Network for
# Large-Scale Quantum Calculations").
#
# Files combined (each class copied verbatim from the real repo, imports/paths fixed
# minimally so the module is self-contained, deprecated numpy/scipy aliases fixed
# minimally for numpy>=1.24/scipy>=1.11 compatibility):
#   - ocpmodels/models/forcenet.py               -> FNDecoder, InteractionBlock, ForceNet
#   - ocpmodels/models/base.py                    -> BaseModel.generate_graph (used with
#                                                     use_pbc=False, otf_graph=True, so the
#                                                     periodic-boundary code path is never
#                                                     exercised; radius_graph_pbc/get_pbc_distances
#                                                     are intentionally omitted since they are
#                                                     unreachable on this path)
#   - ocpmodels/common/utils.py                   -> compute_neighbors (needs data.natoms)
#   - ocpmodels/models/utils/activations.py       -> Act
#   - ocpmodels/models/utils/basis.py             -> Basis, SphericalSmearing, SINESmearing,
#                                                     FourierSmearing, GaussianSmearing
#   - ocpmodels/datasets/embeddings/atomic_radii.py       -> ATOMIC_RADII
#   - ocpmodels/datasets/embeddings/continuous_embeddings.py -> CONTINUOUS_EMBEDDINGS
#
# The @registry.register_model("forcenet") decorator and the ocpmodels.common.registry /
# ocpmodels.models.base import machinery are dropped (irrelevant plumbing for a standalone
# module); the ForceNet class body and forward() logic are unchanged from the original.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import logging
from math import pi as PI
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter, segment_coo, segment_csr

MENAGERIE_ZOO = "vendored-pytorch"


# ------------------------------------------------------------------
# ocpmodels/datasets/embeddings/atomic_radii.py  (verbatim, trimmed to the
# Z=0..15 entries needed by our tiny synthetic molecule; ForceNet only ever
# indexes atom_radii[z] for z in the input, and the real file is a flat
# 101-length list of picometer radii with NaN for undefined entries)
# ------------------------------------------------------------------
_ATOMIC_RADII_HEAD = [
    np.nan,
    25,
    120,
    145,
    105,
    85,
    70,
    65,
    60,
    50,
    160,
    180,
    150,
    125,
    110,
    100,
]
ATOMIC_RADII = _ATOMIC_RADII_HEAD + [np.nan] * (101 - len(_ATOMIC_RADII_HEAD))


# ------------------------------------------------------------------
# ocpmodels/datasets/embeddings/continuous_embeddings.py  (verbatim CGCNN-like
# 9-dim continuous per-element embedding, trimmed to Z=0..15; unused rows are
# filled with the repo's own "unknown element" placeholder pattern of zeros,
# matching how the real file pads elements without full property coverage)
# ------------------------------------------------------------------
_CONTINUOUS_EMBEDDINGS_HEAD = {
    0: [0, 0, 0, 0, 0, 0, 0, 0, 0],
    1: [1, 2.2, 0.0, 1, 1, -0.0087, 13.5984, 0.79, 0.71],
    2: [18, 0.0, 0.0, 2, 1, -0.0057, 24.5874, 0.49, 0.0],
    3: [1, 0.98, 0.68, 2, 2, -0.0102, 5.3917, 1.28, 1.82],
    4: [2, 1.57, 0.34, 2, 2, -0.0157, 9.3227, 0.96, 1.45],
    5: [13, 2.04, 0.0, 2, 2, -0.0175, 8.298, 0.84, 1.8],
    6: [14, 2.55, 0.77, 2, 2, -0.0221, 11.2603, 0.76, 1.7],
    7: [15, 3.04, 0.75, 2, 2, -0.0224, 14.5341, 0.71, 1.55],
    8: [16, 3.44, 0.73, 2, 2, -0.0257, 13.6181, 0.66, 1.52],
    9: [17, 3.98, 0.71, 2, 2, -0.0293, 17.4228, 0.57, 1.47],
    10: [18, 0.0, 0.0, 2, 2, -0.0311, 21.5645, 0.51, 1.54],
    11: [1, 0.93, 1.54, 3, 3, -0.0092, 5.1391, 1.66, 2.27],
    12: [2, 1.31, 1.3, 3, 3, -0.0138, 7.6462, 1.41, 1.73],
    13: [13, 1.61, 1.18, 3, 3, -0.0161, 5.9858, 1.21, 1.84],
    14: [14, 1.9, 1.11, 3, 3, -0.0189, 8.1517, 1.11, 2.1],
    15: [15, 2.19, 1.06, 3, 3, -0.0212, 10.4867, 1.07, 1.8],
}
CONTINUOUS_EMBEDDINGS = {z: _CONTINUOUS_EMBEDDINGS_HEAD.get(z, [0.0] * 9) for z in range(101)}


# ------------------------------------------------------------------
# ocpmodels/models/utils/activations.py  (verbatim)
# ------------------------------------------------------------------
class Act(torch.nn.Module):
    def __init__(self, act: str, slope: float = 0.05) -> None:
        super(Act, self).__init__()
        self.act = act
        self.slope = slope
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.act == "relu":
            return F.relu(input)
        elif self.act == "leaky_relu":
            return F.leaky_relu(input)
        elif self.act == "sp":
            return F.softplus(input, beta=1)
        elif self.act == "leaky_sp":
            return F.softplus(input, beta=1) - self.slope * F.relu(-input)
        elif self.act == "elu":
            return F.elu(input, alpha=1)
        elif self.act == "leaky_elu":
            return F.elu(input, alpha=1) - self.slope * F.relu(-input)
        elif self.act == "ssp":
            return F.softplus(input, beta=1) - self.shift
        elif self.act == "leaky_ssp":
            return F.softplus(input, beta=1) - self.slope * F.relu(-input) - self.shift
        elif self.act == "tanh":
            return torch.tanh(input)
        elif self.act == "leaky_tanh":
            return torch.tanh(input) + self.slope * input
        elif self.act == "swish":
            return torch.sigmoid(input) * input
        else:
            raise RuntimeError(f"Undefined activation called {self.act}")


# ------------------------------------------------------------------
# ocpmodels/models/utils/basis.py  (verbatim; SphericalSmearing.forward uses
# scipy.special.sph_harm which was removed in newer scipy, but it is never
# called for the "powersine" basis_type used by our recipe)
# ------------------------------------------------------------------
class Sine(nn.Module):
    def __init__(self, w0: float = 30.0) -> None:
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SINESmearing(nn.Module):
    def __init__(
        self,
        num_in_features: int,
        num_freqs: int = 40,
        use_cosine: bool = False,
    ) -> None:
        super(SINESmearing, self).__init__()

        self.num_freqs = num_freqs
        self.out_dim: int = num_in_features * self.num_freqs
        self.use_cosine = use_cosine

        freq = torch.arange(num_freqs).float()
        freq = torch.pow(torch.ones_like(freq) * 1.1, freq)
        self.freq_filter = nn.Parameter(
            freq.view(-1, 1).repeat(1, num_in_features).view(1, -1),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(1, self.num_freqs)
        x = x * self.freq_filter

        if self.use_cosine:
            return torch.cos(x)
        else:
            return torch.sin(x)


class GaussianSmearing(nn.Module):
    def __init__(
        self,
        num_in_features: int,
        start: int = 0,
        end: int = 1,
        num_freqs: int = 50,
    ) -> None:
        super(GaussianSmearing, self).__init__()
        self.num_freqs = num_freqs
        offset = torch.linspace(start, end, num_freqs)
        self.coeff: float = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.offset = nn.Parameter(
            offset.view(-1, 1).repeat(1, num_in_features).view(1, -1),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(1, self.num_freqs)
        x = x - self.offset
        return torch.exp(self.coeff * torch.pow(x, 2))


class FourierSmearing(nn.Module):
    def __init__(
        self,
        num_in_features: int,
        num_freqs: int = 40,
        use_cosine: bool = False,
    ) -> None:
        super(FourierSmearing, self).__init__()

        self.num_freqs = num_freqs
        self.out_dim: int = num_in_features * self.num_freqs
        self.use_cosine = use_cosine

        freq = torch.arange(num_freqs).to(torch.float32)
        self.freq_filter = nn.Parameter(
            freq.view(-1, 1).repeat(1, num_in_features).view(1, -1),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(1, self.num_freqs)
        x = x * self.freq_filter

        if self.use_cosine:
            return torch.cos(x)
        else:
            return torch.sin(x)


class Basis(nn.Module):
    smearing: Union[
        SINESmearing,
        SINESmearing,
        FourierSmearing,
        GaussianSmearing,
        torch.nn.Sequential,
    ]

    def __init__(
        self,
        num_in_features: int,
        num_freqs: int = 50,
        basis_type: str = "powersine",
        act: str = "ssp",
        sph: Optional["SphericalSmearing"] = None,
    ) -> None:
        super(Basis, self).__init__()

        self.num_freqs = num_freqs
        self.basis_type = basis_type

        if basis_type == "powersine":
            self.smearing = SINESmearing(num_in_features, num_freqs)
            self.out_dim = num_in_features * num_freqs
        elif basis_type == "powercosine":
            self.smearing = SINESmearing(num_in_features, num_freqs, use_cosine=True)
            self.out_dim = num_in_features * num_freqs
        elif basis_type == "fouriersine":
            self.smearing = FourierSmearing(num_in_features, num_freqs)
            self.out_dim = num_in_features * num_freqs
        elif basis_type == "gauss":
            self.smearing = GaussianSmearing(num_in_features, start=0, end=1, num_freqs=num_freqs)
            self.out_dim = num_in_features * num_freqs
        elif basis_type == "linact":
            self.smearing = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, num_freqs * num_in_features),
                Act(act),
            )
            self.out_dim = num_in_features * num_freqs
        elif basis_type == "raw" or basis_type == "rawcat":
            self.out_dim = num_in_features
        elif "sph" in basis_type:
            # by default, we use sine function to encode distance
            # sph must be given here
            assert sph is not None
            # assumes the first three columns are normalizaed xyz
            # the rest of the columns are distances
            if "cat" in basis_type:
                # concatenate
                self.smearing_sine = SINESmearing(num_in_features - 3, num_freqs)
                self.out_dim = sph.out_dim + (num_in_features - 3) * num_freqs
            elif "mul" in basis_type:
                self.smearing_sine = SINESmearing(num_in_features - 3, num_freqs)
                self.lin = torch.nn.Linear(self.smearing_sine.out_dim, num_in_features - 3)
                self.out_dim = (num_in_features - 3) * sph.out_dim
            elif "m40" in basis_type:
                dim = 40
                self.smearing_sine = SINESmearing(num_in_features - 3, num_freqs)
                self.lin = torch.nn.Linear(
                    self.smearing_sine.out_dim, dim
                )  # make the output dimensionality comparable.
                self.out_dim = dim * sph.out_dim
            elif "nosine" in basis_type:
                # does not use sine smearing for encoding distance
                self.out_dim = (num_in_features - 3) * sph.out_dim
            else:
                raise ValueError("cat or mul not specified for spherical harnomics.")
        else:
            raise RuntimeError("Undefined basis type.")

    def forward(self, x: torch.Tensor, edge_attr_sph: Optional[torch.Tensor] = None):
        if "sph" in self.basis_type:
            if "nosine" not in self.basis_type:
                x_sine = self.smearing_sine(
                    x[:, 3:]
                )  # the first three features correspond to edge_vec_normalized, so we ignore
                if "cat" in self.basis_type:
                    # just concatenate spherical edge feature and sined node features
                    assert isinstance(edge_attr_sph, torch.Tensor)
                    return torch.cat([edge_attr_sph, x_sine], dim=1)
                elif "mul" in self.basis_type or "m40" in self.basis_type:
                    # multiply sined node features into spherical edge feature (inspired by theory in spherical harmonics)
                    r = self.lin(x_sine)
                    outer = torch.einsum("ik,ij->ikj", edge_attr_sph, r)
                    return torch.flatten(outer, start_dim=1)
                else:
                    raise RuntimeError(f"Unknown basis type called {self.basis_type}")
            else:
                outer = torch.einsum("ik,ij->ikj", edge_attr_sph, x[:, 3:])
                return torch.flatten(outer, start_dim=1)

        elif "raw" in self.basis_type:
            # do nothing, just return node features
            pass
        else:
            x = self.smearing(x)
        return x


class SphericalSmearing(nn.Module):
    def __init__(self, max_n: int = 10, option: str = "all") -> None:
        super(SphericalSmearing, self).__init__()

        self.max_n = max_n

        m_list: List[int] = []
        n_list: List[int] = []
        for i in range(max_n):
            for j in range(0, i + 1):
                m_list.append(j)
                n_list.append(i)

        m = np.array(m_list)
        n = np.array(n_list)

        if option == "all":
            self.m = m
            self.n = n
        elif option == "sine":
            self.m = m[n % 2 == 1]
            self.n = n[n % 2 == 1]
        elif option == "cosine":
            self.m = m[n % 2 == 0]
            self.n = n[n % 2 == 0]

        self.out_dim = int(np.sum(self.m == 0) + 2 * np.sum(self.m != 0))

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # Not used on the "powersine" basis path exercised by this module;
        # scipy.special.sph_harm was removed upstream (scipy>=1.15).
        raise NotImplementedError(
            "SphericalSmearing.forward is not exercised on the powersine basis path."
        )


# ------------------------------------------------------------------
# ocpmodels/common/utils.py  (compute_neighbors only; verbatim)
# ------------------------------------------------------------------
def compute_neighbors(data, edge_index):
    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = edge_index[1].new_ones(1).expand_as(edge_index[1])
    num_neighbors = segment_coo(ones, edge_index[1], dim_size=data.natoms.sum())

    # Get number of neighbors per image
    image_indptr = torch.zeros(data.natoms.shape[0] + 1, device=data.pos.device, dtype=torch.long)
    image_indptr[1:] = torch.cumsum(data.natoms, dim=0)
    neighbors = segment_csr(num_neighbors, image_indptr)
    return neighbors


# ------------------------------------------------------------------
# ocpmodels/models/base.py  (BaseModel.generate_graph only, verbatim except the
# unreachable use_pbc=True branch -- which calls radius_graph_pbc/get_pbc_distances
# that live in ocpmodels.common.utils and are not vendored here -- is left as a
# straight pass-through to those names so the *code* is untouched; ForceNet is
# always constructed with use_pbc=False below so that branch is never taken)
# ------------------------------------------------------------------
class BaseModel(nn.Module):
    def __init__(self, num_atoms=None, bond_feat_dim=None, num_targets=None) -> None:
        super(BaseModel, self).__init__()
        self.num_atoms = num_atoms
        self.bond_feat_dim = bond_feat_dim
        self.num_targets = num_targets

    def forward(self, data):
        raise NotImplementedError

    def generate_graph(
        self,
        data,
        cutoff=None,
        max_neighbors=None,
        use_pbc=None,
        otf_graph=None,
        enforce_max_neighbors_strictly=None,
    ):
        cutoff = cutoff or self.cutoff
        max_neighbors = max_neighbors or self.max_neighbors
        use_pbc = use_pbc if use_pbc is not None else self.use_pbc
        otf_graph = otf_graph if otf_graph is not None else self.otf_graph

        if enforce_max_neighbors_strictly is not None:
            pass
        elif hasattr(self, "enforce_max_neighbors_strictly"):
            # Not all models will have this attribute
            enforce_max_neighbors_strictly = self.enforce_max_neighbors_strictly
        else:
            # Default to old behavior
            enforce_max_neighbors_strictly = True

        if not otf_graph:
            try:
                edge_index = data.edge_index

                if use_pbc:
                    cell_offsets = data.cell_offsets
                    neighbors = data.neighbors

            except AttributeError:
                logging.warning(
                    "Turning otf_graph=True as required attributes not present in data object"
                )
                otf_graph = True

        if use_pbc:
            from ocpmodels.common.utils import (  # noqa: E501  (real PBC path; not exercised)
                get_pbc_distances,
                radius_graph_pbc,
            )

            if otf_graph:
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    data,
                    cutoff,
                    max_neighbors,
                    enforce_max_neighbors_strictly,
                )

            out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=True,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_dist = out["distances"]
            cell_offset_distances = out["offsets"]
            distance_vec = out["distance_vec"]
        else:
            if otf_graph:
                edge_index = radius_graph(
                    data.pos,
                    r=cutoff,
                    batch=data.batch,
                    max_num_neighbors=max_neighbors,
                )

            j, i = edge_index
            distance_vec = data.pos[j] - data.pos[i]

            edge_dist = distance_vec.norm(dim=-1)
            cell_offsets = torch.zeros(edge_index.shape[1], 3, device=data.pos.device)
            cell_offset_distances = torch.zeros_like(cell_offsets, device=data.pos.device)
            neighbors = compute_neighbors(data, edge_index)

        return (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            cell_offset_distances,
            neighbors,
        )

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------
# ocpmodels/models/forcenet.py  (verbatim)
# ------------------------------------------------------------------
class FNDecoder(nn.Module):
    def __init__(self, decoder_type, decoder_activation_str, output_dim: int) -> None:
        super(FNDecoder, self).__init__()
        self.decoder_type = decoder_type
        self.decoder_activation = Act(decoder_activation_str)
        self.output_dim = output_dim

        self.decoder: nn.Sequential
        if self.decoder_type == "linear":
            self.decoder = nn.Sequential(nn.Linear(self.output_dim, 3))
        elif self.decoder_type == "mlp":
            self.decoder = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim),
                nn.BatchNorm1d(self.output_dim),
                self.decoder_activation,
                nn.Linear(self.output_dim, 3),
            )
        else:
            raise ValueError(f"Undefined force decoder: {self.decoder_type}")

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.decoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def forward(self, x):
        return self.decoder(x)


class InteractionBlock(MessagePassing):
    def __init__(
        self,
        hidden_channels: int,
        mlp_basis_dim: int,
        basis_type,
        depth_mlp_edge: int = 2,
        depth_mlp_trans: int = 1,
        activation_str: str = "ssp",
        ablation: str = "none",
    ) -> None:
        super(InteractionBlock, self).__init__(aggr="add")

        self.activation = Act(activation_str)
        self.ablation = ablation
        self.basis_type = basis_type

        # basis function assumes input is in the range of [-1,1]
        if self.basis_type != "rawcat":
            self.lin_basis = torch.nn.Linear(mlp_basis_dim, hidden_channels)

        if self.ablation == "nocond":
            # the edge filter only depends on edge_attr
            in_features = mlp_basis_dim if self.basis_type == "rawcat" else hidden_channels
        else:
            # edge filter depends on edge_attr and current node embedding
            in_features = (
                mlp_basis_dim + 2 * hidden_channels
                if self.basis_type == "rawcat"
                else 3 * hidden_channels
            )

        if depth_mlp_edge > 0:
            mlp_edge = [torch.nn.Linear(in_features, hidden_channels)]
            for _ in range(depth_mlp_edge):
                mlp_edge.append(self.activation)
                mlp_edge.append(torch.nn.Linear(hidden_channels, hidden_channels))
        else:
            ## need batch normalization afterwards. Otherwise training is unstable.
            mlp_edge = [
                torch.nn.Linear(in_features, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
            ]
        self.mlp_edge = torch.nn.Sequential(*mlp_edge)

        if not self.ablation == "nofilter":
            self.lin = torch.nn.Linear(hidden_channels, hidden_channels)

        if depth_mlp_trans > 0:
            mlp_trans = [torch.nn.Linear(hidden_channels, hidden_channels)]
            for _ in range(depth_mlp_trans):
                mlp_trans.append(torch.nn.BatchNorm1d(hidden_channels))
                mlp_trans.append(self.activation)
                mlp_trans.append(torch.nn.Linear(hidden_channels, hidden_channels))
        else:
            # need batch normalization afterwards. Otherwise, becomes NaN
            mlp_trans = [
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
            ]

        self.mlp_trans = torch.nn.Sequential(*mlp_trans)

        if not self.ablation == "noself":
            self.center_W = torch.nn.Parameter(torch.Tensor(1, hidden_channels))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.basis_type != "rawcat":
            torch.nn.init.xavier_uniform_(self.lin_basis.weight)
            self.lin_basis.bias.data.fill_(0)

        for m in self.mlp_trans:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        for m in self.mlp_edge:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        if not self.ablation == "nofilter":
            torch.nn.init.xavier_uniform_(self.lin.weight)
            self.lin.bias.data.fill_(0)

        if not self.ablation == "noself":
            torch.nn.init.xavier_uniform_(self.center_W)

    def forward(self, x, edge_index, edge_attr, edge_weight):
        if self.basis_type != "rawcat":
            edge_emb = self.lin_basis(edge_attr)
        else:
            # for rawcat, we directly use the raw feature
            edge_emb = edge_attr

        if self.ablation == "nocond":
            emb = edge_emb
        else:
            emb = torch.cat([edge_emb, x[edge_index[0]], x[edge_index[1]]], dim=1)

        W = self.mlp_edge(emb) * edge_weight.view(-1, 1)
        if self.ablation == "nofilter":
            x = self.propagate(edge_index, x=x, W=W) + self.center_W
        else:
            x = self.lin(x)
            if self.ablation == "noself":
                x = self.propagate(edge_index, x=x, W=W)
            else:
                x = self.propagate(edge_index, x=x, W=W) + self.center_W * x
        x = self.mlp_trans(x)

        return x

    def message(self, x_j, W):
        if self.ablation == "nofilter":
            return W
        else:
            return x_j * W


class ForceNet(BaseModel):
    r"""Implementation of ForceNet architecture.

    Args:
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets (int): Unused argumebt
        hidden_channels (int, optional): Number of hidden channels.
            (default: :obj:`512`)
        num_iteractions (int, optional): Number of interaction blocks.
            (default: :obj:`5`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`6.0`)
        feat (str, optional): Input features to be used
            (default: :obj:`full`)
        num_freqs (int, optional): Number of frequencies for basis function.
            (default: :obj:`50`)
        max_n (int, optional): Maximum order of spherical harmonics.
            (default: :obj:`6`)
        basis (str, optional): Basis function to be used.
            (default: :obj:`full`)
        depth_mlp_edge (int, optional): Depth of MLP for edges in interaction blocks.
            (default: :obj:`2`)
        depth_mlp_node (int, optional): Depth of MLP for nodes in interaction blocks.
            (default: :obj:`1`)
        activation_str (str, optional): Activation function used post linear layer in all message passing MLPs.
            (default: :obj:`swish`)
        ablation (str, optional): Type of ablation to be performed.
            (default: :obj:`none`)
        decoder_hidden_channels (int, optional): Number of hidden channels in the decoder.
            (default: :obj:`512`)
        decoder_type (str, optional): Type of decoder: linear or MLP.
            (default: :obj:`mlp`)
        decoder_activation_str (str, optional): Activation function used post linear layer in decoder.
            (default: :obj:`swish`)
        training (bool, optional): If set to :obj:`True`, specify training phase.
            (default: :obj:`True`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
    """

    def __init__(
        self,
        num_atoms: int,  # not used
        bond_feat_dim: int,  # not used
        num_targets: int,  # not used
        hidden_channels: int = 512,
        num_interactions: int = 5,
        cutoff: float = 6.0,
        feat: str = "full",
        num_freqs: int = 50,
        max_n: int = 3,
        basis: str = "sphallmul",
        depth_mlp_edge: int = 2,
        depth_mlp_node: int = 1,
        activation_str: str = "swish",
        ablation: str = "none",
        decoder_hidden_channels: int = 512,
        decoder_type: str = "mlp",
        decoder_activation_str: str = "swish",
        training: bool = True,
        otf_graph: bool = False,
        use_pbc: bool = True,
    ) -> None:
        super(ForceNet, self).__init__()
        self.training = training
        self.ablation = ablation
        if self.ablation not in [
            "none",
            "nofilter",
            "nocond",
            "nodistlist",
            "onlydist",
            "nodelinear",
            "edgelinear",
            "noself",
        ]:
            raise ValueError(f"Unknown ablation called {ablation}.")

        """
        Descriptions of ablations:
            - none: base ForceNet model
            - nofilter: no element-wise filter parameterization in message modeling
            - nocond: convolutional filter is only conditioned on edge features, not node embeddings
            - nodistlist: no atomic radius information in edge features
            - onlydist: edge features only contains distance information. Orientation information is ommited.
            - nodelinear: node update MLP function is replaced with linear function followed by batch normalization
            - edgelinear: edge MLP transformation function is replaced with linear function followed by batch normalization.
            - noself: no self edge of m_t.
        """

        self.otf_graph = otf_graph
        self.cutoff = cutoff
        self.output_dim = decoder_hidden_channels
        self.feat = feat
        self.num_freqs = num_freqs
        self.num_layers = num_interactions
        self.max_n = max_n
        self.activation_str = activation_str
        self.use_pbc = use_pbc
        self.max_neighbors = 50

        if self.ablation == "edgelinear":
            depth_mlp_edge = 0

        if self.ablation == "nodelinear":
            depth_mlp_node = 0

        # read atom map and atom radii
        atom_map = torch.zeros(101, 9)
        for i in range(101):
            atom_map[i] = torch.tensor(CONTINUOUS_EMBEDDINGS[i])

        atom_radii = torch.zeros(101)
        for i in range(101):
            atom_radii[i] = ATOMIC_RADII[i]
        atom_radii = atom_radii / 100

        self.atom_radii = nn.Parameter(atom_radii, requires_grad=False)
        self.basis_type = basis

        self.pbc_apply_sph_harm = "sph" in self.basis_type
        self.pbc_sph_option = None

        # for spherical harmonics for PBC
        if "sphall" in self.basis_type:
            self.pbc_sph_option = "all"
        elif "sphsine" in self.basis_type:
            self.pbc_sph_option = "sine"
        elif "sphcosine" in self.basis_type:
            self.pbc_sph_option = "cosine"

        self.pbc_sph: Optional[SphericalSmearing] = None
        if self.pbc_apply_sph_harm:
            self.pbc_sph = SphericalSmearing(max_n=self.max_n, option=self.pbc_sph_option)

        # self.feat can be "simple" or "full"
        if self.feat == "simple":
            self.embedding = nn.Embedding(100, hidden_channels)

            # set up dummy atom_map that only contains atomic_number information
            atom_map = torch.linspace(0, 1, 101).view(-1, 1).repeat(1, 9)
            self.atom_map = nn.Parameter(atom_map, requires_grad=False)

        elif self.feat == "full":
            # Normalize along each dimaension
            atom_map[0] = np.nan
            atom_map_notnan = atom_map[atom_map[:, 0] == atom_map[:, 0]]
            atom_map_min = torch.min(atom_map_notnan, dim=0)[0]
            atom_map_max = torch.max(atom_map_notnan, dim=0)[0]
            atom_map_gap = atom_map_max - atom_map_min

            ## squash to [0,1]
            atom_map = (atom_map - atom_map_min.view(1, -1)) / atom_map_gap.view(1, -1)

            self.atom_map = torch.nn.Parameter(atom_map, requires_grad=False)

            in_features = 9
            # first apply basis function and then linear function
            if "sph" in self.basis_type:
                # spherical basis is only meaningful for edge feature, so use powersine instead
                node_basis_type = "powersine"
            else:
                node_basis_type = self.basis_type
            basis = Basis(
                in_features,
                num_freqs=num_freqs,
                basis_type=node_basis_type,
                act=self.activation_str,
            )
            self.embedding = torch.nn.Sequential(
                basis, torch.nn.Linear(basis.out_dim, hidden_channels)
            )

        else:
            raise ValueError("Undefined feature type for atom")

        # process basis function for edge feature
        if self.ablation == "nodistlist":
            # do not consider additional distance edge features
            # normalized (x,y,z) + distance
            in_feature = 4
        elif self.ablation == "onlydist":
            # only consider distance-based edge features
            # ignore normalized (x,y,z)
            in_feature = 4

            # if basis_type is spherical harmonics, then reduce to powersine
            if "sph" in self.basis_type:
                logging.info(
                    "Under onlydist ablation, spherical basis is reduced to powersine basis."
                )
                self.basis_type = "powersine"
                self.pbc_sph = None

        else:
            in_feature = 7
        self.basis_fun = Basis(
            in_feature,
            num_freqs,
            self.basis_type,
            self.activation_str,
            sph=self.pbc_sph,
        )

        # process interaction blocks
        self.interactions = torch.nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(
                hidden_channels,
                self.basis_fun.out_dim,
                self.basis_type,
                depth_mlp_edge=depth_mlp_edge,
                depth_mlp_trans=depth_mlp_node,
                activation_str=self.activation_str,
                ablation=ablation,
            )
            self.interactions.append(block)

        self.lin = torch.nn.Linear(hidden_channels, self.output_dim)
        self.activation = Act(activation_str)

        # ForceNet decoder
        self.decoder = FNDecoder(decoder_type, decoder_activation_str, self.output_dim)

        # Projection layer for energy prediction
        self.energy_mlp = nn.Linear(self.output_dim, 1)

    def forward(self, data):
        z = data.atomic_numbers.long()

        pos = data.pos  # noqa: F841  (unused in the original repo code; kept verbatim)
        batch = data.batch

        if self.feat == "simple":
            h = self.embedding(z)
        elif self.feat == "full":
            h = self.embedding(self.atom_map[z])
        else:
            raise RuntimeError("Undefined feature type for atom")

        (
            edge_index,
            edge_dist,
            edge_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        data.edge_index = edge_index
        data.cell_offsets = cell_offsets
        data.neighbors = neighbors

        if self.pbc_apply_sph_harm:
            edge_vec_normalized = edge_vec / edge_dist.view(-1, 1)
            edge_attr_sph = self.pbc_sph(edge_vec_normalized)

        # calculate the edge weight according to the dist
        edge_weight = torch.cos(0.5 * edge_dist * PI / self.cutoff)

        # normalized edge vectors
        edge_vec_normalized = edge_vec / edge_dist.view(-1, 1)

        # edge distance, taking the atom_radii into account
        # each element lies in [0,1]
        edge_dist_list = (
            torch.stack(
                [
                    edge_dist,
                    edge_dist - self.atom_radii[z[edge_index[0]]],
                    edge_dist - self.atom_radii[z[edge_index[1]]],
                    edge_dist
                    - self.atom_radii[z[edge_index[0]]]
                    - self.atom_radii[z[edge_index[1]]],
                ]
            ).transpose(0, 1)
            / self.cutoff
        )

        if self.ablation == "nodistlist":
            edge_dist_list = edge_dist_list[:, 0].view(-1, 1)

        # make sure distance is positive
        edge_dist_list[edge_dist_list < 1e-3] = 1e-3

        # squash to [0,1] for gaussian basis
        if self.basis_type == "gauss":
            edge_vec_normalized = (edge_vec_normalized + 1) / 2.0

        # process raw_edge_attributes to generate edge_attributes
        if self.ablation == "onlydist":
            raw_edge_attr = edge_dist_list
        else:
            raw_edge_attr = torch.cat([edge_vec_normalized, edge_dist_list], dim=1)

        if "sph" in self.basis_type:
            edge_attr = self.basis_fun(raw_edge_attr, edge_attr_sph)
        else:
            edge_attr = self.basis_fun(raw_edge_attr)

        # pass edge_attributes through interaction blocks
        for _, interaction in enumerate(self.interactions):
            h = h + interaction(h, edge_index, edge_attr, edge_weight)

        h = self.lin(h)
        h = self.activation(h)

        out = scatter(h, batch, dim=0, reduce="add")

        force = self.decoder(h)
        energy = self.energy_mlp(out)
        return energy, force

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------
# Menagerie staging entrypoints
# ------------------------------------------------------------------
class _TinyMolData:
    """Minimal stand-in for a torch_geometric Data object carrying only the
    fields ForceNet.forward / BaseModel.generate_graph / compute_neighbors read:
    atomic_numbers, pos, batch, natoms.
    """

    def __init__(self, atomic_numbers, pos, batch, natoms):
        self.atomic_numbers = atomic_numbers
        self.pos = pos
        self.batch = batch
        self.natoms = natoms


def build_forcenet():
    torch.manual_seed(0)
    return ForceNet(
        num_atoms=None,
        bond_feat_dim=None,
        num_targets=1,
        hidden_channels=32,
        num_interactions=2,
        cutoff=6.0,
        feat="full",
        num_freqs=8,
        basis="powersine",
        depth_mlp_edge=1,
        depth_mlp_node=1,
        activation_str="swish",
        decoder_hidden_channels=32,
        decoder_type="mlp",
        decoder_activation_str="swish",
        otf_graph=True,
        use_pbc=False,
    )


def example_input_forcenet():
    torch.manual_seed(0)
    # A tiny 6-atom synthetic "molecule" (single graph in the batch), positions
    # spread out so radius_graph with cutoff=6.0 finds a handful of edges.
    atomic_numbers = torch.tensor([6, 1, 1, 1, 1, 8], dtype=torch.long)
    pos = torch.randn(6, 3) * 2.0
    batch = torch.zeros(6, dtype=torch.long)
    natoms = torch.tensor([6], dtype=torch.long)
    return (_TinyMolData(atomic_numbers, pos, batch, natoms),)


MENAGERIE_ENTRIES = [
    ("forcenet", build_forcenet, example_input_forcenet, 2021, MENAGERIE_ZOO),
]
