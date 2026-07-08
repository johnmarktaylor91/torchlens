# SOURCE: vendored from Imfinethankyou1/GeoTMI @ a2f2e5bed5ecfdbb78de7530a52a54e1051ea3c2 (main)
#
# GeoTMI (Geometry via Transferring Molecular Information): a DimeNet++-based
# interatomic-property predictor extended with an iterative *position-denoising*
# head -- each interaction block additionally predicts a per-edge coefficient
# (`rbf_att1`/`rbf_att2` -> `pos_update_coeff`) used to update atom positions
# between message-passing steps, so the model can predict properties from a
# cheap/approximate geometry while implicitly refining it towards the expensive
# reference geometry (Kim, Ryu et al., ICLR 2023 workshop, "Geometry via
# Transferring Molecular Information via Positional Denoising").
#
# Files combined (each class copied verbatim from the real repo, imports fixed
# minimally for the installed torch_geometric/torch_sparse versions -- the real
# `.acts.swish` module was removed upstream in newer torch_geometric and is
# replaced with the mathematically-identical `torch.nn.functional.silu`
# (swish(x) = x*sigmoid(x) = SiLU(x)); `EmbeddingBlock`/`SphericalBasisLayer`/
# `ResidualLayer` are reused directly from torch_geometric's own dimenet module,
# exactly as the real repo does):
#   - QM9M/dimenet++/models/envelope.py             -> Envelope
#   - QM9M/dimenet++/models/bessel_basis_layer.py    -> BesselBasisLayer
#   - QM9M/dimenet++/models/interaction_block_pp.py  -> InteractionPPBlock (the
#                                                        position-denoising head)
#   - QM9M/dimenet++/models/output_block_pp.py       -> OutputPPBlock
#   - QM9M/dimenet++/models/dimenet_pp.py            -> DimeNetPlusPlus (GeoTMI's
#                                                        forward(), including the
#                                                        update_pos loop)
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

from math import pi as PI
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, ModuleList, Parameter
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.models.dimenet import EmbeddingBlock, ResidualLayer, SphericalBasisLayer
from torch_scatter import scatter
from torch_sparse import SparseTensor

MENAGERIE_ZOO = "vendored-pytorch"

# swish == SiLU: x * sigmoid(x). Upstream imported this from the now-removed
# torch_geometric.nn.acts module; torch.nn.functional.silu is the identical
# function under a different name/location.
swish = F.silu


# ------------------------------------------------------------------
# QM9M/dimenet++/models/envelope.py  (verbatim)
# ------------------------------------------------------------------
class Envelope(Module):
    def __init__(self, exponent: int):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x: Tensor) -> Tensor:
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        grad_checker = (x < 1.0).to(x.dtype)
        return (1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2) * grad_checker


# ------------------------------------------------------------------
# QM9M/dimenet++/models/bessel_basis_layer.py  (verbatim)
# ------------------------------------------------------------------
class BesselBasisLayer(Module):
    def __init__(self, num_radial: int, cutoff: float = 5.0, envelope_exponent: int = 5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = Parameter(Tensor(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


# ------------------------------------------------------------------
# QM9M/dimenet++/models/interaction_block_pp.py  (verbatim; this is GeoTMI's
# architectural addition over stock DimeNet++ -- the rbf_att1/rbf_att2 head
# producing pos_update_coeff)
# ------------------------------------------------------------------
class Swish(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor):
        return x * torch.sigmoid(x)


class InteractionPPBlock(Module):
    def __init__(
        self,
        hidden_channels: int,
        int_emb_size: int,
        basis_emb_size: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        act: Callable,
    ):
        super().__init__()
        self.act = act

        # Transformation of Bessel and spherical basis representations:
        self.lin_rbf1 = Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = Linear(basis_emb_size, hidden_channels, bias=False)

        self.lin_sbf1 = Linear(num_spherical * num_radial, basis_emb_size, bias=False)
        self.lin_sbf2 = Linear(basis_emb_size, int_emb_size, bias=False)

        # Hidden transformation of input message:
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets:
        self.lin_down = Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = Linear(int_emb_size, hidden_channels, bias=False)

        # Residual layers before and after skip connection:
        self.layers_before_skip = ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)]
        )
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)]
        )

        # position-denoising head: predicts a per-edge coefficient used to
        # update atom positions between message-passing steps (GeoTMI's
        # contribution over stock DimeNet++)
        self.rbf_att1 = Linear(hidden_channels, hidden_channels, bias=True)
        self.rbf_att2 = Linear(hidden_channels, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.rbf_att1.weight, scale=2.0)
        glorot_orthogonal(self.rbf_att2.weight, scale=2.0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()

    def forward(
        self,
        x: Tensor,
        rbf: Tensor,
        sbf: Tensor,
        idx_kj: Tensor,
        idx_ji: Tensor,
        update_pos: bool = False,
    ) -> Tensor:
        # Initial transformation:
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        # Transformation via Bessel basis:
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down project embedding and generating triple-interactions:
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis:
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings:
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        pos_update_coeff = None
        if update_pos:
            pos_update_coeff = self.act(self.rbf_att1(h))
            pos_update_coeff = torch.tanh(self.rbf_att2(pos_update_coeff))

        return h, pos_update_coeff


# ------------------------------------------------------------------
# QM9M/dimenet++/models/output_block_pp.py  (verbatim)
# ------------------------------------------------------------------
class OutputPPBlock(Module):
    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_emb_channels: int,
        out_channels: int,
        num_layers: int,
        act: Callable,
    ):
        super().__init__()
        self.act = act

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)

        # The up-projection layer:
        self.lin_up = Linear(hidden_channels, out_emb_channels, bias=False)
        self.lins = ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(out_emb_channels, out_emb_channels))
        self.lin = Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        self.lin.weight.data.fill_(0)

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor, num_nodes: Optional[int] = None) -> Tensor:
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


# ------------------------------------------------------------------
# QM9M/dimenet++/models/dimenet_pp.py  (verbatim, DimeNetPlusPlus class; this
# IS GeoTMI's model class -- the repo names it DimeNetPlusPlus because it is
# architecturally DimeNet++ + the position-denoising head above)
# ------------------------------------------------------------------
class DimeNetPlusPlus(Module):
    r"""DimeNet++ implementation based on https://github.com/klicperajo/dimenet.
    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Embedding size used for interaction triplets
        basis_emb_size (int): Embedding size used in the basis transformation
        out_emb_channels(int): Embedding size used for atoms in the output block
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (function, optional): The activation funtion.
            (default: :obj:`swish`)
    """

    def __init__(
        self,
        hidden_channels=128,
        out_channels=1,
        num_blocks=4,
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        cutoff=5.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        act=swish,
    ):
        super(DimeNetPlusPlus, self).__init__()

        self.cutoff = cutoff

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff, envelope_exponent)
        self.act = act
        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)

        self.output_blocks = ModuleList(
            [
                OutputPPBlock(
                    num_radial,
                    hidden_channels,
                    out_emb_channels,
                    out_channels,
                    num_output_layers,
                    act,
                )
                for _ in range(num_blocks + 1)
            ]
        )

        self.interaction_blocks = ModuleList(
            [
                InteractionPPBlock(
                    hidden_channels,
                    int_emb_size,
                    basis_emb_size,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act,
                )
                for _ in range(num_blocks)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    def triplets(
        self,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def get_angle(self, pos: Tensor, idx_i: Tensor, idx_j: Tensor, idx_k: Tensor) -> Tensor:
        pos_i = pos[idx_i].detach()
        pos_j = pos[idx_j].detach()
        pos_ji, pos_kj = (
            pos[idx_j].detach() - pos_i,
            pos[idx_k].detach() - pos_j,
        )

        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1) + 1e-12
        angle = torch.atan2(b, a)
        return angle

    def forward(
        self,
        z,
        pos,
        edge_index,
        batch=None,
        update_pos: bool = False,
    ):
        j, i = edge_index

        _, _, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(edge_index, num_nodes=z.size(0))

        # calculate distance
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # calculate angles
        angle = self.get_angle(pos, idx_i, idx_j, idx_k)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # embedding block
        x = self.emb(z.long(), rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # interaction blocks
        dist_history = []
        for (
            interaction_block,
            output_block,
        ) in zip(
            self.interaction_blocks,
            self.output_blocks[1:],
        ):
            x, pos_update_coeff = interaction_block(x, rbf, sbf, idx_kj, idx_ji, update_pos)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))

            if update_pos:
                pos_delta = scatter((pos[i] - pos[j]) * pos_update_coeff, i, dim=0, reduce="mean")
                pos = pos + pos_delta

                dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
                angle = self.get_angle(pos, idx_i, idx_j, idx_k)

                rbf = self.rbf(dist)
                sbf = self.sbf(dist, angle, idx_kj)

                dist_history.append(dist)

        energy = P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)
        return energy, dist_history


# ------------------------------------------------------------------
# Menagerie staging entrypoints
# ------------------------------------------------------------------
def build_geotmi():
    torch.manual_seed(0)
    return DimeNetPlusPlus(
        hidden_channels=16,
        out_channels=1,
        num_blocks=2,
        int_emb_size=8,
        basis_emb_size=4,
        out_emb_channels=16,
        num_spherical=3,
        num_radial=4,
        cutoff=6.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=1,
        num_output_layers=1,
        act=swish,
    )


def example_input_geotmi():
    torch.manual_seed(0)
    # A tiny 6-atom synthetic molecule with a fully-connected edge_index (so the
    # triplets() helper has real k->j->i chains to build), update_pos=True to
    # exercise GeoTMI's position-denoising loop.
    z = torch.tensor([6, 1, 1, 1, 1, 8], dtype=torch.long)
    pos = torch.randn(6, 3) * 2.0
    n = z.size(0)
    src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    mask = src != dst
    edge_index = torch.stack([src[mask], dst[mask]], dim=0)
    batch = torch.zeros(n, dtype=torch.long)
    return (z, pos, edge_index, batch, True)


MENAGERIE_ENTRIES = [
    ("geotmi", build_geotmi, example_input_geotmi, 2023, MENAGERIE_ZOO),
]
