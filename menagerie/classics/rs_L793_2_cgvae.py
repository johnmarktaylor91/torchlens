# SOURCE: vendored from https://github.com/wwang2/CoarseGrainingVAE @ main
# Files: CoarseGrainingVAE/cgvae.py (CGequiVAE, EquiEncoder, EquivariantDecoder),
#        CoarseGrainingVAE/conv.py (make_directed, to_module, preprocess_r,
#        InvariantMessage, EquiMessageBlock, UpdateBlock, ContractiveMessageBlock),
#        CoarseGrainingVAE/modules.py (Dense, DistanceEmbed, PainnRadialBasis,
#        CosineEnvelope, layer_types, gaussian_smearing helpers).
#
# CGVAE (Wang, Charron, Husic, Olsson, Noe, Clementi et al., ICML 2022,
# "Coarse-Grained Geometric Networks for Molecular Super-Resolution") is an
# equivariant variational autoencoder for geometric super-resolution: mapping
# coarse-grained (CG) molecular coordinates back to fine-grained (all-atom)
# coordinates. An EquiEncoder performs PaiNN-style equivariant (scalar s /
# vector v) message passing over the all-atom graph, contracts atom messages
# onto CG beads (ContractiveMessageBlock), producing a per-CG-bead latent
# code S_I; a reparametrized latent z_sample is then decoded by an
# EquivariantDecoder (equivariant message passing over the CG graph) whose
# per-atom vector channel is read out as the reconstructed atomic
# displacement from each atom's parent CG bead (`CGequiVAE.decoder`).
#
# No architecture was altered. This file vendors the `cross_flag=True`
# EquivariantDecoder path (`EquiMessageCross`) actually used in the CGVAE
# training scripts (see `scripts/run.sh` / `CoarseGrainingVAE/run_pdb.py`,
# which construct `EquivariantDecoder(..., cross_flag=True)`), the
# `CGequiVAE` top-level VAE forward, and the `EquiEncoder`. `PCN` (protein
# completion network, a variant head defined in the same cgvae.py that skips
# the VAE encoder entirely), `CGprior` (an optional learned Gaussian prior
# over the CG latents, wired in only when `--prior_type` is set), the
# `EquivariantPsuedoDecoder`/`ENDecoder`/`DenseEquiMessageBlock` alternate
# decoder heads, and the dataset/training/plotting utilities are dropped --
# they are optional/alternate configurations, not part of the default CGVAE
# architecture exercised here (`prior_net=None` is the CGequiVAE default).
# The `.cpu()` calls the original authors sprinkled onto some neighbor-index
# gathers (an artifact of their own mixed CPU/GPU indexing, not part of the
# architecture) are preserved verbatim.

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_
from torch_scatter import scatter_add, scatter_mean

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# CoarseGrainingVAE/modules.py
# ---------------------------------------------------------------------------


class shifted_softplus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return F.softplus(input) - np.log(2.0)


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


def zeros_initializer(tensor):
    return constant_(tensor, val=0.0)


layer_types = {
    "linear": nn.Linear,
    "Tanh": nn.Tanh,
    "ReLU": nn.ReLU,
    "shifted_softplus": shifted_softplus,
    "sigmoid": nn.Sigmoid,
    "Dropout": nn.Dropout,
    "LeakyReLU": nn.LeakyReLU,
    "ELU": nn.ELU,
    "swish": Swish,
}


class CosineEnvelope(nn.Module):
    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, d):
        output = 0.5 * (torch.cos((np.pi * d / self.cutoff)) + 1)
        exclude = d >= self.cutoff
        output = torch.where(exclude, torch.zeros_like(output), output)
        return output


class Dense(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        dropout_rate=0.0,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super().__init__(in_features, out_features, bias)

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        y = super().forward(inputs)
        if hasattr(self, "dropout"):
            y = self.dropout(y)
        if self.activation:
            y = self.activation(y)
        return y


class PainnRadialBasis(nn.Module):
    def __init__(self, n_rbf, cutoff):
        super().__init__()
        self.n = torch.arange(1, n_rbf + 1).float()
        self.cutoff = cutoff

    def forward(self, dist):
        shape_d = dist.unsqueeze(-1)
        n = self.n.to(dist.device)
        coef = n * np.pi / self.cutoff
        device = shape_d.device

        denom = torch.where(shape_d == 0, torch.tensor(1.0, device=device), shape_d)
        num = torch.where(shape_d == 0, coef, torch.sin(coef * shape_d))

        output = torch.where(shape_d >= self.cutoff, torch.tensor(0.0, device=device), num / denom)
        return output


class DistanceEmbed(nn.Module):
    def __init__(self, n_rbf, cutoff, feat_dim, dropout):
        super().__init__()
        rbf = PainnRadialBasis(n_rbf=n_rbf, cutoff=cutoff)
        dense = Dense(in_features=n_rbf, out_features=feat_dim, bias=True, dropout_rate=dropout)
        self.block = nn.Sequential(rbf, dense)
        self.f_cut = CosineEnvelope(cutoff=cutoff)

    def forward(self, dist):
        rbf_feats = self.block(dist)
        envelope = self.f_cut(dist).reshape(-1, 1)
        output = rbf_feats * envelope
        return output


# ---------------------------------------------------------------------------
# CoarseGrainingVAE/conv.py
# ---------------------------------------------------------------------------


def make_directed(nbr_list):
    gtr_ij = (nbr_list[:, 0] > nbr_list[:, 1]).any().item()
    gtr_ji = (nbr_list[:, 1] > nbr_list[:, 0]).any().item()
    directed = gtr_ij and gtr_ji

    if directed:
        return nbr_list, directed

    new_nbrs = torch.cat([nbr_list, nbr_list.flip(1)], dim=0)
    return new_nbrs, directed


def to_module(activation):
    return layer_types[activation]()


def preprocess_r(r_ij):
    dist = ((r_ij**2 + 1e-8).sum(-1)) ** 0.5
    unit = r_ij / dist.reshape(-1, 1)
    return dist, unit


class InvariantMessage(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim, activation, n_rbf, cutoff, dropout):
        super().__init__()

        self.inv_dense = nn.Sequential(
            Dense(
                in_features=in_feat_dim,
                out_features=in_feat_dim,
                bias=True,
                dropout_rate=dropout,
                activation=to_module(activation),
            ),
            Dense(
                in_features=in_feat_dim, out_features=out_feat_dim, bias=True, dropout_rate=dropout
            ),
        )

        self.dist_embed = DistanceEmbed(
            n_rbf=n_rbf, cutoff=cutoff, feat_dim=out_feat_dim, dropout=dropout
        )

        self.dist_filter = Dense(
            in_features=in_feat_dim, out_features=out_feat_dim, bias=True, dropout_rate=0.0
        )

        self.offset = torch.linspace(0.0, cutoff, in_feat_dim)

    def forward(self, s_j, dist, nbrs):
        phi = self.inv_dense(s_j)[nbrs[:, 1].cpu()]
        w_s = self.dist_embed(dist)
        output = phi * w_s
        return output


class EquiMessageBlock(nn.Module):
    def __init__(self, feat_dim, activation, n_rbf, cutoff, dropout):
        super().__init__()
        self.inv_message = InvariantMessage(
            in_feat_dim=feat_dim,
            out_feat_dim=feat_dim * 3,
            activation=activation,
            n_rbf=n_rbf,
            cutoff=cutoff,
            dropout=dropout,
        )

        self.h_att = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, feat_dim)
        )
        self.v_att = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, s_j, v_j, r_ij, nbrs, edge_wgt=None):
        dist, unit = preprocess_r(r_ij)
        inv_out = self.inv_message(s_j=s_j, dist=dist, nbrs=nbrs)
        graph_size = s_j.shape[0]

        inv_out = inv_out.reshape(inv_out.shape[0], 3, s_j.shape[-1])

        split_0 = inv_out[:, 0, :].unsqueeze(-1)
        split_1 = inv_out[:, 1, :]
        split_2 = inv_out[:, 2, :].unsqueeze(-1)

        unit_add = split_2 * unit.unsqueeze(1)
        delta_v_ij = unit_add + split_0 * v_j[nbrs[:, 1].cpu()]
        delta_s_ij = split_1

        if edge_wgt is not None:
            v_edge_wgt = edge_wgt[..., None, None]
            h_edge_wgt = edge_wgt[..., None]
        else:
            v_edge_wgt = 1
            h_edge_wgt = 1

        delta_v_i = scatter_add(
            src=delta_v_ij * v_edge_wgt, index=nbrs[:, 0], dim=0, dim_size=graph_size
        )
        delta_s_i = scatter_add(
            src=delta_s_ij * h_edge_wgt, index=nbrs[:, 0], dim=0, dim_size=graph_size
        )

        return delta_s_i, delta_v_i


class EquiMessageCross(nn.Module):
    def __init__(self, feat_dim, activation, n_rbf, cutoff, dropout):
        super().__init__()
        self.inv_message = InvariantMessage(
            in_feat_dim=feat_dim,
            out_feat_dim=feat_dim * 4,
            activation=activation,
            n_rbf=n_rbf,
            cutoff=cutoff,
            dropout=dropout,
        )

    def forward(self, s_j, v_j, r_ij, nbrs, edge_wgt=None):
        dist, unit = preprocess_r(r_ij)
        inv_out = self.inv_message(s_j=s_j, dist=dist, nbrs=nbrs)

        inv_out = inv_out.reshape(inv_out.shape[0], 4, s_j.shape[1])

        split_0 = inv_out[:, 0, :].unsqueeze(-1)
        split_1 = inv_out[:, 1, :]
        split_2 = inv_out[:, 2, :].unsqueeze(-1)
        split_3 = inv_out[:, 3, :].unsqueeze(-1)

        unit_add = split_2 * unit.unsqueeze(1)
        delta_v_ij = (
            unit_add
            + split_0 * v_j[nbrs[:, 1].cpu()]
            + split_3 * torch.cross(v_j[nbrs[:, 0]], v_j[nbrs[:, 1].cpu()])
        )
        delta_s_ij = split_1

        if edge_wgt is not None:
            v_edge_wgt = edge_wgt[..., None, None]
            h_edge_wgt = edge_wgt[..., None]
        else:
            v_edge_wgt = 1
            h_edge_wgt = 1

        graph_size = s_j.shape[0]
        dv = scatter_add(src=delta_v_ij * v_edge_wgt, index=nbrs[:, 0], dim=0, dim_size=graph_size)
        dh = scatter_add(src=delta_s_ij * h_edge_wgt, index=nbrs[:, 0], dim=0, dim_size=graph_size)

        return dh, dv


class UpdateBlock(nn.Module):
    def __init__(self, feat_dim, activation, dropout):
        super().__init__()
        self.u_mat = Dense(in_features=feat_dim, out_features=feat_dim, bias=False)
        self.v_mat = Dense(in_features=feat_dim, out_features=feat_dim, bias=False)
        self.s_dense = nn.Sequential(
            Dense(
                in_features=2 * feat_dim,
                out_features=feat_dim,
                bias=True,
                dropout_rate=dropout,
                activation=to_module(activation),
            ),
            Dense(in_features=feat_dim, out_features=3 * feat_dim, bias=True, dropout_rate=dropout),
        )

    def forward(self, s_i, v_i):
        v_tranpose = v_i.transpose(1, 2).reshape(-1, v_i.shape[1])

        num_feats = v_i.shape[1]
        u_v = self.u_mat(v_tranpose).reshape(-1, 3, num_feats).transpose(1, 2)
        v_v = self.v_mat(v_tranpose).reshape(-1, 3, num_feats).transpose(1, 2)

        v_v_norm = ((v_v**2 + 1e-10).sum(-1)) ** 0.5
        s_stack = torch.cat([s_i, v_v_norm], dim=-1)

        split = self.s_dense(s_stack).reshape(s_i.shape[0], 3, -1)

        a_vv = split[:, 0, :].unsqueeze(-1)
        delta_v_i = u_v * a_vv

        a_sv = split[:, 1, :]
        a_ss = split[:, 2, :]

        inner = (u_v * v_v).sum(-1)
        delta_s_i = inner * a_sv + a_ss

        return delta_s_i, delta_v_i


class ContractiveMessageBlock(nn.Module):
    def __init__(self, feat_dim, activation, n_rbf, cutoff, dropout):
        super().__init__()

        self.inv_dense = nn.Sequential(
            Dense(
                in_features=feat_dim,
                out_features=feat_dim,
                bias=True,
                dropout_rate=dropout,
                activation=to_module(activation),
            ),
            Dense(in_features=feat_dim, out_features=3 * feat_dim, bias=True, dropout_rate=dropout),
        )

        self.dist_embed = DistanceEmbed(
            n_rbf=n_rbf, cutoff=cutoff, feat_dim=3 * feat_dim, dropout=dropout
        )

    def forward(self, s_i, v_i, r_iI, mapping):
        dist, unit = preprocess_r(r_iI)
        phi = self.inv_dense(s_i)

        w_s = self.dist_embed(dist)

        inv_out = phi * w_s
        inv_out = inv_out.reshape(inv_out.shape[0], 3, -1)

        split_0 = inv_out[:, 0, :].unsqueeze(-1)
        split_1 = inv_out[:, 1, :]
        split_2 = inv_out[:, 2, :].unsqueeze(-1)

        unit_add = split_2 * unit.unsqueeze(1)
        delta_v_iI = unit_add + split_0 * v_i
        delta_s_iI = split_1

        delta_v_I = scatter_add(src=delta_v_iI, index=mapping, dim=0)
        delta_s_I = scatter_add(src=delta_s_iI, index=mapping, dim=0)

        return delta_s_I, delta_v_I


# ---------------------------------------------------------------------------
# CoarseGrainingVAE/cgvae.py
# ---------------------------------------------------------------------------


class EquivariantDecoder(nn.Module):
    def __init__(self, n_atom_basis, n_rbf, cutoff, num_conv, activation, cross_flag=True):
        nn.Module.__init__(self)

        if cross_flag:
            self.message_blocks = nn.ModuleList(
                [
                    EquiMessageCross(
                        feat_dim=n_atom_basis,
                        activation=activation,
                        n_rbf=n_rbf,
                        cutoff=cutoff,
                        dropout=0.0,
                    )
                    for _ in range(num_conv)
                ]
            )
        else:
            self.message_blocks = nn.ModuleList(
                [
                    EquiMessageBlock(
                        feat_dim=n_atom_basis,
                        activation=activation,
                        n_rbf=n_rbf,
                        cutoff=cutoff,
                        dropout=0.0,
                    )
                    for _ in range(num_conv)
                ]
            )

        self.update_blocks = nn.ModuleList(
            [
                UpdateBlock(feat_dim=n_atom_basis, activation=activation, dropout=0.0)
                for _ in range(num_conv)
            ]
        )

        self.n_atom_basis = n_atom_basis

    def forward(self, cg_xyz, CG_nbr_list, mapping, H):
        CG_nbr_list, _ = make_directed(CG_nbr_list)
        r_ij = cg_xyz[CG_nbr_list[:, 1]] - cg_xyz[CG_nbr_list[:, 0]]

        V = torch.zeros(H.shape[0], H.shape[1], 3).to(H.device)

        for i, message_block in enumerate(self.message_blocks):
            dH_message, dV_message = message_block(
                s_j=H, v_j=V, r_ij=r_ij, nbrs=CG_nbr_list, edge_wgt=None
            )
            H = H + dH_message
            V = V + dV_message

            dH_update, dV_update = self.update_blocks[i](s_i=H, v_i=V)
            H = H + dH_update
            V = V + dV_update

        return H, V


class EquiEncoder(nn.Module):
    def __init__(self, n_conv, n_atom_basis, n_rbf, activation, cutoff, dir_mp=False, cg_mp=False):
        super().__init__()

        self.atom_embed = nn.Embedding(100, n_atom_basis, padding_idx=0)
        self.dist_embed = DistanceEmbed(
            n_rbf=n_rbf, cutoff=cutoff, feat_dim=n_atom_basis, dropout=0.0
        )

        self.message_blocks = nn.ModuleList(
            [
                EquiMessageBlock(
                    feat_dim=n_atom_basis,
                    activation=activation,
                    n_rbf=n_rbf,
                    cutoff=cutoff,
                    dropout=0.0,
                )
                for _ in range(n_conv)
            ]
        )

        self.update_blocks = nn.ModuleList(
            [
                UpdateBlock(feat_dim=n_atom_basis, activation=activation, dropout=0.0)
                for _ in range(n_conv)
            ]
        )

        self.cg_message_blocks = nn.ModuleList(
            [
                EquiMessageBlock(
                    feat_dim=n_atom_basis,
                    activation=activation,
                    n_rbf=n_rbf,
                    cutoff=cutoff,
                    dropout=0.0,
                )
                for _ in range(n_conv)
            ]
        )

        self.cg_update_blocks = nn.ModuleList(
            [
                UpdateBlock(feat_dim=n_atom_basis, activation=activation, dropout=0.0)
                for _ in range(n_conv)
            ]
        )

        self.cgmessage_layers = nn.ModuleList(
            [
                ContractiveMessageBlock(
                    feat_dim=n_atom_basis,
                    activation=activation,
                    n_rbf=n_rbf,
                    cutoff=20.0,
                    dropout=0.0,
                )
                for _ in range(n_conv)
            ]
        )

        self.atom2CGcouplings = nn.ModuleList(
            [
                nn.Sequential(
                    Dense(
                        in_features=n_atom_basis,
                        out_features=n_atom_basis,
                        bias=True,
                        activation=to_module(activation),
                    ),
                    Dense(in_features=n_atom_basis, out_features=n_atom_basis, bias=True),
                )
                for _ in range(n_conv)
            ]
        )

        self.n_conv = n_conv
        self.dir_mp = dir_mp
        self.cg_mp = cg_mp
        self.n_atom_basis = n_atom_basis

    def forward(self, z, xyz, cg_xyz, mapping, nbr_list, cg_nbr_list):
        if not self.dir_mp:
            nbr_list, _ = make_directed(nbr_list)
        cg_nbr_list, _ = make_directed(cg_nbr_list)

        h = self.atom_embed(z.long())
        v = torch.zeros(h.shape[0], h.shape[1], 3).to(h.device)

        r_ij = xyz[nbr_list[:, 1]] - xyz[nbr_list[:, 0]]
        r_iI = xyz - cg_xyz[mapping]

        H, V = None, None
        for i in range(self.n_conv):
            ds_message, dv_message = self.message_blocks[i](s_j=h, v_j=v, r_ij=r_ij, nbrs=nbr_list)
            h = h + ds_message
            v = v + dv_message

            if i == 0:
                H = scatter_mean(h, mapping, dim=0)
                V = scatter_mean(v, mapping, dim=0)

            dH, dV = self.cgmessage_layers[i](h, v, r_iI, mapping)

            H = H + dH
            V = V + dV

        return H, h


class CGequiVAE(nn.Module):
    def __init__(
        self,
        encoder,
        equivaraintconv,
        atom_munet,
        atom_sigmanet,
        n_cgs,
        feature_dim,
        prior_net=None,
        det=False,
        equivariant=True,
        offset=True,
    ):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.equivaraintconv = equivaraintconv
        self.atom_munet = atom_munet
        self.atom_sigmanet = atom_sigmanet

        self.n_cgs = n_cgs
        self.prior_net = prior_net
        self.det = det

        self.offset = offset
        self.equivariant = equivariant
        if equivariant is False:
            self.euclidean = nn.Linear(self.encoder.n_atom_basis, self.encoder.n_atom_basis * 3)

    def get_inputs(self, batch):
        xyz = batch["nxyz"][:, 1:]
        cg_xyz = batch["CG_nxyz"][:, 1:]
        cg_z = batch["CG_nxyz"][:, 0]
        z = batch["nxyz"][:, 0]
        mapping = batch["CG_mapping"]
        nbr_list = batch["nbr_list"]
        CG_nbr_list = batch["CG_nbr_list"]
        num_CGs = batch["num_CGs"]
        return z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs

    def reparametrize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        S_I = eps.mul(sigma).add_(mu)
        return S_I

    def CG2ChannelIdx(self, CG_mapping):
        CG2atomChannel = torch.zeros_like(CG_mapping).to("cpu")
        for cg_type in torch.unique(CG_mapping):
            cg_filter = CG_mapping == cg_type
            num_contri_atoms = cg_filter.sum().item()
            CG2atomChannel[cg_filter] = torch.LongTensor(list(range(num_contri_atoms)))
        return CG2atomChannel.detach()

    def decoder(self, cg_xyz, CG_nbr_list, S_I, s_i, mapping, num_CGs):
        cg_s, cg_v = self.equivaraintconv(cg_xyz, CG_nbr_list, mapping, S_I)

        CG2atomChannel = self.CG2ChannelIdx(mapping)

        if self.equivariant is False:
            dv = self.euclidean(cg_s).reshape(cg_s.shape[0], cg_s.shape[1], 3)
            xyz_rel = dv[mapping, CG2atomChannel, :]
        else:
            xyz_rel = cg_v[mapping, CG2atomChannel, :]

        if self.offset:
            decode_offsets = scatter_mean(xyz_rel, mapping, dim=0)
            xyz_rel = xyz_rel - decode_offsets[mapping]

        xyz_recon = xyz_rel + cg_xyz[mapping]

        return xyz_recon

    def forward(self, batch):
        atomic_nums, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs = self.get_inputs(
            batch
        )

        S_I, s_i = self.encoder(atomic_nums, xyz, cg_xyz, mapping, nbr_list, CG_nbr_list)

        if self.prior_net:
            H_prior_mu, H_prior_sigma = self.prior_net(cg_z, cg_xyz, CG_nbr_list)
        else:
            H_prior_mu, H_prior_sigma = None, None

        z = S_I

        mu = self.atom_munet(z)
        logvar = self.atom_sigmanet(z)
        sigma = 1e-12 + torch.exp(logvar / 2)

        if not self.det:
            z_sample = self.reparametrize(mu, sigma)
        else:
            z_sample = z

        S_I = z_sample

        xyz_recon = self.decoder(cg_xyz, CG_nbr_list, S_I, s_i, mapping, num_CGs)

        return mu, sigma, H_prior_mu, H_prior_sigma, xyz, xyz_recon


# ---------------------------------------------------------------------------
# build_ / example_input_
# ---------------------------------------------------------------------------


def build_cgvae():
    """CGequiVAE as constructed in CoarseGrainingVAE/scripts/run_ala.py /
    run_pdb.py: an EquiEncoder over the all-atom graph feeding a
    ContractiveMessageBlock-based CG contraction, a reparametrized latent,
    and an EquivariantDecoder (cross_flag=True) over the CG graph."""
    torch.manual_seed(0)
    n_atom_basis = 16
    n_rbf = 6
    cutoff = 5.0
    n_conv_enc = 2
    n_conv_dec = 2
    activation = "shifted_softplus"
    n_cgs = 3

    encoder = EquiEncoder(
        n_conv=n_conv_enc,
        n_atom_basis=n_atom_basis,
        n_rbf=n_rbf,
        activation=activation,
        cutoff=cutoff,
    )

    decoder_conv = EquivariantDecoder(
        n_atom_basis=n_atom_basis,
        n_rbf=n_rbf,
        cutoff=cutoff,
        num_conv=n_conv_dec,
        activation=activation,
        cross_flag=True,
    )

    atom_munet = nn.Sequential(
        Dense(n_atom_basis, n_atom_basis, activation=to_module(activation)),
        Dense(n_atom_basis, n_atom_basis),
    )
    atom_sigmanet = nn.Sequential(
        Dense(n_atom_basis, n_atom_basis, activation=to_module(activation)),
        Dense(n_atom_basis, n_atom_basis),
    )

    model = CGequiVAE(
        encoder=encoder,
        equivaraintconv=decoder_conv,
        atom_munet=atom_munet,
        atom_sigmanet=atom_sigmanet,
        n_cgs=n_cgs,
        feature_dim=n_atom_basis,
        prior_net=None,
        det=False,
        equivariant=True,
        offset=True,
    )
    model.eval()
    return model


def example_input_cgvae():
    """A tiny synthetic single-molecule all-atom + CG batch dict, matching
    the `batch` schema `CGequiVAE.get_inputs` expects (as produced by the
    real repo's `CGDataset`/`CG_collate` in CoarseGrainingVAE/data.py, which
    itself needs the original MD trajectory .pkl files -- a data-loading
    concern, not part of the model definition)."""
    torch.manual_seed(0)
    n_atoms = 6
    n_cgs = 3

    nxyz = torch.cat([torch.randint(1, 10, (n_atoms, 1)).float(), torch.randn(n_atoms, 3)], dim=1)
    CG_nxyz = torch.cat([torch.randint(1, 10, (n_cgs, 1)).float(), torch.randn(n_cgs, 3)], dim=1)

    # every atom maps to a CG bead (2 atoms per bead)
    CG_mapping = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

    # simple chain neighbor list for the all-atom graph
    nbr_list = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.long)
    # fully connected CG graph (3 beads)
    CG_nbr_list = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.long)

    batch = {
        "nxyz": nxyz,
        "CG_nxyz": CG_nxyz,
        "CG_mapping": CG_mapping,
        "nbr_list": nbr_list,
        "CG_nbr_list": CG_nbr_list,
        "num_CGs": torch.tensor([n_cgs]),
    }
    return (batch,)


MENAGERIE_ENTRIES = [
    (
        "CGVAE (coarse-grained VAE super-resolution)",
        "build_cgvae",
        "example_input_cgvae",
        2022,
        MENAGERIE_ZOO,
    ),
]
