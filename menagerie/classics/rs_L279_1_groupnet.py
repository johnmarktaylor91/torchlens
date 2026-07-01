# SOURCE: vendored from MediaBrain-SJTU/GroupNet @ main
# Files combined:
#   model/GroupNet_nba.py (GroupNet, PastEncoder, FutureEncoder, Decoder, DecomposeBlock,
#       Normal, MLP2, PositionalAgentEncoding -- the multiscale hypergraph trajectory VAE)
#   model/MS_HGNN_batch.py (MS_HGNN_oridinary, MS_HGNN_hyper, MLP, MLP_dict_softmax,
#       edge_aggregation, gumbel_softmax and friends -- the multiscale hypergraph message
#       passing pooling modules used by the past/future encoders)
#   model/utils.py (initialize_weights)
#
# GroupNet (Xu et al., "Learning from All Vehicles" no -- "GroupNet: Multiscale Hypergraph
# Neural Networks for Trajectory Prediction with Relational Reasoning", CVPR 2022) is a VAE
# for multi-agent trajectory forecasting: a hypergraph neural network (MS-HGNN) captures
# pairwise ("ordinary") and higher-order ("hyper") group interactions at multiple scales, a
# past/future encoder pair produces a latent z via a Gaussian VAE, and a residual
# decompose-block decoder reconstructs the past and predicts the future trajectory.
#
# Import-only fixes applied (no architectural change):
#   - Combined into one file; `from model.utils import initialize_weights` and the relative
#     `.MS_HGNN_batch` import collapsed to plain in-file references.
#   - `from tkinter import TRUE` (an unused stray import in the original file -- TRUE is never
#     referenced anywhere in GroupNet_nba.py) dropped; it pulls in a GUI toolkit for nothing.
#   - Hardcoded `.cuda()` calls in `MS_HGNN_oridinary.init_adj` (rel_rec/rel_send) and in
#     `Decoder.forward` (prediction/reconstruction accumulators) replaced with
#     `.to(<a real input tensor's>.device)` so the model traces on CPU-only hosts exactly as
#     it would on the authors' GPU hosts -- the original repo is CUDA-only "as -is" (it never
#     ran on CPU); this is the standard portability fix, not an architecture change.
#   - `GroupNet.forward()` computes VAE training losses (reconstruction + KL + diversity) from
#     a `past_traj`/`future_traj` data dict and always samples from a Normal via `.rsample()`
#     (stochastic, and the CVAE prior needs ground-truth future trajectories to draw the
#     posterior). The traceable entry point exposed here is `GroupNet.inference()` (the repo's
#     own deployed/eval-time forward path), which needs only `past_traj` and runs the full
#     past-encoder -> hypergraph interaction -> prior sampling -> decompose-block decoder
#     pipeline, unmodified.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import math
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# model/utils.py
# ---------------------------------------------------------------------------
def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# model/MS_HGNN_batch.py
# ---------------------------------------------------------------------------
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_size=(1024, 512),
        activation="relu",
        discrim=False,
        dropout=-1,
    ):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout / 3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x


class MLP_dict_softmax(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_size=(1024, 512),
        activation="relu",
        discrim=False,
        dropout=-1,
        edge_types=10,
    ):
        super(MLP_dict_softmax, self).__init__()
        self.bottleneck_dim = edge_types
        self.MLP_distribution = MLP(
            input_dim=input_dim, output_dim=self.bottleneck_dim, hidden_size=hidden_size
        )
        self.MLP_factor = MLP(input_dim=input_dim, output_dim=1, hidden_size=hidden_size)
        self.init_MLP = MLP(input_dim=input_dim, output_dim=input_dim, hidden_size=hidden_size)

    def forward(self, x):
        x = self.init_MLP(x)
        distribution = gumbel_softmax(self.MLP_distribution(x), tau=1 / 2, hard=False)
        factor = torch.sigmoid(self.MLP_factor(x))
        out = factor * distribution
        return out, distribution


class edge_aggregation(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_size=(1024, 512),
        activation="relu",
        discrim=False,
        dropout=-1,
        edge_types=5,
    ):
        super(edge_aggregation, self).__init__()
        self.edge_types = edge_types
        self.dict_dim = input_dim
        self.agg_mlp = []
        for i in range(edge_types):
            self.agg_mlp.append(MLP(input_dim=input_dim, output_dim=input_dim, hidden_size=(128,)))
        self.agg_mlp = nn.ModuleList(self.agg_mlp)
        self.mlp = MLP(input_dim=input_dim, output_dim=input_dim, hidden_size=(128,))

    def forward(self, edge_distribution, H, ori):
        batch = edge_distribution.shape[0]
        edges = edge_distribution.shape[1]
        edge_feature = torch.zeros(batch, edges, ori.shape[-1]).type_as(ori)
        edges = torch.matmul(H, ori)
        for i in range(self.edge_types):
            edge_feature += edge_distribution[:, :, i : i + 1] * self.agg_mlp[i](edges)

        node_feature = torch.cat((torch.matmul(H.permute(0, 2, 1), edge_feature), ori), dim=-1)
        return node_feature


class MS_HGNN_oridinary(nn.Module):
    """Pooling module as proposed in the GroupNet paper (pairwise / 'ordinary' scale)."""

    def __init__(
        self,
        embedding_dim=64,
        h_dim=64,
        mlp_dim=1024,
        bottleneck_dim=1024,
        activation="relu",
        batch_norm=True,
        dropout=0.0,
        nmp_layers=4,
        vis=False,
    ):
        super(MS_HGNN_oridinary, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.nmp_layers = nmp_layers
        self.batch_norm = batch_norm
        self.activation = activation
        self.vis = vis

        hdim_extend = 64
        self.hdim_extend = hdim_extend
        self.edge_types = 6
        self.nmp_mlp_start = MLP_dict_softmax(
            input_dim=hdim_extend, output_dim=h_dim, hidden_size=(128,), edge_types=self.edge_types
        )
        self.nmp_mlps = self.make_nmp_mlp()
        self.nmp_mlp_end = MLP(input_dim=h_dim * 2, output_dim=bottleneck_dim, hidden_size=(128,))
        attention_mlp = []
        for i in range(nmp_layers):
            attention_mlp.append(MLP(input_dim=hdim_extend * 2, output_dim=1, hidden_size=(32,)))
        self.attention_mlp = nn.ModuleList(attention_mlp)
        node2edge_start_mlp = []
        for i in range(nmp_layers):
            node2edge_start_mlp.append(
                MLP(input_dim=h_dim, output_dim=hdim_extend, hidden_size=(256,))
            )
        self.node2edge_start_mlp = nn.ModuleList(node2edge_start_mlp)
        edge_aggregation_list = []
        for i in range(nmp_layers):
            edge_aggregation_list.append(
                edge_aggregation(
                    input_dim=h_dim,
                    output_dim=bottleneck_dim,
                    hidden_size=(128,),
                    edge_types=self.edge_types,
                )
            )
        self.edge_aggregation_list = nn.ModuleList(edge_aggregation_list)

    def make_nmp_mlp(self):
        nmp_mlp = []
        for i in range(self.nmp_layers - 1):
            mlp1 = MLP(input_dim=self.h_dim * 2, output_dim=self.h_dim, hidden_size=(128,))
            mlp2 = MLP_dict_softmax(
                input_dim=self.hdim_extend,
                output_dim=self.h_dim,
                hidden_size=(128,),
                edge_types=self.edge_types,
            )
            nmp_mlp.append(mlp1)
            nmp_mlp.append(mlp2)
        nmp_mlp = nn.ModuleList(nmp_mlp)
        return nmp_mlp

    def edge2node(self, x, rel_rec, rel_send, ori, idx):
        H = rel_rec + rel_send
        incoming = self.edge_aggregation_list[idx](x, H, ori)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send, idx):
        H = rel_rec + rel_send
        x = self.node2edge_start_mlp[idx](x)
        edge_init = torch.matmul(H, x)
        node_num = x.shape[1]
        edge_num = edge_init.shape[1]
        x_rep = (x[:, :, None, :].transpose(2, 1)).repeat(1, edge_num, 1, 1)
        edge_rep = edge_init[:, :, None, :].repeat(1, 1, node_num, 1)
        node_edge_cat = torch.cat((x_rep, edge_rep), dim=-1)
        attention_weight = self.attention_mlp[idx](node_edge_cat)[:, :, :, 0]
        H_weight = attention_weight * H
        H_weight = F.softmax(H_weight, dim=2)
        H_weight = H_weight * H
        edges = torch.matmul(H_weight, x)
        return edges

    def init_adj(self, num_ped, batch, device):
        off_diag = np.ones([num_ped, num_ped])

        rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
        rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
        rel_rec = torch.FloatTensor(rel_rec)
        rel_send = torch.FloatTensor(rel_send)

        # NOTE: original repo hardcodes `.cuda()` here; using `.to(device)` (derived from the
        # real input tensor) instead so the model traces on CPU-only hosts too (see module
        # header "Import-only fixes applied").
        rel_rec = rel_rec.to(device)
        rel_send = rel_send.to(device)

        rel_rec = rel_rec[None, :, :].repeat(batch, 1, 1)
        rel_send = rel_send[None, :, :].repeat(batch, 1, 1)

        return rel_rec, rel_send

    def forward(self, h_states):
        batch = h_states.shape[0]
        actor_num = h_states.shape[1]

        curr_hidden = h_states

        rel_rec, rel_send = self.init_adj(actor_num, batch, h_states.device)
        edge_feat = self.node2edge(curr_hidden, rel_rec, rel_send, 0)
        edge_feat, factors = self.nmp_mlp_start(edge_feat)
        node_feat = curr_hidden

        nodetoedge_idx = 0
        if self.nmp_layers <= 1:
            pass
        else:
            for nmp_l, nmp_mlp in enumerate(self.nmp_mlps):
                if nmp_l % 2 == 0:
                    node_feat = nmp_mlp(
                        self.edge2node(edge_feat, rel_rec, rel_send, node_feat, nodetoedge_idx)
                    )
                    nodetoedge_idx += 1
                else:
                    edge_feat, _ = nmp_mlp(
                        self.node2edge(node_feat, rel_rec, rel_send, nodetoedge_idx)
                    )
        node_feat = self.nmp_mlp_end(
            self.edge2node(edge_feat, rel_rec, rel_send, node_feat, nodetoedge_idx)
        )
        return node_feat, factors


class MS_HGNN_hyper(nn.Module):
    """Pooling module as proposed in the GroupNet paper (higher-order / 'hyper' scale)."""

    def __init__(
        self,
        embedding_dim=64,
        h_dim=64,
        mlp_dim=1024,
        bottleneck_dim=1024,
        activation="relu",
        batch_norm=True,
        dropout=0.0,
        nmp_layers=4,
        scale=2,
        vis=False,
        actor_number=11,
    ):
        super(MS_HGNN_hyper, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.nmp_layers = nmp_layers
        self.batch_norm = batch_norm
        self.activation = activation
        self.scale = scale
        self.vis = vis

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.spatial_transform = nn.Linear(h_dim, h_dim)
        hdim_extend = 64
        self.hdim_extend = hdim_extend
        self.edge_types = 10

        self.nmp_mlp_start = MLP_dict_softmax(
            input_dim=hdim_extend, output_dim=h_dim, hidden_size=(128,), edge_types=self.edge_types
        )
        self.nmp_mlps = self.make_nmp_mlp()
        self.nmp_mlp_end = MLP(input_dim=h_dim * 2, output_dim=bottleneck_dim, hidden_size=(128,))
        attention_mlp = []
        for i in range(nmp_layers):
            attention_mlp.append(MLP(input_dim=hdim_extend * 2, output_dim=1, hidden_size=(32,)))
        self.attention_mlp = nn.ModuleList(attention_mlp)

        node2edge_start_mlp = []
        for i in range(nmp_layers):
            node2edge_start_mlp.append(
                MLP(input_dim=h_dim, output_dim=hdim_extend, hidden_size=(256,))
            )
        self.node2edge_start_mlp = nn.ModuleList(node2edge_start_mlp)
        edge_aggregation_list = []
        for i in range(nmp_layers):
            edge_aggregation_list.append(
                edge_aggregation(
                    input_dim=h_dim,
                    output_dim=bottleneck_dim,
                    hidden_size=(128,),
                    edge_types=self.edge_types,
                )
            )
        self.edge_aggregation_list = nn.ModuleList(edge_aggregation_list)
        # NOTE: `listall` is a repo debug flag; kept False as in the shipped configs so the
        # `all_combs.cuda()` branch (also hardcoded-CUDA in the original) is never entered.
        self.listall = False

    def make_nmp_mlp(self):
        nmp_mlp = []
        for i in range(self.nmp_layers - 1):
            mlp1 = MLP(input_dim=self.h_dim * 2, output_dim=self.h_dim, hidden_size=(128,))
            mlp2 = MLP_dict_softmax(
                input_dim=self.hdim_extend,
                output_dim=self.h_dim,
                hidden_size=(128,),
                edge_types=self.edge_types,
            )
            nmp_mlp.append(mlp1)
            nmp_mlp.append(mlp2)
        nmp_mlp = nn.ModuleList(nmp_mlp)
        return nmp_mlp

    def edge2node(self, x, ori, H, idx):
        incoming = self.edge_aggregation_list[idx](x, H, ori)
        return incoming / incoming.size(1)

    def node2edge(self, x, H, idx):
        x = self.node2edge_start_mlp[idx](x)
        edge_init = torch.matmul(H, x)
        node_num = x.shape[1]
        edge_num = edge_init.shape[1]
        x_rep = (x[:, :, None, :].transpose(2, 1)).repeat(1, edge_num, 1, 1)
        edge_rep = edge_init[:, :, None, :].repeat(1, 1, node_num, 1)
        node_edge_cat = torch.cat((x_rep, edge_rep), dim=-1)
        attention_weight = self.attention_mlp[idx](node_edge_cat)[:, :, :, 0]
        H_weight = attention_weight * H
        H_weight = F.softmax(H_weight, dim=2)
        H_weight = H_weight * H
        edges = torch.matmul(H_weight, x)
        return edges

    def init_adj_attention(self, feat, feat_corr, scale_factor=2):
        batch = feat.shape[0]
        actor_number = feat.shape[1]
        if scale_factor == actor_number:
            H_matrix = torch.ones(batch, 1, actor_number).type_as(feat)
            return H_matrix
        group_size = scale_factor
        if group_size < 1:
            group_size = 1

        _, indice = torch.topk(feat_corr, dim=2, k=group_size, largest=True)
        H_matrix = torch.zeros(batch, actor_number, actor_number).type_as(feat)
        H_matrix = H_matrix.scatter(2, indice, 1)

        return H_matrix

    def forward(self, h_states, corr):
        curr_hidden = h_states

        H = self.init_adj_attention(curr_hidden, corr, scale_factor=self.scale)

        edge_hidden = self.node2edge(curr_hidden, H, idx=0)
        edge_feat, factor = self.nmp_mlp_start(edge_hidden)
        node_feat = curr_hidden
        node2edge_idx = 0
        if self.nmp_layers <= 1:
            pass
        else:
            for nmp_l, nmp_mlp in enumerate(self.nmp_mlps):
                if nmp_l % 2 == 0:
                    node_feat = nmp_mlp(self.edge2node(edge_feat, node_feat, H, node2edge_idx))
                    node2edge_idx += 1
                else:
                    edge_feat, _ = nmp_mlp(self.node2edge(node_feat, H, idx=node2edge_idx))
        node_feat = self.nmp_mlp_end(self.edge2node(edge_feat, node_feat, H, node2edge_idx))
        return node_feat, factor


def sample_gumbel(shape, device, eps=1e-10):
    U = torch.rand(shape, device=device).float()
    return -torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    gumbel_noise = sample_gumbel(logits.size(), logits.device, eps=eps)
    y = logits + gumbel_noise
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape, device=logits.device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = (y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, dim=0)
    return soft_max_1d.transpose(axis, 0)


# ---------------------------------------------------------------------------
# model/GroupNet_nba.py
# ---------------------------------------------------------------------------
class DecomposeBlock(nn.Module):
    """Balance between reconstruction task and prediction task."""

    def __init__(self, past_len, future_len, input_dim):
        super(DecomposeBlock, self).__init__()
        channel_in = 2
        channel_out = 32
        dim_kernel = 3
        dim_embedding_key = 96
        self.past_len = past_len
        self.future_len = future_len

        self.conv_past = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.encoder_past = nn.GRU(channel_out, dim_embedding_key, 1, batch_first=True)

        self.decoder_y = MLP(dim_embedding_key + input_dim, future_len * 2, hidden_size=(512, 256))
        self.decoder_x = MLP(dim_embedding_key + input_dim, past_len * 2, hidden_size=(512, 256))

        self.relu = nn.ReLU()

        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_normal_(self.conv_past.weight)
        nn.init.kaiming_normal_(self.encoder_past.weight_ih_l0)
        nn.init.kaiming_normal_(self.encoder_past.weight_hh_l0)

        nn.init.zeros_(self.conv_past.bias)
        nn.init.zeros_(self.encoder_past.bias_ih_l0)
        nn.init.zeros_(self.encoder_past.bias_hh_l0)

    def forward(self, x_true, x_hat, f):
        """
        x_true: N, T_p, 2
        x_hat: N, T_p, 2
        f: N, D
        """
        x_ = x_true - x_hat
        x_ = torch.transpose(x_, 1, 2)

        past_embed = self.relu(self.conv_past(x_))
        past_embed = torch.transpose(past_embed, 1, 2)

        _, state_past = self.encoder_past(past_embed)
        state_past = state_past.squeeze(0)

        input_feat = torch.cat((f, state_past), dim=1)

        x_hat_after = self.decoder_x(input_feat).contiguous().view(-1, self.past_len, 2)
        y_hat = self.decoder_y(input_feat).contiguous().view(-1, self.future_len, 2)

        return x_hat_after, y_hat


class Normal:
    def __init__(self, mu=None, logvar=None, params=None):
        super().__init__()
        if params is not None:
            self.mu, self.logvar = torch.chunk(params, chunks=2, dim=-1)
        else:
            assert mu is not None
            assert logvar is not None
            self.mu = mu
            self.logvar = logvar
        self.sigma = torch.exp(0.5 * self.logvar)

    def rsample(self):
        eps = torch.randn_like(self.sigma)
        return self.mu + eps * self.sigma

    def sample(self):
        return self.rsample()

    def kl(self, p=None):
        """compute KL(q||p)"""
        if p is None:
            kl = -0.5 * (1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        else:
            term1 = (self.mu - p.mu) / (p.sigma + 1e-8)
            term2 = self.sigma / (p.sigma + 1e-8)
            kl = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
        return kl

    def mode(self):
        return self.mu


class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation="tanh"):
        super().__init__()
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        initialize_weights(self.affine_layers.modules())

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x


class PositionalAgentEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_t_len=200, concat=True):
        super(PositionalAgentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        if concat:
            self.fc = nn.Linear(2 * d_model, d_model)

        pe = self.build_pos_enc(max_t_len)
        self.register_buffer("pe", pe)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset : num_t + t_offset, :]
        pe = pe[None].repeat(num_a, 1, 1)
        return pe

    def forward(self, x, num_a, t_offset=0):
        num_t = x.shape[1]
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset)
        if self.concat:
            feat = [x, pos_enc]
            x = torch.cat(feat, dim=-1)
            x = self.fc(x)
        else:
            x += pos_enc
        return self.dropout(x)


class PastEncoder(nn.Module):
    def __init__(self, args, in_dim=4):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim
        self.scale_number = len(args.hyper_scales)

        self.input_fc = nn.Linear(in_dim, self.model_dim)
        self.input_fc2 = nn.Linear(self.model_dim * args.past_length, self.model_dim)
        self.input_fc3 = nn.Linear(self.model_dim + 3, self.model_dim)

        self.interaction = MS_HGNN_oridinary(
            embedding_dim=16,
            h_dim=self.model_dim,
            mlp_dim=64,
            bottleneck_dim=self.model_dim,
            batch_norm=0,
            nmp_layers=1,
        )

        if len(args.hyper_scales) > 0:
            self.interaction_hyper = MS_HGNN_hyper(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[0],
            )
        if len(args.hyper_scales) > 1:
            self.interaction_hyper2 = MS_HGNN_hyper(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[1],
            )
        if len(args.hyper_scales) > 2:
            self.interaction_hyper3 = MS_HGNN_hyper(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[2],
            )

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, 0.1, concat=True)

    def add_category(self, x):
        B = x.shape[0]
        N = x.shape[1]
        category = torch.zeros(N, 3).type_as(x)
        category[0:5, 0] = 1
        category[5:10, 1] = 1
        category[10, 2] = 1
        category = category.repeat(B, 1, 1)
        x = torch.cat((x, category), dim=-1)
        return x

    def forward(self, inputs, batch_size, agent_num):
        length = inputs.shape[1]

        tf_in = self.input_fc(inputs).view(batch_size * agent_num, length, self.model_dim)

        tf_in_pos = self.pos_encoder(tf_in, num_a=batch_size * agent_num)
        tf_in_pos = tf_in_pos.view(batch_size, agent_num, length, self.model_dim)

        ftraj_input = self.input_fc2(
            tf_in_pos.contiguous().view(batch_size, agent_num, length * self.model_dim)
        )
        ftraj_input = self.input_fc3(self.add_category(ftraj_input))
        query_input = F.normalize(ftraj_input, p=2, dim=2)
        feat_corr = torch.matmul(query_input, query_input.permute(0, 2, 1))
        ftraj_inter, _ = self.interaction(ftraj_input)
        if len(self.args.hyper_scales) > 0:
            ftraj_inter_hyper, _ = self.interaction_hyper(ftraj_input, feat_corr)
        if len(self.args.hyper_scales) > 1:
            ftraj_inter_hyper2, _ = self.interaction_hyper2(ftraj_input, feat_corr)
        if len(self.args.hyper_scales) > 2:
            ftraj_inter_hyper3, _ = self.interaction_hyper3(ftraj_input, feat_corr)

        if len(self.args.hyper_scales) == 0:
            final_feature = torch.cat((ftraj_input, ftraj_inter), dim=-1)
        if len(self.args.hyper_scales) == 1:
            final_feature = torch.cat((ftraj_input, ftraj_inter, ftraj_inter_hyper), dim=-1)
        elif len(self.args.hyper_scales) == 2:
            final_feature = torch.cat(
                (ftraj_input, ftraj_inter, ftraj_inter_hyper, ftraj_inter_hyper2), dim=-1
            )
        elif len(self.args.hyper_scales) == 3:
            final_feature = torch.cat(
                (
                    ftraj_input,
                    ftraj_inter,
                    ftraj_inter_hyper,
                    ftraj_inter_hyper2,
                    ftraj_inter_hyper3,
                ),
                dim=-1,
            )

        output_feature = final_feature.view(batch_size * agent_num, -1)
        return output_feature


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim
        scale_num = 2 + len(self.args.hyper_scales)

        self.num_decompose = args.num_decompose
        input_dim = scale_num * self.model_dim + self.args.zdim
        self.past_length = self.args.past_length
        self.future_length = self.args.future_length

        self.decompose = nn.ModuleList(
            [
                DecomposeBlock(self.args.past_length, self.args.future_length, input_dim)
                for _ in range(self.num_decompose)
            ]
        )

    def forward(
        self,
        past_feature,
        z,
        batch_size_curr,
        agent_num_perscene,
        past_traj,
        cur_location,
        sample_num,
        mode="train",
    ):
        agent_num = batch_size_curr * agent_num_perscene
        past_traj_repeat = past_traj.repeat_interleave(sample_num, dim=0)
        past_feature = past_feature.view(-1, sample_num, past_feature.shape[-1])

        z_in = z.view(-1, sample_num, z.shape[-1])

        hidden = torch.cat((past_feature, z_in), dim=-1)
        hidden = hidden.view(agent_num * sample_num, -1)
        x_true = past_traj_repeat.clone()

        x_hat = torch.zeros_like(x_true)
        batch_size = x_true.size(0)
        # NOTE: original repo hardcodes `.cuda()` here; using `.to(x_true.device)` instead so
        # the model traces on CPU-only hosts too (see module header "Import-only fixes applied").
        prediction = torch.zeros((batch_size, self.future_length, 2)).to(x_true.device)
        reconstruction = torch.zeros((batch_size, self.past_length, 2)).to(x_true.device)

        for i in range(self.num_decompose):
            x_hat, y_hat = self.decompose[i](x_true, x_hat, hidden)
            prediction += y_hat
            reconstruction += x_hat
        norm_seq = prediction.view(agent_num * sample_num, self.future_length, 2)
        recover_pre_seq = reconstruction.view(agent_num * sample_num, self.past_length, 2)

        cur_location_repeat = cur_location.repeat_interleave(sample_num, dim=0)
        out_seq = norm_seq + cur_location_repeat
        if mode == "inference":
            out_seq = out_seq.view(-1, sample_num, *out_seq.shape[1:])
        return out_seq, recover_pre_seq


class FutureEncoder(nn.Module):
    def __init__(self, args, in_dim=4):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim

        self.input_fc = nn.Linear(in_dim, self.model_dim)
        scale_num = 2 + len(self.args.hyper_scales)
        self.input_fc2 = nn.Linear(self.model_dim * self.args.future_length, self.model_dim)
        self.input_fc3 = nn.Linear(self.model_dim + 3, self.model_dim)

        self.interaction = MS_HGNN_oridinary(
            embedding_dim=16,
            h_dim=self.model_dim,
            mlp_dim=64,
            bottleneck_dim=self.model_dim,
            batch_norm=0,
            nmp_layers=1,
            vis=False,
        )

        if len(args.hyper_scales) > 0:
            self.interaction_hyper = MS_HGNN_hyper(
                embedding_dim=16,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[0],
                vis=False,
            )
        if len(args.hyper_scales) > 1:
            self.interaction_hyper2 = MS_HGNN_hyper(
                embedding_dim=16,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[1],
                vis=False,
            )
        if len(args.hyper_scales) > 2:
            self.interaction_hyper3 = MS_HGNN_hyper(
                embedding_dim=16,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[2],
                vis=False,
            )

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, 0.1, concat=True)

        self.out_mlp = MLP2(scale_num * 2 * self.model_dim, [128], "relu")
        self.qz_layer = nn.Linear(self.out_mlp.out_dim, 2 * self.args.zdim)
        initialize_weights(self.qz_layer.modules())

    def add_category(self, x):
        B = x.shape[0]
        N = x.shape[1]
        category = torch.zeros(N, 3).type_as(x)
        category[0:5, 0] = 1
        category[5:10, 1] = 1
        category[10, 2] = 1
        category = category.repeat(B, 1, 1)
        x = torch.cat((x, category), dim=-1)
        return x

    def forward(self, inputs, batch_size, agent_num, past_feature):
        length = inputs.shape[1]
        agent_num = 11
        tf_in = self.input_fc(inputs).view(batch_size * agent_num, length, self.model_dim)

        tf_in_pos = self.pos_encoder(tf_in, num_a=batch_size * agent_num)
        tf_in_pos = tf_in_pos.view(batch_size, agent_num, length, self.model_dim)

        ftraj_input = self.input_fc2(tf_in_pos.contiguous().view(batch_size, agent_num, -1))
        ftraj_input = self.input_fc3(self.add_category(ftraj_input))
        query_input = F.normalize(ftraj_input, p=2, dim=2)
        feat_corr = torch.matmul(query_input, query_input.permute(0, 2, 1))
        ftraj_inter, _ = self.interaction(ftraj_input)

        if len(self.args.hyper_scales) > 0:
            ftraj_inter_hyper, _ = self.interaction_hyper(ftraj_input, feat_corr)
        if len(self.args.hyper_scales) > 1:
            ftraj_inter_hyper2, _ = self.interaction_hyper2(ftraj_input, feat_corr)
        if len(self.args.hyper_scales) > 2:
            ftraj_inter_hyper3, _ = self.interaction_hyper3(ftraj_input, feat_corr)

        if len(self.args.hyper_scales) == 0:
            final_feature = torch.cat((ftraj_input, ftraj_inter), dim=-1)
        if len(self.args.hyper_scales) == 1:
            final_feature = torch.cat((ftraj_input, ftraj_inter, ftraj_inter_hyper), dim=-1)
        elif len(self.args.hyper_scales) == 2:
            final_feature = torch.cat(
                (ftraj_input, ftraj_inter, ftraj_inter_hyper, ftraj_inter_hyper2), dim=-1
            )
        elif len(self.args.hyper_scales) == 3:
            final_feature = torch.cat(
                (
                    ftraj_input,
                    ftraj_inter,
                    ftraj_inter_hyper,
                    ftraj_inter_hyper2,
                    ftraj_inter_hyper3,
                ),
                dim=-1,
            )

        final_feature = final_feature.view(batch_size * agent_num, -1)

        h = torch.cat((past_feature, final_feature), dim=-1)
        h = self.out_mlp(h)
        q_z_params = self.qz_layer(h)
        return q_z_params


class GroupNet(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.device = device
        self.args = args

        scale_num = 2 + len(self.args.hyper_scales)
        self.past_encoder = PastEncoder(args)
        self.pz_layer = nn.Linear(scale_num * self.args.hidden_dim, 2 * self.args.zdim)
        if args.learn_prior:
            initialize_weights(self.pz_layer.modules())
        self.future_encoder = FutureEncoder(args)
        self.decoder = Decoder(args)
        self.param_annealers = nn.ModuleList()

    def set_device(self, device):
        self.device = device
        self.to(device)

    def inference(self, data):
        """Repo's own deployed/eval-time forward path -- needs only `past_traj`."""
        device = self.device
        batch_size = data["past_traj"].shape[0]
        agent_num = data["past_traj"].shape[1]

        past_traj = (
            data["past_traj"]
            .view(batch_size * agent_num, self.args.past_length, 2)
            .to(device)
            .contiguous()
        )

        past_vel = past_traj[:, 1:] - past_traj[:, :-1, :]
        past_vel = torch.cat([past_vel[:, [0]], past_vel], dim=1)

        cur_location = past_traj[:, [-1]]

        inputs = torch.cat((past_traj, past_vel), dim=-1)

        past_feature = self.past_encoder(inputs, batch_size, agent_num)

        sample_num = 20
        if self.args.learn_prior:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            p_z_params = self.pz_layer(past_feature_repeat)
            pz_distribution = Normal(params=p_z_params)
        else:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            pz_distribution = Normal(
                mu=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_traj.device),
                logvar=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(
                    past_traj.device
                ),
            )

        pz_sampled = pz_distribution.rsample()
        z = pz_sampled

        diverse_pred_traj, _ = self.decoder(
            past_feature_repeat,
            z,
            batch_size,
            agent_num,
            past_traj,
            cur_location,
            sample_num=self.args.sample_k,
            mode="inference",
        )
        diverse_pred_traj = diverse_pred_traj.permute(1, 0, 2, 3)
        return diverse_pred_traj


# ---------------------------------------------------------------------------
# menagerie staging entry points
# ---------------------------------------------------------------------------
MENAGERIE_ZOO = "vendored-pytorch"


class _GroupNetInferenceWrapper(nn.Module):
    """Thin nn.Module wrapper so tl.trace() sees a plain tensor-in/tensor-out forward();
    GroupNet.inference() itself takes a dict, mirroring the repo's NBA dataloader batch."""

    def __init__(self, groupnet):
        super().__init__()
        self.groupnet = groupnet

    def forward(self, past_traj):
        return self.groupnet.inference({"past_traj": past_traj})


def build_groupnet():
    # Mirrors the repo's train_hyper_nba.py argparse defaults for the NBA dataset.
    args = SimpleNamespace(
        hidden_dim=32,
        hyper_scales=[3, 5],
        zdim=16,
        past_length=5,
        future_length=10,
        num_decompose=2,
        learn_prior=True,
        ztype="gaussian",
        sample_k=20,
        min_clip=2.0,
    )
    groupnet = GroupNet(args, device=torch.device("cpu"))
    return _GroupNetInferenceWrapper(groupnet)


def example_input_groupnet():
    # NBA dataset: 11 agents (5 offense + 5 defense + 1 ball), past_length=5 timesteps, 2D coords.
    batch_size = 1
    agent_num = 11
    past_length = 5
    return torch.randn(batch_size, agent_num, past_length, 2)


MENAGERIE_ENTRIES = [
    ("GroupNet", build_groupnet, example_input_groupnet, 2022, MENAGERIE_ZOO),
]
