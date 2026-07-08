# SOURCE: vendored from Vicky-51/GRELEN @ main
#
# Vendors the real nn.Module classes from model/GRELEN.py (MLP, Graph_learner attention-based
# graph structure learner, DCGRUCell_ diffusion-convolutional GRU cell, EncoderModel, and the
# top-level Grelen model -- a variational graph-recurrent model that jointly learns a latent
# multi-relation graph over sensor/entity nodes and per-node temporal dynamics via DCGRU,
# used for multivariate time-series anomaly detection, e.g. on the SWaT dataset per the paper
# IJCAI'22 "GRELEN: Graph Recurrent Encoder for Latent ENtities"). The `gumbel_softmax` helper
# (from lib/utils.py, itself lifted verbatim from an upstream pytorch PR) is vendored alongside
# since `Grelen.forward` calls it directly.
#
# Fixes applied (mechanical only, no architecture change): `DCGRUCell_._build_sparse_matrix`
# referenced a module-global `device` name that does not exist anywhere in the original repo
# (a latent bug in the upstream code -- this static method is never called from `Grelen.forward`,
# so it was simply removed rather than guessing at intent); `lib.utils`'s module-level
# `device = torch.device(...)` global (used throughout `gumbel_softmax`) is preserved verbatim
# as a module-level constant here since that is exactly how the original code used it.
#
# Repo: https://github.com/Vicky-51/GRELEN @ main
# Files: model/GRELEN.py, lib/utils.py (gumbel_softmax + helpers)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

MENAGERIE_ZOO = "vendored-pytorch"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- lib/utils.py ----
def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from Gumbel(0, 1)
    """
    U = torch.rand(shape).float().to(device)
    return -torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Draw a sample from the Gumbel-Softmax distribution
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps).to(device)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return F.softmax(y / tau, dim=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(device)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0).to(device)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


# ---- model/GRELEN.py ----
class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class Graph_learner(nn.Module):
    def __init__(self, n_in, n_hid, n_head_dim, head, do_prob=0.0):  # n_in = T
        super(Graph_learner, self).__init__()
        self.n_hid = n_hid
        self.head = head
        self.n_in = n_in
        self.n_head_dim = n_head_dim

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.Wq = nn.Linear(n_hid, n_head_dim * head)
        self.Wk = nn.Linear(n_hid, n_head_dim * head)
        for m in [self.Wq, self.Wk]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs):  # inputs: [B, N, T(features)]
        X = self.mlp1(inputs)
        Xq = self.Wq(X)  # [B, N, n_hid_subspace]
        Xk = self.Wk(X)
        B, N, n_hid = Xq.shape
        Xq = Xq.view(B, N, self.head, self.n_head_dim)  # [B, N, head, head_dim]
        Xk = Xk.view(B, N, self.head, self.n_head_dim)
        Xq = Xq.permute(0, 2, 1, 3)  # [B, head, N, head_dim]
        Xk = Xk.permute(0, 2, 1, 3)
        probs = torch.matmul(Xq, Xk.transpose(-1, -2))  # [B, head, N, N]

        return probs


class DCGRUCell_(torch.nn.Module):
    def __init__(
        self,
        device,
        num_units,
        max_diffusion_step,
        num_nodes,
        nonlinearity="tanh",
        filter_type="laplacian",
        use_gc_for_ru=True,
    ):
        """
        Adapted from Pytorch implementation of DCGRU Cell
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == "tanh" else torch.relu
        self.device = device
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru

        self._gconv_0 = nn.Linear(
            self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units * 2
        )
        self._gconv_1 = nn.Linear(
            self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units * 2
        )
        self._gconv_c_0 = nn.Linear(
            self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units
        )
        self._gconv_c_1 = nn.Linear(
            self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def _calculate_random_walk0(self, adj_mx, B):
        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[1])).unsqueeze(0).repeat(B, 1, 1).to(
            self.device
        )
        d = torch.sum(adj_mx, 1)
        d_inv = 1.0 / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(self.device), d_inv)
        d_mat_inv = torch.diag_embed(d_inv)
        random_walk_mx = torch.matmul(d_mat_inv, adj_mx)
        return random_walk_mx

    def forward(self, inputs, hx, adj):
        output_size = 2 * self._num_units
        fn = self._gconv
        value = torch.sigmoid(fn(inputs, adj, hx, output_size, bias_start=1.0))

        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv_c(inputs, adj, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    def _gconv(self, inputs, adj_mx, state, output_size, bias_start=0.0):
        B = inputs.shape[0]
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        adj_mx0 = self._calculate_random_walk0(adj_mx, B)
        adj_mx1 = self._calculate_random_walk0(adj_mx.permute(0, 2, 1), B)

        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state  # [B, N, 2 * C]
        x0_0 = torch.unsqueeze(x, 0)
        x1_0 = torch.unsqueeze(x, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            x0_1 = torch.matmul(adj_mx0, x0_0)
            x1_1 = torch.matmul(adj_mx1, x1_0)

            x0_0 = torch.cat([x0_0, x0_1], dim=0)
            x1_0 = torch.cat([x1_0, x1_1], dim=0)

            for k in range(2, self._max_diffusion_step + 1):
                x0_2 = torch.matmul(adj_mx0, x0_1)
                x1_2 = torch.matmul(adj_mx1, x1_1)
                x0_0 = torch.cat([x0_0, x0_1], dim=0)
                x1_0 = torch.cat([x1_0, x1_1], dim=0)
                x0_1 = x0_2
                x1_1 = x1_2
        num_matrices = self._max_diffusion_step + 1  # Adds for x itself.
        x0_0 = x0_0.permute(1, 2, 3, 0)
        x1_0 = x1_0.permute(1, 2, 3, 0)
        x0_0 = torch.reshape(x0_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])
        x1_0 = torch.reshape(x1_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        x0_0 = self._gconv_0(x0_0)
        x1_0 = self._gconv_1(x1_0)

        return torch.reshape(x0_0 + x1_0, [batch_size, self._num_nodes * output_size])

    def _gconv_c(self, inputs, adj_mx, state, output_size, bias_start=0.0):
        B = inputs.shape[0]
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        adj_mx0 = self._calculate_random_walk0(adj_mx, B)
        adj_mx1 = self._calculate_random_walk0(adj_mx.permute(0, 2, 1), B)

        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state  # [B, N, 2 * C]
        x0_0 = torch.unsqueeze(x, 0)
        x1_0 = torch.unsqueeze(x, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            x0_1 = torch.matmul(adj_mx0, x0_0)
            x1_1 = torch.matmul(adj_mx1, x1_0)

            x0_0 = torch.cat([x0_0, x0_1], dim=0)
            x1_0 = torch.cat([x1_0, x1_1], dim=0)

            for k in range(2, self._max_diffusion_step + 1):
                x0_2 = torch.matmul(adj_mx0, x0_1)
                x1_2 = torch.matmul(adj_mx1, x1_1)
                x0_0 = torch.cat([x0_0, x0_1], dim=0)
                x1_0 = torch.cat([x1_0, x1_1], dim=0)
                x0_1 = x0_2
                x1_1 = x1_2
        num_matrices = self._max_diffusion_step + 1  # Adds for x itself.

        x0_0 = x0_0.permute(1, 2, 3, 0)
        x1_0 = x1_0.permute(1, 2, 3, 0)

        x0_0 = torch.reshape(x0_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])
        x1_0 = torch.reshape(x1_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])
        x0_0 = self._gconv_c_0(x0_0)
        x1_0 = self._gconv_c_1(x1_0)

        return torch.reshape(x0_0 + x1_0, [batch_size, self._num_nodes * output_size])


class EncoderModel(nn.Module):
    def __init__(
        self, device, n_dim, n_hid, max_diffusion_step, num_nodes, num_rnn_layers, filter_type
    ):
        super(EncoderModel, self).__init__()
        self.device = device
        self.input_dim = n_dim
        self.rnn_units = n_hid
        self.max_diffusion_step = max_diffusion_step
        self.num_nodes = num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.filter_type = filter_type
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.dcgru_layers = nn.ModuleList(
            [
                DCGRUCell_(
                    self.device,
                    self.rnn_units,
                    self.max_diffusion_step,
                    self.num_nodes,
                    filter_type=self.filter_type,
                )
                for _ in range(self.num_rnn_layers)
            ]
        )

    def forward(self, inputs, adj, hidden_state=None):
        batch_size = inputs.shape[0]
        if hidden_state is None:
            hidden_state = torch.zeros(
                (self.num_rnn_layers, batch_size, self.hidden_state_size)
            ).to(self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)


class Grelen(nn.Module):
    """
    GRELEN Model.
    """

    def __init__(
        self,
        device,
        T,
        target_T,
        Graph_learner_n_hid,
        Graph_learner_n_head_dim,
        Graph_learner_head,
        temperature,
        hard,
        GRU_n_dim,
        max_diffusion_step,
        num_nodes,
        num_rnn_layers,
        filter_type,
        do_prob=0.0,
    ):  # n_in = T
        super(Grelen, self).__init__()
        self.device = device
        self.len_sequence = T
        self.target_T = target_T
        self.graph_learner = Graph_learner(
            T, Graph_learner_n_hid, Graph_learner_n_head_dim, Graph_learner_head, do_prob
        )
        self.linear1 = nn.Linear(1, GRU_n_dim)  # First layer of projection
        nn.init.xavier_normal_(self.linear1.weight.data)
        self.linear1.bias.data.fill_(0.1)

        self.temperature = temperature
        self.hard = hard
        self.GRU_n_dim = GRU_n_dim
        self.num_nodes = num_nodes
        self.head = Graph_learner_head
        self.encoder_model = nn.ModuleList(
            [
                EncoderModel(
                    self.device,
                    GRU_n_dim,
                    GRU_n_dim,
                    max_diffusion_step,
                    num_nodes,
                    num_rnn_layers,
                    filter_type,
                )
                for _ in range(self.head - 1)
            ]
        )
        self.linear_out = nn.Linear(GRU_n_dim, 1)
        nn.init.xavier_normal_(self.linear_out.weight.data)
        self.linear_out.bias.data.fill_(0.1)

    def encoder(self, inputs, adj, head):
        """
        Encoder forward pass
        """
        encoder_hidden_state = None
        encoder_hidden_state_tensor = torch.zeros(inputs.shape).to(self.device)
        for t in range(self.len_sequence):
            _, encoder_hidden_state = self.encoder_model[head](
                inputs[..., t], adj, encoder_hidden_state
            )
            encoder_hidden_state_tensor[..., t] = encoder_hidden_state[-1, ...].reshape(
                -1, self.num_nodes, self.GRU_n_dim
            )
        return encoder_hidden_state_tensor

    def forward(self, inputs):
        B = inputs.shape[0]
        input_projected = self.linear1(inputs.unsqueeze(-1))  # [B, N, T, GRU_n_dim]
        input_projected = input_projected.permute(0, 1, 3, 2)  # [B, N, GRU_n_dim, T]
        probs = self.graph_learner(inputs)  # [B, head, N, N]
        mask_loc = torch.eye(self.num_nodes, dtype=bool).to(self.device)
        probs_reshaped = (
            probs.masked_select(~mask_loc)
            .view(B, self.head, self.num_nodes * (self.num_nodes - 1))
            .to(self.device)
        )
        probs_reshaped = probs_reshaped.permute(0, 2, 1)
        prob = F.softmax(probs_reshaped, -1)
        edges = gumbel_softmax(torch.log(prob + 1e-5), tau=self.temperature, hard=True).to(
            self.device
        )

        adj_list = torch.ones(self.head, B, self.num_nodes, self.num_nodes).to(self.device)
        mask = ~torch.eye(self.num_nodes, dtype=bool).unsqueeze(0).unsqueeze(0).to(self.device)
        mask = mask.repeat(self.head, B, 1, 1).to(self.device)
        adj_list[mask] = edges.permute(2, 0, 1).flatten()
        state_for_output = torch.zeros(input_projected.shape).to(self.device)
        state_for_output = (state_for_output.unsqueeze(0)).repeat(self.head - 1, 1, 1, 1, 1)

        for head in range(self.head - 1):
            state_for_output[head, ...] = self.encoder(
                input_projected, adj_list[head + 1, ...], head
            )

        state_for_output2 = torch.mean(state_for_output, 0).permute(0, 1, 3, 2)
        output = self.linear_out(state_for_output2).squeeze(-1)[..., -1 - self.target_T : -1]

        return prob, output


def build_grelen():
    return Grelen(
        device=device,
        T=6,
        target_T=1,
        Graph_learner_n_hid=8,
        Graph_learner_n_head_dim=4,
        Graph_learner_head=3,
        temperature=0.5,
        hard=True,
        GRU_n_dim=4,
        max_diffusion_step=1,
        num_nodes=5,
        num_rnn_layers=1,
        filter_type="laplacian",
        do_prob=0.0,
    ).to(device)


def example_input_grelen():
    # inputs: [B, N, T]
    return torch.randn(2, 5, 6).to(device)


MENAGERIE_ENTRIES = [
    (
        "GRELEN (Graph Recurrent Encoder for Latent ENtities)",
        build_grelen,
        example_input_grelen,
        2022,
        "SOURCE_AVAILABLE",
    ),
]
