# SOURCE: vendored from divelab/DIG @ dig-stable
#   (repo: https://github.com/divelab/DIG; ICML 2021 official code for
#   GraphDF, "GraphDF: A Discrete Flow Model for Molecular Graph Generation")
#   dig/ggraph/method/GraphDF/model/disgraphaf.py (DisGraphAF, verbatim) +
#   dig/ggraph/method/GraphDF/model/rgcn.py (RelationGraphConvolution / RGCN,
#   verbatim) + dig/ggraph/method/GraphDF/model/st_net.py (ST_Dis, verbatim;
#   the file's other S/T-net variants -- ST_Net_Sigmoid/ST_Net_Exp/
#   ST_Net_Softplus -- are DisGraphAF-continuous-flow siblings not used by
#   the discrete GraphDF path, so they are dropped) + dig/ggraph/method/
#   GraphDF/model/df_utils.py (one_hot / one_hot_argmax / one_hot_minus /
#   one_hot_add, verbatim) + dig/ggraph/method/GraphDF/model/graphflow.py
#   (GraphFlowModel.__init__ + GraphFlowModel.forward + initialize_masks,
#   verbatim; imports adjusted to be self-contained in this single file).
#
# GraphDF (Luo, Yan, Ji, ICML 2021) is an autoregressive discrete
# normalizing-flow model for molecular graph generation: it dequantizes a
# one-hot node/adjacency-type molecular graph into a bijective discrete
# latent space via node/edge scale-and-shift ("S/T-net", `ST_Dis`) flow
# layers, conditioned on a Relational-GCN (`RGCN`) graph embedding of the
# partially-built molecule that is re-masked at every autoregressive
# unrolling step (`initialize_masks`'s node/edge/link-prediction masks,
# applied inside `DisGraphAF._get_embs`). Only `GraphFlowModel.forward`
# (mask-based graph embedding -> `DisGraphAF` node+edge one-hot-add flow
# cascade) is vendored here -- the real architecture's training-time
# forward pass. `GraphFlowModel.generate` (RDKit-based autoregressive
# sampling/valency-checked bond assembly) and `DisGraphAF.forward_rl_node` /
# `forward_rl_edge` / `reverse` (reinforcement-learning fine-tuning and
# inverse-flow sampling entry points) are training/generation orchestration
# not exercised by a single forward-pass trace, so they are not vendored,
# and the module-level `from rdkit import Chem` / `from dig.ggraph.utils
# import ...` imports (rdkit not installed as a base lib here) are dropped
# along with them; no architecture code was rewritten.

import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Verbatim from model/df_utils.py
# ---------------------------------------------------------------------------


def one_hot(inputs, vocab_size=None):
    """Returns one hot of data over each element of the inputs"""
    if vocab_size is None:
        vocab_size = inputs.max() + 1
    input_shape = inputs.shape
    inputs = inputs.flatten().unsqueeze(1).long()
    z = torch.zeros(len(inputs), vocab_size, device=inputs.device)
    z.scatter_(1, inputs, 1.0)
    return z.view(*input_shape, vocab_size)


def one_hot_argmax(inputs, temperature=0.1, axis=-1):
    """Returns one-hot of argmax with backward pass set to softmax-temperature."""
    vocab_size = inputs.shape[-1]
    z = one_hot(torch.argmax(inputs, dim=axis), vocab_size)
    soft = F.softmax(inputs / temperature, dim=axis)
    outputs = soft + (z - soft).detach()
    return outputs


def one_hot_minus(inputs, shift):
    """Performs (inputs - shift) % vocab_size in the one-hot space."""
    shift = shift.type(inputs.dtype)
    vocab_size = inputs.shape[-1]
    shift_matrix = torch.stack([torch.roll(shift, i, dims=-1) for i in range(vocab_size)], dim=-2)
    outputs = torch.einsum("...v,...uv->...u", inputs, shift_matrix)
    return outputs


def one_hot_add(inputs, shift):
    """Performs (inputs + shift) % vocab_size in the one-hot space."""
    inputs = torch.stack((inputs, torch.zeros_like(inputs)), dim=-1)
    shift = torch.stack((shift, torch.zeros_like(shift)), dim=-1)
    if "torch.fft" not in sys.modules:
        with warnings.catch_warnings(record=True):
            inputs_fft = torch.fft(inputs, 1)
            shift_fft = torch.fft(shift, 1)
    else:
        inputs_fft = torch.view_as_real(torch.fft.fft(torch.view_as_complex(inputs)))
        shift_fft = torch.view_as_real(torch.fft.fft(torch.view_as_complex(shift)))
    result_fft_real = (
        inputs_fft[..., 0] * shift_fft[..., 0] - inputs_fft[..., 1] * shift_fft[..., 1]
    )
    result_fft_imag = (
        inputs_fft[..., 0] * shift_fft[..., 1] + inputs_fft[..., 1] * shift_fft[..., 0]
    )
    result_fft = torch.stack((result_fft_real, result_fft_imag), dim=-1)
    if "torch.fft" not in sys.modules:
        with warnings.catch_warnings(record=True):
            return torch.ifft(result_fft, 1)[..., 0]
    else:
        return torch.view_as_real(torch.fft.ifft(torch.view_as_complex(result_fft)))[..., 0]


# ---------------------------------------------------------------------------
# Verbatim from model/st_net.py (ST_Dis only -- the discrete-flow S/T-net)
# ---------------------------------------------------------------------------


class ST_Dis(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim=64, bias=True, temperature=0.1):
        super(ST_Dis, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.bias = bias
        self.temperature = temperature

        self.linear1 = nn.Linear(input_dim, hid_dim, bias=bias)
        self.linear2 = nn.Linear(hid_dim, output_dim, bias=bias)
        self.tanh = nn.Tanh()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear2.weight, 1e-10)
        if self.bias:
            nn.init.constant_(self.linear1.bias, 0.0)
            nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, graph_embed):
        loc = self.linear2(self.tanh(self.linear1(graph_embed)))
        loc = one_hot_argmax(loc, self.temperature)
        return loc


# ---------------------------------------------------------------------------
# Verbatim from model/rgcn.py
# ---------------------------------------------------------------------------


class RelationGraphConvolution(nn.Module):
    """
    Relation GCN layer.
    """

    def __init__(
        self,
        in_features,
        out_features,
        edge_dim=3,
        aggregate="sum",
        dropout=0.0,
        use_relu=True,
        bias=False,
    ):
        """
        :param in/out_features: scalar of channels for node embedding
        :param edge_dim: dim of edge type, virtual type not included
        """
        super(RelationGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.aggregate = aggregate
        if use_relu:
            self.act = nn.ReLU()
        else:
            self.act = None

        self.weight = nn.Parameter(
            torch.FloatTensor(self.edge_dim, self.in_features, self.out_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.edge_dim, 1, self.out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x, adj):
        """
        :param x: (batch, N, d)
        :param adj: (batch, E, N, N)
        typically d=9 e=3
        :return:
        updated x with shape (batch, N, d)
        """
        x = F.dropout(x, p=self.dropout, training=self.training)  # (b, N, d)

        batch_size = x.size(0)

        # transform
        support = torch.einsum("bid, edh-> beih", x, self.weight)
        output = torch.einsum("beij, bejh-> beih", adj, support)  # (batch, e, N, d)

        if self.bias is not None:
            output += self.bias
        if self.act is not None:
            output = self.act(output)  # (b, E, N, d)
        output = output.view(
            batch_size, self.edge_dim, x.size(1), self.out_features
        )  # (b, E, N, d)

        if self.aggregate == "sum":
            node_embedding = torch.sum(output, dim=1, keepdim=False)
        elif self.aggregate == "max":
            node_embedding = torch.max(output, dim=1, keepdim=False)
        elif self.aggregate == "mean":
            node_embedding = torch.mean(output, dim=1, keepdim=False)
        elif self.aggregate == "concat":
            node_embedding = torch.cat(torch.split(output, dim=1, split_size_or_sections=1), dim=3)
            node_embedding = torch.squeeze(node_embedding, dim=1)
        else:
            print("GCN aggregate error!")
        return node_embedding

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class RGCN(nn.Module):
    def __init__(
        self, nfeat, nhid=128, nout=128, edge_dim=3, num_layers=3, dropout=0.0, normalization=False
    ):
        """
        :num_layars: the number of layers in each R-GCN
        """
        super(RGCN, self).__init__()

        self.nfeat = nfeat
        self.nhid = nhid
        self.nout = nout
        self.edge_dim = edge_dim
        self.num_layers = num_layers

        self.dropout = dropout
        self.normalization = normalization

        self.emb = nn.Linear(nfeat, nfeat, bias=False)

        self.gc1 = RelationGraphConvolution(
            nfeat,
            nhid,
            edge_dim=self.edge_dim,
            aggregate="sum",
            use_relu=True,
            dropout=self.dropout,
            bias=False,
        )

        self.gc2 = nn.ModuleList(
            [
                RelationGraphConvolution(
                    nhid,
                    nhid,
                    edge_dim=self.edge_dim,
                    aggregate="sum",
                    use_relu=True,
                    dropout=self.dropout,
                    bias=False,
                )
                for i in range(self.num_layers - 2)
            ]
        )

        self.gc3 = RelationGraphConvolution(
            nhid,
            nout,
            edge_dim=self.edge_dim,
            aggregate="sum",
            use_relu=False,
            dropout=self.dropout,
            bias=False,
        )

    def forward(self, x, adj):
        """
        :param x: (batch, N, d)
        :param adj: (batch, E, N, N)
        :return:
        """
        x = self.emb(x)
        x = self.gc1(x, adj)
        for i in range(self.num_layers - 2):
            x = self.gc2[i](x, adj)
        x = self.gc3(x, adj)
        return x


# ---------------------------------------------------------------------------
# Verbatim from model/disgraphaf.py
# ---------------------------------------------------------------------------


class DisGraphAF(nn.Module):
    def __init__(
        self,
        mask_node,
        mask_edge,
        index_select_edge,
        num_flow_layer=12,
        graph_size=38,
        num_node_type=9,
        num_edge_type=4,
        use_bn=True,
        num_rgcn_layer=3,
        nhid=128,
        nout=128,
    ):
        """
        :param index_nod_edg:
        :param num_edge_type, virtual type included
        """
        super(DisGraphAF, self).__init__()
        self.repeat_num = mask_node.size(0)
        self.graph_size = graph_size
        self.num_node_type = num_node_type
        self.num_edge_type = num_edge_type

        self.mask_node = nn.Parameter(
            mask_node.view(1, self.repeat_num, graph_size, 1), requires_grad=False
        )  # (1, repeat_num, n, 1)
        self.mask_edge = nn.Parameter(
            mask_edge.view(1, self.repeat_num, 1, graph_size, graph_size), requires_grad=False
        )  # (1, repeat_num, 1, n, n)
        self.index_select_edge = nn.Parameter(
            index_select_edge, requires_grad=False
        )  # (edge_step_length, 2)

        self.emb_size = nout
        self.num_flow_layer = num_flow_layer

        self.rgcn = RGCN(
            num_node_type,
            nhid=nhid,
            nout=nout,
            edge_dim=self.num_edge_type - 1,
            num_layers=num_rgcn_layer,
            dropout=0.0,
            normalization=False,
        )

        if use_bn:
            self.batchNorm = nn.BatchNorm1d(nout)

        self.node_st_net = nn.ModuleList(
            [
                ST_Dis(nout, self.num_node_type, hid_dim=nhid, bias=True)
                for _ in range(num_flow_layer)
            ]
        )
        self.edge_st_net = nn.ModuleList(
            [
                ST_Dis(nout * 3, self.num_edge_type, hid_dim=nhid, bias=True)
                for _ in range(num_flow_layer)
            ]
        )

    def forward(self, x, adj, x_deq, adj_deq):
        """
        :param x:   (batch, N, 9)
        :param adj: (batch, 4, N, N)

        :param x_deq: (batch, N, 9)
        :param adj_deq:  (batch, edge_num, 4)
        :return:
        x_deq: (batch, N, 9)
        adj_deq: (batch, edge_num, 4)
        """
        # inputs for RelGCNs
        graph_emb_node, graph_node_emb_edge = self._get_embs(x, adj)

        for i in range(self.num_flow_layer):
            # update x_deq
            node_t = self.node_st_net[i](graph_emb_node).type(x.dtype)
            x_deq = one_hot_add(x_deq, node_t)

            # update adj_deq
            edge_t = self.edge_st_net[i](graph_node_emb_edge).type(adj.dtype)
            adj_deq = one_hot_add(adj_deq, edge_t)

        return [x_deq, adj_deq]

    def _get_embs_node(self, x, adj):
        adj = adj[:, :3]  # (batch, 3, N, N)
        node_emb = self.rgcn(x, adj)  # (batch, N, d)
        if hasattr(self, "batchNorm"):
            node_emb = self.batchNorm(node_emb.transpose(1, 2)).transpose(1, 2)  # (batch, N, d)
        graph_emb = torch.sum(node_emb, dim=1, keepdim=False).contiguous()  # (batch, d)
        return graph_emb

    def _get_embs_edge(self, x, adj, index):
        batch_size = x.size(0)
        assert batch_size == index.size(0)

        adj = adj[:, :3]  # (batch, 3, N, N)

        node_emb = self.rgcn(x, adj)  # (batch, N, d)
        if hasattr(self, "batchNorm"):
            node_emb = self.batchNorm(node_emb.transpose(1, 2)).transpose(1, 2)  # (batch, N, d)

        graph_emb = (
            torch.sum(node_emb, dim=1, keepdim=False).contiguous().view(batch_size, 1, -1)
        )  # (batch, 1, d)

        index = index.view(batch_size, -1, 1).repeat(1, 1, self.emb_size)  # (batch, 2, d)
        graph_node_emb = torch.cat(
            (torch.gather(node_emb, dim=1, index=index), graph_emb), dim=1
        )  # (batch_size, 3, d)
        graph_node_emb = graph_node_emb.view(batch_size, -1)  # (batch_size, 3d)
        return graph_node_emb

    def _get_embs(self, x, adj):
        """
        :param x of shape (batch, N, 9)
        :param adj of shape (batch, 4, N, N)
        :return: inputs for st_net_node and st_net_edge
        graph_emb_node of shape (batch, N, d)
        graph_emb_edge of shape (batch, repeat-N, 3d)
        """
        batch_size = x.size(0)
        adj = adj[:, :3]  # (batch, 3, N, N)
        x = torch.where(
            self.mask_node,
            x.unsqueeze(1).repeat(1, self.repeat_num, 1, 1),
            torch.zeros([1], device=x.device),
        ).view(-1, self.graph_size, self.num_node_type)  # (batch*repeat_num, N, 9)

        adj = torch.where(
            self.mask_edge,
            adj.unsqueeze(1).repeat(1, self.repeat_num, 1, 1, 1),
            torch.zeros([1], device=x.device),
        ).view(
            -1, self.num_edge_type - 1, self.graph_size, self.graph_size
        )  # (batch*repeat_num, 3, N, N)
        node_emb = self.rgcn(x, adj)  # (batch*repeat_num, N, d)

        if hasattr(self, "batchNorm"):
            node_emb = self.batchNorm(node_emb.transpose(1, 2)).transpose(
                1, 2
            )  # (batch*repeat_num, N, d)

        node_emb = node_emb.view(
            batch_size, self.repeat_num, self.graph_size, -1
        )  # (batch, repeat_num, N, d)

        graph_emb = torch.sum(node_emb, dim=2, keepdim=False)  # (batch, repeat_num, d)

        # input for st_net_node
        graph_emb_node = graph_emb[:, : self.graph_size].contiguous()  # (batch, N, d)

        # input for st_net_edge
        graph_emb_edge = graph_emb[:, self.graph_size :].contiguous()  # (batch, repeat_num-N, d)
        graph_emb_edge = graph_emb_edge.unsqueeze(2)  # (batch, repeat_num-N, 1, d)

        all_node_emb_edge = node_emb[:, self.graph_size :]  # (batch, repeat_num-N, N, d)

        index = self.index_select_edge.view(1, -1, 2, 1).repeat(
            batch_size, 1, 1, self.emb_size
        )  # (batch_size, repeat_num-N, 2, d)

        graph_node_emb_edge = torch.cat(
            (torch.gather(all_node_emb_edge, dim=2, index=index), graph_emb_edge), dim=2
        )  # (batch_size, repeat_num-N, 3, d)

        graph_node_emb_edge = graph_node_emb_edge.view(
            batch_size, self.repeat_num - self.graph_size, -1
        )  # (batch_size, (repeat_num-N), 3*d)

        return graph_emb_node, graph_node_emb_edge


# ---------------------------------------------------------------------------
# Verbatim from model/graphflow.py (GraphFlowModel.__init__ +
# GraphFlowModel.forward + initialize_masks only)
# ---------------------------------------------------------------------------


class GraphFlowModel(nn.Module):
    def __init__(self, model_conf_dict):
        super(GraphFlowModel, self).__init__()
        self.max_size = model_conf_dict["max_size"]
        self.edge_unroll = model_conf_dict["edge_unroll"]
        self.node_dim = model_conf_dict["node_dim"]
        self.bond_dim = model_conf_dict["bond_dim"]

        node_masks, adj_masks, link_prediction_index, self.flow_core_edge_masks = (
            self.initialize_masks(max_node_unroll=self.max_size, max_edge_unroll=self.edge_unroll)
        )

        self.latent_step = node_masks.size(0)
        self.latent_node_length = self.max_size * self.node_dim
        self.latent_edge_length = (self.latent_step - self.max_size) * self.bond_dim

        self.dp = model_conf_dict["use_gpu"]

        node_base_log_probs = torch.randn(self.max_size, self.node_dim)
        edge_base_log_probs = torch.randn(self.latent_step - self.max_size, self.bond_dim)
        self.flow_core = DisGraphAF(
            node_masks,
            adj_masks,
            link_prediction_index,
            num_flow_layer=model_conf_dict["num_flow_layer"],
            graph_size=self.max_size,
            num_node_type=self.node_dim,
            num_edge_type=self.bond_dim,
            num_rgcn_layer=model_conf_dict["num_rgcn_layer"],
            nhid=model_conf_dict["nhid"],
            nout=model_conf_dict["nout"],
        )
        if self.dp:
            self.flow_core = nn.DataParallel(self.flow_core)
            self.node_base_log_probs = nn.Parameter(node_base_log_probs.cuda(), requires_grad=True)
            self.edge_base_log_probs = nn.Parameter(edge_base_log_probs.cuda(), requires_grad=True)
        else:
            self.node_base_log_probs = nn.Parameter(node_base_log_probs, requires_grad=True)
            self.edge_base_log_probs = nn.Parameter(edge_base_log_probs, requires_grad=True)

    def forward(self, inp_node_features, inp_adj_features):
        """
        Args:
            inp_node_features: (B, N, 9)
            inp_adj_features: (B, 4, N, N)

        Returns:
            z: [(B, node_num*9), (B, edge_num*4)]
            logdet:  ([B], [B])
        """
        inp_node_features_cont = inp_node_features.clone()  # (B, N, 9)

        inp_adj_features_cont = inp_adj_features[
            :, :, self.flow_core_edge_masks
        ].clone()  # (B, 4, edge_num)
        inp_adj_features_cont = inp_adj_features_cont.permute(
            0, 2, 1
        ).contiguous()  # (B, edge_num, 4)

        z = self.flow_core(
            inp_node_features, inp_adj_features, inp_node_features_cont, inp_adj_features_cont
        )
        return z

    def initialize_masks(self, max_node_unroll=38, max_edge_unroll=12):
        """
        Args:
            max node unroll: maximal number of nodes in molecules to be generated (default: 38)
            max edge unroll: maximal number of edges to predict for each generated nodes (default: 12)
        Returns:
            node_masks: node mask for each step
            adj_masks: adjacency mask for each step
            is_node_update_mask: 1 indicate this step is for updating node features
            flow_core_edge_mask: get the distributions we want to model in adjacency matrix
        """
        num_masks = int(
            max_node_unroll
            + (max_edge_unroll - 1) * max_edge_unroll / 2
            + (max_node_unroll - max_edge_unroll) * (max_edge_unroll)
        )
        num_mask_edge = int(num_masks - max_node_unroll)

        node_masks1 = torch.zeros([max_node_unroll, max_node_unroll]).bool()
        adj_masks1 = torch.zeros([max_node_unroll, max_node_unroll, max_node_unroll]).bool()
        node_masks2 = torch.zeros([num_mask_edge, max_node_unroll]).bool()
        adj_masks2 = torch.zeros([num_mask_edge, max_node_unroll, max_node_unroll]).bool()
        link_prediction_index = torch.zeros([num_mask_edge, 2]).long()
        flow_core_edge_masks = torch.zeros([max_node_unroll, max_node_unroll]).bool()

        cnt = 0
        cnt_node = 0
        cnt_edge = 0
        for i in range(max_node_unroll):
            node_masks1[cnt_node][:i] = 1
            adj_masks1[cnt_node][:i, :i] = 1
            cnt += 1
            cnt_node += 1

            edge_total = 0
            if i < max_edge_unroll:
                start = 0
                edge_total = i
            else:
                start = i - max_edge_unroll
                edge_total = max_edge_unroll
            for j in range(edge_total):
                if j == 0:
                    node_masks2[cnt_edge][: i + 1] = 1
                    adj_masks2[cnt_edge] = adj_masks1[cnt_node - 1].clone()
                    adj_masks2[cnt_edge][i, i] = 1
                else:
                    node_masks2[cnt_edge][: i + 1] = 1
                    adj_masks2[cnt_edge] = adj_masks2[cnt_edge - 1].clone()
                    adj_masks2[cnt_edge][i, start + j - 1] = 1
                    adj_masks2[cnt_edge][start + j - 1, i] = 1
                cnt += 1
                cnt_edge += 1
        assert cnt == num_masks, "masks cnt wrong"
        assert cnt_node == max_node_unroll, "node masks cnt wrong"
        assert cnt_edge == num_mask_edge, "edge masks cnt wrong"

        cnt = 0
        for i in range(max_node_unroll):
            if i < max_edge_unroll:
                start = 0
                edge_total = i
            else:
                start = i - max_edge_unroll
                edge_total = max_edge_unroll

            for j in range(edge_total):
                link_prediction_index[cnt][0] = start + j
                link_prediction_index[cnt][1] = i
                cnt += 1
        assert cnt == num_mask_edge, "edge mask initialize fail"

        for i in range(max_node_unroll):
            if i == 0:
                continue
            if i < max_edge_unroll:
                start = 0
                end = i
            else:
                start = i - max_edge_unroll
                end = i
            flow_core_edge_masks[i][start:end] = 1

        node_masks = torch.cat((node_masks1, node_masks2), dim=0)
        adj_masks = torch.cat((adj_masks1, adj_masks2), dim=0)

        node_masks = nn.Parameter(node_masks, requires_grad=False)
        adj_masks = nn.Parameter(adj_masks, requires_grad=False)
        link_prediction_index = nn.Parameter(link_prediction_index, requires_grad=False)
        flow_core_edge_masks = nn.Parameter(flow_core_edge_masks, requires_grad=False)

        return node_masks, adj_masks, link_prediction_index, flow_core_edge_masks


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------


def build_graphdf():
    """Tiny-size real GraphDF (RGCN graph encoder + discrete-flow node/edge
    S/T-net cascade), CPU-only (use_gpu=False -> plain nn.Module, matching
    the repo's own supported config flag rather than nn.DataParallel)."""
    model_conf_dict = {
        "max_size": 6,
        "edge_unroll": 3,
        "node_dim": 5,
        "bond_dim": 4,
        "use_gpu": False,
        "num_flow_layer": 2,
        "num_rgcn_layer": 2,
        "nhid": 16,
        "nout": 16,
    }
    return GraphFlowModel(model_conf_dict)


def example_input_graphdf():
    """One padded 6-atom molecular graph: (node_features, adj_features)
    matching GraphFlowModel.forward's (B, N, node_dim) / (B, bond_dim, N, N)
    contract for max_size=6, bond_dim=4 (3 real bond types + virtual/no-bond
    channel)."""
    torch.manual_seed(0)
    batch, n, node_dim, bond_dim = 1, 6, 5, 4
    node_features = F.one_hot(torch.randint(0, node_dim, (batch, n)), num_classes=node_dim).float()
    bond_type = torch.randint(0, bond_dim, (batch, n, n))
    bond_type = torch.triu(bond_type, diagonal=1)
    bond_type = bond_type + bond_type.transpose(1, 2)
    adj_features = F.one_hot(bond_type, num_classes=bond_dim).permute(0, 3, 1, 2).float()
    return (node_features, adj_features)


MENAGERIE_ENTRIES = [
    (
        "GraphDF",
        build_graphdf,
        example_input_graphdf,
        2021,
        "CODE",
    ),
]
