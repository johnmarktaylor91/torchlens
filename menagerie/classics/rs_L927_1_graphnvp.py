# SOURCE: vendored from hlzhang109/PyTorch-GraphNVP @ master
#   (repo: https://github.com/hlzhang109/PyTorch-GraphNVP)
#   graph_nvp/nvp_model.py (GraphNvpModel) + graph_nvp/coupling.py
#   (Coupling, AffineAdjCoupling, AffineNodeFeatureCoupling,
#   AdditiveAdjCoupling, AdditiveNodeFeatureCoupling, Rescale) +
#   graph_nvp/rgcn.py (RGCN, RelationGraphConvolution, GraphAggregation,
#   GraphLinear, Switch) + graph_nvp/mlp.py (MLP) +
#   graph_nvp/hyperparams.py (Hyperparameters).
# GraphNVP (Madhawa, Ishiguro, Nakago & Abe, "GraphNVP: An Invertible Flow
# Model for Generating Molecular Graphs", arXiv:1905.11600, 2019) is the
# first invertible normalizing-flow model for molecular graph generation: two
# stacks of RealNVP-style coupling layers -- adjacency-tensor couplings
# (masking columns of the N x N adjacency, an MLP over the unmasked entries
# produces affine/additive scale+shift) and node-feature couplings (masking
# node rows, an R-GCN over the (masked feature, full adjacency) pair produces
# the scale+shift) -- map a molecular graph (node-feature matrix + relational
# adjacency tensor) to/from a Gaussian latent space with a tractable
# log-determinant-Jacobian, enabling exact likelihood training and one-shot
# generation-by-sampling (via `.reverse()`).
#
# Code copied verbatim from the real repo; only import paths were flattened
# into this single file, and the module-level `args = argparse.parse_args()`
# singleton (utils/argparser.py, imported via `from utils.argparser import
# args` throughout the real coupling.py/rgcn.py/nvp_model.py) was replaced
# with a plain object carrying just the 5 attributes those files actually
# read (`device`, `num_gcn_layer`, `use_switch`, `apply_batch_norm`,
# `additive_transformations`) -- argparse can't run inside a library import,
# so this is a wiring fix, not an architectural change.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Stand-in for the real repo's `utils.argparser.args` (an argparse Namespace
# built at import time via `args = parser.parse_args()`). Only the fields
# actually read by coupling.py / rgcn.py / nvp_model.py are populated here.
# ---------------------------------------------------------------------------
class _Args:
    device = "cpu"
    num_gcn_layer = 3
    use_switch = False
    apply_batch_norm = False
    additive_transformations = True


args = _Args()


# ---------------------------------------------------------------------------
# Verbatim from graph_nvp/hyperparams.py (JSON load/save stripped -- not
# needed for a tiny in-memory construction; the __init__ signature and
# attribute assignments are unchanged)
# ---------------------------------------------------------------------------
class Hyperparameters:
    def __init__(
        self,
        num_nodes=-1,
        num_relations=-1,
        num_features=-1,
        masks=None,
        num_masks=None,
        mask_size=None,
        num_coupling=None,
        batch_norm=False,
        additive_transformations=False,
        learn_dist=True,
        squeeze_adj=False,
        prior_adj_var=1.0,
        prior_x_var=1.0,
        mlp_channels=None,
        gnn_channels=None,
        seed=1,
    ):
        self.gnn_channels = gnn_channels
        self.mlp_channels = mlp_channels
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.num_features = num_features
        self.masks = masks
        self.num_masks = num_masks
        self.mask_size = mask_size
        self.num_coupling = num_coupling
        self.apply_batch_norm = batch_norm
        self.additive_transformations = additive_transformations
        self.learn_dist = learn_dist
        self.squeeze_adj = squeeze_adj
        self.prior_adj_var = prior_adj_var
        self.prior_x_var = prior_x_var
        self.seed = seed


# ---------------------------------------------------------------------------
# Verbatim from graph_nvp/mlp.py
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, units, in_size=None):
        super(MLP, self).__init__()
        assert isinstance(units, (tuple, list))
        assert len(units) >= 1
        n_layers = len(units)

        units_list = [in_size] + list(units)
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(units_list[i], units_list[i + 1]))
        self.layers = nn.Sequential(*layers)
        self.n_layers = n_layers

    def forward(self, x):
        output = self.layers(x)
        return output


# ---------------------------------------------------------------------------
# Verbatim from graph_nvp/rgcn.py
# ---------------------------------------------------------------------------
def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class Switch(nn.Module):
    def __init__(self):
        super(Switch, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class RelationGraphConvolution(nn.Module):
    """Relation GCN layer."""

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
        super(RelationGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.aggregate = aggregate
        if use_relu:
            self.act = nn.ReLU()
        elif args.use_switch:
            self.act = Switch()
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
        self.graph_linear_self = GraphLinear(in_features, out_features)
        self.graph_linear_edge = GraphLinear(in_features, out_features * edge_dim)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x, adj):
        """
        :param x: (batch, N, d)
        :param adj: (batch, E, N, N)
        """
        x = F.dropout(x, p=self.dropout, training=self.training)  # (b, N, d)

        mb, node, ch = x.shape

        hs = self.graph_linear_self(x)
        m = self.graph_linear_edge(x)
        m = m.view(mb, node, self.out_features, self.edge_dim)
        m = m.permute(0, 3, 1, 2)
        hr = torch.matmul(adj, m)
        hr = torch.sum(hr, 1)
        return hs + hr

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GraphAggregation(nn.Module):
    def __init__(self, in_features=128, out_features=64, b_dim=4, dropout=0.0):
        super(GraphAggregation, self).__init__()
        self.sigmoid_linear = nn.Sequential(nn.Linear(in_features, out_features), nn.Sigmoid())
        self.tanh_linear = nn.Sequential(nn.Linear(in_features, out_features), nn.Tanh())
        self.dropout = nn.Dropout(dropout)
        self.switch = Switch()

    def forward(self, input, activation):
        i = self.sigmoid_linear(input)
        j = self.tanh_linear(input)
        output = torch.sum(torch.mul(i, j), 1)
        if args.use_switch:
            output = self.switch(output)
        else:
            output = activation(output) if activation is not None else output
        output = self.dropout(output)
        return output


class GraphLinear(nn.Module):
    """Graph Linear layer. Applies an affine transformation to the third
    axis of a 3-dimensional input, analogous to chainer's GraphLinear."""

    def __init__(self, *argv, **kwargs):
        super(GraphLinear, self).__init__()
        self.linear = nn.Linear(*argv, **kwargs)

    def __call__(self, x):
        s0, s1, s2 = x.size()
        x = x.view(s0 * s1, s2)
        x = self.linear(x)
        x = x.view(s0, s1, -1)
        return x


class RGCN(nn.Module):
    def __init__(
        self,
        nfeat,
        nhid=256,
        nout=128,
        aggout=64,
        edge_dim=3,
        num_layers=3,
        dropout=0.0,
        normalization=False,
    ):
        super(RGCN, self).__init__()

        self.nfeat = nfeat
        self.nhid = nhid
        self.nout = nout
        self.edge_dim = edge_dim
        self.num_layers = num_layers

        self.dropout = dropout
        self.emb = Linear(nfeat, nfeat, bias=False)

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

        self.agg = GraphAggregation(nout, aggout, b_dim=edge_dim, dropout=dropout)
        self.output_layer = nn.Linear(aggout, 1)

    def forward(self, x, adj):
        """
        :param x: (batch, N, d)
        :param adj: (batch, E, N, N)
        """
        x = self.emb(x)
        x = self.gc1(x, adj)
        for i in range(self.num_layers - 2):
            x = self.gc2[i](x, adj)
        x = self.gc3(x, adj)
        x = self.agg(x, torch.tanh)
        return x


# ---------------------------------------------------------------------------
# Verbatim from graph_nvp/coupling.py
# ---------------------------------------------------------------------------
def create_inv_masks(masks):
    inversed_masks = masks.clone()
    inversed_masks[inversed_masks > 0] = 2
    inversed_masks[inversed_masks == 0] = 1
    inversed_masks[inversed_masks == 2] = 0
    return inversed_masks


class Rescale(nn.Module):
    def __init__(self):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.zeros([1]))

    def forward(self, x):
        if torch.isnan(torch.exp(self.weight)).any():
            raise RuntimeError("Rescale factor has NaN entries")
        x = self.weight.exp() * x
        return x


class Coupling(nn.Module):
    def __init__(self, num_nodes, num_relations, num_features, mask, batch_norm=False):
        super(Coupling, self).__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.num_bonds = num_relations
        self.num_features = num_features

        self.adj_size = self.num_nodes * self.num_nodes * self.num_relations
        self.x_size = self.num_nodes * self.num_features
        self.apply_batch_norm = batch_norm
        self.mask = mask.to(args.device)
        self.inversed_mask = create_inv_masks(self.mask).to(args.device)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def reverse(self):
        raise NotImplementedError


class AffineAdjCoupling(Coupling):
    def __init__(
        self,
        num_nodes,
        num_relations,
        num_features,
        mask,
        batch_norm=False,
        num_masked_cols=1,
        ch_list=None,
    ):
        super(AffineAdjCoupling, self).__init__(
            num_nodes, num_relations, num_features, mask, batch_norm=batch_norm
        )
        self.num_masked_cols = num_masked_cols
        self.ch_list = ch_list
        self.adj_size = num_nodes * num_nodes * num_relations
        self.out_size = num_nodes * num_relations
        self.in_size = self.adj_size - self.out_size

        self.mlp = MLP(ch_list, in_size=self.in_size)
        self.lin = nn.Linear(ch_list[-1], 2 * self.out_size)
        self.scale_factor = torch.zeros(1, device=args.device)
        self.batch_norm = nn.BatchNorm1d(self.in_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.rescale = Rescale()

    def forward(self, adj):
        masked_adj = adj[:, :, self.mask > 0].to(args.device)
        log_s, t = self._s_t_functions(masked_adj)
        t = t.expand(adj.shape)
        s = self.sigmoid(log_s + 2)
        s = s.expand(adj.shape)
        log_det_jacobian = torch.sum(torch.log(torch.abs(s)), axis=(1, 2, 3))
        return adj, log_det_jacobian

    def reverse(self, adj):
        masked_adj = adj[:, :, self.mask > 0].to(args.device)
        log_s, t = self._s_t_functions(masked_adj)
        t = t.expand(adj.shape)
        s = self.sigmoid(log_s + 2)
        s = s.expand(adj.shape)
        adj = adj * self.mask + (((adj - t) / s) * self.inversed_mask)
        return adj, None

    def _s_t_functions(self, adj):
        x = adj.view(adj.shape[0], -1).to(args.device)
        if self.apply_batch_norm:
            x = self.batch_norm(x)
        y = self.mlp(x)
        y = self.tanh(y)
        y = self.lin(y)
        y = self.rescale(y)
        s = y[:, : self.out_size]
        t = y[:, self.out_size :]
        s = s.view(y.shape[0], self.num_relations, self.num_nodes, 1).to(args.device)
        t = t.view(y.shape[0], self.num_relations, self.num_nodes, 1).to(args.device)
        return s, t


class AffineNodeFeatureCoupling(Coupling):
    def __init__(
        self,
        num_nodes,
        num_bonds,
        num_features,
        mask,
        batch_norm=False,
        input_type="float",
        num_masked_cols=1,
        ch_list=None,
    ):
        super(AffineNodeFeatureCoupling, self).__init__(
            num_nodes, num_bonds, num_features, mask, batch_norm=batch_norm
        )
        self.num_masked_cols = num_masked_cols
        self.out_size = num_features * num_masked_cols
        self.rgcn = RGCN(
            num_features,
            nhid=128,
            nout=ch_list["hidden"][0],
            edge_dim=self.num_bonds,
            num_layers=args.num_gcn_layer,
            dropout=0.0,
            normalization=False,
        ).to(args.device)
        self.lin1 = nn.Linear(ch_list["hidden"][0], out_features=ch_list["hidden"][1])
        self.lin2 = nn.Linear(ch_list["hidden"][1], out_features=2 * self.out_size)
        self.scale_factor = torch.zeros(1, device=args.device)
        self.batch_norm = nn.BatchNorm1d(ch_list["hidden"][0])
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.rescale = Rescale()

    def forward(self, x, adj):
        masked_x = x * self.mask
        s, t = self._s_t_functions(masked_x, adj)
        x = masked_x + x * (s * self.inversed_mask) + t * self.inversed_mask
        log_det_jacobian = torch.sum(torch.log(torch.abs(s)), axis=(1, 2))
        return x, log_det_jacobian

    def reverse(self, y, adj):
        masked_y = y * self.mask
        s, t = self._s_t_functions(masked_y, adj)
        x = masked_y + (((y - t) / s) * self.inversed_mask)
        return x, None

    def _s_t_functions(self, x, adj):
        h = self.rgcn(x, adj)
        batch_size = x.shape[0]
        if self.apply_batch_norm:
            h = self.batch_norm(h)
        h = self.lin1(h)
        h = self.tanh(h)
        h = self.lin2(h)
        h = self.rescale(h)
        s = h[:, : self.out_size]
        t = h[:, self.out_size :]
        s = self.sigmoid(s + 2)

        t = t.view(batch_size, 1, self.out_size)
        t = t.expand(batch_size, int(self.num_nodes / self.num_masked_cols), self.out_size).to(
            args.device
        )
        s = s.view(batch_size, 1, self.out_size)
        s = s.expand(batch_size, int(self.num_nodes / self.num_masked_cols), self.out_size).to(
            args.device
        )
        return s, t


class AdditiveAdjCoupling(Coupling):
    def __init__(
        self,
        num_nodes,
        num_relations,
        num_features,
        mask,
        batch_norm=False,
        num_masked_cols=1,
        ch_list=None,
    ):
        super(AdditiveAdjCoupling, self).__init__(
            num_nodes, num_relations, num_features, mask, batch_norm=batch_norm
        )
        self.num_masked_cols = num_masked_cols
        self.adj_size = num_nodes * num_nodes * num_relations
        self.out_size = num_nodes * num_relations
        self.in_size = self.adj_size - self.out_size
        self.mlp = MLP(ch_list, in_size=self.in_size)
        self.lin = nn.Linear(ch_list[-1], out_features=self.out_size)
        self.batch_norm = nn.BatchNorm1d(self.in_size)
        self.scale_factor = torch.zeros(1, device=args.device)
        self.tanh = nn.Tanh()
        self.rescale = Rescale()

    def forward(self, adj):
        masked_adj = adj[:, :, self.mask > 0].to(args.device)
        t = self._s_t_functions(masked_adj)
        t = t.expand(adj.shape)
        adj = adj + t * self.inversed_mask
        return adj, torch.zeros(1, device=args.device)

    def reverse(self, adj):
        masked_adj = adj[:, :, self.mask > 0].to(args.device)
        t = self._s_t_functions(masked_adj)
        t = t.expand(adj.shape)
        adj = adj - t * self.inversed_mask
        return adj, None

    def _s_t_functions(self, adj):
        adj = adj.view(adj.shape[0], -1)
        x = adj.clone()
        if self.apply_batch_norm:
            x = self.batch_norm(x)
        y = self.mlp(x)
        y = self.tanh(y)
        y = self.lin(y)
        y = self.rescale(y)
        y = y.view(y.shape[0], self.num_relations, self.num_nodes, 1)
        return y


class AdditiveNodeFeatureCoupling(Coupling):
    def __init__(
        self,
        num_nodes,
        num_bonds,
        num_features,
        mask,
        batch_norm=False,
        ch_list=None,
        input_type="float",
        num_masked_cols=1,
    ):
        super(AdditiveNodeFeatureCoupling, self).__init__(
            num_nodes, num_bonds, num_features, mask, batch_norm=batch_norm
        )
        self.num_masked_cols = num_masked_cols
        self.out_size = num_features * num_masked_cols
        self.rgcn = RGCN(
            num_features,
            nhid=128,
            nout=ch_list["hidden"][0],
            edge_dim=self.num_bonds,
            num_layers=args.num_gcn_layer,
            dropout=0.0,
            normalization=False,
        ).to(args.device)
        self.lin1 = nn.Linear(ch_list["hidden"][0], out_features=ch_list["hidden"][1])
        self.lin2 = nn.Linear(ch_list["hidden"][1], out_features=self.out_size)
        self.scale_factor = torch.zeros(1, device=args.device)
        self.batch_norm = nn.BatchNorm1d(ch_list["hidden"][0])
        self.tanh = nn.Tanh()
        self.rescale = Rescale()

    def forward(self, x, adj):
        masked_x = x * self.mask
        batch_size = x.shape[0]
        t = self._s_t_functions(masked_x, adj)
        t = t.view(batch_size, 1, self.out_size)
        t = t.expand(batch_size, int(self.num_nodes / self.num_masked_cols), self.out_size)
        if self.num_masked_cols > 1:
            t = t.view(batch_size, self.num_nodes, self.num_features)
        x = x + t * self.inversed_mask
        return x, torch.zeros(1, device=args.device)

    def reverse(self, y, adj):
        masked_y = y * self.mask
        batch_size = y.shape[0]
        t = self._s_t_functions(masked_y, adj)
        t = t.view(batch_size, 1, self.out_size)
        t = t.expand(batch_size, int(self.num_nodes / self.num_masked_cols), self.out_size)
        if self.num_masked_cols > 1:
            t = t.view(batch_size, self.num_nodes, self.num_features)
        y = y - t * self.inversed_mask
        return y, None

    def _s_t_functions(self, x, adj):
        h = self.rgcn(x, adj)
        if self.apply_batch_norm:
            h = self.batch_norm(h)
        h = self.lin1(h)
        h = self.tanh(h)
        h = self.lin2(h)
        h = self.rescale(h)
        return h


# ---------------------------------------------------------------------------
# Verbatim from graph_nvp/nvp_model.py (gaussian_nll dropped -- training-loss
# helper, not part of the forward architecture)
# ---------------------------------------------------------------------------
class GraphNvpModel(nn.Module):
    def __init__(self, hyperparams: Hyperparameters):
        super(GraphNvpModel, self).__init__()
        self.hyperparams = hyperparams
        self._init_params(hyperparams)
        self._need_initialization = False
        if self.masks is None:
            self._need_initialization = True
            self.masks = dict()
            self.masks["node"] = self._create_masks("node")
            self.masks["channel"] = self._create_masks("channel")
        self.num_bonds = self.num_relations
        self.num_atoms = self.num_nodes
        assert self.num_bonds + 1 == self.num_features
        self.adj_size = self.num_atoms * self.num_atoms * self.num_relations
        self.x_size = self.num_atoms * self.num_features
        self.prior_ln_var = nn.Parameter(torch.zeros([1]))
        nn.init.constant_(self.prior_ln_var, 1e-5)
        self.constant_pi = torch.tensor([3.1415926535], device=args.device, dtype=torch.float32)
        # AffineNodeFeatureCoupling found to be unstable.
        channel_coupling = AffineNodeFeatureCoupling
        node_coupling = AffineAdjCoupling
        if self.additive_transformations:
            channel_coupling = AdditiveNodeFeatureCoupling
            node_coupling = AdditiveAdjCoupling
        transforms = []
        for i in range(self.num_coupling["channel"]):
            transforms += [
                channel_coupling(
                    self.num_nodes,
                    self.num_relations,
                    self.num_features,
                    self.masks["channel"][i % self.num_masks["channel"]],
                    num_masked_cols=int(self.num_nodes / self.num_masks["channel"]),
                    ch_list=self.gnn_channels,
                    batch_norm=self.apply_batch_norm,
                )
            ]
        for i in range(self.num_coupling["node"]):
            transforms += [
                node_coupling(
                    self.num_nodes,
                    self.num_relations,
                    self.num_features,
                    self.masks["node"][i % self.num_masks["node"]],
                    num_masked_cols=int(self.num_nodes / self.num_masks["channel"]),
                    batch_norm=self.apply_batch_norm,
                    ch_list=self.mlp_channels,
                )
            ]
        self.transforms = nn.ModuleList(transforms)

    def forward(self, adj, x):
        h = x.clone()
        sum_log_det_jacs_x = torch.zeros(h.shape[0], device=args.device, requires_grad=True)
        sum_log_det_jacs_adj = torch.zeros(h.shape[0], device=args.device, requires_grad=True)
        # forward step of channel-coupling layers
        for i in range(self.num_coupling["channel"]):
            h, log_det_jacobians = self.transforms[i](h, adj)
            sum_log_det_jacs_x = sum_log_det_jacs_x + log_det_jacobians
        for i in range(self.num_coupling["channel"], len(self.transforms)):
            adj, log_det_jacobians = self.transforms[i](adj)
            sum_log_det_jacs_adj = sum_log_det_jacs_adj + log_det_jacobians

        adj = adj.view(adj.shape[0], -1)
        h = h.view(h.shape[0], -1)
        out = [h, adj]
        return out, [sum_log_det_jacs_x, sum_log_det_jacs_adj]

    def reverse(self, z, x_size, true_adj=None):
        batch_size = z.shape[0]
        z_x = z[:, :x_size]
        z_adj = z[:, x_size:]
        if true_adj is None:
            h_adj = z_adj.view(batch_size, self.num_relations, self.num_nodes, self.num_nodes)
            for i in reversed(range(self.num_coupling["channel"], len(self.transforms))):
                h_adj, log_det_jacobians = self.transforms[i].reverse(h_adj)
            adj = h_adj
            adj = adj + adj.permute(0, 1, 3, 2)
            adj = adj / 2.0
            adj = F.softmax(adj, dim=1)
        else:
            adj = true_adj

        h_x = z_x.view(batch_size, self.num_nodes, self.num_features)
        for i in reversed(range(self.num_coupling["channel"])):
            h_x, log_det_jacobians = self.transforms[i].reverse(h_x, adj)
        return adj, h_x

    def _init_params(self, hyperparams):
        self.num_nodes = hyperparams.num_nodes
        self.num_relations = hyperparams.num_relations
        self.num_features = hyperparams.num_features
        self.masks = hyperparams.masks

        self.apply_batch_norm = args.apply_batch_norm
        self.additive_transformations = args.additive_transformations

        self.num_masks = hyperparams.num_masks
        self.num_coupling = hyperparams.num_coupling
        self.mask_size = hyperparams.mask_size
        self.mlp_channels = hyperparams.mlp_channels
        self.gnn_channels = hyperparams.gnn_channels

    def _create_masks(self, type):
        masks = []
        num_cols = int(self.num_nodes / self.hyperparams.num_masks[type])
        if type == "node":
            for i in range(self.hyperparams.num_masks[type]):
                node_mask = torch.ones([self.num_nodes, self.num_nodes])
                for j in range(num_cols):
                    node_mask[:, i + j] = 0.0
                masks.append(node_mask)
        elif type == "channel":
            num_cols = int(self.num_nodes / self.hyperparams.num_masks[type])
            for i in range(self.hyperparams.num_masks[type]):
                ch_mask = torch.ones([self.num_nodes, self.num_features])
                for j in range(num_cols):
                    ch_mask[i * num_cols + j, :] = 0.0
                masks.append(ch_mask)
        return masks


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------

# Tiny QM9-scale hyperparameters: 9 atoms, 4 bond types + "no bond" = 5
# features (num_bonds + 1 == num_features, asserted in the real model), a
# handful of coupling layers with small MLP/GNN channel widths.
#
# num_masks['channel'] == num_nodes so num_masked_cols = num_nodes //
# num_masks['channel'] == 1: the real repo's `AdditiveNodeFeatureCoupling`
# only reaches its `t.view(batch_size, self.num_nodes, self.num_features)`
# reshape when `num_masked_cols > 1`, and that reshape then always fails
# (`.expand()` leaves a broadcast/stride-0 dim, which `.view()` -- as
# opposed to `.reshape()` -- rejects regardless of whether the target shape
# element-count matches) -- reproducible standalone even with the paper's
# own num_atoms=9/num_channel_masks=4 defaults, so this is a latent bug in
# the upstream repo's `main` branch, not an artifact of this vendoring.
# num_masks['channel']=num_nodes keeps num_masked_cols==1, which skips that
# branch and exercises the same real code along its only currently-working
# path. num_masks['node']=num_nodes matches the real repo's own QM9 default
# (`num_node_masks=9` for `num_atoms=9`, see utils/argparser.py) so
# `AffineAdjCoupling`/`AdditiveAdjCoupling`'s `in_size = adj_size - out_size`
# assumption (exactly one node's worth of relation-entries masked per
# coupling layer) holds.
_NUM_NODES = 9
_NUM_RELATIONS = 4
_NUM_FEATURES = _NUM_RELATIONS + 1


def build_graphnvp():
    args.device = "cpu"
    args.additive_transformations = True
    args.apply_batch_norm = False
    args.num_gcn_layer = 3
    hyperparams = Hyperparameters(
        num_nodes=_NUM_NODES,
        num_relations=_NUM_RELATIONS,
        num_features=_NUM_FEATURES,
        masks=None,
        num_masks={"node": _NUM_NODES, "channel": _NUM_NODES},
        mask_size=None,
        num_coupling={"channel": 2, "node": 2},
        batch_norm=False,
        additive_transformations=True,
        mlp_channels=[32, 32],
        # gnn_channels['hidden'][0] must match RGCN's `aggout` (default 64,
        # unrelated to the `nout` passed at construction) -- see the real
        # repo's train_mle.py defaults (`gnn_channels = {'hidden': [64,
        # 128]}`), which this mirrors at a smaller width.
        gnn_channels={"hidden": [64, 32]},
    )
    return GraphNvpModel(hyperparams)


def example_input_graphnvp():
    rng = torch.Generator().manual_seed(0)
    batch_size = 2
    x = torch.randn(batch_size, _NUM_NODES, _NUM_FEATURES, generator=rng)
    adj = torch.rand(batch_size, _NUM_RELATIONS, _NUM_NODES, _NUM_NODES, generator=rng)
    return (adj, x)


MENAGERIE_ENTRIES = [
    (
        "GraphNVP",
        build_graphnvp,
        example_input_graphnvp,
        2019,
        "CODE",
    ),
]
