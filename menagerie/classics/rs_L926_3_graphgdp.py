# SOURCE: vendored from GRAPH-0/GraphGDP @ main
#   (repo: https://github.com/GRAPH-0/GraphGDP; ICDM 2022 official code for
#   GraphGDP, "GraphGDP: Generative Diffusion Processes for Permutation
#   Invariant Graph Generation")
#   models/pgsn.py (PGSN, verbatim) + models/layers.py (get_act / conv1x1 /
#   get_timestep_embedding, verbatim) + models/gnns.py (pos_gnn, verbatim) +
#   models/trans_layers.py (PosTransLayer, verbatim) + models/utils.py
#   (mask_adj2node / get_rw_feat, verbatim; the `register_model` decorator
#   registry and the module-level `import sde_lib` -- the repo's own
#   training-time SDE/score-function module, not needed by a plain forward
#   pass -- are dropped), copied verbatim (imports only adjusted to be
#   self-contained in this single file; `ml_collections.ConfigDict` is not
#   an installed base lib here, so the config object passed to `PGSN` is a
#   plain nested `types.SimpleNamespace` with the identical dotted-attribute
#   shape the repo's own `configs/vp_ego_small_pgsn.py` builds -- PGSN only
#   ever does attribute reads on `config`, never anything ConfigDict-
#   specific, so this is a config-container substitution, not an
#   architecture change).
#
# GraphGDP (Huang, Sun, Xie, Cao, ICDM 2022) is a score-based / diffusion
# graph generative model: a "position enhanced graph score network" (PGSN)
# predicts the denoising score of a noised graph adjacency matrix at a
# given diffusion timestep. It embeds the (continuous) timestep via
# sinusoidal positional embeddings, derives per-node degree and k-step
# random-walk positional-encoding (RWPE) / shortest-path-distance (SPD)
# features from a discretized version of the noisy dense adjacency matrix,
# projects the dense edge features (raw + SPD) with 1x1 convolutions, then
# runs a stack of permutation-equivariant `PosTransLayer` graph-transformer
# message-passing layers (attention over node+position features, edge-
# feature-gated) to jointly refine node and edge representations, finally
# projecting back to a symmetric per-edge score-map matching the input
# adjacency shape. Only `PGSN.forward` is vendored here -- the model's
# entire architecture (score prediction has no separate "generation"
# method; sampling lives in the repo's `sampling.py` reverse-SDE solver,
# which is training/inference orchestration around this same forward call,
# not exercised by a single forward-pass trace); no architecture code was
# rewritten.

import functools
import math
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptTensor
from torch_geometric.utils import dense_to_sparse, softmax
from torch_scatter import scatter

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Verbatim from models/utils.py (register_model / mask_adj2node / get_rw_feat)
# ---------------------------------------------------------------------------

_MODELS = {}


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


@torch.no_grad()
def mask_adj2node(adj_mask):
    """Convert batched adjacency mask matrices to batched node mask matrices.

    Args:
        adj_mask: [B, N, N] Batched adjacency mask matrices without self-loop edge.

    Output:
        node_mask: [B, N] Batched node mask matrices indicating the valid nodes.
    """
    batch_size, max_num_nodes, _ = adj_mask.shape

    node_mask = adj_mask[:, 0, :].clone()
    node_mask[:, 0] = 1

    return node_mask


@torch.no_grad()
def get_rw_feat(k_step, dense_adj):
    """Compute k_step Random Walk for given dense adjacency matrix."""
    rw_list = []
    deg = dense_adj.sum(-1, keepdims=True)
    AD = dense_adj / (deg + 1e-8)
    rw_list.append(AD)

    for _ in range(k_step):
        rw = torch.bmm(rw_list[-1], AD)
        rw_list.append(rw)
    rw_map = torch.stack(rw_list[1:], dim=1)  # [B, k_step, N, N]

    rw_landing = torch.diagonal(rw_map, offset=0, dim1=2, dim2=3)  # [B, k_step, N]
    rw_landing = rw_landing.permute(0, 2, 1)  # [B, N, rw_depth]

    # get the shortest path distance indices
    tmp_rw = rw_map.sort(dim=1)[0]
    spd_ind = (tmp_rw <= 0).sum(dim=1)  # [B, N, N]

    spd_onehot = torch.nn.functional.one_hot(spd_ind, num_classes=k_step + 1).to(torch.float)
    spd_onehot = spd_onehot.permute(0, 3, 1, 2)  # [B, kstep, N, N]

    return rw_landing, spd_onehot


# ---------------------------------------------------------------------------
# Verbatim from models/layers.py
# ---------------------------------------------------------------------------


def get_act(config):
    """Get activation functions from the config file."""
    if config.model.nonlinearity.lower() == "elu":
        return nn.ELU()
    elif config.model.nonlinearity.lower() == "relu":
        return nn.ReLU()
    elif config.model.nonlinearity.lower() == "lrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif config.model.nonlinearity.lower() == "swish":
        return nn.SiLU()
    elif config.model.nonlinearity.lower() == "tanh":
        return nn.Tanh()
    else:
        raise NotImplementedError("activation function does not exist!")


def conv1x1(in_planes, out_planes, stride=1, bias=True, dilation=1, padding=0):
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=bias,
        dilation=dilation,
        padding=padding,
    )
    return conv


# from DDPM
def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


# ---------------------------------------------------------------------------
# Verbatim from models/trans_layers.py
# ---------------------------------------------------------------------------


class PosTransLayer(MessagePassing):
    """Involving the edge feature and updating position feature. Multiply Msg."""

    _alpha: OptTensor

    def __init__(
        self,
        x_channels: int,
        pos_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        act=None,
        attn_clamp: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(PosTransLayer, self).__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.pos_channels = pos_channels
        self.in_channels = in_channels = x_channels + pos_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.attn_clamp = attn_clamp

        if act is None:
            self.act = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.act = act

        self.lin_key = Linear(in_channels, heads * out_channels)
        self.lin_query = Linear(in_channels, heads * out_channels)
        self.lin_value = Linear(in_channels, heads * out_channels)

        self.lin_edge0 = Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_edge1 = Linear(edge_dim, heads * out_channels, bias=False)

        self.lin_pos = Linear(heads * out_channels, pos_channels, bias=False)

        self.lin_skip = Linear(x_channels, heads * out_channels, bias=bias)
        self.norm1 = nn.GroupNorm(
            num_groups=min(heads * out_channels // 4, 32),
            num_channels=heads * out_channels,
            eps=1e-6,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=min(heads * out_channels // 4, 32),
            num_channels=heads * out_channels,
            eps=1e-6,
        )
        # FFN
        self.FFN = nn.Sequential(
            Linear(heads * out_channels, heads * out_channels),
            self.act,
            Linear(heads * out_channels, heads * out_channels),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_skip.reset_parameters()
        self.lin_edge0.reset_parameters()
        self.lin_edge1.reset_parameters()
        self.lin_pos.reset_parameters()

    def forward(
        self, x: OptTensor, pos: Tensor, edge_index, edge_attr: OptTensor = None
    ) -> Tuple[Tensor, Tensor]:
        H, C = self.heads, self.out_channels

        x_feat = torch.cat([x, pos], -1)
        query = self.lin_query(x_feat).view(-1, H, C)
        key = self.lin_key(x_feat).view(-1, H, C)
        value = self.lin_value(x_feat).view(-1, H, C)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x, out_pos = self.propagate(
            edge_index, query=query, key=key, value=value, pos=pos, edge_attr=edge_attr, size=None
        )

        out_x = out_x.view(-1, self.heads * self.out_channels)

        # skip connection for x
        x_r = self.lin_skip(x)
        out_x = (out_x + x_r) / math.sqrt(2)
        out_x = self.norm1(out_x)

        # FFN
        out_x = (out_x + self.FFN(out_x)) / math.sqrt(2)
        out_x = self.norm2(out_x)

        # skip connection for pos
        out_pos = pos + torch.tanh(pos + out_pos)

        return out_x, out_pos

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        pos_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr,
        size_i: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        edge_attn = self.lin_edge0(edge_attr).view(-1, self.heads, self.out_channels)
        alpha = (query_i * key_j * edge_attn).sum(dim=-1) / math.sqrt(self.out_channels)
        if self.attn_clamp:
            alpha = alpha.clamp(min=-5.0, max=5.0)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # node feature message
        msg = value_j
        msg = msg * self.lin_edge1(edge_attr).view(-1, self.heads, self.out_channels)
        msg = msg * alpha.view(-1, self.heads, 1)

        # node position message
        pos_msg = pos_j * self.lin_pos(msg.reshape(-1, self.heads * self.out_channels))

        return msg, pos_msg

    def aggregate(
        self, inputs: Tuple[Tensor, Tensor], index: Tensor, ptr=None, dim_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        if ptr is not None:
            raise NotImplementedError("Not implement Ptr in aggregate")
        else:
            return (
                scatter(inputs[0], index, 0, dim_size=dim_size, reduce=self.aggr),
                scatter(inputs[1], index, 0, dim_size=dim_size, reduce="mean"),
            )

    def update(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        return inputs

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


# ---------------------------------------------------------------------------
# Verbatim from models/gnns.py
# ---------------------------------------------------------------------------


class pos_gnn(nn.Module):
    def __init__(
        self,
        act,
        x_ch,
        pos_ch,
        out_ch,
        max_node,
        graph_layer,
        n_layers=3,
        edge_dim=None,
        heads=4,
        temb_dim=None,
        dropout=0.1,
        attn_clamp=False,
    ):
        super().__init__()
        self.out_ch = out_ch
        self.Dropout_0 = nn.Dropout(dropout)
        self.act = act
        self.max_node = max_node
        self.n_layers = n_layers

        if temb_dim is not None:
            self.Dense_node0 = nn.Linear(temb_dim, x_ch)
            self.Dense_node1 = nn.Linear(temb_dim, pos_ch)
            self.Dense_edge0 = nn.Linear(temb_dim, edge_dim)
            self.Dense_edge1 = nn.Linear(temb_dim, edge_dim)

        self.convs = nn.ModuleList()
        self.edge_convs = nn.ModuleList()
        self.edge_layer = nn.Linear(edge_dim * 2 + self.out_ch, edge_dim)

        graph_layer_cls = {"PosTransLayer": PosTransLayer}[graph_layer]
        for i in range(n_layers):
            if i == 0:
                self.convs.append(
                    graph_layer_cls(
                        x_ch,
                        pos_ch,
                        self.out_ch // heads,
                        heads,
                        edge_dim=edge_dim * 2,
                        act=act,
                        attn_clamp=attn_clamp,
                    )
                )
            else:
                self.convs.append(
                    graph_layer_cls(
                        self.out_ch,
                        pos_ch,
                        self.out_ch // heads,
                        heads,
                        edge_dim=edge_dim * 2,
                        act=act,
                        attn_clamp=attn_clamp,
                    )
                )
            self.edge_convs.append(nn.Linear(self.out_ch, edge_dim * 2))

    def forward(self, x_degree, x_pos, edge_index, dense_ori, dense_spd, dense_index, temb=None):
        """
        Args:
            x_degree: node degree feature [B*N, x_ch]
            x_pos: node rwpe feature [B*N, pos_ch]
            edge_index: [2, edge_length]
            dense_ori: edge feature [B, N, N, nf//2]
            dense_spd: edge shortest path distance feature [B, N, N, nf//2]
            dense_index
            temb: [B, temb_dim]
        """
        B, N, _, _ = dense_ori.shape

        if temb is not None:
            dense_ori = dense_ori + self.Dense_edge0(self.act(temb))[:, None, None, :]
            dense_spd = dense_spd + self.Dense_edge1(self.act(temb))[:, None, None, :]

            temb = temb.unsqueeze(1).repeat(1, self.max_node, 1)
            temb = temb.reshape(-1, temb.shape[-1])
            x_degree = x_degree + self.Dense_node0(self.act(temb))
            x_pos = x_pos + self.Dense_node1(self.act(temb))

        dense_edge = torch.cat([dense_ori, dense_spd], dim=-1)

        ori_edge_attr = dense_edge
        h = x_degree
        h_pos = x_pos

        for i_layer in range(self.n_layers):
            h_edge = dense_edge[dense_index]
            # update node feature
            h, h_pos = self.convs[i_layer](h, h_pos, edge_index, h_edge)
            h = self.Dropout_0(h)
            h_pos = self.Dropout_0(h_pos)

            # update dense edge feature
            h_dense_node = h.reshape(B, N, -1)
            cur_edge_attr = h_dense_node.unsqueeze(1) + h_dense_node.unsqueeze(2)  # [B, N, N, nf]
            dense_edge = (
                dense_edge + self.act(self.edge_convs[i_layer](cur_edge_attr))
            ) / math.sqrt(2.0)
            dense_edge = self.Dropout_0(dense_edge)

        # Concat edge attribute
        h_dense_edge = torch.cat([ori_edge_attr, dense_edge], dim=-1)
        h_dense_edge = self.edge_layer(h_dense_edge).permute(0, 3, 1, 2)

        return h_dense_edge


# ---------------------------------------------------------------------------
# Verbatim from models/pgsn.py
# ---------------------------------------------------------------------------


@register_model(name="PGSN")
class PGSN(nn.Module):
    """Position enhanced graph score network."""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.act = act = get_act(config)

        # get model construction paras
        self.nf = nf = config.model.nf
        self.num_gnn_layers = num_gnn_layers = config.model.num_gnn_layers
        dropout = config.model.dropout
        self.embedding_type = embedding_type = config.model.embedding_type.lower()
        self.rw_depth = rw_depth = config.model.rw_depth
        self.edge_th = config.model.edge_th

        modules = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == "positional":
            embed_dim = nf
        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")

        # timestep embedding layers
        modules.append(nn.Linear(embed_dim, nf * 4))
        modules.append(nn.Linear(nf * 4, nf * 4))

        # graph size condition embedding
        self.size_cond = size_cond = config.model.size_cond
        if size_cond:
            self.size_onehot = functools.partial(
                nn.functional.one_hot, num_classes=config.data.max_node + 1
            )
            modules.append(nn.Linear(config.data.max_node + 1, nf * 4))
            modules.append(nn.Linear(nf * 4, nf * 4))

        channels = config.data.num_channels
        assert channels == 1, "Without edge features."

        # degree onehot
        self.degree_max = self.config.data.max_node // 2
        self.degree_onehot = functools.partial(
            nn.functional.one_hot, num_classes=self.degree_max + 1
        )

        # project edge features
        modules.append(conv1x1(channels, nf // 2))
        modules.append(conv1x1(rw_depth + 1, nf // 2))

        # project node features
        self.x_ch = nf
        self.pos_ch = nf // 2
        modules.append(nn.Linear(self.degree_max + 1, self.x_ch))
        modules.append(nn.Linear(rw_depth, self.pos_ch))

        # GNN
        modules.append(
            pos_gnn(
                act,
                self.x_ch,
                self.pos_ch,
                nf,
                config.data.max_node,
                config.model.graph_layer,
                num_gnn_layers,
                heads=config.model.heads,
                edge_dim=nf // 2,
                temb_dim=nf * 4,
                dropout=dropout,
                attn_clamp=config.model.attn_clamp,
            )
        )

        # output
        modules.append(conv1x1(nf // 2, nf // 2))
        modules.append(conv1x1(nf // 2, channels))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond, *args, **kwargs):
        mask = kwargs["mask"]
        modules = self.all_modules
        m_idx = 0

        # Sinusoidal positional embeddings
        timesteps = time_cond
        temb = get_timestep_embedding(timesteps, self.nf)

        # time embedding
        temb = modules[m_idx](temb)
        m_idx += 1
        temb = modules[m_idx](self.act(temb))
        m_idx += 1

        if self.size_cond:
            with torch.no_grad():
                node_mask = mask_adj2node(mask.squeeze(1))  # [B, N]
                num_node = torch.sum(node_mask, dim=-1)  # [B]
                num_node = self.size_onehot(num_node.to(torch.long)).to(torch.float)
            num_node_emb = modules[m_idx](num_node)
            m_idx += 1
            num_node_emb = modules[m_idx](self.act(num_node_emb))
            m_idx += 1
            temb = temb + num_node_emb

        if not self.config.data.centered:
            # rescale the input data to [-1, 1]
            x = x * 2.0 - 1.0

        with torch.no_grad():
            # continuous-valued graph adjacency matrices
            cont_adj = ((x + 1.0) / 2.0).clone()
            cont_adj = (cont_adj * mask).squeeze(1)  # [B, N, N]
            cont_adj = cont_adj.clamp(min=0.0, max=1.0)
            if self.edge_th > 0.0:
                cont_adj[cont_adj < self.edge_th] = 0.0

            # discretized graph adjacency matrices
            adj = x.squeeze(1).clone()  # [B, N, N]
            adj[adj >= 0.0] = 1.0
            adj[adj < 0.0] = 0.0
            adj = adj * mask.squeeze(1)

        # extract RWSE and Shortest-Path Distance
        x_pos, spd_onehot = get_rw_feat(self.rw_depth, adj)

        # edge [B, N, N, F]
        dense_edge_ori = modules[m_idx](x).permute(0, 2, 3, 1)
        m_idx += 1
        dense_edge_spd = modules[m_idx](spd_onehot).permute(0, 2, 3, 1)
        m_idx += 1

        # Use Degree as node feature
        x_degree = torch.sum(cont_adj, dim=-1)  # [B, N]
        x_degree = x_degree.clamp(max=float(self.degree_max))
        x_degree = self.degree_onehot(x_degree.to(torch.long)).to(torch.float)  # [B, N, max_node]
        x_degree = modules[m_idx](x_degree)  # projection layer [B, N, nf]
        m_idx += 1

        # pos encoding
        x_pos = modules[m_idx](x_pos)
        m_idx += 1

        # Dense to sparse node [BxN, -1]
        x_degree = x_degree.reshape(-1, self.x_ch)
        x_pos = x_pos.reshape(-1, self.pos_ch)
        dense_index = cont_adj.nonzero(as_tuple=True)
        edge_index, _ = dense_to_sparse(cont_adj)

        # Run GNN layers
        h_dense_edge = modules[m_idx](
            x_degree, x_pos, edge_index, dense_edge_ori, dense_edge_spd, dense_index, temb
        )
        m_idx += 1

        # Output
        h = self.act(modules[m_idx](self.act(h_dense_edge)))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1

        # make edge estimation symmetric
        h = (h + h.transpose(2, 3)) / 2.0 * mask

        assert m_idx == len(modules)

        return h


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------


def _make_config():
    """Plain nested SimpleNamespace mirroring the repo's own
    configs/vp_ego_small_pgsn.py dotted-attribute shape (ml_collections is
    not an installed base lib here; PGSN only ever reads config attributes,
    so this is a config-container substitution, not an architecture
    change), shrunk to a tiny size for a fast trace."""
    config = SimpleNamespace()
    config.model = SimpleNamespace(
        name="PGSN",
        nonlinearity="swish",
        nf=8,
        num_gnn_layers=2,
        size_cond=False,
        embedding_type="positional",
        rw_depth=3,
        graph_layer="PosTransLayer",
        edge_th=-1.0,
        heads=2,
        attn_clamp=False,
        dropout=0.0,
    )
    config.data = SimpleNamespace(
        centered=True,
        max_node=6,
        num_channels=1,
    )
    return config


class _PGSNPositionalWrapper(nn.Module):
    """Menagerie staging adapter only: PGSN.forward requires `mask` as a
    required keyword argument (kwargs['mask']), but the trace harness calls
    the built model with purely positional example-input args. This thin
    wrapper stores a fixed-size all-valid mask (matching the tiny example
    graph's shape, no padding) as a buffer and forwards
    `forward(x, time_cond)` -> `PGSN.forward(x, time_cond, mask=self.mask)`
    unchanged -- no PGSN architecture code is altered."""

    def __init__(self, pgsn: PGSN, mask: torch.Tensor):
        super().__init__()
        self.pgsn = pgsn
        self.register_buffer("mask", mask)

    def forward(self, x, time_cond):
        return self.pgsn(x, time_cond, mask=self.mask)


def build_graphgdp():
    """Tiny-size real GraphGDP / PGSN (sinusoidal timestep embed + degree
    and RWPE/SPD graph feature extraction + PosTransLayer graph-transformer
    score network), wrapped only to make the required `mask` kwarg
    positional-call-friendly for tracing (see _PGSNPositionalWrapper)."""
    batch, n = 1, 6
    mask = torch.ones(batch, 1, n, n)
    return _PGSNPositionalWrapper(PGSN(_make_config()), mask)


def example_input_graphgdp():
    """One padded 6-node noisy adjacency-matrix batch matching PGSN.forward's
    (x, time_cond, mask=...) contract: x is [B, 1, N, N] (a noised, roughly
    symmetric adjacency in [-1, 1] since data.centered=True), time_cond is a
    per-graph continuous diffusion timestep in [0, 1]."""
    torch.manual_seed(0)
    batch, n = 1, 6
    raw = torch.rand(batch, n, n) * 2.0 - 1.0
    adj = (raw + raw.transpose(1, 2)) / 2.0
    x = adj.unsqueeze(1)  # [B, 1, N, N]
    time_cond = torch.tensor([0.5])
    return (x, time_cond)


MENAGERIE_ENTRIES = [
    (
        "GraphGDP",
        build_graphgdp,
        example_input_graphgdp,
        2022,
        "CODE",
    ),
]
