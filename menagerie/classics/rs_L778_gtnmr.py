# SOURCE: vendored from https://github.com/Anan-Wu-XMU/GT-NMR @ main
# (gt/config/*.py + gt/act/example.py + gt/utils.py + gt/encoder/custom_atom_bond.py +
#  gt/encoder/rrwp_encoder.py + gt/layer/other_attn_layer.py + gt/layer/grit_layer.py +
#  gt/head/node_regression_head.py + gt/network/grit_model.py + gt/transform/rrwp.py)
#
# GT-NMR: Graph transformer-based model for NMR spectroscopy prediction
# (Anan Wu et al., XMU). The repo listed under the queue name "GT-NMR" was an
# empty stub (README only); the actual model code lives in the differently-cased
# fork `Anan-Wu-XMU/GT-NMR`, which trains/evaluates this same GritTransformer
# architecture on 1H/13C NMR chemical-shift prediction (results/logging.log in
# that repo references `GT-NMR-code-main`).
#
# The architecture is GRIT (Graph Inductive Bias Transformer, Ma et al. ICML 2023)
# wired through PyG's GraphGym registry system exactly as the real repo does:
# `torch_geometric.graphgym.register.register_network('GritTransformer')` etc.
# register real nn.Module classes into GraphGym's global dicts, and the network
# is built via `network_dict['GritTransformer'](dim_in, dim_out)` against a
# `torch_geometric.graphgym.config.cfg` singleton populated from the repo's own
# `configs/gt13C.yaml` values (RRWP positional encoding, custom Atom/Bond node
# and edge encoders, node-regression head). No architecture was altered.
#
# Only minimal changes were made to make the module self-contained for staging:
#   - `gt/__init__.py`'s package-wide `from .loader import *` / `from .train import *`
#     were dropped (those pull in uproot/rdkit-flavoured dataset-loading and the
#     mlflow/wandb training loop -- irrelevant to constructing/tracing the model).
#   - all files were concatenated into one module (was split across
#     gt/config/*.py, gt/act/example.py, gt/utils.py, gt/encoder/*.py,
#     gt/layer/*.py, gt/head/node_regression_head.py, gt/network/grit_model.py,
#     gt/transform/rrwp.py) since torchlens staging validates via direct
#     single-file import.
#   - `full_edge_index`/`pyg_softmax` are duplicated verbatim in both
#     rrwp_encoder.py and grit_layer.py/other_attn_layer.py in the real repo;
#     that duplication is preserved as-is (not de-duped) to keep this a faithful
#     copy of the original files rather than a restructuring.

import warnings as _warnings
from functools import partial
from typing import Any, List, Optional, Union

import numpy as np
import opt_einsum as oe
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_sparse
from ogb.utils.features import get_bond_feature_dims  # noqa: F401 (kept for fidelity; unused at runtime)
from torch import Tensor
from torch.nn import functional as tF  # noqa: F401 (kept for fidelity with rrwp_encoder.py's `F` alias)
from torch_geometric.data import Batch, Data
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import BatchNorm1dNode, new_layer_config
from torch_geometric.graphgym.register import (
    act_dict,
    edge_encoder_dict,
    head_dict,
    layer_dict,
    network_dict,
    node_encoder_dict,
    register_act,
    register_config,
    register_edge_encoder,
    register_head,
    register_layer,
    register_network,
    register_node_encoder,
)
from torch_geometric.utils import (
    add_remaining_self_loops,
    degree,
    remove_self_loops,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_add, scatter_max
from torch_sparse import SparseTensor
from yacs.config import CfgNode as CN

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# gt/config/gt_config.py
# ---------------------------------------------------------------------------


@register_config("cfg_gt")
def set_cfg_gt(cfg):
    """Configuration for Graph Transformer-style models, e.g.:
    - Spectral Attention Network (SAN) Graph Transformer.
    - "vanilla" Transformer / Performer.
    - General Powerful Scalable (GPS) Model.
    """

    cfg.gt = CN()
    cfg.gt.layer_type = "GritTransformer"
    cfg.gt.layers = 3
    cfg.gt.n_heads = 8
    cfg.gt.dim_hidden = 64
    cfg.gt.full_graph = True
    cfg.gt.gamma = 1e-5
    cfg.gt.pna_degrees = []
    cfg.gt.dropout = 0.0
    cfg.gt.attn_dropout = 0.0
    cfg.gt.layer_norm = False
    cfg.gt.batch_norm = True
    cfg.gt.bn_momentum = 0.1
    cfg.gt.bn_no_runner = False
    cfg.gt.residual = True

    cfg.gt.bigbird = CN()
    cfg.gt.bigbird.attention_type = "block_sparse"
    cfg.gt.bigbird.chunk_size_feed_forward = 0
    cfg.gt.bigbird.is_decoder = False
    cfg.gt.bigbird.add_cross_attention = False
    cfg.gt.bigbird.hidden_act = "relu"
    cfg.gt.bigbird.max_position_embeddings = 128
    cfg.gt.bigbird.use_bias = False
    cfg.gt.bigbird.num_random_blocks = 3
    cfg.gt.bigbird.block_size = 3
    cfg.gt.bigbird.layer_norm_eps = 1e-6

    # ------------- Special for GRIT ------------
    cfg.gt.update_e = True
    cfg.gt.attn = CN()
    cfg.gt.attn.use = True
    cfg.gt.attn.sparse = False
    cfg.gt.attn.deg_scaler = True
    cfg.gt.attn.use_bias = False
    cfg.gt.attn.clamp = 5.0
    cfg.gt.attn.act = "signed_sqrt"
    cfg.gt.attn.full_attn = True
    cfg.gt.attn.norm_e = True
    cfg.gt.attn.O_e = True
    cfg.gt.attn.edge_enhance = True


# ---------------------------------------------------------------------------
# gt/config/posenc_config.py
# ---------------------------------------------------------------------------


@register_config("posenc")
def set_cfg_posenc(cfg):
    """Extend configuration with positional encoding options."""

    cfg.posenc_LapPE = CN()
    cfg.posenc_SignNet = CN()
    cfg.posenc_RWSE = CN()
    cfg.posenc_HKdiagSE = CN()
    cfg.posenc_ElstaticSE = CN()
    cfg.posenc_EquivStableLapPE = CN()
    cfg.posenc_RRWP = CN()

    for name in [
        "posenc_LapPE",
        "posenc_SignNet",
        "posenc_RWSE",
        "posenc_HKdiagSE",
        "posenc_ElstaticSE",
        "posenc_RRWP",
    ]:
        pecfg = getattr(cfg, name)
        pecfg.enable = False
        pecfg.model = "none"
        pecfg.dim_pe = 16
        pecfg.layers = 3
        pecfg.n_heads = 4
        pecfg.post_layers = 0
        pecfg.raw_norm_type = "none"
        pecfg.pass_as_var = False

    cfg.posenc_EquivStableLapPE.enable = False
    cfg.posenc_EquivStableLapPE.raw_norm_type = "none"

    for name in ["posenc_LapPE", "posenc_SignNet", "posenc_EquivStableLapPE"]:
        pecfg = getattr(cfg, name)
        pecfg.eigen = CN()
        pecfg.eigen.laplacian_norm = "sym"
        pecfg.eigen.eigvec_norm = "L2"
        pecfg.eigen.max_freqs = 10

    cfg.posenc_SignNet.phi_out_dim = 4
    cfg.posenc_SignNet.phi_hidden_dim = 64

    for name in ["posenc_RWSE", "posenc_HKdiagSE", "posenc_ElstaticSE"]:
        pecfg = getattr(cfg, name)
        pecfg.kernel = CN()
        pecfg.kernel.times = []
        pecfg.kernel.times_func = ""

    cfg.posenc_ElstaticSE.kernel.times_func = "range(10)"

    cfg.posenc_RRWP.enable = False
    cfg.posenc_RRWP.ksteps = 21
    cfg.posenc_RRWP.add_identity = True
    cfg.posenc_RRWP.spd = False


# ---------------------------------------------------------------------------
# gt/config/custom_gnn_config.py, dataset_config.py, defaults_config.py,
# optimizers_config.py, split_config.py, wandb_config.py, mlflow_config.py,
# pretrained_config.py
# ---------------------------------------------------------------------------


@register_config("custom_gnn")
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model."""
    cfg.gnn.residual = False


@register_config("dataset_cfg")
def dataset_cfg(cfg):
    """Dataset-specific config options."""
    cfg.dataset.node_encoder_num_types = 0
    cfg.dataset.edge_encoder_num_types = 0
    cfg.dataset.slic_compactness = 10
    cfg.dataset.pe_transform_on_the_fly = False


@register_config("overwrite_defaults")
def overwrite_defaults_cfg(cfg):
    """Overwrite the default config values that are first set by GraphGym in
    torch_geometric.graphgym.config.set_cfg"""
    cfg.dataset.name = "none"
    cfg.round = 5


@register_config("extended_cfg")
def extended_cfg(cfg):
    """General extended config options."""
    cfg.name_tag = ""
    cfg.best_by_loss = False
    cfg.train.ckpt_best = True
    cfg.tensorboard_each_run = True


@register_config("extended_optim")
def extended_optim_cfg(cfg):
    """Extend optimizer config group that is first set by GraphGym."""
    cfg.optim.batch_accumulation = 1
    cfg.optim.reduce_factor = 0.5
    cfg.optim.schedule_patience = 10
    cfg.optim.min_lr = 0.0
    cfg.optim.num_warmup_epochs = 50
    cfg.optim.clip_grad_norm = False
    cfg.optim.early_stop_by_lr = False
    cfg.optim.early_stop_by_perf = False
    cfg.optim.stop_patience = 100
    cfg.optim.num_cycles = 0.5
    cfg.optim.min_lr_mode = "threshold"


@register_config("split")
def set_cfg_split(cfg):
    """Reconfigure the default config value for dataset split options."""
    cfg.dataset.split_mode = "standard"
    cfg.dataset.split_index = 0
    cfg.dataset.split_dir = "./splits"
    cfg.run_multiple_splits = []


@register_config("cfg_wandb")
def set_cfg_wandb(cfg):
    """Weights & Biases tracker configuration."""
    cfg.wandb = CN()
    cfg.wandb.use = False
    cfg.wandb.entity = "gtransformers"
    cfg.wandb.project = "gtblueprint"
    cfg.wandb.name = ""


@register_config("cfg_mlflow")
def set_cfg_mlflow(cfg):
    """MLflow tracker configuration."""
    cfg.mlflow = CN()
    cfg.mlflow.use = False
    cfg.mlflow.project = " "
    cfg.mlflow.name = " "


@register_config("cfg_pretrained")
def set_cfg_pretrained(cfg):
    """Configuration options for loading a pretrained model."""
    cfg.pretrained = CN()
    cfg.pretrained.dir = ""
    cfg.pretrained.reset_prediction_head = True
    cfg.pretrained.freeze_main = False


# ---------------------------------------------------------------------------
# gt/act/example.py
# ---------------------------------------------------------------------------


class SWISH(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class SignedSqrt(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.sqrt(torch.relu(x)) - torch.sqrt(torch.relu(-x))
        return x


register_act("swish", partial(SWISH, inplace=True))
register_act("lrelu_03", partial(nn.LeakyReLU, negative_slope=0.3, inplace=True))
register_act("lrelu_02", partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))
register_act("gelu", nn.GELU)
register_act("signed_sqrt", SignedSqrt)


# ---------------------------------------------------------------------------
# gt/utils.py (only negate_edge_index is used by grit_layer.py's import)
# ---------------------------------------------------------------------------


def negate_edge_index(edge_index, batch=None):
    """Negate batched sparse adjacency matrices given by edge indices."""
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce="add")
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short, device=edge_index.device)

        flattened_size = n * n
        adj = adj.view([flattened_size])
        _idx1 = idx1[idx0 == i]
        _idx2 = idx2[idx0 == i]
        idx = _idx1 * n + _idx2
        zero = torch.zeros(_idx1.numel(), dtype=torch.short, device=edge_index.device)
        scatter(zero, idx, dim=0, out=adj, reduce="mul")

        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_negative = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_negative


# ---------------------------------------------------------------------------
# gt/encoder/custom_atom_bond.py
# ---------------------------------------------------------------------------


@register_node_encoder("customAtom")
class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, *args, **kwargs):
        super().__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        num_features = cfg.dataset.customAtom_num_features
        for _ in range(num_features):
            emb = torch.nn.Embedding(150, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, batch):
        encoded_features = 0
        for i in range(batch.x.shape[1]):
            encoded_features += self.atom_embedding_list[i](batch.x[:, i])

        batch.x = encoded_features
        return batch


@register_edge_encoder("customBond")
class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()

        num_edge_features = cfg.dataset.customBond_num_features

        self.bond_embedding_list = torch.nn.ModuleList()

        for _ in range(num_edge_features):
            emb = torch.nn.Embedding(50, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, batch):
        bond_embedding = 0
        for i in range(batch.edge_attr.shape[1]):
            edge_attr = batch.edge_attr
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        batch.edge_attr = bond_embedding
        return batch


# ---------------------------------------------------------------------------
# gt/encoder/rrwp_encoder.py
# ---------------------------------------------------------------------------


def full_edge_index(edge_index, batch=None):
    """Return the Full batched sparse adjacency matrices given by edge indices.

    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.
    Implementation inspired by `torch_geometric.utils.to_dense_adj`.
    """

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce="add")
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short, device=edge_index.device)

        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_full = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_full


@register_node_encoder("rrwp_linear")
class RRWPLinearNodeEncoder(torch.nn.Module):
    """FC_1(RRWP) + FC_2 (Node-attr)."""

    def __init__(
        self, emb_dim, out_dim, use_bias=False, batchnorm=False, layernorm=False, pe_name="rrwp"
    ):
        super().__init__()
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.name = pe_name

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        rrwp = batch[f"{self.name}"]
        rrwp = self.fc(rrwp)

        if self.batchnorm:
            rrwp = self.bn(rrwp)

        if self.layernorm:
            rrwp = self.ln(rrwp)

        if "x" in batch:
            batch.x = batch.x + rrwp
        else:
            batch.x = rrwp

        return batch


@register_edge_encoder("rrwp_linear")
class RRWPLinearEdgeEncoder(torch.nn.Module):
    """Merge RRWP with given edge-attr and Zero-padding to all pairs of node."""

    def __init__(
        self,
        emb_dim,
        out_dim,
        batchnorm=False,
        layernorm=False,
        use_bias=False,
        pad_to_full_graph=True,
        fill_value=0.0,
        add_node_attr_as_self_loop=False,
        overwrite_old_attr=False,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
        self.overwrite_old_attr = overwrite_old_attr

        self.batchnorm = batchnorm
        self.layernorm = layernorm
        if self.batchnorm or self.layernorm:
            _warnings.warn(
                "batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info "
            )

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.pad_to_full_graph = pad_to_full_graph
        self.fill_value = 0.0

        padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
        self.register_buffer("padding", padding)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)

        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        rrwp_idx = batch.rrwp_index
        rrwp_val = batch.rrwp_val
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        rrwp_val = self.fc(rrwp_val)

        if edge_attr is None:
            edge_attr = edge_index.new_zeros(edge_index.size(1), rrwp_val.size(1))

        if self.overwrite_old_attr:
            out_idx, out_val = rrwp_idx, rrwp_val
        else:
            edge_index, edge_attr = add_remaining_self_loops(
                edge_index, edge_attr, num_nodes=batch.num_nodes, fill_value=0.0
            )
            out_idx, out_val = torch_sparse.coalesce(
                torch.cat([edge_index, rrwp_idx], dim=1),
                torch.cat([edge_attr, rrwp_val], dim=0),
                batch.num_nodes,
                batch.num_nodes,
                op="add",
            )

        if self.pad_to_full_graph:
            edge_index_full = full_edge_index(out_idx, batch=batch.batch)
            edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
            out_idx = torch.cat([out_idx, edge_index_full], dim=1)
            out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            out_idx, out_val = torch_sparse.coalesce(
                out_idx, out_val, batch.num_nodes, batch.num_nodes, op="add"
            )

        if self.batchnorm:
            out_val = self.bn(out_val)

        if self.layernorm:
            out_val = self.ln(out_val)

        batch.edge_index, batch.edge_attr = out_idx, out_val
        return batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(pad_to_full_graph={self.pad_to_full_graph},"
            f"fill_value={self.fill_value},"
            f"{self.fc.__repr__()})"
        )


# ---------------------------------------------------------------------------
# gt/layer/other_attn_layer.py (MultiHeadAttentionLayerGraphormerSparse is
# imported by grit_layer.py's optional graphormer_attn branch)
# ---------------------------------------------------------------------------


def pyg_softmax(src, index, num_nodes=None):
    """Computes a sparsely evaluated softmax."""
    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out


class MultiHeadAttentionLayerGraphormerSparse(nn.Module):
    """Multi-Head Graph Attention Layer. Scaled Dot-product."""

    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        use_bias,
        clamp=None,
        dropout=0.0,
        act=None,
        edge_enhance=False,
        **kwargs,
    ):
        super().__init__()

        clamp = None

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        if act is None:
            self.act = nn.Identity()
        else:
            self.act = act_dict[act]()

        if self.edge_enhance:
            self.VeRow = nn.Parameter(
                torch.zeros(self.out_dim, self.num_heads, self.out_dim), requires_grad=True
            )
            nn.init.xavier_normal_(self.VeRow)

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]
        dest = batch.Q_h[batch.edge_index[1]]
        score = src * dest / np.sqrt(self.out_dim)
        score = score.sum(dim=-1, keepdim=True)

        if batch.get("E", None) is not None:
            E_b = batch.E.view(-1, self.num_heads, 1)
            score = score + E_b

        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        score = pyg_softmax(score, batch.edge_index[1])
        score = self.dropout(score)

        batch.attn = score

        msg = batch.V_h[batch.edge_index[0]] * score
        batch.wV = torch.zeros_like(batch.V_h)
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce="add")

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)

        V_h = self.V(batch.x)
        if batch.get("edge_attr", None) is not None:
            batch.E = self.E(batch.edge_attr)
        else:
            batch.E = None

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(batch)
        h_out = batch.wV
        e_out = batch.get("wE", None)

        return h_out, e_out


# ---------------------------------------------------------------------------
# gt/layer/grit_layer.py
# ---------------------------------------------------------------------------


class MultiHeadAttentionLayerGritSparse(nn.Module):
    """Proposed Attention Computation for GRIT."""

    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        use_bias,
        clamp=5.0,
        dropout=0.0,
        act=None,
        edge_enhance=True,
        sqrt_relu=False,
        signed_sqrt=True,
        cfg=CN(),
        **kwargs,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1), requires_grad=True)
        nn.init.xavier_normal_(self.Aw)

        if act is None:
            self.act = nn.Identity()
        else:
            self.act = act_dict[act]()

        if self.edge_enhance:
            self.VeRow = nn.Parameter(
                torch.zeros(self.out_dim, self.num_heads, self.out_dim), requires_grad=True
            )
            nn.init.xavier_normal_(self.VeRow)

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]
        dest = batch.Q_h[batch.edge_index[1]]
        score = src + dest

        if batch.get("E", None) is not None:
            batch.E = batch.E.view(-1, self.num_heads, self.out_dim * 2)
            E_w, E_b = batch.E[:, :, : self.out_dim], batch.E[:, :, self.out_dim :]
            score = score * E_w
            score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
            score = score + E_b

        score = self.act(score)
        e_t = score

        if batch.get("E", None) is not None:
            batch.wE = score.flatten(1)

        score = oe.contract("ehd, dhc->ehc", score, self.Aw, backend="torch")
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        score = pyg_softmax(score, batch.edge_index[1])
        score = self.dropout(score)
        batch.attn = score

        msg = batch.V_h[batch.edge_index[0]] * score
        batch.wV = torch.zeros_like(batch.V_h)
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce="add")

        if self.edge_enhance and batch.E is not None:
            rowV = scatter(e_t * score, batch.edge_index[1], dim=0, reduce="add")
            rowV = oe.contract("nhd, dhc -> nhc", rowV, self.VeRow, backend="torch")
            batch.wV = batch.wV + rowV

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)

        V_h = self.V(batch.x)
        if batch.get("edge_attr", None) is not None:
            batch.E = self.E(batch.edge_attr)
        else:
            batch.E = None

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(batch)
        h_out = batch.wV
        e_out = batch.get("wE", None)

        return h_out, e_out


@register_layer("GritTransformer")
class GritTransformerLayer(nn.Module):
    """Proposed Transformer Layer for GRIT."""

    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        dropout=0.0,
        attn_dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        residual=True,
        act="relu",
        norm_e=True,
        O_e=True,
        cfg=dict(),
        **kwargs,
    ):
        super().__init__()

        self.debug = False
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.update_e = cfg.get("update_e", True)
        self.bn_momentum = cfg.bn_momentum
        self.bn_no_runner = cfg.bn_no_runner
        self.rezero = cfg.get("rezero", False)

        self.act = act_dict[act]() if act is not None else nn.Identity()
        if cfg.get("attn", None) is None:
            cfg.attn = dict()
        self.use_attn = cfg.attn.get("use", True)
        self.deg_scaler = cfg.attn.get("deg_scaler", True)

        self.attention = MultiHeadAttentionLayerGritSparse(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=cfg.attn.get("use_bias", False),
            dropout=attn_dropout,
            clamp=cfg.attn.get("clamp", 5.0),
            act=cfg.attn.get("act", "relu"),
            edge_enhance=cfg.attn.get("edge_enhance", True),
            sqrt_relu=cfg.attn.get("sqrt_relu", False),
            signed_sqrt=cfg.attn.get("signed_sqrt", False),
            scaled_attn=cfg.attn.get("scaled_attn", False),
            no_qk=cfg.attn.get("no_qk", False),
        )

        if cfg.attn.get("graphormer_attn", False):
            self.attention = MultiHeadAttentionLayerGraphormerSparse(
                in_dim=in_dim,
                out_dim=out_dim // num_heads,
                num_heads=num_heads,
                use_bias=cfg.attn.get("use_bias", False),
                dropout=attn_dropout,
                clamp=cfg.attn.get("clamp", 5.0),
                act=cfg.attn.get("act", "relu"),
                edge_enhance=True,
                sqrt_relu=cfg.attn.get("sqrt_relu", False),
                signed_sqrt=cfg.attn.get("signed_sqrt", False),
                scaled_attn=cfg.attn.get("scaled_attn", False),
                no_qk=cfg.attn.get("no_qk", False),
            )

        self.O_h = nn.Linear(out_dim // num_heads * num_heads, out_dim)
        if O_e:
            self.O_e = nn.Linear(out_dim // num_heads * num_heads, out_dim)
        else:
            self.O_e = nn.Identity()

        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, out_dim // num_heads * num_heads, 2))
            nn.init.xavier_normal_(self.deg_coef)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(
                out_dim,
                track_running_stats=not self.bn_no_runner,
                eps=1e-5,
                momentum=cfg.bn_momentum,
            )
            self.batch_norm1_e = (
                nn.BatchNorm1d(
                    out_dim,
                    track_running_stats=not self.bn_no_runner,
                    eps=1e-5,
                    momentum=cfg.bn_momentum,
                )
                if norm_e
                else nn.Identity()
            )

        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(
                out_dim,
                track_running_stats=not self.bn_no_runner,
                eps=1e-5,
                momentum=cfg.bn_momentum,
            )

        if self.rezero:
            self.alpha1_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha2_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha1_e = nn.Parameter(torch.zeros(1, 1))

    def forward(self, batch):
        h = batch.x
        num_nodes = batch.num_nodes
        log_deg = get_log_deg(batch)

        h_in1 = h
        e_in1 = batch.get("edge_attr", None)
        e = None

        h_attn_out, e_attn_out = self.attention(batch)

        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        if self.deg_scaler:
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)
        if e_attn_out is not None:
            e = e_attn_out.flatten(1)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.O_e(e)

        if self.residual:
            if self.rezero:
                h = h * self.alpha1_h
            h = h_in1 + h
            if e is not None:
                if self.rezero:
                    e = e * self.alpha1_e
                e = e + e_in1

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            if e is not None:
                e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            if e is not None:
                e = self.batch_norm1_e(e)

        h_in2 = h
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            if self.rezero:
                h = h * self.alpha2_h
            h = h_in2 + h

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        if self.update_e:
            batch.edge_attr = e
        else:
            batch.edge_attr = e_in1

        return batch

    def __repr__(self):
        return "{}(in_channels={}, out_channels={}, heads={}, residual={})\n[{}]".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.num_heads,
            self.residual,
            super().__repr__(),
        )


@torch.no_grad()
def get_log_deg(batch):
    if "log_deg" in batch:
        log_deg = batch.log_deg
    elif "deg" in batch:
        deg = batch.deg
        log_deg = torch.log(deg + 1).unsqueeze(-1)
    else:
        _warnings.warn(
            "Compute the degree on the fly; Might be problematric if have applied edge-padding to complete graphs"
        )
        deg = pyg.utils.degree(batch.edge_index[1], num_nodes=batch.num_nodes, dtype=torch.float)
        log_deg = torch.log(deg + 1)
    log_deg = log_deg.view(batch.num_nodes, 1)
    return log_deg


# ---------------------------------------------------------------------------
# gt/head/node_regression_head.py
# ---------------------------------------------------------------------------


@register_head("node_regression_head")
class SANnoderegressionHead(nn.Module):
    """SAN prediction head for node regression tasks."""

    def __init__(self, dim_in, dim_out, L=2):
        super().__init__()
        self.deg_scaler = False
        self.fwl = False
        self.name = cfg.dataset.name
        self.mode = cfg.train.mode

        list_FC_layers = [
            nn.Linear(dim_in // 2**l, dim_in // 2 ** (l + 1), bias=True) for l in range(L)
        ]  # noqa: E741 (kept for fidelity)
        list_FC_layers.append(nn.Linear(dim_in // 2**L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = act_dict[cfg.gnn.act]()

    def _apply_index(self, batch):
        return batch.x, batch.y

    def _apply_mask(self, batch):
        if batch.mask_node is not None:
            x = batch.x
            y = batch.y
            mask = batch.mask_node
            x = x[mask]
            y = y[mask]
            batch.x = x
            batch.y = y
        return batch

    def _apply_infer_mask(self, batch):
        if batch.infer_mask is not None:
            x = batch.x
            y = batch.y
            mask = batch.infer_mask
            x = x[mask]
            batch.x = x
            batch.y = y
        return batch

    def forward(self, batch):
        if self.mode == "custom":
            batch = self._apply_mask(batch)
        else:
            batch = self._apply_infer_mask(batch)

        graph_emb = batch.x
        for l in range(self.L):  # noqa: E741 (kept for fidelity)
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        batch.x = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label


# ---------------------------------------------------------------------------
# gt/network/grit_model.py
# ---------------------------------------------------------------------------


class FeatureEncoder(torch.nn.Module):
    """Encoding node and edge features."""

    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            NodeEncoder = node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_inner, -1, -1, has_act=False, has_bias=False, cfg=cfg
                    )
                )
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            if "PNA" in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            EdgeEncoder = edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_edge, -1, -1, has_act=False, has_bias=False, cfg=cfg
                    )
                )

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network("GritTransformer")
class GritTransformer(torch.nn.Module):
    """The proposed GritTransformer (Graph Inductive Bias Transformer)."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        self.ablation = True
        self.ablation = False

        if cfg.posenc_RRWP.enable:
            self.rrwp_abs_encoder = node_encoder_dict["rrwp_linear"](
                cfg.posenc_RRWP.ksteps, cfg.gnn.dim_inner
            )
            rel_pe_dim = cfg.posenc_RRWP.ksteps
            self.rrwp_rel_encoder = edge_encoder_dict["rrwp_linear"](
                rel_pe_dim,
                cfg.gnn.dim_edge,
                pad_to_full_graph=cfg.gt.attn.full_attn,
                add_node_attr_as_self_loop=False,
                fill_value=0.0,
            )

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, (
            "The inner and hidden dims must match."
        )

        global_model_type = cfg.gt.get("layer_type", "GritTransformer")

        TransformerLayer = layer_dict.get(global_model_type)

        layers = []
        for l in range(cfg.gt.layers):  # noqa: E741 (kept for fidelity)
            layers.append(
                TransformerLayer(
                    in_dim=cfg.gt.dim_hidden,
                    out_dim=cfg.gt.dim_hidden,
                    num_heads=cfg.gt.n_heads,
                    dropout=cfg.gt.dropout,
                    act=cfg.gnn.act,
                    attn_dropout=cfg.gt.attn_dropout,
                    layer_norm=cfg.gt.layer_norm,
                    batch_norm=cfg.gt.batch_norm,
                    residual=True,
                    norm_e=cfg.gt.attn.norm_e,
                    O_e=cfg.gt.attn.O_e,
                    cfg=cfg.gt,
                )
            )

        self.layers = torch.nn.Sequential(*layers)
        GNNHead = head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)

        return batch


# ---------------------------------------------------------------------------
# gt/transform/rrwp.py
# ---------------------------------------------------------------------------


def add_node_attr(data: Data, value: Any, attr_name: Optional[str] = None) -> Data:
    if attr_name is None:
        if "x" in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data


@torch.no_grad()
def add_full_rrwp(
    data,
    walk_length=8,
    attr_name_abs="rrwp",
    attr_name_rel="rrwp",
    add_identity=True,
    spd=False,
    **kwargs,
):
    device = data.edge_index.device  # noqa: F841 (unused in original; kept for fidelity)
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=(num_nodes, num_nodes))

    deg = adj.sum(dim=1)
    deg_inv = 1.0 / adj.sum(dim=1)
    deg_inv[deg_inv == float("inf")] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list = []
    i = 0
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=torch.float))
        i = i + 1

    out = adj
    pe_list.append(adj)

    if walk_length > 2:
        for j in range(i + 1, walk_length):
            out = out @ adj
            pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1)

    abs_pe = pe.diagonal().transpose(0, 1)

    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)

    if spd:
        spd_idx = walk_length - torch.arange(walk_length)
        val = (rel_pe_val > 0).type(torch.float) * spd_idx.unsqueeze(0)
        val = torch.argmax(val, dim=-1)
        rel_pe_val = F.one_hot(val, walk_length).type(torch.float)
        abs_pe = torch.zeros_like(abs_pe)

    data = add_node_attr(data, abs_pe, attr_name=attr_name_abs)
    data = add_node_attr(data, rel_pe_idx, attr_name=f"{attr_name_rel}_index")
    data = add_node_attr(data, rel_pe_val, attr_name=f"{attr_name_rel}_val")
    data.log_deg = torch.log(deg + 1)
    data.deg = deg.type(torch.long)

    return data


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------

_ATOM_FEATS = 6
_BOND_FEATS = 2
_RRWP_STEPS = 4
_DIM_HIDDEN = 16
_N_LAYERS = 2
_N_HEADS = 2


def _build_gtnmr_cfg():
    """Populate the GraphGym `cfg` singleton with the same field values the
    real repo's configs/gt13C.yaml uses, sized down for a fast trace."""
    set_cfg(cfg)
    cfg.dataset.node_encoder = True
    cfg.dataset.node_encoder_name = "customAtom"
    cfg.dataset.node_encoder_bn = False
    cfg.dataset.edge_encoder = True
    cfg.dataset.edge_encoder_name = "customBond"
    cfg.dataset.edge_encoder_bn = False
    cfg.dataset.customAtom_num_features = _ATOM_FEATS
    cfg.dataset.customBond_num_features = _BOND_FEATS
    cfg.dataset.task = "node"
    cfg.dataset.task_type = "regression"
    cfg.dataset.name = "gt13C_menagerie"
    cfg.train.mode = "custom"
    cfg.posenc_RRWP.enable = True
    cfg.posenc_RRWP.ksteps = _RRWP_STEPS
    cfg.posenc_RRWP.add_identity = True
    cfg.model.type = "GritTransformer"
    cfg.model.loss_fun = "l1"
    cfg.model.edge_decoding = "dot"
    cfg.gt.layer_type = "GritTransformer"
    cfg.gt.layers = _N_LAYERS
    cfg.gt.n_heads = _N_HEADS
    cfg.gt.dim_hidden = _DIM_HIDDEN
    cfg.gt.dropout = 0.0
    cfg.gt.layer_norm = False
    cfg.gt.batch_norm = True
    cfg.gt.update_e = True
    cfg.gt.attn_dropout = 0.0
    cfg.gt.attn.clamp = 5.0
    cfg.gt.attn.act = "signed_sqrt"
    cfg.gt.attn.full_attn = True
    cfg.gt.attn.edge_enhance = True
    cfg.gt.attn.O_e = True
    cfg.gt.attn.norm_e = True
    cfg.gt.attn.signed_sqrt = True
    cfg.gnn.head = "node_regression_head"
    cfg.gnn.layers_pre_mp = 0
    cfg.gnn.layers_post_mp = 2
    cfg.gnn.dim_inner = _DIM_HIDDEN
    cfg.gnn.batchnorm = True
    cfg.gnn.act = "signed_sqrt"
    cfg.gnn.dropout = 0.0
    cfg.gnn.agg = "mean"
    cfg.accelerator = "cpu"
    cfg.share.dim_in = _ATOM_FEATS
    cfg.share.dim_out = 1
    return cfg


class GTNMRWrapper(nn.Module):
    """Thin wrapper so torchlens sees a plain nn.Module(x) -> tensor forward.

    The real repo's training entrypoint (`torch_geometric.graphgym.model_builder
    .create_model`) wraps `GritTransformer` in a PyTorch-Lightning `GraphGymModule`
    whose forward returns `(pred, label)` on a mutated-in-place PyG `Batch`. We call
    the real `GritTransformer` directly (no architecture difference from the
    Lightning wrapper -- it just calls `self.model(batch)`) and return only the
    prediction tensor.
    """

    def __init__(self, grit_transformer: nn.Module):
        super().__init__()
        self.grit = grit_transformer

    def forward(self, batch: Batch) -> Tensor:
        # GritTransformer.forward runs the encoder/layers/post_mp children in
        # sequence; `post_mp` (SANnoderegressionHead) is the last child and its
        # forward returns `(pred, label)` directly rather than a Batch, so
        # GritTransformer's own return value is already that tuple.
        pred, _label = self.grit(batch)
        return pred


def build_gtnmr():
    torch.manual_seed(0)
    _build_gtnmr_cfg()
    model = network_dict["GritTransformer"](dim_in=_ATOM_FEATS, dim_out=1)
    model.eval()
    return GTNMRWrapper(model)


def example_input_gtnmr():
    _build_gtnmr_cfg()  # idempotent; ensures cfg matches build_gtnmr() if called standalone
    torch.manual_seed(0)

    # A tiny synthetic 5-atom ring-like molecule graph (real NMR training data
    # comes from RDKit-parsed .mol/.pickle files via the repo's own loader,
    # which needs rdkit; we are not vendoring the data pipeline, only the
    # model, so we fabricate tensors with the same shapes/dtypes the real
    # AtomEncoder/BondEncoder/RRWP encoders expect).
    n_nodes = 5
    x = torch.randint(0, 10, (n_nodes, _ATOM_FEATS), dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long
    )
    edge_attr = torch.randint(0, 5, (edge_index.size(1), _BOND_FEATS), dtype=torch.long)
    y = torch.randn(n_nodes, 1)
    mask_node = torch.ones(n_nodes, dtype=torch.bool)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.mask_node = mask_node
    data.edge_weight = None
    data = add_full_rrwp(data, walk_length=_RRWP_STEPS, add_identity=True)
    batch = Batch.from_data_list([data])
    batch.split = "train"
    return (batch,)


MENAGERIE_ENTRIES = [
    ("GT-NMR (GritTransformer)", "build_gtnmr", "example_input_gtnmr", 2024, MENAGERIE_ZOO),
]
