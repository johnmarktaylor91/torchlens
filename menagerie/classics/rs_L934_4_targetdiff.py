# SOURCE: vendored from https://github.com/guanjq/targetdiff @ main
# (models/common.py, models/uni_transformer.py, models/molopt_score_model.py)
"""TargetDiff: 3D Equivariant Diffusion for Target-Aware Molecule Generation and
Affinity Prediction (Guan et al., ICLR 2023). TargetDiff jointly diffuses ligand atom
3D coordinates and atom types conditioned on a fixed protein-pocket context, denoised
by an SE(3)-equivariant "Uni-Transformer" (`UniTransformerO2TwoUpdateGeneral`) that
alternates x2h (geometry -> feature) and h2x (feature -> geometry) equivariant
attention updates over a protein+ligand context graph. This module vendors the real
`ScorePosNet3D` denoising network (`models/molopt_score_model.py`), its default
`uni_o2` refine net (`models/uni_transformer.py`), and shared helpers
(`models/common.py`) verbatim -- only the *diffusion sampling/training loop*
(forward process, ELBO loss, `q_v_*`/`p_v_*` posterior helpers, guidance sampling)
is omitted, since that is a training/inference *procedure* built on top of the
network, not part of the network architecture itself. `ScorePosNet3D.forward()` is
called directly here exactly as upstream defines it -- one equivariant denoising
step over protein+ligand atoms at a given diffusion timestep.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, radius_graph
from torch_scatter import scatter_softmax, scatter_sum


# ---------------------------------------------------------------------------
# models/common.py
# ---------------------------------------------------------------------------
class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, fixed_offset=True):
        super().__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        if fixed_offset:
            offset = torch.tensor(
                [0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10]
            )
        else:
            offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "silu": nn.SiLU(),
}


class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""

    def __init__(
        self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn="relu", act_last=False
    ):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(NONLINEARITIES[act_fn])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def outer_product(*vectors):
    for index, vector in enumerate(vectors):
        if index == 0:
            out = vector.unsqueeze(-1)
        else:
            out = out * vector.unsqueeze(1)
            out = out.view(out.shape[0], -1).unsqueeze(-1)
    return out.squeeze()


def compose_context(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
    sort_idx = torch.sort(batch_ctx, stable=True).indices

    mask_ligand = torch.cat(
        [
            torch.zeros([batch_protein.size(0)], device=batch_protein.device).bool(),
            torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
        ],
        dim=0,
    )[sort_idx]

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]

    return h_ctx, pos_ctx, batch_ctx, mask_ligand


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


def hybrid_edge_connection(ligand_pos, protein_pos, k, ligand_index, protein_index):
    dst = torch.repeat_interleave(ligand_index, len(ligand_index))
    src = ligand_index.repeat(len(ligand_index))
    mask = dst != src
    dst, src = dst[mask], src[mask]
    ll_edge_index = torch.stack([src, dst])

    ligand_protein_pos_dist = torch.unsqueeze(ligand_pos, 1) - torch.unsqueeze(protein_pos, 0)
    ligand_protein_pos_dist = torch.norm(ligand_protein_pos_dist, p=2, dim=-1)
    knn_p_idx = torch.topk(ligand_protein_pos_dist, k=k, largest=False, dim=1).indices
    knn_p_idx = protein_index[knn_p_idx]
    knn_l_idx = torch.unsqueeze(ligand_index, 1)
    knn_l_idx = knn_l_idx.repeat(1, k)
    pl_edge_index = torch.stack([knn_p_idx, knn_l_idx], dim=0)
    pl_edge_index = pl_edge_index.view(2, -1)
    return ll_edge_index, pl_edge_index


def batch_hybrid_edge_connection(x, k, mask_ligand, batch, add_p_index=False):
    batch_size = batch.max().item() + 1
    batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index = [], [], []
    with torch.no_grad():
        for i in range(batch_size):
            ligand_index = ((batch == i) & (mask_ligand == 1)).nonzero()[:, 0]
            protein_index = ((batch == i) & (mask_ligand == 0)).nonzero()[:, 0]
            ligand_pos, protein_pos = x[ligand_index], x[protein_index]
            ll_edge_index, pl_edge_index = hybrid_edge_connection(
                ligand_pos, protein_pos, k, ligand_index, protein_index
            )
            batch_ll_edge_index.append(ll_edge_index)
            batch_pl_edge_index.append(pl_edge_index)
            if add_p_index:
                all_pos = torch.cat([protein_pos, ligand_pos], 0)
                p_edge_index = knn_graph(all_pos, k=k, flow="source_to_target")
                p_edge_index = p_edge_index[:, p_edge_index[1] < len(protein_pos)]
                p_src, p_dst = p_edge_index
                all_index = torch.cat([protein_index, ligand_index], 0)
                p_edge_index = torch.stack([all_index[p_src], all_index[p_dst]], 0)
                batch_p_edge_index.append(p_edge_index)

    if add_p_index:
        edge_index = [
            torch.cat([ll, pl, p], -1)
            for ll, pl, p in zip(batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index)
        ]
    else:
        edge_index = [
            torch.cat([ll, pl], -1) for ll, pl in zip(batch_ll_edge_index, batch_pl_edge_index)
        ]
    edge_index = torch.cat(edge_index, -1)
    return edge_index


# ---------------------------------------------------------------------------
# models/uni_transformer.py
# ---------------------------------------------------------------------------
class BaseX2HAttLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_heads,
        edge_feat_dim,
        r_feat_dim,
        act_fn="relu",
        norm=True,
        ew_net_type="r",
        out_fc=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.ew_net_type = ew_net_type
        self.out_fc = out_fc

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == "r":
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == "m":
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.hv_func(kv_input)

        if self.ew_net_type == "r":
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == "m":
            e_w = self.ew_net(v[..., : self.hidden_dim])
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.0
        v = v * e_w
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)

        m = alpha.unsqueeze(-1) * v
        output = scatter_sum(m, dst, dim=0, dim_size=N)
        output = output.view(-1, self.output_dim)
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))

        output = output + h
        return output


class BaseH2XAttLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_heads,
        edge_feat_dim,
        r_feat_dim,
        act_fn="relu",
        norm=True,
        ew_net_type="r",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim
        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == "r":
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.xv_func(kv_input)
        if self.ew_net_type == "r":
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == "m":
            e_w = 1.0
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.0
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)

        m = alpha.unsqueeze(-1) * v
        output = scatter_sum(m, dst, dim=0, dim_size=N)
        return output.mean(1)


class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_heads,
        num_r_gaussian,
        edge_feat_dim,
        act_fn="relu",
        norm=True,
        num_x2h=1,
        num_h2x=1,
        r_min=0.0,
        r_max=10.0,
        num_node_types=8,
        ew_net_type="r",
        x2h_out_fc=True,
        sync_twoup=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(
            self.r_min, self.r_max, num_gaussians=num_r_gaussian
        )

        self.x2h_layers = nn.ModuleList()
        for _ in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    n_heads,
                    edge_feat_dim,
                    r_feat_dim=num_r_gaussian * 4,
                    act_fn=act_fn,
                    norm=norm,
                    ew_net_type=self.ew_net_type,
                    out_fc=self.x2h_out_fc,
                )
            )
        self.h2x_layers = nn.ModuleList()
        for _ in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    n_heads,
                    edge_feat_dim,
                    r_feat_dim=num_r_gaussian * 4,
                    act_fn=act_fn,
                    norm=norm,
                    ew_net_type=self.ew_net_type,
                )
            )

    def forward(self, h, x, edge_attr, edge_index, mask_ligand, e_w=None, fix_x=False):
        src, dst = edge_index
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr
        else:
            edge_feat = None

        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        h_in = h
        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            h_out = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w)
            h_in = h_out
        x2h_out = h_in

        new_h = h if self.sync_twoup else x2h_out
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w)
            if not fix_x:
                x = x + delta_x * mask_ligand[:, None]
            rel_x = x[dst] - x[src]
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        return x2h_out, x


class UniTransformerO2TwoUpdateGeneral(nn.Module):
    def __init__(
        self,
        num_blocks,
        num_layers,
        hidden_dim,
        n_heads=1,
        k=32,
        num_r_gaussian=50,
        edge_feat_dim=0,
        num_node_types=8,
        act_fn="relu",
        norm=True,
        cutoff_mode="radius",
        ew_net_type="r",
        num_init_x2h=1,
        num_init_h2x=0,
        num_x2h=1,
        num_h2x=1,
        r_max=10.0,
        x2h_out_fc=True,
        sync_twoup=False,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types
        self.cutoff_mode = cutoff_mode
        self.k = k
        self.ew_net_type = ew_net_type

        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_init_x2h = num_init_x2h
        self.num_init_h2x = num_init_h2x
        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        self.distance_expansion = GaussianSmearing(0.0, r_max, num_gaussians=num_r_gaussian)
        if self.ew_net_type == "global":
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)

        self.init_h_emb_layer = self._build_init_h_layer()
        self.base_block = self._build_share_blocks()

    def _build_init_h_layer(self):
        return AttentionLayerO2TwoUpdateNodeGeneral(
            self.hidden_dim,
            self.n_heads,
            self.num_r_gaussian,
            self.edge_feat_dim,
            act_fn=self.act_fn,
            norm=self.norm,
            num_x2h=self.num_init_x2h,
            num_h2x=self.num_init_h2x,
            r_max=self.r_max,
            num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type,
            x2h_out_fc=self.x2h_out_fc,
            sync_twoup=self.sync_twoup,
        )

    def _build_share_blocks(self):
        base_block = []
        for _ in range(self.num_layers):
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim,
                self.n_heads,
                self.num_r_gaussian,
                self.edge_feat_dim,
                act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h,
                num_h2x=self.num_h2x,
                r_max=self.r_max,
                num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type,
                x2h_out_fc=self.x2h_out_fc,
                sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)

    def _connect_edge(self, x, mask_ligand, batch):
        if self.cutoff_mode == "radius":
            edge_index = radius_graph(x, r=self.r_max, batch=batch, flow="source_to_target")
        elif self.cutoff_mode == "knn":
            edge_index = knn_graph(x, k=self.k, batch=batch, flow="source_to_target")
        elif self.cutoff_mode == "hybrid":
            edge_index = batch_hybrid_edge_connection(
                x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True
            )
        else:
            raise ValueError(f"Not supported cutoff mode: {self.cutoff_mode}")
        return edge_index

    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type

    def forward(self, h, x, mask_ligand, batch, return_all=False, fix_x=False):
        all_x = [x]
        all_h = [h]

        for _b_idx in range(self.num_blocks):
            edge_index = self._connect_edge(x, mask_ligand, batch)
            src, dst = edge_index

            edge_type = self._build_edge_type(edge_index, mask_ligand)
            if self.ew_net_type == "global":
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
                dist_feat = self.distance_expansion(dist)
                logits = self.edge_pred_layer(dist_feat)
                e_w = torch.sigmoid(logits)
            else:
                e_w = None

            for layer in self.base_block:
                h, x = layer(h, x, edge_type, edge_index, mask_ligand, e_w=e_w, fix_x=fix_x)
            all_x.append(x)
            all_h.append(h)

        outputs = {"x": x, "h": h}
        if return_all:
            outputs.update({"all_x": all_x, "all_h": all_h})
        return outputs


# ---------------------------------------------------------------------------
# models/molopt_score_model.py -- SinusoidalPosEmb + ScorePosNet3D
# (variance-schedule buffer setup mirrors upstream `__init__`; the diffusion
# forward-process / ELBO-loss / posterior-sampling methods are omitted --
# those are the training/sampling *procedure*, not the network architecture.
# `forward()` below is upstream's `ScorePosNet3D.forward` verbatim: one
# equivariant denoising step over composed protein+ligand context.)
# ---------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ScorePosNet3D(nn.Module):
    """Real `ScorePosNet3D` model-definition subset: atom/time embeddings +
    `uni_o2` refine net + v_inference head, exactly as upstream builds and
    calls them in `forward()`. Diffusion-schedule buffers are still
    registered (matching the real `__init__`) even though `forward()` alone
    does not consume the full noise schedule (that happens in the upstream
    training-loss / sampling methods, intentionally omitted here)."""

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):
        super().__init__()
        self.config = config

        self.hidden_dim = config["hidden_dim"]
        self.num_classes = ligand_atom_feature_dim
        if config["node_indicator"]:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)

        self.time_emb_dim = config["time_emb_dim"]
        self.time_emb_mode = config["time_emb_mode"]
        if self.time_emb_dim > 0:
            if self.time_emb_mode == "simple":
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)
            elif self.time_emb_mode == "sin":
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
                )
                self.ligand_atom_emb = nn.Linear(
                    ligand_atom_feature_dim + self.time_emb_dim, emb_dim
                )
            else:
                raise NotImplementedError
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        self.refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config["num_blocks"],
            num_layers=config["num_layers"],
            hidden_dim=config["hidden_dim"],
            n_heads=config["n_heads"],
            k=config["knn"],
            edge_feat_dim=config["edge_feat_dim"],
            num_r_gaussian=config["num_r_gaussian"],
            num_node_types=config["num_node_types"],
            act_fn=config["act_fn"],
            norm=config["norm"],
            cutoff_mode=config["cutoff_mode"],
            ew_net_type=config["ew_net_type"],
            num_x2h=config["num_x2h"],
            num_h2x=config["num_h2x"],
            r_max=config["r_max"],
            x2h_out_fc=config["x2h_out_fc"],
            sync_twoup=config["sync_twoup"],
        )
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )
        self.node_indicator = config["node_indicator"]
        self.num_timesteps = config["num_diffusion_timesteps"]

    def forward(
        self,
        protein_pos,
        protein_v,
        batch_protein,
        init_ligand_pos,
        init_ligand_v,
        batch_ligand,
        time_step=None,
        return_all=False,
        fix_x=False,
    ):
        init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        if self.time_emb_dim > 0:
            if self.time_emb_mode == "simple":
                input_ligand_feat = torch.cat(
                    [init_ligand_v, (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)],
                    -1,
                )
            elif self.time_emb_mode == "sin":
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        if self.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            init_ligand_h = torch.cat(
                [init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1
            )

        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        outputs = self.refine_net(
            h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x
        )
        final_pos, final_h = outputs["x"], outputs["h"]
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        final_ligand_v = self.v_inference(final_ligand_h)

        return {
            "pred_ligand_pos": final_ligand_pos,
            "pred_ligand_v": final_ligand_v,
            "final_h": final_h,
            "final_ligand_h": final_ligand_h,
        }


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
_PROTEIN_FEAT_DIM = 8
_LIGAND_FEAT_DIM = 6
# Mirrors the real configs/training.yml defaults (scaled down: hidden_dim
# 128->32, num_layers 9->2, n_heads 16->2, knn 32->4, num_r_gaussian 20->8),
# including time_emb_dim=0 (real default -- no time embedding branch at all).
_CONFIG = dict(
    hidden_dim=32,
    node_indicator=True,
    time_emb_dim=0,
    time_emb_mode="simple",
    num_blocks=1,
    num_layers=2,
    n_heads=2,
    knn=4,
    edge_feat_dim=4,
    # NOTE: GaussianSmearing's default fixed_offset=True ignores num_gaussians
    # and always emits exactly 20 features (real upstream quirk) -- must match.
    num_r_gaussian=20,
    num_node_types=_LIGAND_FEAT_DIM,
    act_fn="relu",
    norm=True,
    cutoff_mode="knn",
    ew_net_type="global",
    num_x2h=1,
    num_h2x=1,
    r_max=10.0,
    x2h_out_fc=True,
    sync_twoup=False,
    num_diffusion_timesteps=1000,
)


def build_targetdiff():
    return ScorePosNet3D(
        _CONFIG,
        protein_atom_feature_dim=_PROTEIN_FEAT_DIM,
        ligand_atom_feature_dim=_LIGAND_FEAT_DIM,
    ).eval()


def example_input_targetdiff():
    n_protein, n_ligand = 12, 6
    protein_pos = torch.randn(n_protein, 3)
    protein_v = torch.randn(n_protein, _PROTEIN_FEAT_DIM)
    batch_protein = torch.zeros(n_protein, dtype=torch.long)
    init_ligand_pos = torch.randn(n_ligand, 3)
    init_ligand_v = torch.randint(0, _LIGAND_FEAT_DIM, (n_ligand,))
    batch_ligand = torch.zeros(n_ligand, dtype=torch.long)
    time_step = torch.zeros(1, dtype=torch.long)
    return (
        protein_pos,
        protein_v,
        batch_protein,
        init_ligand_pos,
        init_ligand_v,
        batch_ligand,
        time_step,
    )


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("TargetDiff", "build_targetdiff", "example_input_targetdiff", 2023, "vendored"),
]
