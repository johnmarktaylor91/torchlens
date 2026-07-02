# SOURCE: vendored from https://github.com/nilotpal09/HGPflow @ main
# (hgpflow_v2/models/hgpflow_model.py + models/node_prep/{node_prep_model.py,
#  node_prep_model_mini.py} + models/hg_learner/{hg_learn_model.py,
#  iterative_refiner.py} + models/hyperedge_learner/hyperedge_model.py +
#  models/helpers/{dense.py,diffusion_transformer.py,attention.py,
#  graph_operations.py,utils.py} + utility/var_transformation.py)
#
# HGPflow: Hypergraph Particle Flow. Learns a soft hyperedge incidence matrix
# over calorimeter-topocluster/track "nodes" via an iterative refiner
# (DeepSet node updates + transformer edge updates + an incidence-prediction
# MLP), then reads out per-hyperedge kinematics/class via a second
# hyperedge-learner stage (proxy-kinematics init nets + optional transformer +
# regression/classification heads). This is the real HEP particle-flow
# reconstruction architecture from the repo; only the `mini` node-prep variant
# is vendored (the `cell_v1`/`cell_v2` variants operate on raw calorimeter
# cells instead of topoclusters and are not used by the shipped
# `configs/clic/model_configs/model_stage1_mini.yml`).
#
# Two deliberate, narrowly-scoped deviations from the real repo, both
# preserving the architecture exactly:
#   1. `enable_flash_attn` is forced to False everywhere. The real repo's own
#      `configs/clic/model_configs/model_stage1_mini.yml` sets
#      `node_prep_model.transformer.mha_config.enable_flash_attn: True` but
#      `hg_model.transformer_e.mha_config.enable_flash_attn: False`; `DiTLayer`
#      (in diffusion_transformer.py, vendored below) implements both as
#      alternate self-attention backends behind this one config flag, so using
#      the (already-shipped-in-the-repo) False branch everywhere selects the
#      real base-torch `MultiheadAttention` path instead of the optional
#      `flash_attn` package, which is not installed in this environment.
#   2. Dense/transformer widths are shrunk (192 -> 24, 384 -> 32, etc.) purely
#      for trace speed; every module, mechanism, and code path is unchanged.
#
# The `hgpflow_v2/dataset/*` loaders (which read ROOT files via uproot/awkward)
# are not vendored -- only the model. The `batch` dict the model consumes is
# fabricated here with tensors matching the exact shapes/keys the real
# `PflowDatasetMini` produces (see dataset_mini.py's `init_var_list`:
# 14 track features, 9 topo features).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# models/helpers/utils.py
# ---------------------------------------------------------------------------


def add_dims(x, ndim):
    """Adds dimensions to a tensor to match the shape of another tensor."""
    if (dim_diff := ndim - x.dim()) < 0:
        raise ValueError(f"Target ndim ({ndim}) is larger than input ndim ({x.dim()})")

    if dim_diff > 0:
        x = x.view(x.shape[0], *dim_diff * (1,), *x.shape[1:])

    return x


def masked_softmax(x, mask, dim=-1):
    """Applies softmax over a tensor without including padded elements."""
    if mask is not None:
        mask = add_dims(mask, x.dim())
        x = x.masked_fill(mask, -torch.inf)

    x = F.softmax(x, dim=dim)

    if mask is not None:
        x = x.masked_fill(mask, 0)

    return x


def merge_masks(q_mask, kv_mask, attn_mask, q_shape, k_shape, device):
    """Create a full attention mask which incoporates the padding information.

    Using pytorch transformer convention:
        False: Real node
        True:  Zero padded
    """
    merged_mask = None

    if q_mask is not None or kv_mask is not None:
        if q_mask is None:
            q_mask = torch.full(q_shape[:-1], False, device=device)
        if kv_mask is None:
            kv_mask = torch.full(k_shape[:-1], False, device=device)
        merged_mask = q_mask.unsqueeze(-1) | kv_mask.unsqueeze(-2)

    if attn_mask is not None:
        merged_mask = attn_mask if merged_mask is None else attn_mask | merged_mask

    return merged_mask


def attach_context(x, context):
    """Concatenates a context tensor to an input tensor with considerations for
    broadcasting."""
    if context is None:
        raise RuntimeError("Expected context is missing from forward pass")

    if (dim_diff := x.dim() - context.dim()) < 0:
        raise ValueError(
            f"Provided context has more dimensions ({context.dim()}) than inputs ({x.dim()})"
        )

    if dim_diff > 0:
        context = add_dims(context, x.dim())
        context = context.expand(*x.shape[:-1], -1)

    return torch.cat([x, context], dim=-1)


def padded_to_packed(seq, inv_mask):
    # inv_mask: True for valid tokens
    seqlens = inv_mask.sum(dim=-1)
    maxlen = seqlens.max()
    culens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return seq[inv_mask], culens, maxlen


def packed_to_padded(unpadded_seq, inv_mask):
    # inv_mask: True for valid tokens
    shape = (*inv_mask.shape, unpadded_seq.shape[-1])
    out = torch.zeros(shape, dtype=unpadded_seq.dtype, device=unpadded_seq.device)
    out[inv_mask] = unpadded_seq
    return out


# ---------------------------------------------------------------------------
# models/helpers/dense.py
# ---------------------------------------------------------------------------


class Dense(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layers,
        activation="LeakyReLU",
        final_activation=None,
        norm_layer=None,
        norm_final_layer=False,
        dropout=0.0,
        context_dim=0,
    ):
        """A simple fully connected feed forward neural network, which can take
        in additional contextual information."""
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim

        node_list = [input_dim + context_dim, *hidden_layers, output_dim]

        layers = []

        num_layers = len(node_list) - 1
        for i in range(num_layers):
            is_final_layer = i == num_layers - 1

            if norm_layer and (norm_final_layer or not is_final_layer):
                layers.append(getattr(nn, norm_layer)(node_list[i], elementwise_affine=False))

            if dropout and (norm_final_layer or not is_final_layer):
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(node_list[i], node_list[i + 1]))

            if not is_final_layer:
                layers.append(getattr(nn, activation)())
            elif final_activation:
                layers.append(getattr(nn, final_activation)())

        self.net = nn.Sequential(*layers)

    def forward(self, x, context=None):
        if self.context_dim:
            x = attach_context(x, context)
        return self.net(x)


# ---------------------------------------------------------------------------
# models/helpers/attention.py
# ---------------------------------------------------------------------------


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        attention,
        edge_embed_dim=0,
        q_dim=None,
        k_dim=None,
        v_dim=None,
        out_proj=True,
        update_edges=False,
    ):
        """Generic multihead attention."""
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        if edge_embed_dim % num_heads != 0:
            raise ValueError(
                f"edge_embed_dim {edge_embed_dim} must be divisible by num_heads {num_heads}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.out_proj = out_proj
        self.edge_embed_dim = edge_embed_dim
        self.edge_head_dim = edge_embed_dim // num_heads
        self.k_dim = k_dim or embed_dim
        self.v_dim = v_dim or embed_dim
        self.scale = (
            embed_dim**0.5 / num_heads**0.5
        )  # kept semantically equivalent to math.sqrt(head_dim)
        self.scale = (embed_dim // num_heads) ** 0.5
        self.update_edges = update_edges

        if q_dim is None:
            self.q_dim = self.embed_dim
        else:
            self.q_dim = q_dim
            assert (self.q_dim == self.num_heads) or (self.out_proj == True), (  # noqa: E712 (kept for fidelity)
                f"q_dim {self.q_dim} must be equal to num_heads {self.num_heads} "
                f"if out_proj is False"
            )

        self.attention = attention

        self.linear_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_k = nn.Linear(self.k_dim, self.embed_dim)
        self.linear_v = nn.Linear(self.v_dim, self.embed_dim)
        if self.edge_embed_dim > 0:
            self.linear_e = nn.Linear(self.edge_embed_dim, self.num_heads)
            self.linear_g = nn.Linear(self.edge_embed_dim, self.num_heads)
            if self.update_edges:
                self.linear_e_out = nn.Linear(self.num_heads, self.edge_embed_dim)
            else:
                self.register_buffer("linear_e_out", None)
        if self.out_proj:
            self.linear_out = nn.Linear(self.embed_dim, self.q_dim)
        else:
            self.register_buffer("linear_out", None)

    def input_projections(self, q, k, v):
        """Perform input linear projections, output shapes are (B,L,H,HD)."""
        shape = (k.shape[0], -1, self.num_heads, self.head_dim)
        q_proj = self.linear_q(q).view(shape).transpose(1, 2)
        k_proj = self.linear_k(k).view(shape).transpose(1, 2)
        v_proj = self.linear_v(v).view(shape).transpose(1, 2)
        return q_proj, k_proj, v_proj

    def forward(
        self,
        q,
        k=None,
        v=None,
        edges=None,
        q_mask=None,
        kv_mask=None,
        attn_mask=None,
        attn_bias=None,
    ):
        """Full forward pass through the model."""
        if k is None:
            k = q
            if kv_mask is None:
                kv_mask = q_mask
        v = v if v is not None else k

        b_size, _seq_len, _features = q.shape

        attn_mask = merge_masks(q_mask, kv_mask, attn_mask, q.shape, k.shape, q.device)

        q_proj, k_proj, v_proj = self.input_projections(q, k, v)

        if edges is not None:
            e = self.linear_e(edges)
            g = torch.sigmoid(self.linear_g(edges))
            attn_bias = e if attn_bias is None else attn_bias + e

        attn_weights = self.attention(
            q_proj, k_proj, self.scale, attn_mask, attn_bias, self.update_edges
        )
        if self.update_edges:
            attn_weights, attn_scores = attn_weights

        if edges is not None:
            attn_weights = attn_weights * g.permute(0, 3, 1, 2)

        out = torch.matmul(attn_weights, v_proj)
        out = out.transpose(1, 2).contiguous().view(b_size, -1, self.embed_dim)

        edge_out = None
        if self.update_edges:
            edge_out = self.linear_e_out(attn_scores.permute(0, 2, 3, 1))

        if self.out_proj:
            out = self.linear_out(out)

        if edges is not None:
            return out, edge_out

        return out


class ScaledDotProductAttention(nn.Module):
    """Scaled dot product attention, commonly used in transformers."""

    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q,
        k,
        scale,
        mask=None,
        attn_bias=None,
        return_scores=False,
    ):
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attn_bias is not None:
            scores = scores + attn_bias.permute(0, 3, 1, 2)

        scores = self.dropout(scores)

        attention_weights = masked_softmax(scores, mask)

        if return_scores:
            return attention_weights, scores

        return attention_weights


# ---------------------------------------------------------------------------
# models/helpers/diffusion_transformer.py
# ---------------------------------------------------------------------------


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTLayer(nn.Module):
    def __init__(self, embed_dim, context_dim, mha_config, dense_config=None):
        super().__init__()
        self.embed_dim = embed_dim

        self.enable_flash_attn = mha_config.get("enable_flash_attn", False)
        mha_config.pop("enable_flash_attn", None)
        if self.enable_flash_attn:
            # NOTE (menagerie staging): the real repo imports
            # `.attention_flash_varlen.MultiheadFlashAttentionVarLen` here,
            # which needs the optional `flash_attn` package. Every staged
            # config below forces `enable_flash_attn=False`, so this branch
            # is unreachable in the staged trace but is kept verbatim as a
            # faithful copy of the real DiTLayer.
            from .attention_flash_varlen import MultiheadFlashAttentionVarLen  # noqa: F401

            raise RuntimeError("flash attention path not vendored for menagerie staging")
        else:
            mha_config["attention"] = ScaledDotProductAttention()
            self.mha = MultiheadAttention(embed_dim, **mha_config)

        if dense_config:
            dense_config["input_dim"] = embed_dim
            dense_config["output_dim"] = embed_dim
            self.dense = Dense(**dense_config)
        else:
            self.register_buffer("dense", None)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(context_dim, 6 * embed_dim, bias=True)
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.adaLN_modulation[1].weight.data.zero_()
        self.adaLN_modulation[1].bias.data.zero_()

    def forward(
        self, q, q_mask=None, k=None, kv_mask=None, context=None, attn_mask=None, attn_bias=None
    ):
        """
        if k is provided, then we will have cross-attention
        """

        if self.enable_flash_attn:
            assert k is None, "Flash attention does not support cross-attention"

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            context
        ).chunk(6, dim=1)
        if k is None:  # self-attention
            if self.enable_flash_attn:
                if q_mask is None:
                    not_q_mask = torch.full(q.shape[:-1], True, dtype=torch.bool, device=q.device)
                else:
                    not_q_mask = ~q_mask
                q_modulated = modulate(self.norm1(q), shift_msa, scale_msa)
                q_packed, culens, maxlen = padded_to_packed(q_modulated, not_q_mask)
                q_attn = self.mha(q_packed, culens, maxlen)
                q_attn = packed_to_padded(q_attn, not_q_mask)
            else:
                q_attn = self.mha(
                    q=modulate(self.norm1(q), shift_msa, scale_msa),
                    q_mask=q_mask,
                    attn_mask=attn_mask,
                    attn_bias=attn_bias,
                )

        else:  # cross-attention
            q_attn = self.mha(
                q=q,
                k=modulate(self.norm1(k), shift_msa, scale_msa),
                q_mask=q_mask,
                kv_mask=kv_mask,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )

        q = q + gate_msa.unsqueeze(1) * q_attn

        if self.dense:
            q_mlp = self.dense(modulate(self.norm2(q), shift_mlp, scale_mlp), context)
            q = q + gate_mlp.unsqueeze(1) * q_mlp

        return q


class DiTEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_layers,
        mha_config,
        dense_config=None,
        context_dim=0,
        out_dim=0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

        self.layers = nn.ModuleList(
            [
                DiTLayer(
                    embed_dim,
                    context_dim,
                    dict(mha_config),
                    dense_config,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        if self.out_dim:
            self.final_linear = nn.Linear(self.embed_dim, self.out_dim)

    def forward(self, q, **kwargs):
        for layer in self.layers:
            q = layer(q, **kwargs)
        q = self.final_norm(q)

        if self.out_dim:
            q = self.final_linear(q)
        return q


# ---------------------------------------------------------------------------
# models/helpers/graph_operations.py
# ---------------------------------------------------------------------------


def custom_update_all(edge_fn, node_fn, **kwargs):
    """
    kwargs with prefix 'efn_' will be passed to edge_fn
    kwargs with prefix 'nfn_' will be passed to node_fn
    """
    edge_fn_kwargs = {}
    node_fn_kwargs = {}
    for k, v in kwargs.items():
        if k.startswith("efn_"):
            edge_fn_kwargs[k[4:]] = v
        elif k.startswith("nfn_"):
            node_fn_kwargs[k[4:]] = v
        else:
            raise ValueError("invalid kwarg prefix")

    # B, N, M, D
    edge_msg = edge_fn(**edge_fn_kwargs)

    # B, N, D
    node_fn_kwargs["edge_msg"] = edge_msg
    dst_feat = node_fn(**node_fn_kwargs)

    return dst_feat


def custom_copy_u(src_feat, num_nodes_dst, src_mask=None, dst_mask=None, edge_mask=None):
    """Function to copy the src features to the edges."""
    M = num_nodes_dst
    msg = src_feat.unsqueeze(2).repeat(1, 1, M, 1)
    if edge_mask is not None:
        msg = msg * edge_mask.unsqueeze(-1)
    return msg


def custom_u_mul_e(
    src_feat, num_nodes_dst, edge_feat, src_mask=None, dst_mask=None, edge_mask=None
):
    """Function to copy the src feature multiplied by an edge feature to the edges."""
    M = num_nodes_dst
    msg = src_feat.unsqueeze(2).repeat(1, 1, M, 1) * edge_feat
    if edge_mask is not None:
        msg = msg * edge_mask.unsqueeze(-1)
    return msg


def custom_sum_mailbox(edge_msg, edge_mask=None):
    """Sum incoming messages over the source-node dimension."""
    return edge_msg.sum(dim=1)


# ---------------------------------------------------------------------------
# utility/var_transformation.py
# ---------------------------------------------------------------------------


class VarTransformation:
    """
    trans: tranforming the quantities
        eg. x -> log(x), pow(e,m) etc
    scale: scaling the quantities
        eg. x -> (x - mean(x)) / std(x)
    forward: trans + scale
    """

    def __init__(self, config):
        self.config = config
        self.scale_mode = config["scale_mode"]
        self.transformation = config["transformation"]

    def trans(self, x):
        if self.transformation is None:
            return x
        if self.transformation == "pow(x,m)":
            return x ** self.config["m"]
        return x

    def inv_trans(self, x):
        if self.transformation is None:
            return x
        if self.transformation == "pow(x,m)":
            return x ** (1 / self.config["m"])
        return x

    def scale(self, x):
        if self.scale_mode is None:
            return x
        elif self.scale_mode == "min_max":
            if self.config["range"] == [0, 1]:
                return (x - self.config["min"]) / (self.config["max"] - self.config["min"])
            elif self.config["range"] == [-1, 1]:
                return (x - self.config["min"]) / (self.config["max"] - self.config["min"]) * 2 - 1
        elif self.scale_mode == "standard":
            return (x - self.config["mean"]) / self.config["std"]
        return x

    def inv_scale(self, x):
        if self.scale_mode is None:
            return x
        elif self.scale_mode == "min_max":
            if self.config["range"] == [0, 1]:
                return x * (self.config["max"] - self.config["min"]) + self.config["min"]
            elif self.config["range"] == [-1, 1]:
                return (x + 1) / 2 * (self.config["max"] - self.config["min"]) + self.config["min"]
        elif self.scale_mode == "standard":
            return x * self.config["std"] + self.config["mean"]
        return x

    def forward(self, x):
        x = self.trans(x)
        x = self.scale(x)
        return x

    def inverse(self, x):
        x = self.inv_scale(x)
        x = self.inv_trans(x)
        return x


# ---------------------------------------------------------------------------
# models/node_prep/node_prep_model_mini.py + node_prep_model.py
# ---------------------------------------------------------------------------


class NodePrepModelMini(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.track_init_net = Dense(**config["track_init_net"])
        self.topo_init_net = Dense(**config["topo_init_net"])

        self.node_transformer = DiTEncoder(**config["transformer"])

    def forward(self, batch):
        track_feat = self.track_init_net(batch["track"]["feat0"])
        topo_feat = self.topo_init_net(batch["topo"]["feat0"])

        track_mask = batch["node"]["is_track"].unsqueeze(-1)
        topo_mask = batch["node"]["is_topo"].unsqueeze(-1)
        node_feat = track_feat * track_mask + topo_feat * topo_mask

        # transformer
        node_global = node_feat.mean(dim=1)  # there is no padding

        # identify what's track and what's topo
        node_feat = torch.cat([node_feat, track_mask.float(), topo_mask.float()], dim=-1)

        node_feat = self.node_transformer(q=node_feat, context=node_global)

        if self.config.get("add_skip_feat", False):
            node_feat = torch.cat([node_feat, batch["node"]["skip_feat0"]], dim=-1)

        return node_feat


class NodePrepModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        model_dict = {
            "mini": NodePrepModelMini,
            # 'cell_v1'/'cell_v2' variants intentionally not vendored (raw
            # calorimeter-cell node prep -- not used by the mini config path).
        }

        self.model = model_dict[config["type"]](config)

    def forward(self, batch):
        return self.model(batch)


# ---------------------------------------------------------------------------
# models/hg_learner/iterative_refiner.py
# ---------------------------------------------------------------------------


class OutCatLinear(nn.Module):
    def __init__(self, d_e, d_n, d_i, d_out):
        super().__init__()
        self.proj_e = nn.Linear(d_e, d_out)
        self.proj_n = nn.Linear(d_n, d_out)
        self.proj_i = nn.Linear(d_i, d_out)

    def forward(self, inputs):
        e_t, n_t, i_t = inputs
        o0 = self.proj_n(n_t).unsqueeze(1)
        o1 = self.proj_e(e_t).unsqueeze(2)
        o2 = self.proj_i(i_t.unsqueeze(3))
        return o0 + o1 + o2


class DeepSetLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features, out_features)
        self.layer2 = nn.Linear(in_features, out_features)

    def forward(self, x):
        x0 = self.layer1(x)
        x1 = self.layer2(x - x.mean(dim=1, keepdim=True))
        x = x0 + x1
        return x


class DeepSet(nn.Module):
    def __init__(self, d_in, d_hids):
        super().__init__()
        layers = []
        layers.append(DeepSetLayer(d_in, d_hids[0]))
        for i in range(1, len(d_hids)):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LayerNorm(d_hids[i - 1]))
            layers.append(DeepSetLayer(d_hids[i - 1], d_hids[i]))

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)


class HypergraphRefiner(nn.Module):
    def __init__(self, dim, config):
        super().__init__()

        self.config = config

        self.mlp_n = DeepSet(3 * dim, config["deepset_n"]["hidden_layers"])
        self.transformer_e = DiTEncoder(**config["transformer_e"])

        self.norm_pre_n = nn.LayerNorm(3 * dim)
        self.norm_pre_e = nn.LayerNorm(2 * dim)
        self.norm_n = nn.LayerNorm(dim)
        self.norm_e = nn.LayerNorm(dim)

        self.mlp_incidence = nn.Sequential(
            OutCatLinear(dim, dim, 1, dim), nn.ReLU(inplace=True), nn.Linear(dim, 1)
        )
        self.edge_indicator = Dense(**config["edge_indicator"])

    def get_track_eye(self, track_mask, i_t_shape):
        b_size, e_size, n_size = i_t_shape

        track_eye = torch.eye(n_size, device=track_mask.device).unsqueeze(0).repeat(b_size, 1, 1)

        track_eye = torch.nn.functional.pad(track_eye, (0, 0, 0, e_size - n_size))
        track_eye = track_eye * track_mask.unsqueeze(1)

        ch_mask_from_tracks = track_eye.sum(dim=2, keepdim=True).bool()

        return track_eye, ch_mask_from_tracks

    def set_track(self, i_t, track_mask, track_eye):
        i_t = i_t * (~track_mask).unsqueeze(1)
        i_t = i_t + track_eye

        return i_t

    def forward(self, inputs, e_t, n_t, i_t, track_mask, track_eye, ch_mask_from_tracks):
        b_size, e_size, n_size = i_t.shape

        i_t = self.mlp_incidence((e_t, n_t, i_t)).squeeze(3)

        i_t = nn.Softmax(dim=1)(i_t)

        i_t = self.set_track(i_t, track_mask, track_eye)

        i_t_sum = i_t.sum(dim=2, keepdim=True)
        e_ind_logit = self.edge_indicator(torch.cat([e_t, i_t_sum], dim=2))

        e_ind_logit = e_ind_logit * (~ch_mask_from_tracks) + ch_mask_from_tracks * 1e6

        e_ind = F.sigmoid(e_ind_logit)

        im_t = i_t * e_ind

        updates_e = torch.einsum("ben,bnd->bed", im_t, n_t)
        e_t = self.transformer_e(q=torch.cat([e_t, updates_e], dim=-1), context=inputs.mean(dim=1))

        updates_n = torch.einsum("ben,bed->bnd", im_t, e_t)
        n_t = self.norm_n(
            n_t + self.mlp_n(self.norm_pre_n(torch.cat([inputs, n_t, updates_n], dim=-1)))
        )

        pred_is_charged = track_eye.sum(dim=2).bool()
        pred = (i_t, e_ind_logit.squeeze(-1), pred_is_charged)

        return pred, e_t, n_t, i_t


class IterativeRefiner(nn.Module):
    def __init__(self, config, max_edges):
        super().__init__()
        self.config = config
        self.n_edges = max_edges

        self.T = self.config["T_TOTAL"]
        self.d_in = self.config["d_in"]
        self.d_hid = self.config["d_hid"]

        self.t_backprops_last = [False] * (self.T - 1) + [True]

        self.proj_inputs = nn.Linear(self.d_in, self.d_hid)
        self.refiner = HypergraphRefiner(self.d_hid, config)

        self.edges_mu = nn.Parameter(torch.randn(1, 1, self.d_hid))
        self.edges_logsigma = nn.Parameter(torch.zeros(1, 1, self.d_hid))
        nn.init.xavier_uniform_(self.edges_logsigma)

    def get_initial(self, inputs, track_mask, n_edges=None):
        b, n_v, _, device = *inputs.shape, inputs.device
        n_e = n_edges if n_edges is not None else self.n_edges

        mu = self.edges_mu.expand(b, n_e, -1)
        sigma = self.edges_logsigma.exp().expand(b, n_e, -1)
        e_t = mu + sigma * torch.randn(mu.shape, device=device)

        v_t = self.proj_inputs(inputs)
        i_t = torch.zeros(b, n_e, n_v, device=device) + 1.0 / self.n_edges

        track_eye, ch_mask_from_tracks = self.refiner.get_track_eye(track_mask, i_t.shape)
        i_t = self.refiner.set_track(i_t, track_mask, track_eye)
        return e_t, v_t, i_t, track_eye, ch_mask_from_tracks

    def refine(
        self, inputs, e_t, v_t, i_t, track_mask, track_eye, ch_mask_from_tracks, t_backprops
    ):
        inputs = self.proj_inputs(inputs)
        pred_bp = []

        for t, do_bp in enumerate(t_backprops):
            if not do_bp:
                with torch.no_grad():
                    _, e_t, v_t, i_t = self.refiner(
                        inputs, e_t, v_t, i_t, track_mask, track_eye, ch_mask_from_tracks
                    )
            else:
                p, e_t, v_t, i_t = self.refiner(
                    inputs, e_t, v_t, i_t, track_mask, track_eye, ch_mask_from_tracks
                )
                pred_bp.append((t, p))

        return pred_bp, (e_t, v_t, i_t)

    def forward(self, inputs, track_mask):
        e_t, v_t, i_t, track_eye, ch_mask_from_tracks = self.get_initial(inputs, track_mask)
        return self.refine(
            inputs, e_t, v_t, i_t, track_mask, track_eye, ch_mask_from_tracks, self.t_backprops_last
        )


class HGLearnModel(nn.Module):
    def __init__(self, config, max_edges):
        super().__init__()
        self.config = config

        model_dict = {
            "iterative_refiner": IterativeRefiner,
        }
        self.model = model_dict[config["type"]](config, max_edges)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# ---------------------------------------------------------------------------
# models/hyperedge_learner/hyperedge_model.py
# ---------------------------------------------------------------------------


class HyperedgeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.proxy_ch_kin_init_net = Dense(**config["proxy_ch_kin_init_net"])
        self.proxy_neut_kin_init_net = Dense(**config["proxy_neut_kin_init_net"])
        self.proxy_em_frac_init_net = Dense(**config["proxy_em_frac_init_net"])
        self.e_t_init_net = Dense(**config["e_t_init_net"])
        self.inc_times_node_feat_init_net = Dense(**config["inc_times_node_feat_init_net"])

        if "transformer" in config:
            self.transformer = DiTEncoder(**config["transformer"])
            self.ind_threshold = config["ind_threshold"]

        self.ch_class_net = Dense(**config["class_nets"]["ch_class_net"])
        self.neut_class_net = Dense(**config["class_nets"]["neut_class_net"])

        if config.get("kin_nets", None) is not None:
            if "ch_kin_net" in config["kin_nets"]:
                self.ch_kin_net = Dense(**config["kin_nets"]["ch_kin_net"])
            elif "ch_pt_net" in config["kin_nets"]:
                self.ch_pt_net = Dense(**config["kin_nets"]["ch_pt_net"])

            if "neut_kin_net" in config["kin_nets"]:
                self.neut_kin_net = Dense(**config["kin_nets"]["neut_kin_net"])
            elif "neut_ke_net" in config["kin_nets"]:
                self.neut_ke_net = Dense(**config["kin_nets"]["neut_ke_net"])

    def get_part_init_feat(
        self, proxy_kin, proxy_is_charged, e_t, inc_times_node_feat, proxy_em_frac
    ):
        ch_proxy_kin, neut_proxy_kin = proxy_kin

        proxy_pt = ch_proxy_kin[..., 0].unsqueeze(-1)
        proxy_ke = neut_proxy_kin[..., 0].unsqueeze(-1)

        proxy_eta = ch_proxy_kin[..., 1].unsqueeze(-1) * proxy_is_charged.unsqueeze(
            -1
        ) + neut_proxy_kin[..., 1].unsqueeze(-1) * (~proxy_is_charged.unsqueeze(-1))

        proxy_phi = ch_proxy_kin[..., 2].unsqueeze(-1) * proxy_is_charged.unsqueeze(
            -1
        ) + neut_proxy_kin[..., 2].unsqueeze(-1) * (~proxy_is_charged.unsqueeze(-1))
        proxy_cosphi = torch.cos(proxy_phi)
        proxy_sinphi = torch.sin(proxy_phi)

        proxy_ch_kin_inp = torch.cat([proxy_pt, proxy_eta, proxy_cosphi, proxy_sinphi], dim=-1)

        proxy_neut_kin_inp = torch.cat([proxy_ke, proxy_eta, proxy_cosphi, proxy_sinphi], dim=-1)

        proxy_ch_kin_init = self.proxy_ch_kin_init_net(proxy_ch_kin_inp)
        proxy_neut_kin_init = self.proxy_neut_kin_init_net(proxy_neut_kin_inp)
        proxy_kin_init = proxy_ch_kin_init * proxy_is_charged.unsqueeze(
            -1
        ) + proxy_neut_kin_init * (~proxy_is_charged.unsqueeze(-1))

        proxy_em_frac_init = self.proxy_em_frac_init_net((proxy_em_frac.unsqueeze(-1) * 2) - 1)
        e_t_init = self.e_t_init_net(e_t)
        i_t_times_node_feat_init = self.inc_times_node_feat_init_net(
            torch.clamp(inc_times_node_feat, -1, 1)
        )

        part_init_feat = torch.cat(
            [proxy_kin_init, proxy_em_frac_init, e_t_init, i_t_times_node_feat_init], dim=-1
        )

        return part_init_feat, (proxy_pt, proxy_ke, proxy_eta, proxy_phi)

    def forward(
        self,
        proxy_kin,
        proxy_is_charged,
        e_t,
        inc_times_node_feat,
        proxy_em_frac,
        node_feat_sum,
        ind,
    ):
        part_feat, (proxy_pt, proxy_ke, proxy_eta, proxy_phi) = self.get_part_init_feat(
            proxy_kin, proxy_is_charged, e_t, inc_times_node_feat, proxy_em_frac
        )

        if hasattr(self, "transformer"):
            not_part_mask = ind < self.ind_threshold
            part_feat = self.transformer(
                q=torch.cat(
                    [part_feat, proxy_is_charged.unsqueeze(-1), ~proxy_is_charged.unsqueeze(-1)],
                    dim=-1,
                ),
                q_mask=not_part_mask,
                context=node_feat_sum,
            )

        if hasattr(self, "ch_kin_net"):
            ch_del_kin = self.ch_kin_net(part_feat)

            ch_pred_pt = proxy_pt + ch_del_kin[..., 0:1]
            ch_pred_eta = proxy_eta + ch_del_kin[..., 1:2]
            ch_pred_phi = proxy_phi + ch_del_kin[..., 2:3]

            ch_pred_kin = torch.cat([ch_pred_pt, ch_pred_eta, ch_pred_phi], dim=-1)

        elif hasattr(self, "ch_pt_net"):
            ch_del_pt = self.ch_pt_net(part_feat)
            ch_pred_pt = proxy_pt + ch_del_pt

            ch_pred_kin = torch.cat([ch_pred_pt, proxy_eta, proxy_phi], dim=-1)

        else:
            ch_pred_kin = torch.cat([proxy_pt, proxy_eta, proxy_phi], dim=-1)

        if hasattr(self, "neut_kin_net"):
            neut_del_kin = self.neut_kin_net(part_feat)

            neut_pred_ke = proxy_ke + neut_del_kin[..., 0:1]
            neut_pred_eta = proxy_eta + neut_del_kin[..., 1:2]
            neut_pred_phi = proxy_phi + neut_del_kin[..., 2:3]

            neut_pred_kin = torch.cat([neut_pred_ke, neut_pred_eta, neut_pred_phi], dim=-1)

        elif hasattr(self, "neut_ke_net"):
            neut_del_ke = self.neut_ke_net(part_feat)
            neut_pred_ke = proxy_ke + neut_del_ke

            neut_pred_kin = torch.cat([neut_pred_ke, proxy_eta, proxy_phi], dim=-1)

        else:
            neut_pred_kin = torch.cat([proxy_ke, proxy_eta, proxy_phi], dim=-1)

        pred_kin = (ch_pred_kin, neut_pred_kin)

        ch_class_logits = self.ch_class_net(part_feat)
        neut_class_logits = self.neut_class_net(part_feat)
        class_logits = (ch_class_logits, neut_class_logits)

        return pred_kin, class_logits


# ---------------------------------------------------------------------------
# models/hgpflow_model.py
# ---------------------------------------------------------------------------


class HGPFlowModel(nn.Module):
    def __init__(self, config_v, config_ms1, config_ms2, class_mass_dict):
        super().__init__()

        self.epsilon = 1e-8
        self.config_ms1 = config_ms1
        self.config_ms2 = config_ms2
        self.class_mass_dict = class_mass_dict
        self.max_particles = config_v["max_particles"]

        self.node_prep_model = NodePrepModel(config_ms1["node_prep_model"])
        self.hg_model = HGLearnModel(config_ms1["hg_model"], self.max_particles)
        if config_ms2 is not None:
            self.hyperedge_model = HyperedgeModel(config_ms2["hyperedge_model"])

        self.transform_funcs = {
            "pt": VarTransformation(config_v["transformation_dict"]["pt"]),
            "e": VarTransformation(config_v["transformation_dict"]["e"]),
            "eta": VarTransformation(config_v["transformation_dict"]["eta"]),
        }

    def disable_gradients(self, component):
        if component == "hg_learner":
            for p in self.node_prep_model.parameters():
                p.requires_grad = False
            for p in self.hg_model.parameters():
                p.requires_grad = False
        elif component == "hyperedge_learner":
            for p in self.hyperedge_model.parameters():
                p.requires_grad = False
        else:
            raise ValueError(f"Component {component} not recognized")

    def infer(self, batch):
        with torch.no_grad():
            (pred_inc, pred_ind, pred_is_charged), proxy_kin, pred_kin, pred_class_logits = (
                self.forward(batch)
            )

            ch_pred_class = torch.argmax(pred_class_logits[0], dim=-1)
            neut_pred_class = torch.argmax(pred_class_logits[1], dim=-1) + 3
            pred_class = ch_pred_class * pred_is_charged + neut_pred_class * (~pred_is_charged)

            proxy_ptetaphi_raw = self.get_ptetaphi_raw_from_kin(
                proxy_kin, pred_class, unnormalize=True
            )
            pred_ptetaphi_raw = self.get_ptetaphi_raw_from_kin(
                pred_kin, pred_class, unnormalize=True
            )

            return (
                (pred_inc, pred_ind, pred_is_charged),
                proxy_ptetaphi_raw,
                pred_ptetaphi_raw,
                pred_class,
            )

    def forward(self, batch):
        (
            (pred_inc, pred_ind, pred_is_charged),
            (proxy_kin, proxy_is_charged, proxy_em_frac, e_t, inc_times_node_feat, node_feat_sum),
        ) = self.forward_pre_stage2(batch)

        pred_kin, pred_class_logits = self.hyperedge_model(
            proxy_kin,
            proxy_is_charged,
            e_t,
            inc_times_node_feat,
            proxy_em_frac,
            node_feat_sum,
            pred_ind,
        )

        return (pred_inc, pred_ind, pred_is_charged), proxy_kin, pred_kin, pred_class_logits

    def forward_pre_stage2(self, batch):
        node_feat = self.node_prep_model(batch)

        preds_list, (e_t, _, _) = self.hg_model(node_feat, batch["node"]["is_track"])
        pred_inc, pred_ind_logit, pred_is_charged = preds_list[-1][1]
        pred_ind = torch.sigmoid(pred_ind_logit)

        proxy_kin, proxy_is_charged, proxy_em_frac = self.compute_proxies(batch, pred_inc)

        # (b, n_hyperedge, n_node) * (b, n_node, d_hid) -> (b, n_hyperedge, d_hid)
        inc_times_node_feat = torch.bmm(pred_inc, node_feat)
        node_feat_sum = node_feat.sum(dim=1)

        return (pred_inc, pred_ind, pred_is_charged), (
            proxy_kin,
            proxy_is_charged,
            proxy_em_frac,
            e_t,
            inc_times_node_feat,
            node_feat_sum,
        )

    def compute_proxies(self, batch, inc):
        """
        Args:
            inc: (B, n_hyperedge, n_node) (normalized and with track hard coded)
        Returns proxy (ch_pt, eta, phi, neut_e)
        """
        bs, n_hedge, n_node = inc.size()

        inc_energy_raw = inc * batch["topo"]["e_raw"].unsqueeze(-1).permute(0, 2, 1)

        inc_energy_raw = inc_energy_raw * batch["node"]["is_topo"].unsqueeze(1)
        inc = inc_energy_raw / (inc_energy_raw.sum(dim=2, keepdim=True) + 1e-8)

        track_eye = torch.eye(n_node, device=inc.device).unsqueeze(0).repeat(bs, 1, 1)
        track_eye = torch.nn.functional.pad(track_eye, (0, 0, 0, n_hedge - n_node))
        track_eye = track_eye * batch["node"]["is_track"].unsqueeze(1)

        proxy_is_charged = track_eye.sum(dim=2).bool()

        charged_proxy_pt = custom_update_all(
            custom_copy_u,
            custom_sum_mailbox,
            efn_src_feat=batch["track"]["pt"].unsqueeze(-1),
            efn_src_mask=batch["node"]["is_track"],
            efn_num_nodes_dst=n_hedge,
            efn_dst_mask=None,
            efn_edge_mask=track_eye.permute(0, 2, 1),
        )

        charged_proxy_eta = custom_update_all(
            custom_copy_u,
            custom_sum_mailbox,
            efn_src_feat=batch["track"]["eta"].unsqueeze(-1),
            efn_src_mask=batch["node"]["is_track"],
            efn_num_nodes_dst=n_hedge,
            efn_dst_mask=None,
            efn_edge_mask=track_eye.permute(0, 2, 1),
        )

        charged_proxy_phi = custom_update_all(
            custom_copy_u,
            custom_sum_mailbox,
            efn_src_feat=batch["track"]["phi"].unsqueeze(-1),
            efn_src_mask=batch["node"]["is_track"],
            efn_num_nodes_dst=n_hedge,
            efn_dst_mask=None,
            efn_edge_mask=track_eye.permute(0, 2, 1),
        )

        charged_proxy_kin = torch.cat(
            [charged_proxy_pt, charged_proxy_eta, charged_proxy_phi], dim=-1
        ) * proxy_is_charged.unsqueeze(-1)

        node_topo_mask = batch["node"]["is_topo"].bool()
        node_topo_to_hedge_mask = (
            batch["node"]["is_topo"].unsqueeze(-1).repeat(1, 1, n_hedge).bool()
        )

        neut_proxy_ke_raw = inc_energy_raw.sum(dim=2, keepdim=True)

        neut_proxy_eta_raw = custom_update_all(
            custom_u_mul_e,
            custom_sum_mailbox,
            efn_src_feat=batch["topo"]["eta_raw"].unsqueeze(-1),
            efn_src_mask=node_topo_mask,
            efn_num_nodes_dst=n_hedge,
            efn_dst_mask=None,
            efn_edge_feat=inc.permute(0, 2, 1).unsqueeze(-1),
            efn_edge_mask=node_topo_to_hedge_mask,
            nfn_edge_mask=node_topo_to_hedge_mask,
        )

        neut_proxy_cosphi = custom_update_all(
            custom_u_mul_e,
            custom_sum_mailbox,
            efn_src_feat=torch.cos(batch["topo"]["phi"]).unsqueeze(-1),
            efn_src_mask=node_topo_mask,
            efn_num_nodes_dst=n_hedge,
            efn_dst_mask=None,
            efn_edge_feat=inc.permute(0, 2, 1).unsqueeze(-1),
            efn_edge_mask=node_topo_to_hedge_mask,
            nfn_edge_mask=node_topo_to_hedge_mask,
        )
        neut_proxy_sinphi = custom_update_all(
            custom_u_mul_e,
            custom_sum_mailbox,
            efn_src_feat=torch.sin(batch["topo"]["phi"]).unsqueeze(-1),
            efn_src_mask=node_topo_mask,
            efn_num_nodes_dst=n_hedge,
            efn_dst_mask=None,
            efn_edge_feat=inc.permute(0, 2, 1).unsqueeze(-1),
            efn_edge_mask=node_topo_to_hedge_mask,
            nfn_edge_mask=node_topo_to_hedge_mask,
        )
        neut_proxy_phi = torch.atan2(neut_proxy_sinphi, neut_proxy_cosphi)

        neut_proxy_ke = self.transform_funcs["e"].forward(torch.clamp(neut_proxy_ke_raw, 0.0, None))
        neut_proxy_eta = self.transform_funcs["eta"].forward(neut_proxy_eta_raw)
        neut_proxy_kin = torch.cat([neut_proxy_ke, neut_proxy_eta, neut_proxy_phi], dim=-1) * (
            ~proxy_is_charged
        ).unsqueeze(-1)

        proxy_kin = (charged_proxy_kin, neut_proxy_kin)

        topo_em_fracs = batch["topo"]["em_frac"].unsqueeze(1)
        i_t_energy_em_raw = inc_energy_raw * topo_em_fracs
        proxy_ke_em_raw = i_t_energy_em_raw.sum(dim=2)

        proxy_ke_raw = inc_energy_raw.sum(dim=2)
        proxy_em_frac = proxy_ke_em_raw / (proxy_ke_raw + 1e-8)

        return proxy_kin, proxy_is_charged, proxy_em_frac

    def get_ptetaphi_raw_from_kin(self, kin, class_label, unnormalize=False):
        ch_kin, neut_kin = kin
        if unnormalize:
            ch_pt_raw = self.transform_funcs["pt"].inverse(ch_kin[..., 0:1])
            ch_eta_raw = self.transform_funcs["eta"].inverse(ch_kin[..., 1:2])

            neut_ke_raw = self.transform_funcs["e"].inverse(neut_kin[..., 0:1])
            neut_eta_raw = self.transform_funcs["eta"].inverse(neut_kin[..., 1:2])
        else:
            ch_pt_raw = ch_kin[..., 0:1]
            ch_eta_raw = ch_kin[..., 1:2]

            neut_ke_raw = neut_kin[..., 0:1]
            neut_eta_raw = neut_kin[..., 1:2]

        ch_ptetaphi_raw = torch.cat([ch_pt_raw, ch_eta_raw, ch_kin[..., 2:3]], dim=-1)
        neut_pt_raw = neut_ke_raw / torch.cosh(neut_eta_raw)

        m_neut = self.class_mass_dict[3]
        nh_mask = class_label == 3
        neut_pt_raw[nh_mask] = torch.sqrt(
            torch.clamp(neut_ke_raw[nh_mask] ** 2 + 2 * m_neut * neut_ke_raw[nh_mask], 0, None)
        ) / torch.cosh(neut_eta_raw[nh_mask])

        neut_ptetaphi_raw = torch.cat([neut_pt_raw, neut_eta_raw, neut_kin[..., 2:3]], dim=-1)

        ptetaphi_raw = ch_ptetaphi_raw * (class_label < 3).unsqueeze(-1) + neut_ptetaphi_raw * (
            class_label >= 3
        ).unsqueeze(-1)

        return ptetaphi_raw


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------
#
# Sizes below are shrunk from the real `configs/clic/model_configs/
# model_stage1_mini.yml` + `model_stage2.yml` (e.g. embed_dim 192 -> 24) for
# a fast trace, but every field name and every module/mechanism is the same.
# `enable_flash_attn` is forced False (see module docstring for rationale).

_D_HID = 24
_N_TRACK_FEAT = 14
_N_TOPO_FEAT = 9
_N_HYPEREDGE = 6
_N_TRACK = 3
_N_TOPO = 5
_N_NODE = _N_TRACK + _N_TOPO


def _identity_transform_cfg():
    return {"scale_mode": None, "transformation": None}


def _config_v():
    return {
        "max_particles": _N_HYPEREDGE,
        "transformation_dict": {
            "pt": _identity_transform_cfg(),
            "e": _identity_transform_cfg(),
            "eta": _identity_transform_cfg(),
        },
    }


def _config_ms1():
    return {
        "node_prep_model": {
            "type": "mini",
            "track_init_net": {
                "input_dim": _N_TRACK_FEAT,
                "output_dim": _D_HID,
                "hidden_layers": [_D_HID],
                "activation": "LeakyReLU",
                "norm_layer": "LayerNorm",
            },
            "topo_init_net": {
                "input_dim": _N_TOPO_FEAT,
                "output_dim": _D_HID,
                "hidden_layers": [_D_HID],
                "activation": "LeakyReLU",
                "norm_layer": "LayerNorm",
            },
            "transformer": {
                # node_feat is track/topo-init-net output (_D_HID wide) then cat'd
                # with 2 float mask columns (is_track, is_topo) before the
                # transformer, exactly as NodePrepModelMini.forward does -- so
                # embed_dim must be _D_HID + 2, matching the real repo's own
                # `out_dim: 177  # we'll cat the node skip_feat0 to it` pattern
                # of "input width includes the appended columns".
                "embed_dim": _D_HID + 2,
                "num_layers": 2,
                # num_heads=2 (not 4, unlike hg_model's transformers below) so
                # that embed_dim (_D_HID + 2 mask columns) divides evenly; the
                # real repo's own mini config likewise uses a different
                # num_heads per transformer block (each DiTEncoder instance
                # gets its own independent mha_config).
                "mha_config": {"enable_flash_attn": False, "num_heads": 2},
                "dense_config": {
                    "hidden_layers": [_D_HID],
                    "activation": "LeakyReLU",
                    "norm_layer": "LayerNorm",
                },
                "context_dim": _D_HID,
                "out_dim": _D_HID,
            },
            "add_skip_feat": False,
        },
        "hg_model": {
            "type": "iterative_refiner",
            "T_TOTAL": 3,
            "T_BPTT": 2,
            "N_BPTT": 1,
            "d_in": _D_HID,
            "d_hid": _D_HID,
            "init_edges": {"type": "random", "embedding_dim": 5},
            "deepset_n": {"hidden_layers": [_D_HID, _D_HID]},
            "transformer_e": {
                "embed_dim": 2 * _D_HID,
                "num_layers": 1,
                "mha_config": {"enable_flash_attn": False, "num_heads": 4},
                "dense_config": {
                    "hidden_layers": [_D_HID],
                    "activation": "LeakyReLU",
                    "norm_layer": "LayerNorm",
                },
                "context_dim": _D_HID,
                "out_dim": _D_HID,
            },
            "edge_indicator": {
                "input_dim": _D_HID + 1,
                "output_dim": 1,
                "hidden_layers": [_D_HID, 8],
                "activation": "LeakyReLU",
                "norm_layer": "LayerNorm",
            },
        },
    }


def _config_ms2():
    return {
        "hyperedge_model": {
            "ind_threshold": 0.4,
            "proxy_ch_kin_init_net": {
                "input_dim": 4,
                "output_dim": 8,
                "hidden_layers": [8],
                "activation": "LeakyReLU",
                "norm_layer": "LayerNorm",
            },
            "proxy_neut_kin_init_net": {
                "input_dim": 4,
                "output_dim": 8,
                "hidden_layers": [8],
                "activation": "LeakyReLU",
                "norm_layer": "LayerNorm",
            },
            "proxy_em_frac_init_net": {
                "input_dim": 1,
                "output_dim": 8,
                "hidden_layers": [8],
                "activation": "LeakyReLU",
                "norm_layer": "LayerNorm",
            },
            "e_t_init_net": {
                "input_dim": _D_HID,
                "output_dim": 8,
                "hidden_layers": [8],
                "activation": "LeakyReLU",
                "norm_layer": "LayerNorm",
            },
            "inc_times_node_feat_init_net": {
                "input_dim": _D_HID,
                "output_dim": 8,
                "hidden_layers": [8],
                "activation": "LeakyReLU",
                "norm_layer": "LayerNorm",
            },
            "kin_nets": {
                "ch_pt_net": {
                    "input_dim": 32,
                    "output_dim": 1,
                    "hidden_layers": [16, 8],
                    "activation": "LeakyReLU",
                    "norm_layer": "LayerNorm",
                },
                "neut_ke_net": {
                    "input_dim": 32,
                    "output_dim": 1,
                    "hidden_layers": [16, 8],
                    "activation": "LeakyReLU",
                    "norm_layer": "LayerNorm",
                },
            },
            "class_nets": {
                "ch_class_net": {
                    "input_dim": 32,
                    "output_dim": 3,
                    "hidden_layers": [16, 8],
                    "activation": "LeakyReLU",
                    "norm_layer": "LayerNorm",
                },
                "neut_class_net": {
                    "input_dim": 32,
                    "output_dim": 2,
                    "hidden_layers": [16, 8],
                    "activation": "LeakyReLU",
                    "norm_layer": "LayerNorm",
                },
            },
        },
    }


_CLASS_MASS_DICT = {3: 0.939565}  # neutron mass in GeV (as in the repo's helper_dicts.py)


def build_hgpflow():
    torch.manual_seed(0)
    model = HGPFlowModel(_config_v(), _config_ms1(), _config_ms2(), _CLASS_MASS_DICT)
    model.eval()
    return model


def example_input_hgpflow():
    torch.manual_seed(0)
    b = 1

    is_track = torch.zeros(b, _N_NODE, dtype=torch.bool)
    is_track[:, :_N_TRACK] = True
    is_topo = ~is_track

    track_feat0 = torch.randn(b, _N_NODE, _N_TRACK_FEAT)
    topo_feat0 = torch.randn(b, _N_NODE, _N_TOPO_FEAT)

    track_pt = torch.rand(b, _N_NODE) + 0.1
    track_eta = torch.randn(b, _N_NODE) * 0.5
    track_phi = torch.rand(b, _N_NODE) * 3.14159

    topo_e_raw = torch.rand(b, _N_NODE) + 0.1
    topo_eta_raw = torch.randn(b, _N_NODE) * 0.5
    topo_phi = torch.rand(b, _N_NODE) * 3.14159
    topo_em_frac = torch.rand(b, _N_NODE)

    batch = {
        "node": {
            "is_track": is_track,
            "is_topo": is_topo,
        },
        "track": {
            "feat0": track_feat0,
            "pt": track_pt,
            "eta": track_eta,
            "phi": track_phi,
        },
        "topo": {
            "feat0": topo_feat0,
            "e_raw": topo_e_raw,
            "eta_raw": topo_eta_raw,
            "phi": topo_phi,
            "em_frac": topo_em_frac,
        },
    }
    return (batch,)


MENAGERIE_ENTRIES = [
    (
        "HGPflow (Hypergraph Particle Flow)",
        "build_hgpflow",
        "example_input_hgpflow",
        2024,
        MENAGERIE_ZOO,
    ),
]
