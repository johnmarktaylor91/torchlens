# SOURCE: vendored from zackchen-lb/GTA @ main
#
# Vendors the real nn.Module classes implementing GTA (Graph Temporal Attention), a graph
# transformer for multivariate time-series anomaly/deviation detection over industrial-control
# / IoT sensor networks (models/gta.py's `GTA`), which composes: an `AdaGCNConv`
# (torch_geometric MessagePassing graph conv with a learned per-edge Gumbel-softmax gate over
# `logits`) + dilated `TemporalBlock` 1D convs (models/tconv.py) into a `GraphTemporalEmbedding`
# that jointly learns a sparse sensor graph and temporal receptive field, feeding an Informer
# encoder-decoder (models/model.py, models/encoder.py, models/decoder.py, models/attn.py --
# ProbSparse/Full attention + distilling conv encoder) borrowed from Zhou et al.'s Informer
# repo (this repo's own doc comment: "Most of the models are copied from
# https://github.com/ratschlab/RGAN" applies only to a different file; models/model.py is the
# well-known zhouhaoyi/Informer2020 encoder-decoder, vendored here verbatim per this repo).
#
# Fixes applied (mechanical only, no architecture change):
#  1. `GraphTemporalEmbedding.forward` references `self.gc_modules[0]` (plural, with an "s")
#     on its first-level branch, but `__init__` only ever defines a single shared
#     `self.gc_module` (singular) -- the plural name is never assigned anywhere in the class
#     (a latent AttributeError bug in the upstream code; note the adjacent commented-out
#     `# self.gc_modules.extend(...)` shows the author considered, then abandoned, a
#     per-level module list in favor of the single shared `gc_module` actually built in
#     `__init__`). Renamed the one erroneous `self.gc_modules[0]` call-site to
#     `self.gc_module` (matching every other use in the same method) so the real architecture
#     (single shared learned-graph conv module reused across all dilation levels) actually
#     runs, rather than papering over it with a synthetic reimplementation.
#  2. `utils/masking.py`'s `ProbMask.__init__` passes `dytpe=torch.bool` (typo) to
#     `torch.ones(...)`, an invalid keyword that would TypeError before the mask is even
#     built; corrected to `dtype=torch.bool`.
#  3. `GTA.forward`'s two `.double()` casts on `dec_inp` (upstream numeric-precision choice,
#     unrelated to the graph/temporal/attention architecture) are dropped since every layer
#     here (Linear/Conv1d/LayerNorm) is float32 by default and the two dtypes cannot be mixed;
#     this only pins the working dtype, it does not change control flow or module structure.
#
# Repo: https://github.com/zackchen-lb/GTA @ main
# Files: models/gta.py, models/tconv.py, models/model.py, models/encoder.py, models/decoder.py,
#        models/attn.py, models/embed.py, utils/masking.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils import weight_norm
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import glorot, zeros

MENAGERIE_ZOO = "vendored-pytorch"


# ---- utils/masking.py ----
class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


# ---- models/embed.py ----
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", data="ETTh", dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # self.temporal_embedding = TemporalEmbedding(...)  # unused: caller never provides x_mark features

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


# ---- models/attn.py ----
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, float("-inf"))

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L, E = K.shape
        _, _, S, _ = Q.shape

        K_expand = K.unsqueeze(-3).expand(B, H, S, L, E)
        indx_sample = torch.randint(L, (S, sample_k))
        K_sample = K_expand[:, :, torch.arange(S).unsqueeze(1), indx_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L)
        M_top = M.topk(n_top, sorted=False)[1]

        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.sum(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert L_Q == L_V
            contex = V.cumsum(dim=-1)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)

        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = (
            torch.matmul(attn, V)
        )
        return context_in

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape

        queries = queries.view(B, H, L, -1)
        keys = keys.view(B, H, S, -1)
        values = values.view(B, H, S, -1)

        U = self.factor * int(math.ceil(math.log(S)))
        u = self.factor * int(math.ceil(math.log(L)))
        U = min(U, S)
        u = min(u, L)

        scores_top, index = self._prob_QK(queries, keys, u, U)
        scale = self.scale or 1.0 / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        context = self._get_initial_context(values, L)
        context = self._update_context(context, values, scores_top, index, L, attn_mask)

        return context.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(queries, keys, values, attn_mask).view(B, L, -1)

        return self.out_projection(out)


# ---- models/encoder.py ----
class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in, out_channels=c_in, kernel_size=3, padding=2, padding_mode="circular"
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        x = x + self.dropout(self.attention(x, x, x, attn_mask=attn_mask))

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
            x = self.attn_layers[-1](x)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


# ---- models/decoder.py ----
class DecoderLayer(nn.Module):
    def __init__(
        self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask))
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask))

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


# ---- models/model.py (Informer, from zhouhaoyi/Informer2020, vendored per this repo) ----
class Informer(nn.Module):
    def __init__(
        self,
        enc_in,
        dec_in,
        c_out,
        seq_len,
        label_len,
        out_len,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_layers=2,
        d_ff=512,
        dropout=0.0,
        attn="prob",
        embed="fixed",
        data="ETTh",
        activation="gelu",
        device=torch.device("cpu"),
    ):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn

        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, data, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, data, dropout)

        Attn = ProbAttention if attn == "prob" else FullAttention

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, factor, attention_dropout=dropout), d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            [ConvLayer(d_model) for _ in range(e_layers - 1)],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, factor, attention_dropout=dropout), d_model, n_heads
                    ),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout), d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        return dec_out[:, -self.pred_len :, :]


# ---- models/tconv.py ----
class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size=3, stride=1, dilation=1, padding=1, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                padding_mode="circular",
            )
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                padding_mode="circular",
            )
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# ---- models/gta.py ----
class AdaGCNConv(MessagePassing):
    def __init__(
        self,
        num_nodes,
        in_channels,
        out_channels,
        improved=False,
        add_self_loops=False,
        normalize=True,
        bias=True,
        init_method="all",
    ):
        super(AdaGCNConv, self).__init__(aggr="add", node_dim=0)
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.bias = bias
        self.init_method = init_method

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self._init_graph_logits_()

        self.reset_parameters()

    def _init_graph_logits_(self):
        if self.init_method == "all":
            logits = 0.8 * torch.ones(self.num_nodes**2, 2)
            logits[:, 1] = 0
        elif self.init_method == "random":
            logits = 1e-3 * torch.randn(self.num_nodes**2, 2)
        elif self.init_method == "equal":
            logits = 0.5 * torch.ones(self.num_nodes**2, 2)
        else:
            raise NotImplementedError("Initial Method %s is not implemented" % self.init_method)

        self.register_parameter("logits", Parameter(logits, requires_grad=True))

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        if self.normalize:
            edge_index, edge_weight = gcn_norm(
                edge_index,
                edge_weight,
                x.size(self.node_dim),
                self.improved,
                self.add_self_loops,
                dtype=x.dtype,
            )

        z = torch.nn.functional.gumbel_softmax(self.logits, hard=True)

        x = torch.matmul(x, self.weight)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None, z=z)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, edge_weight, z):
        if edge_weight is None:
            return x_j * z[:, 0].contiguous().view([-1] + [1] * (x_j.dim() - 1))
        else:
            return (
                edge_weight.view([-1] + [1] * (x_j.dim() - 1))
                * x_j
                * z[:, 0].contiguous().view([-1] + [1] * (x_j.dim() - 1))
            )

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class GraphTemporalEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_nodes,
        seq_len,
        num_levels,
        kernel_size=3,
        dropout=0.02,
        device=torch.device("cpu"),
    ):
        super(GraphTemporalEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.num_levels = num_levels
        self.device = device
        assert (kernel_size - 1) // 2

        self.tc_modules = torch.nn.ModuleList([])
        self.gc_module = AdaGCNConv(num_nodes, seq_len, seq_len)
        for i in range(num_levels):
            dilation_size = 2**i
            self.tc_modules.extend(
                [
                    TemporalBlock(
                        num_nodes,
                        num_nodes,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=dilation_size,
                        padding=(kernel_size - 1) * dilation_size // 2,
                        dropout=dropout,
                    )
                ]
            )

        source_nodes, target_nodes = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                source_nodes.append(j)
                target_nodes.append(i)
        self.edge_index = torch.tensor(
            [source_nodes, target_nodes], dtype=torch.long, device=self.device
        )

    def forward(self, x):
        # >> (bsz, seq_len, num_nodes)
        x = x.permute(0, 2, 1)  # >> (bsz, num_nodes, seq_len)

        x = self.tc_modules[0](x)  # >> (bsz, num_nodes, seq_len)
        # NOTE (fix): upstream reads `self.gc_modules[0]` here, an attribute that is never
        # defined anywhere in this class (only the singular `self.gc_module`, built once above,
        # is ever assigned) -- renamed to `self.gc_module` to match every other call-site below.
        x = self.gc_module(x.transpose(0, 1), self.edge_index).transpose(
            0, 1
        )  # >> (bsz, num_nodes, seq_len)

        for i in range(1, self.num_levels):
            x = self.tc_modules[i](x)  # >> (bsz, num_nodes, seq_len)
            x = self.gc_module(x.transpose(0, 1), self.edge_index).transpose(
                0, 1
            )  # >> (bsz, num_nodes, seq_len)

        return x.transpose(1, 2)


class GTA(torch.nn.Module):
    def __init__(
        self,
        num_nodes,
        seq_len,
        label_len,
        out_len,
        num_levels,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_layers=2,
        d_ff=512,
        dropout=0.0,
        attn="prob",
        embed="fixed",
        data="ETTh",
        activation="gelu",
        device=torch.device("cpu"),
    ):
        super(GTA, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.label_len = label_len
        self.out_len = out_len
        self.num_levels = num_levels
        self.device = device

        self.gt_embedding = GraphTemporalEmbedding(
            num_nodes, seq_len, num_levels, kernel_size=3, dropout=dropout, device=device
        )
        self.model = Informer(
            num_nodes,
            num_nodes,
            num_nodes,
            seq_len,
            label_len,
            out_len,
            factor,
            d_model,
            n_heads,
            e_layers,
            d_layers,
            d_ff,
            dropout,
            attn,
            embed,
            data,
            activation,
            device,
        )

    def forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = self.gt_embedding(batch_x)  # >> (bsz, seq, num_nodes)
        dec_inp = torch.zeros_like(batch_y[:, -self.out_len :, :]).to(self.device)
        dec_inp = torch.cat([batch_y[:, : self.label_len, :], dec_inp], dim=1).to(self.device)
        output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        return output


def build_gta():
    # Tiny sizes throughout (real defaults per main_gta_dad.py's argparse: num_levels=3,
    # d_model=128, n_heads=8, e_layers=3, d_layers=2, d_ff=128, attn='prob') shrunk for a
    # fast trace; architecture and control flow unchanged.
    return GTA(
        num_nodes=6,
        seq_len=12,
        label_len=6,
        out_len=4,
        num_levels=2,
        factor=5,
        d_model=16,
        n_heads=2,
        e_layers=2,
        d_layers=1,
        d_ff=16,
        dropout=0.0,
        attn="prob",
        embed="fixed",
        data="ETTh",
        activation="gelu",
        device=torch.device("cpu"),
    )


def example_input_gta():
    num_nodes, seq_len, label_len, out_len = 6, 12, 6, 4
    batch_x = torch.randn(2, seq_len, num_nodes)
    batch_y = torch.randn(2, label_len + out_len, num_nodes)
    batch_x_mark = torch.randn(2, seq_len, 4)
    batch_y_mark = torch.randn(2, label_len + out_len, 4)
    return (batch_x, batch_y, batch_x_mark, batch_y_mark)


MENAGERIE_ENTRIES = [
    ("GTA (Graph Temporal Attention)", build_gta, example_input_gta, 2021, MENAGERIE_ZOO),
]
