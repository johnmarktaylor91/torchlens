# SOURCE: vendored from lijfrank/EmotionIC @ b426452bda8d3e2b5b9896698a616e8be38aa20c
# https://raw.githubusercontent.com/lijfrank/EmotionIC/b426452bda8d3e2b5b9896698a616e8be38aa20c/model.py
# https://raw.githubusercontent.com/lijfrank/EmotionIC/b426452bda8d3e2b5b9896698a616e8be38aa20c/model_gru.py
# https://raw.githubusercontent.com/lijfrank/EmotionIC/b426452bda8d3e2b5b9896698a616e8be38aa20c/model_attention.py
#
# Frank et al. (Science China Information Sciences, 2023) "EmotionIC: emotional inertia
# and contagion-driven dependency modelling for emotion recognition in conversation" --
# a Transformer-based "global" self/other-conversation-role attention branch
# (ConvPos-relative-position multi-head attention, dispatched from `TransformerEncoder`
# with `attn_type='convpos'` as used by the real `EmotionIC.__init__`) fused with a
# custom "individual" branch (`ConvGRU`: a GRU-cell variant whose recurrence graph is
# built per-conversation from same-speaker "inertia" and cross-speaker "contagion"
# links, not a plain chronological chain), followed by LayerNorm fusion and a linear
# classification head.
#
# This file vendors exactly the subset of `model.py` + `model_gru.py` (`ConvGRU`,
# `ConvGRU`'s cell) + `model_attention.py` (`TransformerEncoder`/`TransformerLayer` and
# the `convpos` attention path it dispatches to, `RelativeSinusoidalPositionalEmbedding`)
# actually reached by the real `EmotionIC(attn_type='convpos', pos_embed=None)`
# construction in the source `model.py`. Sibling classes not on that path
# (`ConvGRU_self`, `ConvGRU_other`, `MultiHeadAttn_Order`, `ConvPosMultiHeadAttn_Seg`,
# etc.) are omitted -- they belong to other attention/GRU variants swept by the same
# repo's ablations, not to the shipped `EmotionIC` architecture.
#
# No architectural changes were made; only mechanical fixes for import isolation:
#   - The two upstream files import each other and `torch.nn.GRU` at module scope; those
#     imports are inlined here since this is a single self-contained staging module.
#   - `EmotionIC.__init__` hardcoded `attn_type`/`pos_embed`/`after_norm` as local
#     variables (not exposed as constructor args) in the original -- kept identical,
#     just documented, since that's exactly how the source is constructed and trained.

import math
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# model_gru.py (ConvGRU branch: "individual"/self-contagion GRU)
# ---------------------------------------------------------------------------


class ConvRNNCellBase(nn.Module):
    __constants__ = ["input_size", "hidden_size", "bias"]

    input_size: int
    hidden_size: int
    bias: bool
    weight_ih: torch.Tensor
    weight_hh: torch.Tensor

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool, num_chunks_x: int, num_chunks_y: int
    ) -> None:
        super(ConvRNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(num_chunks_x * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(num_chunks_y * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(num_chunks_x * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(num_chunks_y * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        s = "{input_size}, {hidden_size}"
        if "bias" in self.__dict__ and self.bias is not True:
            s += ", bias={bias}"
        if "nonlinearity" in self.__dict__ and self.nonlinearity != "tanh":
            s += ", nonlinearity={nonlinearity}"
        return s.format(**self.__dict__)

    def check_forward_input(self, input: torch.Tensor) -> None:
        if input.size(-1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(-1), self.input_size
                )
            )

    def check_forward_hidden(
        self, input: torch.Tensor, hx: torch.Tensor, hidden_label: str = ""
    ) -> None:
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)
                )
            )

        if hx.size(-1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(-1), self.hidden_size
                )
            )

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)

        for name, weight in self.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(weight, gain=1.0)
            else:
                nn.init.uniform_(weight, -stdv, stdv)


class ConvGRUCell(ConvRNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super(ConvGRUCell, self).__init__(
            input_size, hidden_size, bias, num_chunks_x=4, num_chunks_y=6
        )

    def _conv_gru_cell(self, x: torch.Tensor, hx: torch.Tensor, hy: torch.Tensor):
        W_ir, W_is, W_iz, W_il = torch.chunk(self.weight_ih, chunks=4, dim=0)
        b_ir, b_is, b_iz, b_il = torch.chunk(self.bias_ih, chunks=4, dim=0)

        W_hr, W_hs, W_hrz, W_hsz, W_hn, W_hm = torch.chunk(self.weight_hh, chunks=6, dim=0)
        b_hr, b_hs, b_hrz, b_hsz, b_hn, b_hm = torch.chunk(self.bias_hh, chunks=6, dim=0)

        r = torch.sigmoid(torch.matmul(x, W_ir.T) + b_ir + torch.matmul(hy, W_hr.T) + b_hr)
        s = torch.sigmoid(torch.matmul(x, W_is.T) + b_is + torch.matmul(hx, W_hs.T) + b_hs)

        z = torch.sigmoid(
            torch.matmul(x, W_iz.T)
            + b_iz
            + torch.matmul(hy, W_hrz.T)
            + b_hrz
            + torch.matmul(hx, W_hsz.T)
            + b_hsz
        )
        n = torch.tanh(
            torch.matmul(x, W_il.T)
            + b_il
            + r * (torch.matmul(hy, W_hn.T) + b_hn)
            + s * (torch.matmul(hx, W_hm.T) + b_hm)
        )

        h_ = (1 - z) * n + z * hx

        return h_

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
        hy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        if hy is None:
            hy = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        self.check_forward_hidden(input, hx, "")
        self.check_forward_hidden(input, hy, "")
        return self._conv_gru_cell(input, hx, hy)


class ConvGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super(ConvGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.device = "cpu"

        self.layers = nn.ModuleList(
            [ConvGRUCell(input_size, hidden_size, bias)]
            + [ConvGRUCell(hidden_size, hidden_size, bias) for _ in range(num_layers - 1)]
        )
        if self.bidirectional:
            self.layers_r = nn.ModuleList(
                [ConvGRUCell(input_size, hidden_size, bias)]
                + [ConvGRUCell(hidden_size, hidden_size, bias) for _ in range(num_layers - 1)]
            )
        else:
            self.register_parameter("layers_r", None)

        self.dropout_layer = nn.Dropout(self.dropout)

    def _reverse(self, input: torch.Tensor, umask: torch.Tensor):
        conv_len = torch.sum(umask, dim=0)
        batch_size = input.size(1)
        input_ret = torch.zeros_like(input).to(self.device)

        for i in range(batch_size):
            input_ret[torch.arange(conv_len[i] - 1, -1, -1), i] = input[
                torch.arange(conv_len[i]), i
            ]

        return input_ret

    def _bulid_conv_id(self, qmask: torch.Tensor, umask: torch.Tensor) -> torch.Tensor:
        max_p = torch.max(qmask).item()
        min_p = torch.min(qmask).item()
        assert min_p >= 0

        seq_len, batch_size = qmask.size()
        mem_near = (-1) * torch.ones(batch_size, max_p + 1, dtype=torch.int8).to(self.device)
        conv_id = torch.zeros_like(qmask).to(self.device)
        batch_id = torch.arange(batch_size)

        for i in range(seq_len):
            conv_id[i] = mem_near[batch_id, qmask[i]]
            mem_near[batch_id, qmask[i]] = i

        conv_id += 1
        return conv_id * umask

    def _build_conv_influ(
        self,
        qmask: torch.Tensor,
        umask: torch.Tensor,
        conv_id: torch.Tensor,
        self_shift=3,
        self_scale=1,
        other_shift=0,
        other_scale=2,
    ) -> torch.Tensor:
        def conv_sigmoid(distance: torch.Tensor, shift=0, scale=1, pos=True) -> torch.Tensor:
            reversal = 1.0 if pos else -1.0
            return 1 / (1 + torch.exp(reversal * (-distance + shift) / scale))

        seq_len, batch_size = qmask.size()
        seq_range = torch.arange(seq_len).to(qmask.device)
        zeros_pad = torch.zeros(batch_size, dtype=torch.long).to(qmask.device)

        self_influ_scale = (seq_range[:, None] - conv_id) + 1e2 * (conv_id == 0)
        other_influ_scale = torch.zeros_like(qmask)

        conv_len, last_conv = zeros_pad, -1 * torch.ones_like(qmask[0])

        for i in range(1, seq_len):
            other_influ_scale[i] = torch.where(qmask[i] == last_conv, conv_len + 1, zeros_pad)

            conv_len = torch.where(qmask[i] == qmask[i - 1], conv_len, other_influ_scale[i])

            last_conv = torch.where(qmask[i] != qmask[i - 1], qmask[i - 1], last_conv)

        return (
            torch.stack(
                (
                    conv_sigmoid(self_influ_scale, pos=False, shift=self_shift, scale=self_scale),
                    conv_sigmoid(other_influ_scale, pos=True, shift=other_shift, scale=other_scale),
                )
            )
            * umask[None, :, :]
        )

    def _compute_grucell(
        self,
        layer_id: int,
        input: torch.Tensor,
        conv_id: torch.Tensor,
        conv_inertia: torch.Tensor,
        conv_contagion: torch.Tensor,
        umask: torch.Tensor,
        influ_scale: torch.Tensor,
        bidirectional: bool = False,
    ) -> torch.Tensor:
        assert 0 <= layer_id < self.num_layers
        assert input.size(-1) == self.layers[layer_id].input_size

        seq_len, batch_size = conv_id.size()
        h_ = [torch.zeros(batch_size, self.hidden_size).to(self.device)]

        for j in range(seq_len):
            umask_j = umask[j].unsqueeze(1)
            hx = (
                umask_j
                * torch.cat([h_[k][i].unsqueeze(0) for i, k in enumerate(conv_id[j])])
                * (conv_inertia[j])[:, None]
            )
            hy = umask_j * h_[-1] * (conv_contagion[j])[:, None]
            h_t = (
                self.layers[layer_id](
                    umask_j * input[j],
                    hx * influ_scale[0, j][:, None],
                    hy * influ_scale[1, j][:, None],
                )
                if not bidirectional
                else self.layers_r[layer_id](
                    umask_j * input[j],
                    hx * influ_scale[0, j][:, None],
                    hy * influ_scale[1, j][:, None],
                )
            )
            h_.append(h_t)
        h_.pop(0)

        return torch.cat([h_t.unsqueeze(0) for h_t in h_])

    def forward(self, input: torch.Tensor, qmask: torch.Tensor, umask: torch.Tensor):
        assert input.size()[:-1] == qmask.size() == umask.size()

        self.device = input.device
        seq_length, batch_size = umask.shape
        seq_range = torch.arange(seq_length).to(self.device)

        if self.batch_first:
            h_ = input.transpose(0, 1)
            qmask = qmask.transpose(0, 1)
            umask = umask.transpose(0, 1)
        else:
            h_ = input

        conv_id = self._bulid_conv_id(qmask, umask)
        conv_bool = (seq_range[1:][:, None] != conv_id[1:]).T * umask[1:].T
        conv_end = [torch.nonzero(conv_batch).squeeze(1) for conv_batch in conv_bool]
        conv_conti = []
        for idx, conv_e in enumerate(conv_end):
            if conv_e.dim() == 0:
                conv_conti.append(torch.ones(1, dtype=torch.bool))
            else:
                conv_conti.append(qmask[conv_e[:-1], idx] == qmask[conv_e[1:] + 1, idx])
        conv_inertia = conv_id > 0
        for idx in range(batch_size):
            conv_inertia[conv_end[idx][1:] + 1, idx] = conv_conti[idx]
        conv_contagion = conv_id != seq_range[:, None]

        influ_scale = self._build_conv_influ(qmask, umask, conv_id)

        for i in range(self.num_layers):
            h_ = self._compute_grucell(
                i, h_, conv_id, conv_inertia, conv_contagion, umask, influ_scale
            )
            if i + 1 != self.num_layers:
                h_ = self.dropout_layer(h_)

        if self.bidirectional:
            h_r = self._reverse(input, umask)
            qmask_r = self._reverse(qmask, umask)

            conv_id_r = self._bulid_conv_id(qmask_r, umask)
            conv_bool_r = (seq_range[1:][:, None] != conv_id_r[1:]).T * umask[1:].T
            conv_end_r = [torch.nonzero(conv_batch).squeeze(1) for conv_batch in conv_bool_r]
            conv_conti_r = []
            for idx, conv_e in enumerate(conv_end_r):
                if conv_e.dim() == 0:
                    conv_conti_r.append(torch.ones(1, dtype=torch.bool))
                else:
                    conv_conti_r.append(qmask_r[conv_e[:-1], idx] == qmask_r[conv_e[1:] + 1, idx])
            conv_inertia_r = conv_id_r > 0
            for idx in range(batch_size):
                conv_inertia_r[conv_end_r[idx][1:] + 1, idx] = conv_conti_r[idx]
            conv_contagion_r = conv_id_r != seq_range[:, None]

            influ_scale_r = self._build_conv_influ(qmask_r, umask, conv_id_r)

            for i in range(self.num_layers):
                h_r = self._compute_grucell(
                    i, h_r, conv_id_r, conv_inertia_r, conv_contagion_r, umask, influ_scale_r, True
                )
                if i + 1 != self.num_layers:
                    h_r = self.dropout_layer(h_r)
            h_r = self._reverse(h_r, umask)
            h_ = torch.cat([h_, h_r], dim=-1)

        return h_ if not self.batch_first else h_.transpose(0, 1)


# ---------------------------------------------------------------------------
# model_attention.py (TransformerEncoder + convpos relative-position attention:
# the "global"/conversation-wide branch), restricted to the `attn_type='convpos'`,
# `pos_embed=None` path actually used by `EmotionIC.__init__`.
# ---------------------------------------------------------------------------


def compute_squared_EDM_method4(n):
    X = np.expand_dims(np.arange(n), -1)
    G = np.dot(X, X.T)
    H = np.tile(np.diag(G), (n, 1))
    return torch.FloatTensor(H + H.T - 2 * G)


class RelativeEmbedding(nn.Module):
    def forward(self, input):
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + seq_len
        if max_pos > self.origin_shift:
            weights = self.get_embedding(
                max_pos * 2,
                self.embedding_dim,
                self.padding_idx,
            )
            weights = weights.to(self._float_tensor)
            del self.weights
            self.origin_shift = weights.size(0) // 2
            self.register_buffer("weights", weights)

        positions = (
            torch.arange(int(-seq_len / 2), round(seq_len / 2 + 1e-5)).to(input.device).long()
            + self.origin_shift
        )
        embed = self.weights.index_select(0, positions.long()).detach()
        return embed


class RelativeSinusoidalPositionalEmbedding(RelativeEmbedding):
    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size % 2 == 0
        weights = self.get_embedding(
            init_size + 1,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer("weights", weights)
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(-num_embeddings // 2, num_embeddings // 2, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        self.origin_shift = num_embeddings // 2 + 1
        return emb


class ConvPosMultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, scale=False):
        super().__init__()
        assert d_model % n_head == 0

        self.shift = nn.Parameter(torch.abs(torch.randn(1)) + 0.001)
        self.bias = nn.Parameter(-torch.abs(torch.randn(1)))

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 5 * d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_model // n_head, 0, 1200)
        self.qk_pos = nn.Linear(d_model // n_head, 3 * d_model, bias=False)

        self.device = "cpu"

        if scale:
            self.scale = math.sqrt(2 * d_model // n_head)
        else:
            self.scale = 1

    def forward(self, x, mask, qmask, use_Gaussian=False):
        self.device = x.device

        batch_size, max_len, d_model = x.size()
        x = self.qkv_linear(x)
        q, k, k_1, k_2, v = torch.chunk(x, 5, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        k_1 = k_1.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        k_2 = k_2.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        y = self.qk_pos(self.pos_embed(mask))
        q_p, k1_p, k2_p = torch.chunk(y, 3, dim=-1)
        q_p = q_p.view(max_len, self.n_head, -1).transpose(0, 1)
        k1_p = k1_p.view(max_len, self.n_head, -1).permute(1, 2, 0)
        k2_p = k2_p.view(max_len, self.n_head, -1).permute(1, 2, 0)

        attn_self = torch.matmul(q, k_1) + torch.matmul(q_p, k1_p)
        attn_others = torch.matmul(q, k_2) + torch.matmul(q_p, k2_p)

        qmask_array = 1.0 * (
            qmask.unsqueeze(-1).repeat(1, 1, max_len) == qmask.unsqueeze(-2).repeat(1, max_len, 1)
        )

        attn_self.masked_fill_(mask=qmask_array[:, None].eq(0.0), value=0.0)
        attn_others.masked_fill_(mask=qmask_array[:, None].eq(1.0), value=0.0)
        attn = attn_self + attn_others

        attn = attn / self.scale
        attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float("-inf"))

        if use_Gaussian:
            square_distance = compute_squared_EDM_method4(max_len).to(q.device)
            shift_M = self.shift * torch.ones(max_len, max_len).to(q.device)
            bias_M = self.bias * torch.ones(max_len, max_len).to(q.device)
            dis_M = -(shift_M * square_distance + bias_M)
            dis_M_ = dis_M.unsqueeze(0).unsqueeze(0).repeat((batch_size, self.n_head, 1, 1))
            attn += dis_M_

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v)
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.fc(v)

        return v


class TransformerLayer(nn.Module):
    def __init__(self, d_model, self_attn, feedforward_dim, after_norm, dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = self_attn

        self.after_norm = after_norm

        self.ffn = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask, qmask, use_Gaussian=False):
        residual = x
        if not self.after_norm:
            x = self.norm1(x)

        x = self.self_attn(x, mask, qmask, use_Gaussian)
        x = x + residual
        if self.after_norm:
            x = self.norm1(x)
        residual = x
        if not self.after_norm:
            x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        if self.after_norm:
            x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        n_head,
        feedforward_dim,
        dropout,
        after_norm=True,
        attn_type="naive",
        scale=False,
        dropout_attn=None,
        pos_embed=None,
        batch_first=True,
    ):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.d_model = d_model
        self.batch_first = batch_first

        if pos_embed is None:
            self.pos_embed = None
        elif pos_embed == "sin":
            self.pos_embed = (
                None  # non-convpos position-embedding variants omitted; unused by EmotionIC
            )
        elif pos_embed == "fix":
            self.pos_embed = (
                None  # non-convpos position-embedding variants omitted; unused by EmotionIC
            )

        if attn_type == "convpos":
            self_attn = ConvPosMultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)
        else:
            raise ValueError(
                'attn_type not supported by this vendored subset (only "convpos" is '
                "kept, matching the real EmotionIC construction): {}".format(attn_type)
            )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model, deepcopy(self_attn), feedforward_dim, after_norm, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask, qmask, use_Gaussian=False):
        if not self.batch_first:
            x = x.transpose(0, 1)
            mask = mask.transpose(0, 1)
            qmask = qmask.transpose(0, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed(mask)

        for layer in self.layers:
            x = layer(x, mask, qmask, use_Gaussian)

        return x if self.batch_first else x.transpose(0, 1)


# ---------------------------------------------------------------------------
# model.py (EmotionIC itself)
# ---------------------------------------------------------------------------


class EmotionIC(nn.Module):
    def __init__(
        self,
        hidden_dim,
        output_dim,
        trans_n_layers,
        indi_n_layer,
        dropout=0.6,
        attn_drop=0.6,
        feed_drop=0.6,
        rnn_drop=0.6,
        use_dropout=False,
    ):
        super(EmotionIC, self).__init__()

        self.trans_n_layers = trans_n_layers
        self.indi_n_layer = indi_n_layer

        dim = 1024

        self.embedding_dim = dim
        self.num_head = 8
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.dropout = dropout
        self.dropout_attn = attn_drop
        self.dropout_feed = feed_drop
        self.dropout_rnn = rnn_drop

        after_norm = 1
        attn_type = "convpos"
        pos_embed = None
        self.global_encoder = TransformerEncoder(
            self.trans_n_layers,
            self.embedding_dim,
            self.num_head,
            feedforward_dim=self.hidden_dim,
            dropout=self.dropout_feed,
            after_norm=after_norm,
            attn_type=attn_type,
            scale=attn_type == "adatrans",
            dropout_attn=self.dropout_attn,
            pos_embed=pos_embed,
            batch_first=False,
        )

        self.conv_GRU_indi = ConvGRU(
            self.embedding_dim,
            self.hidden_dim,
            num_layers=self.indi_n_layer,
            bidirectional=False,
            dropout=self.dropout_rnn if use_dropout else 0.0,
        )

        self.LN_glob = nn.LayerNorm(self.embedding_dim, elementwise_affine=True)
        self.LN_local = nn.LayerNorm(self.hidden_dim, elementwise_affine=True)
        self.LN_origin = nn.LayerNorm(self.embedding_dim, elementwise_affine=True)

        self.fc_embed = nn.Linear(1 * self.hidden_dim + 2 * self.embedding_dim, self.hidden_dim)

        self.fc_out = nn.Sequential(
            nn.Dropout(self.dropout), nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, text, umask, qmask):
        glob_hidden = self.global_encoder(text, umask, qmask)
        indi_hidden = self.conv_GRU_indi(text, qmask, umask)
        text = self.LN_origin(text)
        fc_embeds = self.fc_embed(
            torch.cat((self.LN_glob(glob_hidden), self.LN_local(indi_hidden), text), dim=-1)
        )
        fc_out = self.fc_out(fc_embeds)

        return F.log_softmax(fc_out, 2), fc_embeds


def build_emotionic():
    return EmotionIC(hidden_dim=32, output_dim=6, trans_n_layers=1, indi_n_layer=1)


def example_input_emotionic():
    seq_len = 5
    batch = 2
    dim = 1024
    text = torch.randn(seq_len, batch, dim)
    umask = torch.ones(seq_len, batch, dtype=torch.long)
    qmask = torch.zeros(seq_len, batch, dtype=torch.long)
    qmask[:, 0] = torch.tensor([0, 1, 0, 1, 0])
    qmask[:, 1] = torch.tensor([1, 0, 1, 0, 1])
    return (text, umask, qmask)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("EmotionIC", "build_emotionic", "example_input_emotionic", 2023, "vendored"),
]
