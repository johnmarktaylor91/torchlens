# SOURCE: vendored from yoniLc/ECCT @ 361b9910e8429a7a77af8153a91ca87818067092
# (Model.py -- full file: Encoder, SublayerConnection, EncoderLayer, MultiHeadedAttention,
#  PositionwiseFeedForward, ECC_Transformer; Codes.py::sign_to_bin)
"""Error Correction Code Transformer (ECCT) for algebraic block-code decoding
(Choukroun & Wolf, "Error Correction Code Transformer", NeurIPS 2022, arXiv:2203.14966).
Official repo: https://github.com/yoniLc/ECCT (``Model.py`` + ``Codes.py`` @ main).

ECCT reformulates soft-decision decoding of a linear block code as a masked
Transformer-encoder sequence task: the input is the concatenation of the channel LLR
magnitudes and the syndrome (computed from the received word against the code's
parity-check matrix), embedded per-position via a learned scale vector
(``src_embed``), then passed through ``N_dec`` masked self-attention encoder layers
where the attention mask is derived from the code's Tanner graph (``get_mask``,
built from ``pc_matrix``) so each bit/check-node position only attends to positions
it is structurally connected to. A final linear projection maps back to
``n`` output logits (the estimated multiplicative noise on the codeword).

``sign_to_bin`` is vendored from ``Codes.py`` (used by ``ECC_Transformer.loss``, kept for
architectural completeness even though it is not exercised by a bare forward pass). The
data-generation / BCH-BER-training harness (``Main.py``) and the on-disk parity-check-matrix
loader (``Get_Generator_and_Parity``, which reads ``Codes_DB/*.alist``/``*.txt`` files) are
not part of the model and are not vendored; the staging build helper below constructs a
small synthetic binary parity-check matrix directly (same role as ``Get_Generator_and_Parity``'s
output: an ``(n-k) x n`` binary array used only to build the attention mask and to size
``src_embed``/``out_fc`` -- no architectural difference from using a real BCH/LDPC/POLAR
matrix loaded from ``Codes_DB``). No layer, head count, or forward-pass control-flow was
changed from the official repo.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm


def sign_to_bin(x):
    return 0.5 * (1 - x)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if N > 1:
            self.norm2 = LayerNorm(layer.size)

    def forward(self, x, mask):
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, mask)
            if idx == len(self.layers) // 2 and len(self.layers) > 1:
                x = self.norm2(x)
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))  # noqa: E741
        ]

        x, self.attn = self.attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class ECC_Transformer(nn.Module):
    def __init__(self, args, dropout=0):
        super().__init__()
        code = args.code
        c = copy.deepcopy
        attn = MultiHeadedAttention(args.h, args.d_model)
        ff = PositionwiseFeedForward(args.d_model, args.d_model * 4, dropout)

        self.src_embed = torch.nn.Parameter(
            torch.empty((code.n + code.pc_matrix.size(0), args.d_model))
        )
        self.decoder = Encoder(EncoderLayer(args.d_model, c(attn), c(ff), dropout), args.N_dec)
        self.oned_final_embed = torch.nn.Sequential(*[nn.Linear(args.d_model, 1)])
        self.out_fc = nn.Linear(code.n + code.pc_matrix.size(0), code.n)

        self.get_mask(code)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, magnitude, syndrome):
        emb = torch.cat([magnitude, syndrome], -1).unsqueeze(-1)
        emb = self.src_embed.unsqueeze(0) * emb
        emb = self.decoder(emb, self.src_mask)
        return self.out_fc(self.oned_final_embed(emb).squeeze(-1))

    def loss(self, z_pred, z2, y):
        loss = F.binary_cross_entropy_with_logits(z_pred, sign_to_bin(torch.sign(z2)))
        x_pred = sign_to_bin(torch.sign(-z_pred * torch.sign(y)))
        return loss, x_pred

    def get_mask(self, code, no_mask=False):
        if no_mask:
            self.src_mask = None
            return

        def build_mask(code):
            mask_size = code.n + code.pc_matrix.size(0)
            mask = torch.eye(mask_size, mask_size)
            for ii in range(code.pc_matrix.size(0)):
                idx = torch.where(code.pc_matrix[ii] > 0)[0]
                for jj in idx:
                    for kk in idx:
                        if jj != kk:
                            mask[jj, kk] += 1
                            mask[kk, jj] += 1
                            mask[code.n + ii, jj] += 1
                            mask[jj, code.n + ii] += 1
            src_mask = ~(mask > 0).unsqueeze(0).unsqueeze(0)
            return src_mask

        src_mask = build_mask(code)
        self.register_buffer("src_mask", src_mask)


MENAGERIE_ZOO = "vendored-pytorch"


class _Code:
    """Minimal stand-in for the repo's ``args.code`` namespace: only the two fields
    ``ECC_Transformer`` actually reads (``n``, ``pc_matrix``) are populated."""

    pass


class _Args:
    """Minimal stand-in for the repo's argparse ``args`` namespace: only the fields
    ``ECC_Transformer.__init__`` reads."""

    pass


def build_ecct():
    # A tiny (7,4) Hamming-code-shaped binary parity-check matrix (3 x 7): small enough
    # for a fast trace while giving the Tanner-graph attention mask real structure (each
    # check row connects several distinct bit columns, same role as a real BCH/LDPC/POLAR
    # matrix loaded from Codes_DB by the repo's Get_Generator_and_Parity).
    n = 7
    pc_matrix = torch.tensor(
        [
            [1, 1, 1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 1, 0, 0, 1],
        ],
        dtype=torch.float32,
    )

    code = _Code()
    code.n = n
    code.pc_matrix = pc_matrix

    args = _Args()
    args.code = code
    args.h = 4
    args.d_model = 32
    args.N_dec = 2

    return ECC_Transformer(args, dropout=0.0)


def example_input_ecct():
    # magnitude: (B, n) channel LLR magnitudes; syndrome: (B, n-k) parity-check syndrome
    # bits -- both real-valued per the repo's forward signature (magnitude, syndrome).
    n = 7
    num_checks = 3
    magnitude = torch.rand(1, n)
    syndrome = torch.randint(0, 2, (1, num_checks)).float()
    return (magnitude, syndrome)


MENAGERIE_ENTRIES = [
    (
        "Error Correction Code Transformer (ECCT)",
        "build_ecct",
        "example_input_ecct",
        2022,
        "vendored-pytorch",
    ),
]
