# SOURCE: vendored from yoniLc/DDECC @ main
# (DDECC.py, Codes.py)
#
# "Denoising Diffusion Error Correction Codes" (DDECC / DDECCT), Choukroun &
# Wolf, ICLR 2023, arXiv:2209.13533. yoniLc/DDECC is the OFFICIAL repo. The
# DDECCT model is a transformer-encoder-style denoiser used as the reverse
# process of a diffusion model over channel-noise magnitude/syndrome
# features for soft error-correcting-code decoding; it reuses (a locally
# defined) multi-head self-attention encoder stack conditioned on a
# diffusion time-step embedding and a code-derived attention mask built
# from the parity-check matrix.
#
# Vendored verbatim below: DDECCT, Encoder, EncoderLayer, SublayerConnection,
# MultiHeadedAttention, PositionwiseFeedForward, clones, EMA (DDECC.py), plus
# sign_to_bin/bin_to_sign/row_reduce/get_generator (Codes.py) needed to build
# a code object's generator/parity matrices. Only the relative
# `from Codes import sign_to_bin, bin_to_sign` import was flattened into this
# single file; no architectural changes. `ConditionalLinear`/`ConditionalModel`
# in the original file are an unrelated toy 2D diffusion baseline (not part of
# the DDECCT graph) and are omitted.
#
# Ref: https://github.com/yoniLc/DDECC/blob/main/DDECC.py
# Ref: https://github.com/yoniLc/DDECC/blob/main/Codes.py

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# From Codes.py (verbatim, only the pieces DDECCT / code construction need).
# ---------------------------------------------------------------------------
def sign_to_bin(x):
    return 0.5 * (1 - x)


def bin_to_sign(x):
    return 1 - 2 * x


def row_reduce(mat, ncols=None):
    assert mat.ndim == 2
    ncols = mat.shape[1] if ncols is None else ncols
    mat_row_reduced = mat.copy()
    p = 0
    for j in range(ncols):
        idxs = p + np.nonzero(mat_row_reduced[p:, j])[0]
        if idxs.size == 0:
            continue
        mat_row_reduced[[p, idxs[0]], :] = mat_row_reduced[[idxs[0], p], :]
        idxs = np.nonzero(mat_row_reduced[:, j])[0].tolist()
        idxs.remove(p)
        mat_row_reduced[idxs, :] = mat_row_reduced[idxs, :] ^ mat_row_reduced[p, :]
        p += 1
        if p == mat_row_reduced.shape[0]:
            break
    return mat_row_reduced, p


def get_generator(pc_matrix_):
    assert pc_matrix_.ndim == 2
    pc_matrix = pc_matrix_.copy().astype(bool).transpose()
    pc_matrix_I = np.concatenate((pc_matrix, np.eye(pc_matrix.shape[0], dtype=bool)), axis=-1)
    pc_matrix_I, p = row_reduce(pc_matrix_I, ncols=pc_matrix.shape[1])
    return row_reduce(pc_matrix_I[p:, pc_matrix.shape[1] :])[0]


# ---------------------------------------------------------------------------
# From DDECC.py (verbatim, minus the unrelated ConditionalLinear /
# ConditionalModel toy-2D-diffusion baseline classes).
# ---------------------------------------------------------------------------
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if N > 1:
            self.norm2 = LayerNorm(layer.size)

    def forward(self, x, mask, time_emb):
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, mask)
            # x = time_emb*x
            if idx == len(self.layers) // 2 and len(self.layers) > 1:
                x = self.norm2(x)
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
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
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class DDECCT(nn.Module):
    def __init__(self, args, device, dropout=0):
        super(DDECCT, self).__init__()
        ####
        self.n_steps = args.N_steps
        self.d_model = args.d_model
        self.sigma = args.sigma
        self.register_buffer("pc_matrix", args.code.pc_matrix.transpose(0, 1).float())
        self.device = device
        #
        betas = torch.linspace(1e-3, 1e-2, self.n_steps)
        betas = betas * 0 + self.sigma
        self.betas = betas.view(-1, 1)
        self.betas_bar = torch.cumsum(self.betas, 0).view(-1, 1)
        self.ema = EMA(0.9, flag_run=True)
        ###
        self.line_search = False
        ###
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
        self.time_embed = nn.Embedding(self.n_steps, args.d_model)

        self.get_mask(code)
        ###
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, y, time_step):
        magnitude = torch.abs(y)
        syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long().float(), self.pc_matrix) % 2
        syndrome = bin_to_sign(syndrome)
        emb = torch.cat([magnitude, syndrome], -1).unsqueeze(-1)
        emb = self.src_embed.unsqueeze(0) * emb
        time_emb = self.time_embed(time_step).view(-1, 1, self.d_model)
        emb = time_emb * emb
        emb = self.decoder(emb, self.src_mask, time_emb)
        return self.out_fc(self.oned_final_embed(emb).squeeze(-1))

    def p_sample(self, yt):
        # Single sampling from the real p dist.
        sum_syndrome = (
            (torch.matmul(sign_to_bin(torch.sign(yt.to(self.device))), self.pc_matrix) % 2)
            .round()
            .long()
            .sum(-1)
        )
        t = sum_syndrome
        # Model output
        noise_mul_pred = self(yt.to(self.device), sum_syndrome.to(self.device)).cpu()
        noise_add_pred = yt - torch.sign(-noise_mul_pred * torch.sign(yt))
        factor = torch.sqrt(self.betas_bar[t]) * self.betas[t] / (self.betas_bar[t] + self.betas[t])
        alpha_final = 1
        if self.line_search:
            alpha = torch.linspace(1, 20, 20).unsqueeze(0).unsqueeze(0)
            new_synd = (
                (
                    torch.matmul(
                        sign_to_bin(
                            torch.sign(
                                yt.unsqueeze(-1) - alpha * (noise_add_pred * factor).unsqueeze(-1)
                            )
                        ).permute(0, 2, 1),
                        self.pc_matrix.cpu(),
                    )
                    % 2
                )
                .round()
                .long()
                .sum(-1)
            )
            alpha_final = alpha.squeeze(0)[:, new_synd.argmin(-1).unsqueeze(-1)].squeeze(0)
        yt_1 = yt - alpha_final * noise_add_pred * factor
        yt_1[t == 0] = yt[t == 0]
        return (yt_1), t

    def p_sample_loop(self, cur_y):
        # Iterative sampling from the real p dist.
        res = []
        synd_all = []
        for it in range(self.pc_matrix.shape[1]):
            cur_y, curr_synd = self.p_sample(cur_y)
            synd_all.append(curr_synd)
            res.append(cur_y)
        synd_all = torch.stack(synd_all).t().cpu()
        aa = (synd_all == 0).int() * 2 - 1
        idx = torch.arange(aa.shape[1], 0, -1)
        idx_conv = torch.argmax(aa * idx, 1, keepdim=True)
        return cur_y, res, idx_conv.view(-1), synd_all

    def loss(self, x_0):
        t = torch.randint(0, self.n_steps, size=(x_0.shape[0] // 2 + 1,))
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[: x_0.shape[0]].long()
        e = torch.randn_like(x_0)
        noise_factor = torch.sqrt(self.betas_bar[t]).to(x_0.device)
        h = torch.from_numpy(np.random.rayleigh(x_0.size(0), x_0.size(1))).float()
        h = 1.0
        yt = h * x_0 * 1 + e * noise_factor
        sum_syndrome = (
            (torch.matmul(sign_to_bin(torch.sign(yt.to(self.device))), self.pc_matrix) % 2)
            .sum(-1)
            .long()
        )
        output = self(yt.to(self.device), sum_syndrome.to(self.device))
        z_mul = yt * x_0
        return F.binary_cross_entropy_with_logits(
            output, sign_to_bin(torch.sign(z_mul.to(self.device)))
        )

    def get_mask(self, code, no_mask=False):
        if no_mask:
            self.src_mask = None
            return

        def build_mask(code):
            mask_size = code.n + code.pc_matrix.size(0)
            mask = torch.eye(mask_size, mask_size)
            for ii in range(code.n - code.k):
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


class EMA(object):
    def __init__(self, mu=0.999, flag_run=True):
        self.mu = mu
        self.shadow = {}
        self.flag_run = flag_run

    def register(self, module):
        if self.flag_run:
            for name, param in module.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone()

    def update(self, module):
        if self.flag_run:
            for name, param in module.named_parameters():
                if param.requires_grad:
                    self.shadow[name].data = (1.0 - self.mu) * param.data + self.mu * self.shadow[
                        name
                    ].data


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo). The real Main.py
# CLI builds `args.code` from a stored (n,k) parity-check matrix loaded off
# disk (Codes_DB/*.alist or *.txt, e.g. BCH/POLAR/LDPC codes) and sizes
# `args.N_steps` off `code.pc_matrix.shape[0] + 5`. We use a tiny, valid
# (7,4) Hamming code parity-check matrix (a well-known textbook code, not an
# architectural choice) built with the repo's own `get_generator` so
# `DDECCT.get_mask` / `forward` operate on the real shapes, and shrink
# d_model/heads/N_dec for a fast CPU trace while keeping the full
# transformer-encoder + diffusion-time-embedding + parity-mask architecture.
# ---------------------------------------------------------------------------
class _Args:
    pass


class _Code:
    pass


def _build_hamming74_code():
    H = np.array(
        [
            [1, 1, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1],
        ],
        dtype=int,
    )
    G = get_generator(H)
    code = _Code()
    code.n = 7
    code.k = 4
    code.code_type = "HAMMING"
    code.generator_matrix = torch.from_numpy(G.astype(float)).transpose(0, 1).long()
    code.pc_matrix = torch.from_numpy(H.astype(float)).long()
    return code


def build_ddecct():
    torch.manual_seed(0)
    np.random.seed(0)
    code = _build_hamming74_code()
    args = _Args()
    args.code = code
    args.h = 2
    args.d_model = 8
    args.N_dec = 2
    args.sigma = 0.01
    args.N_steps = int(code.pc_matrix.shape[0]) + 5
    model = DDECCT(args, device="cpu")
    model.eval()
    return model


def example_input_ddecct():
    torch.manual_seed(0)
    y = torch.randn(2, 7)
    time_step = torch.randint(0, 8, (2,))
    return (y, time_step)


MENAGERIE_ENTRIES = [
    (
        "DDECC.DDECCT (diffusion error-correction decoder)",
        "build_ddecct",
        "example_input_ddecct",
        2023,
        MENAGERIE_ZOO,
    ),
]
