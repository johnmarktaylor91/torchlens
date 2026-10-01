# FAITHFUL PORT of deepmodeling/CrystalFormer @ main (original framework: JAX / Haiku)
#
# CrystalFormer: "autoregressive transformer conditioning on Wyckoff positions
# and space group symmetry for crystal generation" (ICLR 2024). Source files:
#   crystalformer/src/transformer.py  (make_transformer -> the network fn)
#   crystalformer/src/attention.py    (MultiHeadAttention, with RoPE)
#   crystalformer/src/rope.py         (sine_table / apply_rotary_embedding)
#   crystalformer/src/wyckoff.py      (wmax_table / dof0_table static masks)
# fetched from https://raw.githubusercontent.com/deepmodeling/CrystalFormer/main/...
#
# CrystalFormer is written in JAX + dm-haiku, not PyTorch (the queue.tsv
# framework label was wrong -- verified directly against the repo). JAX/Haiku
# is a different deep-learning framework entirely, not a missing base-lib pip
# package, so per the ladder this is a rung-3 FAITHFUL PORT: every real
# computation in `network()` (the hk.transform'd forward function) and
# `MultiHeadAttention.__call__` (with rotary position embeddings) is
# transcribed into torch, preserving shapes, masking logic, normalization
# order, and the interleaved-token (W/A/X/Y/Z per atom) sequence structure
# exactly as authored. No architecture was invented or guessed from a paper
# description -- every branch below has a corresponding line in the real
# JAX source.
#
# Notes on the port:
#  - hk.Linear/hk.Sequential/hk.LayerNorm -> nn.Linear/nn.Sequential/nn.LayerNorm.
#  - hk.get_parameter(fixed-shape embedding tables) -> nn.Parameter of the
#    same shape (g_embedding_table, w_embedding_table, a_embedding_table,
#    c_embedding_uncond).
#  - `wmax_table` (230,) and `dof0_table` (230, 28) in the real repo are
#    static, non-trainable lookup tables computed offline from a bundled
#    Wyckoff-position CSV (crystallographic space-group data, not learned
#    architecture). That CSV is not shippable data here, so these are
#    represented as fixed (non-trainable) buffers of the same shape/dtype
#    role, populated with structurally-plausible placeholder values (a
#    handful of allowed Wyckoff slots per space group, and dof==0 marked for
#    the first slot) -- every masking/renormalization computation that reads
#    them is transcribed verbatim.
#  - RoPE (`sine_table` / `apply_rotary_embedding` from rope.py) is
#    transcribed verbatim into torch.
#  - `jax.scipy.special.logsumexp(...)` -> `torch.logsumexp(...)`;
#    `jax.nn.gelu` -> `F.gelu`; `jax.nn.softplus` -> `F.softplus`;
#    `jax.nn.softmax` -> `F.softmax`.
#  - Haiku's dropout-at-train-time (`hk.dropout` gated by `is_train`) is
#    ported as `nn.Dropout` gated by `self.training`, matching the semantics.
#  - The unconditional/conditional composition-embedding `jnp.where` switch
#    and every one of the five Wyckoff/atom-type constraint masks in the
#    output head are preserved exactly (steps (1)-(5) in the original code,
#    each still ported as its own explicit masked-logsumexp renormalization).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


# ---------------------------------------------------------------------------
# rope.py -- verbatim rotary position embedding helpers
# ---------------------------------------------------------------------------


def sine_table(features, length, min_timescale=1.0, max_timescale=10000.0, device=None, dtype=None):
    fraction = torch.arange(0, features, 2, dtype=torch.float32, device=device) / features
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction
    rotational_frequency = 1.0 / timescale
    sinusoid_inp = torch.einsum(
        "i,j->ij",
        torch.arange(length, dtype=torch.float32, device=device),
        rotational_frequency,
    )
    sinusoid_inp = torch.cat([sinusoid_inp, sinusoid_inp], dim=-1)
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_embedding(q, k, cos, sin):
    qlen, qheads, d = q.shape
    klen, kheads, kd = k.shape

    qcos = cos[:qlen, :].unsqueeze(1).expand(qlen, qheads, d)
    qsin = sin[:qlen, :].unsqueeze(1).expand(qlen, qheads, d)
    kcos = cos[:klen, :].unsqueeze(1).expand(klen, kheads, kd)
    ksin = sin[:klen, :].unsqueeze(1).expand(klen, kheads, kd)

    out_q = (q * qcos) + (rotate_half(q) * qsin)
    out_k = (k * kcos) + (rotate_half(k) * ksin)
    return out_q, out_k


# ---------------------------------------------------------------------------
# attention.py -- verbatim multi-head attention with RoPE
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """(Multi-Head) Attention module, ported from
    crystalformer/src/attention.py (itself adapted from dm-haiku's example
    transformer attention module), including rotary position embeddings."""

    def __init__(
        self,
        num_heads,
        key_size,
        model_size=None,
        value_size=None,
        with_bias=True,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads
        self.dropout_rate = dropout_rate

        # linear projections take the model_size input embedding to num_heads*head_size
        self.query_proj = nn.Linear(self.model_size, num_heads * self.key_size, bias=with_bias)
        self.key_proj = nn.Linear(self.model_size, num_heads * self.key_size, bias=with_bias)
        self.value_proj = nn.Linear(self.model_size, num_heads * self.value_size, bias=with_bias)
        self.final_proj = nn.Linear(num_heads * self.value_size, self.model_size, bias=with_bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None):
        seq_len_q = query.shape[-2]
        seq_len_k = key.shape[-2]

        query_heads = self.query_proj(query).reshape(seq_len_q, self.num_heads, self.key_size)
        key_heads = self.key_proj(key).reshape(seq_len_k, self.num_heads, self.key_size)
        value_heads = self.value_proj(value).reshape(seq_len_k, self.num_heads, self.value_size)

        sin, cos = sine_table(features=self.key_size, length=seq_len_q, device=query.device)
        query_heads, key_heads = apply_rotary_embedding(query_heads, key_heads, cos, sin)

        attn_logits = torch.einsum("thd,Thd->htT", query_heads, key_heads)
        attn_logits = attn_logits / math.sqrt(self.key_size)

        if mask is not None:
            attn_logits = torch.where(mask, attn_logits, torch.full_like(attn_logits, -1e30))
        attn_weights = F.softmax(attn_logits, dim=-1)

        if self.training:
            attn_weights = self.dropout(attn_weights)

        attn = torch.einsum("htT,Thd->thd", attn_weights, value_heads)
        attn = attn.reshape(seq_len_q, -1)

        return self.final_proj(attn)


# ---------------------------------------------------------------------------
# transformer.py -- verbatim CrystalFormer network
# ---------------------------------------------------------------------------


class CrystalFormer(nn.Module):
    """Space-group-conditioned autoregressive crystal-structure transformer,
    ported from crystalformer/src/transformer.py:make_transformer's inner
    `network()` function (the real Haiku-transformed forward pass)."""

    def __init__(
        self,
        Nf,
        Kx,
        Kl,
        n_max,
        h0_size,
        num_layers,
        num_heads,
        key_size,
        model_size,
        embed_size,
        atom_types,
        wyck_types,
        dropout_rate=0.0,
        attn_dropout=0.1,
        widening_factor=4,
        sigmamin=1e-3,
    ):
        super().__init__()
        self.Nf = Nf
        self.Kx = Kx
        self.Kl = Kl
        self.n_max = n_max
        self.num_layers = num_layers
        self.atom_types = atom_types
        self.wyck_types = wyck_types
        self.dropout_rate = dropout_rate
        self.sigmamin = sigmamin

        self.coord_types = 3 * Kx
        self.lattice_types = Kl + 2 * 6 * Kl
        self.output_size = max(atom_types + self.lattice_types, self.coord_types, wyck_types)

        init_std = 0.01

        # embedding tables (hk.get_parameter)
        self.g_embedding_table = nn.Parameter(torch.randn(230, embed_size) * init_std)
        self.w_embedding_table = nn.Parameter(torch.randn(wyck_types, embed_size) * init_std)
        self.a_embedding_table = nn.Parameter(torch.randn(atom_types, embed_size) * init_std)
        self.c_embedding_uncond = nn.Parameter(torch.randn(embed_size) * init_std)

        # static Wyckoff-constraint lookup tables (see module docstring: real
        # values are computed offline from a bundled crystallographic CSV in
        # the source repo; placeholders here preserve shape/dtype/role only)
        wmax_table = torch.full((230,), 4, dtype=torch.long)
        dof0_table = torch.zeros((230, 28), dtype=torch.bool)
        dof0_table[:, 1] = True
        self.register_buffer("wmax_table", wmax_table)
        self.register_buffer("dof0_table", dof0_table)

        # g_logit head
        self.g_head = nn.Sequential(
            nn.Linear(embed_size, h0_size),
            nn.GELU(),
            nn.Linear(h0_size, 230),
        )

        # w_logit head (input: concat(c_embeddings, g_embeddings) = 2*embed_size)
        self.w_head = nn.Sequential(
            nn.Linear(2 * embed_size, h0_size),
            nn.GELU(),
            nn.Linear(h0_size, wyck_types),
        )

        wa_in = 3 * embed_size + 1  # c_emb + g_emb + w_emb + multiplicity scalar
        self.hW_proj = nn.Linear(wa_in, model_size)

        ha_in = 3 * embed_size  # c_emb + g_emb + a_emb
        self.hA_proj = nn.Linear(ha_in, model_size)

        hxyz_in = 2 * embed_size + 2 * Nf  # c_emb + g_emb + sin/cos Fourier features
        self.hX_proj = nn.Linear(hxyz_in, model_size)
        self.hY_proj = nn.Linear(hxyz_in, model_size)
        self.hZ_proj = nn.Linear(hxyz_in, model_size)

        self.attn_blocks = nn.ModuleList(
            [
                MultiHeadAttention(
                    num_heads=num_heads,
                    key_size=key_size,
                    model_size=model_size,
                    dropout_rate=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.attn_norms = nn.ModuleList([nn.LayerNorm(model_size) for _ in range(num_layers)])
        self.dense_norms = nn.ModuleList([nn.LayerNorm(model_size) for _ in range(num_layers)])
        self.dense_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(model_size, widening_factor * model_size),
                    nn.GELU(),
                    nn.Linear(widening_factor * model_size, model_size),
                )
                for _ in range(num_layers)
            ]
        )
        self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_layers)])
        self.dense_dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_layers)])

        self.final_norm = nn.LayerNorm(model_size)
        self.out_proj = nn.Linear(model_size, self.output_size)

    def _renormalize(self, h_x):
        n = h_x.shape[0]
        x_logit, x_loc, x_kappa = torch.split(
            h_x[:, : self.coord_types], [self.Kx, self.Kx, self.Kx], dim=-1
        )
        x_logit = x_logit - torch.logsumexp(x_logit, dim=1, keepdim=True)
        x_kappa = F.softplus(x_kappa)
        h_x = torch.cat(
            [
                x_logit,
                x_loc,
                x_kappa,
                torch.zeros(n, self.output_size - self.coord_types, device=h_x.device),
            ],
            dim=-1,
        )
        return h_x

    def forward(self, composition, G, XYZ, A, W, M):
        """
        Args:
            composition: (atom_types,)
            G: scalar long, space group id 1..230
            XYZ: (n, 3) fractional coordinates
            A: (n,) element type
            W: (n,) wyckoff position index
            M: (n,) multiplicities (float)
        Returns:
            g_logit: (230,)
            h: (5n+1, output_size)
        """
        n = XYZ.shape[0]
        X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]

        g_idx = G - 1
        w_max = self.wmax_table[g_idx]

        g_embeddings = self.g_embedding_table[g_idx]
        w_embeddings = self.w_embedding_table[W]
        a_embeddings = self.a_embedding_table[A]

        is_comp_provided = composition.sum() > 0
        c_embeddings_cond = (self.a_embedding_table * composition[:, None]).sum(dim=0)
        c_embeddings = torch.where(is_comp_provided, c_embeddings_cond, self.c_embedding_uncond)

        g_logit = self.g_head(c_embeddings)
        g_logit = g_logit - torch.logsumexp(g_logit, dim=0)

        w_logit = self.w_head(torch.cat([c_embeddings, g_embeddings], dim=0))
        w_mask = (torch.arange(self.wyck_types, device=w_logit.device) > 0) & (
            torch.arange(self.wyck_types, device=w_logit.device) <= w_max
        )
        w_logit = torch.where(w_mask, w_logit, w_logit - 1e10)
        w_logit = w_logit - torch.logsumexp(w_logit, dim=0)

        h0 = torch.cat(
            [
                w_logit.unsqueeze(0),
                torch.zeros(1, self.output_size - self.wyck_types, device=w_logit.device),
            ],
            dim=-1,
        )
        if n == 0:
            return g_logit, h0

        mask = torch.tril(torch.ones(1, 5 * n, 5 * n, dtype=torch.bool, device=XYZ.device))

        c_rep = c_embeddings.unsqueeze(0).expand(n, -1)
        g_rep = g_embeddings.unsqueeze(0).expand(n, -1)

        hW = torch.cat([c_rep, g_rep, w_embeddings, M.reshape(n, 1)], dim=1)
        hW = self.hW_proj(hW)

        hA = torch.cat([c_rep, g_rep, a_embeddings], dim=1)
        hA = self.hA_proj(hA)

        def fourier_feats(coord):
            feats = []
            for f in range(1, self.Nf + 1):
                feats.append(torch.sin(2 * math.pi * coord[:, None] * f))
                feats.append(torch.cos(2 * math.pi * coord[:, None] * f))
            return torch.cat(feats, dim=1)

        hX = torch.cat([c_rep, g_rep, fourier_feats(X)], dim=1)
        hX = self.hX_proj(hX)
        hY = torch.cat([c_rep, g_rep, fourier_feats(Y)], dim=1)
        hY = self.hY_proj(hY)
        hZ = torch.cat([c_rep, g_rep, fourier_feats(Z)], dim=1)
        hZ = self.hZ_proj(hZ)

        h = torch.stack([hW, hA, hX, hY, hZ], dim=1)  # (n, 5, model_size)
        h = h.reshape(5 * n, -1)

        for i in range(self.num_layers):
            h_norm = self.attn_norms[i](h)
            h_attn = self.attn_blocks[i](h_norm, h_norm, h_norm, mask=mask)
            if self.training:
                h_attn = self.attn_dropouts[i](h_attn)
            h = h + h_attn

            h_norm = self.dense_norms[i](h)
            h_dense = self.dense_blocks[i](h_norm)
            if self.training:
                h_dense = self.dense_dropouts[i](h_dense)
            h = h + h_dense

        h = self.final_norm(h)
        h = self.out_proj(h)  # (5n, output_size)

        h = h.reshape(n, 5, -1)
        h_al, h_x, h_y, h_z, w_logit_seq = (
            h[:, 0, :],
            h[:, 1, :],
            h[:, 2, :],
            h[:, 3, :],
            h[:, 4, :],
        )

        h_x = self._renormalize(h_x)
        h_y = self._renormalize(h_y)
        h_z = self._renormalize(h_z)

        a_logit = h_al[:, : self.atom_types]
        w_logit_seq = w_logit_seq[:, : self.wyck_types]

        # (1) impose W_0 <= W_1 <= W_2 (or strict < when dof==0)
        idx_range = torch.arange(1, self.wyck_types, device=h.device).reshape(
            1, self.wyck_types - 1
        )
        w_mask_less_equal = idx_range < W[:, None]
        w_mask_less = idx_range <= W[:, None]
        dof0_flags = self.dof0_table[g_idx, W].unsqueeze(-1)
        w_mask1 = torch.where(dof0_flags, w_mask_less, w_mask_less_equal)
        w_mask1 = torch.cat([torch.zeros(n, 1, dtype=torch.bool, device=h.device), w_mask1], dim=1)
        w_logit_seq = w_logit_seq - torch.where(
            w_mask1, torch.full_like(w_logit_seq, 1e10), torch.zeros_like(w_logit_seq)
        )
        w_logit_seq = w_logit_seq - torch.logsumexp(w_logit_seq, dim=1, keepdim=True)

        # (2) enhance probability of pad atoms if already a type-0 atom present
        pad_col = torch.where(
            W == 0, torch.ones(n, device=h.device), torch.zeros(n, device=h.device)
        ).reshape(n, 1)
        w_mask2 = torch.cat(
            [pad_col, torch.zeros(n, self.wyck_types - 1, device=h.device)], dim=1
        ).bool()
        w_logit_seq = torch.where(w_mask2, torch.full_like(w_logit_seq, 1e10), w_logit_seq)
        w_logit_seq = w_logit_seq - torch.logsumexp(w_logit_seq, dim=1, keepdim=True)

        # (3) mask out positions past w_max for this space group
        w_idx = torch.arange(self.wyck_types, device=h.device)
        w_logit_seq = torch.where(w_idx <= w_max, w_logit_seq, w_logit_seq - 1e10)
        w_logit_seq = w_logit_seq - torch.logsumexp(w_logit_seq, dim=1, keepdim=True)

        # (4) if w != 0 mask out the pad atom slot, else mask out true atoms
        a_mask = torch.cat(
            [
                (W > 0).reshape(n, 1),
                (W == 0).reshape(n, 1).expand(n, self.atom_types - 1),
            ],
            dim=1,
        )
        a_logit = a_logit + torch.where(
            a_mask, torch.full_like(a_logit, -1e10), torch.zeros_like(a_logit)
        )
        a_logit = a_logit - torch.logsumexp(a_logit, dim=1, keepdim=True)

        # (5) composition constraint (conditional vs unconditional generation)
        comp_constraint_mask = torch.where(
            composition == 0,
            torch.ones_like(composition, dtype=torch.float32),
            torch.zeros_like(composition, dtype=torch.float32),
        )
        effective_mask = torch.where(
            is_comp_provided, comp_constraint_mask, torch.zeros_like(comp_constraint_mask)
        )
        a_mask2 = effective_mask.reshape(1, self.atom_types).expand(n, -1).clone()
        a_mask2[:, 0] = 0
        a_logit = a_logit + torch.where(
            a_mask2.bool(), torch.full_like(a_logit, -1e10), torch.zeros_like(a_logit)
        )
        a_logit = a_logit - torch.logsumexp(a_logit, dim=1, keepdim=True)

        w_logit_seq = torch.cat(
            [w_logit_seq, torch.zeros(n, self.output_size - self.wyck_types, device=h.device)],
            dim=-1,
        )

        # lattice part
        l_logit, mu, sigma = torch.split(
            h_al[:, self.atom_types : self.atom_types + self.lattice_types],
            [self.Kl, self.Kl * 6, self.Kl * 6],
            dim=-1,
        )
        l_logit = l_logit - torch.logsumexp(l_logit, dim=1, keepdim=True)
        sigma = F.softplus(sigma) + self.sigmamin

        h_al = torch.cat(
            [
                a_logit,
                l_logit,
                mu,
                sigma,
                torch.zeros(
                    n, self.output_size - self.atom_types - self.lattice_types, device=h.device
                ),
            ],
            dim=-1,
        )

        h_out = torch.stack([h_al, h_x, h_y, h_z, w_logit_seq], dim=1)  # (n, 5, output_size)
        h_out = h_out.reshape(5 * n, self.output_size)

        h_out = torch.cat([h0, h_out], dim=0)

        return g_logit, h_out


def build_crystalformer():
    return CrystalFormer(
        Nf=2,
        Kx=4,
        Kl=2,
        n_max=6,
        h0_size=16,
        num_layers=2,
        num_heads=2,
        key_size=8,
        model_size=16,
        embed_size=8,
        atom_types=10,
        wyck_types=6,
        dropout_rate=0.0,
        attn_dropout=0.0,
        widening_factor=2,
    )


def example_input_crystalformer():
    torch.manual_seed(0)
    atom_types = 10
    n = 3
    composition = torch.zeros(atom_types)
    G = torch.tensor(123)
    XYZ = torch.rand(n, 3)
    A = torch.randint(0, atom_types, (n,))
    W = torch.randint(0, 6, (n,))
    M = torch.rand(n)
    return (composition, G, XYZ, A, W, M)


MENAGERIE_ENTRIES = [
    (
        "CrystalFormer",
        build_crystalformer,
        example_input_crystalformer,
        2024,
        "PORT",
    ),
]
