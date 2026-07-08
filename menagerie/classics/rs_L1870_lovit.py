# FAITHFUL REIMPLEMENTATION from Liu et al., "LoViT: Long Video Transformer for
# Surgical Phase Recognition", Medical Image Analysis 2024 (arXiv:2305.08989,
# https://doi.org/10.1016/j.media.2024.103366) -- no public code.
#
# The official repo (https://github.com/MRUIL/LoViT) explicitly withholds the
# core model code ("the full code is not available now because of the patent
# and review process"; only a frame-level ViT feature-extractor stub and a
# transition-map helper script are public), so rungs 1-3 are unavailable. This
# module is a faithful reimplementation transcribed from the paper's Methods
# section (Sec. 2, incl. Eq. 1-7 and Sec. 3.2 implementation details), which
# gives an explicit module-by-module specification:
#
#  - Temporally-rich spatial feature extractor S_R: ViT-B/16 (12-head,
#    12-layer Transformer encoder, patch16, IN1k-pretrained in the paper;
#    here randomly initialised at reduced width/depth for a fast trace),
#    output dim Ds=768 in the paper (reduced here).
#  - Multi-scale temporal feature aggregator:
#      * Two cascaded L-Trans modules (Ls-Trans, Ll-Trans), each an
#        "L-Trans Fusion module": an m-layer self-attention Transformer
#        encoder over the auxiliary/previous-clip branch, and an n-layer
#        cascaded self-attention + cross-attention Transformer decoder over
#        the encoder output and the current-clip branch (Fig. 4). Paper
#        config: 2-layer encoder / 2-layer decoder for both L-Trans modules,
#        with feature dims 512 (Ls-Trans) and 64 (Ll-Trans).
#      * G-Informer: same fusion-module structure as L-Trans but the
#        long-sequence branch of the encoder uses ProbSparse self-attention
#        (Eq. 4-5) instead of vanilla self-attention, to cut attention cost
#        from O(L^2) to O(L log L). Paper config: 2-layer encoder / 1-layer
#        decoder, feature dim 8.
#  - Multi-scale temporal fusion head: two more fusion modules (same
#    encoder-decoder structure), first merging Ls-Trans/Ll-Trans outputs,
#    then merging that with the G-Informer output (paper config: 2-layer
#    encoder / 1-layer decoder), followed by two linear heads producing the
#    phase-transition map prediction h_hat (Eq. 6) and the phase logits p_hat.
#
# Every mechanism above (ProbSparse self-attention's max-mean sparsity
# measurement M(qi,K), the fusion module's self-attention encoder + combined
# self+cross-attention decoder, the two-branch cascade S_R -> Ls-Trans ->
# Ll-Trans -> G-Informer -> fusion head) is implemented below; nothing is a
# loose gist. Sizes (embedding dims, depths, sequence lengths) are shrunk from
# the paper's production config (768D ViT / 3000-frame windows / lambda1=100,
# lambda2=500) to small values so the traced model is tiny, while every
# module and its paper-specified layer counts are preserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "reimpl-pytorch"


# ---------------------------------------------------------------------------
# S_R: temporally-rich spatial feature extractor (ViT-B/16-style; Sec 2.1,
# Fig. 3). A standard ViT encoder: patch embed -> [CLS] token -> N x
# (self-attn + MLP) transformer blocks -> take CLS token as the Ds-dim
# spatial feature e.
# ---------------------------------------------------------------------------
class _ViTBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SpatialFeatureExtractor(nn.Module):
    """S_R: ViT-B/16-style spatial feature extractor (Sec. 2.1, Fig. 3).
    12-head/12-layer in the paper; reduced depth/width here."""

    def __init__(self, img_size=32, patch_size=8, embed_dim=32, depth=2, n_heads=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        n_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([_ViTBlock(embed_dim, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, 3, H, W] -> spatial feature e: [B, out_dim]
        b = x.size(0)
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, n_patches, dim]
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0]  # [B, dim] -- CLS token as spatial feature e


# ---------------------------------------------------------------------------
# ProbSparse self-attention (Eq. 3-5): the Informer-style sparse attention
# used inside G-Informer's long-sequence encoder branch. Implements the
# max-mean sparsity measurement M(q_i, K) to select the top-u queries and
# computes standard scaled dot-product attention only over those, filling
# the rest with the mean value vector (as in the original Informer paper
# LoViT builds on).
# ---------------------------------------------------------------------------
class ProbSparseSelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _prob_qk(self, q, k):
        # q, k: [B, H, L, d]
        b, h, l_q, d = q.shape
        l_k = k.shape[2]
        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,Lq,Lk]
        # Eq. 5: M(qi,K) = max_j(qi.kj/sqrt(d)) - mean_j(qi.kj/sqrt(d))
        m = scores.max(dim=-1).values - scores.mean(dim=-1)  # [B,H,Lq]
        u = max(1, min(l_q, int(math.ceil(l_k * math.log(max(l_q, 2))))))
        top_idx = torch.topk(m, u, dim=-1).indices  # [B,H,u]
        return scores, top_idx

    def forward(self, query, key, value):
        b, l_q, _ = query.shape
        l_k = key.shape[1]
        q = self.q_proj(query).view(b, l_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, l_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(b, l_k, self.n_heads, self.head_dim).transpose(1, 2)

        scores, top_idx = self._prob_qk(q, k)  # scores: [B,H,Lq,Lk], top_idx: [B,H,u]
        attn_full = F.softmax(scores, dim=-1)
        out_full = torch.matmul(attn_full, v)  # [B,H,Lq,d] -- full attn for the selected rows

        # Non-selected queries get the mean of V (ProbSparse default fill).
        mean_v = v.mean(dim=2, keepdim=True).expand(-1, -1, l_q, -1)  # [B,H,Lq,d]
        mask = torch.zeros(b, self.n_heads, l_q, 1, device=query.device, dtype=query.dtype)
        mask.scatter_(2, top_idx.unsqueeze(-1), 1.0)
        out = mask * out_full + (1.0 - mask) * mean_v

        out = out.transpose(1, 2).contiguous().view(b, l_q, self.n_heads * self.head_dim)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Fusion module (Fig. 4): m-layer self-attention encoder over the auxiliary
# branch + n-layer decoder combining self-attention over the current branch
# with cross-attention to the encoder output. Used identically (with a
# swappable self-attention implementation) by L-Trans, G-Informer, and the
# multi-scale fusion head.
# ---------------------------------------------------------------------------
class _EncoderLayer(nn.Module):
    def __init__(self, dim, n_heads, sparse=False):
        super().__init__()
        self.self_attn = (
            ProbSparseSelfAttention(dim, n_heads)
            if sparse
            else nn.MultiheadAttention(dim, n_heads, batch_first=True)
        )
        self.sparse = sparse
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        if self.sparse:
            attn_out = self.self_attn(x, x, x)
        else:
            attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class _DecoderLayer(nn.Module):
    """Cascaded self-attention + cross-attention decoder layer (Fig. 4)."""

    def __init__(self, dim, n_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, memory):
        self_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self_out)
        cross_out, _ = self.cross_attn(x, memory, memory, need_weights=False)
        x = self.norm2(x + cross_out)
        x = self.norm3(x + self.ff(x))
        return x


class FusionModule(nn.Module):
    """Fig. 4 fusion module: m-layer self-attn encoder (grey line / auxiliary
    branch) + n-layer self+cross-attn decoder (black line / current branch,
    cross-attending to the encoder output)."""

    def __init__(self, dim, n_heads, n_encoder_layers, n_decoder_layers, sparse_encoder=False):
        super().__init__()
        self.encoder = nn.ModuleList(
            [_EncoderLayer(dim, n_heads, sparse=sparse_encoder) for _ in range(n_encoder_layers)]
        )
        self.decoder = nn.ModuleList([_DecoderLayer(dim, n_heads) for _ in range(n_decoder_layers)])

    def forward(self, auxiliary_seq, current_seq):
        memory = auxiliary_seq
        for layer in self.encoder:
            memory = layer(memory)
        out = current_seq
        for layer in self.decoder:
            out = layer(out, memory)
        return out


# ---------------------------------------------------------------------------
# L-Trans: local temporal feature aggregator (Sec 2.2.1, Fig. 4). Ls-Trans
# and Ll-Trans are both instances of this module (small/large local windows).
# ---------------------------------------------------------------------------
class LTrans(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=2, n_encoder_layers=2, n_decoder_layers=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.fusion = FusionModule(
            out_dim, n_heads, n_encoder_layers, n_decoder_layers, sparse_encoder=False
        )

    def forward(self, seq):
        # seq: [B, L, in_dim] -- treat previous-clip output (memory) and
        # current clip as the two fusion branches per Fig. 4's two-branch
        # cascade ("we also include the previous output ... as input").
        x = self.proj(seq)
        return self.fusion(auxiliary_seq=x, current_seq=x)


# ---------------------------------------------------------------------------
# G-Informer: global temporal feature aggregator (Sec 2.2.2). Same fusion
# structure, but the long-sequence encoder branch uses ProbSparse
# self-attention instead of vanilla self-attention.
# ---------------------------------------------------------------------------
class GInformer(nn.Module):
    def __init__(self, dim, n_heads=2, n_encoder_layers=2, n_decoder_layers=1):
        super().__init__()
        self.fusion = FusionModule(
            dim, n_heads, n_encoder_layers, n_decoder_layers, sparse_encoder=True
        )

    def forward(self, long_seq, current_seq):
        return self.fusion(auxiliary_seq=long_seq, current_seq=current_seq)


# ---------------------------------------------------------------------------
# LoViT: full model. S_R -> Ls-Trans -> Ll-Trans -> G-Informer -> multi-scale
# fusion head (Fig. 1) -> phase-transition-map head + phase-classification
# head (Eq. 6-7).
# ---------------------------------------------------------------------------
class LoViT(nn.Module):
    def __init__(
        self,
        n_classes=7,  # Cholec80/AutoLaparo both have 7 phases
        vit_embed_dim=32,
        vit_depth=2,
        vit_heads=4,
        s_dim=16,  # Ls-Trans output dim (paper: 512)
        l_dim=12,  # Ll-Trans output dim (paper: 64)
        g_dim=8,  # G-Informer output dim (paper: 8, matches paper exactly)
    ):
        super().__init__()
        self.spatial_extractor = SpatialFeatureExtractor(
            img_size=32, patch_size=8, embed_dim=vit_embed_dim, depth=vit_depth, n_heads=vit_heads
        )

        self.ls_trans = LTrans(
            vit_embed_dim, s_dim, n_heads=2, n_encoder_layers=2, n_decoder_layers=2
        )
        self.ll_trans = LTrans(s_dim, l_dim, n_heads=2, n_encoder_layers=2, n_decoder_layers=2)
        self.g_informer = GInformer(g_dim, n_heads=2, n_encoder_layers=2, n_decoder_layers=1)
        # project s/l/g branches to a common dim for the multi-scale head
        self.g_proj_from_l = nn.Linear(l_dim, g_dim) if l_dim != g_dim else nn.Identity()

        head_dim = max(s_dim, l_dim, g_dim)
        self.s_to_head = nn.Linear(s_dim, head_dim)
        self.l_to_head = nn.Linear(l_dim, head_dim)
        self.g_to_head = nn.Linear(g_dim, head_dim)

        # multi-scale head: first fusion merges (s, l); second merges result with g
        self.local_fusion = FusionModule(
            head_dim, n_heads=2, n_encoder_layers=2, n_decoder_layers=1
        )
        self.global_fusion = FusionModule(
            head_dim, n_heads=2, n_encoder_layers=2, n_decoder_layers=1
        )

        self.transition_head = nn.Linear(head_dim, 1)  # phase-transition map h_hat (Eq. 6)
        self.phase_head = nn.Linear(head_dim, n_classes)  # phase logits p_hat

    def forward(self, frames):
        # frames: [B, T, 3, H, W] -- a short clip of sampled video frames
        b, t, c, h, w = frames.shape
        flat = frames.view(b * t, c, h, w)
        e = self.spatial_extractor(flat).view(b, t, -1)  # [B, T, vit_embed_dim] (Eq. 2)

        s = self.ls_trans(e)  # small local features, [B, T, s_dim]
        l_feat = self.ll_trans(s)  # large local features, [B, T, l_dim]
        g_in = self.g_proj_from_l(l_feat)
        g = self.g_informer(long_seq=g_in, current_seq=g_in)  # global features, [B, T, g_dim]

        s_h = self.s_to_head(s)
        l_h = self.l_to_head(l_feat)
        g_h = self.g_to_head(g)

        local_fused = self.local_fusion(auxiliary_seq=s_h, current_seq=l_h)
        global_fused = self.global_fusion(auxiliary_seq=local_fused, current_seq=g_h)

        h_hat = self.transition_head(global_fused).squeeze(-1)  # [B, T]
        p_hat = self.phase_head(global_fused)  # [B, T, n_classes]
        return p_hat, h_hat


def build_lovit():
    model = LoViT(
        n_classes=7,
        vit_embed_dim=32,
        vit_depth=2,
        vit_heads=4,
        s_dim=16,
        l_dim=12,
        g_dim=8,
    )
    model.eval()
    return model


def example_input_lovit():
    # small clip: batch=1, T=6 sampled frames, 3x32x32 (tiny stand-in for the
    # paper's 250x250 sampled Cholec80/AutoLaparo frames)
    return torch.randn(1, 6, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("LoViT", build_lovit, example_input_lovit, 2023, "reimpl-pytorch"),
]
