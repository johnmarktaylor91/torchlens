"""DMR: Deep Match to Rank for Sequential CTR Prediction.

Lv et al., AAAI 2020 (Alibaba).
Paper: https://ojs.aaai.org/index.php/AAAI/article/view/5346
       "Deep Match to Rank Model for Personalized Click-Through Rate Prediction"
Source: https://github.com/lvze92/DMR (TF1 official);
        PyTorch port: https://github.com/reczoo/FuxiCTR

DMR's distinctive architecture: a CTR model that EXPLICITLY BRIDGES collaborative
filtering (user-item matching) with ranking, via two parallel modules:

1. USER-TO-ITEM (U2I) MATCH MODULE:
   - An inner-product user-item matching score: <user_embedding, item_embedding>
   - This captures CF-style similarity signals
   - The score is injected as an auxiliary feature into the ranking DNN

2. ITEM-TO-ITEM (I2I) MATCH MODULE (User Interest Representation):
   - An attention-based interest extraction over click HISTORY sequence:
     for each historical item h_i, compute attention weight with target item t:
     a_i = softmax(<e_{h_i}, e_t>)  (same inner product attention as DIN)
   - Weighted sum of history embeddings: v_u = sum_i a_i * e_{h_i}
   - This is similar to DIN but the key insight is using the SAME item embedding
     space for both history (keys) and target (query), creating an
     item-to-item bridge signal

3. RANKING DNN:
   - Concatenates: user profile embedding + I2I interest rep + U2I match score
     + target item embedding + context features
   - Stacked fully-connected layers -> sigmoid -> click probability

The I2I and U2I modules provide complementary matching signals that are fused
into a standard DNN ranker.

Simplifications: embedding_dim=16, history_len=4, 2 user features, 2 item features,
1 context feature; DNN layers [64, 32]; random init, CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemEmbedding(nn.Module):
    """Item embedding: maps item_id + category_id -> embedding vector."""

    def __init__(self, item_vocab: int, cate_vocab: int, emb_dim: int) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(item_vocab, emb_dim, padding_idx=0)
        self.cate_emb = nn.Embedding(cate_vocab, emb_dim // 2, padding_idx=0)
        self.proj = nn.Linear(emb_dim + emb_dim // 2, emb_dim)

    def forward(self, item_id: torch.Tensor, cate_id: torch.Tensor) -> torch.Tensor:
        # item_id, cate_id: (B,) or (B, L)
        e_item = self.item_emb(item_id)  # (..., emb_dim)
        e_cate = self.cate_emb(cate_id)  # (..., emb_dim//2)
        return self.proj(torch.cat([e_item, e_cate], dim=-1))


class I2IAttentionModule(nn.Module):
    """Item-to-Item attention for user interest representation (DIN-style).

    Given target item embedding e_t and history item embeddings e_h (B, L, D),
    compute attention-weighted sum: v_u = sum_i softmax(<e_t, e_{h_i}>) * e_{h_i}.

    The I2I bridge: using the same embedding space for query (target) and keys (history).
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        # Activation unit: MLP on concat([e_t, e_h, e_t - e_h, e_t * e_h])
        self.attn_mlp = nn.Sequential(
            nn.Linear(4 * emb_dim, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        target_emb: torch.Tensor,  # (B, D)
        hist_emb: torch.Tensor,  # (B, L, D)
        hist_mask: torch.Tensor,  # (B, L) bool mask (True = valid)
    ) -> torch.Tensor:
        B, L, D = hist_emb.shape
        t = target_emb.unsqueeze(1).expand(B, L, D)  # (B, L, D)

        # Activation unit: element-wise interactions
        feat = torch.cat([t, hist_emb, t - hist_emb, t * hist_emb], dim=-1)  # (B, L, 4D)
        scores = self.attn_mlp(feat).squeeze(-1)  # (B, L)

        # Mask and softmax
        scores = scores.masked_fill(~hist_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)  # (B, L)
        attn = torch.nan_to_num(attn, nan=0.0)

        # Weighted sum
        v_u = (attn.unsqueeze(-1) * hist_emb).sum(dim=1)  # (B, D)
        return v_u


class U2IMatchModule(nn.Module):
    """User-to-Item matching via inner-product.

    Computes a scalar match score between a user embedding and the target item embedding.
    This score bridges CF matching into the ranking model.
    """

    def __init__(self, emb_dim: int, user_dim: int) -> None:
        super().__init__()
        # Project user features to item embedding space
        self.user_proj = nn.Linear(user_dim, emb_dim)

    def forward(self, user_feat: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        # user_feat: (B, user_dim), item_emb: (B, D)
        u = self.user_proj(user_feat)  # (B, D)
        # Inner product match score
        score = (u * item_emb).sum(dim=-1, keepdim=True)  # (B, 1)
        return score, u


class DMR(nn.Module):
    """Deep Match to Rank model.

    Inputs:
      - user_feat: (B, user_dim) -- user profile features (pre-embedded)
      - item_id, cate_id: (B,) -- target item
      - hist_item, hist_cate: (B, L) -- click history
      - hist_mask: (B, L) -- valid history mask
      - neg_hist_item, neg_hist_cate: (B, L) -- negative sampled history
        (used in I2I auxiliary loss during training; ignored at inference)
      - context_feat: (B, ctx_dim) -- context features

    Output: (B, 1) -- click probability logit
    """

    def __init__(
        self,
        item_vocab: int = 128,
        cate_vocab: int = 32,
        emb_dim: int = 16,
        user_feat_dim: int = 16,  # pre-embedded user profile
        ctx_feat_dim: int = 8,
        dnn_layers: list = None,
    ) -> None:
        super().__init__()
        if dnn_layers is None:
            dnn_layers = [64, 32]

        self.emb_dim = emb_dim
        self.item_emb = ItemEmbedding(item_vocab, cate_vocab, emb_dim)

        # I2I: attention-based user interest from history
        self.i2i = I2IAttentionModule(emb_dim)

        # U2I: CF-style match between user profile and target item
        self.u2i = U2IMatchModule(emb_dim, user_feat_dim)

        # Ranking DNN input: [user_feat | I2I v_u | U2I u_proj | U2I score | target_emb | ctx]
        dnn_in = user_feat_dim + emb_dim + emb_dim + 1 + emb_dim + ctx_feat_dim

        layers = []
        prev = dnn_in
        for h in dnn_layers:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.dnn = nn.Sequential(*layers)

    def forward(
        self,
        user_feat: torch.Tensor,  # (B, user_feat_dim)
        item_id: torch.Tensor,  # (B,)
        cate_id: torch.Tensor,  # (B,)
        hist_item: torch.Tensor,  # (B, L)
        hist_cate: torch.Tensor,  # (B, L)
        hist_mask: torch.Tensor,  # (B, L) bool
        context_feat: torch.Tensor,  # (B, ctx_feat_dim)
    ) -> torch.Tensor:
        # Target item embedding
        e_t = self.item_emb(item_id, cate_id)  # (B, D)

        # History item embeddings
        e_h = self.item_emb(hist_item, hist_cate)  # (B, L, D)

        # I2I: attention-weighted user interest representation
        v_u = self.i2i(e_t, e_h, hist_mask)  # (B, D)

        # U2I: match score + projected user embedding
        u2i_score, u_proj = self.u2i(user_feat, e_t)  # (B, 1), (B, D)

        # Concatenate all features for DNN
        dnn_in = torch.cat(
            [
                user_feat,  # user profile
                v_u,  # I2I interest representation
                u_proj,  # U2I projected user embedding
                u2i_score,  # U2I match scalar
                e_t,  # target item embedding
                context_feat,  # context
            ],
            dim=-1,
        )

        return self.dnn(dnn_in)  # (B, 1) -- click logit


def build_dmr() -> nn.Module:
    return DMR(
        item_vocab=128,
        cate_vocab=32,
        emb_dim=16,
        user_feat_dim=16,
        ctx_feat_dim=8,
        dnn_layers=[64, 32],
    )


def example_input_dmr():
    B, L = 2, 4
    user_feat = torch.randn(B, 16)
    item_id = torch.randint(1, 128, (B,))
    cate_id = torch.randint(1, 32, (B,))
    hist_item = torch.randint(1, 128, (B, L))
    hist_cate = torch.randint(1, 32, (B, L))
    hist_mask = torch.ones(B, L, dtype=torch.bool)
    hist_mask[0, 2:] = False  # some padding
    context_feat = torch.randn(B, 8)
    return [user_feat, item_id, cate_id, hist_item, hist_cate, hist_mask, context_feat]


MENAGERIE_ENTRIES = [
    ("DMR (Deep Match to Rank)", "build_dmr", "example_input_dmr", "2020", "DC"),
]
