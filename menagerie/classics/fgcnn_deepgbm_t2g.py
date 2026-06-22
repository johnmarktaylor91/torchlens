"""FGCNN, DeepGBM-NN, and T2G-Former: three CTR/tabular deep learning models.

FGCNN:
  Liu et al., "Feature Generation by Convolutional Neural Network for Click-Through
  Rate Prediction", WWW 2019. arXiv:1904.04447
  Source: https://github.com/xue-pai/FuxiCTR

  CNN over the embedding matrix (fields as spatial dim) + recombination layer
  that generates NEW features, then combined with inner product DNN (IPNN/DNN).

DeepGBM:
  Ke et al., "DeepGBM: A Deep Learning Framework Distilled by GBDT for Online
  Prediction Tasks", KDD 2019. arXiv:1909.01083
  Source: https://github.com/motefly/DeepGBM

  Two-branch model:
    GBDT2NN: NN approximating tree leaf indices from categorical features.
             Dense-net (FC layers) that mimics GBDT leaf outputs.
    CatNN:   FM-style categorical feature interaction branch.
  Both branches concatenated -> logit.
  NOTE: actual GBDT training is not reproduced (requires LightGBM);
  we implement the GBDT2NN dense-net + CatNN NN portions as random-init,
  documenting the tree-distillation as the training-time step.

T2G-Former:
  Ji et al., "T2G-Former: Organizing Tabular Features into Relation Graphs
  Addresses the Challenge of Missing Homophily", AAAI 2023. arXiv:2211.16887
  Source: https://github.com/jyansir/t2g-former

  Builds a feature-interaction graph (estimated relation matrix) and runs
  a graph-transformer (multi-head attention over tabular feature tokens).
  Includes a learnable relation estimator + graph-aware attention.

All models:
  - Tabular input: (B, num_fields) integer field indices.
  - Compact scale: num_fields=8-10, embed_dim=8, batch=2.
  - Random init, CPU, trace+draw verified 2026-06-21.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared hypers
NUM_FIELDS = 8
EMBED_DIM = 8
VOCAB = 50
BATCH = 2


def _mlp(in_dim: int, hidden: List[int], out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ============================================================
# 1. FGCNN (Feature Generation CNN)
# ============================================================


class RecombinationLayer(nn.Module):
    """FGCNN recombination layer: generates new feature combinations.

    After CNN extracts local patterns in the embedding map,
    the recombination layer takes CNN outputs (B, C_out, F', E) and
    uses a dense projection to generate K new features of dimension E
    (Eq 5-6 in paper). These new features augment the original embedding.
    """

    def __init__(
        self, cnn_out_channels: int, cnn_out_fields: int, embed_dim: int, num_new_features: int
    ) -> None:
        super().__init__()
        in_dim = cnn_out_channels * cnn_out_fields * embed_dim
        out_dim = num_new_features * embed_dim
        self.proj = nn.Linear(in_dim, out_dim)
        self.embed_dim = embed_dim
        self.num_new_features = num_new_features

    def forward(self, cnn_out: torch.Tensor) -> torch.Tensor:
        # cnn_out: (B, C, F', E) -- CNN output over embedding map
        B = cnn_out.shape[0]
        flat = cnn_out.flatten(1)  # (B, C*F'*E)
        new_feats = F.relu(self.proj(flat))  # (B, K*E)
        return new_feats.view(B, self.num_new_features, self.embed_dim)  # (B, K, E)


class FGCNNModel(nn.Module):
    """FGCNN: CNN over embedding map + recombination + IPNN/DNN.

    Architecture (Fig 1):
      Embedding -> reshape to (B, 1, F, E) (single-channel image) ->
        [Conv2d(1, C, kernel=(k,1)) + MaxPool over F -> Recombination] x num_cnn_layers ->
        new_features (from recombination) concat with original embeddings ->
        InnerProduct DNN (IP-NN) -> logit

    The key contribution: CNN generates extra features that augment the original
    embedding before the final DNN, capturing local correlations.
    """

    def __init__(
        self,
        num_fields: int = NUM_FIELDS,
        embed_dim: int = EMBED_DIM,
        vocab: int = VOCAB,
        cnn_channels: List[int] = None,
        cnn_kernel_heights: List[int] = None,
        num_new_features: int = 4,
        dnn_hidden: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [4, 4]
        if cnn_kernel_heights is None:
            cnn_kernel_heights = [3, 3]
        if dnn_hidden is None:
            dnn_hidden = [64, 32]

        self.embedding = nn.Embedding(vocab, embed_dim)
        self.num_fields = num_fields
        self.embed_dim = embed_dim

        # CNN layers over the (F, E) embedding image
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.recomb_layers = nn.ModuleList()

        in_channels = 1
        cur_fields = num_fields
        total_new_features = 0

        for i, (out_ch, kh) in enumerate(zip(cnn_channels, cnn_kernel_heights)):
            # Conv over (F, E): kernel height over F-axis, width=1 (keep E)
            padding = kh // 2
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_ch, kernel_size=(kh, 1), padding=(padding, 0))
            )
            # Max-pool halves F (if pool_size=2)
            pool_size = min(2, cur_fields)
            self.pool_layers.append(nn.MaxPool2d(kernel_size=(pool_size, 1), stride=(pool_size, 1)))
            pooled_fields = max(1, cur_fields // pool_size)
            self.recomb_layers.append(
                RecombinationLayer(out_ch, pooled_fields, embed_dim, num_new_features)
            )
            total_new_features += num_new_features
            in_channels = out_ch
            cur_fields = pooled_fields

        # Inner-product neural network (IPNN) input:
        # original F embeddings + all generated features
        total_fields = num_fields + total_new_features
        # Inner product: all pairwise dot products -> num_pairs scalar features
        num_pairs = total_fields * (total_fields - 1) // 2
        dnn_in = total_fields * embed_dim + num_pairs
        self.dnn = _mlp(dnn_in, dnn_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, nF)
        emb = self.embedding(x.long())  # (B, nF, E)
        B, nF, E = emb.shape

        # CNN path: treat embedding as (B, 1, nF, E) image
        cnn_in = emb.unsqueeze(1)  # (B, 1, nF, E)
        generated_features = []

        h = cnn_in
        for conv, pool, recomb in zip(self.conv_layers, self.pool_layers, self.recomb_layers):
            h = torch.relu(conv(h))  # (B, C_out, nF', E)
            h = pool(h)  # (B, C_out, F'//2, E)
            new_feats = recomb(h)  # (B, K, E)
            generated_features.append(new_feats)

        # Concatenate original + generated features: (B, nF + total_new, E)
        all_fields = [emb] + generated_features
        all_emb = torch.cat(all_fields, dim=1)  # (B, total_F, E)

        # IPNN: inner products + flatten
        flat = all_emb.flatten(1)  # (B, total_F * E)
        # Pairwise inner products
        n = all_emb.shape[1]
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((all_emb[:, i, :] * all_emb[:, j, :]).sum(-1, keepdim=True))
        ip = torch.cat(pairs, dim=-1)  # (B, num_pairs)

        h_dnn = torch.cat([flat, ip], dim=-1)  # (B, total_F*E + num_pairs)
        return self.dnn(h_dnn).squeeze(-1)


# ============================================================
# 2. DeepGBM-NN
# ============================================================


class GBDT2NN(nn.Module):
    """GBDT2NN: Dense NN approximating GBDT leaf outputs.

    The GBDT2NN branch (Section 3.1 of DeepGBM paper) takes the same
    continuous/embedding features and learns to approximate what GBDT
    trees would output (leaf index distributions). At inference,
    the NN approximates the tree ensemble.

    In this random-init reproduction:
    - Input: flattened embeddings (same as a GBDT-style feature vector)
    - Architecture: stacked FC blocks mimicking the 'leaf embedding' lookup
      that GBDT2NN uses after tree-distillation training
    - num_trees: how many trees are distilled (each has leaf_dim leaf nodes)
    - Output: (B, num_trees * leaf_dim) -- approximate leaf activations
    """

    def __init__(
        self, in_dim: int, num_trees: int = 4, leaf_dim: int = 4, hidden_dim: int = 32
    ) -> None:
        super().__init__()
        self.num_trees = num_trees
        self.leaf_dim = leaf_dim
        # Per-tree FC: approximate leaf-index distribution
        self.tree_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, leaf_dim),
                    nn.Softmax(dim=-1),  # leaf probability distribution
                )
                for _ in range(num_trees)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        tree_outputs = [net(x) for net in self.tree_nets]  # list of (B, leaf_dim)
        return torch.cat(tree_outputs, dim=-1)  # (B, num_trees * leaf_dim)


class CatNN(nn.Module):
    """CatNN branch: categorical FM interaction network (DeepGBM Section 3.2).

    Simple FM-style second-order interaction on categorical embeddings.
    """

    def __init__(self, num_fields: int, embed_dim: int, out_dim: int = 16) -> None:
        super().__init__()
        # FM second-order: 0.5*(sum^2 - sum of squares) -> embed_dim
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # emb: (B, F, E)
        sum_emb = emb.sum(dim=1)  # (B, E)
        sum_sq = (emb**2).sum(dim=1)  # (B, E)
        fm_out = 0.5 * (sum_emb**2 - sum_sq)  # (B, E)
        return F.relu(self.proj(fm_out))  # (B, out_dim)


class DeepGBMNN(nn.Module):
    """DeepGBM-NN: GBDT2NN + CatNN branches -> concat -> logit.

    Architecture (Fig 2):
      Embedding -> flatten ->
        GBDT2NN(flat): tree-distillation approximation network
        CatNN(emb):    FM categorical branch
      Concat -> Linear -> logit

    The GBDT training (LightGBM on numerical features) is the training-time
    distillation step; we reproduce the NN architectural form with random init.
    """

    def __init__(
        self,
        num_fields: int = NUM_FIELDS,
        embed_dim: int = EMBED_DIM,
        vocab: int = VOCAB,
        num_trees: int = 4,
        leaf_dim: int = 4,
        cat_out_dim: int = 16,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim)
        flat = num_fields * embed_dim
        self.gbdt2nn = GBDT2NN(flat, num_trees, leaf_dim)
        self.catnn = CatNN(num_fields, embed_dim, cat_out_dim)
        combined = num_trees * leaf_dim + cat_out_dim
        self.out = nn.Sequential(
            nn.Linear(combined, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x.long())  # (B, F, E)
        flat = emb.flatten(1)  # (B, F*E)
        gbdt_out = self.gbdt2nn(flat)  # (B, T*leaf_dim)
        cat_out = self.catnn(emb)  # (B, cat_out_dim)
        h = torch.cat([gbdt_out, cat_out], dim=-1)
        return self.out(h).squeeze(-1)


# ============================================================
# 3. T2G-Former (Table-to-Graph Transformer)
# ============================================================


class RelationEstimator(nn.Module):
    """T2G-Former relation estimator: learns pairwise feature relations.

    Builds a feature interaction graph by estimating a relation matrix R
    from current token representations. R_{ij} reflects the relevance of
    feature j to feature i (Section 3.2 of paper).
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        # Bilinear relation: R_ij = sigmoid(h_i W h_j)
        self.W = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, F, E)
        q = self.W(h)  # (B, F, E)
        # Relation matrix: (B, F, F)
        R = torch.bmm(q, h.transpose(1, 2)) / math.sqrt(h.shape[-1])
        return torch.sigmoid(R)  # (B, F, F) in (0,1)


class GraphTransformerLayer(nn.Module):
    """T2G-Former graph-transformer layer: graph-aware multi-head attention.

    Standard MHA but attention weights are modulated by the graph relation matrix:
      A_ij = softmax((q_i k_j / sqrt(d)) * R_ij)
    where R is the estimated feature relation matrix.
    """

    def __init__(self, embed_dim: int, num_heads: int = 2) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wo = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, h: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        # h: (B, nF, E), R: (B, nF, nF) relation matrix
        B, nF, E = h.shape
        H, D = self.num_heads, self.head_dim

        def split_heads(t):
            return t.view(B, nF, H, D).transpose(1, 2)  # (B, H, nF, D)

        q = split_heads(self.Wq(h))
        k = split_heads(self.Wk(h))
        v = split_heads(self.Wv(h))

        # Attention scores
        scores = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, nF, nF)

        # Graph modulation: broadcast R over heads
        R_h = R.unsqueeze(1).expand_as(scores)  # (B, H, nF, nF)
        scores = scores * R_h  # element-wise graph gate

        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # (B, H, nF, D)
        out = out.transpose(1, 2).contiguous().view(B, nF, E)
        out = self.Wo(out)

        h = self.norm(h + out)
        h = self.norm2(h + self.ff(h))
        return h


class T2GFormer(nn.Module):
    """T2G-Former: relation-estimated feature graph + graph-transformer over tabular.

    Architecture (Fig 2):
      Embedding -> [RelationEstimator + GraphTransformerLayer] x num_layers ->
        CLS token pooling (or mean pooling) -> DNN -> logit

    The key distinctive primitive: the RelationEstimator builds a dynamic
    feature-interaction graph from learned representations, which then guides
    graph-aware attention in the GraphTransformerLayer.
    """

    def __init__(
        self,
        num_fields: int = NUM_FIELDS,
        embed_dim: int = EMBED_DIM,
        vocab: int = VOCAB,
        num_layers: int = 2,
        num_heads: int = 2,
        dnn_hidden: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if dnn_hidden is None:
            dnn_hidden = [32]
        self.embedding = nn.Embedding(vocab, embed_dim)
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.relation_estimators = nn.ModuleList(
            [RelationEstimator(embed_dim) for _ in range(num_layers)]
        )
        self.graph_transformer_layers = nn.ModuleList(
            [GraphTransformerLayer(embed_dim, num_heads) for _ in range(num_layers)]
        )

        # Output: CLS token -> DNN -> logit
        self.dnn = _mlp(embed_dim, dnn_hidden, 1)
        self.num_fields = num_fields

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x.long())  # (B, F, E)
        B = emb.shape[0]
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, E)
        h = torch.cat([cls, emb], dim=1)  # (B, F+1, E)

        for rel_est, gtr_layer in zip(self.relation_estimators, self.graph_transformer_layers):
            # Estimate relation on full sequence (including CLS)
            R = rel_est(h)  # (B, F+1, F+1)
            h = gtr_layer(h, R)  # (B, F+1, E)

        cls_out = h[:, 0, :]  # (B, E) -- CLS token
        return self.dnn(cls_out).squeeze(-1)


# ============================================================
# Build functions and example inputs
# ============================================================


def build_fgcnn() -> nn.Module:
    """FGCNN: CNN over embedding map + recombination + IPNN."""
    return FGCNNModel()


def example_input_fgcnn() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, NUM_FIELDS))


def build_deepgbm_nn() -> nn.Module:
    """DeepGBM-NN: GBDT2NN (tree-distillation dense net) + CatNN FM branch."""
    return DeepGBMNN()


def example_input_deepgbm_nn() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, NUM_FIELDS))


def build_t2g_former() -> nn.Module:
    """T2G-Former: relation-estimated feature graph + graph-transformer."""
    return T2GFormer()


def example_input_t2g_former() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, NUM_FIELDS))


MENAGERIE_ENTRIES = [
    (
        "FGCNN (Feature Generation CNN: conv over embedding map + recombination + IPNN)",
        "build_fgcnn",
        "example_input_fgcnn",
        "2019",
        "DC",
    ),
    (
        "DeepGBM-NN (GBDT2NN tree-distillation dense net + CatNN FM branch)",
        "build_deepgbm_nn",
        "example_input_deepgbm_nn",
        "2019",
        "DC",
    ),
    (
        "T2G-Former (Feature-interaction graph estimated from tokens + graph-transformer)",
        "build_t2g_former",
        "example_input_t2g_former",
        "2023",
        "DC",
    ),
]
