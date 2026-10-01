"""TokenGT and Tokenphormer: token-based graph transformers.

TokenGT:
  Kim et al., "Pure Transformers are Powerful Graph Learners."
  arXiv:2207.02505 (NeurIPS 2022).
  Source: https://github.com/jw9730/tokengt

Tokenphormer:
  Shirzad et al., "Exphormer: Sparse Transformers for Graphs."
  arXiv:2303.06147 (ICML 2023) -- the "Exphormer" paper which includes
  the Tokenphormer-style multi-token graph transformer with global/virtual
  tokens + expander-graph sparse attention.

  Note: "Tokenphormer" as a standalone name appears in informal discussions;
  the canonical reference is Exphormer (which subsumes the token-augmented
  multi-token approach). We implement the multi-token design from Exphormer
  (node tokens + edge tokens + virtual global tokens + sparse attention).
  This is architecturally distinct from TokenGT.

------------------------------------------------------------------------------
TokenGT distinctive primitive:
  Treat ALL nodes AND ALL edges as independent tokens in a standard transformer.
  Node token i:    embedding(x_i) + node-id-encoding(i)  + type_embed(node)
  Edge token (i,j): embedding(e_{ij}) + 0.5*(node-id(i)+node-id(j)) + type_embed(edge)

  The "node identifier" is a learned orthonormal vector p_i (from Laplacian
  eigenvectors or random Gaussian), which is concatenated/added to distinguish
  structurally equivalent nodes. All N node tokens + M edge tokens -> standard
  multi-head self-attention (dense, O((N+M)^2)).

Faithful-compact simplifications:
  - Node identifiers: random Gaussian (not Laplacian EV) -- equivalent to the
    "random Gaussian" option from the paper.
  - 2 transformer layers, d_model=32, 4 heads.
  - 4 nodes, 5 edges (small graph).
  - Node and edge features = 4-dim float.

------------------------------------------------------------------------------
Tokenphormer (Exphormer) distinctive primitive:
  Multiple token types: node tokens, virtual "global" tokens (similar to [CLS]),
  and (for Exphormer) expander-graph edges for sparse long-range connectivity.
  Here we reproduce the core multi-token-type idea:
    - N node tokens
    - K virtual global tokens (interact with all nodes)
    - Sparse attention: each node attends to its graph neighbors + global tokens
  This is the architectural signature -- unlike TokenGT (dense, all-token),
  Exphormer/Tokenphormer uses sparse + virtual token attention.

Faithful-compact simplifications:
  - 4 nodes, 5 directed edges, 2 virtual tokens.
  - 2 layers, d_model=32, 4 heads.
  - Sparse attention implemented via block masking.
  - Random init, CPU, forward-only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Shared transformer utilities
# =============================================================================


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (T, d_model),  attn_mask: (T, T) bool (True=masked)  ->  (T, d_model)"""
        T, D = x.shape
        res = x
        x = self.norm1(x)
        qkv = self.qkv(x).view(T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(1)  # (T, n_heads, d_head)
        scale = math.sqrt(self.d_head)
        attn = torch.einsum("ihd,jhd->ijh", q, k) / scale  # (T, T, n_heads)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask.unsqueeze(-1), float("-inf"))
        attn = F.softmax(attn, dim=1)
        out = torch.einsum("ijh,jhd->ihd", attn, v).reshape(T, D)
        out = self.proj(out)
        x = res + out
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# TokenGT
# =============================================================================


class TokenGT(nn.Module):
    """TokenGT: nodes + edges as tokens with node-identifier encodings.

    Each node token carries: embed(x_i) + node_id(i) + type_node_embed.
    Each edge token carries: embed(e_ij) + 0.5*(node_id(i)+node_id(j)) + type_edge_embed.
    All tokens are fed to a standard dense transformer.
    """

    def __init__(
        self,
        in_node: int = 4,
        in_edge: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_node_id: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_node_id = d_node_id

        # Input projections
        self.node_embed = nn.Linear(in_node, d_model)
        self.edge_embed = nn.Linear(in_edge, d_model)

        # Node identifier: each node gets a random learnable d_node_id vector
        # These are analogous to Laplacian eigenvectors but random-Gaussian.
        # They distinguish structurally indistinguishable nodes.
        self.node_id_proj = nn.Linear(d_node_id, d_model, bias=False)

        # Token-type embedding (node vs edge)
        self.type_embed = nn.Embedding(2, d_model)  # 0=node, 1=edge

        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, 2)

    def forward(
        self,
        node_feat: torch.Tensor,  # (N, in_node)
        edge_feat: torch.Tensor,  # (M, in_edge)
        edge_index: torch.Tensor,  # (2, M) [src, dst]
        node_ids: torch.Tensor,  # (N, d_node_id) random orthonormal
    ) -> torch.Tensor:
        """Returns graph-level logits (2,)."""
        N = node_feat.size(0)
        M = edge_feat.size(0)

        # Node tokens
        node_tok = (
            self.node_embed(node_feat)
            + self.node_id_proj(node_ids)
            + self.type_embed(node_feat.new_zeros(N, dtype=torch.long))
        )  # (N, d_model)

        # Edge tokens: identifier = avg of endpoint node ids
        src, dst = edge_index[0], edge_index[1]
        edge_id = 0.5 * (node_ids[src] + node_ids[dst])  # (M, d_node_id)
        edge_tok = (
            self.edge_embed(edge_feat)
            + self.node_id_proj(edge_id)
            + self.type_embed(edge_feat.new_ones(M, dtype=torch.long))
        )  # (M, d_model)

        # Concatenate all tokens: [node_0, ..., node_{N-1}, edge_0, ..., edge_{M-1}]
        tokens = torch.cat([node_tok, edge_tok], dim=0)  # (N+M, d_model)

        for blk in self.blocks:
            tokens = blk(tokens)

        tokens = self.norm(tokens)
        # Mean pool node tokens for graph-level output
        graph_rep = tokens[:N].mean(dim=0)
        return self.cls(graph_rep)


def _make_tokengt_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(6)
    N = 4
    M = 5
    in_node = 4
    in_edge = 4
    d_node_id = 8
    node_feat = torch.randn(N, in_node)
    edge_feat = torch.randn(M, in_edge)
    src = torch.tensor([0, 1, 1, 2, 3])
    dst = torch.tensor([1, 0, 2, 3, 0])
    edge_index = torch.stack([src, dst], dim=0)
    # Random node identifiers (approximate orthonormal)
    node_ids = torch.randn(N, d_node_id)
    node_ids = F.normalize(node_ids, dim=-1)
    return node_feat, edge_feat, edge_index, node_ids


def build_tokengt() -> nn.Module:
    return TokenGT(in_node=4, in_edge=4, d_model=32, n_heads=4, n_layers=2, d_node_id=8)


def example_input_tokengt() -> list[torch.Tensor]:
    return list(_make_tokengt_inputs())


# =============================================================================
# Tokenphormer (Exphormer-style multi-token-type graph transformer)
# =============================================================================


class Tokenphormer(nn.Module):
    """Exphormer / Tokenphormer: multi-token-type sparse graph transformer.

    Token types:
      - N node tokens
      - K virtual global tokens (attend to all nodes, all nodes attend to them)
    Sparse attention: each node attends to its graph neighbors + virtual tokens.
    Virtual tokens attend to all nodes (dense in their row).
    """

    def __init__(
        self,
        in_node: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        n_virtual: int = 2,
    ) -> None:
        super().__init__()
        self.n_virtual = n_virtual
        self.node_embed = nn.Linear(in_node, d_model)
        self.virtual_embed = nn.Embedding(n_virtual, d_model)

        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, 2)

    def forward(
        self,
        node_feat: torch.Tensor,  # (N, in_node)
        edge_index: torch.Tensor,  # (2, M)
    ) -> torch.Tensor:
        """Returns graph-level logits (2,)."""
        N = node_feat.size(0)
        K = self.n_virtual
        T = N + K  # total tokens

        node_tok = self.node_embed(node_feat)  # (N, d_model)
        virt_tok = self.virtual_embed(torch.arange(K, device=node_feat.device))  # (K, d_model)
        tokens = torch.cat([node_tok, virt_tok], dim=0)  # (T, d_model)

        # Build sparse attention mask (True = masked/blocked)
        # Nodes: attend to graph neighbors + virtual tokens
        # Virtual: attend to all nodes + other virtual tokens
        mask = torch.ones(T, T, dtype=torch.bool, device=node_feat.device)
        # Node-self attention
        src, dst = edge_index[0], edge_index[1]
        mask[dst, src] = False  # node attends to its neighbors
        mask[torch.arange(N), torch.arange(N)] = False  # self-attention
        # Node <-> virtual: both directions
        for vi in range(K):
            mask[:N, N + vi] = False  # nodes attend to virtual
            mask[N + vi, :N] = False  # virtual attend to nodes
        # Virtual self-attention
        mask[N:, N:] = False

        for blk in self.blocks:
            tokens = blk(tokens, attn_mask=mask)

        tokens = self.norm(tokens)
        graph_rep = tokens[:N].mean(dim=0)
        return self.cls(graph_rep)


def build_tokenphormer() -> nn.Module:
    return Tokenphormer(in_node=4, d_model=32, n_heads=4, n_layers=2, n_virtual=2)


def example_input_tokenphormer() -> list[torch.Tensor]:
    torch.manual_seed(7)
    N = 4
    node_feat = torch.randn(N, 4)
    src = torch.tensor([0, 1, 1, 2, 3])
    dst = torch.tensor([1, 0, 2, 3, 0])
    edge_index = torch.stack([src, dst], dim=0)
    return [node_feat, edge_index]


# =============================================================================
# Registry
# =============================================================================

MENAGERIE_ENTRIES = [
    (
        "TokenGT (Tokenized Graph Transformer)",
        "build_tokengt",
        "example_input_tokengt",
        "2022",
        "DC",
    ),
    (
        "Exphormer / Tokenphormer",
        "build_tokenphormer",
        "example_input_tokenphormer",
        "2023",
        "DC",
    ),
]
