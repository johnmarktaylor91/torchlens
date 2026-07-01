# SOURCE: vendored from luckyhzt/unicolor @ dfd77f2f1498c1944133dfcaf0a873e1a3871189
# https://raw.githubusercontent.com/luckyhzt/unicolor/dfd77f2f1498c1944133dfcaf0a873e1a3871189/framework/hybrid_tran/models/transformer.py
#
# Weng, Wu, Ji, Kot & Shi 2022 (SIGGRAPH Asia 2022) "UniColor: A Unified Framework for
# Multi-Modal Colorization with Transformer" -- `HybridTran`, the paper's Hybrid
# (Vision) Transformer that fuses a grayscale-image token stream with a color-token
# stream (from a separately-trained "Chroma-VQGAN" codebook, `framework/chroma_vqgan/`)
# plus optional point/stroke/text-derived color "condition" tokens
# (`self.cond_input`/`self.cond_emb`, scattered into the color-token grid at
# `cond_indices` before the transformer blocks) into one causal-self-attention
# transformer stack, then predicts color-token logits over the color half of the
# sequence. This is the concrete architectural contribution described in the paper
# (Sec. 3.2 "Hybrid Transformer") and used as the core predictive model in
# `framework/hybrid_tran/models/colorization.py::Colorization.forward` (the
# `Chroma_VQGAN` encoder that produces the token indices `HybridTran` consumes is a
# separately pretrained tokenizer, not part of this architecture; `HybridTran` is traced
# here directly on token-index / feature inputs exactly as `Colorization.forward` calls
# it: `self.hybrid_tran(idx_color, f_gray, cond, cond_indices)`).
#
# `HybridTran`/`SelfAttention`/`AttentionBlock` are reproduced verbatim from the
# vendored file; no computation changed.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridTran(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, input_shape, dim_gray, vocab_color):
        super().__init__()
        seq_length = input_shape[0] * input_shape[1]  # size of one image

        # input embedding stem
        self.tok_emb = nn.Embedding(vocab_color + 1, embed_dim)  # One more index for masked tokens
        self.gray_input = nn.Linear(dim_gray, embed_dim)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, 2 * seq_length, embed_dim)
        )  # 2 seq_length for gray and color
        # Condition
        self.cond_input = nn.Linear(3, embed_dim)
        self.cond_emb = nn.Parameter(torch.zeros(embed_dim))
        # transformer
        self.blocks = nn.ModuleList(
            [AttentionBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        # decoder head
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_color, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, color, gray, cond=None, cond_indices=None):
        h_gray = self.gray_input(gray)
        h_color = self.tok_emb(color)

        # Insert conditions
        if cond is not None and cond.shape[0] > 0:
            h_cond = self.cond_input(cond)
            for i, ind in enumerate(cond_indices):
                b, r, c = ind
                h_color[b, r, c, :] = h_cond[i, :] + self.cond_emb

        # forward the GPT model
        x = torch.cat([h_gray, h_color.flatten(1, 2)], dim=1)

        length = x.shape[1]
        # Positional embeddings
        position_embeddings = self.pos_emb[
            :, :length, :
        ]  # each position maps to a (learnable) vector
        x = x + position_embeddings
        # Transformers forward
        for block in self.blocks:
            x = block(x)
        # Output logits
        x = self.ln_f(x)
        x = x[:, -length // 2 :, :]
        logits = self.head(x)

        return logits


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        # output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = num_heads

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, head, len, dim)
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, head, len, dim)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, head, len, dim)

        # causal self-attention; Self-attend: (B, head, len, dim) x (B, head, dim, len) -> (B, head, len, len)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, head, len, len) x (B, head, len, dim) -> (B, head, len, dim)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # layers
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        attn = self.attn(self.ln1(x))
        x = x + attn
        x = x + self.mlp(self.ln2(x))

        return x


# --- staging harness (torchlens menagerie build/example entry points) ---


def build_unicolor():
    torch.manual_seed(0)
    return HybridTran(
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        input_shape=(4, 4),
        dim_gray=16,
        vocab_color=48,
    )


def example_input_unicolor():
    torch.manual_seed(0)
    batch_size = 2
    grid_h, grid_w = 4, 4
    seq_length = grid_h * grid_w
    dim_gray = 16
    vocab_color = 48
    color = torch.randint(
        low=0, high=vocab_color, size=(batch_size, grid_h, grid_w), dtype=torch.long
    )
    gray = torch.randn(batch_size, seq_length, dim_gray)
    cond = torch.empty(0, 3)
    cond_indices = torch.empty(0, 3, dtype=torch.long)
    return (color, gray, cond, cond_indices)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("UniColor", "build_unicolor", "example_input_unicolor", 2022, MENAGERIE_ZOO),
]
