"""Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Detection.

Liu et al., 2023.
Paper: https://arxiv.org/abs/2303.05499
Source: https://github.com/IDEA-Research/GroundingDINO

Grounding DINO is a dual-encoder-single-decoder open-set object detector that
fuses language into a DETR-style detection transformer:

  1. Image backbone (Swin Transformer) -> multi-scale image features.
  2. Text backbone (BERT) -> text token features.  (Here we replace BERT and its
     tokenizer with a fixed random token-embedding bank, since HF_HUB_OFFLINE=1
     forbids downloading a tokenizer/checkpoint; the architecture role -- a
     sequence of contextual text token embeddings -- is preserved.)
  3. Feature enhancer (cross-modality encoder): each layer applies
       - deformable / standard self-attention on image tokens,
       - vanilla self-attention on text tokens,
       - bidirectional image<->text cross-attention (text-to-image and
         image-to-text), aligning the two modalities.
  4. Language-guided query selection: pick the image tokens most similar to the
     text features to initialise decoder queries.
  5. Cross-modality decoder: each query passes through self-attention,
     image cross-attention, an additional text cross-attention, and an FFN.
  6. Heads: a box regression head and a contrastive class head (queries dotted
     against text features) produce open-set boxes + class logits.

We build a COMPACT but faithful Swin-T image backbone, a random text-embedding
text branch, a 2-layer feature enhancer with full bidirectional fusion, query
selection, and a 2-layer cross-modality decoder.  Deformable attention is
approximated by standard multi-head attention over the flattened image tokens to
stay traceable, while the dual-encoder + cross-modality fusion + decoder
structure is kept faithful.

ONE published variant captured here: grounding_dino_src_swin_t (Swin-Tiny).
The wrapper takes a SINGLE image tensor and internally generates the fixed
random text token ids/embeddings, returning a single tensor (box predictions).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Swin-T image backbone (compact, faithful structure)
# ============================================================


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def _window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class _Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bn, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(bn, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(bn, n, c)
        return self.proj(out)


class _SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 4, shift_size: int = 0) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, dim * 4)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows = _window_partition(x, self.window_size).view(
            -1, self.window_size * self.window_size, c
        )
        attn_windows = self.attn(windows).view(-1, self.window_size, self.window_size, c)
        x = _window_reverse(attn_windows, self.window_size, h, w)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(b, h * w, c)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class _PatchMerging(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, _, c = x.shape
        x = x.view(b, h, w, c)
        x = torch.cat(
            [x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]],
            -1,
        )
        x = x.view(b, -1, 4 * c)
        return self.reduction(self.norm(x))


class _SwinStage(nn.Module):
    def __init__(
        self, dim: int, depth: int, num_heads: int, window_size: int, downsample: bool
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _SwinBlock(
                    dim,
                    num_heads,
                    window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                )
                for i in range(depth)
            ]
        )
        self.downsample = _PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor, h: int, w: int):
        for blk in self.blocks:
            x = blk(x, h, w)
        b, _, c = x.shape
        feat = x.transpose(1, 2).view(b, c, h, w)
        if self.downsample is not None:
            x = self.downsample(x, h, w)
            h, w = h // 2, w // 2
        return x, feat, h, w


class _PatchEmbed(nn.Module):
    def __init__(self, in_ch: int = 3, embed_dim: int = 96, patch_size: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), h, w


class _SwinTBackbone(nn.Module):
    """Compact Swin-Tiny image backbone returning the last 3 feature levels."""

    def __init__(self, embed_dim: int = 96, window_size: int = 4) -> None:
        super().__init__()
        depths = [2, 2, 2, 2]
        num_heads = [3, 6, 12, 24]
        self.patch_embed = _PatchEmbed(3, embed_dim, 4)
        dims = [embed_dim * (2**i) for i in range(4)]
        self.feature_dims = dims[1:]  # Swin-T detection uses stages 2,3,4
        self.stages = nn.ModuleList(
            [
                _SwinStage(dims[i], depths[i], num_heads[i], window_size, downsample=(i < 3))
                for i in range(4)
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x, h, w = self.patch_embed(x)
        feats = []
        for stage in self.stages:
            x, feat, h, w = stage(x, h, w)
            feats.append(feat)
        return feats[1:]  # 3 multi-scale features (strides 8, 16, 32)


# ============================================================
# Feature enhancer: bidirectional cross-modality fusion
# ============================================================


class _MHA(nn.Module):
    """Standard multi-head attention (used for self- and cross-attention)."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        b, nq, c = query.shape
        nk = key.shape[1]
        q = self.q(query).reshape(b, nq, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(key).reshape(b, nk, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(key).reshape(b, nk, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, nq, c)
        return self.proj(out)


class _FeatureEnhancerLayer(nn.Module):
    """One enhancer layer: image self-attn, text self-attn, bidirectional fusion.

    (Deformable image self-attention is approximated by standard self-attention
    over the flattened image tokens to keep the trace clean.)
    """

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        # image self-attention (stands in for deformable self-attention)
        self.img_self = _MHA(dim, num_heads)
        self.img_norm1 = nn.LayerNorm(dim)
        # text self-attention
        self.txt_self = _MHA(dim, num_heads)
        self.txt_norm1 = nn.LayerNorm(dim)
        # bidirectional cross-attention
        self.i2t = _MHA(dim, num_heads)  # image queries attend to text
        self.t2i = _MHA(dim, num_heads)  # text queries attend to image
        self.img_norm2 = nn.LayerNorm(dim)
        self.txt_norm2 = nn.LayerNorm(dim)
        # FFNs
        self.img_ffn = _Mlp(dim, dim * 4)
        self.txt_ffn = _Mlp(dim, dim * 4)
        self.img_norm3 = nn.LayerNorm(dim)
        self.txt_norm3 = nn.LayerNorm(dim)

    def forward(self, img: torch.Tensor, txt: torch.Tensor):
        # self-attention
        img = img + self.img_self(self.img_norm1(img), self.img_norm1(img))
        txt = txt + self.txt_self(self.txt_norm1(txt), self.txt_norm1(txt))
        # bidirectional cross-attention (fusion)
        img_n, txt_n = self.img_norm2(img), self.txt_norm2(txt)
        img = img + self.i2t(img_n, txt_n)
        txt = txt + self.t2i(txt_n, img_n)
        # FFNs
        img = img + self.img_ffn(self.img_norm3(img))
        txt = txt + self.txt_ffn(self.txt_norm3(txt))
        return img, txt


# ============================================================
# Cross-modality decoder
# ============================================================


class _DecoderLayer(nn.Module):
    """Query: self-attn -> image cross-attn -> text cross-attn -> FFN."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.self_attn = _MHA(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.img_cross = _MHA(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.txt_cross = _MHA(dim, num_heads)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = _Mlp(dim, dim * 4)
        self.norm4 = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        q = q + self.self_attn(self.norm1(q), self.norm1(q))
        q = q + self.img_cross(self.norm2(q), img)
        q = q + self.txt_cross(self.norm3(q), txt)
        q = q + self.ffn(self.norm4(q))
        return q


# ============================================================
# Full Grounding DINO model
# ============================================================


class GroundingDINO(nn.Module):
    """Compact, faithful Grounding DINO (Swin-T image + random-embedding text)."""

    def __init__(
        self,
        embed_dim: int = 96,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_enhancer_layers: int = 2,
        num_decoder_layers: int = 2,
        num_queries: int = 50,
        text_len: int = 16,
        vocab_size: int = 1000,
        num_classes: int = 80,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.text_len = text_len

        # Image backbone
        self.backbone = _SwinTBackbone(embed_dim=embed_dim, window_size=4)
        bb_dims = self.backbone.feature_dims  # 3 levels
        # input projections of each level -> hidden_dim (DETR-style)
        self.input_proj = nn.ModuleList([nn.Conv2d(d, hidden_dim, 1) for d in bb_dims])

        # Text backbone replacement: fixed random token embedding bank + a small
        # contextualizing transformer self-attention (BERT-role stand-in).
        self.register_buffer("text_token_ids", torch.randint(0, vocab_size, (1, text_len)))
        self.text_embed = nn.Embedding(vocab_size, hidden_dim)
        self.text_encoder = _MHA(hidden_dim, num_heads)
        self.text_norm = nn.LayerNorm(hidden_dim)

        # Feature enhancer (cross-modality encoder)
        self.enhancer = nn.ModuleList(
            [_FeatureEnhancerLayer(hidden_dim, num_heads) for _ in range(num_enhancer_layers)]
        )

        # Language-guided query selection projections
        self.query_select_img = nn.Linear(hidden_dim, hidden_dim)
        self.query_select_txt = nn.Linear(hidden_dim, hidden_dim)
        # learned query content embeddings (anchor/content split, DINO-style)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Cross-modality decoder
        self.decoder = nn.ModuleList(
            [_DecoderLayer(hidden_dim, num_heads) for _ in range(num_decoder_layers)]
        )

        # Heads
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )
        # contrastive class head: project queries, dot with text features
        self.class_proj = nn.Linear(hidden_dim, hidden_dim)

    def _encode_text(self, batch: int) -> torch.Tensor:
        ids = self.text_token_ids.expand(batch, -1)  # (B, L)
        txt = self.text_embed(ids)  # (B, L, C)
        txt = txt + self.text_encoder(self.text_norm(txt), self.text_norm(txt))
        return txt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]

        # 1. image backbone -> multi-scale features, project + flatten + concat
        feats = self.backbone(x)
        img_tokens = []
        for proj, f in zip(self.input_proj, feats):
            t = proj(f).flatten(2).transpose(1, 2)  # (B, Hf*Wf, C)
            img_tokens.append(t)
        img = torch.cat(img_tokens, dim=1)  # (B, sum N, C)

        # 2. text branch (BERT-role random embeddings)
        txt = self._encode_text(b)  # (B, L, C)

        # 3. feature enhancer: bidirectional cross-modality fusion
        for layer in self.enhancer:
            img, txt = layer(img, txt)

        # 4. language-guided query selection: score image tokens vs text, pick top-k
        img_s = self.query_select_img(img)  # (B, N, C)
        txt_s = self.query_select_txt(txt)  # (B, L, C)
        sim = img_s @ txt_s.transpose(1, 2)  # (B, N, L)
        token_score = sim.max(dim=2).values  # (B, N)
        topk = min(self.num_queries, img.shape[1])
        idx = token_score.topk(topk, dim=1).indices  # (B, k)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        selected = torch.gather(img, 1, idx_exp)  # (B, k, C) positional/anchor part
        # combine selected image-token init with learned content queries
        content = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)[:, :topk, :]
        q = selected + content

        # 5. cross-modality decoder
        for layer in self.decoder:
            q = layer(q, img, txt)

        # 6. heads -> boxes (sigmoid cxcywh) + contrastive class logits
        boxes = self.bbox_head(q).sigmoid()  # (B, k, 4)
        cls = self.class_proj(q) @ txt.transpose(1, 2)  # (B, k, L) contrastive logits
        # return a single tensor: concat boxes + per-query max class logit
        cls_score = cls.max(dim=2, keepdim=True).values  # (B, k, 1)
        out = torch.cat([boxes, cls_score], dim=2)  # (B, k, 5)
        return out


def build(variant: str = "swin_t") -> nn.Module:
    """Build Grounding DINO (Swin-T variant)."""
    return GroundingDINO(
        embed_dim=96,
        hidden_dim=256,
        num_heads=8,
        num_enhancer_layers=2,
        num_decoder_layers=2,
        num_queries=50,
        text_len=16,
        vocab_size=1000,
        num_classes=80,
    )


# ============================================================
# Menagerie wiring: zero-arg builders + example inputs + entries.
# ============================================================


def build_grounding_dino() -> nn.Module:
    """Build Grounding DINO (Swin-T image backbone + random-embedding text)."""
    return build("swin_t")


def example_input() -> torch.Tensor:
    """Example RGB image tensor ``(1, 3, 128, 128)`` (text is generated internally)."""
    return torch.randn(1, 3, 128, 128)


MENAGERIE_ENTRIES = [
    (
        "Grounding DINO (open-set detection, Swin-T + cross-modality decoder)",
        "build_grounding_dino",
        "example_input",
        "2023",
        "DC",
    ),
]
