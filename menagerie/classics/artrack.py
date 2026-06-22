"""ARTrack: Autoregressive Visual Tracking.

Wei et al., CVPR 2023.
Paper: https://arxiv.org/abs/2303.10826
Source: https://github.com/MIV-XJTU/ARTrack

ARTrack recasts visual tracking as an autoregressive coordinate-sequence
prediction problem.  Its distinctive primitive:

  1. **Encoder**: extracts joint template+search features (two-stream with
     independent ViT-like patch embedding blocks; features are concatenated
     and processed together, or alternatively a one-stream ViT for ViT-B variant).
  2. **Autoregressive coordinate decoder**: a transformer decoder that generates
     the bounding-box coordinate token sequence (top-left x, top-left y, width,
     height) as discrete tokens one at a time.  Each decoding step attends over
     the encoder's context features and the previously generated coordinate tokens.
  3. **Coordinate vocabulary**: coordinates are discretized into V=1000 bins;
     the decoder is a seq2seq model over this vocabulary.

Two variants are implemented:
  * ARTrack: two-stream patch-embed encoder (template + search processed
    separately then concatenated) + AR coord decoder.
  * ARTrack-ViT-B: one-stream ViT encoder (template and search tokens
    concatenated before the transformer, processed jointly) + AR coord decoder.
    This is the higher-performing published default.

Architecture notes / simplifications:
  - Patch embedding + positional encoding at small dim (d_model=64).
  - Encoder: 2 transformer layers (vs 12 for ViT-Base in the paper).
  - Decoder: 2 transformer layers.
  - Coordinate vocab V=32 (paper uses 1000; reduced for compact graph).
  - Spatial resolution: 4x4 template + 8x8 search patches (paper: 8x8 + 16x16).
  - No pre-trained weights; random init.
  - Coordinate sequence length = 4 (cx, cy, w, h tokens).
  - trace+draw verified 2026-06-21.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared components
# ============================================================


class PatchEmbed(nn.Module):
    """Flatten image patches into tokens."""

    def __init__(self, img_size: int, patch_size: int, in_ch: int, embed_dim: int) -> None:
        super().__init__()
        assert img_size % patch_size == 0
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (N, embed_dim, H/p, W/p)
        return x.flatten(2).transpose(1, 2)  # (N, n_patches, embed_dim)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(*([self.norm1(x)] * 3))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class ARCoordDecoder(nn.Module):
    """Autoregressive coordinate decoder.

    Generates bounding-box coordinate tokens one at a time using a
    transformer decoder.  At each step, previously generated tokens
    (plus a learnable start token) serve as causal queries attending
    over encoder context memory.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ffn_dim: int,
        num_layers: int,
        coord_vocab: int,
        seq_len: int = 4,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.coord_vocab = coord_vocab

        # Learnable start token + coordinate token embeddings
        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.coord_embed = nn.Embedding(coord_vocab, d_model)

        # Causal positional embeddings for query sequence
        self.pos_embed = nn.Embedding(seq_len + 1, d_model)

        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim, dropout=0.0, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Projection to coordinate logits
        self.out_proj = nn.Linear(d_model, coord_vocab)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """Autoregressively generate coordinate tokens.

        Args:
            memory: (N, S, d_model) encoder output (template+search features).
        Returns:
            coord_logits: (N, seq_len, coord_vocab) -- logits for each position.
        """
        N = memory.shape[0]
        device = memory.device

        # Start with the start token
        query = self.start_token.expand(N, -1, -1)  # (N, 1, d)
        pos = self.pos_embed(torch.zeros(1, 1, dtype=torch.long, device=device))
        query = query + pos

        all_logits = []
        prev_tokens = query

        for step in range(self.seq_len):
            # Causal mask: queries attend only to previous positions
            tgt_len = prev_tokens.shape[1]
            causal_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool), diagonal=1
            )

            dec_out = self.decoder(
                prev_tokens,
                memory,
                tgt_mask=causal_mask.float().masked_fill(causal_mask, float("-inf")),
            )
            # Take the last token's output
            step_out = dec_out[:, -1:, :]  # (N, 1, d)
            logits = self.out_proj(step_out)  # (N, 1, coord_vocab)
            all_logits.append(logits)

            # Greedy next token (during tracing; training uses teacher forcing)
            next_token_idx = logits.argmax(dim=-1)  # (N, 1)
            next_embed = self.coord_embed(next_token_idx)  # (N, 1, d)
            step_pos = self.pos_embed(torch.full((1, 1), step + 1, dtype=torch.long, device=device))
            next_embed = next_embed + step_pos
            prev_tokens = torch.cat([prev_tokens, next_embed], dim=1)

        return torch.cat(all_logits, dim=1)  # (N, seq_len, coord_vocab)


# ============================================================
# ARTrack: Two-stream encoder + AR coord decoder
# ============================================================


class ARTrackEncoder(nn.Module):
    """Two-stream encoder: template + search patches processed separately then
    concatenated into joint context for the AR decoder.

    Uses different patch sizes for template (coarser, smaller token count) and
    search (finer, larger token count), as in the ARTrack paper.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ffn_dim: int,
        num_layers: int,
        tmpl_size: int,
        srch_size: int,
        tmpl_patch: int,
        srch_patch: int,
    ) -> None:
        super().__init__()
        # Separate patch embeddings for template and search
        self.tmpl_patch_embed = nn.Conv2d(3, d_model, kernel_size=tmpl_patch, stride=tmpl_patch)
        self.srch_patch_embed = nn.Conv2d(3, d_model, kernel_size=srch_patch, stride=srch_patch)
        n_tmpl = (tmpl_size // tmpl_patch) ** 2
        n_srch = (srch_size // srch_patch) ** 2
        self.tmpl_pos = nn.Parameter(torch.zeros(1, n_tmpl, d_model))
        self.srch_pos = nn.Parameter(torch.zeros(1, n_srch, d_model))
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model, nhead, ffn_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def _tokenize(self, img: torch.Tensor, embed: nn.Conv2d) -> torch.Tensor:
        x = embed(img)  # (N, d, H/p, W/p)
        return x.flatten(2).transpose(1, 2)  # (N, n, d)

    def forward(self, template: torch.Tensor, search: torch.Tensor) -> torch.Tensor:
        tmpl_tokens = self._tokenize(template, self.tmpl_patch_embed) + self.tmpl_pos
        srch_tokens = self._tokenize(search, self.srch_patch_embed) + self.srch_pos
        x = torch.cat([tmpl_tokens, srch_tokens], dim=1)  # (N, T+S, d)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class ARTrackModel(nn.Module):
    """ARTrack: two-stream encoder -> autoregressive coordinate decoder.

    Template: 32x32 with patch_size=8 -> 16 tokens.
    Search: 32x32 with patch_size=4 -> 64 tokens.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        ffn_dim: int = 128,
        num_enc_layers: int = 2,
        num_dec_layers: int = 2,
        coord_vocab: int = 32,
        tmpl_size: int = 32,
        srch_size: int = 32,
        tmpl_patch: int = 8,
        srch_patch: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = ARTrackEncoder(
            d_model, nhead, ffn_dim, num_enc_layers, tmpl_size, srch_size, tmpl_patch, srch_patch
        )
        self.decoder = ARCoordDecoder(d_model, nhead, ffn_dim, num_dec_layers, coord_vocab)

    def forward(self, template: torch.Tensor, search: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(template, search)  # (N, T+S, d)
        coord_logits = self.decoder(memory)  # (N, 4, coord_vocab)
        return coord_logits


class ARTrackWrapper(nn.Module):
    """Wrapper to accept stacked (template, search) input for TorchLens trace.

    Input: (N, 2, 3, 32, 32) -- both template and search at 32x32.
    Template patch embedding uses patch_size=8 -> 4x4=16 tokens.
    Search patch embedding uses patch_size=4 -> 8x8=64 tokens.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = ARTrackModel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 2, 3, H, W) -- dim 1 = [template, search]
        template = x[:, 0]  # (N, 3, 32, 32) -- uses patch_size=8 -> 16 tokens
        search = x[:, 1]  # (N, 3, 32, 32) -- uses patch_size=4 -> 64 tokens
        return self.model(template, search)


# ============================================================
# ARTrack-ViT-B: One-stream ViT encoder + AR coord decoder
# ============================================================


class ARTrackViTBModel(nn.Module):
    """ARTrack-ViT-B: one-stream ViT joint encoder -> autoregressive coord decoder.

    Template and search patches are concatenated BEFORE the transformer and
    processed jointly in a single ViT stream.  This is the high-performing
    default from the ARTrack paper.  Both template and search use the same
    shared patch embedding (patch_size=4, 32x32 -> 64 tokens each).
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        ffn_dim: int = 128,
        num_enc_layers: int = 2,
        num_dec_layers: int = 2,
        coord_vocab: int = 32,
        img_size: int = 32,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        n_patches = (img_size // patch_size) ** 2

        # Single shared patch embedding for both template and search
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.tmpl_pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        self.srch_pos = nn.Parameter(torch.zeros(1, n_patches, d_model))

        # One-stream ViT joint encoder (template+search tokens processed together)
        self.enc_layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model, nhead, ffn_dim) for _ in range(num_enc_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        # AR coord decoder
        self.decoder = ARCoordDecoder(d_model, nhead, ffn_dim, num_dec_layers, coord_vocab)

    def _embed(self, img: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(img)  # (N, d, H/p, W/p)
        tokens = x.flatten(2).transpose(1, 2)  # (N, n, d)
        return tokens + pos

    def forward(self, template: torch.Tensor, search: torch.Tensor) -> torch.Tensor:
        tmpl_tokens = self._embed(template, self.tmpl_pos)
        srch_tokens = self._embed(search, self.srch_pos)
        # Joint one-stream: concatenate BEFORE the ViT layers
        x = torch.cat([tmpl_tokens, srch_tokens], dim=1)  # (N, T+S, d)
        for layer in self.enc_layers:
            x = layer(x)
        memory = self.norm(x)
        coord_logits = self.decoder(memory)  # (N, 4, coord_vocab)
        return coord_logits


class ARTrackViTBWrapper(nn.Module):
    """Wrapper to accept stacked (template, search) input for TorchLens trace."""

    def __init__(self) -> None:
        super().__init__()
        self.model = ARTrackViTBModel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        template = x[:, 0]
        search = x[:, 1]
        return self.model(template, search)


# ============================================================
# Zero-arg builders and example inputs
# ============================================================


def build_artrack() -> nn.Module:
    """Build ARTrack (two-stream encoder + AR coordinate decoder)."""
    return ARTrackWrapper()


def build_artrack_vit_b() -> nn.Module:
    """Build ARTrack-ViT-B (one-stream ViT joint encoder + AR coordinate decoder)."""
    return ARTrackViTBWrapper()


def example_input_artrack() -> torch.Tensor:
    """Stacked (template, search) both 32x32.

    Template uses patch_size=8 -> 16 tokens; search uses patch_size=4 -> 64 tokens.
    Shape: (1, 2, 3, 32, 32).
    """
    return torch.randn(1, 2, 3, 32, 32)


def example_input_artrack_vit_b() -> torch.Tensor:
    """Stacked (template, search) both at 32x32 for ARTrack-ViT-B."""
    return torch.randn(1, 2, 3, 32, 32)


# ============================================================
# Menagerie entries
# ============================================================

MENAGERIE_ENTRIES = [
    (
        "ARTrack (autoregressive coord-sequence tracking, two-stream encoder)",
        "build_artrack",
        "example_input_artrack",
        "2023",
        "DC",
    ),
    (
        "ARTrack-ViT-B (autoregressive tracking, one-stream ViT joint encoder)",
        "build_artrack_vit_b",
        "example_input_artrack_vit_b",
        "2023",
        "DC",
    ),
]
