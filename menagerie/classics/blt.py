"""BLT (Byte Latent Transformer, tokenizer-free byte LLM).

Pagnoni, Pasunuru, Rodriguez, Nguyen, Muller, Li, Zhou, Yu, Weston, Zettlemoyer,
Ghosh, Lewis, Holtzman, Iyer (Meta / Stanford, 2024),
"Byte Latent Transformer: Patches Scale Better Than Tokens."
Paper: https://arxiv.org/abs/2412.09871
Reference impl: transformers ``BltForCausalLM`` (``itazap/blt-1b-hf``)

BLT is a tokenizer-free byte-level LLM.  Instead of a fixed BPE vocabulary it
ingests raw UTF-8 bytes (vocab 256 + a few special ids, HF default ``260``) and
groups them dynamically into *patches* (in the paper, via an entropy model).
The compute is split across three transformers:

  * **Local encoder** -- a small byte-level transformer that contextualises byte
    embeddings, then pools each group of bytes into a single patch representation
    (cross-attention from learned patch queries to encoder byte states).
  * **Latent global transformer** -- a larger transformer that runs over the
    (far fewer) patch representations, where most of the model capacity lives.
  * **Local decoder** -- a byte-level transformer that un-patches: it re-expands
    patch representations back to the byte grid via cross-attention and predicts
    the next byte at every position over the byte vocabulary.

This faithful-but-compact reimplementation builds all three transformers from
scratch (random embeddings, no HF download) and uses a *fixed* patch size to
group bytes into patches (the entropy patcher is training-time machinery; fixed
patching preserves the encode -> latent -> decode topology that defines BLT).
The output is per-byte logits ``(batch, seq_len, vocab_size)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _TransformerStack(nn.Module):
    """A stack of pre-norm multi-head self-attention transformer blocks."""

    def __init__(self, dim: int, n_layers: int, n_heads: int, ff_mult: int = 4) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * ff_mult,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class _CrossAttention(nn.Module):
    """Pre-norm multi-head cross-attention block (queries attend to a memory)."""

    def __init__(self, dim: int, n_heads: int, ff_mult: int = 4) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=0.0, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        attn_out, _ = self.attn(qn, kvn, kvn, need_weights=False)
        q = q + attn_out
        q = q + self.ff(self.norm_ff(q))
        return q


class BLT(nn.Module):
    """Byte Latent Transformer: local-encoder / latent-global / local-decoder.

    Compact faithful reimplementation with fixed patch grouping.
    """

    def __init__(
        self,
        vocab_size: int = 260,
        byte_dim: int = 128,
        patch_dim: int = 256,
        patch_size: int = 4,
        max_bytes: int = 256,
        max_patches: int = 64,
        n_encoder_layers: int = 2,
        n_global_layers: int = 4,
        n_decoder_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        self.byte_dim = byte_dim
        self.patch_dim = patch_dim

        # Byte-level embeddings + positional embeddings.
        self.byte_embed = nn.Embedding(vocab_size, byte_dim)
        self.byte_pos = nn.Parameter(torch.zeros(1, max_bytes, byte_dim))
        self.patch_pos = nn.Parameter(torch.zeros(1, max_patches, patch_dim))

        # Local encoder over bytes.
        self.local_encoder = _TransformerStack(byte_dim, n_encoder_layers, n_heads)

        # Patchify: pool each group of `patch_size` byte states into a patch query,
        # then project byte_dim -> patch_dim and refine with cross-attention.
        self.to_patch = nn.Linear(byte_dim, patch_dim)
        self.enc_to_patchdim = nn.Linear(byte_dim, patch_dim)
        self.patch_cross_attn = _CrossAttention(patch_dim, n_heads)

        # Latent global transformer over patch representations.
        self.global_transformer = _TransformerStack(patch_dim, n_global_layers, n_heads)

        # Local decoder: re-expand patches to bytes via cross-attention, then
        # byte-level self-attention, then per-byte logits.
        self.patch_to_byte = nn.Linear(patch_dim, byte_dim)
        self.decoder_cross_attn = _CrossAttention(byte_dim, n_heads)
        self.local_decoder = _TransformerStack(byte_dim, n_decoder_layers, n_heads)
        self.norm_out = nn.LayerNorm(byte_dim)
        self.lm_head = nn.Linear(byte_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """``input_ids``: ``(batch, seq_len)`` int64 byte ids in ``[0, vocab)``."""
        b, n = input_ids.shape
        # Pad sequence up to a multiple of patch_size so it groups cleanly.
        pad = (-n) % self.patch_size
        if pad:
            input_ids = torch.cat([input_ids, input_ids.new_zeros(b, pad)], dim=1)
        n_pad = input_ids.shape[1]
        n_patches = n_pad // self.patch_size

        # --- Local encoder over bytes ---
        h = self.byte_embed(input_ids) + self.byte_pos[:, :n_pad]
        h = self.local_encoder(h)  # (b, n_pad, byte_dim)

        # --- Patchify: mean-pool each patch_size window -> patch query ---
        grouped = h.view(b, n_patches, self.patch_size, self.byte_dim)
        patch_q = self.to_patch(grouped.mean(dim=2))  # (b, n_patches, patch_dim)
        patch_q = patch_q + self.patch_pos[:, :n_patches]
        # Cross-attend patch queries to encoder byte states for richer patches.
        enc_kv = self.enc_to_patchdim(h)  # (b, n_pad, patch_dim)
        patches = self.patch_cross_attn(patch_q, enc_kv)  # (b, n_patches, patch_dim)

        # --- Latent global transformer over patches ---
        patches = self.global_transformer(patches)  # (b, n_patches, patch_dim)

        # --- Local decoder: un-patch back to bytes ---
        patch_mem = self.patch_to_byte(patches)  # (b, n_patches, byte_dim)
        # Byte queries (encoder states) cross-attend to processed patches.
        dec = self.decoder_cross_attn(h, patch_mem)  # (b, n_pad, byte_dim)
        dec = self.local_decoder(dec)
        dec = self.norm_out(dec)
        logits = self.lm_head(dec)  # (b, n_pad, vocab)
        return logits[:, :n]  # drop padding -> (b, seq_len, vocab)


def build_blt() -> nn.Module:
    """Build a compact faithful BLT (tokenizer-free byte LLM)."""
    return BLT(
        vocab_size=260,
        byte_dim=128,
        patch_dim=256,
        patch_size=4,
        n_encoder_layers=2,
        n_global_layers=4,
        n_decoder_layers=2,
        n_heads=4,
    )


def example_input() -> torch.Tensor:
    """Example byte sequence ``(1, 64)`` of int64 byte ids in ``[0, 256)``."""
    return torch.randint(0, 256, (1, 64), dtype=torch.int64)


MENAGERIE_ENTRIES = [
    (
        "BLT (Byte Latent Transformer, tokenizer-free byte LLM)",
        "build_blt",
        "example_input",
        "2024",
        "DC",
    ),
]
