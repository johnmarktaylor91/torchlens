"""CoCa: Contrastive Captioners are Image-Text Foundation Models.

Yu et al., 2022, arXiv:2205.01917.

CoCa combines a vision encoder, a text decoder whose first layers are unimodal
language-model layers, and later multimodal decoder layers with cross-attention
to image tokens.  It uses a shared graph for contrastive image/text embeddings
and autoregressive caption logits.  This compact random-init reconstruction keeps
that split decoder and dual-head structure without pretrained weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CoCaBlock(nn.Module):
    """Decoder block with optional image cross-attention."""

    def __init__(self, dim: int, heads: int, cross_attend: bool) -> None:
        """Initialize the decoder block.

        Parameters
        ----------
        dim:
            Token embedding width.
        heads:
            Number of attention heads.
        cross_attend:
            Whether to include image cross-attention.
        """

        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = (
            nn.MultiheadAttention(dim, heads, batch_first=True) if cross_attend else None
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(
        self, text: torch.Tensor, image_tokens: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply causal self-attention, optional cross-attention, and MLP.

        Parameters
        ----------
        text:
            Text tokens of shape ``(batch, length, dim)``.
        image_tokens:
            Encoded image tokens of shape ``(batch, patches, dim)``.
        mask:
            Causal attention mask.

        Returns
        -------
        torch.Tensor
            Updated text tokens.
        """

        text = (
            text + self.self_attn(self.ln1(text), self.ln1(text), self.ln1(text), attn_mask=mask)[0]
        )
        if self.cross_attn is not None:
            text = text + self.cross_attn(self.ln2(text), image_tokens, image_tokens)[0]
        text = text + self.ff(self.ln3(text))
        return text


class CompactCoCa(nn.Module):
    """Compact CoCa-style image-text encoder-decoder."""

    def __init__(self, vocab: int = 128, dim: int = 48, depth: int = 4) -> None:
        """Initialize compact CoCa.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Embedding width.
        depth:
            Number of decoder blocks.
        """

        super().__init__()
        self.patch = nn.Conv2d(3, dim, kernel_size=8, stride=8)
        self.image_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, 4 * dim, batch_first=True, norm_first=True),
            num_layers=1,
        )
        self.token = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(1, 12, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [CoCaBlock(dim, 4, cross_attend=i >= depth // 2) for i in range(depth)]
        )
        self.text_proj = nn.Linear(dim, dim, bias=False)
        self.image_proj = nn.Linear(dim, dim, bias=False)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Run CoCa and concatenate caption logits with contrastive scores.

        Parameters
        ----------
        inputs:
            Tuple ``(image, token_ids)``.

        Returns
        -------
        torch.Tensor
            Flattened caption logits followed by one image/text similarity score.
        """

        image, token_ids = inputs
        image_tokens = self.patch(image).flatten(2).transpose(1, 2)
        image_tokens = self.image_encoder(image_tokens)
        text = self.token(token_ids) + self.pos[:, : token_ids.shape[1]]
        mask = torch.full(
            (token_ids.shape[1], token_ids.shape[1]), float("-inf"), device=text.device
        ).triu(1)
        for block in self.blocks:
            text = block(text, image_tokens, mask)
        image_emb = nn.functional.normalize(self.image_proj(image_tokens.mean(dim=1)), dim=-1)
        text_emb = nn.functional.normalize(self.text_proj(text[:, 0]), dim=-1)
        contrastive = (image_emb * text_emb).sum(dim=-1, keepdim=True)
        logits = self.lm_head(text).flatten(1)
        return torch.cat([logits, contrastive], dim=-1)


def build() -> nn.Module:
    """Build a compact random-init CoCa model.

    Returns
    -------
    nn.Module
        CoCa reconstruction.
    """

    return CompactCoCa()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Create an image/text example input.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Image tensor and token ids.
    """

    return torch.randn(1, 3, 32, 32), torch.randint(0, 128, (1, 10))


MENAGERIE_ENTRIES = [
    ("CoCa-ContrastiveCaptioner", "build", "example_input", "2022", "VL"),
]
