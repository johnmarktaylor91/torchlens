"""Kyutai Hibiki streaming speech-to-speech translation, compact form.

Hibiki (Labiausse et al., 2025) is a decoder-only simultaneous speech
translation model built on the Moshi multistream idea: a causal global
Transformer processes source/target streams synchronously, while a local
Transformer models the hierarchy of text and audio-codec tokens at each frame.
This reconstruction keeps the multistream causal decoder and per-frame local
token hierarchy without requiring the install-hostile Moshi package.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LocalHierarchy(nn.Module):
    """Per-frame local Transformer over text and audio codebook tokens."""

    def __init__(self, dim: int = 64, heads: int = 4, codebooks: int = 4) -> None:
        """Initialize local hierarchy layers.

        Parameters
        ----------
        dim:
            Token width.
        heads:
            Attention heads.
        codebooks:
            Number of audio codebook streams.
        """
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, codebooks + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True)
        self.local = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, frame_state: Tensor) -> Tensor:
        """Expand a frame state into hierarchical text/audio tokens.

        Parameters
        ----------
        frame_state:
            Global stream state of shape ``(batch, frames, dim)``.

        Returns
        -------
        Tensor
            Local token states ``(batch, frames, codebooks + 1, dim)``.
        """
        batch, frames, dim = frame_state.shape
        queries = self.query.expand(batch * frames, -1, -1)
        local_in = queries + frame_state.reshape(batch * frames, 1, dim)
        local_out = self.local(local_in)
        return local_out.reshape(batch, frames, queries.shape[1], dim)


class CompactHibiki(nn.Module):
    """Decoder-only multistream speech translation model."""

    def __init__(self, vocab: int = 128, dim: int = 64, streams: int = 3, heads: int = 4) -> None:
        """Initialize embeddings, causal global decoder, and local hierarchy.

        Parameters
        ----------
        vocab:
            Discrete audio/text token vocabulary size.
        dim:
            Token width.
        streams:
            Source speech, target speech, and target text streams.
        heads:
            Attention heads.
        """
        super().__init__()
        self.streams = streams
        self.token = nn.Embedding(vocab, dim)
        self.stream_embed = nn.Parameter(torch.randn(streams, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True)
        self.global_decoder = nn.TransformerEncoder(layer, num_layers=2)
        self.local = LocalHierarchy(dim, heads, codebooks=4)
        self.to_text = nn.Linear(dim, vocab)
        self.to_audio = nn.Linear(dim, vocab)

    def forward(self, tokens: Tensor) -> Tensor:
        """Run causal multistream decoding.

        Parameters
        ----------
        tokens:
            Token tensor of shape ``(batch, streams, frames)``.

        Returns
        -------
        Tensor
            Stacked logits for local text/audio hierarchy.
        """
        batch, streams, frames = tokens.shape
        embedded = self.token(tokens) + self.stream_embed[:streams][None, :, None]
        sequence = embedded.permute(0, 2, 1, 3).reshape(batch, frames * streams, -1)
        mask = torch.full((frames * streams, frames * streams), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1)
        decoded = self.global_decoder(sequence, mask=mask)
        frame_state = decoded.reshape(batch, frames, streams, -1).mean(dim=2)
        local = self.local(frame_state)
        text_logits = self.to_text(local[:, :, :1])
        audio_logits = self.to_audio(local[:, :, 1:])
        return torch.cat((text_logits, audio_logits), dim=2)


def build() -> nn.Module:
    """Build a compact Hibiki model.

    Returns
    -------
    nn.Module
        Random-init Hibiki reconstruction.
    """
    return CompactHibiki()


def example_input() -> Tensor:
    """Return source/target stream token input.

    Returns
    -------
    Tensor
        Integer token tensor ``(batch, streams, frames)``.
    """
    return torch.randint(0, 128, (1, 3, 8))


MENAGERIE_ENTRIES = [
    ("kyutai_hibiki_streaming_st", "build", "example_input", "2025", "audio/speech"),
]
