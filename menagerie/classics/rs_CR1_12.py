# FAITHFUL REIMPLEMENTATION from https://aclanthology.org/2020.iwslt-1.28.pdf (no public code)
"""Toy ACT simultaneous translation Transformer."""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ZOO = "reimpl-pytorch"


class ACTDecoderLayer(nn.Module):
    """Transformer decoder layer with Graves-style adaptive halting over source prefixes."""

    def __init__(self, d_model: int, nhead: int, threshold: float, max_steps: int) -> None:
        """Initialize the ACT decoder layer.

        Parameters
        ----------
        d_model:
            Hidden width.
        nhead:
            Number of attention heads.
        threshold:
            Halting probability threshold.
        max_steps:
            Maximum read/ponder steps.
        """
        super().__init__()
        self.threshold = threshold
        self.max_steps = max_steps
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model)
        )
        self.halt = nn.Linear(d_model, 1)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])

    def forward(self, target: Tensor, memory: Tensor) -> Tensor:
        """Apply self-attention, ACT-controlled encoder attention, and feed-forward updates.

        Parameters
        ----------
        target:
            Target token states of shape ``(batch, target_len, d_model)``.
        memory:
            Source encoder states of shape ``(batch, source_len, d_model)``.

        Returns
        -------
        Tensor
            Updated target states.
        """
        self_out = self.self_attn(target, target, target, need_weights=False)[0]
        target = self.norms[0](target + self_out)
        accumulated = torch.zeros(target.shape[0], target.shape[1], 1, device=target.device)
        remainder = torch.ones_like(accumulated)
        weighted_state = torch.zeros_like(target)
        state = target
        source_len = memory.shape[1]
        for step in range(self.max_steps):
            prefix_len = min(step + 1, source_len)
            attended = self.cross_attn(
                state, memory[:, :prefix_len], memory[:, :prefix_len], need_weights=False
            )[0]
            proposal = self.norms[1](state + attended)
            halt_prob = torch.sigmoid(self.halt(proposal))
            still_running = (accumulated < self.threshold).to(target.dtype)
            new_accumulated = accumulated + halt_prob * still_running
            over_threshold = (new_accumulated > self.threshold).to(target.dtype)
            update_weight = torch.where(
                over_threshold > 0,
                remainder,
                halt_prob * still_running,
            )
            weighted_state = weighted_state + update_weight * proposal
            remainder = torch.clamp(remainder - update_weight, min=0.0)
            accumulated = torch.minimum(new_accumulated, torch.ones_like(new_accumulated))
            state = proposal
        target = torch.where(accumulated > 0, weighted_state, state)
        return self.norms[2](target + self.ff(target))


class ACTSimultaneousTransformer(nn.Module):
    """Transformer NMT toy model with ACT as learned read/write policy."""

    def __init__(self, vocab_size: int = 32, d_model: int = 16, nhead: int = 2) -> None:
        """Initialize the model.

        Parameters
        ----------
        vocab_size:
            Source and target vocabulary size.
        d_model:
            Embedding dimension.
        nhead:
            Number of attention heads.
        """
        super().__init__()
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.decoder = ACTDecoderLayer(d_model, nhead, threshold=0.9, max_steps=4)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, sample: tuple[Tensor, Tensor]) -> Tensor:
        """Run simultaneous translation logits.

        Parameters
        ----------
        sample:
            Tuple of source token ids and previous target token ids.

        Returns
        -------
        Tensor
            Token logits with shape ``(batch, target_len, vocab_size)``.
        """
        source, target = sample
        memory = self.encoder(self.src_embed(source))
        decoded = self.decoder(self.tgt_embed(target), memory)
        return self.output(decoded)


def build_act_nmt() -> ACTSimultaneousTransformer:
    """Build a tiny ACT simultaneous NMT model.

    Returns
    -------
    ACTSimultaneousTransformer
        Model instance.
    """
    return ACTSimultaneousTransformer()


def example_input_act_nmt() -> tuple[Tensor, Tensor]:
    """Create example source and target token ids.

    Returns
    -------
    tuple[Tensor, Tensor]
        Source and target id tensors.
    """
    return (
        torch.tensor([[1, 5, 8, 2, 0]], dtype=torch.long),
        torch.tensor([[1, 7, 3, 2]], dtype=torch.long),
    )


MENAGERIE_ENTRIES = [
    (
        "Adaptive Computation Time for NMT (ACT-NMT)",
        build_act_nmt,
        example_input_act_nmt,
        2020,
        "REIMPL",
    )
]
