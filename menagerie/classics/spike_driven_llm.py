"""Spike-Driven LLM: a spiking GPT-style decoder language model.

Spiking-decoder LM in the SpikeGPT / SpikeDecoder family.
References:
  * SpikeGPT (Zhu et al., 2023): https://arxiv.org/abs/2302.13939
  * SpikeDecoder: Realizing the GPT Architecture with SNNs (2026):
    https://arxiv.org/abs/2606.12287

A fully spike-driven autoregressive Transformer decoder for language modeling:
token embedding -> N spiking decoder blocks (spiking self-attention + spiking
GLU/MLP, causal) -> spike-form LM head over the vocabulary.  The distinctive
mechanism vs. an ANN GPT is that every nonlinearity is a LIF spiking neuron and
the attention is *softmax-free* spike self-attention (Spiking Self-Attention,
SSA): Q/K/V are spike trains, attention is computed as spike-form matmuls with a
spiking neuron after the product, and causal masking enforces autoregression.

This faithful random-init reimplementation captures the spiking GPT decoder
spine.  A leading TIME axis (T spiking timesteps) is unrolled by TorchLens.
Sequence length, embed dim and depth are kept small so the unrolled graph
renders quickly; the dynamics are identical at any T / sequence length.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics._snn_neurons import LIFNeuron


class _SpikingSelfAttention(nn.Module):
    """Causal Spiking Self-Attention (SSA): softmax-free spike attention.

    Q, K, V are spike trains from LIF neurons.  The attention map is the
    spike-form scaled dot product with a causal mask; a spiking neuron is applied
    to the attention output (no softmax).
    """

    def __init__(self, dim: int = 128, heads: int = 4, seq: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_lif = LIFNeuron(beta=0.9, threshold=1.0)
        self.k_lif = LIFNeuron(beta=0.9, threshold=1.0)
        self.v_lif = LIFNeuron(beta=0.9, threshold=1.0)
        self.attn_lif = LIFNeuron(beta=0.9, threshold=1.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_lif = LIFNeuron(beta=0.9, threshold=1.0)
        mask = torch.tril(torch.ones(seq, seq)).view(1, 1, 1, seq, seq)
        self.register_buffer("causal_mask", mask)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        t, b, n, _ = x.shape
        return x.reshape(t, b, n, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, N, D)
        q = self._heads(self.q_lif(self.q_proj(x)))
        k = self._heads(self.k_lif(self.k_proj(x)))
        v = self._heads(self.v_lif(self.v_proj(x)))
        # spike-form attention scores (no softmax), causal masked
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        scores = scores * self.causal_mask
        out = torch.matmul(scores, v)  # (T, B, H, N, hd)
        out = self.attn_lif(out)
        t, b, hh, n, hd = out.shape
        out = out.permute(0, 1, 3, 2, 4).reshape(t, b, n, self.dim)
        out = self.proj_lif(self.proj(out))
        return out


class _SpikingGLU(nn.Module):
    """Spiking gated MLP (GLU-style) feed-forward block."""

    def __init__(self, dim: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.fc_in = nn.Linear(dim, hidden)
        self.fc_gate = nn.Linear(dim, hidden)
        self.lif_in = LIFNeuron(beta=0.9, threshold=1.0)
        self.lif_gate = LIFNeuron(beta=0.9, threshold=1.0)
        self.fc_out = nn.Linear(hidden, dim)
        self.lif_out = LIFNeuron(beta=0.9, threshold=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.lif_gate(self.fc_gate(x))
        h = self.lif_in(self.fc_in(x))
        h = h * g
        return self.lif_out(self.fc_out(h))


class _SpikingDecoderBlock(nn.Module):
    """A spiking GPT decoder block: residual SSA + residual spiking GLU."""

    def __init__(self, dim: int = 128, heads: int = 4, hidden: int = 256, seq: int = 16) -> None:
        super().__init__()
        self.attn = _SpikingSelfAttention(dim, heads, seq)
        self.glu = _SpikingGLU(dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.glu(x)
        return x


class SpikeDrivenLLM(nn.Module):
    """Spike-driven GPT decoder language model (random init)."""

    def __init__(
        self,
        vocab_size: int = 256,
        seq: int = 16,
        dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        hidden: int = 256,
        time: int = 2,
    ) -> None:
        super().__init__()
        self.seq = seq
        self.time = time
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq, dim))
        self.embed_lif = LIFNeuron(beta=0.9, threshold=1.0)
        self.blocks = nn.ModuleList(
            [_SpikingDecoderBlock(dim, heads, hidden, seq) for _ in range(depth)]
        )
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, N) long token ids
        emb = self.token_embed(idx) + self.pos_embed  # (B, N, D)
        # spike over T timesteps
        emb_t = emb.unsqueeze(0).expand(self.time, -1, -1, -1)  # (T, B, N, D)
        x = self.embed_lif(emb_t)
        for blk in self.blocks:
            x = blk(x)
        # temporal mean -> LM logits over the vocabulary
        x = x.mean(dim=0)  # (B, N, D)
        return self.head(x)  # (B, N, vocab)


def build_spike_driven_llm() -> nn.Module:
    """Build the spike-driven GPT decoder language model (random init)."""
    return SpikeDrivenLLM(vocab_size=256, seq=16, dim=128, depth=2, heads=4, hidden=256, time=2)


def example_input() -> torch.Tensor:
    """Example token-id sequence ``(1, 16)`` (long); embedded then spiked over T=2."""
    return torch.randint(0, 256, (1, 16))


MENAGERIE_ENTRIES = [
    (
        "Spike-Driven LLM (spiking GPT decoder language model)",
        "build_spike_driven_llm",
        "example_input",
        "2023",
        "DC",
    ),
]
