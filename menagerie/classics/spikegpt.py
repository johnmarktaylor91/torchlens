"""SpikeGPT: spiking RWKV-style language model with binary spike token-shift.

Zhu et al., "SpikeGPT: Generative Pre-trained Language Model with Spiking
Neural Networks", arXiv 2023.
Paper: https://arxiv.org/abs/2302.13939
Source: https://github.com/ridgerchu/SpikeGPT

SpikeGPT adapts the RWKV architecture (token-shift + linear recurrence, no
softmax) to spiking neural networks.  Every dense activation in RWKV is
replaced by a LIF spiking neuron, making the entire token-shift and
receptance/WKV linear-attention computation spike-domain.

Distinctive mechanism:
  - **Token-shift:** input token embedding is mixed with the previous token's
    embedding (``alpha * x[s] + (1-alpha) * x[s-1]``) before computing the
    RWKV R/K/V projections.  Gives a temporal memory without full attention.
  - **Spiking receptance / WKV:** R, K, V from LIF(Linear) are binary spike
    trains.  The WKV linear-recurrence ``wkv[s] = (decay * wkv[s-1] + k[s]*v[s])``
    is accumulated per-token, then gated by the spike receptance R.
  - **Spiking ChannelMix (FFN):** token-shift -> LIF(Linear) gate * LIF(Linear)
    value, addition-only, pure spike domain.

The 216M configuration has d_model=768, 24 layers, vocab=50257.  This compact
proxy uses d_model=64, 2 blocks, vocab=256, seq_len=16 to keep the unrolled
graph tractable.  Architecture and dynamics are identical; only scale differs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics._snn_neurons import LIFNeuron


# ============================================================
# SpikeGPT building blocks
# ============================================================


class _TokenShift(nn.Module):
    """Token-shift: mix current token with previous token embedding.

    Given (B, S, D): shifted[s] = alpha * x[s] + (1-alpha) * x[s-1],
    where x[-1] = zeros.  Alpha is a learnable per-channel scalar in (0,1).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.full((d_model,), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        prev = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)
        alpha = self.alpha.sigmoid()
        return alpha * x + (1.0 - alpha) * prev


class _SpikeMLP(nn.Module):
    """LIF(Linear) spiking projection -- processes (B, S, D_in) -> (B, S, D_out).

    The S (sequence) dimension is passed as the TIME axis of the LIF neuron so
    TorchLens unrolls the spiking recurrence over the token sequence.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.lif = LIFNeuron(beta=0.9, threshold=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D_in)
        y = self.linear(x)  # (B, S, D_out)
        # permute to (S, B, D_out) so LIF unrolls over S
        y = y.permute(1, 0, 2)  # (S, B, D_out)
        s = self.lif(y)  # (S, B, D_out)
        return s.permute(1, 0, 2)  # (B, S, D_out)


class WKVAttention(nn.Module):
    """Spiking RWKV-style linear recurrence (WKV) block.

    Per-token causal accumulation: wkv[s] = decay * wkv[s-1] + k[s]*v[s],
    then gated by spike receptance r.  Token-shift before R/K/V.  No softmax.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.shift_r = _TokenShift(d_model)
        self.shift_k = _TokenShift(d_model)
        self.shift_v = _TokenShift(d_model)
        self.r_proj = _SpikeMLP(d_model, d_model)
        self.k_proj = _SpikeMLP(d_model, d_model)
        self.v_proj = _SpikeMLP(d_model, d_model)
        # per-channel learnable decay W in (0, 1)
        self.w = nn.Parameter(-torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        B, S, D = x.shape
        r = self.r_proj(self.shift_r(x))  # (B, S, D) spikes
        k = self.k_proj(self.shift_k(x))  # (B, S, D) spikes
        v = self.v_proj(self.shift_v(x))  # (B, S, D) spikes

        decay = torch.exp(self.w.clamp(max=-0.01))  # (D,)
        # WKV causal recurrence over S
        wkv_acc = torch.zeros(B, D, device=x.device)
        outs = []
        for s in range(S):
            wkv_acc = decay * wkv_acc + k[:, s, :] * v[:, s, :]
            out_s = r[:, s, :] * wkv_acc  # spike gate
            outs.append(out_s)
        out = torch.stack(outs, dim=1)  # (B, S, D)
        return self.out_proj(out)


class ChannelMix(nn.Module):
    """Spiking channel-mix (FFN) block.

    SpikeGPT ChannelMix: token-shift -> spike gate r + spike value k ->
    gated output ``r * v(k)``.  Pure spike-domain, no softmax.
    """

    def __init__(self, d_model: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = d_model * expansion
        self.shift_r = _TokenShift(d_model)
        self.shift_k = _TokenShift(d_model)
        self.r_proj = _SpikeMLP(d_model, d_model)
        self.k_proj = _SpikeMLP(d_model, hidden)
        self.v_proj = _SpikeMLP(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.r_proj(self.shift_r(x))  # (B, S, D) gate spike
        k = self.k_proj(self.shift_k(x))  # (B, S, hidden) spike
        v = self.v_proj(k)  # (B, S, D) spike
        return r * v


class SpikeGPTBlock(nn.Module):
    """One SpikeGPT decoder block: LayerNorm + WKV + LayerNorm + ChannelMix."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.wkv = WKVAttention(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = ChannelMix(d_model, expansion=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.wkv(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SpikeGPT(nn.Module):
    """SpikeGPT: spiking RWKV-style language model.

    Token embedding -> N SpikeGPT blocks (spiking token-shift + WKV + channel-mix)
    -> LayerNorm -> LM head over vocabulary.

    Compact proxy: d_model=64, depth=2, vocab=256, seq_len=16.
    Nominal 216M config: d_model=768, depth=24, vocab=50257.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([SpikeGPTBlock(d_model) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S) integer token ids
        h = self.embed(x)  # (B, S, D)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return self.head(h)  # (B, S, vocab_size)


def build_spikegpt_216m() -> nn.Module:
    """Build SpikeGPT 216M compact proxy (d_model=64, depth=2; nominal 768/24)."""
    return SpikeGPT(vocab_size=256, d_model=64, depth=2)


def example_input() -> torch.Tensor:
    """Example token sequence ``(1, 16)`` int64 ids."""
    return torch.randint(0, 256, (1, 16))


MENAGERIE_ENTRIES = [
    (
        "SpikeGPT 216M (spiking RWKV token-shift + WKV linear-recurrence LM)",
        "build_spikegpt_216m",
        "example_input",
        "2023",
        "DC",
    ),
]
