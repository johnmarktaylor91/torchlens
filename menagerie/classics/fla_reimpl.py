"""Flash Linear Attention family reimplementations.

Paper: representative linear-time sequence models from the FLA roster, including
RetNet retention, Gated Linear Attention, DeltaNet/Gated DeltaNet, Mamba/Mamba2
selective state-space blocks, HGRN, Forgetting Transformer, GSA/KDA, ReBased
linear attention, and Samba-style hybrid SSM-attention.

This Torch-only module replaces unavailable Triton/CUDA FLA kernels with compact
random-init PyTorch modules that keep each family's load-bearing recurrence or
linear-attention structure. It is intended for forward trace validation, not for
kernel parity or pretrained checkpoints.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FLAMixer(nn.Module):
    """Variant-specific linear-time sequence mixer."""

    def __init__(self, hidden_size: int = 64, variant: str = "linear") -> None:
        """Initialize projections and recurrent parameters.

        Parameters
        ----------
        hidden_size:
            Hidden feature width.
        variant:
            FLA family variant.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.variant = variant
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.depthwise = nn.Conv1d(hidden_size, hidden_size, 3, padding=1, groups=hidden_size)
        self.state_a = nn.Parameter(torch.zeros(hidden_size))
        self.slots = nn.Parameter(torch.randn(4, hidden_size) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Mix a sequence with the selected FLA recurrence.

        Parameters
        ----------
        x:
            Sequence tensor ``(batch, time, hidden)``.

        Returns
        -------
        Tensor
            Mixed sequence.
        """
        if self.variant in {"mamba", "mamba2", "samba", "loglinear"}:
            mixed = self._ssm(x)
            if self.variant == "samba":
                mixed = mixed + self._linear_attention(x)
            return self.out(mixed)
        if self.variant in {"delta", "gated_delta", "delta_product", "kda", "comba"}:
            return self.out(self._delta_rule(x))
        if self.variant in {"retnet", "fox"}:
            return self.out(self._retention(x))
        if self.variant in {"gsa"}:
            return self.out(self._slot_attention(x))
        if self.variant in {"hgrn"}:
            return self.out(self._hgrn(x))
        return self.out(self._linear_attention(x))

    def _linear_attention(self, x: Tensor) -> Tensor:
        """Apply positive-feature linear attention.

        Parameters
        ----------
        x:
            Sequence tensor.

        Returns
        -------
        Tensor
            Attention output.
        """
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        kv = torch.zeros(
            x.shape[0], self.hidden_size, self.hidden_size, device=x.device, dtype=x.dtype
        )
        normalizer = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for q_t, k_t, v_t in zip(q.unbind(1), k.unbind(1), v.unbind(1)):
            kv = kv + torch.einsum("bi,bj->bij", k_t, v_t)
            normalizer = normalizer + k_t
            numerator = torch.einsum("bi,bij->bj", q_t, kv)
            denominator = (q_t * normalizer).sum(dim=-1, keepdim=True).clamp_min(1e-4)
            outputs.append(numerator / denominator)
        return torch.stack(outputs, dim=1) * torch.sigmoid(self.gate(x))

    def _delta_rule(self, x: Tensor) -> Tensor:
        """Apply DeltaNet-style fast-weight updates.

        Parameters
        ----------
        x:
            Sequence tensor.

        Returns
        -------
        Tensor
            Fast-weight outputs.
        """
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        fast_weight = torch.zeros(
            x.shape[0], self.hidden_size, self.hidden_size, device=x.device, dtype=x.dtype
        )
        outputs = []
        for q_t, k_t, v_t, gate_t in zip(
            q.unbind(1), k.unbind(1), v.unbind(1), self.gate(x).unbind(1)
        ):
            prediction = torch.einsum("bij,bj->bi", fast_weight, k_t)
            update = (v_t - prediction) * torch.sigmoid(gate_t)
            if self.variant == "delta_product":
                householder = torch.einsum("bi,bj->bij", k_t, k_t)
                fast_weight = fast_weight - fast_weight.bmm(householder)
            fast_weight = fast_weight + torch.einsum("bi,bj->bij", update, k_t)
            outputs.append(torch.einsum("bij,bj->bi", fast_weight, q_t))
        return torch.stack(outputs, dim=1)

    def _retention(self, x: Tensor) -> Tensor:
        """Apply RetNet/FoX exponential retention.

        Parameters
        ----------
        x:
            Sequence tensor.

        Returns
        -------
        Tensor
            Retention outputs.
        """
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        decay = torch.sigmoid(self.state_a).view(1, -1)
        state = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for q_t, k_t, v_t in zip(q.unbind(1), k.unbind(1), v.unbind(1)):
            forget = torch.sigmoid(k_t) if self.variant == "fox" else decay
            state = forget * state + torch.tanh(k_t) * v_t
            outputs.append(torch.tanh(q_t) * state)
        return torch.stack(outputs, dim=1)

    def _ssm(self, x: Tensor) -> Tensor:
        """Apply Mamba-style depthwise convolution plus selective SSM recurrence.

        Parameters
        ----------
        x:
            Sequence tensor.

        Returns
        -------
        Tensor
            State-space outputs.
        """
        conv = self.depthwise(x.transpose(1, 2)).transpose(1, 2)
        candidate = torch.tanh(conv)
        delta = torch.sigmoid(self.gate(x))
        state = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        a = torch.sigmoid(self.state_a).view(1, -1)
        for cand_t, delta_t in zip(candidate.unbind(1), delta.unbind(1)):
            state = a * state + delta_t * cand_t
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    def _slot_attention(self, x: Tensor) -> Tensor:
        """Apply gated slot attention.

        Parameters
        ----------
        x:
            Sequence tensor.

        Returns
        -------
        Tensor
            Slot-mixed sequence.
        """
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        slots = self.slots.unsqueeze(0).expand(x.shape[0], -1, -1)
        outputs = []
        for q_t, k_t, v_t in zip(q.unbind(1), k.unbind(1), v.unbind(1)):
            slot_scores = torch.softmax(torch.einsum("bd,bsd->bs", q_t + k_t, slots), dim=-1)
            context = torch.einsum("bs,bsd->bd", slot_scores, slots) + v_t
            slots = slots + 0.05 * torch.tanh(context).unsqueeze(1)
            outputs.append(context)
        return torch.stack(outputs, dim=1)

    def _hgrn(self, x: Tensor) -> Tensor:
        """Apply HGRN-style gated recurrence.

        Parameters
        ----------
        x:
            Sequence tensor.

        Returns
        -------
        Tensor
            Recurrent outputs.
        """
        value = torch.tanh(self.qkv(x)[..., : self.hidden_size])
        gate = torch.sigmoid(self.gate(x))
        state = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for value_t, gate_t in zip(value.unbind(1), gate.unbind(1)):
            state = gate_t * state + (1.0 - gate_t) * value_t
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class FLABlock(nn.Module):
    """Transformer-style block around an FLA mixer."""

    def __init__(self, hidden_size: int, variant: str) -> None:
        """Initialize normalization, mixer, and MLP.

        Parameters
        ----------
        hidden_size:
            Hidden feature width.
        variant:
            FLA mixer variant.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.mixer = FLAMixer(hidden_size, variant)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual FLA block.

        Parameters
        ----------
        x:
            Sequence tensor.

        Returns
        -------
        Tensor
            Updated sequence.
        """
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class FLAModel(nn.Module):
    """Compact FLA model or causal language model."""

    def __init__(
        self,
        variant: str = "linear",
        causal_lm: bool = False,
        input_features: bool = False,
        hidden_size: int = 64,
        vocab_size: int = 128,
    ) -> None:
        """Initialize embedding/projection, FLA blocks, and output head.

        Parameters
        ----------
        variant:
            FLA mixer variant.
        causal_lm:
            Whether to return vocabulary logits.
        input_features:
            Whether inputs are already hidden features.
        hidden_size:
            Hidden feature width.
        vocab_size:
            Vocabulary size for token models.
        """
        super().__init__()
        self.input_features = input_features
        self.causal_lm = causal_lm
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.feature_proj = nn.Linear(hidden_size, hidden_size)
        self.blocks = nn.ModuleList([FLABlock(hidden_size, variant) for _ in range(2)])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_or_features: Tensor) -> Tensor:
        """Run the FLA sequence model.

        Parameters
        ----------
        tokens_or_features:
            Float token IDs with shape ``(batch, time)`` or hidden features with
            shape ``(batch, time, hidden)``.

        Returns
        -------
        Tensor
            Hidden states or language-model logits.
        """
        if self.input_features or tokens_or_features.ndim == 3:
            hidden = self.feature_proj(tokens_or_features)
        else:
            tokens = tokens_or_features.long().remainder(self.embed.num_embeddings)
            hidden = self.embed(tokens)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden)
        return self.lm_head(hidden) if self.causal_lm else hidden


def build_fla(
    variant: str = "linear", causal_lm: bool = False, input_features: bool = False
) -> nn.Module:
    """Build a compact FLA model.

    Parameters
    ----------
    variant:
        FLA variant.
    causal_lm:
        Whether to return LM logits.
    input_features:
        Whether the input already contains hidden features.

    Returns
    -------
    nn.Module
        FLA model.
    """
    return FLAModel(variant=variant, causal_lm=causal_lm, input_features=input_features)
